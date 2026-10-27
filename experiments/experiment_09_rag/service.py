"""Experiment 9 — PDF question answering with Retrieval-Augmented Generation.

Pipeline preserved from the original notebook (genaiexp9rag.ipynb):
  PyMuPDF page text  ->  word chunks (180 words, 40 overlap)  ->
  sentence-transformers/all-MiniLM-L6-v2 (normalised embeddings)  ->
  FAISS IndexFlatIP (cosine via inner product)  ->  Qwen/Qwen2.5-3B-Instruct answer.

Not carried over: the notebook's hard-coded routing for its own sample PDFs
(e.g. keyword rules for "Futurewise"). Retrieval here is purely semantic over the
documents the user uploads.

Heavy libraries are imported lazily so the dashboard starts without them.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from werkzeug.utils import secure_filename

SYSTEM_PROMPT = "You are a helpful AI assistant. Answer accurately and clearly. If you do not know something, say so."

RAG_PROMPT = """You are a helpful AI assistant answering questions using the user's uploaded documents.

Rules:
- Answer using only the supplied context.
- Do not invent facts.
- If the context does not answer the question, say that the information was not found.
- Cite supporting passages using [Source 1], [Source 2], etc.

Context:
{context}

Question:
{question}

Answer:
"""


class RAGError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 40) -> list[str]:
    """Word-window chunking, identical in behaviour to the notebook."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


@dataclass
class Document:
    doc_id: str
    name: str
    kind: str  # "pdf" | "txt"
    pages: int
    chunks: list = field(default_factory=list)
    index: object = None

    def info(self) -> dict:
        return {"doc_id": self.doc_id, "name": self.name, "kind": self.kind, "pages": self.pages, "chunks": len(self.chunks)}


class RAGService:
    def __init__(self, cfg):
        self.cfg = cfg
        self.upload_dir = Path(cfg["EXPERIMENTS_DIR"]) / "experiment_09_rag" / "uploads"
        self.allowed = set(cfg["ALLOWED_UPLOAD_EXTENSIONS"])
        self.docs: dict[str, Document] = {}
        self._lock = threading.RLock()
        self.embedder = None  # SentenceTransformer-compatible; injectable for tests
        self._gen = None
        self.gen_state = "not_loaded"
        self.gen_error = None

    # ---------------- status ----------------
    def page_context(self) -> dict:
        from experiments.common import has_module

        missing = [m for m in ("sentence_transformers", "faiss", "pymupdf") if not has_module(m)]
        return {
            "missing": missing,
            "documents": self.list_documents(),
            "embed_model": self.cfg["RAG_EMBED_MODEL"],
            "gen_model": self.cfg["RAG_GENERATION_MODEL"],
            "gen_enabled": self.cfg["RAG_GENERATION_ENABLED"],
            "gemini_ok": bool(self.cfg.get("GEMINI_API_KEY")),
            "gemini_model": self.cfg.get("GEMINI_MODEL"),
            "max_upload_mb": self.cfg["MAX_UPLOAD_MB"],
            "extensions": sorted(self.allowed),
        }

    def list_documents(self) -> list[dict]:
        with self._lock:
            return [d.info() for d in self.docs.values()]

    # ---------------- upload ----------------
    def add_document(self, filename: str, data: bytes) -> dict:
        display = secure_filename(filename or "")
        suffix = Path(display).suffix.lower()
        if not display or suffix not in self.allowed:
            raise RAGError("Unsupported file type. Upload a PDF or a plain-text (.txt) file.", 415)
        if not data:
            raise RAGError("The uploaded file is empty.", 400)
        if len(self.docs) >= self.cfg["RAG_MAX_DOCUMENTS"]:
            raise RAGError(f"Document limit reached ({self.cfg['RAG_MAX_DOCUMENTS']}). Remove one first.", 409)

        kind = "pdf" if suffix == ".pdf" else "txt"
        if kind == "pdf":
            pages = self._extract_pdf(data)
        else:
            pages = self._extract_txt(data)
        if not pages:
            raise RAGError(
                "No extractable text found. The file may be a scanned/image-only PDF (OCR is not supported) or empty.", 422
            )

        chunks = []
        for page in pages:
            for chunk_id, text in enumerate(chunk_text(page["text"], self.cfg["RAG_CHUNK_WORDS"], self.cfg["RAG_CHUNK_OVERLAP"])):
                chunks.append({"source": display, "page": page["page"], "chunk_id": chunk_id, "text": text})
        if not chunks:
            raise RAGError("No extractable text found in this document.", 422)

        embeddings = self._embed([c["text"] for c in chunks])
        index = self._build_index(embeddings)

        doc_id = uuid.uuid4().hex[:12]
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        stored = self.upload_dir / f"{doc_id}{suffix}"  # server-generated name; user filename never used on disk
        stored.write_bytes(data)

        doc = Document(doc_id=doc_id, name=display, kind=kind, pages=len(pages), chunks=chunks, index=index)
        with self._lock:
            self.docs[doc_id] = doc
        return doc.info()

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            doc = self.docs.pop(doc_id, None)
        if doc is None:
            raise RAGError("Document not found.", 404)
        for suffix in self.allowed:
            path = self.upload_dir / f"{doc_id}{suffix}"
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass

    # ---------------- extraction ----------------
    def _extract_pdf(self, data: bytes) -> list[dict]:
        if b"%PDF-" not in data[:1024]:
            raise RAGError("This file is not a valid PDF.", 422)
        try:
            try:
                import pymupdf
            except ImportError:
                import fitz as pymupdf  # older PyMuPDF
        except ImportError as exc:
            raise RAGError("PyMuPDF is not installed. Run: pip install -r requirements-ml.txt", 503) from exc
        try:
            pdf = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:  # noqa: BLE001
            raise RAGError("The PDF is corrupt or could not be opened.", 422) from exc
        try:
            if getattr(pdf, "needs_pass", False):
                raise RAGError("Password-protected PDFs are not supported.", 422)
            pages = []
            for number, page in enumerate(pdf, start=1):
                if number > self.cfg["RAG_MAX_PAGES"]:
                    break
                try:
                    text = page.get_text("text").strip()
                except Exception as exc:  # noqa: BLE001
                    raise RAGError("The PDF is corrupt: a page could not be read.", 422) from exc
                if text:
                    pages.append({"page": number, "text": text})
            return pages
        finally:
            pdf.close()

    @staticmethod
    def _extract_txt(data: bytes) -> list[dict]:
        if b"\x00" in data[:4096]:
            raise RAGError("This does not look like a plain-text file.", 422)
        text = data.decode("utf-8", errors="replace").strip()
        return [{"page": 1, "text": text}] if text else []

    # ---------------- embeddings / index ----------------
    def _get_embedder(self):
        if self.embedder is not None:
            return self.embedder
        with self._lock:
            if self.embedder is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RAGError("sentence-transformers is not installed. Run: pip install -r requirements-ml.txt", 503) from exc
                try:
                    self.embedder = SentenceTransformer(self.cfg["RAG_EMBED_MODEL"])
                except OSError as exc:
                    raise RAGError("Could not download the embedding model. Check your internet connection.", 503) from exc
        return self.embedder

    def _embed(self, texts: list[str]):
        import numpy as np

        embedder = self._get_embedder()
        try:
            vectors = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001
            raise RAGError("Embedding failed. See the server log for details.", 500) from exc
        return np.asarray(vectors, dtype="float32")

    @staticmethod
    def _build_index(embeddings):
        try:
            import faiss
        except ImportError as exc:
            raise RAGError("faiss-cpu is not installed. Run: pip install -r requirements-ml.txt", 503) from exc
        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(embeddings)
        return index

    # ---------------- retrieval ----------------
    def retrieve(self, question: str, doc_ids: list[str] | None = None, top_k: int | None = None) -> list[dict]:
        top_k = int(top_k or self.cfg["RAG_TOP_K"])
        with self._lock:
            if doc_ids:
                missing = [d for d in doc_ids if d not in self.docs]
                if missing:
                    raise RAGError("One of the selected documents no longer exists.", 404)
                selected = [self.docs[d] for d in doc_ids]
            else:
                selected = list(self.docs.values())
        if not selected:
            raise RAGError("Upload a document first.", 409)

        query = self._embed([question])
        hits = []
        for doc in selected:
            k = min(top_k, doc.index.ntotal)
            if k <= 0:
                continue
            scores, indices = doc.index.search(query, k)
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                item = dict(doc.chunks[int(idx)])
                item["score"] = float(score)
                item["kind"] = doc.kind
                hits.append(item)
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:top_k]

    # ---------------- generation ----------------
    def _load_generator(self):
        if self._gen is not None:
            return self._gen
        with self._lock:
            if self._gen is not None:
                return self._gen
            self.gen_state = "loading"
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                self.gen_state, self.gen_error = "error", "torch/transformers not installed"
                raise RAGError(self.gen_error, 503) from exc
            try:
                name = self.cfg["RAG_GENERATION_MODEL"]
                tokenizer = AutoTokenizer.from_pretrained(name)
                kwargs = {"device_map": "auto"}
                try:
                    model = AutoModelForCausalLM.from_pretrained(name, dtype="auto", **kwargs)
                except TypeError:
                    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", **kwargs)
                model.eval()
            except Exception as exc:  # noqa: BLE001
                self.gen_state, self.gen_error = "error", f"{type(exc).__name__}"
                raise RAGError("The generation model could not be loaded (download, memory or missing 'accelerate').", 503) from exc
            self._gen = (torch, tokenizer, model)
            self.gen_state = "ready"
            return self._gen

    def _generate(self, prompt: str) -> str:
        torch, tokenizer, model = self._load_generator()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=self.cfg["RAG_MAX_NEW_TOKENS"],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def answer(self, question: str, doc_ids: list[str] | None = None, top_k=None, generate: bool = True,
               engine: str = "local", gemini=None) -> dict:
        """engine: "local" (Qwen), "gemini" (cloud, via the `gemini(prompt)->str` callable) or "none"."""
        question = (question or "").strip()
        if not question:
            raise RAGError("Please enter a question.", 400)
        if len(question) > self.cfg["RAG_MAX_QUESTION_CHARS"]:
            raise RAGError(f"Question is too long (max {self.cfg['RAG_MAX_QUESTION_CHARS']} characters).", 400)
        try:
            top_k = max(1, min(10, int(top_k))) if top_k is not None else self.cfg["RAG_TOP_K"]
        except (TypeError, ValueError):
            top_k = self.cfg["RAG_TOP_K"]

        hits = self.retrieve(question, doc_ids, top_k)
        sources = [
            {
                "label": f"Source {i}",
                "source": h["source"],
                "page": h["page"],
                "location": f"PDF page {h['page']}" if h["kind"] == "pdf" else "text file",
                "similarity": round(h["score"], 3),
                "excerpt": h["text"],
            }
            for i, h in enumerate(hits, start=1)
        ]
        if not hits:
            return {"answer": None, "generation": {"used": False, "reason": "No relevant passages were found."}, "sources": []}

        if engine not in ("local", "gemini", "none"):
            raise RAGError("'engine' must be 'local', 'gemini' or 'none'.", 400)
        if not generate:
            engine = "none"
        generation = {"used": False, "engine": engine, "model": None, "reason": ""}
        answer = None
        if engine == "none":
            generation["reason"] = "Retrieval only: showing the retrieved passages without a generated answer."
        else:
            context = "\n\n".join(f"[{s['label']}: {s['source']}, {s['location']}]\n{s['excerpt']}" for s in sources)
            prompt = RAG_PROMPT.format(context=context, question=question)
            if engine == "local":
                generation["model"] = self.cfg["RAG_GENERATION_MODEL"]
                if not self.cfg["RAG_GENERATION_ENABLED"]:
                    generation["reason"] = "Local generation is disabled in the server configuration (RAG_GENERATION=off). Showing retrieved passages only."
                else:
                    try:
                        answer = self._generate(prompt)
                        generation["used"] = True
                    except RAGError as exc:
                        generation["reason"] = exc.message + " Showing retrieved passages only."
                    except Exception:  # noqa: BLE001
                        generation["reason"] = "Local generation failed (see server log). Showing retrieved passages only."
            else:  # gemini: only when the user explicitly chose it
                generation["model"] = "Google Gemini (cloud)"
                if gemini is None:
                    generation["reason"] = "Gemini is not available. Showing retrieved passages only."
                else:
                    try:
                        answer = gemini(prompt)
                        generation["used"] = True
                    except Exception as exc:  # noqa: BLE001 - AssistantError carries a safe message
                        generation["reason"] = (getattr(exc, "message", None) or "Gemini request failed.") + " Showing retrieved passages only."
        return {"answer": answer, "generation": generation, "sources": sources}


def create_service(cfg) -> RAGService:
    return RAGService(cfg)
