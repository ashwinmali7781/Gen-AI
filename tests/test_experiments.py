"""Offline experiment tests. External APIs and ML models are mocked; nothing is downloaded."""
import io
import sys
import types
from pathlib import Path

import pytest

from experiments.common import clamp, safe_join
from experiments.experiment_09_rag.service import RAGError, RAGService, chunk_text

np = pytest.importorskip("numpy")


# ---------------- helpers ----------------
def test_safe_join(tmp_path):
    assert safe_join(tmp_path, "a/b.png") == (tmp_path / "a/b.png").resolve()
    for bad in ["../x", "a/../../x", "", "\x00", "/etc/passwd"]:
        assert safe_join(tmp_path, bad) is None


def test_clamp():
    assert clamp("abc", 0, 1, 0.5) == 0.5
    assert clamp(5, 0, 1, 0.5) == 1
    assert clamp(float("nan"), 0, 1, 0.5) == 0.5
    assert clamp(None, 0, 1, 0.3) == 0.3


# ---------------- Exp 3 ----------------
def test_qa_empty_and_invalid(client):
    assert client.post("/api/exp3/ask", json={"question": ""}).status_code == 400
    assert client.post("/api/exp3/ask", json={"question": "   "}).status_code == 400
    assert client.post("/api/exp3/ask", json={"question": 5}).status_code == 400
    assert client.post("/api/exp3/ask", json={}).status_code == 400
    assert client.post("/api/exp3/ask", data="not json", content_type="text/plain").status_code == 400
    r = client.post("/api/exp3/ask", json={"question": "x" * 2001})
    assert r.status_code == 400 and "too long" in r.get_json()["error"]


def test_qa_missing_torch_gives_clear_error(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    r = client.post("/api/exp3/ask", json={"question": "hello"})
    assert r.status_code == 503
    assert "not installed" in r.get_json()["error"]


class FakeTensor(list):
    shape = (1, 3)


def _install_fake_llm(monkeypatch):
    calls = {"loads": 0, "templates": []}

    class FakeBatch(dict):
        def to(self, device):
            return self

    class FakeTok:
        eos_token_id = 0

        @classmethod
        def from_pretrained(cls, name):
            calls["loads"] += 1
            return cls()

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            calls["templates"].append(messages)
            return "<|user|>" + messages[-1]["content"]

        def __call__(self, prompt, return_tensors=None):
            return FakeBatch(input_ids=types.SimpleNamespace(shape=(1, 3)))

        def decode(self, tokens, skip_special_tokens=True):
            return "  fake answer  "

    class FakeModel:
        @classmethod
        def from_pretrained(cls, name, **kw):
            calls["loads"] += 1
            return cls()

        def to(self, d):
            return self

        def eval(self):
            return self

        def generate(self, **kw):
            calls["gen"] = kw
            return [["a", "b", "c", "d"]]

    class Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False), float16="f16", inference_mode=lambda: Ctx()
    )
    fake_tf = types.SimpleNamespace(AutoTokenizer=FakeTok, AutoModelForCausalLM=FakeModel)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    return calls


def test_qa_model_loaded_once_and_chat_template_used(client, monkeypatch):
    calls = _install_fake_llm(monkeypatch)
    for q in ("one", "two", "three"):
        r = client.post("/api/exp3/ask", json={"question": q, "temperature": 0, "max_new_tokens": 9999})
        assert r.status_code == 200
        assert r.get_json()["answer"] == "fake answer"
    assert calls["loads"] == 2, "tokenizer + model each loaded exactly once for 3 questions"
    assert len(calls["templates"]) == 3
    assert calls["templates"][0][-1] == {"role": "user", "content": "one"}
    assert calls["gen"]["max_new_tokens"] == 512, "max_new_tokens clamped"
    assert calls["gen"]["do_sample"] is False
    status = client.get("/api/exp3/status").get_json()
    assert status["state"] == "ready" and status["device"] == "cpu" and status["cpu_warning"] is True


# ---------------- Exp 4 ----------------
def test_image_empty_prompt(client):
    assert client.post("/api/exp4/generate", json={"prompt": ""}).status_code == 400
    assert client.post("/api/exp4/generate", json={"prompt": 3}).status_code == 400
    assert client.post("/api/exp4/generate", json={"prompt": "x" * 501}).status_code == 400


def test_image_missing_token_message(client):
    r = client.post("/api/exp4/generate", json={"prompt": "a cat"})
    assert r.status_code == 503 and "HF_TOKEN" in r.get_json()["error"]


@pytest.fixture()
def token_app(tmp_path):
    from app import create_app
    from config import TestConfig

    class Cfg(TestConfig):
        EXPERIMENTS_DIR = tmp_path / "e"
        HF_TOKEN = "hf_dummy"

    return create_app(Cfg)


def _fake_hf(monkeypatch, behaviour):
    class FakeClient:
        def __init__(self, **kw):
            FakeClient.kw = kw

        def text_to_image(self, prompt, **kw):
            FakeClient.call = (prompt, kw)
            return behaviour()

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(InferenceClient=FakeClient))
    return FakeClient


def test_image_success_saves_png(token_app, monkeypatch):
    from PIL import Image

    fc = _fake_hf(monkeypatch, lambda: Image.new("RGB", (8, 8), "red"))
    c = token_app.test_client()
    r = c.post("/api/exp4/generate", json={"prompt": "a red square", "size": 512})
    body = r.get_json()
    assert r.status_code == 200 and body["ok"]
    assert fc.call[1]["model"] == "black-forest-labs/FLUX.1-schnell"
    assert fc.kw["api_key"] == "hf_dummy"
    assert "hf_dummy" not in r.get_data(as_text=True)
    got = c.get(body["url"])
    assert got.status_code == 200 and got.mimetype == "image/png"
    assert "attachment" in c.get(body["download_url"]).headers["Content-Disposition"]


def test_image_bad_size(token_app, monkeypatch):
    _fake_hf(monkeypatch, lambda: None)
    r = token_app.test_client().post("/api/exp4/generate", json={"prompt": "x", "size": 99999})
    assert r.status_code == 400


@pytest.mark.parametrize("status,expected", [(401, 401), (403, 401), (402, 402), (404, 404), (429, 429), (422, 400)])
def test_image_provider_errors_mapped(token_app, monkeypatch, status, expected):
    class HTTPErr(Exception):
        response = types.SimpleNamespace(status_code=status)

    def boom():
        raise HTTPErr("secret detail hf_dummy")

    _fake_hf(monkeypatch, boom)
    r = token_app.test_client().post("/api/exp4/generate", json={"prompt": "x"})
    assert r.status_code == expected
    assert "hf_dummy" not in r.get_data(as_text=True)


def test_image_timeout_and_garbage_response(token_app, monkeypatch):
    class InferenceTimeoutError(Exception):
        pass

    def timeout():
        raise InferenceTimeoutError()

    _fake_hf(monkeypatch, timeout)
    assert token_app.test_client().post("/api/exp4/generate", json={"prompt": "x"}).status_code == 504
    _fake_hf(monkeypatch, lambda: "not an image")
    assert token_app.test_client().post("/api/exp4/generate", json={"prompt": "x"}).status_code == 502


# ---------------- Exp 8 ----------------
def test_music_validation(client):
    assert client.post("/api/exp8/generate", json={"prompt": ""}).status_code == 400
    assert client.post("/api/exp8/generate", json={"prompt": "x" * 301}).status_code == 400
    assert client.post("/api/exp8/generate", json={"prompt": 1}).status_code == 400


def test_music_missing_packages_clear_error(client, monkeypatch):
    monkeypatch.setattr("experiments.common.has_module", lambda n: False)
    r = client.post("/api/exp8/generate", json={"prompt": "calm piano"})
    assert r.status_code == 503 and "Missing packages" in r.get_json()["error"]


def test_music_generation_mocked(client, monkeypatch):
    pytest.importorskip("scipy")
    monkeypatch.setattr("experiments.common.has_module", lambda n: True)

    class Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))

    def fake_pipeline(task, model, device):
        assert task == "text-to-audio" and model == "facebook/musicgen-small"

        def run(prompt, forward_params=None):
            run.params = forward_params
            return {"audio": np.sin(np.linspace(0, 100, 32000 * 2)).astype("float32")[None, :], "sampling_rate": 32000}

        return run

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=fake_pipeline))
    r = client.post("/api/exp8/generate", json={"prompt": "calm piano", "seconds": 4})
    body = r.get_json()
    assert r.status_code == 200 and body["seconds_audio"] == 2.0
    got = client.get(body["url"])
    assert got.status_code == 200 and got.data[:4] == b"RIFF"


def test_real_saved_clips_are_valid_wav():
    root = Path(__file__).resolve().parent.parent / "experiments/experiment_08_musicgen/outputs/audio"
    clips = sorted(root.glob("*.wav"))
    assert len(clips) == 4
    for c in clips:
        assert c.read_bytes()[:4] == b"RIFF"


# ---------------- Exp 9 ----------------
class FakeEmbedder:
    """Deterministic bag-of-words embedder so semantic-ish retrieval can be tested offline."""

    VOCAB = ["cat", "dog", "finance", "market", "sentiment", "pizza", "river", "mountain"]

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        out = []
        for t in texts:
            low = t.lower()
            v = np.array([low.count(w) for w in self.VOCAB], dtype="float32") + 1e-3
            out.append(v / np.linalg.norm(v))
        return np.vstack(out)


def make_pdf(pages: list[str]) -> bytes:
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture()
def rag(app):
    pytest.importorskip("faiss")
    svc = app.extensions["labspace"]["services"]["exp9"]
    svc.embedder = FakeEmbedder()
    return svc


def test_chunk_text_matches_notebook_behaviour():
    words = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_text(words, 180, 40)
    assert [len(c.split()) for c in chunks] == [180, 180, 120]
    assert chunks[1].split()[0] == "w140"  # overlap of 40 words
    assert chunk_text("", 180, 40) == []
    assert chunk_text("short text") == ["short text"]
    with pytest.raises(ValueError):
        chunk_text("a b c", 10, 10)


def test_upload_validation(client):
    def up(name, data):
        return client.post("/api/exp9/upload", data={"file": (io.BytesIO(data), name)}, content_type="multipart/form-data")

    assert up("virus.exe", b"MZ").status_code == 415
    assert up("notes.docx", b"PK").status_code == 415
    assert up("noext", b"hello").status_code == 415
    assert up("../../evil.pdf", b"not a pdf").status_code == 422
    assert up("empty.txt", b"").status_code == 400
    assert up("binary.txt", b"\x00\x01\x02" * 50).status_code == 422
    assert up("spaces.txt", b"   \n  ").status_code == 422
    assert client.post("/api/exp9/upload", data={}).status_code == 400


def test_corrupt_pdf_rejected(client):
    pytest.importorskip("pymupdf")
    r = client.post("/api/exp9/upload", data={"file": (io.BytesIO(b"%PDF-1.4 garbage garbage"), "bad.pdf")},
                    content_type="multipart/form-data")
    assert r.status_code == 422


def test_oversized_upload_rejected(app):
    app.config["MAX_CONTENT_LENGTH"] = 1024
    r = app.test_client().post("/api/exp9/upload", data={"file": (io.BytesIO(b"a" * 5000), "big.txt")},
                               content_type="multipart/form-data")
    assert r.status_code == 413 and "too large" in r.get_json()["error"].lower()


def test_pdf_without_text_reports_clearly(rag, client):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    doc.new_page()
    data = doc.tobytes()
    r = client.post("/api/exp9/upload", data={"file": (io.BytesIO(data), "blank.pdf")}, content_type="multipart/form-data")
    assert r.status_code == 422 and "No extractable text" in r.get_json()["error"]


def test_rag_end_to_end_retrieval_with_pages(rag, client):
    pdf = make_pdf(["The cat and the dog play. Cat cat dog.", "Finance market sentiment analysis. Market market.", "Pizza by the river."])
    r = client.post("/api/exp9/upload", data={"file": (io.BytesIO(pdf), "My Doc.pdf")}, content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    doc = r.get_json()["document"]
    assert doc["pages"] == 3 and doc["kind"] == "pdf" and doc["name"] == "My_Doc.pdf"

    r = client.post("/api/exp9/ask", json={"question": "What about the market and finance?", "generate": False})
    body = r.get_json()
    assert r.status_code == 200 and body["answer"] is None
    assert body["sources"][0]["page"] == 2, "semantic top hit is the finance page"
    assert "Market" in body["sources"][0]["excerpt"] or "market" in body["sources"][0]["excerpt"].lower()
    assert body["generation"]["used"] is False and body["generation"]["reason"]
    assert 0 < body["sources"][0]["similarity"] <= 1.0001


def test_rag_uploaded_file_stored_under_server_name(rag, client, app):
    r = client.post("/api/exp9/upload", data={"file": (io.BytesIO(b"cat dog"), "../../weird name.txt")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    stored = list((app.config["EXPERIMENTS_DIR"] / "experiment_09_rag" / "uploads").iterdir())
    assert len(stored) == 1
    assert stored[0].stem == r.get_json()["document"]["doc_id"] and stored[0].suffix == ".txt"


def test_rag_validation_and_delete(rag, client):
    assert client.post("/api/exp9/ask", json={"question": "hi"}).status_code == 409  # nothing uploaded
    up = client.post("/api/exp9/upload", data={"file": (io.BytesIO(b"cat dog river"), "a.txt")}, content_type="multipart/form-data")
    doc_id = up.get_json()["document"]["doc_id"]
    assert client.post("/api/exp9/ask", json={"question": ""}).status_code == 400
    assert client.post("/api/exp9/ask", json={"question": "x" * 1001}).status_code == 400
    assert client.post("/api/exp9/ask", json={"question": "cat?", "doc_ids": "bad"}).status_code == 400
    assert client.post("/api/exp9/ask", json={"question": "cat?", "doc_ids": ["nope"]}).status_code == 404
    assert client.delete(f"/api/exp9/documents/{doc_id}").status_code == 200
    assert client.delete(f"/api/exp9/documents/{doc_id}").status_code == 404
    assert client.get("/api/exp9/documents").get_json()["documents"] == []


def test_rag_generation_failure_falls_back_honestly(rag, client, monkeypatch):
    client.post("/api/exp9/upload", data={"file": (io.BytesIO(b"cat dog river"), "a.txt")}, content_type="multipart/form-data")

    def fail(self, prompt):
        raise RAGError("The generation model could not be loaded.", 503)

    monkeypatch.setattr(RAGService, "_generate", fail)
    body = client.post("/api/exp9/ask", json={"question": "cat?", "generate": True}).get_json()
    assert body["answer"] is None and body["generation"]["used"] is False
    assert "could not be loaded" in body["generation"]["reason"] and body["sources"]


def test_rag_generation_prompt_is_grounded(rag, client, monkeypatch):
    client.post("/api/exp9/upload", data={"file": (io.BytesIO(b"cat dog river"), "a.txt")}, content_type="multipart/form-data")
    seen = {}
    monkeypatch.setattr(RAGService, "_generate", lambda self, prompt: seen.setdefault("p", prompt) and "grounded answer")
    body = client.post("/api/exp9/ask", json={"question": "cat?", "generate": True}).get_json()
    assert body["answer"] == "grounded answer" and body["generation"]["used"] is True
    assert "[Source 1: a.txt" in seen["p"] and "Do not invent facts" in seen["p"]


# ---------------- Assistant ----------------
def test_assistant_missing_key_and_validation(client):
    assert client.post("/api/assistant/ask", json={"question": ""}).status_code == 400
    r = client.post("/api/assistant/ask", json={"question": "what is a GAN?"})
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.get_json()["error"]
    assert client.post("/api/assistant/ask", json={}).status_code == 400


# ---------------- Gemini mode (SDK mocked: no real key, no network) ----------------
@pytest.fixture()
def gemini_app(tmp_path):
    from app import create_app
    from config import TestConfig

    class Cfg(TestConfig):
        EXPERIMENTS_DIR = tmp_path / "e"
        GEMINI_API_KEY = "AIza_TEST_KEY_NOT_REAL"
        GEMINI_MODEL = "test-model"

    return create_app(Cfg)


def _fake_genai(monkeypatch, text="gemini says hi", exc=None):
    seen = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            seen.update(model=model, contents=contents, system=config.system_instruction)
            if exc:
                raise exc
            return types.SimpleNamespace(text=text)

    class FakeClient:
        def __init__(self, api_key):
            seen["key"] = api_key
            self.models = FakeModels()

    genai = types.SimpleNamespace(Client=FakeClient)
    gtypes = types.SimpleNamespace(GenerateContentConfig=lambda **kw: types.SimpleNamespace(**kw))
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)
    genai.types = gtypes
    return seen


def test_exp3_gemini_mode_labelled_and_uses_key_server_side(gemini_app, monkeypatch):
    seen = _fake_genai(monkeypatch)
    r = gemini_app.test_client().post("/api/exp3/ask", json={"question": "hello", "engine": "gemini"})
    body = r.get_json()
    assert r.status_code == 200 and body["answer"] == "gemini says hi"
    assert body["engine"] == "gemini" and body["model"] == "test-model" and "Gemini" in body["provider"]
    assert seen["key"] == "AIza_TEST_KEY_NOT_REAL" and seen["contents"] == "hello"
    assert "AIza_TEST" not in r.get_data(as_text=True), "key never returned to the browser"


def test_exp3_gemini_validation_and_missing_key(client, gemini_app, monkeypatch):
    assert gemini_app.test_client().post("/api/exp3/ask", json={"question": "", "engine": "gemini"}).status_code == 400
    assert gemini_app.test_client().post("/api/exp3/ask", json={"question": "hi", "engine": "bogus"}).status_code == 400
    r = client.post("/api/exp3/ask", json={"question": "hi", "engine": "gemini"})  # no key configured
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.get_json()["error"]


@pytest.mark.parametrize("code,status", [(403, 401), (404, 404), (429, 429), (500, 502)])
def test_gemini_errors_mapped_without_leaking(gemini_app, monkeypatch, code, status):
    class APIError(Exception):
        pass

    err = APIError("boom AIza_TEST_KEY_NOT_REAL")
    err.code = code
    _fake_genai(monkeypatch, exc=err)
    r = gemini_app.test_client().post("/api/exp3/ask", json={"question": "hi", "engine": "gemini"})
    assert r.status_code == status and "AIza_TEST" not in r.get_data(as_text=True)


def test_gemini_empty_response_is_an_error(gemini_app, monkeypatch):
    _fake_genai(monkeypatch, text="")
    assert gemini_app.test_client().post("/api/exp3/ask", json={"question": "hi", "engine": "gemini"}).status_code == 502


def test_exp3_default_engine_and_hint_rendering(client, gemini_app, monkeypatch):
    monkeypatch.setattr("experiments.common.has_module", lambda n: False)  # simulate no torch
    page = client.get("/experiment/exp3").data
    assert b"No Gemini key yet" in page and b"aistudio.google.com/apikey" in page
    page2 = gemini_app.test_client().get("/experiment/exp3").data
    assert b"Gemini key found" in page2 and b"AIza_TEST" not in page2
    assert b'value="gemini" checked' in page2, "Gemini preselected only because torch is missing"


def test_exp9_gemini_answer_labelled_and_grounded(gemini_app, monkeypatch):
    pytest.importorskip("faiss")
    seen = _fake_genai(monkeypatch, text="doc answer [Source 1]")
    svc = gemini_app.extensions["labspace"]["services"]["exp9"]
    svc.embedder = FakeEmbedder()
    c = gemini_app.test_client()
    c.post("/api/exp9/upload", data={"file": (io.BytesIO(b"cat cat dog"), "a.txt")}, content_type="multipart/form-data")
    body = c.post("/api/exp9/ask", json={"question": "what about cat?", "engine": "gemini"}).get_json()
    assert body["answer"] == "doc answer [Source 1]"
    g = body["generation"]
    assert g["used"] and g["engine"] == "gemini" and "sent to Google" in g["provider"]
    assert "[Source 1: a.txt" in seen["contents"] and "Do not invent facts" in seen["contents"]
    assert body["sources"], "retrieved passages still returned and shown"


def test_exp9_gemini_failure_falls_back_to_passages(gemini_app, monkeypatch):
    pytest.importorskip("faiss")

    class APIError(Exception):
        code = 429

    _fake_genai(monkeypatch, exc=APIError("quota"))
    svc = gemini_app.extensions["labspace"]["services"]["exp9"]
    svc.embedder = FakeEmbedder()
    c = gemini_app.test_client()
    c.post("/api/exp9/upload", data={"file": (io.BytesIO(b"cat dog"), "a.txt")}, content_type="multipart/form-data")
    body = c.post("/api/exp9/ask", json={"question": "cat?", "engine": "gemini"}).get_json()
    assert body["answer"] is None and body["generation"]["used"] is False
    assert "rate limit" in body["generation"]["reason"].lower() and body["sources"]


def test_exp9_engine_validation_and_no_silent_gemini(client, rag):
    client.post("/api/exp9/upload", data={"file": (io.BytesIO(b"cat dog"), "a.txt")}, content_type="multipart/form-data")
    assert client.post("/api/exp9/ask", json={"question": "cat?", "engine": "bogus"}).status_code == 400
    body = client.post("/api/exp9/ask", json={"question": "cat?", "engine": "none"}).get_json()
    assert body["answer"] is None and body["generation"]["engine"] == "none" and body["sources"]
    # old boolean flag still works and never routes to Gemini
    old = client.post("/api/exp9/ask", json={"question": "cat?", "generate": False}).get_json()
    assert old["generation"]["engine"] == "none"


def test_assistant_dashboard_uses_generate(gemini_app, monkeypatch):
    _fake_genai(monkeypatch, text="a GAN is ...")
    r = gemini_app.test_client().post("/api/assistant/ask", json={"question": "what is a GAN?"})
    assert r.status_code == 200 and "a GAN is" in r.get_json()["answer"]
