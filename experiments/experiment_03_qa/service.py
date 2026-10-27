"""Experiment 3 — TinyLlama chat Q&A through Hugging Face Transformers.

The model is loaded lazily, exactly once per process, and guarded by a lock.
Importing this module never imports torch/transformers, so the dashboard starts
even when those packages are absent.
"""
from __future__ import annotations

import threading
import time

from experiments.common import clamp

SYSTEM_PROMPT = "You are a helpful, concise assistant. If you are unsure, say so."


class QAError(Exception):
    """Error carrying an HTTP status and a user-safe message."""

    def __init__(self, message: str, status: int = 500):
        super().__init__(message)
        self.message = message
        self.status = status


class QAService:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_id = cfg["QA_MODEL"]
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._torch = None
        self.state = "not_loaded"  # not_loaded | loading | ready | error
        self.device = None
        self.error: str | None = None

    # ---------------- status ----------------
    def status(self) -> dict:
        cuda = None
        try:
            import torch  # noqa: WPS433

            cuda = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            pass
        return {
            "state": self.state,
            "model": self.model_id,
            "device": self.device or ("cuda" if cuda else "cpu" if cuda is not None else "unknown"),
            "cuda_available": cuda,
            "error": self.error,
            "cpu_warning": cuda is False,
        }

    def page_context(self) -> dict:
        from experiments.common import has_module

        local_ok = has_module("torch") and has_module("transformers")
        gemini_ok = bool(self.cfg.get("GEMINI_API_KEY"))
        return {
            "qa_status": self.status(),
            "defaults": self.defaults(),
            "local_ok": local_ok,
            "gemini_ok": gemini_ok,
            "gemini_model": self.cfg.get("GEMINI_MODEL"),
            # Prefer the required local model when it is installed; otherwise Gemini if a key exists.
            "default_engine": "local" if (local_ok or not gemini_ok) else "gemini",
        }

    def validate_question(self, question) -> str:
        question = (question or "").strip()
        if not question:
            raise QAError("Please enter a question.", 400)
        if len(question) > self.cfg["QA_MAX_QUESTION_CHARS"]:
            raise QAError(f"Question is too long (max {self.cfg['QA_MAX_QUESTION_CHARS']} characters).", 400)
        return question

    def defaults(self) -> dict:
        return {
            "max_new_tokens": self.cfg["QA_MAX_NEW_TOKENS"],
            "temperature": self.cfg["QA_TEMPERATURE"],
            "top_p": self.cfg["QA_TOP_P"],
        }

    # ---------------- loading ----------------
    def ensure_loaded(self) -> None:
        if self.state == "ready":
            return
        with self._lock:
            if self.state == "ready":
                return
            self.state = "loading"
            self.error = None
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                self._fail("torch/transformers are not installed. Run: pip install -r requirements-ml.txt", exc)
            try:
                use_cuda = torch.cuda.is_available()
                self.device = "cuda" if use_cuda else "cpu"
                tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                kwargs = {}
                if use_cuda:
                    kwargs["dtype"] = torch.float16
                try:
                    model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
                except TypeError:  # older transformers used torch_dtype
                    if use_cuda:
                        kwargs = {"torch_dtype": torch.float16}
                    model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
                model.to(self.device)
                model.eval()
            except OSError as exc:
                self._fail("Could not download or read the model. Check your internet connection and disk space (about 2.2 GB).", exc)
            except MemoryError as exc:
                self._fail("Not enough memory to load the model.", exc)
            except RuntimeError as exc:
                self._fail("Model could not be loaded (possibly out of memory).", exc)
            except Exception as exc:  # noqa: BLE001
                self._fail("Unexpected error while loading the model.", exc)
            self._tokenizer, self._model, self._torch = tokenizer, model, torch
            self.state = "ready"

    def _fail(self, message: str, exc: Exception):
        self.state = "error"
        self.error = message
        raise QAError(message, 503) from exc

    # ---------------- generation ----------------
    def build_prompt(self, question: str) -> str:
        """Format with the model's own chat template (Zephyr-style for TinyLlama-Chat)."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def ask(self, question: str, max_new_tokens=None, temperature=None, top_p=None) -> dict:
        question = self.validate_question(question)

        max_new_tokens = int(clamp(max_new_tokens, 16, 512, self.cfg["QA_MAX_NEW_TOKENS"]))
        temperature = clamp(temperature, 0.0, 1.5, self.cfg["QA_TEMPERATURE"])
        top_p = clamp(top_p, 0.1, 1.0, self.cfg["QA_TOP_P"])

        self.ensure_loaded()
        torch = self._torch
        started = time.time()
        try:
            prompt = self.build_prompt(question)
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self._tokenizer.eos_token_id,
            }
            if temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                gen_kwargs.update(do_sample=False)
            with self._lock, torch.inference_mode():
                output = self._model.generate(**inputs, **gen_kwargs)
            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except MemoryError as exc:
            raise QAError("Out of memory while generating. Try a shorter answer length.", 507) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise QAError("GPU/CPU ran out of memory while generating. Try a shorter answer length.", 507) from exc
            raise QAError("Generation failed. See the server log for details.", 500) from exc
        except Exception as exc:  # noqa: BLE001
            raise QAError("Generation failed. See the server log for details.", 500) from exc

        if not answer:
            answer = "(The model returned an empty answer. Try rephrasing the question.)"
        return {
            "answer": answer,
            "model": self.model_id,
            "device": self.device,
            "seconds": round(time.time() - started, 2),
            "params": {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p},
        }


def create_service(cfg) -> QAService:
    return QAService(cfg)
