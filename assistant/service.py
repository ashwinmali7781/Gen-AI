"""Google Gemini helper (optional).

Used for: the dashboard concept assistant, and the clearly-labelled "Gemini mode" of
Experiment 3 (chat) and Experiment 9 (answer generation) for machines that cannot
download the local models. It never runs unless the user picks it, and results are
always labelled as coming from Gemini. The API key stays on the server.
"""
from __future__ import annotations

SYSTEM_INSTRUCTION = (
    "You are a teaching assistant inside an AI/ML experiment dashboard. Explain machine-learning "
    "concepts (transformers, diffusion models, GANs, StyleGAN, MusicGen, RAG, embeddings, FAISS) "
    "clearly and briefly for a college student. If unsure, say so."
)

PROVIDER_LABEL = "Google Gemini (cloud API, not an open-source local model)"


class AssistantError(Exception):
    def __init__(self, message: str, status: int = 500):
        super().__init__(message)
        self.message = message
        self.status = status


class AssistantService:
    def __init__(self, cfg):
        self.cfg = cfg

    def enabled(self) -> bool:
        return bool(self.cfg.get("GEMINI_API_KEY"))

    def model(self) -> str:
        return self.cfg["GEMINI_MODEL"]

    def generate(self, prompt: str, system_instruction: str | None = None, max_output_tokens: int | None = None) -> str:
        """Single Gemini call. Raises AssistantError with a user-safe message (never the key)."""
        if not self.enabled():
            raise AssistantError(
                "Gemini is not configured. Paste your key into the .env file as GEMINI_API_KEY=... and restart the server.", 503)
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise AssistantError("google-genai is not installed. Run: pip install -r requirements-optional.txt", 503) from exc
        try:
            client = genai.Client(api_key=self.cfg["GEMINI_API_KEY"])
            config = types.GenerateContentConfig(
                system_instruction=system_instruction, max_output_tokens=max_output_tokens)
            response = client.models.generate_content(model=self.model(), contents=prompt, config=config)
            text = (getattr(response, "text", None) or "").strip()
        except Exception as exc:  # noqa: BLE001 - never leak provider errors or keys
            code = getattr(exc, "code", None)
            if code in (400,) and "API key" in str(exc):
                raise AssistantError("Gemini rejected the API key. Check GEMINI_API_KEY in .env.", 401) from exc
            if code in (401, 403):
                raise AssistantError("Gemini rejected the API key or it lacks access. Check GEMINI_API_KEY in .env.", 401) from exc
            if code == 404:
                raise AssistantError(f"Gemini model '{self.model()}' was not found. Set GEMINI_MODEL in .env to a current model name.", 404) from exc
            if code == 429:
                raise AssistantError("Gemini rate limit or quota reached. Wait a moment and try again.", 429) from exc
            raise AssistantError("The Gemini request failed. Check your internet connection and GEMINI_MODEL.", 502) from exc
        if not text:
            raise AssistantError("Gemini returned an empty response (it may have been blocked). Try rephrasing.", 502)
        return text

    def ask(self, question: str) -> dict:
        question = (question or "").strip()
        if not question:
            raise AssistantError("Please enter a question.", 400)
        if len(question) > 1500:
            raise AssistantError("Question is too long (max 1500 characters).", 400)
        text = self.generate(question, SYSTEM_INSTRUCTION)
        return {"answer": text, "model": self.model(), "provider": PROVIDER_LABEL}


def create_service(cfg) -> AssistantService:
    return AssistantService(cfg)
