"""Central configuration. Secrets come only from environment variables / .env."""
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class Config:
    BASE_DIR = BASE_DIR
    EXPERIMENTS_DIR = BASE_DIR / "experiments"

    # A random key is generated when none is configured (sessions are not used for anything critical).
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
    TESTING = False

    # ---- secrets (server-side only, never rendered into templates) ----
    HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

    # ---- uploads ----
    MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 16)
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".txt"}

    # ---- Experiment 3: TinyLlama chat Q&A ----
    QA_MODEL = os.environ.get("QA_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    QA_MAX_NEW_TOKENS = _int("QA_MAX_NEW_TOKENS", 256)
    QA_TEMPERATURE = _float("QA_TEMPERATURE", 0.7)
    QA_TOP_P = _float("QA_TOP_P", 0.9)
    QA_MAX_QUESTION_CHARS = 2000

    # ---- Experiment 4: hosted diffusion (Hugging Face Inference Providers) ----
    IMAGE_MODEL = os.environ.get("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
    IMAGE_PROVIDER = os.environ.get("HF_IMAGE_PROVIDER", "auto")
    IMAGE_TIMEOUT_S = _int("HF_IMAGE_TIMEOUT_S", 120)
    IMAGE_MAX_PROMPT_CHARS = 500
    IMAGE_KEEP_LAST = 40  # generated images retained on disk

    # ---- Experiment 8: MusicGen ----
    MUSICGEN_MODEL = os.environ.get("MUSICGEN_MODEL", "facebook/musicgen-small")
    MUSICGEN_MAX_SECONDS = _int("MUSICGEN_MAX_SECONDS", 15)
    MUSICGEN_MAX_PROMPT_CHARS = 300
    MUSICGEN_KEEP_LAST = 20

    # ---- Experiment 9: RAG (defaults mirror the original notebook) ----
    RAG_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    RAG_GENERATION_MODEL = os.environ.get("RAG_GENERATION_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    RAG_GENERATION_ENABLED = os.environ.get("RAG_GENERATION", "on").lower() not in {"off", "0", "false"}
    RAG_CHUNK_WORDS = 180
    RAG_CHUNK_OVERLAP = 40
    RAG_TOP_K = 5
    RAG_MAX_PAGES = 600
    RAG_MAX_QUESTION_CHARS = 1000
    RAG_MAX_DOCUMENTS = 10
    RAG_MAX_NEW_TOKENS = _int("RAG_MAX_NEW_TOKENS", 250)


class TestConfig(Config):
    TESTING = True
    SECRET_KEY = "test-secret"
    HF_TOKEN = ""
    GEMINI_API_KEY = ""
