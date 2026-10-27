"""Verified inventory of the integrated experiments.

Every entry was checked against the notebooks in
https://github.com/TheSarthakShinde/GENAI- (Experiment 1 is intentionally excluded).
"""
from __future__ import annotations

from pathlib import Path

from .common import IMAGE_EXTS, AUDIO_EXTS, has_module, list_files

REPO_URL = "https://github.com/TheSarthakShinde/GENAI-"
COLAB_BASE = "https://colab.research.google.com/github/TheSarthakShinde/GENAI-/blob/main/"

EXPERIMENTS: list[dict] = [
    {
        "id": "exp3",
        "number": 3,
        "slug": "transformer-qa",
        "title": "Transformer Q&A (TinyLlama)",
        "summary": "Chat with the pretrained TinyLlama-1.1B-Chat model through Hugging Face Transformers.",
        "tags": ["transformer", "llm", "chat", "tinyllama"],
        "notebook": "genai3_.ipynb",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "kind": "live",
        "package": "experiments.experiment_03_qa",
        "folder": "experiment_03_qa",
        "requires": ["torch", "transformers"],
        "gemini_mode": True,
        "uses": "Open-source model (local) or optional Gemini mode",
    },
    {
        "id": "exp4",
        "number": 4,
        "slug": "text-to-image",
        "title": "Text-to-Image Diffusion",
        "summary": "Generate images from a prompt with FLUX.1-schnell via hosted inference; the original SD v1.5 output is kept as a saved example.",
        "tags": ["diffusion", "image", "flux", "stable diffusion"],
        "notebook": "Exe_4_ImageGeneration.ipynb",
        "model": "black-forest-labs/FLUX.1-schnell (hosted) · original notebook: runwayml/stable-diffusion-v1-5",
        "kind": "hybrid",
        "package": "experiments.experiment_04_diffusion",
        "folder": "experiment_04_diffusion",
        "requires": ["huggingface_hub", "PIL"],
        "needs_token": True,
        "saved_dirs": ["outputs/saved"],
        "saved_exts": IMAGE_EXTS,
        "uses": "Open-source model (hosted via Hugging Face)",
    },
    {
        "id": "exp6",
        "number": 6,
        "slug": "dcgan-faces",
        "title": "DCGAN Face Generation (CelebA)",
        "summary": "Deep Convolutional GAN trained on CelebA. Browse the genuine training snapshots and generated faces from the notebook.",
        "tags": ["gan", "dcgan", "celeba", "faces"],
        "notebook": "genaiexp6.ipynb",
        "model": "DCGAN generator/discriminator trained from scratch (15 epochs, 64×64)",
        "kind": "saved",
        "package": "experiments.experiment_06_dcgan",
        "folder": "experiment_06_dcgan",
        "requires": [],
        "saved_dirs": ["outputs/dataset", "outputs/training", "outputs/results"],
        "saved_exts": IMAGE_EXTS,
        "uses": "Saved notebook outputs (no training here)",
    },
    {
        "id": "exp7",
        "number": 7,
        "slug": "stylegan-faces",
        "title": "StyleGAN Face Generation (CelebA)",
        "summary": "Style-based GAN with a mapping network and AdaIN. Compare early vs final training samples; training is not re-run.",
        "tags": ["gan", "stylegan", "celeba", "faces"],
        "notebook": "genai_exp7_.ipynb",
        "model": "Compact StyleGAN-style generator trained from scratch (15 epochs, 64×64)",
        "kind": "saved",
        "package": "experiments.experiment_07_stylegan",
        "folder": "experiment_07_stylegan",
        "requires": [],
        "saved_dirs": ["outputs/dataset", "outputs/training", "outputs/results"],
        "saved_exts": IMAGE_EXTS,
        "uses": "Saved notebook outputs (no training here)",
    },
    {
        "id": "exp8",
        "number": 8,
        "slug": "musicgen",
        "title": "Text-to-Music (MusicGen)",
        "summary": "Meta's pretrained MusicGen-small turns a text description into audio. Listen to the notebook's saved clips or generate a short one locally.",
        "tags": ["audio", "music", "musicgen", "transformer"],
        "notebook": "genai_exp_8.ipynb",
        "model": "facebook/musicgen-small",
        "kind": "hybrid",
        "package": "experiments.experiment_08_musicgen",
        "folder": "experiment_08_musicgen",
        "requires": ["torch", "transformers", "scipy"],
        "saved_dirs": ["outputs/audio"],
        "saved_exts": AUDIO_EXTS,
        "uses": "Open-source model (local) + saved clips",
    },
    {
        "id": "exp9",
        "number": 9,
        "slug": "pdf-rag",
        "title": "PDF Question Answering (RAG)",
        "summary": "Upload a PDF or text file, retrieve relevant chunks with MiniLM embeddings + FAISS, and answer from them.",
        "tags": ["rag", "faiss", "embeddings", "pdf", "retrieval"],
        "notebook": "genaiexp9rag.ipynb",
        "model": "all-MiniLM-L6-v2 + FAISS · generator: Qwen/Qwen2.5-3B-Instruct",
        "kind": "live",
        "package": "experiments.experiment_09_rag",
        "folder": "experiment_09_rag",
        "requires": ["sentence_transformers", "faiss", "pymupdf"],
        "gemini_mode": True,
        "uses": "Local retrieval + local Qwen or optional Gemini answers",
    },
]

_BY_ID = {e["id"]: e for e in EXPERIMENTS}


def get_experiment(exp_id: str) -> dict | None:
    return _BY_ID.get(exp_id)


def outputs_root(exp: dict, experiments_dir: Path) -> Path:
    return Path(experiments_dir) / exp["folder"] / "outputs"


def colab_url(exp: dict) -> str:
    return COLAB_BASE + exp["notebook"]


def saved_count(exp: dict, experiments_dir: Path) -> int:
    total = 0
    for rel in exp.get("saved_dirs", []):
        total += len(list_files(Path(experiments_dir) / exp["folder"] / rel, exp.get("saved_exts", ())))
    return total


def describe_status(exp: dict, config) -> dict:
    """Honest status. Never says 'working': dependency presence is not a successful test."""
    experiments_dir = Path(config["EXPERIMENTS_DIR"])
    missing = [m for m in exp.get("requires", []) if not has_module(m)]
    saved = saved_count(exp, experiments_dir) if exp.get("saved_dirs") else 0
    token_missing = exp.get("needs_token") and not config.get("HF_TOKEN")
    kind = exp["kind"]

    if kind == "saved":
        if saved:
            return {"level": "saved", "label": "Saved outputs", "detail": f"{saved} genuine notebook outputs available. No live inference.", "missing": []}
        return {"level": "setup", "label": "Outputs missing", "detail": "Saved output folders are empty; see the experiment README.", "missing": []}

    if kind == "hybrid":
        live_ok = not missing and not token_missing
        live_bits = []
        if missing:
            live_bits.append("missing packages: " + ", ".join(missing))
        if token_missing:
            live_bits.append("HF_TOKEN not set")
        if live_ok:
            detail = "Live generation configured (not yet tested). " if saved else "Live generation configured (not yet tested)."
            if saved:
                detail += f"{saved} saved outputs also available."
            return {"level": "ready", "label": "Live (untested) + saved", "detail": detail, "missing": []}
        if saved:
            return {"level": "saved", "label": "Saved outputs only", "detail": f"{saved} saved outputs available. Live generation off: " + "; ".join(live_bits) + ".", "missing": missing}
        return {"level": "setup", "label": "Setup required", "detail": "; ".join(live_bits) + ".", "missing": missing}

    # live
    gemini_ok = bool(config.get("GEMINI_API_KEY")) and exp.get("gemini_mode")
    if missing and gemini_ok and exp["id"] == "exp3":
        return {"level": "ready", "label": "Gemini mode available", "detail": "Local TinyLlama packages not installed; Gemini mode (cloud, no download) can answer. Not tested until you run it.", "missing": missing}
    if missing:
        return {"level": "setup", "label": "Setup required", "detail": "Missing packages: " + ", ".join(missing) + ". See README.", "missing": missing}
    return {"level": "ready", "label": "Live (dependencies found)", "detail": "Dependencies detected. Model downloads on first use; not tested until you run it.", "missing": []}
