"""Startup diagnostics. Does NOT load any model unless you pass --load-qa.

    python diagnose.py            # environment + configuration report
    python diagnose.py --load-qa  # additionally load TinyLlama and answer one test question
"""
import argparse
import importlib.util
import platform
import sys
from pathlib import Path

from config import Config
from experiments import registry


def have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def hf_cached(model_id: str) -> bool:
    folder = "models--" + model_id.replace("/", "--")
    candidates = [Path.home() / ".cache" / "huggingface" / "hub" / folder]
    return any(p.is_dir() for p in candidates)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-qa", action="store_true", help="load TinyLlama and run one question")
    args = parser.parse_args()

    print(f"Python {platform.python_version()} on {platform.system()} {platform.release()}")
    if sys.version_info < (3, 10):
        print("  WARNING: Python 3.10+ is recommended.")

    print("\nPackages")
    for mod, label in [("flask", "Flask"), ("dotenv", "python-dotenv"), ("PIL", "Pillow"), ("huggingface_hub", "huggingface_hub"),
                       ("torch", "torch"), ("transformers", "transformers"), ("accelerate", "accelerate"),
                       ("sentence_transformers", "sentence-transformers"), ("faiss", "faiss-cpu"), ("pymupdf", "pymupdf"),
                       ("scipy", "scipy"), ("numpy", "numpy"), ("google.genai", "google-genai (optional)")]:
        try:
            ok = have(mod)
        except ModuleNotFoundError:
            ok = False
        print(f"  [{'x' if ok else ' '}] {label}")

    print("\nHardware")
    if have("torch"):
        import torch

        if torch.cuda.is_available():
            print(f"  CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  No CUDA GPU detected: local models (Exp 3, 8, 9) will run on the CPU and be slow.")
    else:
        print("  torch not installed: local models unavailable (run: pip install -r requirements-ml.txt).")

    print("\nConfiguration (values hidden)")
    print(f"  HF_TOKEN set:        {bool(Config.HF_TOKEN)}  (needed for Exp 4 hosted generation)")
    print(f"  GEMINI_API_KEY set:  {bool(Config.GEMINI_API_KEY)}  (optional assistant)")
    print(f"  .env file present:   {(Config.BASE_DIR / '.env').is_file()}")

    print("\nModel cache (first use downloads weights)")
    for label, model in [("Exp 3 TinyLlama", Config.QA_MODEL), ("Exp 8 MusicGen", Config.MUSICGEN_MODEL),
                         ("Exp 9 embeddings", Config.RAG_EMBED_MODEL), ("Exp 9 generator", Config.RAG_GENERATION_MODEL)]:
        print(f"  {label:18s} {model}: {'cached' if hf_cached(model) else 'not downloaded yet'}")

    print("\nExperiments")
    cfg = {"EXPERIMENTS_DIR": Config.EXPERIMENTS_DIR, "HF_TOKEN": Config.HF_TOKEN}
    for exp in registry.EXPERIMENTS:
        st = registry.describe_status(exp, cfg)
        print(f"  Exp {exp['number']}: {st['label']} — {st['detail']}")

    if args.load_qa:
        print("\nLoading TinyLlama (this can take minutes on first run)...")
        from experiments.experiment_03_qa.service import QAError, QAService

        svc = QAService({k: getattr(Config, k) for k in dir(Config) if k.isupper()})
        try:
            result = svc.ask("Say hello in one short sentence.", max_new_tokens=32)
            print("  OK:", result["answer"], f"({result['seconds']}s on {result['device']})")
        except QAError as exc:
            print("  FAILED:", exc.message)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

