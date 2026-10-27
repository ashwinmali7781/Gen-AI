# LabSpace — AI/ML Experiment Hub

One Flask web app that showcases six experiments from the notebook repository
[TheSarthakShinde/GENAI-](https://github.com/TheSarthakShinde/GENAI-). Built for a college practical demonstration on Windows + VS Code.
Experiment 1 is intentionally not included.

This is a **showcase**, not a training platform. Nothing is trained when you run it. Training results come from outputs
that were saved inside your original notebooks.

## 1. Experiment inventory (verified against the notebooks)

The notebooks were opened and read before building. Two experiments differ from the original project brief, so the
brief's names were **not** used. Real contents:

| # | Notebook | What it really is | In this app |
|---|----------|-------------------|-------------|
| 3 | `genai3_.ipynb` | Q&A chat with `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Gradio UI in the notebook) | **Live**: custom Flask chat, TinyLlama loaded once. Optional labelled **Gemini mode** (no download) |
| 4 | `Exe_4_ImageGeneration.ipynb` | Text-to-image with **Stable Diffusion v1.5** + Gradio | **Hosted FLUX.1-schnell** (free-to-try, no local weights) + your saved SD v1.5 output |
| 6 | `genaiexp6.ipynb` | **DCGAN on CelebA** (not a VAE) | **Saved outputs**: before/after, **training replay** (plays the 15 saved epoch snapshots), loss curves, generated faces |
| 7 | `genai_exp7_.ipynb` | **StyleGAN-style GAN on CelebA** (not text-to-video) | **Saved outputs**: before/after comparison, **training replay**, epochs, truncation, style mixing, noise variation |
| 8 | `genai_exp_8.ipynb` | **MusicGen text-to-music** (`facebook/musicgen-small`) (not VAE/MNIST) | **Saved clips** (4 genuine audio files) + optional live generation |
| 9 | `genaiexp9rag.ipynb` | PDF RAG: PyMuPDF → 180-word chunks → `all-MiniLM-L6-v2` → FAISS → `Qwen/Qwen2.5-3B-Instruct` | **Live**: upload PDF/TXT, ask, see retrieved passages. Answer engine: retrieval only / **Gemini** / local Qwen |

There is **no VAE/MNIST notebook and no text-to-video notebook** in the repository, so neither is in the app.
Decisions agreed with you: Exp 7 shows saved before/after images instead of re-running StyleGAN; Exp 8 uses MusicGen; Exp 4 uses a free, easy hosted model.

## 2. What was extracted from your notebooks

Extracted by script from the notebook output cells (genuine, unaltered): 1 SD v1.5 image, 20 DCGAN images, 20 StyleGAN images, 4 MusicGen WAV clips.

- The notebooks never saved an image from a completely *untrained* network. The "before training" side is the **end of epoch 1** (the earliest saved snapshot), and it is labelled that way.
- The **Training replay** player on Exp 6/7 just steps through those saved epoch images (play/pause/slider/speed). It is labelled "replay of saved snapshots, not a live run" on the page; nothing is trained or generated.
- Exp 6's training animation was not saved; one static frame from that cell is shown and labelled as such.
- MusicGen clip prompts are the ones written in the notebook, matched by cell order.

## 3. Features and honest limitations

- Status chips never say "working". They say things like *Saved outputs*, *Setup required*, *Live (dependencies found)*. A feature is only proven by you running it.
- **Not tested in the build environment:** real TinyLlama, real MusicGen, real Qwen/MiniLM downloads, and real FLUX calls (no GPU, no model downloads, no HF token there). These paths were tested with mocks only. See section 10.
- Exp 3/8/9 on CPU are slow (see section 8). Exp 9's 3B generator needs about 6 GB of memory; set `RAG_GENERATION_MODEL=Qwen/Qwen2.5-1.5B-Instruct` or `RAG_GENERATION=off` on weak machines. Retrieval-only mode still shows real FAISS results.
- Exp 9 keeps uploaded documents **in memory** (index cleared on restart). The original file is also stored under a server-generated name in `experiments/experiment_09_rag/uploads/`. Scanned (image-only) PDFs are rejected: no OCR.
- Exp 9 deliberately does **not** copy the notebook's hard-coded routing for your sample PDFs (e.g. "if 'futurewise' in question"). Retrieval is purely semantic over what you upload.
- Experiment 4: the hosted route needs a Hugging Face account/token and uses provider credits that may be limited or billed. Whether FLUX.1-schnell is available to *your* account/provider can only be known by trying. If it fails you get a clear error and the saved SD output stays visible.
- Flask's built-in server is for local demos only. Not for production.

## 4. Folder structure

```
ai_ml_experiment_hub/
  app.py  config.py  diagnose.py  smoke_test.py
  requirements.txt  requirements-ml.txt  requirements-dev.txt  requirements-optional.txt
  .env.example  .gitignore  README.md
  assistant/                      optional Gemini concept explainer
  experiments/
    registry.py  common.py  gallery.py
    experiment_03_qa/             TinyLlama chat           (routes.py, service.py, README.md)
    experiment_04_diffusion/      FLUX hosted + saved SD   outputs/saved, outputs/generated
    experiment_06_dcgan/          saved outputs only       outputs/{dataset,training,results}
    experiment_07_stylegan/       saved outputs only       outputs/{dataset,training,results}
    experiment_08_musicgen/       saved clips + live       outputs/audio, outputs/generated
    experiment_09_rag/            RAG                      uploads/
  templates/  static/{css,js,images}/  tests/
```

## 5. Install on Windows (PowerShell, VS Code terminal)

Use **Python 3.10 – 3.12** (3.11 is a safe choice). The suite was run here on Python 3.13; PyTorch/FAISS wheels for very new Python versions are not guaranteed, so prefer 3.11 for the ML packages.

```powershell
cd path\to\ai_ml_experiment_hub
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip

# A) Core: dashboard, galleries, saved outputs, hosted image generation (small install)
pip install -r requirements.txt

# B) Local models: Exp 3 TinyLlama, Exp 8 live MusicGen, Exp 9 RAG (large: PyTorch etc.)
pip install -r requirements-ml.txt

# C) Optional Gemini assistant
pip install -r requirements-optional.txt
```

NVIDIA GPU: install the CUDA build of PyTorch first using the selector at https://pytorch.org/get-started/locally/, then run step B. The plain `pip` wheel on Windows is CPU-only.

## 6. Configure keys

```powershell
Copy-Item .env.example .env
notepad .env
```

| Variable | Needed for | Notes |
|----------|-----------|-------|
| `FLASK_SECRET_KEY` | Flask | any long random string |
| `HF_TOKEN` | Exp 4 live generation | fine-grained token with **"Make calls to Inference Providers"**: https://huggingface.co/settings/tokens |
| `GEMINI_API_KEY` | optional assistant | leave empty to disable |

Keys are read only on the server and never put in pages or JavaScript. `.env` is git-ignored and excluded from the ZIP. The app starts fine with every key missing.

## 6a. Gemini mode (no big downloads) — where to paste your API key

If you do not want to download TinyLlama (2.2 GB) or Qwen (6 GB), use Gemini mode for Exp 3 and Exp 9 answers.

1. Create a key at https://aistudio.google.com/apikey (check Google's current free-tier limits yourself; I could not confirm them).
2. Open the **`.env`** file in the project folder and paste it: `GEMINI_API_KEY=your_key_here` (no quotes, no spaces). The `.env.example` has a marked "PASTE YOUR GEMINI API KEY HERE" block.
3. `pip install -r requirements-optional.txt`, then restart `python app.py`.
4. On Exp 3 or Exp 9 choose **Gemini (cloud API)** under "Answer engine".

Rules this app follows:
- Gemini is **never used silently**. You select it, and every answer is labelled "Answered by Gemini (cloud)". Exp 3 in Gemini mode is **not** the TinyLlama experiment; if your viva asks for TinyLlama, use the TinyLlama option.
- Exp 9 sends **the retrieved passages and your question to Google** in Gemini mode. Do not use it for confidential documents.
- Exp 9 retrieval (MiniLM + FAISS) always runs locally, so it still needs `requirements-ml.txt` (PyTorch, and about 90 MB for MiniLM). Gemini only replaces the 6 GB Qwen answer step.
- Exp 3 in Gemini mode needs only `requirements.txt` + `requirements-optional.txt` (no torch).
- Default model is `gemini-3.5-flash-lite`. If Google retires the name you get a "model was not found" message; set `GEMINI_MODEL=` in `.env` to a current model from https://ai.google.dev/models.
- Tested with the SDK mocked and the real `google-genai` 2.33 call shapes checked. A **live Gemini call was not made** (no key and no network in the build environment).

## 7. Run

```powershell
python diagnose.py          # optional: shows packages, GPU/CPU, keys set (values hidden), model cache
python app.py               # then open http://127.0.0.1:5000
```

`python diagnose.py --load-qa` additionally loads TinyLlama and answers one question (downloads ~2.2 GB first time).

## 8. First run, downloads, CPU vs GPU

| Feature | Download (first use) | Needs | CPU behaviour |
|---------|---------------------|-------|---------------|
| Exp 3 TinyLlama | ~2.2 GB | ~5 GB RAM (fp32 on CPU) | works; tens of seconds per answer or more |
| Exp 8 MusicGen-small | ~2.3 GB | a few GB RAM | very slow; a few seconds of audio can take minutes. Duration capped (default 15 s) |
| Exp 9 MiniLM embeddings | ~90 MB | small | fine |
| Exp 9 Qwen2.5-3B | ~6 GB | ~6+ GB RAM/VRAM | very slow on CPU; switch off or use the 1.5B model |
| Exp 4 FLUX | none (hosted) | internet + HF token | n/a |

Models are cached by Hugging Face (usually `C:\Users\<you>\.cache\huggingface`) and loaded **once per server process**. GPU is used automatically when CUDA is available.

## 9. Adding or replacing saved outputs

Drop files into the experiment's `outputs/` folders and reload; no code change. Images: `.png/.jpg/.webp`; audio: `.wav/.mp3`.
Epoch snapshots must be named `epoch_01_samples.png` … `epoch_15_samples.png` for the before/after comparison.
Empty folders show a message telling you where to put files.

## 10. Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest -q          # offline suite: no model download, no API calls, no keys
python smoke_test.py         # in-process request of every page and key endpoint
```

What the offline suite covers: all routes and 404s, static files, empty-folder galleries, path traversal and non-media blocking, missing-key errors, empty/invalid input to every API, upload validation (type, empty, binary, corrupt PDF, oversized, no-text PDF), a real PyMuPDF → chunking → FAISS retrieval run (with a stand-in embedder), TinyLlama loaded-once and chat-template use (mocked model), FLUX success/error mapping (mocked client), MusicGen WAV writing (mocked pipeline), token never appearing in pages, a broken experiment not breaking the dashboard, Gemini mode (labelling, key never returned, error mapping, no silent use, Exp 9 fallback to passages).

What it **cannot** prove: real model quality, real downloads, real Hugging Face provider availability, real Gemini answers. Run those yourself once (section 7) before the demo.

Packages the suite was run with (plus google-genai 2.33.0 for call-shape checks): Python 3.13.17, Flask 3.1.3, Werkzeug 3.1.9, python-dotenv 1.2.3, Pillow 12.3.0, huggingface_hub 2.3.0, numpy 2.5.3, scipy 1.18.1, PyMuPDF 2.0, faiss-cpu 1.15.1, pytest 9.1.1. `torch`, `transformers`, `sentence-transformers` and `accelerate` were **not** installed or tested here; `requirements-ml.txt` uses lower bounds only.

## 11. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `torch/transformers are not installed` | `pip install -r requirements-ml.txt` inside the activated venv |
| Exp 3 very slow | no GPU detected; install CUDA PyTorch or lower *Max new tokens* |
| `Could not download ... model` | check internet/disk space; retry (Hugging Face resumes) |
| Exp 4 `HF_TOKEN is not set` | create `.env` from `.env.example`, restart |
| Exp 4 `Authentication failed` | token needs the Inference Providers permission |
| Exp 4 `not available on the selected provider` / 402 / 429 | provider or credit issue; try `HF_IMAGE_PROVIDER=fal-ai` (or another) in `.env`, or wait |
| Exp 9 `No extractable text` | scanned PDF; use a text PDF or `.txt` |
| Exp 9 generator fails to load | pick **Gemini** or **Retrieval only** as the answer engine, or set `RAG_GENERATION_MODEL=Qwen/Qwen2.5-1.5B-Instruct` |
| `Gemini rejected the API key` | re-paste the key in `.env`, no quotes/spaces, restart |
| `Gemini model '...' was not found` | set `GEMINI_MODEL=` in `.env` to a current model name |
| Port 5000 busy | edit the `port=` in the last line of `app.py` |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

## 12. Licences and access

- TinyLlama, Qwen2.5-3B-Instruct, MusicGen (weights licence: check the model card, MusicGen weights are CC-BY-NC), all-MiniLM-L6-v2 and FLUX.1-schnell each have their own licence on their Hugging Face page. Check them before any use beyond a classroom demo.
- CelebA is research/non-commercial data; the face images shown are model outputs and dataset samples from your notebooks.
