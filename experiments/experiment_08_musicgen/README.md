# Experiment 8 — Text-to-Music (MusicGen)
Source notebook: `genai_exp_8.ipynb` (`facebook/musicgen-small`, transformers text-to-audio pipeline).
- `outputs/audio/musicgen_clip_1..4.wav`: the notebook's real outputs (32 kHz mono). Prompts recorded in `service.py` by cell order.
- Live generation: same pipeline, loaded once, duration capped by `MUSICGEN_MAX_SECONDS`. Slow on CPU. Untested with the real model here.
