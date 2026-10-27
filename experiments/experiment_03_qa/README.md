# Experiment 3 — Transformer Q&A (TinyLlama)
Source notebook: `genai3_.ipynb` (TinyLlama-1.1B-Chat + Gradio). Here: Flask API + custom chat UI, same pretrained model.
- `service.py`: lazy, lock-guarded one-time load; CUDA if available else CPU; prompt via the model's chat template; clamped generation params; mapped errors (missing packages, download failure, OOM).
- `routes.py`: `GET /api/exp3/status`, `POST /api/exp3/load`, `POST /api/exp3/ask` (JSON `question`, optional `max_new_tokens`, `temperature`, `top_p`).
- Not tested here with the real model (no download). Test with `python diagnose.py --load-qa`.
- Note: the notebook header links Qwen3 in a markdown cell, but the code uses TinyLlama; TinyLlama is what is integrated.

- Gemini mode: choose "Gemini (cloud API)" on the page. Labelled in every answer; never used unless selected. Needs `GEMINI_API_KEY`.
