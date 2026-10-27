# Experiment 9 — PDF Question Answering (RAG)
Source notebook: `genaiexp9rag.ipynb`.
Preserved: PyMuPDF page text, 180-word chunks / 40 overlap, `sentence-transformers/all-MiniLM-L6-v2` (normalised), FAISS `IndexFlatIP`, `Qwen/Qwen2.5-3B-Instruct`, grounded prompt with [Source n] citations.
Changed: no sample-PDF keyword routing; multi-upload; upload validation; retrieval-only fallback if the generator cannot load.
`uploads/` stores files under server-generated names. Index is in memory only.

Answer engine (user-selected): retrieval only / Gemini (cloud; passages are sent to Google) / local Qwen. Retrieval always local.
