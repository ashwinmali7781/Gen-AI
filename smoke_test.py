"""Offline smoke test: starts the app in-process and requests every page/endpoint.

    python smoke_test.py

It never downloads a model, never calls a paid API, and works without any API keys.
Exit code 0 = all checks passed.
"""
import html
import sys

from app import create_app
from config import TestConfig
from experiments import registry


def main() -> int:
    app = create_app(TestConfig)
    client = app.test_client()
    results = []

    def check(label, ok, detail=""):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))

    r = client.get("/")
    check("GET / (dashboard)", r.status_code == 200 and b"LabSpace" in r.data, f"HTTP {r.status_code}")
    check("Dashboard lists all experiments", r.data.count(b'class="card"') == len(registry.EXPERIMENTS))

    for exp in registry.EXPERIMENTS:
        r = client.get(f"/experiment/{exp['id']}")
        check(f"GET /experiment/{exp['id']}", r.status_code == 200 and html.escape(exp["title"]).encode() in r.data, f"HTTP {r.status_code}")

    for bad in ("exp1", "exp2", "nope"):
        check(f"GET /experiment/{bad} -> 404", client.get(f"/experiment/{bad}").status_code == 404)

    for path in ("/static/css/style.css", "/static/js/main.js", "/static/js/qa.js", "/static/js/image_generation.js",
                 "/static/js/music_generation.js", "/static/js/rag.js", "/static/js/replay.js", "/static/images/logo.svg"):
        check(f"GET {path}", client.get(path).status_code == 200)

    r = client.get("/healthz")
    check("GET /healthz", r.status_code == 200 and r.get_json()["ok"] is True, f"load_errors={r.get_json()['load_errors']}")

    r = client.post("/api/exp3/ask", json={"question": ""})
    check("Exp 3 rejects empty question", r.status_code == 400)
    r = client.post("/api/exp4/generate", json={"prompt": "a cat"})
    check("Exp 4 reports missing HF_TOKEN clearly", r.status_code == 503 and "HF_TOKEN" in r.get_json()["error"])
    r = client.post("/api/exp9/upload", data={})
    check("Exp 9 rejects missing upload", r.status_code == 400)
    r = client.post("/api/assistant/ask", json={"question": "hi"})
    check("Assistant reports missing GEMINI_API_KEY", r.status_code == 503 and "GEMINI_API_KEY" in r.get_json()["error"])

    r = client.get("/outputs/exp8/audio/musicgen_clip_1.wav")
    check("Saved MusicGen clip is served", r.status_code == 200 and r.mimetype.startswith("audio/"))
    r = client.get("/outputs/exp6/../../config.py")
    check("Path traversal blocked", r.status_code == 404)

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
