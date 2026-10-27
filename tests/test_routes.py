import pytest

from experiments import registry

IDS = [e["id"] for e in registry.EXPERIMENTS]


def test_dashboard_ok(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.data.count(b'class="card"') == 6


def test_experiment_one_excluded():
    assert "exp1" not in IDS
    assert IDS == ["exp3", "exp4", "exp6", "exp7", "exp8", "exp9"]


@pytest.mark.parametrize("exp_id", IDS)
def test_every_experiment_page_loads(client, exp_id):
    assert client.get(f"/experiment/{exp_id}").status_code == 200


@pytest.mark.parametrize("bad", ["exp1", "exp2", "exp10", "nope", "..%2Fetc"])
def test_unknown_experiment_404(client, bad):
    assert client.get(f"/experiment/{bad}").status_code == 404


@pytest.mark.parametrize("path", [
    "/static/css/style.css", "/static/js/main.js", "/static/js/qa.js", "/static/js/image_generation.js",
    "/static/js/music_generation.js", "/static/js/rag.js", "/static/js/replay.js", "/static/images/logo.svg"])
def test_static_files(client, path):
    assert client.get(path).status_code == 200


def test_unknown_route_404_html_and_json(client):
    assert b"Page not found" in client.get("/nope").data
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.get_json()["ok"] is False


def test_healthz_no_secrets(client):
    body = client.get("/healthz").get_json()
    assert body["ok"] and body["load_errors"] == {}
    assert body["hf_token_configured"] is False and body["gemini_configured"] is False


def test_pages_never_contain_token(tmp_path):
    from app import create_app
    from config import TestConfig

    class Cfg(TestConfig):
        EXPERIMENTS_DIR = tmp_path / "e"
        HF_TOKEN = "hf_SECRET_TOKEN_VALUE"
        GEMINI_API_KEY = "AIza_SECRET_KEY_VALUE"

    c = create_app(Cfg).test_client()
    for path in ["/", *[f"/experiment/{i}" for i in IDS], "/healthz"]:
        data = c.get(path).data
        assert b"SECRET_TOKEN_VALUE" not in data and b"SECRET_KEY_VALUE" not in data


def test_optional_services_do_not_block_dashboard(client):
    assert client.get("/").status_code == 200
    assert b"Assistant not configured" in client.get("/").data


def test_broken_experiment_does_not_break_dashboard(tmp_path, monkeypatch):
    import importlib

    from app import create_app
    from config import TestConfig

    mod = importlib.import_module("experiments.experiment_03_qa.service")
    monkeypatch.setattr(mod, "create_service", lambda cfg: (_ for _ in ()).throw(RuntimeError("boom")))

    class Cfg(TestConfig):
        EXPERIMENTS_DIR = tmp_path / "e"

    c = create_app(Cfg).test_client()
    assert c.get("/").status_code == 200
    page = c.get("/experiment/exp3")
    assert page.status_code == 200 and b"failed to load" in page.data
    assert c.get("/experiment/exp6").status_code == 200


def test_output_path_traversal_blocked(client, app):
    root = app.config["EXPERIMENTS_DIR"] / "experiment_08_musicgen" / "outputs"
    (root / "audio").mkdir(parents=True, exist_ok=True)
    (root.parent / "secret.wav").write_bytes(b"RIFFxxxx")
    for url in ["/outputs/exp8/../secret.wav", "/outputs/exp8/audio/../../secret.wav",
                "/outputs/exp8/%2e%2e/secret.wav", "/outputs/exp8//etc/passwd", "/outputs/exp8/audio/..%2f..%2fsecret.wav"]:
        assert client.get(url, follow_redirects=True).status_code == 404, url


def test_output_rejects_non_media_extension(client, app):
    root = app.config["EXPERIMENTS_DIR"] / "experiment_06_dcgan" / "outputs"
    root.mkdir(parents=True, exist_ok=True)
    (root / "notes.py").write_text("print(1)")
    assert client.get("/outputs/exp6/notes.py").status_code == 404


def test_output_serves_and_downloads(client, app):
    d = app.config["EXPERIMENTS_DIR"] / "experiment_08_musicgen" / "outputs" / "audio"
    d.mkdir(parents=True, exist_ok=True)
    (d / "a.wav").write_bytes(b"RIFF....WAVE")
    assert client.get("/outputs/exp8/audio/a.wav").status_code == 200
    r = client.get("/outputs/exp8/audio/a.wav?download=1")
    assert "attachment" in r.headers.get("Content-Disposition", "")


def test_saved_galleries_handle_empty_folders(client):
    for exp_id in ("exp6", "exp7"):
        r = client.get(f"/experiment/{exp_id}")
        assert r.status_code == 200
        assert b"No images found" in r.data
        assert b"Comparison unavailable" in r.data
    assert b"No saved clips found" in client.get("/experiment/exp8").data
    assert b"No saved outputs" in client.get("/experiment/exp4").data


def test_real_saved_outputs_present(real_client):
    r6 = real_client.get("/experiment/exp6").data
    r7 = real_client.get("/experiment/exp7").data
    assert b"No images found" not in r6 and b"No images found" not in r7
    assert b"Before vs after training" in r7
    assert real_client.get("/experiment/exp8").data.count(b"<audio") >= 4


def test_status_never_claims_working(real_client):
    page = real_client.get("/").data.lower()
    assert b">working<" not in page and b"fully working" not in page


def test_no_secret_in_repo_files():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    assert not (root / ".env").exists()
    pattern = re.compile(r"hf_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_\-]{30,}")
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".js", ".html", ".md", ".txt", ".example", ".css", ".svg"} and "__pycache__" not in p.parts:
            assert not pattern.search(p.read_text(encoding="utf-8", errors="ignore")), p


@pytest.mark.parametrize("exp_id", ["exp6", "exp7"])
def test_replay_real_frames_and_honest_label(real_client, exp_id):
    html = real_client.get(f"/experiment/{exp_id}").data
    assert b'id="replay"' in html and b"replay of saved snapshots" in html.lower()
    assert b"epoch_15_samples.png" in html and b"not a live run" in html
    assert b"/static/js/replay.js" in html


@pytest.mark.parametrize("exp_id", ["exp6", "exp7"])
def test_replay_empty_folder_message(client, exp_id):
    html = client.get(f"/experiment/{exp_id}").data
    assert b"No snapshots found" in html and b'id="replay"' not in html
