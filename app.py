"""LabSpace — AI/ML Experiment Hub. Flask application factory.

Run locally:  python app.py      (development server, local demos only)
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from config import Config
from experiments import registry
from experiments.common import SERVABLE_EXTS, safe_join

log = logging.getLogger("labspace")


def create_app(config_object=None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(config_object or Config)

    state = {"services": {}, "load_errors": {}}
    app.extensions["labspace"] = state

    # Output/upload folders are created on demand.
    for exp in registry.EXPERIMENTS:
        registry.outputs_root(exp, app.config["EXPERIMENTS_DIR"]).mkdir(parents=True, exist_ok=True)
    (Path(app.config["EXPERIMENTS_DIR"]) / "experiment_09_rag" / "uploads").mkdir(parents=True, exist_ok=True)

    _register_experiments(app, state)
    _register_assistant(app, state)
    _register_main_routes(app)
    _register_error_handlers(app)

    @app.context_processor
    def inject_globals():
        return {
            "app_name": "LabSpace",
            "app_tagline": "AI/ML Experiment Hub",
            "nav_experiments": registry.EXPERIMENTS,
            "gemini_enabled": bool(app.config.get("GEMINI_API_KEY")),
        }

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        return resp

    return app


def _register_experiments(app: Flask, state: dict) -> None:
    """A failure in one experiment must never take down the dashboard."""
    for exp in registry.EXPERIMENTS:
        try:
            service_mod = importlib.import_module(exp["package"] + ".service")
            routes_mod = importlib.import_module(exp["package"] + ".routes")
            state["services"][exp["id"]] = service_mod.create_service(app.config)
            app.register_blueprint(routes_mod.bp)
        except Exception as exc:  # noqa: BLE001 - isolate experiment failures
            log.exception("Experiment %s failed to load", exp["id"])
            state["load_errors"][exp["id"]] = f"{type(exc).__name__}: {exc}"


def _register_assistant(app: Flask, state: dict) -> None:
    try:
        from assistant import routes as a_routes, service as a_service

        state["services"]["assistant"] = a_service.create_service(app.config)
        app.register_blueprint(a_routes.bp)
    except Exception as exc:  # noqa: BLE001
        log.exception("Assistant failed to load")
        state["load_errors"]["assistant"] = f"{type(exc).__name__}: {exc}"


def _register_main_routes(app: Flask) -> None:
    @app.get("/")
    def index():
        cards = []
        for exp in registry.EXPERIMENTS:
            cards.append({**exp, "status": registry.describe_status(exp, app.config), "colab": registry.colab_url(exp)})
        return render_template("index.html", cards=cards, repo_url=registry.REPO_URL)

    @app.get("/experiment/<exp_id>")
    def experiment_detail(exp_id: str):
        exp = registry.get_experiment(exp_id)
        if exp is None:
            abort(404)
        state = app.extensions["labspace"]
        service = state["services"].get(exp_id)
        context = {}
        if service is not None and hasattr(service, "page_context"):
            try:
                context = service.page_context()
            except Exception as exc:  # noqa: BLE001
                log.exception("page_context failed for %s", exp_id)
                state["load_errors"][exp_id] = f"{type(exc).__name__}: {exc}"
        return render_template(
            "experiment.html",
            exp=exp,
            status=registry.describe_status(exp, app.config),
            colab=registry.colab_url(exp),
            load_error=state["load_errors"].get(exp_id),
            ctx=context,
            repo_url=registry.REPO_URL,
        )

    @app.get("/outputs/<exp_id>/<path:relative>")
    def serve_output(exp_id: str, relative: str):
        """Serve saved/generated media from an experiment's outputs folder only."""
        exp = registry.get_experiment(exp_id)
        if exp is None:
            abort(404)
        root = registry.outputs_root(exp, app.config["EXPERIMENTS_DIR"])
        target = safe_join(root, relative)
        if target is None or not target.is_file() or target.suffix.lower() not in SERVABLE_EXTS:
            abort(404)
        as_attachment = request.args.get("download") == "1"
        return send_from_directory(root.resolve(), target.relative_to(root.resolve()).as_posix(), as_attachment=as_attachment)

    @app.get("/healthz")
    def healthz():
        state = app.extensions["labspace"]
        return jsonify(
            {
                "ok": True,
                "experiments": len(registry.EXPERIMENTS),
                "load_errors": state["load_errors"],
                "hf_token_configured": bool(app.config.get("HF_TOKEN")),
                "gemini_configured": bool(app.config.get("GEMINI_API_KEY")),
            }
        )


def _register_error_handlers(app: Flask) -> None:
    def wants_json() -> bool:
        return request.path.startswith("/api/")

    @app.errorhandler(404)
    def not_found(_e):
        if wants_json():
            return jsonify({"ok": False, "error": "Not found."}), 404
        return render_template("404.html"), 404

    @app.errorhandler(413)
    def too_large(_e):
        limit = app.config["MAX_UPLOAD_MB"]
        message = f"File is too large. The limit is {limit} MB."
        if wants_json():
            return jsonify({"ok": False, "error": message}), 413
        return render_template("errors/500.html", message=message, code=413), 413

    @app.errorhandler(405)
    def method_not_allowed(_e):
        if wants_json():
            return jsonify({"ok": False, "error": "Method not allowed."}), 405
        return render_template("404.html"), 405

    @app.errorhandler(500)
    def server_error(_e):
        if wants_json():
            return jsonify({"ok": False, "error": "Internal server error."}), 500
        return render_template("errors/500.html", message="Something went wrong on the server.", code=500), 500


app = create_app()

if __name__ == "__main__":
    # Flask's development server: local demonstrations only, not production.
    app.run(host="127.0.0.1", port=5000, debug=False)
