import logging

from flask import Blueprint, current_app, jsonify, request

from experiments.common import json_error

from .service import ImageError

bp = Blueprint("exp4_api", __name__, url_prefix="/api/exp4")
log = logging.getLogger("labspace.exp4")


def _svc():
    return current_app.extensions["labspace"]["services"]["exp4"]


@bp.get("/status")
def status():
    ctx = _svc().page_context()
    return jsonify({"ok": True, "live_configured": ctx["live_configured"], "reason": ctx["live_reason"], "model": ctx["model"]})


@bp.post("/generate")
def generate():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return json_error("Send a JSON body with a 'prompt' field.", 400)
    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        return json_error("'prompt' must be a string.", 400)
    try:
        result = _svc().generate(prompt, payload.get("size", 768))
    except ImageError as exc:
        return json_error(exc.message, exc.status)
    return jsonify({"ok": True, **result})
