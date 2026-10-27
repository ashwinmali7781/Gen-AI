from flask import Blueprint, current_app, jsonify, request

from experiments.common import json_error

from .service import MusicError

bp = Blueprint("exp8_api", __name__, url_prefix="/api/exp8")


def _svc():
    return current_app.extensions["labspace"]["services"]["exp8"]


@bp.get("/clips")
def clips():
    return jsonify({"ok": True, "clips": _svc().saved_clips()})


@bp.post("/generate")
def generate():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return json_error("Send a JSON body with a 'prompt' field.", 400)
    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        return json_error("'prompt' must be a string.", 400)
    try:
        result = _svc().generate(prompt, payload.get("seconds", 8))
    except MusicError as exc:
        return json_error(exc.message, exc.status)
    return jsonify({"ok": True, **result})
