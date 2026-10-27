from flask import Blueprint, current_app, jsonify, request

from experiments.common import json_error

from .service import AssistantError

bp = Blueprint("assistant_api", __name__, url_prefix="/api/assistant")


@bp.post("/ask")
def ask():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("question"), str):
        return json_error("Send a JSON body with a 'question' string.", 400)
    svc = current_app.extensions["labspace"]["services"]["assistant"]
    try:
        result = svc.ask(payload["question"])
    except AssistantError as exc:
        return json_error(exc.message, exc.status)
    return jsonify({"ok": True, **result})
