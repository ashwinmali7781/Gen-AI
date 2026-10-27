import time

from flask import Blueprint, current_app, jsonify, request

from assistant.service import PROVIDER_LABEL, AssistantError
from experiments.common import json_error

from .service import SYSTEM_PROMPT, QAError

bp = Blueprint("exp3_api", __name__, url_prefix="/api/exp3")


def _svc():
    return current_app.extensions["labspace"]["services"]["exp3"]


@bp.get("/status")
def status():
    return jsonify({"ok": True, **_svc().status()})


@bp.post("/load")
def load():
    try:
        _svc().ensure_loaded()
    except QAError as exc:
        return json_error(exc.message, exc.status, **_svc().status())
    return jsonify({"ok": True, **_svc().status()})


@bp.post("/ask")
def ask():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return json_error("Send a JSON body with a 'question' field.", 400)
    question = payload.get("question")
    if not isinstance(question, str):
        return json_error("'question' must be a string.", 400)
    engine = payload.get("engine", "local")
    if engine not in ("local", "gemini"):
        return json_error("'engine' must be 'local' (TinyLlama) or 'gemini'.", 400)

    if engine == "gemini":
        # Explicit, user-chosen alternative; always labelled in the response. Never used silently.
        try:
            question = _svc().validate_question(question)
            gemini = current_app.extensions["labspace"]["services"]["assistant"]
            started = time.time()
            answer = gemini.generate(question, SYSTEM_PROMPT, max_output_tokens=int(payload.get("max_new_tokens") or 512))
        except QAError as exc:
            return json_error(exc.message, exc.status)
        except AssistantError as exc:
            return json_error(exc.message, exc.status)
        except (TypeError, ValueError):
            return json_error("'max_new_tokens' must be a number.", 400)
        return jsonify({"ok": True, "answer": answer, "engine": "gemini", "model": gemini.model(),
                        "device": "cloud (Gemini API)", "provider": PROVIDER_LABEL,
                        "seconds": round(time.time() - started, 2)})

    try:
        result = _svc().ask(
            question,
            max_new_tokens=payload.get("max_new_tokens"),
            temperature=payload.get("temperature"),
            top_p=payload.get("top_p"),
        )
    except QAError as exc:
        return json_error(exc.message, exc.status)
    return jsonify({"ok": True, "engine": "local", **result})
