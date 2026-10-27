from flask import Blueprint, current_app, jsonify, request

from experiments.common import json_error

from .service import RAGError

bp = Blueprint("exp9_api", __name__, url_prefix="/api/exp9")


def _svc():
    return current_app.extensions["labspace"]["services"]["exp9"]


@bp.get("/documents")
def documents():
    return jsonify({"ok": True, "documents": _svc().list_documents()})


@bp.post("/upload")
def upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        return json_error("No file was uploaded. Choose a PDF or .txt file.", 400)
    try:
        info = _svc().add_document(file.filename, file.read())
    except RAGError as exc:
        return json_error(exc.message, exc.status)
    return jsonify({"ok": True, "document": info, "documents": _svc().list_documents()})


@bp.delete("/documents/<doc_id>")
def delete_document(doc_id: str):
    try:
        _svc().delete_document(doc_id)
    except RAGError as exc:
        return json_error(exc.message, exc.status)
    return jsonify({"ok": True, "documents": _svc().list_documents()})


@bp.post("/ask")
def ask():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return json_error("Send a JSON body with a 'question' field.", 400)
    question = payload.get("question")
    if not isinstance(question, str):
        return json_error("'question' must be a string.", 400)
    doc_ids = payload.get("doc_ids")
    if doc_ids is not None and (not isinstance(doc_ids, list) or not all(isinstance(d, str) for d in doc_ids)):
        return json_error("'doc_ids' must be a list of document ids.", 400)
    engine = payload.get("engine")
    if engine is None:  # backward compatible with the older boolean flag
        engine = "local" if payload.get("generate", True) else "none"
    if engine not in ("local", "gemini", "none"):
        return json_error("'engine' must be 'local', 'gemini' or 'none'.", 400)
    gemini_svc = current_app.extensions["labspace"]["services"].get("assistant")
    gemini = None
    if engine == "gemini" and gemini_svc is not None:
        gemini = lambda prompt: gemini_svc.generate(prompt, "You answer strictly from the supplied document context.", 700)  # noqa: E731
    try:
        result = _svc().answer(question, doc_ids or None, payload.get("top_k"), True, engine, gemini)
    except RAGError as exc:
        return json_error(exc.message, exc.status)
    if engine == "gemini":
        result["generation"]["model"] = gemini_svc.model() if gemini_svc else None
        result["generation"]["provider"] = "Google Gemini (cloud). The retrieved passages were sent to Google."
    return jsonify({"ok": True, **result})
