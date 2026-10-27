from flask import Blueprint, current_app, jsonify

bp = Blueprint("exp7_api", __name__, url_prefix="/api/exp7")


@bp.get("/gallery")
def gallery():
    svc = current_app.extensions["labspace"]["services"]["exp7"]
    return jsonify({"ok": True, "gallery": svc.gallery(), "compare": svc.compare()})
