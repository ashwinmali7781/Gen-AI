from flask import Blueprint, current_app, jsonify

bp = Blueprint("exp6_api", __name__, url_prefix="/api/exp6")


@bp.get("/gallery")
def gallery():
    svc = current_app.extensions["labspace"]["services"]["exp6"]
    return jsonify({"ok": True, "gallery": svc.gallery(), "compare": svc.compare()})
