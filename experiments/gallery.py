"""Gallery builder for experiments that showcase saved notebook outputs."""
from __future__ import annotations

import re
from pathlib import Path

from .common import IMAGE_EXTS, list_files

_EPOCH_RE = re.compile(r"epoch_(\d+)_samples", re.I)


def _caption(name: str, captions: dict, epoch_total: int | None) -> str:
    if name in captions:
        return captions[name]
    match = _EPOCH_RE.search(name)
    if match and epoch_total:
        return f"Generated faces from fixed noise at the end of epoch {int(match.group(1))} of {epoch_total}."
    return Path(name).stem.replace("_", " ")


def build_gallery(exp_id: str, outputs_dir: Path, sections: list[dict], captions: dict, epoch_total: int | None = None) -> list[dict]:
    """sections: [{"key","title","subdir","description"}]. Empty folders give empty item lists."""
    result = []
    for sec in sections:
        folder = Path(outputs_dir) / sec["subdir"]
        items = [
            {
                "url": f"/outputs/{exp_id}/{sec['subdir']}/{p.name}",
                "name": p.name,
                "caption": _caption(p.name, captions, epoch_total),
            }
            for p in list_files(folder, IMAGE_EXTS)
        ]
        result.append({**sec, "items": items, "folder_hint": f"experiments/…/outputs/{sec['subdir']}/"})
    return result


def build_compare(exp_id: str, outputs_dir: Path, early: str, final: str, early_label: str, final_label: str) -> dict | None:
    """Side-by-side early/final snapshot, only when both files exist."""
    root = Path(outputs_dir)
    if not (root / early).is_file() or not (root / final).is_file():
        return None
    return {
        "early": {"url": f"/outputs/{exp_id}/{early}", "label": early_label},
        "final": {"url": f"/outputs/{exp_id}/{final}", "label": final_label},
    }


def build_replay(exp_id: str, outputs_dir: Path, epoch_total: int | None = None) -> list[dict]:
    """Ordered epoch snapshots for the replay player. Empty list if none are saved."""
    frames = []
    for p in list_files(Path(outputs_dir) / "training", IMAGE_EXTS):
        m = _EPOCH_RE.search(p.name)
        if m:
            frames.append((int(m.group(1)), p.name))
    frames.sort()
    return [{"epoch": n, "url": f"/outputs/{exp_id}/training/{name}",
             "label": f"Epoch {n}" + (f" of {epoch_total}" if epoch_total else "")} for n, name in frames]
