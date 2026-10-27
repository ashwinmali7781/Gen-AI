"""Small helpers shared by all experiments. No heavy imports here."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable

from flask import jsonify

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac"}
SERVABLE_EXTS = IMAGE_EXTS | AUDIO_EXTS


def json_error(message: str, status: int = 400, **extra):
    payload = {"ok": False, "error": message}
    payload.update(extra)
    return jsonify(payload), status


def clamp(value, low, high, default):
    """Coerce value to float and clamp; fall back to default on bad input."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return max(low, min(high, number))


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def safe_join(root: Path, relative: str) -> Path | None:
    """Resolve `relative` under `root`; return None if it would escape root."""
    if not relative or "\x00" in relative:
        return None
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def list_files(directory: Path, exts: Iterable[str]) -> list[Path]:
    """Sorted files with given extensions. Missing/empty directory -> []."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    allowed = {e.lower() for e in exts}
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in allowed)


def prune_old_files(directory: Path, keep: int, exts: Iterable[str]) -> None:
    files = sorted(list_files(directory, exts), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        try:
            old.unlink()
        except OSError:
            pass
