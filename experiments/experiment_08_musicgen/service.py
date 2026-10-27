"""Experiment 8 — text-to-music with Meta's pretrained MusicGen (facebook/musicgen-small).

Two modes:
  * saved clips: the four genuine audio outputs embedded in the original notebook;
  * live generation: runs the same transformers "text-to-audio" pipeline locally
    (lazy-loaded once; slow on CPU, so durations are capped).
"""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

from experiments.common import AUDIO_EXTS, clamp, list_files, prune_old_files

# Prompts recorded from the notebook, in the order the clips were produced.
SAVED_PROMPTS = {
    "musicgen_clip_1.wav": "an upbeat lo-fi hip-hop track with a relaxed piano melody",
    "musicgen_clip_2.wav": "a cheerful acoustic guitar folk tune",
    "musicgen_clip_3.wav": "an energetic electronic dance track with a heavy bassline",
    "musicgen_clip_4.wav": "a calm classical piano piece",
}
TOKENS_PER_SECOND = 50  # MusicGen audio frame rate


class MusicError(Exception):
    def __init__(self, message: str, status: int = 500):
        super().__init__(message)
        self.message = message
        self.status = status


class MusicService:
    def __init__(self, cfg):
        self.cfg = cfg
        base = Path(cfg["EXPERIMENTS_DIR"]) / "experiment_08_musicgen" / "outputs"
        self.saved_dir = base / "audio"
        self.generated_dir = base / "generated"
        self._lock = threading.Lock()
        self._pipe = None
        self.state = "not_loaded"
        self.device = None
        self.error = None

    # ---------------- page data ----------------
    def saved_clips(self) -> list[dict]:
        clips = []
        for p in list_files(self.saved_dir, AUDIO_EXTS):
            clips.append(
                {
                    "name": p.name,
                    "url": f"/outputs/exp8/audio/{p.name}",
                    "download_url": f"/outputs/exp8/audio/{p.name}?download=1",
                    "prompt": SAVED_PROMPTS.get(p.name, "prompt not recorded"),
                }
            )
        return clips

    def live_available(self) -> tuple[bool, str]:
        from experiments.common import has_module

        missing = [m for m in ("torch", "transformers", "scipy", "numpy") if not has_module(m)]
        if missing:
            return False, "Missing packages: " + ", ".join(missing) + ". Run: pip install -r requirements-ml.txt"
        return True, ""

    def page_context(self) -> dict:
        ok, reason = self.live_available()
        return {
            "clips": self.saved_clips(),
            "live_available": ok,
            "live_reason": reason,
            "model": self.cfg["MUSICGEN_MODEL"],
            "max_seconds": self.cfg["MUSICGEN_MAX_SECONDS"],
            "state": self.state,
        }

    # ---------------- generation ----------------
    def _load(self):
        if self._pipe is not None:
            return
        with self._lock:
            if self._pipe is not None:
                return
            self.state = "loading"
            try:
                import torch
                from transformers import pipeline
            except ImportError as exc:
                self.state, self.error = "error", "torch/transformers not installed"
                raise MusicError("torch/transformers are not installed. Run: pip install -r requirements-ml.txt", 503) from exc
            try:
                cuda = torch.cuda.is_available()
                self.device = "cuda" if cuda else "cpu"
                self._pipe = pipeline("text-to-audio", model=self.cfg["MUSICGEN_MODEL"], device=0 if cuda else -1)
                self.state = "ready"
            except OSError as exc:
                self.state, self.error = "error", "download failed"
                raise MusicError("Could not download the MusicGen model (about 2.3 GB). Check your internet connection and disk space.", 503) from exc
            except Exception as exc:  # noqa: BLE001
                self.state, self.error = "error", type(exc).__name__
                raise MusicError("MusicGen could not be loaded. See the server log for details.", 503) from exc

    def generate(self, prompt: str, seconds=8) -> dict:
        prompt = (prompt or "").strip()
        if not prompt:
            raise MusicError("Please enter a music description.", 400)
        if len(prompt) > self.cfg["MUSICGEN_MAX_PROMPT_CHARS"]:
            raise MusicError(f"Description is too long (max {self.cfg['MUSICGEN_MAX_PROMPT_CHARS']} characters).", 400)
        seconds = int(clamp(seconds, 2, self.cfg["MUSICGEN_MAX_SECONDS"], 8))

        ok, reason = self.live_available()
        if not ok:
            raise MusicError(reason, 503)
        self._load()

        started = time.time()
        try:
            import numpy as np
            from scipy.io import wavfile

            with self._lock:
                out = self._pipe(prompt, forward_params={"do_sample": True, "max_new_tokens": seconds * TOKENS_PER_SECOND})
            audio = np.squeeze(np.asarray(out["audio"]))
            if audio.ndim > 1:
                audio = audio[0]
            rate = int(out["sampling_rate"])
            if audio.size == 0:
                raise MusicError("The model produced no audio.", 502)
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("int16")
            self.generated_dir.mkdir(parents=True, exist_ok=True)
            name = f"musicgen_{uuid.uuid4().hex[:12]}.wav"
            wavfile.write(self.generated_dir / name, rate, pcm)
        except MusicError:
            raise
        except MemoryError as exc:
            raise MusicError("Out of memory while generating. Try a shorter duration.", 507) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise MusicError("Out of memory while generating. Try a shorter duration.", 507) from exc
            raise MusicError("Music generation failed. See the server log for details.", 500) from exc
        except Exception as exc:  # noqa: BLE001
            raise MusicError("Music generation failed. See the server log for details.", 500) from exc

        prune_old_files(self.generated_dir, self.cfg["MUSICGEN_KEEP_LAST"], AUDIO_EXTS)
        return {
            "url": f"/outputs/exp8/generated/{name}",
            "download_url": f"/outputs/exp8/generated/{name}?download=1",
            "prompt": prompt,
            "seconds_audio": round(audio.size / rate, 1),
            "seconds_taken": round(time.time() - started, 1),
            "device": self.device,
            "model": self.cfg["MUSICGEN_MODEL"],
        }


def create_service(cfg) -> MusicService:
    return MusicService(cfg)
