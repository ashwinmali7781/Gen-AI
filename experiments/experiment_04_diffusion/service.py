"""Experiment 4 — text-to-image with an open-source diffusion model (FLUX.1-schnell).

Generation runs on Hugging Face Inference Providers: no weights are downloaded
locally. The token stays server-side. If hosted inference is not configured or
fails, the saved notebook output (Stable Diffusion v1.5) remains viewable.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from experiments.common import IMAGE_EXTS, list_files, prune_old_files

ALLOWED_SIZES = {512, 768, 1024}

SAVED_CAPTIONS = {
    "sd15_original_output.png": (
        "Original notebook output — Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5), "
        "prompt “birds”, style “Realistic”, negative prompt “birds”. "
        "This is a saved result from the earlier Colab run, not a live generation."
    ),
}


class ImageError(Exception):
    def __init__(self, message: str, status: int = 500):
        super().__init__(message)
        self.message = message
        self.status = status


class ImageService:
    def __init__(self, cfg):
        self.cfg = cfg
        base = Path(cfg["EXPERIMENTS_DIR"]) / "experiment_04_diffusion" / "outputs"
        self.saved_dir = base / "saved"
        self.generated_dir = base / "generated"

    # ---------------- page data ----------------
    def configured(self) -> tuple[bool, str]:
        if not self.cfg.get("HF_TOKEN"):
            return False, "HF_TOKEN is not set. Add it to your .env file to enable hosted generation."
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            return False, "huggingface_hub is not installed. Run: pip install -r requirements.txt"
        return True, ""

    def saved_gallery(self) -> list[dict]:
        return [
            {
                "url": f"/outputs/exp4/saved/{p.name}",
                "caption": SAVED_CAPTIONS.get(p.name, p.stem.replace("_", " ")),
                "name": p.name,
            }
            for p in list_files(self.saved_dir, IMAGE_EXTS)
        ]

    def page_context(self) -> dict:
        ok, reason = self.configured()
        return {
            "live_configured": ok,
            "live_reason": reason,
            "model": self.cfg["IMAGE_MODEL"],
            "provider": self.cfg["IMAGE_PROVIDER"],
            "saved": self.saved_gallery(),
            "sizes": sorted(ALLOWED_SIZES),
        }

    # ---------------- generation ----------------
    def _client(self):
        from huggingface_hub import InferenceClient

        return InferenceClient(
            provider=self.cfg["IMAGE_PROVIDER"],
            api_key=self.cfg["HF_TOKEN"],
            timeout=self.cfg["IMAGE_TIMEOUT_S"],
        )

    def generate(self, prompt: str, size=768) -> dict:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ImageError("Please enter a prompt.", 400)
        if len(prompt) > self.cfg["IMAGE_MAX_PROMPT_CHARS"]:
            raise ImageError(f"Prompt is too long (max {self.cfg['IMAGE_MAX_PROMPT_CHARS']} characters).", 400)
        try:
            size = int(size)
        except (TypeError, ValueError):
            size = 768
        if size not in ALLOWED_SIZES:
            raise ImageError("Size must be 512, 768 or 1024.", 400)

        ok, reason = self.configured()
        if not ok:
            raise ImageError(reason, 503)

        try:
            image = self._client().text_to_image(
                prompt, model=self.cfg["IMAGE_MODEL"], width=size, height=size
            )
        except Exception as exc:  # noqa: BLE001 - mapped to safe messages below
            raise self._map_error(exc) from exc

        if image is None or not hasattr(image, "save"):
            raise ImageError("The provider returned an unexpected response (no image).", 502)

        self.generated_dir.mkdir(parents=True, exist_ok=True)
        name = f"flux_{uuid.uuid4().hex[:12]}.png"
        try:
            image.save(self.generated_dir / name, format="PNG")
        except Exception as exc:  # noqa: BLE001
            raise ImageError("Could not save the generated image.", 500) from exc
        prune_old_files(self.generated_dir, self.cfg["IMAGE_KEEP_LAST"], IMAGE_EXTS)
        return {
            "url": f"/outputs/exp4/generated/{name}",
            "download_url": f"/outputs/exp4/generated/{name}?download=1",
            "model": self.cfg["IMAGE_MODEL"],
            "prompt": prompt,
            "size": size,
        }

    def _map_error(self, exc: Exception) -> ImageError:
        name = type(exc).__name__
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if "Timeout" in name or isinstance(exc, TimeoutError):
            return ImageError("The image provider timed out. Try again or use a smaller size.", 504)
        if status in (401, 403):
            return ImageError("Authentication failed. Check that HF_TOKEN is valid and has the 'Make calls to Inference Providers' permission.", 401)
        if status == 402:
            return ImageError("Inference credits are exhausted for this account/provider.", 402)
        if status == 404:
            return ImageError(f"Model {self.cfg['IMAGE_MODEL']} is not available on the selected provider.", 404)
        if status == 429:
            return ImageError("Rate limit reached. Wait a moment and try again.", 429)
        if status in (400, 422):
            return ImageError("The provider rejected the request (unsupported model, provider or parameters).", 400)
        if "Connection" in name or "OSError" in name:
            return ImageError("Could not reach the inference service. Check your internet connection.", 503)
        return ImageError("Image generation failed at the provider. See the server log for details.", 502)


def create_service(cfg) -> ImageService:
    return ImageService(cfg)
