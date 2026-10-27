"""Experiment 7 — StyleGAN-style generator on CelebA. Saved notebook outputs only.

StyleGAN training is not executed by this app. The notebook saved one snapshot per
epoch; there is no image from a truly untrained network, so the "before" side of the
comparison is the earliest saved snapshot (end of epoch 1).
"""
from __future__ import annotations

from pathlib import Path

from experiments.gallery import build_compare, build_gallery, build_replay

EXP_ID = "exp7"
EPOCHS = 15

SECTIONS = [
    {"key": "dataset", "title": "Training data (CelebA)", "subdir": "dataset",
     "description": "Real CelebA photos used for training (64×64). Dataset images, not generated."},
    {"key": "training", "title": "Generator progress by epoch", "subdir": "training",
     "description": "Fixed noise vectors rendered by the generator at the end of each of the 15 epochs."},
    {"key": "results", "title": "StyleGAN capabilities & results", "subdir": "results",
     "description": "Loss curves plus the notebook's truncation-trick, style-mixing and noise-variation demonstrations."},
]

CAPTIONS = {
    "celeba_training_batch.png": "Sample CelebA training images (real photos from the dataset, not generated).",
    "loss_curves.png": "Generator and discriminator loss during training.",
    "truncation_trick.png": "Truncation trick: the same latents pulled toward the average style at different psi values.",
    "style_mixing.png": "Style mixing: coarse features from one latent, fine details from another.",
    "noise_variation.png": "Same style vector with different per-layer noise: identity stays, fine texture changes.",
}


class StyleGANService:
    def __init__(self, cfg):
        self.outputs = Path(cfg["EXPERIMENTS_DIR"]) / "experiment_07_stylegan" / "outputs"

    def gallery(self):
        return build_gallery(EXP_ID, self.outputs, SECTIONS, CAPTIONS, EPOCHS)

    def compare(self):
        return build_compare(
            EXP_ID, self.outputs, "training/epoch_01_samples.png", f"training/epoch_{EPOCHS:02d}_samples.png",
            "Early training — end of epoch 1 (earliest saved snapshot)", f"After training — end of epoch {EPOCHS}",
        )

    def page_context(self) -> dict:
        return {"gallery": self.gallery(), "compare": self.compare(), "replay": build_replay(EXP_ID, self.outputs, EPOCHS)}


def create_service(cfg) -> StyleGANService:
    return StyleGANService(cfg)
