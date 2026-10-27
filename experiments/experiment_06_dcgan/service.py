"""Experiment 6 — DCGAN on CelebA. Saved notebook outputs only; nothing is trained or run here."""
from __future__ import annotations

from pathlib import Path

from experiments.gallery import build_compare, build_gallery, build_replay

EXP_ID = "exp6"
EPOCHS = 15

SECTIONS = [
    {"key": "dataset", "title": "Training data (CelebA)", "subdir": "dataset",
     "description": "Real CelebA face images used for training, resized and centre-cropped to 64×64. These are dataset photos, not generated images."},
    {"key": "training", "title": "Generator progress by epoch", "subdir": "training",
     "description": "The same fixed noise vectors fed to the generator at the end of each of the 15 epochs."},
    {"key": "results", "title": "Results", "subdir": "results",
     "description": "Loss curves and faces produced by the trained generator from new random noise."},
]

CAPTIONS = {
    "celeba_sample_faces.png": "Sample CelebA images (real photos from the dataset).",
    "celeba_training_batch.png": "A batch of real, preprocessed CelebA training images (dataset, not generated).",
    "loss_curves.png": "Generator and discriminator loss during DCGAN training.",
    "training_animation_frame.png": "Static frame from the notebook's training-progress animation cell (the animation itself was not saved).",
    "generated_faces.png": "New faces generated from random noise by the trained generator.",
    "single_generated_face.png": "One generated face, saved as a standalone image.",
}


class DCGANService:
    def __init__(self, cfg):
        self.outputs = Path(cfg["EXPERIMENTS_DIR"]) / "experiment_06_dcgan" / "outputs"

    def gallery(self):
        return build_gallery(EXP_ID, self.outputs, SECTIONS, CAPTIONS, EPOCHS)

    def compare(self):
        return build_compare(
            EXP_ID, self.outputs, "training/epoch_01_samples.png", f"training/epoch_{EPOCHS:02d}_samples.png",
            "After epoch 1 (earliest saved snapshot)", f"After epoch {EPOCHS} (end of training)",
        )

    def page_context(self) -> dict:
        return {"gallery": self.gallery(), "compare": self.compare(), "replay": build_replay(EXP_ID, self.outputs, EPOCHS)}


def create_service(cfg) -> DCGANService:
    return DCGANService(cfg)
