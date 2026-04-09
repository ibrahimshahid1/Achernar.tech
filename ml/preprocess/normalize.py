"""
normalize — Input normalization for ML models.
Handles per-channel normalization, background subtraction, and scaling.
"""
import numpy as np


def normalize_image(image: np.ndarray, norm_params: dict | None = None) -> np.ndarray:
    """Normalize an image for model input using stored parameters."""
    raise NotImplementedError("Implement image normalization")


def compute_norm_params(images: list[np.ndarray]) -> dict:
    """Compute normalization parameters (mean, std) from a dataset."""
    raise NotImplementedError("Implement norm param computation")
