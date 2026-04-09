"""
crops — Image crop extraction for ML training.
Extracts star crops from full-frame images for FQE training.
"""
import numpy as np


def extract_star_crops(image: np.ndarray, star_positions: list, crop_size: int = 64) -> list:
    """Extract square crops around detected star positions."""
    raise NotImplementedError("Implement star crop extraction")


def random_background_crops(image: np.ndarray, n_crops: int, crop_size: int = 64) -> list:
    """Extract random background crops for negative examples."""
    raise NotImplementedError("Implement background crop extraction")
