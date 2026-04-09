"""
feature_builder — Section 7.6.2.
Builds the 38-feature vector for focus offset prediction.
"""
import numpy as np


FEATURE_NAMES = [
    "hfr", "fwhm", "star_count", "position_norm", "temperature",
    "hfr_gradient", "hfr_curvature", "hfr_std", "filter_index",
    "session_hours", "altitude", "seeing_estimate", "background_adu",
    "trailing_metric", "cloud_probability",
    # + 23 additional derived features
]


def build_features(sample: dict) -> np.ndarray:
    """Build 38-element feature vector from a focus sample dict."""
    raise NotImplementedError("Implement 38-feature builder")


def build_feature_matrix(samples: list[dict]) -> np.ndarray:
    """Build (N, 38) feature matrix from a list of samples."""
    raise NotImplementedError("Implement batch feature building")
