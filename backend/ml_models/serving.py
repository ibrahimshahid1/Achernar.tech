"""
LightGBM Focus Offset Recommender (FOR) — cloud-side model serving.
Section 7.6.8 — filter recommendation model.
"""
import numpy as np


class FocusOffsetRecommender:
    """LightGBM-based focus offset recommender for cloud serving."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.model = None

    def load(self):
        raise NotImplementedError("Load LightGBM model from path")

    def predict(self, features: np.ndarray) -> dict:
        raise NotImplementedError("Run LightGBM inference")


class AnomalyDetector:
    """Anomaly detection for focus runs and sessions."""

    def __init__(self):
        self.model = None

    def detect(self, features: np.ndarray) -> dict:
        raise NotImplementedError("Run anomaly detection inference")
