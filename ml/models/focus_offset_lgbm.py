"""
focus_offset_lgbm — Section 7.6.2.
LightGBM v1 teacher model for focus offset prediction.
"""
import numpy as np


class FocusOffsetLGBM:
    """
    LightGBM teacher model for focus offset.
    Input: (N, 38) feature matrix
    Output: (N,) predicted offsets
    """
    def __init__(self):
        self.model = None

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs):
        raise NotImplementedError("Implement LightGBM training — Section 7.6.2")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implement LightGBM prediction")

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError
