"""
exposure_bandit — Section 7.6.7.
LinUCB / Thompson sampling bandit for exposure optimization.
"""
import numpy as np


class ExposureBandit:
    """
    Contextual bandit for exposure time and gain optimization.
    Arms: discrete (exposure, gain) pairs
    Context: sky conditions, target type, filter, moon phase
    """
    def __init__(self, arms: list[tuple], alpha: float = 1.0):
        self.arms = arms
        self.alpha = alpha
        raise NotImplementedError("Implement LinUCB exposure bandit — Section 7.6.7")

    def select_arm(self, context: np.ndarray) -> int:
        raise NotImplementedError

    def update(self, arm_idx: int, reward: float, context: np.ndarray):
        raise NotImplementedError
