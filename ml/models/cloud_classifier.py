"""
cloud_classifier — Section 7.6.4.
COC: MobileNetV3-Small cloud condition classifier.
"""
import torch
import torch.nn as nn


class CloudClassifier(nn.Module):
    """
    MobileNetV3-Small cloud/sky classifier.
    Input: (B, 3, 224, 224) RGB sky patches
    Output: (B, 5) class logits [clear, thin, moderate, heavy, overcast]
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        raise NotImplementedError("Implement COC MobileNetV3-Small — Section 7.6.4")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
