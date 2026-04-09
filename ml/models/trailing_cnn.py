"""
trailing_cnn — Section 7.6.5.
TED: CNN for star trailing/elongation detection.
"""
import torch
import torch.nn as nn


class TrailingDetector(nn.Module):
    """
    CNN for trailing/elongation detection.
    Input: (B, 1, 64, 64) star crops
    Output: (B, 3) [trailing_prob, axis_ra_prob, axis_dec_prob]
    """
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Implement TED CNN — Section 7.6.5")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
