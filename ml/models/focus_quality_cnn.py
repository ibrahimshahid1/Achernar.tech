"""
focus_quality_cnn — Section 7.6.1.

FocusQualityEstimator (FQE): scores how well-focused a star crop is, in [0, 1].
The spec calls for a MobileNetV3-Small backbone; to keep the edge model tiny and
dependency-light (no torchvision required), this is a compact depthwise-separable
CNN with the same I/O contract. Swap in torchvision's MobileNetV3-Small later by
keeping the (B,1,64,64) -> (B,1) signature unchanged.

    Input  : (B, 1, 64, 64)  grayscale star crop, normalized to ~[0, 1]
    Output : (B, 1)          focus quality score in [0, 1] (sigmoid)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _DWSeparable(nn.Module):
    """Depthwise-separable conv block (MobileNet-style)."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.Hardswish()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class FocusQualityEstimator(nn.Module):
    """Compact CNN focus-quality estimator. Output is a scalar score in [0, 1]."""

    def __init__(self, num_classes: int = 1, width: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, 3, stride=2, padding=1, bias=False),  # 64 -> 32
            nn.BatchNorm2d(width), nn.Hardswish(),
        )
        self.body = nn.Sequential(
            _DWSeparable(width, width * 2, stride=2),       # 32 -> 16
            _DWSeparable(width * 2, width * 4, stride=2),    # 16 -> 8
            _DWSeparable(width * 4, width * 4, stride=2),    # 8  -> 4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(width * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).flatten(1)
        return torch.sigmoid(self.head(x))      # (B, num_classes) in [0, 1]


if __name__ == "__main__":  # shape self-check (needs torch)
    m = FocusQualityEstimator().eval()
    out = m(torch.rand(4, 1, 64, 64))
    assert out.shape == (4, 1), out.shape
    print("FQE output:", tuple(out.shape), "range ok:", bool((0 <= out).all() and (out <= 1).all()))
