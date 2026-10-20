"""COC public-human segmentation candidates for the Achernar v6 experiment.

RGB images are scaled to [0, 1]. Outputs are *uncalibrated* pixel logits ordered
sky, cloud, contamination. Coverage, optical transparency and whole-frame cloud
presence are different targets; this module does not manufacture the legacy
five transparency classes from segmentation masks.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # Model construction requires the training extras.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


CLASS_NAMES = ("sky", "cloud", "contamination")
IGNORE_INDEX = 255
_VARIANT_WIDTHS = {"tiny": 8, "small": 16}


def variant_config(variant: str = "tiny") -> dict:
    """Return the JSON-safe architecture and tensor contract, without weights."""
    if variant not in _VARIANT_WIDTHS:
        raise ValueError(f"Unknown COC variant {variant!r}; expected tiny or small")
    return {
        "architecture": "depthwise_separable_unet_groupnorm_v1",
        "variant": variant,
        "base_width": _VARIANT_WIDTHS[variant],
        "encoder_widths": [_VARIANT_WIDTHS[variant] * multiplier for multiplier in (1, 2, 4, 8)],
        "input_channels": 3,
        "num_classes": len(CLASS_NAMES),
        "class_names": list(CLASS_NAMES),
        "ignore_index": IGNORE_INDEX,
        "input_layout": "BCHW",
        "input_range": [0.0, 1.0],
        "output": "uncalibrated_pixel_logits",
        "minimum_spatial_size": 16,
        "normalization": "RGB_uint8_divided_by_255",
        "pretrained": False,
    }


def _missing_torch() -> ModuleNotFoundError:
    return ModuleNotFoundError("PyTorch is required to construct the COC segmentation model")


if nn is None:

    class CloudSegmenter:  # type: ignore[no-redef]
        def __init__(self, variant: str = "tiny"):
            variant_config(variant)
            raise _missing_torch()

    class CloudClassifier:  # type: ignore[no-redef]
        def __init__(self, num_classes: int = 5):
            raise NotImplementedError(
                "The legacy five-class COC task has no approved training labels. "
                "Use CloudSegmenter for the separate sky/cloud/contamination segmentation task."
            )

else:

    def _norm(channels: int) -> nn.GroupNorm:
        # Every variant width is divisible by four. Multiple channels per group
        # avoid a degenerate group even for a single, small input image.
        return nn.GroupNorm(num_groups=4, num_channels=channels)

    class _SeparableBlock(nn.Module):
        def __init__(self, incoming: int, outgoing: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(incoming, incoming, 3, padding=1, groups=incoming, bias=False),
                nn.Conv2d(incoming, outgoing, 1, bias=False),
                _norm(outgoing),
                nn.ReLU(inplace=False),
                nn.Conv2d(outgoing, outgoing, 3, padding=1, groups=outgoing, bias=False),
                nn.Conv2d(outgoing, outgoing, 1, bias=False),
                _norm(outgoing),
                nn.ReLU(inplace=False),"but"
            )

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return self.layers(image)

    class CloudSegmenter(nn.Module):
        """Small U-Net family with a fixed, three-class segmentation ontology.

        ``tiny`` and ``small`` differ only in width, allowing capacity comparisons
        on the same training groups. Neither variant downloads pretrained weights.
        Spatial dimensions may be unequal or odd, provided each is at least 16.
        """

        def __init__(self, variant: str = "tiny"):
            super().__init__()
            self.config = variant_config(variant)
            self.variant = variant
            self.num_classes = len(CLASS_NAMES)
            width = self.config["base_width"]
            self.stem = nn.Sequential(
                nn.Conv2d(3, width, 3, padding=1, bias=False),
                _norm(width),
                nn.ReLU(inplace=False),
            )
            self.pool = nn.AvgPool2d(2)
            self.encoder1 = _SeparableBlock(width, width * 2)
            self.encoder2 = _SeparableBlock(width * 2, width * 4)
            self.encoder3 = _SeparableBlock(width * 4, width * 8)
            self.bridge = _SeparableBlock(width * 8, width * 8)
            self.decoder2 = _SeparableBlock(width * 12, width * 4)
            self.decoder1 = _SeparableBlock(width * 6, width * 2)
            self.decoder0 = _SeparableBlock(width * 3, width)
            self.head = nn.Conv2d(width, self.num_classes, 1)

        @staticmethod
        def _upsample(value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            return F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            # Python input guards are intentionally outside the exported graph.
            # Export/runtime adapters must enforce the same declared contract.
            if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
                if image.ndim != 4 or image.shape[1] != 3 or image.shape[0] < 1:
                    raise ValueError("COC expects nonempty BCHW RGB input with three channels")
                if min(image.shape[-2:]) < 16:
                    raise ValueError("COC input height and width must each be at least 16")
                if not image.is_floating_point():
                    raise ValueError("COC expects floating point RGB values scaled to [0, 1]")
                if not bool(torch.isfinite(image).all()):
                    raise ValueError("COC input contains nonfinite pixels")
                if bool((image < 0).any()) or bool((image > 1).any()):
                    raise ValueError("COC RGB input must be scaled to [0, 1]")

            skip0 = self.stem(image)
            skip1 = self.encoder1(self.pool(skip0))
            skip2 = self.encoder2(self.pool(skip1))
            value = self.bridge(self.encoder3(self.pool(skip2)))
            value = self.decoder2(torch.cat((self._upsample(value, skip2), skip2), dim=1))
            value = self.decoder1(torch.cat((self._upsample(value, skip1), skip1), dim=1))
            value = self.decoder0(torch.cat((self._upsample(value, skip0), skip0), dim=1))
            return self.head(value)

    class CloudClassifier(nn.Module):
        """Explicitly unsupported legacy task; no implicit ontology conversion."""

        def __init__(self, num_classes: int = 5):
            super().__init__()
            raise NotImplementedError(
                "The legacy five-class COC task has no approved training labels. "
                "Use CloudSegmenter for the separate sky/cloud/contamination segmentation task."
            )
