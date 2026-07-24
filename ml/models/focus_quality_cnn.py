"""
focus_quality_cnn — Focus Quality Estimator, Section 7.6.1.

Two-head heteroscedastic FQE built on a grayscale-adapted MobileNetV3-Small.

    Input  : (B, 1, H, W)   grayscale star crop, robust-normalized to ~[0, 1]
    Output : mu (B,), log_var (B,)   where mu = predicted log(HFR_px)

Design decisions that deviate from the spec's Listing 8, and why:

  * **Per-star, not whole-frame.** The spec fed a 256x256 frame crop; we score
    individual star crops and pool robustly (SEP/DAOPHOT idiom, Bosse 2018).
  * **Predict log(HFR) in pixels**, not HFR in arc-seconds. HFR scatter is
    roughly multiplicative across a sweep, so log-space homogenizes the
    heteroscedasticity, and pixels avoid needing a pixel scale.
  * **Pixels only — no focuser position.** The spec's metadata vector included
    normalized focuser position. Within a run HFR is monotonic in
    |position - focus|, so feeding position lets the net regress HFR from a
    rig-specific shortcut instead of measuring PSF width (leakage; see
    Oakden-Rayner 2020, Kaufman & Rosset 2012). Position/temperature/filter
    history belong to the Focus Offset Predictor, not the FQE.

The two-head output implements Gaussian mean/variance regression
(Nix & Weigend 1994; Kendall & Gal 2017): mu is the predicted mean of
log(HFR), log_var its aleatoric log-variance.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small
    from torchvision.models import MobileNet_V3_Small_Weights
except ModuleNotFoundError:  # torch/torchvision are ML training extras
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _missing_torch_error() -> ModuleNotFoundError:
    return ModuleNotFoundError(
        "PyTorch + torchvision are required to instantiate FocusQualityEstimator. "
        "Install the ML training extras with `pip install -r ml/requirements.txt`."
    )


if nn is None:

    class FocusQualityEstimator:  # type: ignore[no-redef]
        """Import-safe placeholder used when PyTorch is not installed."""

        def __init__(self, *args, **kwargs):
            raise _missing_torch_error()

else:

    class FocusQualityEstimator(nn.Module):
        """Grayscale MobileNetV3-Small with a heteroscedastic 2-value head.

        Args:
            pretrained: load ImageNet weights (requires network). Defaults to
                False so tests/CI are hermetic; production training should set
                True and average the RGB stem into the grayscale stem.
            dropout: dropout before the final linears.
        """

        def __init__(self, pretrained: bool = False, dropout: float = 0.20):
            super().__init__()
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)

            # Adapt the first conv from 3-channel to 1-channel by averaging the
            # RGB filters — preserves pretrained low-level features on grayscale.
            old_conv = backbone.features[0][0]
            new_conv = nn.Conv2d(
                1,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            with torch.no_grad():
                if weights is not None:
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                else:
                    # Averaging three randomly initialized RGB channels cuts the
                    # stem variance by roughly 3x. Fresh grayscale training needs
                    # a fresh fan-out-aware initialization instead.
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out")
            backbone.features[0][0] = new_conv

            self.features = backbone.features  # -> (B, 576, h', w')
            self.pool = backbone.avgpool  # AdaptiveAvgPool2d(1) -> (B, 576, 1, 1)
            self.head = nn.Sequential(
                nn.Linear(576, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 2),  # [mu_log_hfr, log_var]
            )

        def forward(self, img: "torch.Tensor"):
            x = self.pool(self.features(img)).flatten(1)
            out = self.head(x)
            mu = out[:, 0]
            log_var = out[:, 1]
            return mu, log_var


if __name__ == "__main__":  # shape self-check (needs torch)
    if torch is None:
        raise _missing_torch_error()
    m = FocusQualityEstimator().eval()
    mu, log_var = m(torch.rand(4, 1, 64, 64))
    assert mu.shape == (4,) and log_var.shape == (4,), (mu.shape, log_var.shape)
    print("FQE ok — mu:", tuple(mu.shape), "log_var:", tuple(log_var.shape))
