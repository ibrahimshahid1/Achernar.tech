"""
normalize — input normalization for the FQE (Section 7.6.1, preprocessing step 3).

The edge (daemon) and training MUST use identical preprocessing, so this is the
single source of truth: clip each crop to its own [p_lo, p_hi] percentiles and
scale to [0, 1]. The Rust edge path recomputes the same NumPy-linear percentiles
per crop; fixed per-rig parameters are not part of the current model contract.
"""
from __future__ import annotations

import numpy as np


def robust_normalize(
    image: np.ndarray,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    params: tuple[float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Clip to [p_lo, p_hi] percentiles and scale to [0, 1].

    Returns ``(normalized_float32, (lo, hi))`` so the (lo, hi) used can be
    persisted for identical edge preprocessing. Pass ``params`` to reuse a
    stored (lo, hi) instead of recomputing per-crop.
    """
    img = np.asarray(image, dtype=np.float32)
    if params is not None:
        lo, hi = params
    else:
        finite = img[np.isfinite(img)]
        if finite.size == 0:
            return np.zeros_like(img), (0.0, 1.0)
        lo = float(np.percentile(finite, p_lo))
        hi = float(np.percentile(finite, p_hi))
    if hi <= lo:
        hi = lo + 1.0
    finite_mask = np.isfinite(img)
    with np.errstate(invalid="ignore", over="ignore"):
        out = np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    # Camera pixels are finite, but corrupted/test inputs must match Rust's
    # explicit non-finite policy and must never feed NaN/Inf into ONNX Runtime.
    out[~finite_mask] = 0.0
    return out, (float(lo), float(hi))


def normalize_image(image: np.ndarray, norm_params: dict | None = None) -> np.ndarray:
    """Normalize an image for model input using stored parameters."""
    params = None
    if norm_params is not None and "lo" in norm_params and "hi" in norm_params:
        params = (norm_params["lo"], norm_params["hi"])
    out, _ = robust_normalize(image, params=params)
    return out


def compute_norm_params(images: list[np.ndarray], p_lo: float = 1.0, p_hi: float = 99.0) -> dict:
    """Compute (lo, hi) percentiles over a set of images (for a rig profile)."""
    stacked = np.concatenate([np.asarray(im, dtype=np.float32).ravel() for im in images])
    finite = stacked[np.isfinite(stacked)]
    return {
        "lo": float(np.percentile(finite, p_lo)),
        "hi": float(np.percentile(finite, p_hi)),
        "p_lo": p_lo,
        "p_hi": p_hi,
    }
