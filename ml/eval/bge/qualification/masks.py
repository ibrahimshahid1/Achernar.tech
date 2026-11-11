"""Sky masks for truth-free metrics.

``source_mask``: a method's own detections, each masked by (a) its bounding
box dilated by ``BBOX_PAD_PX`` and (b) the ellipse ``SOURCE_ELLIPSE_SCALE`` x
(a, b, theta) when the catalog carries a/b/theta (SEP), else a circle of radius
``SOURCE_ELLIPSE_SCALE`` x sqrt(npix / pi). Scale 3 matches the v2 source mask
(``MASK_ELLIPSE_SCALE``, spec section 3.8).

``hot_pixel_mask``: persistent hot pixels from the darks of one night: a pixel
is hot when it exceeds the dark's median + ``HOT_SIGMA`` x robust sigma in at
least half of the night's darks (all of them when fewer than 3), dilated by
``HOT_DILATE_PX`` (the bilinear CFA reconstruction spreads one hot photosite
over a 3x3 neighbourhood of the analysis plane).
"""

from __future__ import annotations

import math

import numpy as np

BBOX_PAD_PX = 4
SOURCE_ELLIPSE_SCALE = 3.0
HOT_SIGMA = 5.0
HOT_DILATE_PX = 1
ROBUST_SAMPLE = 2_000_003  # pixels used for a dark's robust sigma (deterministic stride)


def robust_sigma(values) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    med = np.median(v)
    return float(1.482602218505602 * np.median(np.abs(v - med)))


def _ellipse(mask, x, y, a, b, theta, r):
    h, w = mask.shape
    ext = r * max(a, b) + 1.0
    x0, x1 = max(0, int(math.floor(x - ext))), min(w, int(math.ceil(x + ext)) + 1)
    y0, y1 = max(0, int(math.floor(y - ext))), min(h, int(math.ceil(y + ext)) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - x, yy - y
    c, s = math.cos(theta), math.sin(theta)
    u, v = dx * c + dy * s, -dx * s + dy * c
    a2, b2 = max(a * r, 0.5) ** 2, max(b * r, 0.5) ** 2
    mask[y0:y1, x0:x1] |= (u * u / a2 + v * v / b2) <= 1.0


def source_mask(detections, shape, pad=BBOX_PAD_PX, scale=SOURCE_ELLIPSE_SCALE) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    for d in detections or []:
        x, y = float(d["x"]), float(d["y"])
        if all(k in d and d[k] is not None for k in ("xmin", "xmax", "ymin", "ymax")):
            mask[max(0, int(d["ymin"]) - pad):min(h, int(d["ymax"]) + pad + 1),
                 max(0, int(d["xmin"]) - pad):min(w, int(d["xmax"]) + pad + 1)] = True
        a, b, theta = d.get("a"), d.get("b"), d.get("theta")
        if a and b and theta is not None and a > 0 and b > 0:
            _ellipse(mask, x, y, float(a), float(b), float(theta), scale)
        else:
            r = scale * math.sqrt(max(float(d.get("npix") or 1.0), 1.0) / math.pi)
            _ellipse(mask, x, y, r / scale, r / scale, 0.0, scale)
    return mask


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not mask.any():
        return mask
    from scipy import ndimage

    return ndimage.binary_dilation(mask, structure=np.ones((2 * px + 1, 2 * px + 1), bool))


def hot_pixel_mask(dark_iter, shape, n_sigma=HOT_SIGMA, dilate_px=HOT_DILATE_PX):
    """``(mask | None, info)`` from an iterable of ``(image, masks)`` darks of one night."""
    counts = np.zeros(shape, dtype=np.uint16)
    n = 0
    for image, masks in dark_iter:
        if image.shape != tuple(shape):
            continue
        usable = np.isfinite(image)
        for m in masks.values():
            usable &= ~m.astype(bool)
        flat = image[usable]
        step = max(1, flat.size // ROBUST_SAMPLE)
        sample = flat[::step]
        med, sig = float(np.median(sample)), robust_sigma(sample)
        if not np.isfinite(sig) or sig <= 0:
            continue
        counts += (np.nan_to_num(image, nan=-np.inf) > med + n_sigma * sig).astype(np.uint16)
        n += 1
    if n == 0:
        return None, {"n_darks": 0, "reason": "no usable dark of this shape in the night"}
    need = n if n < 3 else int(math.ceil(0.5 * n))
    hot = counts >= need
    info = {"n_darks": n, "rule": f"> median + {n_sigma} robust sigma in >= {need}/{n} darks, dilated {dilate_px} px",
            "n_hot_pixels": int(hot.sum())}
    hot = dilate(hot, dilate_px)
    info["n_masked_pixels"] = int(hot.sum())
    return hot, info
