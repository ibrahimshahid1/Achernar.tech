"""Truth-free metrics on real frames (definitions also in the local doc).

Pull ``z = (I - B_hat) / R_hat`` over the method's source-masked sky. For a
correct model z ~ N(0, 1): mean 0 (bias in sigma units), robust std 1 minus
about 1.3 % from 3-sigma clipped noise estimation, tails 2.7e-3 beyond 3 and
5.7e-7 beyond 5. The analysis plane is a bilinear CFA reconstruction, so its
per-pixel variance differs between CFA phases and neighbours are correlated:
the expected robust std ``s0`` is calibrated on development-night darks/bias
(``calibrate_s0``); the per-phase robust std is reported.

Difference pull ``z = (I1 - I2) / sqrt(R1^2 + R2^2)`` on consecutive frames of
one repeat group, over pixels outside both frames' source masks: B cancels,
so it tests R_hat alone.

Negated residual: the common reference detector (SEP extract with the
daemon's default detection configuration) run on ``-(I - B_hat)`` with
``err = R_hat``: every detection is false; those within 3 HFR of a bright star
(peak >= 20 R_hat) indicate wing over-subtraction.
"""

from __future__ import annotations

import math

import numpy as np

from . import config_sha256
from .masks import robust_sigma

TAIL_REFERENCE = {"p_abs_gt3": 2.6998e-3, "p_abs_gt5": 5.733e-7}
CLIP_FOR_MEAN = 5.0
MIN_SKY_PIXELS = 10_000
BRIGHT_PEAK_SIGMA = 20.0
NEAR_BRIGHT_HFR = 3.0
DEFAULT_HFR_PX = 2.0


def pull_stats(z: np.ndarray) -> dict:
    """Statistics of a pull sample (1-D finite array, or 2-D with NaN for excluded)."""
    if z.ndim == 2:
        phases = []
        for dy in (0, 1):
            for dx in (0, 1):
                sub = z[dy::2, dx::2]
                sub = sub[np.isfinite(sub)]
                phases.append(robust_sigma(sub) if sub.size >= 100 else None)
        z = z[np.isfinite(z)]
    else:
        phases = None
        z = z[np.isfinite(z)]
    n = int(z.size)
    if n < MIN_SKY_PIXELS:
        return {"n_pixels": n, "status": "not_evaluable", "reason": f"{n} sky pixels < {MIN_SKY_PIXELS}"}
    a = np.abs(z)
    clipped = z[a <= CLIP_FOR_MEAN]
    return {
        "status": "completed", "n_pixels": n,
        "mean": float(np.mean(z)), "clipped_mean": float(np.mean(clipped)) if clipped.size else None,
        "median": float(np.median(z)), "std": float(np.std(z)), "robust_std": robust_sigma(z),
        "frac_abs_gt3": float(np.mean(a > 3.0)), "frac_abs_gt5": float(np.mean(a > 5.0)),
        "robust_std_by_cfa_phase_00_01_10_11": phases,
        "definitions": {"clipped_mean": f"mean of z with |z| <= {CLIP_FOR_MEAN}", "robust_std": "1.4826 MAD"},
    }


def frame_pull(image, back, rms, exclude) -> dict:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (image.astype(np.float64) - np.asarray(back, dtype=np.float64)) / np.asarray(rms, dtype=np.float64)
    bad = exclude | ~np.isfinite(z) | ~(np.asarray(rms) > 0)
    z = np.where(bad, np.nan, z)
    return pull_stats(z)


def difference_pull(image1, rms1, exclude1, image2, rms2, exclude2) -> dict:
    if image1.shape != image2.shape:
        return {"status": "not_evaluable", "reason": "frame shapes differ"}
    r1, r2 = np.asarray(rms1, dtype=np.float64), np.asarray(rms2, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (image1.astype(np.float64) - image2.astype(np.float64)) / np.sqrt(r1 * r1 + r2 * r2)
    bad = exclude1 | exclude2 | ~np.isfinite(z)
    return pull_stats(np.where(bad, np.nan, z))


def negated_detections(image, back, rms, exclude, detections, frame_hfr, det_config=None) -> dict:
    """False detections on -(I - B_hat) with the common reference detector."""
    from .. import python_methods as PM

    det = det_config or PM.sep_detection_config_json()
    sep = PM.sep_module()
    data = -(image.astype(np.float32) - np.asarray(back, dtype=np.float32))
    data = np.ascontiguousarray(np.where(exclude, np.float32(0.0), data), dtype=np.float32)
    err = rms if np.ndim(rms) == 0 else np.ascontiguousarray(rms, dtype=np.float32)
    err = float(err) if np.ndim(err) == 0 else err
    previous = sep.get_extract_pixstack()
    try:
        sep.set_extract_pixstack(det["pixstack"])
        objects = sep.extract(data, det["thresh_sigma"], err=err, mask=np.ascontiguousarray(exclude.astype(np.uint8)),
                              maskthresh=0.0, minarea=det["minarea"],
                              filter_kernel=np.asarray(det["filter_kernel"], dtype=np.float32),
                              filter_type=det["filter_type"], deblend_nthresh=det["deblend_nthresh"],
                              deblend_cont=det["deblend_cont"], clean=det["clean"], clean_param=det["clean_param"])
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "reason": f"sep.extract on the negated residual: {type(exc).__name__}: {exc}"}
    finally:
        sep.set_extract_pixstack(previous)
    usable_mpx = float((~exclude).sum()) / 1e6
    xs = np.asarray(objects["x"], dtype=np.float64)
    ys = np.asarray(objects["y"], dtype=np.float64)
    hfr = float(frame_hfr) if frame_hfr else DEFAULT_HFR_PX
    rms_map = np.broadcast_to(np.asarray(rms, dtype=np.float64), image.shape)
    h, w = image.shape
    bright = []
    for d in detections or []:
        xi, yi = min(max(int(round(d["x"])), 0), w - 1), min(max(int(round(d["y"])), 0), h - 1)
        peak = d.get("peak") if d.get("peak") is not None else d.get("peak_residual")
        if peak is not None and rms_map[yi, xi] > 0 and float(peak) >= BRIGHT_PEAK_SIGMA * rms_map[yi, xi]:
            bright.append((float(d["x"]), float(d["y"])))
    near = 0
    if bright and xs.size:
        b = np.asarray(bright)
        dist = np.min(np.hypot(xs[:, None] - b[None, :, 0], ys[:, None] - b[None, :, 1]), axis=1)
        near = int(np.sum(dist <= NEAR_BRIGHT_HFR * hfr))
    return {"status": "completed", "n_detections": int(xs.size), "usable_mpx": usable_mpx,
            "rate_per_mpx": (xs.size / usable_mpx) if usable_mpx > 0 else None, "n_bright_stars": len(bright),
            "n_near_bright": near, "fraction_near_bright": (near / xs.size) if xs.size else None,
            "near_radius": f"{NEAR_BRIGHT_HFR} x frame HFR ({hfr:.2f} px)",
            "detector_config_sha256": config_sha256(det)}


def cv(values) -> float | None:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size < 3 or not np.mean(v) > 0:
        return None
    return float(np.std(v, ddof=1) / np.mean(v))


def fit_sweep(positions, hfrs) -> dict:
    """Hyperbola fit of one sweep with the harness's fitting code (metrics.fit_hyperbola),
    the >= 2-valid-frames-each-side rule (metrics.focus_shift)."""
    from .. import metrics as MT

    vals = [float(v) if v is not None else float("nan") for v in hfrs]
    res = MT.focus_shift(list(map(float, positions)), vals, 0.0)
    hyp = res["hyperbola"]
    out = {"status": hyp["status"], "reason": hyp.get("reason"), "p0": hyp.get("p0"),
           "rms_residual_px": hyp.get("rms_residual"), "a_px": hyp.get("a_px"), "b_steps": hyp.get("b_steps"),
           "valid_frames_left_right": hyp.get("valid_frames_left_right"), "n_frames": res["n_frames"],
           "n_frames_with_hfr": res["n_frames_valid"], "parabola_p0": res["parabola"].get("p0")}
    return out


def cfz_steps(focal_ratio, stepsize_um, wavelength_nm=550.0) -> float | None:
    """Critical focus zone 4 lambda N^2 (full width) in focuser steps."""
    if not focal_ratio or not stepsize_um or stepsize_um <= 0:
        return None
    return 4.0 * (wavelength_nm * 1e-3) * float(focal_ratio) ** 2 / float(stepsize_um)


def finite_median(values):
    v = [x for x in values if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return float(np.median(v)) if v else None
