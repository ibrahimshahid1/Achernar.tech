"""
synthetic_psf — generate synthetic star crops with known HFR (Section 7.7.5 tests).

This is the substitute for real autofocus data during development. It renders
Gaussian/Moffat PSFs at a *known* HFR, arranged into synthetic focus "runs"
(each run is a V-curve of HFR vs. position), so the entire FQE pipeline —
model, losses, aggregation, training, ONNX export, edge inference — can be
built and validated end-to-end before a single real frame exists.

What it CANNOT substitute for: a promotable model. Synthetic PSFs are clean,
single, centered, and lack real optical aberration, field dependence, tracking
error, cosmic rays, and rig-specific PSF structure. Only real multi-rig data
yields weights that generalize (see build_focus_quality_dataset.py docstring).
"""
from __future__ import annotations

import numpy as np

# HFR (px) = sigma * sqrt(2 ln2) for a Gaussian.
_GAUSS_HFR_PER_SIGMA = np.sqrt(2.0 * np.log(2.0))


def _gaussian_crop(size, hfr, cx, cy, amp):
    sigma = hfr / _GAUSS_HFR_PER_SIGMA
    ys, xs = np.mgrid[0:size, 0:size]
    r2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return amp * np.exp(-r2 / (2.0 * sigma**2))


def _moffat_crop(size, hfr, cx, cy, amp, beta):
    # HFR = alpha * sqrt(2^(1/(beta-1)) - 1)  =>  alpha from target HFR.
    alpha = hfr / np.sqrt(2.0 ** (1.0 / (beta - 1.0)) - 1.0)
    ys, xs = np.mgrid[0:size, 0:size]
    r2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return amp * (1.0 + r2 / alpha**2) ** (-beta)


def render_star(
    hfr,
    size=64,
    amp=8000.0,
    sky=100.0,
    read_noise=8.0,
    profile="gaussian",
    beta=2.8,
    jitter=1.5,
    rng=None,
):
    """Render one 16-bit-ish star crop at the given HFR (px), with sky + noise."""
    rng = rng or np.random.default_rng()
    c = size / 2 - 0.5
    cx = c + rng.uniform(-jitter, jitter)
    cy = c + rng.uniform(-jitter, jitter)
    if profile == "moffat":
        star = _moffat_crop(size, hfr, cx, cy, amp, beta)
    else:
        star = _gaussian_crop(size, hfr, cx, cy, amp)
    img = sky + star
    # Shot noise on the star+sky plus Gaussian read noise, clipped to 16-bit.
    img = rng.poisson(np.clip(img, 0, None)).astype(np.float64)
    img += rng.normal(0.0, read_noise, img.shape)
    return np.clip(img, 0.0, 65535.0)


def make_synthetic_dataset(
    n_runs=40,
    positions_per_run=7,
    stars_per_frame=6,
    size=64,
    hfr_min_range=(1.4, 2.6),
    # Chosen so the V-curve spans ~1.5-12 px HFR over +/-3 steps on a 64 px crop
    # (a star must stay well inside the crop): hfr = hfr_min + curv*offset^2.
    curvature_range=(6e-6, 1.8e-5),
    step=250,
    seed=0,
):
    """Build a grouped synthetic dataset of focus runs.

    Each run has a best-focus position and an HFR V-curve
    ``hfr = hfr_min + curvature * (pos - best)^2``. Returns a dict of arrays:

        crops     (N, size, size) float32, robust-normalized to [0, 1]
        log_hfr   (N,)  float32   regression target
        hfr_px    (N,)  float32   raw HFR label
        run_id    (N,)  int64     grouping key (for grouped splits / ranking)
        position  (N,)  int64     focuser position
    """
    from preprocess.normalize import robust_normalize

    rng = np.random.default_rng(seed)
    crops, log_hfr, hfr_px, run_id, position, frame_id = [], [], [], [], [], []

    for run in range(n_runs):
        best = int(rng.integers(8000, 12000))
        hfr_min = float(rng.uniform(*hfr_min_range))
        curv = float(rng.uniform(*curvature_range))
        amp = float(rng.uniform(4000.0, 16000.0))
        sky = float(rng.uniform(60.0, 400.0))
        profile = "moffat" if rng.random() < 0.5 else "gaussian"
        offsets = (np.arange(positions_per_run) - positions_per_run // 2) * step

        for off in offsets:
            pos = best + int(off)
            hfr = hfr_min + curv * (pos - best) ** 2
            frame_key = f"synthetic-run-{run}-position-{pos}"
            for _ in range(stars_per_frame):
                crop = render_star(
                    hfr, size=size, amp=amp, sky=sky, profile=profile, beta=rng.uniform(2.2, 4.0), rng=rng
                )
                norm, _ = robust_normalize(crop)
                crops.append(norm.astype(np.float32))
                log_hfr.append(np.log(hfr))
                hfr_px.append(hfr)
                run_id.append(run)
                position.append(pos)
                frame_id.append(frame_key)

    run_id_array = np.asarray(run_id, dtype=np.int64)
    return {
        "crops": np.stack(crops).astype(np.float32),
        "log_hfr": np.asarray(log_hfr, dtype=np.float32),
        "hfr_px": np.asarray(hfr_px, dtype=np.float32),
        "run_id": run_id_array,
        "run_key": np.asarray([f"synthetic-run-{value}" for value in run_id_array]),
        "split_group": np.asarray([f"synthetic-run-{value}" for value in run_id_array]),
        "position": np.asarray(position, dtype=np.int64),
        "frame_id": np.asarray(frame_id),
    }
