"""Real dark + simulated sky + simulated stars: exact differential truth with real noise.

For a real dark D (real read noise, pattern, hot pixels, amplifier glow), a
known smooth sky G (from ``scenarios.py``: the ``vignetting_background``
generator, and the linear plane of ``gradient_linear_xy``) and known stars S
(``scenarios.add_stars``, PSF of the night's lights or a default Moffat), the
method is run on D and on D + G + S. The dark's own background is unknown, so
the truth is differential and per method::

    B_true(x, y) = B_hat_m(D)(x, y) + G(x, y)

and the error ``B_hat_m(D + G + S) - B_true`` is scored with the synthetic
metrics of ``ml/eval/bge/metrics.py`` (``background_map_metrics``,
``detection_metrics``, ``hfr_metrics``). The noise scale is method-independent:
``sigma_D`` = 1.4826 MAD(D1 - D2) / sqrt(2) from a second dark of the same
night and shape when one exists, else the robust std of D minus its 32 px
block median (recorded). Photon noise of G + S is added when the gain is
known; the added noise then enters R_true but not B_true.
"""

from __future__ import annotations

import math

import numpy as np

from .masks import robust_sigma

DEFAULT_PSF = {"model": "moffat", "fwhm_px": 4.0, "beta": 3.0}
N_STARS_PER_MPX = 40  # star density of a trial (~1000 stars on a 24 Mpx frame, a moderate field)
N_STARS_MIN, N_STARS_MAX = 10, 1000
SKY_SIGMA_MULTIPLE = 50.0  # default sky level above the dark, in sigma_D, when no light sets it
STAR_SNR_RANGE = (10.0, 300.0)


def sigma_dark(dark, masks, other=None) -> tuple:
    from .methods import unusable_mask

    bad = unusable_mask(dark, masks)
    if other is not None and other.shape == dark.shape:
        diff = (dark.astype(np.float64) - other.astype(np.float64))[~bad & np.isfinite(other)]
        return robust_sigma(diff) / math.sqrt(2.0), "pair_difference"
    h, w = dark.shape
    b = 32
    hh, ww = h // b * b, w // b * b
    blocks = np.median(dark[:hh, :ww].reshape(hh // b, b, ww // b, b).swapaxes(1, 2).reshape(hh // b, ww // b, -1),
                       axis=2)
    smooth = np.kron(blocks, np.ones((b, b)))
    res = (dark[:hh, :ww] - smooth)[~bad[:hh, :ww]]
    return robust_sigma(res), "block_median_residual"


def sky_models(width, height, level):
    """``{name: G}`` smooth sky models at mean level ~ ``level`` (ADU)."""
    from .. import scenarios as SC

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    xc, yc = (width - 1) / 2.0, (height - 1) / 2.0
    # Same functional form as scenarios.gradient_linear_xy (plane about the centre), here ~30 % across the frame.
    linear = level * (1.0 + 0.15 * (xx - xc) / max(xc, 1.0) + 0.10 * (yy - yc) / max(yc, 1.0))
    vignetting = SC.vignetting_background(width, height, b0=level, c2=0.25, c4=0.05)
    return {"linear_gradient": linear, "vignetting": vignetting}


def trial_star_count(shape) -> int:
    return int(min(N_STARS_MAX, max(N_STARS_MIN, round(N_STARS_PER_MPX * shape[0] * shape[1] / 1e6))))


def build_trial(dark, masks, sky, psf, sigma, rng, gain=None, n_stars=None):
    """``(image, truth_dict)`` for one dark + sky model + stars trial."""
    from .. import scenarios as SC

    h, w = dark.shape
    unit = SC.render_psf(psf, w / 2.0, h / 2.0, 1.0)
    margin = int(unit.radius) + 8
    min_sep = max(20.0, 2.5 * unit.radius)
    n = trial_star_count(dark.shape) if n_stars is None else n_stars
    while n > 0:
        try:
            xy = SC.place_uniform(rng, n, w, h, margin, min_sep)
            break
        except RuntimeError:
            n //= 2
    else:
        xy = []
    hfr_unit, _ = SC.numerical_hfr(unit, w / 2.0, h / 2.0)
    n_ap = math.pi * max(3.0, 2.0 * (hfr_unit or 2.0)) ** 2
    snr = np.exp(rng.uniform(math.log(STAR_SNR_RANGE[0]), math.log(STAR_SNR_RANGE[1]), len(xy)))
    fluxes = snr * sigma * math.sqrt(n_ap) / 0.5  # about half the flux inside the S/N aperture
    stars = [(x, y, float(f), psf) for (x, y), f in zip(xy, fluxes)]
    src, sources = SC.add_stars((h, w), stars)
    SC.assign_blend_groups(sources, 2.0 * min_sep / 3.0)
    image = dark.astype(np.float64) + sky + src
    noise_note = "none added (gain unknown): noise is the dark's real noise"
    if gain:
        image += rng.standard_normal(image.shape) * np.sqrt(np.clip(sky + src, 0.0, None) / gain)
        noise_note = f"gaussian photon noise of sky + stars added, gain {gain} e-/ADU"
    return np.ascontiguousarray(image, dtype=np.float32), {"sources": sources, "source_image": src,
                                                          "n_stars": len(stars), "noise": noise_note}


def score_trial(dark_out, trial_out, sky, truth, image_shape, usable, sigma, sigma_map=None) -> dict:
    """Synthetic-metric scoring of one trial for one method (differential truth)."""
    from .. import metrics as MT

    if not (dark_out.usable and trial_out.usable):
        why = dark_out.reason if not dark_out.usable else trial_out.reason
        return {"status": "not_evaluable", "reason": f"method unusable on dark or trial: {why}"}
    b_true = np.asarray(dark_out.back_map(image_shape), dtype=np.float64) + sky
    sig_map = sigma_map if sigma_map is not None else np.full(image_shape, sigma, dtype=np.float64)
    under, far = MT.source_masks(truth["source_image"], sig_map, usable)
    star_xy = [(s.x, s.y) for s in truth["sources"]]
    bg = MT.background_map_metrics(trial_out.back_map(image_shape), b_true, usable, under, far, star_xy)
    truth_json = [s.to_json() for s in truth["sources"]]
    out = {"status": "completed", "background": bg, "sigma_d_adu": sigma,
           "bias_sigma": bg.get("bias_median_adu", float("nan")) / sigma if sigma > 0 else None,
           "rmse_sigma": bg.get("rmse_adu", float("nan")) / sigma if sigma > 0 else None}
    if trial_out.detections is not None:
        det = MT.detection_metrics(trial_out.detections, truth_json, image_shape[1], image_shape[0],
                                   truth["source_image"], sig_map)
        out["detection"] = MT.strip_private(det)
    if trial_out.accepted is not None:
        out["hfr"] = MT.hfr_metrics(trial_out.accepted, truth_json)
    return out
