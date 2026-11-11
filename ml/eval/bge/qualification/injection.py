"""Artificial-star injection and recovery in real light frames.

PSF: estimated from the frame's own bright isolated stars, found by the
method-independent reference (``python_sep__n1g1``: SEP background, SEP
extraction and the harness acceptance policy). Up to ``PSF_MAX_STARS`` accepted
stars with aperture S/N >= ``PSF_MIN_SNR``, no other catalog object within
max(15 px, 6 HFR), are each fitted with a circular Moffat (point-sampled model,
free centre, amplitude, alpha, beta in [1.2, 10], least squares on the
background-subtracted stamp). The injected PSF is the Moffat with the median
FWHM and median beta, rendered pixel-integrated by ``scenarios.render_psf``.
With fewer than ``PSF_MIN_FITS`` converged fits the PSF falls back to a
Gaussian of sigma = median HFR / 1.17741 (recorded); with no PSF star the
frame is not evaluable for injection. A Moffat is a poor model of a defocused
donut, so sweep frames are not injected by default.

Placement: seeded (frame sha256, round, seed), uniform over the frame,
outside a margin, farther than ``R_avoid`` from every reference detection at
3 sigma (bounding boxes and 3x ellipses) and every unusable/hot pixel, and
2 R_avoid from each other, R_avoid = max(12 px, render radius + 3 HFR).
Identical injected pixels are given to every method (paired comparison).

Brightness: stratified over S/N bins ``SNR_BIN_EDGES``; the target S/N (log-
uniform within the bin) is ``F_ap / sqrt(N_ap R_ref^2 + F_ap / gain)`` in the
aperture r = max(3, 2 HFR_true) (the harness acceptance S/N aperture), R_ref
the reference noise at the position. Photon noise of the injected star is
added only when ``gain_e_per_adu`` is known (recorded otherwise). The daemon's
focus gates examine only the 48 brightest candidates, so each frame also gets
``bright_rounds`` rounds of ``BRIGHT_ROUND_STARS`` stars at 1.5-4 x the
reference's 48th-brightest flux (``bright_flux_range``); their bin is set from
the achieved S/N (none above 300: they then count for R7-R10, not R6).

Truth per star: position; R7 HFR truth = the noiseless, isolated injected stamp
measured with the daemon-equivalent crop estimator (``crop_estimator_truth``:
zero sky, crop = clamp(2 x the extent of the stamp's pixels above 5 frame sky
sigma, 32, 128) px placed by ``psf_measure.rust_extract_crop``, HFR by
``psf_measure.rust_compute_hfr_with_sky``, the Python twin of
``ClassicalFocusSolver::compute_hfr_with_sky``; coordinator decision (a)); the
total-flux HFR_true = ``scenarios.numerical_hfr`` of the rendered stamp is kept
for the R7-obs observation, the match radii and the apertures; aperture flux =
``sep.sum_circle`` of the noiseless injected-star image in r_flux = max(4, 3
HFR_true) (the same aperture algorithm as the measurement).

Recovery per method: a detection within max(2, 0.5 HFR_true) (one-to-one,
``metrics.match_greedy``); HFR from the method's accepted stars matched the
same way; flux = ``sep.sum_circle(I_inj - B_hat_inj)`` at the true position;
centroid error of the matched detection; dB = mean over r = max(2, 2 HFR_true)
of ``B_hat_inj - B_hat_orig`` divided by the median ``R_hat_orig`` there.
"""

from __future__ import annotations

import math

import numpy as np

SNR_BIN_EDGES = (5.0, 10.0, 20.0, 50.0, 100.0, 300.0)
PSF_MIN_SNR = 50.0
PSF_MAX_STARS = 25
PSF_MIN_FITS = 3
DEFAULT_INJECTIONS_PER_ROUND = 100
MARGIN_PX = 32
AVOID_THRESH_SIGMA = 3.0


def snr_bin(snr: float) -> int | None:
    for i, (lo, hi) in enumerate(zip(SNR_BIN_EDGES, SNR_BIN_EDGES[1:])):
        if lo <= snr < hi:
            return i
    return None


def bin_label(i: int) -> str:
    return f"{SNR_BIN_EDGES[i]:g}-{SNR_BIN_EDGES[i + 1]:g}"


def _moffat_fit(resid, valid, x0, y0):
    from scipy.optimize import least_squares

    h, w = resid.shape
    yy, xx = np.mgrid[0:h, 0:w]
    z = resid[valid]
    xv, yv = xx[valid].astype(np.float64), yy[valid].astype(np.float64)
    amp0 = float(np.max(z)) if z.size else 0.0
    if not amp0 > 0:
        return None

    def model(p):
        amp, xc, yc, alpha, beta = p
        return amp * (1.0 + ((xv - xc) ** 2 + (yv - yc) ** 2) / (alpha * alpha)) ** (-beta)

    p0 = [amp0, x0, y0, 2.0, 3.0]
    lo = [0.0, x0 - 2.0, y0 - 2.0, 0.3, 1.2]
    hi = [np.inf, x0 + 2.0, y0 + 2.0, 30.0, 10.0]
    try:
        sol = least_squares(lambda p: model(p) - z, p0, bounds=(lo, hi), max_nfev=400)
    except Exception:  # noqa: BLE001
        return None
    if not sol.success:
        return None
    amp, xc, yc, alpha, beta = sol.x
    fwhm = 2.0 * alpha * math.sqrt(2.0 ** (1.0 / beta) - 1.0)
    if not (np.isfinite(fwhm) and fwhm > 0):
        return None
    return {"fwhm_px": float(fwhm), "beta": float(beta), "alpha_px": float(alpha)}


def estimate_psf(image, masks, ref) -> dict:
    """PSF model dict for ``scenarios.render_psf`` plus provenance, or ``{"status": "not_evaluable"}``.

    PSF stars are the reference's accepted stars (x, y, hfr_px) with aperture
    S/N >= PSF_MIN_SNR measured here (``sep.sum_circle`` of I - B_ref with
    err = R_ref, r = max(3, 2 HFR)), isolated from every other catalog object."""
    if ref is None or not ref.usable or not ref.accepted:
        return {"status": "not_evaluable", "reason": "reference method unusable or no accepted star on this frame"}
    from .. import python_methods as PM
    from .methods import unusable_mask

    sep = PM.sep_module()
    back = ref.back_map(image.shape)
    rms = ref.rms_map(image.shape)
    bad = unusable_mask(image, masks)
    resid = np.ascontiguousarray(np.where(bad, 0.0, image.astype(np.float64) - np.asarray(back, dtype=np.float64)))
    acc = [a for a in ref.accepted if a.get("hfr_px")]
    xs_a = np.array([a["x"] for a in acc], dtype=np.float64)
    ys_a = np.array([a["y"] for a in acc], dtype=np.float64)
    r_ap = np.maximum(3.0, 2.0 * np.array([a["hfr_px"] for a in acc], dtype=np.float64))
    flux, ferr, _flag = sep.sum_circle(resid, xs_a, ys_a, r_ap, err=np.ascontiguousarray(rms, dtype=np.float64),
                                       subpix=5)
    snr = np.where(ferr > 0, flux / np.where(ferr > 0, ferr, 1.0), 0.0)
    cat = ref.detections or []
    xs = np.array([c["x"] for c in cat], dtype=np.float64)
    ys = np.array([c["y"] for c in cat], dtype=np.float64)
    candidates = []
    for a, f, s_n in zip(acc, flux, snr):
        if s_n < PSF_MIN_SNR:
            continue
        d = np.hypot(xs - a["x"], ys - a["y"]) if xs.size else np.array([])
        d = d[d > 0.5]
        if d.size == 0 or d.min() > max(15.0, 6.0 * a["hfr_px"]):
            candidates.append((float(f), a))
    if not candidates:
        return {"status": "not_evaluable", "reason": f"no isolated accepted reference star with S/N >= {PSF_MIN_SNR}"}
    iso = [a for _f, a in sorted(candidates, key=lambda t: -t[0])[:PSF_MAX_STARS]]
    hfrs = [a["hfr_px"] for a in iso]
    fits = []
    h, w = image.shape
    for c in iso:
        r = int(np.clip(round(5.0 * c["hfr_px"]), 8, 40))
        xi, yi = int(round(c["x"])), int(round(c["y"]))
        x0, x1, y0, y1 = xi - r, xi + r + 1, yi - r, yi + r + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            continue
        stamp = resid[y0:y1, x0:x1]
        valid = ~bad[y0:y1, x0:x1] & np.isfinite(stamp)
        fit = _moffat_fit(stamp, valid, c["x"] - x0, c["y"] - y0)
        if fit:
            fits.append(fit)
    if len(fits) >= PSF_MIN_FITS:
        psf = {"model": "moffat", "fwhm_px": float(np.median([f["fwhm_px"] for f in fits])),
               "beta": float(np.median([f["beta"] for f in fits]))}
        method = f"median of {len(fits)} circular Moffat fits"
    else:
        from .. import scenarios as SC

        psf = {"model": "gaussian", "sigma_px": float(np.median(hfrs)) / SC.GAUSSIAN_HFR_PER_SIGMA}
        method = f"gaussian from median reference HFR ({len(fits)} Moffat fits < {PSF_MIN_FITS})"
    return {"status": "completed", "psf": psf, "method": method, "n_psf_stars": len(iso), "n_fits": len(fits),
            "reference_median_hfr_px": float(np.median(hfrs))}


def avoidance_distance(image, masks, ref, hot=None):
    """Distance (px) from each pixel to the nearest reference source/unusable pixel."""
    from scipy import ndimage

    from .. import python_methods as PM
    from .masks import source_mask
    from .methods import unusable_mask

    bad = unusable_mask(image, masks)
    if hot is not None:
        bad = bad | hot
    sep = PM.sep_module()
    resid = np.ascontiguousarray(np.where(bad, 0.0, image - ref.back_map(image.shape)), dtype=np.float32)
    err = ref.rms_map(image.shape)
    det = PM.sep_detection_config_json()
    previous = sep.get_extract_pixstack()
    sep.set_extract_pixstack(max(det["pixstack"], 3_000_000))
    try:
        objs = sep.extract(resid, AVOID_THRESH_SIGMA, err=np.ascontiguousarray(err, dtype=np.float32),
                           mask=bad.astype(np.uint8), minarea=det["minarea"],
                           filter_kernel=np.asarray(det["filter_kernel"], dtype=np.float32), filter_type="conv",
                           deblend_cont=1.0, clean=False)
    finally:
        sep.set_extract_pixstack(previous)
    rows = [{"x": float(o["x"]), "y": float(o["y"]), "xmin": int(o["xmin"]), "xmax": int(o["xmax"]),
             "ymin": int(o["ymin"]), "ymax": int(o["ymax"]), "a": float(o["a"]), "b": float(o["b"]),
             "theta": float(o["theta"]), "npix": int(o["npix"])} for o in objs]
    blocked = source_mask(rows, image.shape) | bad
    return ndimage.distance_transform_edt(~blocked), len(rows)


#: R7 truth estimator (coordinator decision 2026-11-11 (a)): the Python twin of the
#: daemon's ClassicalFocusSolver::compute_hfr_with_sky, on the daemon's adaptive crop.
CROP_TRUTH_FUNCTION = "ml.labels.psf_measure.rust_compute_hfr_with_sky"
CROP_TRUTH_DETECT_SIGMA = 5.0
CROP_MIN_PX, CROP_MAX_PX = 32, 128  # quality.rs CROP_MIN_PX / CROP_MAX_PX


def crop_estimator_truth(stamp, x, y, width, height, sky_sigma):
    """``(hfr | None, info)``: HFR of the noiseless, isolated injected stamp measured
    with the daemon-equivalent crop estimator (R7 truth).

    Zero sky. Crop size = clamp(2 x the bounding-box extent of the stamp's pixels
    above ``CROP_TRUTH_DETECT_SIGMA`` x ``sky_sigma`` (the frame's sky sigma),
    32, 128) px, placed by ``psf_measure.rust_extract_crop`` (quality.rs
    extract_crop) about the true position; HFR =
    ``psf_measure.rust_compute_hfr_with_sky(crop, level=0, sigma=sky_sigma)``."""
    try:
        from ml.labels import psf_measure as LM

        estimator, extract = LM.rust_compute_hfr_with_sky, LM.rust_extract_crop
        lo, hi = getattr(LM, "FOCUS_CROP_MIN_PX", CROP_MIN_PX), getattr(LM, "FOCUS_CROP_MAX_PX", CROP_MAX_PX)
    except (ImportError, AttributeError) as exc:
        return None, {"reason": f"{CROP_TRUTH_FUNCTION} unavailable: {exc}"}
    if not (sky_sigma and sky_sigma > 0):
        return None, {"reason": "frame sky sigma unavailable"}
    above = np.argwhere(stamp.data > CROP_TRUTH_DETECT_SIGMA * sky_sigma)
    if above.size == 0:
        return None, {"reason": f"no stamp pixel above {CROP_TRUTH_DETECT_SIGMA} sky sigma (undetectable)"}
    extent = int(max(above[:, 1].max() - above[:, 1].min() + 1, above[:, 0].max() - above[:, 0].min() + 1))
    crop_size = int(min(max(2 * extent, lo), hi))
    x0, y0, cw, ch = extract(width, height, x, y, crop_size)
    crop = np.zeros((ch, cw), dtype=np.float64)
    _paste(crop, stamp.data, stamp.ix0 - x0, stamp.iy0 - y0)
    hfr = float(estimator(crop.astype(np.float32), level=0.0, sigma=float(sky_sigma)))
    return (hfr if np.isfinite(hfr) and hfr > 0 else None), {"crop_px": crop_size, "extent_px": extent}


def plan_injections(rng, image, ref, psf, distance, n, gain=None, saturation_adu=None, flux_range=None,
                    sky_sigma=None):
    """List of injected-star dicts and the noiseless injected-star image (float64).

    Default: brightness stratified over the S/N bins. ``flux_range=(lo, hi)``
    (a "bright round"): total flux log-uniform in [lo, hi]; the S/N is then the
    achieved one and the bin is assigned from it (``None`` above the top edge,
    which keeps the star out of the completeness bins but in the S/N >= 20 set)."""
    from .. import scenarios as SC

    h, w = image.shape
    unit = SC.render_psf(psf, w / 2.0, h / 2.0, 1.0)
    hfr_ref, _ = SC.numerical_hfr(unit, w / 2.0, h / 2.0)
    r_avoid = max(12.0, unit.radius + 3.0 * (hfr_ref or 2.0))
    margin = MARGIN_PX + unit.radius
    rms = ref.rms_map(image.shape)
    if sky_sigma is None:  # the frame's sky sigma: median reference noise over finite pixels
        rr = np.asarray(rms)
        sky_sigma = float(np.median(rr[np.isfinite(rr)])) if rr.size else None
    stars_img = np.zeros((h, w), dtype=np.float64)
    placed, stars = [], []
    n_bins = len(SNR_BIN_EDGES) - 1
    tries = 0
    while len(stars) < n and tries < 200 * n:
        tries += 1
        x, y = rng.uniform(margin, w - 1 - margin), rng.uniform(margin, h - 1 - margin)
        if distance[int(round(y)), int(round(x))] <= r_avoid:
            continue
        if placed and np.min(np.hypot(*(np.asarray(placed) - [x, y]).T)) < 2.0 * r_avoid:
            continue
        k = len(stars) % n_bins
        lo, hi = SNR_BIN_EDGES[k], SNR_BIN_EDGES[k + 1]
        target = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        stamp = SC.render_psf(psf, x, y, 1.0)
        hfr_true, _ = SC.numerical_hfr(stamp, x, y)
        r_snr = max(3.0, 2.0 * hfr_true)
        frac = SC.enclosed_flux(stamp, x, y, r_snr) / float(stamp.data.sum())
        sigma = float(rms[int(round(y)), int(round(x))])
        var_bg = math.pi * r_snr * r_snr * sigma * sigma
        if flux_range:
            flux = float(math.exp(rng.uniform(math.log(flux_range[0]), math.log(flux_range[1]))))
            f_ap = flux * frac * float(stamp.data.sum())
            target = f_ap / math.sqrt(var_bg + (f_ap / gain if gain else 0.0))
            k = snr_bin(target)
            flux = flux / float(stamp.data.sum())
        elif gain:
            a = target * target / gain
            f_ap = 0.5 * (a + math.sqrt(a * a + 4.0 * target * target * var_bg))
            flux = f_ap / frac / float(stamp.data.sum())
        else:
            f_ap = target * math.sqrt(var_bg)
            flux = f_ap / frac / float(stamp.data.sum())
        data = stamp.data * flux
        _paste(stars_img, data, stamp.ix0, stamp.iy0)
        placed.append((x, y))
        scaled = SC.Stamp(data, stamp.ix0, stamp.iy0, stamp.radius)
        hfr_crop, crop_info = crop_estimator_truth(scaled, x, y, w, h, sky_sigma)
        stars.append({"x": float(x), "y": float(y), "flux_total": float(data.sum()), "hfr_true_px": float(hfr_true),
                      "hfr_truth_crop_px": hfr_crop, "crop_truth": crop_info,
                      "snr_target": target, "bin": k, "r_snr_px": r_snr, "r_flux_px": max(4.0, 3.0 * hfr_true),
                      "r_db_px": max(2.0, 2.0 * hfr_true), "render_radius_px": float(stamp.radius),
                      "saturated": False, "round_kind": "bright" if flux_range else "stratified"})
    if saturation_adu:
        total = image.astype(np.float64) + stars_img
        for s in stars:
            r = int(math.ceil(s["render_radius_px"]))
            xi, yi = int(round(s["x"])), int(round(s["y"]))
            win = total[max(0, yi - r):yi + r + 1, max(0, xi - r):xi + r + 1]
            s["saturated"] = bool(win.size and np.nanmax(win) >= 0.98 * saturation_adu)
    return stars, stars_img, {"r_avoid_px": r_avoid, "placed": len(stars), "requested": n, "tries": tries,
                              "sky_sigma": sky_sigma, "r7_truth": CROP_TRUTH_FUNCTION}


#: The daemon's focus quality examines only the brightest candidates
#: (``quality.rs`` / ``psf_measure.FOCUS_MAX_STARS_EXAMINED`` = 48): stratified
#: injections in a rich field are detected but never reach the HFR gates. A
#: bright round injects ``BRIGHT_ROUND_STARS`` stars with total flux 1.5-4 x the
#: reference's 48th-brightest detection (peak <= half the saturation level when
#: known) so that R7 (HFR bias) samples stars the gates actually see.
FOCUS_MAX_STARS_EXAMINED = 48
BRIGHT_ROUND_STARS = 40


def bright_flux_range(ref, psf, saturation_adu=None):
    """``(lo, hi)`` total flux of a bright round, or None when the reference has too few detections."""
    from .. import scenarios as SC

    fluxes = sorted((float(c["flux"]) for c in (ref.detections or []) if c.get("flux")), reverse=True)
    if len(fluxes) < FOCUS_MAX_STARS_EXAMINED:
        lo = (fluxes[-1] if fluxes else None)
        if lo is None:
            return None
    else:
        lo = fluxes[FOCUS_MAX_STARS_EXAMINED - 1]
    lo, hi = 1.5 * lo, 4.0 * lo
    if saturation_adu:
        unit = SC.render_psf(psf, 0.0, 0.0, 1.0)
        cap = 0.5 * saturation_adu / max(float(unit.data.max()), 1e-12)
        if cap < lo:
            return None
        hi = min(hi, cap)
    return lo, hi


def _paste(target, data, ix0, iy0):
    h, w = target.shape
    ny, nx = data.shape
    x0, y0, x1, y1 = max(0, ix0), max(0, iy0), min(w, ix0 + nx), min(h, iy0 + ny)
    if x0 < x1 and y0 < y1:
        target[y0:y1, x0:x1] += data[y0 - iy0:y1 - iy0, x0 - ix0:x1 - ix0]


def injected_image(image, masks, stars_img, rng, gain=None, saturation_adu=None):
    out = image.astype(np.float64) + stars_img
    noise = "none (gain_e_per_adu unknown)"
    if gain:
        out += rng.standard_normal(out.shape) * np.sqrt(np.clip(stars_img, 0.0, None) / gain)
        noise = f"gaussian photon noise of the injected flux, gain {gain} e-/ADU"
    new_masks = dict(masks)
    if saturation_adu:
        sat = out >= saturation_adu
        if sat.any():
            new_masks["native_saturation"] = (masks.get("native_saturation", np.zeros(out.shape, np.uint8)).astype(bool)
                                              | sat).astype(np.uint8)
    return np.ascontiguousarray(out, dtype=np.float32), new_masks, noise


def true_aperture_flux(stars_img, stars):
    from .. import python_methods as PM

    sep = PM.sep_module()
    if not stars:
        return
    data = np.ascontiguousarray(stars_img, dtype=np.float64)
    xs = np.array([s["x"] for s in stars])
    ys = np.array([s["y"] for s in stars])
    rs = np.array([s["r_flux_px"] for s in stars])
    flux, _err, _flag = sep.sum_circle(data, xs, ys, rs, subpix=5)
    for s, f in zip(stars, flux):
        s["flux_ap_true"] = float(f)


def score_round(stars, image_inj, out_inj, out_orig) -> list:
    """Per-star recovery records for one method on one injection round."""
    from .. import metrics as MT
    from .. import python_methods as PM

    records = [{"bin": s["bin"], "snr": s["snr_target"], "saturated": s["saturated"], "detected": False,
                "round_kind": s.get("round_kind", "stratified"),
                "hfr_rel": None, "hfr_rel_total_flux": None, "flux_rel": None, "centroid_px": None,
                "delta_b_sigma": None} for s in stars]
    if not stars:
        return records
    xy = [(s["x"], s["y"]) for s in stars]
    radii = [max(2.0, 0.5 * s["hfr_true_px"]) for s in stars]
    if out_inj.status == "completed" and out_inj.detections:
        for i, j, d in MT.match_greedy([(c["x"], c["y"]) for c in out_inj.detections], xy, radii):
            records[j]["detected"] = True
            records[j]["centroid_px"] = d
    if out_inj.status == "completed" and out_inj.accepted:
        acc = out_inj.accepted
        for i, j, _d in MT.match_greedy([(a["x"], a["y"]) for a in acc], xy, radii):
            hfr = acc[i].get("hfr_px")
            if hfr:
                truth = stars[j].get("hfr_truth_crop_px")
                if truth:
                    records[j]["hfr_rel"] = (hfr - truth) / truth  # R7: daemon crop-estimator truth
                records[j]["hfr_rel_total_flux"] = (hfr - stars[j]["hfr_true_px"]) / stars[j]["hfr_true_px"]
    if out_inj.usable:
        sep = PM.sep_module()
        back = out_inj.back_map(image_inj.shape)
        resid = np.ascontiguousarray(image_inj.astype(np.float64) - np.asarray(back, dtype=np.float64))
        resid = np.where(np.isfinite(resid), resid, 0.0)
        xs, ys = np.array([s["x"] for s in stars]), np.array([s["y"] for s in stars])
        flux, _e, _f = sep.sum_circle(resid, xs, ys, np.array([s["r_flux_px"] for s in stars]), subpix=5)
        for rec, s, f in zip(records, stars, flux):
            if s.get("flux_ap_true"):
                rec["flux_rel"] = (float(f) - s["flux_ap_true"]) / s["flux_ap_true"]
    if out_inj.usable and out_orig is not None and out_orig.usable:
        shape = image_inj.shape
        b1, b0, r0 = out_inj.back_map(shape), out_orig.back_map(shape), out_orig.rms_map(shape)
        for rec, s in zip(records, stars):
            r = s["r_db_px"]
            x0, x1 = max(0, int(s["x"] - r - 1)), min(shape[1], int(s["x"] + r + 2))
            y0, y1 = max(0, int(s["y"] - r - 1)), min(shape[0], int(s["y"] + r + 2))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            disk = (xx - s["x"]) ** 2 + (yy - s["y"]) ** 2 <= r * r
            db = np.asarray(b1[y0:y1, x0:x1], dtype=np.float64)[disk] - np.asarray(b0[y0:y1, x0:x1],
                                                                                   dtype=np.float64)[disk]
            sig = float(np.median(np.asarray(r0[y0:y1, x0:x1], dtype=np.float64)[disk]))
            if db.size and sig > 0:
                rec["delta_b_sigma"] = float(np.mean(db) / sig)
    return records


def aggregate(records, min_snr_for_bias=20.0) -> dict:
    """Night-level injection summary for one method."""
    recs = [r for r in records if not r["saturated"]]
    bins = []  # stars without a bin (bright rounds above the top edge) are not in the completeness bins
    for k in range(len(SNR_BIN_EDGES) - 1):
        sel = [r for r in recs if r["bin"] == k]
        n_rec = sum(r["detected"] for r in sel)
        bins.append({"bin": bin_label(k), "lo": SNR_BIN_EDGES[k], "hi": SNR_BIN_EDGES[k + 1], "n_injected": len(sel),
                     "n_recovered": n_rec, "completeness": (n_rec / len(sel)) if sel else None})
    hi = [r for r in recs if r["snr"] >= min_snr_for_bias]

    def med(key, rows):
        v = [r[key] for r in rows if r[key] is not None]
        return (float(np.median(v)) if v else None), len(v)

    hfr, n_hfr = med("hfr_rel", hi)
    hfr_tot, n_hfr_tot = med("hfr_rel_total_flux", hi)
    flux, n_flux = med("flux_rel", hi)
    cen, n_cen = med("centroid_px", hi)
    db, n_db = med("delta_b_sigma", recs)
    by_kind = {}
    for kind in sorted({r.get("round_kind", "stratified") for r in hi}):
        rows = [r for r in hi if r.get("round_kind", "stratified") == kind]
        by_kind[kind] = {"n": len(rows), "median_rel_hfr": med("hfr_rel", rows)[0], "n_hfr": med("hfr_rel", rows)[1],
                         "median_rel_hfr_total_flux_truth": med("hfr_rel_total_flux", rows)[0]}
    return {"n_injected": len(recs), "n_saturated_excluded": len(records) - len(recs), "bins": bins,
            "snr_ge_20_by_round_kind": by_kind,
            f"snr_ge_{min_snr_for_bias:g}": {"n": len(hi), "median_rel_hfr": hfr, "n_hfr": n_hfr,
                                               "hfr_truth": f"{CROP_TRUTH_FUNCTION} on the noiseless stamp (R7)",
                                               "median_rel_hfr_total_flux_truth": hfr_tot,
                                               "n_hfr_total_flux_truth": n_hfr_tot,
                                               "median_rel_flux": flux, "n_flux": n_flux,
                                               "median_centroid_px": cen, "n_centroid": n_cen},
            "delta_b": {"median_sigma": db, "n": n_db}}
