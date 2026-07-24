"""
psf_measure — reference PSF/quality labels for the FQE dataset (Section 7.7.5).

This module produces the *ground-truth labels* that replace the hand-wavy
reciprocal focus scores in the daemon (``achernar-daemon/src/vision/quality.rs``).
It follows the source-detection / background / half-light-radius foundation of
SExtractor (Bertin & Arnouts 1996), SEP (Barbary 2016) and DAOPHOT
(Stetson 1987), and the PSF-width conventions of Law et al. (2006).

Key facts this module encodes (and that the daemon currently gets wrong):

  * **HFR** (half-flux radius) is measured by a curve of growth after *local*
    background subtraction, interpolating between cumulative-flux samples.
  * **HFD = 2 * HFR** by definition — this is what the daemon mislabels as FWHM.
  * **FWHM** is a *fitted* PSF width. ``FWHM = 2 * HFR`` holds only for a
    circular Gaussian, not for a Moffat PSF. We therefore fit a model and
    report the fitted FWHM separately, never ``2 * HFR``.
  * Missing / invalid measurements are ``None`` (never ``0.0``), because 0 is the
    mathematically *best* value for a lower-is-better width metric and would let
    an invalid frame win a focus comparison.

All measurements are in **pixels**. Conversion to arc-seconds (seeing) requires
a known pixel scale and is intentionally *not* done here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional

import numpy as np

try:
    from scipy.optimize import least_squares

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - scipy is a hard dep for fitting
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# Rejection taxonomy — why a candidate star is not usable for a width label.
# --------------------------------------------------------------------------- #
class Rejection(str, Enum):
    SATURATED = "saturated"
    EDGE_TRUNCATED = "edge_truncated"
    LOW_SNR = "low_snr"
    TOO_FEW_PIXELS = "too_few_pixels"
    ELONGATED = "elongated"
    BLENDED = "blended"
    FIT_FAILED = "fit_failed"


@dataclass
class StarMeasurement:
    """Per-star measurement. ``valid`` gates whether it contributes a label."""

    x: float
    y: float
    flux: float
    peak: float
    snr: float
    hfr_px: Optional[float]
    fwhm_px: Optional[float]  # fitted; None if the fit failed / was not run
    moffat_beta: Optional[float]
    ellipticity: float
    valid: bool
    rejections: list[str]


@dataclass
class FrameLabel:
    """Robust frame-level label aggregated over accepted stars."""

    hfr_px_median: Optional[float]
    hfr_px_mad: Optional[float]
    fwhm_px_median: Optional[float]
    fwhm_px_mad: Optional[float]
    ellipticity_median: Optional[float]
    star_count_detected: int
    star_count_used: int
    sky_median_adu: float
    sky_rms_adu: float
    saturated_star_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Background (SExtractor-style robust sky estimate).
# --------------------------------------------------------------------------- #
def estimate_background(image: np.ndarray, sigma: float = 3.0, iters: int = 5) -> tuple[float, float]:
    """Iterative sigma-clipped sky median and RMS (a global approximation of the
    SExtractor mesh background — adequate for small autofocus crops)."""
    data = image[np.isfinite(image)].astype(np.float64).ravel()
    if data.size == 0:
        return 0.0, 0.0
    for _ in range(iters):
        med = np.median(data)
        mad = np.median(np.abs(data - med)) * 1.4826
        if mad <= 0:
            break
        keep = np.abs(data - med) <= sigma * mad
        if keep.all() or keep.sum() < 8:
            break
        data = data[keep]
    med = float(np.median(data))
    rms = float(np.std(data)) if data.size > 1 else 0.0
    return med, rms


# --------------------------------------------------------------------------- #
# Source detection: find star crops in a full frame (for the dataset builder).
# --------------------------------------------------------------------------- #
def extract_centered_crop(
    image: np.ndarray, cx: float, cy: float, crop_size: int = 64
) -> Optional[np.ndarray]:
    """Return a truly centered fixed-size crop, or ``None`` at frame edges.

    Positive coordinates use round-half-up, matching Rust ``f32::round`` in
    ``vision/preprocess.rs``. Training and edge inference must not disagree by a
    pixel or shift an edge source into a context the model never saw.
    """
    image = np.asarray(image)
    if image.ndim != 2 or not np.isfinite(cx) or not np.isfinite(cy):
        return None
    half = crop_size // 2
    center_x = int(np.floor(float(cx) + 0.5))
    center_y = int(np.floor(float(cy) + 0.5))
    x0, y0 = center_x - half, center_y - half
    h, w = image.shape
    if x0 < 0 or y0 < 0 or x0 + crop_size > w or y0 + crop_size > h:
        return None
    return image[y0 : y0 + crop_size, x0 : x0 + crop_size].copy()


def detect_sources(
    image: np.ndarray,
    sky: Optional[float] = None,
    sky_rms: Optional[float] = None,
    threshold_sigma: float = 5.0,
    min_area: int = 3,
    max_area: int = 2000,
    crop_size: int = 64,
    edge_margin: int = 2,
) -> list[tuple[np.ndarray, float, float]]:
    """Detect bright connected components above `sky + k*rms` and return a list
    of ``(crop, cx, cy)`` for each.

    This is a lightweight, SExtractor-*inspired* segmentation (plain
    scipy.ndimage connected components) with NO deblending, matched filter, or
    cosmic-ray rejection — do not mistake it for SEP. For the real SEP path
    (mesh background, deblending, aperture-bounded ``flux_radius``) use
    :func:`measure_frame_sep`. Crops truncated by the frame edge are dropped."""
    from scipy import ndimage

    image = np.asarray(image, dtype=np.float64)
    if sky is None or sky_rms is None:
        sky, sky_rms = estimate_background(image)
    thresh = sky + threshold_sigma * max(sky_rms, 1e-6)
    labels, n = ndimage.label(image > thresh)
    if n == 0:
        return []

    h, w = image.shape
    out: list[tuple[np.ndarray, float, float]] = []
    centroids = ndimage.center_of_mass(image - sky, labels, range(1, n + 1))
    sizes = ndimage.sum(np.ones_like(image), labels, range(1, n + 1))
    for (cy, cx), area in zip(centroids, sizes):
        if area < min_area or area > max_area:
            continue
        crop = extract_centered_crop(image, float(cx), float(cy), crop_size)
        if crop is None:
            continue  # would be edge-truncated
        out.append((crop, float(cx), float(cy)))
    return out


# --------------------------------------------------------------------------- #
# Curve-of-growth half-flux radius (the primary FQE label).
# --------------------------------------------------------------------------- #
def _cog_once(resid_flat, r_flat, max_radius):
    """One curve-of-growth pass within ``max_radius``. Returns (hfr, total_flux).

    The flux is summed UNCLIPPED so zero-mean sky noise cancels instead of
    forming a positive pedestal. The half-flux crossing is taken as the first
    radius where the cumulative reaches half of the aperture total; that crossing
    lies in the steeply-rising stellar core, so it is robust to noise wiggles in
    the wings."""
    keep = r_flat <= max_radius
    if keep.sum() < 4:
        return None, 0.0
    r = r_flat[keep]
    v = resid_flat[keep]
    order = np.argsort(r)
    r_sorted = r[order]
    cumulative = np.cumsum(v[order])
    total = float(cumulative[-1])
    if total <= 0:
        return None, total

    half = 0.5 * total
    reached = cumulative >= half
    if not reached.any():
        return float(r_sorted[-1]), total
    idx = int(np.argmax(reached))
    if idx == 0:
        return float(r_sorted[0]), total
    c0, c1 = cumulative[idx - 1], cumulative[idx]
    r0, r1 = r_sorted[idx - 1], r_sorted[idx]
    if c1 == c0:
        return float(r1), total
    frac = (half - c0) / (c1 - c0)
    return float(r0 + frac * (r1 - r0)), total


def curve_of_growth_hfr(
    crop: np.ndarray,
    sky: float,
    sky_rms: Optional[float] = None,
    max_radius: Optional[float] = None,
    centroid: Optional[tuple[float, float]] = None,
) -> Optional[float]:
    """Aperture-bounded, locally-background-subtracted half-flux radius (px).

    Two properties that the previous implementation got wrong and that this one
    guarantees (see the astronomical-quality-metrics research note):

    * **No noise pedestal.** The sky-subtracted flux is integrated *without*
      clipping negatives to zero, so background noise cancels in the mean. The
      old ``clip(crop - sky, 0)`` rectified every positive noise excursion into a
      pedestal that grew HFR with the noise level (a true 3 px star measured
      16 px at high noise).
    * **Bounded aperture.** Integration is limited to a finite radius adapted to
      the star (~4x an initial HFR estimate), so enlarging the crop cannot
      accumulate more noise-only annuli (the old code grew HFR from 3.6 px to
      32 px purely by changing the crop size).

    Returns ``None`` when no positive stellar flux is present — an explicit
    "no width", never a spuriously-perfect 0.
    """
    crop = np.asarray(crop, dtype=np.float64)
    h, w = crop.shape
    resid = crop - sky
    ys, xs = np.mgrid[0:h, 0:w]

    # Robust centroid from the thresholded core. Clipping is fine for *locating*
    # the star; the bug was clipping when *integrating* its flux.
    if centroid is None:
        thr = 3.0 * max(sky_rms, 1e-6) if sky_rms is not None else 0.0
        core = np.clip(resid - thr, 0.0, None)
        core_total = float(core.sum())
        if core_total <= 0:
            return None
        cx = float((xs * core).sum() / core_total)
        cy = float((ys * core).sum() / core_total)
    else:
        cx, cy = centroid

    r_flat = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).ravel()
    resid_flat = resid.ravel()
    hard_cap = min(h, w) / 2.0 - 1.0

    if max_radius is not None:
        hfr, _ = _cog_once(resid_flat, r_flat, min(max_radius, hard_cap))
        return hfr

    # Pass 1: rough HFR within a modest cap. Pass 2: bound the aperture to ~4x
    # that estimate so noise-only outer annuli stay out of the integral.
    est, total = _cog_once(resid_flat, r_flat, min(hard_cap, 20.0))
    if est is None or total <= 0:
        return None
    aperture = float(np.clip(4.0 * est, 6.0, hard_cap))
    hfr, _ = _cog_once(resid_flat, r_flat, aperture)
    return hfr


# --------------------------------------------------------------------------- #
# Second-moment shape (ellipticity) — cheap, no fit required.
# --------------------------------------------------------------------------- #
def second_moment_shape(crop: np.ndarray, sky: float) -> tuple[float, float]:
    """Return (ellipticity = 1 - b/a, elongation = a/b) from flux-weighted
    second moments. Elongation flags trailed/blended stars before they poison
    a width label."""
    h, w = crop.shape
    ys, xs = np.mgrid[0:h, 0:w]
    f = np.clip(crop.astype(np.float64) - sky, 0.0, None)
    total = f.sum()
    if total <= 0:
        return 0.0, 1.0
    cx = (xs * f).sum() / total
    cy = (ys * f).sum() / total
    mxx = (f * (xs - cx) ** 2).sum() / total
    myy = (f * (ys - cy) ** 2).sum() / total
    mxy = (f * (xs - cx) * (ys - cy)).sum() / total
    common = np.sqrt(max(((mxx - myy) / 2) ** 2 + mxy**2, 0.0))
    a2 = (mxx + myy) / 2 + common
    b2 = (mxx + myy) / 2 - common
    a = np.sqrt(max(a2, 1e-12))
    b = np.sqrt(max(b2, 1e-12))
    ellipticity = float(1.0 - b / a) if a > 0 else 0.0
    elongation = float(a / b) if b > 0 else np.inf
    return ellipticity, elongation


# --------------------------------------------------------------------------- #
# Circular Moffat fit -> fitted FWHM + beta. Falls back to a Gaussian-moment
# FWHM when scipy is unavailable or the fit fails.
# --------------------------------------------------------------------------- #
def _moffat_radial(r, amp, alpha, beta, offset):
    return offset + amp * (1.0 + (r / alpha) ** 2) ** (-beta)


def fit_moffat_fwhm(crop: np.ndarray, sky: float) -> tuple[Optional[float], Optional[float]]:
    """Fit a circular Moffat and return (fwhm_px, beta).

    ``FWHM = 2 * alpha * sqrt(2**(1/beta) - 1)``. Returns (None, None) if the
    fit is not usable. Note this is a genuinely different quantity from
    ``2 * HFR`` unless the PSF is Gaussian (beta -> inf)."""
    h, w = crop.shape
    ys, xs = np.mgrid[0:h, 0:w]
    f = np.clip(crop.astype(np.float64) - sky, 0.0, None)
    total = f.sum()
    if total <= 0:
        return None, None
    cx = (xs * f).sum() / total
    cy = (ys * f).sum() / total
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).ravel()
    z = (crop.astype(np.float64) - sky).ravel()

    if not _HAVE_SCIPY:
        return _gaussian_moment_fwhm(crop, sky), None

    amp0 = float(z.max())
    if amp0 <= 0:
        return None, None
    p0 = np.array([amp0, 2.0, 2.5, 0.0])

    def resid(p):
        amp, alpha, beta, offset = p
        alpha = max(alpha, 1e-3)
        beta = max(beta, 0.5)
        return _moffat_radial(r, amp, alpha, beta, offset) - z

    try:
        sol = least_squares(
            resid,
            p0,
            bounds=([0.0, 1e-3, 0.6, -abs(amp0)], [np.inf, max(h, w), 20.0, abs(amp0)]),
            max_nfev=200,
        )
    except Exception:
        return _gaussian_moment_fwhm(crop, sky), None
    if not sol.success:
        return _gaussian_moment_fwhm(crop, sky), None

    _, alpha, beta, _ = sol.x
    if beta <= 0.5 or alpha <= 0:
        return _gaussian_moment_fwhm(crop, sky), None
    fwhm = float(2.0 * alpha * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    return fwhm, float(beta)


def _gaussian_moment_fwhm(crop: np.ndarray, sky: float) -> Optional[float]:
    """FWHM from the flux-weighted radial second moment assuming a Gaussian
    core: FWHM = 2*sqrt(2 ln 2) * sigma."""
    h, w = crop.shape
    ys, xs = np.mgrid[0:h, 0:w]
    f = np.clip(crop.astype(np.float64) - sky, 0.0, None)
    total = f.sum()
    if total <= 0:
        return None
    cx = (xs * f).sum() / total
    cy = (ys * f).sum() / total
    r2 = (xs - cx) ** 2 + (ys - cy) ** 2
    sigma = np.sqrt((f * r2).sum() / total / 2.0)  # 2D -> per-axis variance
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma)


def _count_core_peaks(crop: np.ndarray, sky: float, sky_rms: float, min_sep: int = 3) -> int:
    """Count distinct local maxima above sky + 5*rms — a cheap deblend test for
    the crop path. Two peaks => a close double whose merged profile broadens HFR."""
    try:
        from scipy.ndimage import maximum_filter
    except ImportError:  # pragma: no cover
        return 1
    crop = np.asarray(crop, dtype=np.float64)
    thresh = sky + 5.0 * max(sky_rms, 1e-6)
    footprint = 2 * min_sep + 1
    is_peak = (crop == maximum_filter(crop, size=footprint)) & (crop > thresh)
    # Collapse plateau peaks that touch into a single detection.
    from scipy.ndimage import label as _label

    _, n = _label(is_peak)
    return int(n)


# --------------------------------------------------------------------------- #
# Per-star measurement with source-validity gates.
# --------------------------------------------------------------------------- #
def measure_star(
    crop: np.ndarray,
    *,
    sky: Optional[float] = None,
    sky_rms: Optional[float] = None,
    saturation_level: float,
    min_snr: float = 5.0,
    max_ellipticity: float = 0.35,
    edge_margin: int = 2,
    fit_fwhm: bool = True,
) -> StarMeasurement:
    """Measure one star crop and decide whether it may contribute a width label.

    The gates implement the research note's source-validity checks: reject
    saturated, edge-truncated, low-SNR, blended/elongated stars *before* they
    can bias the aggregate HFR or become a selected focus star."""
    crop = np.asarray(crop, dtype=np.float64)
    h, w = crop.shape
    if sky is None or sky_rms is None:
        sky, sky_rms = estimate_background(crop)

    rejections: list[str] = []

    signal = np.clip(crop - sky, 0.0, None)
    total = float(signal.sum())
    peak = float(crop.max())
    noise = max(sky_rms, 1e-6)

    if peak >= saturation_level:
        rejections.append(Rejection.SATURATED.value)

    ys, xs = np.mgrid[0:h, 0:w]
    if total > 0:
        cx = float((xs * signal).sum() / total)
        cy = float((ys * signal).sum() / total)
        if (
            cx < edge_margin
            or cy < edge_margin
            or cx > (w - 1 - edge_margin)
            or cy > (h - 1 - edge_margin)
        ):
            rejections.append(Rejection.EDGE_TRUNCATED.value)
    else:
        cx, cy = w / 2.0, h / 2.0

    # Aperture SNR proxy (sky-noise-dominated; no gain/read-noise model — hence
    # a *proxy*): sky-subtracted flux summed inside a fixed core aperture around
    # the centroid, over the propagated sky noise. The sum is *unclipped* so that
    # zero-mean sky noise cancels; clipping to >=0 first would rectify noise and
    # spuriously inflate the SNR of a blank/buried patch.
    r_ap = max(4.0, min(h, w) / 8.0)
    aperture = ((xs - cx) ** 2 + (ys - cy) ** 2) <= r_ap**2
    ap_signal = float((crop[aperture] - sky).sum())
    n_ap = float(aperture.sum())
    # "significant" pixels: those above a 3-sigma detection threshold.
    core_pixels = int((signal[aperture] > 3.0 * noise).sum())
    snr = ap_signal / (noise * float(np.sqrt(max(n_ap, 1.0))))

    if snr < min_snr:
        rejections.append(Rejection.LOW_SNR.value)
    if core_pixels < 5:
        rejections.append(Rejection.TOO_FEW_PIXELS.value)

    ellipticity, elongation = second_moment_shape(crop, sky)
    if ellipticity > max_ellipticity:
        rejections.append(Rejection.ELONGATED.value)

    # Blend gate: a clean star has one core peak. Two or more well-separated
    # peaks above 5-sigma mean a close double whose blended profile would give a
    # spuriously broad HFR — reject it (SEP does this by deblending; the
    # crop path detects and drops it).
    if _count_core_peaks(crop, sky, noise) >= 2:
        rejections.append(Rejection.BLENDED.value)

    hfr = curve_of_growth_hfr(crop, sky, sky_rms=noise, centroid=(cx, cy))
    fwhm, beta = (None, None)
    if fit_fwhm and not rejections:
        fwhm, beta = fit_moffat_fwhm(crop, sky)
        if fwhm is None:
            rejections.append(Rejection.FIT_FAILED.value)

    valid = len(rejections) == 0 and hfr is not None and hfr > 0
    return StarMeasurement(
        x=cx,
        y=cy,
        flux=total,
        peak=peak,
        snr=float(snr),
        hfr_px=hfr,
        fwhm_px=fwhm,
        moffat_beta=beta,
        ellipticity=float(ellipticity),
        valid=valid,
        rejections=rejections,
    )


def _robust_median_mad(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)) * 1.4826)
    return med, mad


def aggregate_frame(
    measurements: list[StarMeasurement],
    sky_median: float,
    sky_rms: float,
) -> FrameLabel:
    """Robustly pool accepted stars into a frame label (median + MAD). This is
    the transparent robust pooling that Bosse et al. (2018) recommend *before*
    any learned patch weighting."""
    used = [m for m in measurements if m.valid]
    hfrs = [m.hfr_px for m in used if m.hfr_px is not None]
    fwhms = [m.fwhm_px for m in used if m.fwhm_px is not None]
    ells = [m.ellipticity for m in used]
    saturated = sum(1 for m in measurements if Rejection.SATURATED.value in m.rejections)

    hfr_med, hfr_mad = _robust_median_mad(hfrs)
    fwhm_med, fwhm_mad = _robust_median_mad(fwhms)
    ell_med, _ = _robust_median_mad(ells)

    return FrameLabel(
        hfr_px_median=hfr_med,
        hfr_px_mad=hfr_mad,
        fwhm_px_median=fwhm_med,
        fwhm_px_mad=fwhm_mad,
        ellipticity_median=ell_med,
        star_count_detected=len(measurements),
        star_count_used=len(used),
        sky_median_adu=float(sky_median),
        sky_rms_adu=float(sky_rms),
        saturated_star_fraction=(saturated / len(measurements)) if measurements else 0.0,
    )


# --------------------------------------------------------------------------- #
# Real SEP path: mesh background + deblending extraction + flux_radius.
# This is the labeler the dataset builder should use on full frames. Unlike the
# crop helpers above, it genuinely uses SEP (Barbary 2016) for a spatial mesh
# background (handles gradients/vignetting), matched-filter detection WITH
# deblending (separates close doubles), and an aperture-bounded half-light
# radius. `require_sep=False` lets callers degrade to the crop path if SEP is
# unavailable.
# --------------------------------------------------------------------------- #
@dataclass
class SepSource:
    x: float
    y: float
    flux: float
    peak: float
    snr: float
    hfr_px: Optional[float]
    fwhm_px: Optional[float]
    ellipticity: float
    valid: bool
    rejections: list[str]


def measure_frame_sep(
    image: np.ndarray,
    *,
    saturation_level: float,
    thresh_sigma: float = 5.0,
    deblend_cont: float = 0.005,
    min_area: int = 5,
    max_ellipticity: float = 0.35,
    min_snr: float = 5.0,
    fit_fwhm: bool = True,
) -> tuple[list[SepSource], FrameLabel]:
    """Extract and measure every source in a full frame with SEP.

    Returns ``(sources, frame_label)``. Each source carries SEP-derived HFR
    (``sep.flux_radius``, aperture-bounded and locally background subtracted),
    ellipticity from the second-moment ``a``/``b``, an optional fitted Moffat
    FWHM, and rejection flags for saturation, edge truncation, elongation, and
    SEP's own blend (``OBJ_MERGED``) / truncation (``OBJ_TRUNC``) bits.
    """
    import sep

    data = np.ascontiguousarray(image, dtype=np.float32)
    bkg = sep.Background(data)
    data_sub = data - bkg.back()
    rms = float(bkg.globalrms)

    objects = sep.extract(
        data_sub,
        thresh_sigma,
        err=rms,
        minarea=min_area,
        deblend_nthresh=32,
        deblend_cont=deblend_cont,
    )
    h, w = data.shape
    sources: list[SepSource] = []

    if len(objects) > 0:
        x = objects["x"]
        y = objects["y"]
        # Half-light radius, bounded to 6x the isophotal semi-major axis. Let SEP
        # measure the normalization flux inside that aperture; objects["flux"] is
        # isophotal and undercounts Moffat wings, biasing HFR low.
        rmax = np.maximum(6.0 * objects["a"], 6.0)
        rhalf, rflag = sep.flux_radius(data_sub, x, y, rmax, 0.5, normflux=None, subpix=5)
        aperture_radius = np.maximum(3.0 * objects["a"], 4.0)
        aperture_flux, aperture_fluxerr, aperture_flag = sep.sum_circle(
            data_sub,
            x,
            y,
            aperture_radius,
            err=bkg.rms(),
            subpix=5,
        )

        for i in range(len(objects)):
            rej: list[str] = []
            a = float(objects["a"][i])
            b = float(objects["b"][i])
            ellip = float(1.0 - b / a) if a > 0 else 0.0
            iy = int(np.clip(np.floor(float(y[i]) + 0.5), 0, h - 1))
            ix = int(np.clip(np.floor(float(x[i]) + 0.5), 0, w - 1))
            peak = float(objects["peak"][i]) + float(bkg.back()[iy, ix])
            fluxerr = float(aperture_fluxerr[i])
            snr = (
                float(aperture_flux[i]) / fluxerr
                if np.isfinite(fluxerr) and fluxerr > 0
                else 0.0
            )

            flag = int(objects["flag"][i])
            if flag & sep.OBJ_MERGED:
                rej.append(Rejection.BLENDED.value)
            if flag & sep.OBJ_TRUNC:
                rej.append(Rejection.EDGE_TRUNCATED.value)
            if int(rflag[i]) & sep.APER_TRUNC or int(aperture_flag[i]) & sep.APER_TRUNC:
                if Rejection.EDGE_TRUNCATED.value not in rej:
                    rej.append(Rejection.EDGE_TRUNCATED.value)
            if peak >= saturation_level:
                rej.append(Rejection.SATURATED.value)
            if not np.isfinite(snr) or snr < min_snr:
                rej.append(Rejection.LOW_SNR.value)
            if ellip > max_ellipticity:
                rej.append(Rejection.ELONGATED.value)

            hfr = float(rhalf[i]) if np.isfinite(rhalf[i]) and rhalf[i] > 0 and int(rflag[i]) == 0 else None
            fwhm = None
            if fit_fwhm and not rej and hfr is not None:
                half = 24
                center_x = int(np.floor(float(x[i]) + 0.5))
                center_y = int(np.floor(float(y[i]) + 0.5))
                x0, y0 = center_x - half, center_y - half
                if 0 <= x0 and 0 <= y0 and x0 + 2 * half <= w and y0 + 2 * half <= h:
                    crop = data[y0 : y0 + 2 * half, x0 : x0 + 2 * half]
                    fwhm, _ = fit_moffat_fwhm(crop, float(bkg.globalback))

            valid = not rej and hfr is not None
            sources.append(
                SepSource(
                    x=float(x[i]),
                    y=float(y[i]),
                    flux=float(aperture_flux[i]),
                    peak=peak,
                    snr=snr,
                    hfr_px=hfr,
                    fwhm_px=fwhm,
                    ellipticity=ellip,
                    valid=valid,
                    rejections=rej,
                )
            )

    used = [s for s in sources if s.valid]
    hfr_med, hfr_mad = _robust_median_mad([s.hfr_px for s in used if s.hfr_px is not None])
    fwhm_med, fwhm_mad = _robust_median_mad([s.fwhm_px for s in used if s.fwhm_px is not None])
    ell_med, _ = _robust_median_mad([s.ellipticity for s in used])
    saturated = sum(1 for s in sources if Rejection.SATURATED.value in s.rejections)
    label = FrameLabel(
        hfr_px_median=hfr_med,
        hfr_px_mad=hfr_mad,
        fwhm_px_median=fwhm_med,
        fwhm_px_mad=fwhm_mad,
        ellipticity_median=ell_med,
        star_count_detected=len(sources),
        star_count_used=len(used),
        sky_median_adu=float(bkg.globalback),
        sky_rms_adu=rms,
        saturated_star_fraction=(saturated / len(sources)) if sources else 0.0,
    )
    return sources, label
