"""Seeded synthetic frames with known background, noise and sources.

Every generator is deterministic for a given seed. Random draws come from
independent child streams of ``numpy.random.SeedSequence(seed)`` (positions,
fluxes, noise, defects), so changing one draw does not shift the others.

Conventions
-----------
* Images are float32, row-major, shape ``(height, width)``; ``x`` is the
  column and ``y`` the row, 0-based at pixel centres (SEP's convention).
  Pixel ``(j, i)`` covers ``[i-0.5, i+0.5] x [j-0.5, j+0.5]``.
* Units are ADU. ``background`` is the truth B(x,y) including any pedestal.
* ``noise_std`` is the expected per-pixel standard deviation of the noise
  process (analog, before any ADC quantization); ``noise_std_sky`` is the same
  with source photons excluded, i.e. the quantity a background RMS map
  estimates. For Gaussian scenarios they are equal.
* Sources are rendered pixel-integrated: Gaussian exactly (erf), other models
  by midpoint super-sampling. ``hfr_true_px`` is computed numerically from
  the noiseless rendered stamp of the isolated source (no background, no
  noise, no neighbours): the radius enclosing half of the stamp's flux inside
  ``hfr_aperture_px`` (the render radius), treating each pixel's flux as
  uniform over the pixel (m x m sub-pixel quadrature).

Parameters are chosen for physical plausibility (16-bit sky levels around
1000 ADU with 10 ADU noise, 14-bit saturation, gradients of tens of percent
across a frame, read noise of a few electrons) and are recorded in each
scenario's metadata. They are not tuned to favour any method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, special

from . import GENERATOR_VERSION, sha256_array

DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 384
MASK_NAMES = ("bad_pixel", "invalid", "edge", "native_saturation")
#: Fraction of model flux the Moffat render radius must enclose.
MOFFAT_ENCLOSED = 0.999


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class TruthSource:
    id: int
    x: float
    y: float
    #: Nominal model flux (ADU).
    flux: float
    #: Sum of the rendered stamp (ADU), including any part off the frame.
    flux_rendered: float
    kind: str  # "star" | "extended"
    psf: dict
    hfr_true_px: float | None
    hfr_aperture_px: float | None
    #: Continuous-profile half-flux radius where a closed form exists.
    hfr_analytic_px: float | None
    #: Brightest rendered pixel of the isolated source (ADU, no background).
    peak_adu: float
    #: Radius enclosing the rendered stamp (px).
    render_radius_px: float
    blend_group: int = -1
    saturated: bool = False
    #: Generator-specific truth (BGE v2 sets: r_out_px, vane_width_frac,
    #: n_vanes, isolated, pair_id, separation_px, flux_ratio, ...). Emitted
    #: at the top level of to_json() only when present, so pre-v2 scenarios
    #: serialise exactly as before.
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        out = self._base_json()
        out.update(self.extra)
        return out

    def _base_json(self) -> dict:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "flux": self.flux,
            "flux_rendered": self.flux_rendered,
            "kind": self.kind,
            "psf": self.psf,
            "hfr_true_px": self.hfr_true_px,
            "hfr_aperture_px": self.hfr_aperture_px,
            "hfr_analytic_px": self.hfr_analytic_px,
            "peak_adu": self.peak_adu,
            "render_radius_px": self.render_radius_px,
            "blend_group": self.blend_group,
            "saturated": self.saturated,
        }


@dataclass
class Scenario:
    id: str
    description: str
    image: np.ndarray
    background: np.ndarray | None
    noise_std: np.ndarray | None
    noise_std_sky: np.ndarray | None
    source_image: np.ndarray | None
    masks: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    #: Digitized scenarios: the analog sky sigma (secondary truth; the primary
    #: ``noise_std_sky`` is the effective std of the digitized values).
    noise_std_sky_analog: np.ndarray | None = None
    #: Digitized scenarios: expected digitized level E[step*round(X/step)].
    background_digitized: np.ndarray | None = None

    def scoring_background(self):
        """Truth B used for scoring: the analog B(x,y), or the expected
        digitized level where quantization makes the analog level
        unrecoverable (``metadata['quantization_limited_level']``)."""
        if self.metadata.get("quantization_limited_level") and self.background_digitized is not None:
            return self.background_digitized
        return self.background

    def footprint_noise(self):
        """Noise scale for source footprints and the bright-star cut: the
        analog sigma where the digitized std is degenerate, else the sky std."""
        return self.noise_std_sky_analog if self.noise_std_sky_analog is not None else self.noise_std_sky

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1]) if self.image.ndim == 2 else 0

    @property
    def saturation_adu(self):
        return self.metadata.get("saturation_adu")

    def combined_mask(self) -> np.ndarray:
        """Pixels no method may use: any detector mask or a non-finite value."""
        mask = ~np.isfinite(self.image)
        for name in MASK_NAMES:
            if name in self.masks:
                mask |= self.masks[name].astype(bool)
        return mask

    def input_hashes(self) -> dict:
        hashes = {"image": sha256_array(self.image.astype("<f4", copy=False))}
        for name, mask in sorted(self.masks.items()):
            hashes[f"mask:{name}"] = sha256_array(mask.astype(np.uint8, copy=False))
        return hashes

    def truth_summary(self) -> dict:
        stars = [s for s in self.sources if s.kind == "star"]
        return {
            "n_sources": len(self.sources),
            "n_stars": len(stars),
            "n_extended": len(self.sources) - len(stars),
            "n_saturated": sum(1 for s in self.sources if s.saturated),
            "n_multi_member_blend_groups": sum(1 for c in group_sizes(self.sources).values() if c > 1),
        }


def group_sizes(sources) -> dict:
    counts = {}
    for s in sources:
        counts[s.blend_group] = counts.get(s.blend_group, 0) + 1
    return counts


@dataclass
class FocusSweep:
    id: str
    description: str
    positions: list
    p0_true: float
    frames: list  # list[Scenario], one per position
    metadata: dict = field(default_factory=dict)


def _streams(seed: int, n: int = 5):
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]


def _base_metadata(scenario_id: str, seed: int, params: dict, **extra) -> dict:
    meta = {
        "scenario": scenario_id,
        "units": "adu",
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "rng": "numpy default_rng(SeedSequence(seed).spawn(5)) [positions, fluxes, noise, defects, spare]",
        "numpy_version": np.__version__,
        "params": params,
        "saturation_adu": None,
        "gain_e_per_adu": None,
        "read_noise_e": None,
        "quantization_step_adu": None,
        "noise_model": None,
    }
    meta.update(extra)
    return meta


# --------------------------------------------------------------------------
# Profiles and rendering
# --------------------------------------------------------------------------


def moffat_alpha(fwhm: float, beta: float) -> float:
    return fwhm / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))


def moffat_hfr_analytic(fwhm: float, beta: float) -> float:
    """Continuous half-flux radius of an untruncated Moffat profile."""
    return moffat_alpha(fwhm, beta) * math.sqrt(2.0 ** (1.0 / (beta - 1.0)) - 1.0)


GAUSSIAN_HFR_PER_SIGMA = math.sqrt(2.0 * math.log(2.0))  # 1.17741...


def _bounds(x0: float, y0: float, radius: float):
    ix0 = math.floor(x0 - radius)
    iy0 = math.floor(y0 - radius)
    nx = math.ceil(x0 + radius) - ix0 + 1
    ny = math.ceil(y0 + radius) - iy0 + 1
    return ix0, iy0, nx, ny


def _offsets(n: int) -> np.ndarray:
    return (np.arange(n) + 0.5) / n - 0.5


def _supersample(intensity, x0, y0, ix0, iy0, nx, ny, n):
    """Mean of ``intensity(dx, dy)`` over n x n midpoints of each pixel."""
    off = _offsets(n)
    xs = ((ix0 + np.arange(nx))[:, None] + off[None, :]).ravel() - x0
    ys = ((iy0 + np.arange(ny))[:, None] + off[None, :]).ravel() - y0
    values = intensity(xs[None, :], ys[:, None])
    return values.reshape(ny, n, nx, n).mean(axis=(1, 3))


def _oversample_for(width_px: float) -> int:
    """Sub-sampling so the sub-pixel step is <= width/10 (capped at 9)."""
    return int(min(9, max(1, math.ceil(10.0 / max(width_px, 1e-3)))))


@dataclass
class Stamp:
    data: np.ndarray  # (ny, nx), ADU per pixel, noiseless
    ix0: int
    iy0: int
    radius: float


def render_gaussian(x0, y0, sigma, flux) -> Stamp:
    radius = max(6.0 * sigma, 4.0)
    ix0, iy0, nx, ny = _bounds(x0, y0, radius)
    xc = ix0 + np.arange(nx)
    yc = iy0 + np.arange(ny)
    gx = special.ndtr((xc + 0.5 - x0) / sigma) - special.ndtr((xc - 0.5 - x0) / sigma)
    gy = special.ndtr((yc + 0.5 - y0) / sigma) - special.ndtr((yc - 0.5 - y0) / sigma)
    return Stamp(flux * np.outer(gy, gx), ix0, iy0, radius)


def moffat_render_radius(fwhm, beta) -> float:
    alpha = moffat_alpha(fwhm, beta)
    return alpha * math.sqrt((1.0 - MOFFAT_ENCLOSED) ** (1.0 / (1.0 - beta)) - 1.0)


def render_moffat(x0, y0, fwhm, beta, flux) -> Stamp:
    alpha = moffat_alpha(fwhm, beta)
    radius = max(moffat_render_radius(fwhm, beta), 4.0)
    ix0, iy0, nx, ny = _bounds(x0, y0, radius)
    norm = flux * (beta - 1.0) / (math.pi * alpha * alpha)

    def intensity(dx, dy):
        return norm * (1.0 + (dx * dx + dy * dy) / (alpha * alpha)) ** (-beta)

    data = _supersample(intensity, x0, y0, ix0, iy0, nx, ny, _oversample_for(fwhm))
    return Stamp(data, ix0, iy0, radius)


def render_annulus(x0, y0, r_out, r_in, sigma_seeing, flux) -> Stamp:
    """Defocused star: uniform annular pupil image blurred by Gaussian seeing.

    Geometric defocus of an obstructed aperture gives a uniform annulus of
    outer radius ``r_out`` and inner radius ``r_in``; atmospheric seeing is
    modelled as a Gaussian of ``sigma_seeing`` px. Below 0.25 px outer radius
    the pupil image is unresolved and the profile is the seeing Gaussian.
    """
    if r_out <= 0.25:
        stamp = render_gaussian(x0, y0, sigma_seeing, flux)
        return stamp
    radius = r_out + 5.0 * sigma_seeing + 2.0
    ix0, iy0, nx, ny = _bounds(x0, y0, radius)
    n = int(min(32, max(8, math.ceil(8.0 / r_out))))

    def coverage(dx, dy):
        r2 = dx * dx + dy * dy
        return ((r2 <= r_out * r_out) & (r2 >= r_in * r_in)).astype(np.float64)

    cover = _supersample(coverage, x0, y0, ix0, iy0, nx, ny, n)
    total = cover.sum()
    blurred = ndimage.gaussian_filter(cover, sigma_seeing, mode="constant", truncate=5.0)
    return Stamp(flux * blurred / total, ix0, iy0, radius)


def render_exponential_disk(x0, y0, scale_px, axis_ratio, pa_deg, central_adu) -> Stamp:
    """Elliptical exponential disk (Sersic n=1), a smooth galaxy model."""
    radius = 10.0 * scale_px
    ix0, iy0, nx, ny = _bounds(x0, y0, radius)
    c, s = math.cos(math.radians(pa_deg)), math.sin(math.radians(pa_deg))

    def intensity(dx, dy):
        u = dx * c + dy * s
        v = -dx * s + dy * c
        return central_adu * np.exp(-np.sqrt(u * u + (v / axis_ratio) ** 2) / scale_px)

    return Stamp(_supersample(intensity, x0, y0, ix0, iy0, nx, ny, 3), ix0, iy0, radius)


def render_gaussian_blob(x0, y0, sigma, amplitude_adu) -> Stamp:
    """Diffuse nebula-like emission: a broad Gaussian sampled at pixel centres."""
    radius = 5.0 * sigma
    ix0, iy0, nx, ny = _bounds(x0, y0, radius)
    xs = ix0 + np.arange(nx) - x0
    ys = iy0 + np.arange(ny) - y0
    data = amplitude_adu * np.exp(-0.5 * (ys[:, None] ** 2 + xs[None, :] ** 2) / sigma**2)
    return Stamp(data, ix0, iy0, radius)


def circle_pixel_overlap(x0, x1, y0, y1, r):
    """Exact area of the disk ``u^2+v^2 <= r^2`` inside boxes ``[x0,x1] x [y0,y1]``.

    Vectorised over arrays of boxes (``x0 <= x1``, ``y0 <= y1``). Uses the
    closed-form integral of the circle segment above a horizontal line,
    ``U(h)`` for ``h >= 0``, and splits boxes that straddle ``v = 0``.
    """
    x0, x1, y0, y1 = (np.asarray(a, dtype=np.float64) for a in (x0, x1, y0, y1))
    r = float(r)
    if r <= 0.0:
        return np.zeros(np.broadcast(x0, x1, y0, y1).shape)

    def up(h):
        # Area of the disk above v = h (h >= 0) between u = x0 and u = x1.
        s = np.sqrt(np.clip(r * r - h * h, 0.0, None))

        def g(x):
            return 0.5 * (np.sqrt(np.clip(1.0 - (x / r) ** 2, 0.0, None)) * x * r
                          + r * r * np.arcsin(np.clip(x / r, -1.0, 1.0)) - 2.0 * h * x)

        return g(np.clip(x1, -s, s)) - g(np.clip(x0, -s, s))

    above = y0 >= 0.0
    below = y1 <= 0.0
    a, b = np.abs(y0), np.abs(y1)
    return np.where(
        above,
        up(np.where(above, y0, 0.0)) - up(np.where(above, y1, 0.0)),
        np.where(
            below,
            up(np.where(below, b, 0.0)) - up(np.where(below, a, 0.0)),
            2.0 * up(np.zeros_like(a)) - up(a) - up(b),
        ),
    )


class _CurveOfGrowth:
    """Enclosed flux E(r) of a stamp, each pixel's flux uniform over the pixel.

    Pixels entirely inside radius r contribute their full flux (prefix sums
    over the pixels' farthest-corner distance); only pixels the circle cuts
    need the exact overlap area.
    """

    def __init__(self, stamp: Stamp, x0: float, y0: float):
        ny, nx = stamp.data.shape
        xl = np.broadcast_to(((stamp.ix0 + np.arange(nx)) - 0.5 - x0)[None, :], (ny, nx)).ravel()
        yl = np.broadcast_to(((stamp.iy0 + np.arange(ny)) - 0.5 - y0)[:, None], (ny, nx)).ravel()
        xh, yh = xl + 1.0, yl + 1.0
        flux = stamp.data.ravel().astype(np.float64)
        near_x = np.where((xl <= 0) & (xh >= 0), 0.0, np.minimum(np.abs(xl), np.abs(xh)))
        near_y = np.where((yl <= 0) & (yh >= 0), 0.0, np.minimum(np.abs(yl), np.abs(yh)))
        self.d_min = np.hypot(near_x, near_y)
        self.d_max = np.hypot(np.maximum(np.abs(xl), np.abs(xh)), np.maximum(np.abs(yl), np.abs(yh)))
        order = np.argsort(self.d_max, kind="stable")
        self.sorted_dmax = self.d_max[order]
        self.prefix = np.concatenate([[0.0], np.cumsum(flux[order])])
        self.xl, self.xh, self.yl, self.yh, self.flux = xl, xh, yl, yh, flux

    def __call__(self, r: float) -> float:
        full = self.prefix[np.searchsorted(self.sorted_dmax, r, side="right")]
        cut = (self.d_min < r) & (self.d_max > r)
        if not cut.any():
            return float(full)
        area = circle_pixel_overlap(self.xl[cut], self.xh[cut], self.yl[cut], self.yh[cut], r)
        return float(full + np.sum(self.flux[cut] * area))


def enclosed_flux(stamp: Stamp, x0: float, y0: float, r: float) -> float:
    """Flux inside radius ``r`` with each pixel's flux uniform over the pixel."""
    return _CurveOfGrowth(stamp, x0, y0)(r)


def numerical_hfr(stamp: Stamp, x0: float, y0: float, r_ap: float | None = None):
    """Half-flux radius of a noiseless stamp inside ``r_ap`` (default: render radius).

    Each pixel's flux is spread uniformly over the pixel (the assumption
    aperture photometry makes); the enclosed flux uses exact circle-pixel
    overlap areas and the radius is found by Brent's method to 1e-6 px.
    Returns ``(hfr, r_ap)``.
    """
    from scipy.optimize import brentq

    r_ap = stamp.radius if r_ap is None else r_ap
    cog = _CurveOfGrowth(stamp, x0, y0)
    target = 0.5 * cog(r_ap)
    if not target > 0.0:
        return None, r_ap
    hfr = brentq(lambda r: cog(r) - target, 1e-6, r_ap, xtol=1e-6)
    return float(hfr), r_ap


def _paste(target: np.ndarray, stamp: Stamp) -> None:
    h, w = target.shape
    ny, nx = stamp.data.shape
    x_lo, y_lo = max(0, stamp.ix0), max(0, stamp.iy0)
    x_hi, y_hi = min(w, stamp.ix0 + nx), min(h, stamp.iy0 + ny)
    if x_lo >= x_hi or y_lo >= y_hi:
        return
    target[y_lo:y_hi, x_lo:x_hi] += stamp.data[
        y_lo - stamp.iy0 : y_hi - stamp.iy0, x_lo - stamp.ix0 : x_hi - stamp.ix0
    ]


def pupil_shadow_coverage(x0, y0, r_out, obstruction, sigma_seeing, n_vanes=0, vane_width_frac=0.0,
                          vane_angle_rad=0.0, modulation=0.0, modulation_angle_rad=0.0, hole_offset_px=(0.0, 0.0)):
    """Pre-seeing pupil shadow of :func:`render_pupil_shadow`: per-pixel
    covered area (px^2, times the brightness modulation) on the stamp grid,
    as ``(coverage, ix0, iy0, radius)``."""
    radius = r_out + 5.0 * sigma_seeing + 2.0
    ix0, iy0, nx, ny = _bounds(x0, y0, radius)
    n = int(min(32, max(8, math.ceil(8.0 / max(r_out, 1e-3)))))
    r_in = obstruction * r_out
    half_width = 0.5 * vane_width_frac * 2.0 * r_out
    sub = 1.0 / n
    hx, hy = hole_offset_px

    def intensity(dx, dy):
        r2 = dx * dx + dy * dy
        cover = ((r2 <= r_out * r_out) & ((dx - hx) ** 2 + (dy - hy) ** 2 >= r_in * r_in)).astype(np.float64)
        for k in range(int(n_vanes)):
            theta = vane_angle_rad + 2.0 * math.pi * k / n_vanes
            u = dx * math.cos(theta) + dy * math.sin(theta)
            v = -dx * math.sin(theta) + dy * math.cos(theta)
            frac = np.clip((np.minimum(v + 0.5 * sub, half_width) - np.maximum(v - 0.5 * sub, -half_width)) / sub,
                           0.0, 1.0)
            cover = cover * (1.0 - np.where(u > 0.0, frac, 0.0))
        if modulation:
            cover = cover * (1.0 + modulation * np.cos(np.arctan2(dy, dx) - modulation_angle_rad))
        return cover

    return _supersample(intensity, x0, y0, ix0, iy0, nx, ny, n), ix0, iy0, radius


def render_pupil_shadow(x0, y0, r_out, obstruction, sigma_seeing, flux, n_vanes=0, vane_width_frac=0.0,
                        vane_angle_rad=0.0, modulation=0.0, modulation_angle_rad=0.0, hole_offset_px=(0.0, 0.0)):
    """Geometric-defocus star image: the shadow of the pupil, blurred by seeing.

    The pupil image is a disk of radius ``r_out`` minus the obstruction disk
    (radius ``obstruction * r_out``, centred at ``hole_offset_px`` from the
    disk centre) minus ``n_vanes`` radial vane strips of width
    ``w_img = vane_width_frac * 2 * r_out`` (the vane width as a fraction of
    the aperture D, scaled to the pupil image of diameter ``2 r_out``) from
    the centre to the rim, at angles ``vane_angle_rad + 2 pi k / n_vanes``.
    ``modulation`` multiplies the brightness by
    ``1 + modulation * cos(phi - modulation_angle_rad)`` (coma-like azimuthal
    asymmetry). The shadow is midpoint super-sampled (vane strips
    anti-aliased by their exact 1-D overlap with each sub-cell), normalised
    to ``flux`` and convolved with a Gaussian of ``sigma_seeing`` px.
    """
    cover, ix0, iy0, radius = pupil_shadow_coverage(
        x0, y0, r_out, obstruction, sigma_seeing, n_vanes, vane_width_frac, vane_angle_rad, modulation,
        modulation_angle_rad, hole_offset_px)
    total = cover.sum()
    blurred = ndimage.gaussian_filter(cover, sigma_seeing, mode="constant", truncate=5.0)
    return Stamp(flux * blurred / total, ix0, iy0, radius)


def render_psf(psf: dict, x0: float, y0: float, flux: float) -> Stamp:
    model = psf["model"]
    if model == "gaussian":
        return render_gaussian(x0, y0, psf["sigma_px"], flux)
    if model == "moffat":
        return render_moffat(x0, y0, psf["fwhm_px"], psf["beta"], flux)
    if model == "annulus":
        return render_annulus(x0, y0, psf["r_out_px"], psf["r_in_px"], psf["sigma_seeing_px"], flux)
    if model == "pupil_shadow":
        return render_pupil_shadow(
            x0, y0, psf["r_out_px"], psf["obstruction"], psf["sigma_seeing_px"], flux,
            n_vanes=psf.get("n_vanes", 0), vane_width_frac=psf.get("vane_width_frac", 0.0),
            vane_angle_rad=psf.get("vane_angle_rad", 0.0), modulation=psf.get("modulation", 0.0),
            modulation_angle_rad=psf.get("modulation_angle_rad", 0.0),
            hole_offset_px=tuple(psf.get("hole_offset_px", (0.0, 0.0))))
    raise ValueError(f"unknown psf model {model!r}")


def analytic_hfr(psf: dict):
    if psf["model"] == "gaussian":
        return GAUSSIAN_HFR_PER_SIGMA * psf["sigma_px"]
    if psf["model"] == "moffat":
        return moffat_hfr_analytic(psf["fwhm_px"], psf["beta"])
    return None


def stamp_centroid(stamp: Stamp):
    """Flux centroid of a noiseless stamp (pixel-centre coordinates)."""
    ny, nx = stamp.data.shape
    total = float(stamp.data.sum())
    xs = stamp.ix0 + np.arange(nx)
    ys = stamp.iy0 + np.arange(ny)
    return float((stamp.data.sum(axis=0) * xs).sum() / total), float((stamp.data.sum(axis=1) * ys).sum() / total)


def add_stars(shape, stars, start_id=0):
    """Render ``stars`` = [(x, y, flux, psf)] into a noiseless source image.

    ``pupil_shadow`` stars take their truth HFR about the stamp's flux
    centroid (the aperture truth a centroiding measurement can reach for an
    asymmetric, e.g. comatic, donut); every other model about ``(x, y)``.
    """
    image = np.zeros(shape, dtype=np.float64)
    sources = []
    for k, (x, y, flux, psf) in enumerate(stars):
        stamp = render_psf(psf, x, y, flux)
        if psf["model"] == "pupil_shadow":
            hfr, r_ap = numerical_hfr(stamp, *stamp_centroid(stamp))
        else:
            hfr, r_ap = numerical_hfr(stamp, x, y)
        _paste(image, stamp)
        sources.append(
            TruthSource(
                id=start_id + k,
                x=float(x),
                y=float(y),
                flux=float(flux),
                flux_rendered=float(stamp.data.sum()),
                kind="star",
                psf=dict(psf),
                hfr_true_px=hfr,
                hfr_aperture_px=float(r_ap),
                hfr_analytic_px=analytic_hfr(psf),
                peak_adu=float(stamp.data.max()),
                render_radius_px=float(stamp.radius),
            )
        )
    return image, sources


def assign_blend_groups(sources, link_px: float) -> None:
    """Single-linkage groups of sources closer than ``link_px``."""
    n = len(sources)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n:
        xy = np.array([[s.x, s.y] for s in sources])
        for i in range(n):
            d = np.hypot(xy[i + 1 :, 0] - xy[i, 0], xy[i + 1 :, 1] - xy[i, 1])
            for j in np.nonzero(d < link_px)[0]:
                a, b = find(i), find(i + 1 + int(j))
                if a != b:
                    parent[b] = a
    roots = {}
    for i, s in enumerate(sources):
        s.blend_group = roots.setdefault(find(i), len(roots))


def place_uniform(rng, n, width, height, margin, min_sep=0.0, existing=(), max_tries=200_000):
    pts = list(existing)
    out = []
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(f"could not place {n} points with min_sep={min_sep}")
        x = rng.uniform(margin, width - 1 - margin)
        y = rng.uniform(margin, height - 1 - margin)
        if min_sep > 0 and pts:
            arr = np.asarray(pts)
            if np.min(np.hypot(arr[:, 0] - x, arr[:, 1] - y)) < min_sep:
                continue
        pts.append((x, y))
        out.append((x, y))
    return out


def log_uniform(rng, lo, hi, n):
    return np.exp(rng.uniform(math.log(lo), math.log(hi), n))


def _grid(width, height):
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    return xx, yy


# --------------------------------------------------------------------------
# Noise and detector models
# --------------------------------------------------------------------------


def gaussian_noise(rng, sigma_map):
    return rng.standard_normal(sigma_map.shape) * sigma_map


def poisson_read(rng, pedestal_adu, sky_adu, source_adu, gain_e_per_adu, read_noise_e):
    """Photon (Poisson) and read (Gaussian) noise, returned in ADU.

    ``sky_adu`` and ``source_adu`` exclude the pedestal. Returns the noisy
    frame, the expected total std and the expected sky-only std (ADU).
    """
    lam_sky = np.clip(sky_adu * gain_e_per_adu, 0.0, None)
    lam_src = np.clip(source_adu * gain_e_per_adu, 0.0, None)
    electrons = rng.poisson(lam_sky + lam_src).astype(np.float64)
    electrons += rng.standard_normal(electrons.shape) * read_noise_e
    image = pedestal_adu + electrons / gain_e_per_adu
    std = np.sqrt(lam_sky + lam_src + read_noise_e**2) / gain_e_per_adu
    std_sky = np.sqrt(lam_sky + read_noise_e**2) / gain_e_per_adu
    return image, std, std_sky


def _finish(
    scenario_id,
    description,
    image,
    background,
    noise_std,
    noise_std_sky,
    source_image,
    sources,
    metadata,
    masks=None,
    noise_std_sky_analog=None,
    background_digitized=None,
):
    sat = metadata.get("saturation_adu")
    if sat is not None and source_image is not None and background is not None:
        noiseless = background + source_image
        for s in sources:
            r = max(2.0, 2.0 * (s.hfr_true_px or 2.0))
            x_lo, x_hi = max(0, int(s.x - r)), min(image.shape[1], int(s.x + r) + 2)
            y_lo, y_hi = max(0, int(s.y - r)), min(image.shape[0], int(s.y + r) + 2)
            if x_lo < x_hi and y_lo < y_hi:
                s.saturated = bool(noiseless[y_lo:y_hi, x_lo:x_hi].max() >= sat)
    as32 = lambda a: None if a is None else np.ascontiguousarray(a, dtype=np.float32)  # noqa: E731
    return Scenario(
        id=scenario_id,
        description=description,
        image=as32(image),
        background=as32(background),
        noise_std=as32(noise_std),
        noise_std_sky=as32(noise_std_sky),
        source_image=as32(source_image),
        masks={k: np.ascontiguousarray(v, dtype=np.uint8) for k, v in (masks or {}).items()},
        sources=sources,
        metadata=metadata,
        noise_std_sky_analog=as32(noise_std_sky_analog),
        background_digitized=as32(background_digitized),
    )


def _star_list(rng_pos, rng_flux, n, width, height, flux_lo, flux_hi, psf, margin=16, min_sep=20.0):
    xy = place_uniform(rng_pos, n, width, height, margin, min_sep)
    fluxes = log_uniform(rng_flux, flux_lo, flux_hi, n)
    return [(x, y, float(f), psf) for (x, y), f in zip(xy, fluxes)]


def _gaussian_scene(
    scenario_id,
    description,
    seed,
    background,
    sigma_map,
    stars,
    params,
    link_px=10.0,
    **meta,
):
    rngs = _streams(seed)
    shape = background.shape
    src, sources = add_stars(shape, stars)
    assign_blend_groups(sources, link_px)
    image = background + src + gaussian_noise(rngs[2], sigma_map)
    metadata = _base_metadata(scenario_id, seed, params, noise_model="gaussian", **meta)
    return _finish(scenario_id, description, image, background, sigma_map, sigma_map, src, sources, metadata)


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------

SKY_ADU = 1000.0
SKY_SIGMA_ADU = 10.0
GAUSS_PSF = {"model": "gaussian", "sigma_px": 2.0}
MOFFAT_PSF = {"model": "moffat", "fwhm_px": 5.0, "beta": 3.0}


def uniform_gaussian_sky(seed=101, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    params = {"sky_adu": SKY_ADU, "sigma_adu": SKY_SIGMA_ADU, "n_stars": 0}
    bg = np.full((height, width), SKY_ADU)
    sig = np.full((height, width), SKY_SIGMA_ADU)
    return _gaussian_scene(
        "uniform_gaussian_sky",
        "Constant sky 1000 ADU, Gaussian noise 10 ADU, no sources (every detection is false).",
        seed, bg, sig, [], params,
    )


def poisson_read_low_counts(seed=102, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    gain, rn, pedestal, sky_e = 1.0, 1.8, 100.0, 4.0
    psf = {"model": "gaussian", "sigma_px": 1.8}
    params = {"pedestal_adu": pedestal, "sky_e_per_px": sky_e, "gain_e_per_adu": gain,
              "read_noise_e": rn, "n_stars": 25, "flux_e": [200.0, 20000.0], "psf": psf,
              "adc_quantization": "omitted (see quantization scenarios)"}
    rngs = _streams(seed)
    stars = _star_list(rngs[0], rngs[1], 25, width, height, 200.0, 20000.0, psf)
    src, sources = add_stars((height, width), stars)
    assign_blend_groups(sources, 10.0)
    sky = np.full((height, width), sky_e / gain)
    image, std, std_sky = poisson_read(rngs[2], pedestal, sky, src, gain, rn)
    meta = _base_metadata("poisson_read_low_counts", seed, params, noise_model="poisson+read",
                          gain_e_per_adu=gain, read_noise_e=rn)
    return _finish("poisson_read_low_counts",
                   "Short exposure: pedestal 100 ADU + 4 e-/px sky, Poisson photon noise and 1.8 e- read noise, gain 1 e-/ADU; 25 faint stars.",
                   image, pedestal + sky, std, std_sky, src, sources, meta)


def _quantized(scenario_id, description, seed, level, analog_sigma, step, n_stars, flux_range, width, height):
    psf = {"model": "gaussian", "sigma_px": 1.8}
    params = {"background_adu": level, "analog_sigma_adu": analog_sigma, "quantization_step_adu": step,
              "n_stars": n_stars, "flux_adu": list(flux_range), "psf": psf}
    rngs = _streams(seed)
    stars = _star_list(rngs[0], rngs[1], n_stars, width, height, *flux_range, psf)
    src, sources = add_stars((height, width), stars)
    assign_blend_groups(sources, 10.0)
    bg = np.full((height, width), level)
    sig = np.full((height, width), analog_sigma)
    analog = bg + src + gaussian_noise(rngs[2], sig)
    image = step * np.round(analog / step)
    # Truth for an estimator of the digitized frame: the exact mean and std of
    # step*round(X/step), X ~ N(mu, sigma), per pixel (sources included in
    # noise_std; sky only in noise_std_sky). The analog sigma stays secondary.
    level_dig, sky_std = digitized_moments(np.array(level), analog_sigma, step)
    _, total_std = digitized_moments(bg + src, analog_sigma, step)
    limited = bool(abs(float(level_dig) - level) > QUANT_LEVEL_LIMIT_STEPS * step)
    scoreable = bool(float(sky_std) >= QUANT_NOISE_MIN_STEPS * step)
    meta = _base_metadata(scenario_id, seed, params, noise_model="gaussian+adc_rounding",
                          quantization_step_adu=step)
    meta.update({
        "noise_truth": "effective digitized std: exact std of step*round(X/step), X ~ N(B, analog sigma)",
        "analog_sigma_adu": analog_sigma,
        "effective_sky_std_adu": float(sky_std),
        "sheppard_sky_std_adu": math.sqrt(analog_sigma**2 + step**2 / 12.0),
        "expected_digitized_level_adu": float(level_dig),
        "quantization_limited_level": limited,
        "noise_scoreable": scoreable,
        "quantization_rules": f"level quantization-limited if |E_dig - B| > {QUANT_LEVEL_LIMIT_STEPS} step; "
                              f"noise not scoreable if effective std < {QUANT_NOISE_MIN_STEPS} step",
    })
    return _finish(scenario_id, description, image, bg, total_std, np.full_like(bg, float(sky_std)), src, sources,
                   meta, noise_std_sky_analog=sig, background_digitized=np.full_like(bg, float(level_dig)))


#: The analog level is "quantization-limited" when the expected digitized
#: level differs from it by more than this fraction of a step.
QUANT_LEVEL_LIMIT_STEPS = 0.01
#: An effective digitized noise below this fraction of a step is not a
#: scoreable noise truth (the ADC has swallowed the noise).
QUANT_NOISE_MIN_STEPS = 0.01


def digitized_moments(mu, sigma: float, step: float):
    """Exact mean and std of ``step*round(X/step)`` for ``X ~ N(mu, sigma)``.

    Vectorised over ``mu``; sums the bin probabilities within 8 sigma (the
    neglected tail mass is < 1e-15). Ties of ``round`` have probability 0.
    """
    mu = np.asarray(mu, dtype=np.float64)
    if sigma <= 0:
        mean = step * np.round(mu / step)
        return mean, np.zeros_like(mu)
    half = int(math.ceil(8.0 * sigma / step)) + 1
    k0 = np.round(mu / step)
    offsets = np.arange(-half, half + 1).reshape((-1,) + (1,) * mu.ndim)
    k = k0[None, ...] + offsets
    p = special.ndtr(((k + 0.5) * step - mu) / sigma) - special.ndtr(((k - 0.5) * step - mu) / sigma)
    values = k * step
    mean = np.sum(p * values, axis=0) / np.sum(p, axis=0)
    var = np.sum(p * (values - mean) ** 2, axis=0) / np.sum(p, axis=0)
    return mean, np.sqrt(var)


def quantization_coarse(seed=103, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    return _quantized(
        "quantization_coarse",
        "Background 200.3 ADU with 0.5 ADU analog noise rounded to 1 ADU steps (noise below one ADU step); 25 stars.",
        seed, 200.3, 0.5, 1.0, 25, (50.0, 5000.0), width, height)


def quantization_plateau(seed=104, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    return _quantized(
        "quantization_plateau",
        "Zero-scatter plateau: 200.2 ADU with 0.05 ADU analog noise rounds to exactly 200 everywhere off-source; 10 stars.",
        seed, 200.2, 0.05, 1.0, 10, (2000.0, 20000.0), width, height)


def gradient_linear_xy(seed=105, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    gx, gy = 0.4, 0.25
    xx, yy = _grid(width, height)
    bg = SKY_ADU + gx * (xx - (width - 1) / 2) + gy * (yy - (height - 1) / 2)
    rngs = _streams(seed)
    stars = _star_list(rngs[0], rngs[1], 40, width, height, 2e3, 2e5, GAUSS_PSF)
    params = {"b0_adu": SKY_ADU, "gx_adu_per_px": gx, "gy_adu_per_px": gy, "sigma_adu": SKY_SIGMA_ADU,
              "n_stars": 40, "flux_adu": [2e3, 2e5], "psf": GAUSS_PSF}
    return _gaussian_scene(
        "gradient_linear_xy",
        "Linear sky gradient 0.4 ADU/px in x and 0.25 ADU/px in y about 1000 ADU (about 20% across the frame); 40 Gaussian stars.",
        seed, bg, np.full_like(bg, SKY_SIGMA_ADU), stars, params)


def vignetting_background(width, height, b0=SKY_ADU, c2=0.25, c4=0.05):
    xx, yy = _grid(width, height)
    xc, yc = (width - 1) / 2, (height - 1) / 2
    rho2 = ((xx - xc) ** 2 + (yy - yc) ** 2) / (xc**2 + yc**2)
    return b0 * (1.0 - c2 * rho2 - c4 * rho2**2)


def vignetting_radial(seed=106, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    bg = vignetting_background(width, height)
    rngs = _streams(seed)
    stars = _star_list(rngs[0], rngs[1], 40, width, height, 2e3, 2e5, GAUSS_PSF)
    params = {"model": "B0*(1-0.25 rho^2-0.05 rho^4), rho=r/r_corner", "b0_adu": SKY_ADU,
              "sigma_adu": SKY_SIGMA_ADU, "n_stars": 40, "flux_adu": [2e3, 2e5], "psf": GAUSS_PSF}
    return _gaussian_scene(
        "vignetting_radial",
        "Curved radial vignetting: sky falls 30% from centre to corner (quadratic + quartic); constant 10 ADU noise; 40 stars.",
        seed, bg, np.full_like(bg, SKY_SIGMA_ADU), stars, params)


def amp_glow_corner(seed=107, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    gain, rn, pedestal, sky_e, glow_e, glow_scale = 1.0, 3.0, 500.0, 300.0, 600.0, 70.0
    xx, yy = _grid(width, height)
    d = np.hypot(xx - (width - 1), yy - (height - 1))
    sky = np.full((height, width), sky_e) + glow_e * np.exp(-d / glow_scale)
    rngs = _streams(seed)
    stars = _star_list(rngs[0], rngs[1], 40, width, height, 2e3, 2e5, GAUSS_PSF)
    src, sources = add_stars((height, width), stars)
    assign_blend_groups(sources, 10.0)
    image, std, std_sky = poisson_read(rngs[2], pedestal, sky / gain, src, gain, rn)
    params = {"pedestal_adu": pedestal, "sky_e": sky_e, "glow_e": glow_e, "glow_scale_px": glow_scale,
              "glow_corner": "bottom-right (x=W-1, y=H-1)", "gain_e_per_adu": gain, "read_noise_e": rn,
              "n_stars": 40, "flux_adu": [2e3, 2e5], "psf": GAUSS_PSF}
    meta = _base_metadata("amp_glow_corner", seed, params, noise_model="poisson+read",
                          gain_e_per_adu=gain, read_noise_e=rn)
    return _finish("amp_glow_corner",
                   "Amplifier-glow-like corner: +600 e- exponential (70 px scale) on 300 e- sky, Poisson+read noise; 40 stars.",
                   image, pedestal + sky / gain, std, std_sky, src, sources, meta)


def steep_ridge_background(seed=120, width=512, height=512):
    """W1's reproduction of a negative SEP RMS spline (coordinator contract note).

    A vertical ridge 800 ADU high with a 60 px 1/e half-width on 1000 ADU,
    Gaussian noise 5 ADU, no sources. Per-mesh sigmas jump sharply between
    neighbouring 64 px meshes; the bicubic RMS spline can undershoot below 0.
    """
    xx, _ = _grid(width, height)
    bg = 1000.0 + 800.0 * np.exp(-(((xx - 256.0) / 60.0) ** 2))
    params = {"model": "1000 + 800 exp(-((x-256)/60)^2) ADU", "sigma_adu": 5.0, "n_stars": 0,
              "provenance": "W1 reproduction relayed by the coordinator (min R -0.946, globalrms 37.6 at bw=64)"}
    return _gaussian_scene(
        "steep_ridge_background",
        "Steep vertical sky ridge (+800 ADU, 60 px) on 1000 ADU, 5 ADU noise, 512x512, no sources.",
        seed, bg, np.full_like(bg, 5.0), [], params)


def steep_ridge_amp600(seed=122, width=512, height=512):
    """D-018 fixture: a 600 ADU ridge whose RMS spline undershoots while
    staying positive (review B: min R 0.549 against a 4.865 minimum mesh
    sigma, ~100 false detections on this starless frame). An honest reference
    reports ``invalid_rms_map`` and runs no extraction."""
    xx, _ = _grid(width, height)
    bg = 1000.0 + 600.0 * np.exp(-(((xx - 256.0) / 60.0) ** 2))
    params = {"model": "1000 + 600 exp(-((x-256)/60)^2) ADU", "sigma_adu": 5.0, "n_stars": 0,
              "provenance": "review B1 / decision D-018 (positive RMS-spline undershoot)",
              "designed_outcome": "python_sep and rust_sep: noise unresolved, invalid_rms_map, no extraction"}
    return _gaussian_scene(
        "steep_ridge_amp600",
        "Steep vertical sky ridge (+600 ADU, 60 px) on 1000 ADU, 5 ADU noise, 512x512, no sources "
        "(positive RMS-spline undershoot).",
        seed, bg, np.full_like(bg, 5.0), [], params)


def curved_bowl(seed=121, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """Strongly curved background 1000 + 0.02 r^2 ADU (r from the frame centre).

    A stress case (about +2000 ADU at the corners, e.g. a light leak or a bad
    flat) for the mesh median filter plus spline (W1 reports tens of ADU of
    bias at some stars).
    """
    xx, yy = _grid(width, height)
    r2 = (xx - (width - 1) / 2) ** 2 + (yy - (height - 1) / 2) ** 2
    bg = 1000.0 + 0.02 * r2
    rngs = _streams(seed)
    stars = _star_list(rngs[0], rngs[1], 40, width, height, 2e3, 2e5, GAUSS_PSF)
    params = {"model": "1000 + 0.02 r^2 ADU, r from frame centre", "sigma_adu": SKY_SIGMA_ADU, "n_stars": 40,
              "flux_adu": [2e3, 2e5], "psf": GAUSS_PSF, "class": "stress"}
    return _gaussian_scene(
        "curved_bowl",
        "Stress: strongly curved bowl 1000 + 0.02 r^2 ADU (about +2000 ADU at the corners); 40 stars.",
        seed, bg, np.full_like(bg, SKY_SIGMA_ADU), stars, params)


def heteroscedastic_noise(seed=108, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    xx, yy = _grid(width, height)
    sig = 6.0 + 12.0 * xx / (width - 1) + 4.0 * np.sin(math.pi * yy / (height - 1))
    bg = np.full((height, width), SKY_ADU)
    rngs = _streams(seed)
    stars = _star_list(rngs[0], rngs[1], 40, width, height, 2e3, 2e5, GAUSS_PSF)
    params = {"sky_adu": SKY_ADU, "sigma_model": "6 + 12 x/(W-1) + 4 sin(pi y/(H-1)) ADU",
              "n_stars": 40, "flux_adu": [2e3, 2e5], "psf": GAUSS_PSF}
    return _gaussian_scene(
        "heteroscedastic_noise",
        "Flat sky with spatially varying noise R(x,y) from 6 to 22 ADU; 40 stars.",
        seed, bg, sig, stars, params)


def sparse_gaussian_stars(seed=109, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    bg = np.full((height, width), SKY_ADU)
    rngs = _streams(seed)
    stars = _star_list(rngs[0], rngs[1], 40, width, height, 2e3, 2e5, GAUSS_PSF)
    params = {"sky_adu": SKY_ADU, "sigma_adu": SKY_SIGMA_ADU, "n_stars": 40, "flux_adu": [2e3, 2e5],
              "psf": GAUSS_PSF, "min_separation_px": 20}
    return _gaussian_scene(
        "sparse_gaussian_stars",
        "40 isolated Gaussian stars (sigma 2 px, FWHM 4.7 px), flux 2e3-2e5 ADU, flat sky.",
        seed, bg, np.full_like(bg, SKY_SIGMA_ADU), stars, params)


def moffat_stars(seed=110, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    bg = np.full((height, width), SKY_ADU)
    rngs = _streams(seed)
    stars = _star_list(rngs[0], rngs[1], 40, width, height, 2e3, 2e5, MOFFAT_PSF)
    params = {"sky_adu": SKY_ADU, "sigma_adu": SKY_SIGMA_ADU, "n_stars": 40, "flux_adu": [2e3, 2e5],
              "psf": MOFFAT_PSF, "min_separation_px": 20}
    return _gaussian_scene(
        "moffat_stars",
        "40 isolated Moffat stars (FWHM 5 px, beta 3; extended wings), flux 2e3-2e5 ADU, flat sky.",
        seed, bg, np.full_like(bg, SKY_SIGMA_ADU), stars, params)


def defocused_donut_stars(seed=111, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    psf = {"model": "annulus", "r_out_px": 14.0, "r_in_px": 5.0, "sigma_seeing_px": 1.2}
    bg = np.full((height, width), SKY_ADU)
    rngs = _streams(seed)
    stars = _star_list(rngs[0], rngs[1], 25, width, height, 5e4, 1e6, psf, margin=24, min_sep=50.0)
    params = {"sky_adu": SKY_ADU, "sigma_adu": SKY_SIGMA_ADU, "n_stars": 25, "flux_adu": [5e4, 1e6],
              "psf": psf, "min_separation_px": 50}
    return _gaussian_scene(
        "defocused_donut_stars",
        "25 strongly defocused stars: annulus r_out 14 px, r_in 5 px (36% obstruction), 1.2 px seeing; flux 5e4-1e6 ADU.",
        seed, bg, np.full_like(bg, SKY_SIGMA_ADU), stars, params, link_px=30.0)


def crowded_field(seed=112, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    psf = {"model": "moffat", "fwhm_px": 4.0, "beta": 3.0}
    rngs = _streams(seed)
    singles = place_uniform(rngs[0], 300, width, height, margin=8)
    stars = [(x, y, float(f), psf) for (x, y), f in zip(singles, log_uniform(rngs[1], 1e3, 1e5, 300))]
    for _ in range(20):
        (x, y), = place_uniform(rngs[0], 1, width, height, margin=16)
        sep_px = rngs[0].uniform(3.0, 8.0)
        ang = rngs[0].uniform(0.0, 2 * math.pi)
        f1 = float(log_uniform(rngs[1], 5e3, 1e5, 1)[0])
        ratio = rngs[1].uniform(0.2, 1.0)
        stars.append((x, y, f1, psf))
        stars.append((x + sep_px * math.cos(ang), y + sep_px * math.sin(ang), f1 * ratio, psf))
    bg = np.full((height, width), SKY_ADU)
    params = {"sky_adu": SKY_ADU, "sigma_adu": SKY_SIGMA_ADU, "n_single": 300, "n_close_doubles": 20,
              "double_separation_px": [3.0, 8.0], "double_flux_ratio": [0.2, 1.0], "flux_adu": [1e3, 1e5],
              "psf": psf, "blend_link_px": 8.0}
    return _gaussian_scene(
        "crowded_field",
        "Crowded field: 300 random Moffat stars (FWHM 4 px) plus 20 close doubles at 3-8 px separation; blend groups link at 8 px (2 FWHM).",
        seed, bg, np.full_like(bg, SKY_SIGMA_ADU), stars, params, link_px=8.0)


def extended_emission(seed=113, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    rngs = _streams(seed)
    bg = np.full((height, width), SKY_ADU)
    stars = _star_list(rngs[0], rngs[1], 30, width, height, 2e3, 2e5, GAUSS_PSF)
    src, sources = add_stars((height, width), stars)
    galaxy = render_exponential_disk(330.0, 190.0, 25.0, 0.6, 30.0, 400.0)
    nebula = render_gaussian_blob(120.0, 260.0, 60.0, 120.0)
    for k, (stamp, name, x, y, psf) in enumerate(
        [(galaxy, "galaxy", 330.0, 190.0, {"model": "exponential_disk", "scale_px": 25.0, "axis_ratio": 0.6,
                                           "pa_deg": 30.0, "central_adu": 400.0}),
         (nebula, "nebula", 120.0, 260.0, {"model": "gaussian_blob", "sigma_px": 60.0, "amplitude_adu": 120.0})]):
        _paste(src, stamp)
        sources.append(TruthSource(id=len(sources), x=x, y=y, flux=float(stamp.data.sum()),
                                   flux_rendered=float(stamp.data.sum()), kind="extended",
                                   psf=dict(psf, name=name), hfr_true_px=None, hfr_aperture_px=None,
                                   hfr_analytic_px=None, peak_adu=float(stamp.data.max()),
                                   render_radius_px=float(stamp.radius)))
    assign_blend_groups(sources, 10.0)
    image = bg + src + gaussian_noise(rngs[2], np.full_like(bg, SKY_SIGMA_ADU))
    params = {"sky_adu": SKY_ADU, "sigma_adu": SKY_SIGMA_ADU, "n_stars": 30, "psf": GAUSS_PSF,
              "galaxy": "exponential disk h=25 px, q=0.6, PA 30 deg, I0=400 ADU at (330,190)",
              "nebula": "Gaussian blob sigma=60 px, 120 ADU at (120,260)",
              "truth_convention": "galaxy and nebula are sources, not background"}
    meta = _base_metadata("extended_emission", seed, params, noise_model="gaussian")
    sig = np.full_like(bg, SKY_SIGMA_ADU)
    return _finish("extended_emission",
                   "Flat sky with a bright exponential-disk galaxy and a broad diffuse nebula (declared sources in the truth); 30 stars.",
                   image, bg, sig, sig, src, sources, meta)


def native_clipping(seed=114, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    sat = 16383.0
    rngs = _streams(seed)
    bg = np.full((height, width), SKY_ADU)
    sig = np.full_like(bg, SKY_SIGMA_ADU)
    stars = _star_list(rngs[0], rngs[1], 40, width, height, 5e4, 2e6, GAUSS_PSF)
    src, sources = add_stars((height, width), stars)
    assign_blend_groups(sources, 10.0)
    analog = bg + src + gaussian_noise(rngs[2], sig)
    clipped = analog >= sat
    image = np.minimum(analog, sat)
    params = {"sky_adu": SKY_ADU, "sigma_adu": SKY_SIGMA_ADU, "saturation_adu": sat, "n_stars": 40,
              "flux_adu": [5e4, 2e6], "psf": GAUSS_PSF}
    meta = _base_metadata("native_clipping", seed, params, noise_model="gaussian", saturation_adu=sat)
    return _finish("native_clipping",
                   "14-bit clipping at 16383 ADU: bright Gaussian stars with clipped cores flagged in native_saturation.",
                   image, bg, sig, sig, src, sources, meta, masks={"native_saturation": clipped})


def bad_pixels_invalid(seed=115, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    rngs = _streams(seed)
    bg = np.full((height, width), SKY_ADU)
    sig = np.full_like(bg, SKY_SIGMA_ADU)
    stars = _star_list(rngs[0], rngs[1], 40, width, height, 2e3, 2e5, GAUSS_PSF)
    src, sources = add_stars((height, width), stars)
    assign_blend_groups(sources, 10.0)
    image = bg + src + gaussian_noise(rngs[2], sig)
    d = rngs[3]
    bad = np.zeros((height, width), bool)
    invalid = np.zeros((height, width), bool)
    edge = np.zeros((height, width), bool)
    for _ in range(30):  # hot clusters
        x, y = int(d.integers(5, width - 8)), int(d.integers(5, height - 8))
        w, h = int(d.integers(1, 4)), int(d.integers(1, 4))
        image[y:y + h, x:x + w] += d.uniform(3000.0, 20000.0)
        bad[y:y + h, x:x + w] = True
    for _ in range(5):  # dead clusters
        x, y = int(d.integers(5, width - 8)), int(d.integers(5, height - 8))
        image[y:y + 2, x:x + 2] = 0.0
        bad[y:y + 2, x:x + 2] = True
    for _ in range(2):  # partial hot columns
        x, y = int(d.integers(10, width - 10)), int(d.integers(10, height - 80))
        image[y:y + 60, x] += d.uniform(300.0, 800.0)
        bad[y:y + 60, x] = True
    flat = d.choice(width * height, 260, replace=False)
    image.flat[flat[:200]] = np.nan
    image.flat[flat[200:250]] = np.inf
    image.flat[flat[250:]] = -np.inf
    invalid.flat[flat] = True
    bx, by = int(d.integers(20, width - 40)), int(d.integers(20, height - 40))
    image[by:by + 10, bx:bx + 10] = np.nan
    invalid[by:by + 10, bx:bx + 10] = True
    border = 4
    edge[:border, :] = edge[-border:, :] = edge[:, :border] = edge[:, -border:] = True
    image[edge] = 100.0
    params = {"sky_adu": SKY_ADU, "sigma_adu": SKY_SIGMA_ADU, "n_stars": 40, "psf": GAUSS_PSF,
              "hot_clusters": "30 x (1-3)x(1-3) px, +3000..20000 ADU", "dead_clusters": "5 x 2x2 px at 0 ADU",
              "hot_columns": "2 x 60 px, +300..800 ADU", "nonfinite": "200 NaN, 50 +Inf, 10 -Inf, one 10x10 NaN block",
              "edge": "4 px border at 100 ADU (non-imaging reference pixels)"}
    meta = _base_metadata("bad_pixels_invalid", seed, params, noise_model="gaussian")
    return _finish("bad_pixels_invalid",
                   "Hot/dead pixel clusters and hot columns (bad_pixel mask), NaN/Inf pixels (invalid mask), 4 px reference border (edge mask); 40 stars.",
                   image, bg, sig, sig, src, sources, meta,
                   masks={"bad_pixel": bad, "invalid": invalid, "edge": edge})


def tiny_frame(seed=116):
    width, height = 100, 80
    bg = np.full((height, width), SKY_ADU)
    rngs = _streams(seed)
    stars = _star_list(rngs[0], rngs[1], 3, width, height, 2e4, 2e4, GAUSS_PSF, margin=16, min_sep=20.0)
    params = {"width": width, "height": height, "sky_adu": SKY_ADU, "sigma_adu": SKY_SIGMA_ADU, "n_stars": 3,
              "designed_outcome": "smaller than two 64 px meshes per axis: a spatial mesh model is unsupported"}
    return _gaussian_scene(
        "tiny_frame",
        "100x80 frame, smaller than two 64 px meshes per axis; 3 stars.",
        seed, bg, np.full_like(bg, SKY_SIGMA_ADU), stars, params)


def all_masked(seed=117, width=256, height=192):
    rngs = _streams(seed)
    bg = np.full((height, width), SKY_ADU)
    sig = np.full_like(bg, SKY_SIGMA_ADU)
    image = bg + gaussian_noise(rngs[2], sig)
    params = {"designed_outcome": "no usable pixels: an honest method reports the background unavailable"}
    meta = _base_metadata("all_masked", seed, params, noise_model="gaussian")
    return _finish("all_masked", "Every pixel flagged invalid (finite values underneath).",
                   image, bg, sig, sig, np.zeros_like(bg), [], meta,
                   masks={"invalid": np.ones((height, width), bool)})


def all_nonfinite(seed=118, width=256, height=192):
    image = np.full((height, width), np.nan)
    image[::7, ::5] = np.inf
    params = {"designed_outcome": "no finite pixel: an honest method reports the background unavailable"}
    meta = _base_metadata("all_nonfinite", seed, params, noise_model=None)
    return _finish("all_nonfinite", "Every pixel NaN or +Inf, no mask supplied (non-finite values only).",
                   image, None, None, None, None, [], meta)


def zero_size(seed=119):
    params = {"designed_outcome": "malformed geometry: no method can measure anything"}
    meta = _base_metadata("zero_size", seed, params, bridge="excluded (Python-only status case)")
    return _finish("zero_size", "0x0 frame (Python-only; excluded from the Rust bridge manifest).",
                   np.zeros((0, 0)), None, None, None, None, [], meta)


SCENARIOS = {
    f.__name__: f
    for f in (
        uniform_gaussian_sky,
        poisson_read_low_counts,
        quantization_coarse,
        quantization_plateau,
        gradient_linear_xy,
        vignetting_radial,
        curved_bowl,
        steep_ridge_background,
        steep_ridge_amp600,
        amp_glow_corner,
        heteroscedastic_noise,
        sparse_gaussian_stars,
        moffat_stars,
        defocused_donut_stars,
        crowded_field,
        extended_emission,
        native_clipping,
        bad_pixels_invalid,
        tiny_frame,
        all_masked,
        all_nonfinite,
        zero_size,
    )
}


def build_scenario(name: str, seed: int | None = None) -> Scenario:
    builder = SCENARIOS[name]
    return builder() if seed is None else builder(seed=seed)


# --------------------------------------------------------------------------
# Focus sweeps
# --------------------------------------------------------------------------

SWEEP_POSITIONS = list(range(4000, 5201, 100))  # 13 focuser positions (steps)
SWEEP_P0 = 4630.0  # true optimum, deliberately off the sampling grid


def moffat_sweep_fwhm(p, p0=SWEEP_P0, w0=3.2, k=0.040):
    """FWHM(p) = sqrt(w0^2 + k^2 (p - p0)^2) px; k in px per focuser step."""
    return math.sqrt(w0 * w0 + (k * (p - p0)) ** 2)


def _sweep(sweep_id, description, seed, psf_for_position, n_bright, n_faint, params, min_sep, p0=SWEEP_P0):
    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    rngs = _streams(seed)
    xy = place_uniform(rngs[0], n_bright + n_faint, width, height, margin=40, min_sep=min_sep)
    fluxes = np.concatenate([log_uniform(rngs[1], 2e5, 1e6, n_bright), log_uniform(rngs[1], 3e3, 2e4, n_faint)])
    xx, yy = _grid(width, height)
    bg = SKY_ADU + 0.3 * (xx - (width - 1) / 2) + 0.2 * (yy - (height - 1) / 2)
    sig = np.full_like(bg, SKY_SIGMA_ADU)
    frames = []
    for idx, p in enumerate(SWEEP_POSITIONS):
        psf = psf_for_position(p)
        stars = [(x, y, float(f), psf) for (x, y), f in zip(xy, fluxes)]
        src, sources = add_stars((height, width), stars)
        assign_blend_groups(sources, 2.0 * min_sep / 3.0)
        noise_rng = np.random.default_rng(np.random.SeedSequence([seed, idx]))
        image = bg + src + gaussian_noise(noise_rng, sig)
        frame_id = f"{sweep_id}_p{p}"
        meta = _base_metadata(frame_id, seed, dict(params, position=p, psf=psf), noise_model="gaussian",
                              focus_sweep=sweep_id, focus_position=p, p0_true=p0,
                              noise_seed=f"SeedSequence([{seed}, {idx}])")
        frames.append(_finish(frame_id, f"{description} Position {p}.", image, bg, sig, sig, src, sources, meta))
    return FocusSweep(sweep_id, description, list(SWEEP_POSITIONS), p0, frames,
                      metadata={"params": params, "seed": seed, "generator_version": GENERATOR_VERSION,
                                "n_bright": n_bright, "n_faint": n_faint})


def sweep_moffat_fixed(seed=201):
    params = {"psf": "moffat beta=3", "fwhm_model": "sqrt(3.2^2 + (0.040 (p-p0))^2) px", "p0": SWEEP_P0,
              "background": "1000 + 0.3(x-xc) + 0.2(y-yc) ADU, constant over the sweep", "sigma_adu": SKY_SIGMA_ADU,
              "stars": "24 bright, flux 2e5-1e6 ADU, >= 55 px apart"}
    return _sweep("sweep_moffat_fixed", "Moffat focus sweep, fixed bright star set, gradient sky.", seed,
                  lambda p: {"model": "moffat", "fwhm_px": moffat_sweep_fwhm(p), "beta": 3.0}, 24, 0, params, 55.0)


def sweep_moffat_changing_set(seed=202):
    params = {"psf": "moffat beta=3", "fwhm_model": "sqrt(3.2^2 + (0.040 (p-p0))^2) px", "p0": SWEEP_P0,
              "background": "1000 + 0.3(x-xc) + 0.2(y-yc) ADU, constant over the sweep", "sigma_adu": SKY_SIGMA_ADU,
              "stars": "24 bright (2e5-1e6 ADU) + 40 faint (3e3-2e4 ADU) that fall below detection when defocused, >= 30 px apart"}
    return _sweep("sweep_moffat_changing_set",
                  "Moffat focus sweep in which faint stars drop out of detection away from focus.", seed,
                  lambda p: {"model": "moffat", "fwhm_px": moffat_sweep_fwhm(p), "beta": 3.0}, 24, 40, params, 30.0)


def donut_sweep_psf(p, p0=SWEEP_P0, c=0.028, obstruction=0.35, seeing=1.3):
    r_out = c * abs(p - p0)
    return {"model": "annulus", "r_out_px": r_out, "r_in_px": obstruction * r_out, "sigma_seeing_px": seeing}


def sweep_donut(seed=203):
    params = {"psf": "annulus r_out=0.028|p-p0| px, r_in=0.35 r_out, Gaussian seeing 1.3 px", "p0": SWEEP_P0,
              "background": "1000 + 0.3(x-xc) + 0.2(y-yc) ADU, constant over the sweep", "sigma_adu": SKY_SIGMA_ADU,
              "stars": "24 bright, flux 2e5-1e6 ADU, >= 55 px apart"}
    return _sweep("sweep_donut", "Geometric-defocus (donut) sweep of an obstructed aperture, fixed bright set.",
                  seed, donut_sweep_psf, 24, 0, params, 55.0)


SWEEPS = {f.__name__: f for f in (sweep_moffat_fixed, sweep_moffat_changing_set, sweep_donut)}


# --------------------------------------------------------------------------
# BGE v2 scenario sets (docs/bge-v2-spec.md §6): development variants and the
# held-out generators. Generator version stays bge-fixtures-v1: no pixel of
# a pre-existing scenario changes.
# --------------------------------------------------------------------------

#: The 22 core scenarios and 3 sweeps that existed before BGE v2 (A8 compares
#: v2 with v1 on exactly these).
PRE_EXISTING_SCENARIOS = tuple(SCENARIOS)
PRE_EXISTING_SWEEPS = tuple(SWEEPS)
#: Seeds already used by any generator before BGE v2 (101-122, 201-203, 301,
#: 401). The v2 development and held-out seeds are disjoint from these and
#: from each other (checked by the tests).
PRE_EXISTING_SEEDS = tuple(range(101, 123)) + (201, 202, 203, 301, 401)
DEV_V2_SEEDS = {
    "dev_newtonian_spider_donuts": 8101, "dev_refractor_disks": 8102, "dev_comatic_donuts": 8103,
    "dev_close_doubles": 8104, "dev_sweep_newtonian": 8105, "dev_amp_glow_ridge": 8107,
    "dev_dense_defocused_field": 8108,
}
HOLDOUT_SEEDS = {
    "ho_newtonian_spider_donuts": 9101, "ho_refractor_disks": 9102, "ho_comatic_donuts": 9103,
    "ho_close_doubles": 9104, "ho_sweep_newtonian": 9105, "ho_sweep_defocused_wings": 9106,
    "ho_amp_glow_ridge": 9107, "ho_dense_defocused_field": 9108,
}
SPLIT_DEV = "dev"
SPLIT_HOLDOUT = "holdout"
#: The daemon's gates stop after 16 used stars (quality.rs MAX_STARS_USED)
#: and examine at most 48 candidates. Frames whose acceptance is a criterion
#: hold at most this many acceptable sources, so the cap can never decide a
#: truth source's acceptance or rejection.
GATE_MAX_STARS_USED = 16
#: A truth source is "isolated" when every other source k satisfies
#: d_jk > 2 L_j + L_k, with L the light radius (pupil shadows: r_out + 3 sigma;
#: Moffat: 1.5 FWHM; Gaussian: 3 sigma). The measurement crop of the gates
#: spans about 2 L about the star, so no neighbour light enters it.
ISOLATION_RULE = "d_jk > 2 L_j + L_k; L = r_out + 3 sigma_seeing (pupil shadow), 1.5 FWHM (Moffat), 3 sigma (Gaussian)"


@dataclass
class ScenarioSet:
    """A group of frames scored together (e.g. several spider-donut frames,
    one spider orientation each)."""

    id: str
    description: str
    frames: list
    split: str
    metadata: dict = field(default_factory=dict)


def light_radius(psf: dict) -> float:
    if psf["model"] in ("pupil_shadow", "annulus"):
        return psf["r_out_px"] + 3.0 * psf["sigma_seeing_px"]
    if psf["model"] == "moffat":
        return 1.5 * psf["fwhm_px"]
    return 3.0 * psf["sigma_px"]


def mark_isolation(sources) -> None:
    """Set ``extra['isolated']`` per :data:`ISOLATION_RULE`."""
    if not sources:
        return
    xy = np.array([[s.x, s.y] for s in sources])
    light = np.array([light_radius(s.psf) for s in sources])
    for j, s in enumerate(sources):
        d = np.hypot(xy[:, 0] - s.x, xy[:, 1] - s.y)
        others = np.arange(len(sources)) != j
        s.extra["isolated"] = bool(np.all(d[others] > 2.0 * light[j] + light[others]))


def _frame_rngs(seed: int, index: int):
    return [np.random.default_rng(s) for s in np.random.SeedSequence([seed, index]).spawn(5)]


def _sloped_sky(width, height, gx=0.0, gy=0.0, level=SKY_ADU):
    xx, yy = _grid(width, height)
    return level + gx * (xx - (width - 1) / 2) + gy * (yy - (height - 1) / 2)


def _v2_frame(frame_id, description, seed, index, split, set_id, stars, params, width, height, *, sky=None,
              sigma_adu=SKY_SIGMA_ADU, link_px=10.0, extras=None, starless=False, rngs=None):
    """Render one v2 frame: ``stars`` = [(x, y, flux, psf)] on ``sky`` with
    Gaussian noise from stream 2 of ``SeedSequence([seed, index])``."""
    rngs = rngs or _frame_rngs(seed, index)
    bg = _sloped_sky(width, height) if sky is None else sky
    sig = np.full_like(bg, sigma_adu)
    src, sources = add_stars((height, width), stars)
    assign_blend_groups(sources, link_px)
    for s, extra in zip(sources, extras or [{}] * len(sources)):
        s.extra.update(extra)
        if s.psf["model"] == "pupil_shadow":
            for key in ("r_out_px", "obstruction", "n_vanes", "vane_width_frac", "sigma_seeing_px"):
                s.extra.setdefault(key, s.psf.get(key))
            s.extra.setdefault("hfr_centre", "flux_centroid")
    mark_isolation(sources)
    image = bg + src + gaussian_noise(rngs[2], sig)
    meta = _base_metadata(frame_id, seed, dict(params, frame_index=index), noise_model="gaussian",
                          split=split, scenario_set=set_id, starless=bool(starless), pre_existing=False,
                          frame_seed=f"SeedSequence([{seed}, {index}])", isolation_rule=ISOLATION_RULE)
    return _finish(frame_id, description, image, bg, sig, sig, src, sources, meta)


def _stratified(rng, lo, hi, n):
    """One uniform draw in each of ``n`` equal strata of [lo, hi], shuffled."""
    values = lo + (hi - lo) * (np.arange(n) + rng.uniform(0.0, 1.0, n)) / n
    return values[rng.permutation(n)]


def _pupil_psf(r_out, obstruction, seeing, n_vanes=0, vane_width_frac=0.0, vane_angle_rad=0.0, modulation=0.0,
               modulation_angle_rad=0.0, hole_offset_px=(0.0, 0.0)) -> dict:
    return {"model": "pupil_shadow", "r_out_px": float(r_out), "obstruction": float(obstruction),
            "sigma_seeing_px": float(seeing), "n_vanes": int(n_vanes), "vane_width_frac": float(vane_width_frac),
            "vane_angle_rad": float(vane_angle_rad), "modulation": float(modulation),
            "modulation_angle_rad": float(modulation_angle_rad),
            "hole_offset_px": [float(hole_offset_px[0]), float(hole_offset_px[1])]}


def _donut_positions(rng, r_out, seeing, width, height, n_max=GATE_MAX_STARS_USED):
    light = r_out + 3.0 * seeing
    render = r_out + 5.0 * seeing + 2.0
    min_sep = 3.2 * light
    margin = 1.3 * render + 6.0
    n = int(min(n_max, max(4, 0.3 * width * height / (min_sep * min_sep))))
    return place_uniform(rng, n, width, height, margin, min_sep)


def _donut_set(set_id, description, seed, split, n_frames, *, r_range, obstruction_range, vanes, width=800,
               height=600, comatic=False, flux_range=(5e4, 1e6)):
    """Frames of geometric-defocus donuts, one telescope state per frame.

    ``vanes``: ``None`` (no spider) or a list of per-frame vane counts. Per
    frame: one ``r_out`` (stratified over ``r_range``), obstruction, vane
    width (0.5-1.5% of D), vane angle (shared by every star of the frame) and
    seeing sigma (1.0-1.8 px). Comatic sets add, per star, a +-40%
    brightness modulation at a random azimuth and an obstruction offset of
    up to 0.2 r_out.
    """
    setup = _streams(seed)[4]
    r_outs = _stratified(setup, r_range[0], r_range[1], n_frames)
    frames = []
    for idx in range(n_frames):
        rngs = _frame_rngs(seed, idx)
        p = rngs[4]
        obstruction = float(p.uniform(*obstruction_range)) if obstruction_range else 0.0
        seeing = float(p.uniform(1.0, 1.8))
        n_vanes = int(vanes[idx]) if vanes else 0
        vane_width = float(p.uniform(0.005, 0.015)) if n_vanes else 0.0
        vane_angle = float(p.uniform(0.0, 2.0 * math.pi / n_vanes)) if n_vanes else 0.0
        r_out = float(r_outs[idx])
        xy = _donut_positions(rngs[0], r_out, seeing, width, height)
        fluxes = log_uniform(rngs[1], flux_range[0], flux_range[1], len(xy))
        stars, extras = [], []
        for (x, y), f in zip(xy, fluxes):
            if comatic:
                offset = float(p.uniform(0.0, 0.2)) * r_out
                direction = float(p.uniform(0.0, 2.0 * math.pi))
                psf = _pupil_psf(r_out, obstruction, seeing, modulation=0.4,
                                 modulation_angle_rad=float(p.uniform(0.0, 2.0 * math.pi)),
                                 hole_offset_px=(offset * math.cos(direction), offset * math.sin(direction)))
                extras.append({"hole_offset_frac": offset / r_out, "modulation": 0.4})
            else:
                psf = _pupil_psf(r_out, obstruction, seeing, n_vanes, vane_width, vane_angle)
                extras.append({})
            stars.append((x, y, float(f), psf))
        params = {"r_out_px": r_out, "obstruction": obstruction, "sigma_seeing_px": seeing, "n_vanes": n_vanes,
                  "vane_width_frac": vane_width, "vane_angle_rad": vane_angle, "comatic": comatic,
                  "flux_adu": list(flux_range), "n_stars": len(stars), "sky_adu": SKY_ADU, "sigma_adu": SKY_SIGMA_ADU}
        frame_id = f"{set_id}_f{idx:02d}"
        frames.append(_v2_frame(frame_id, f"{description} Frame {idx}: r_out {r_out:.1f} px.", seed, idx, split,
                                set_id, stars, params, width, height, extras=extras, link_px=2.0 * r_out,
                                rngs=rngs))
    return ScenarioSet(set_id, description, frames, split,
                       {"seed": seed, "n_frames": n_frames, "r_range_px": list(r_range),
                        "obstruction_range": list(obstruction_range) if obstruction_range else None,
                        "vanes": list(vanes) if vanes else None, "comatic": comatic,
                        "generator_version": GENERATOR_VERSION})


def _vane_counts(seed, n_frames, n_three):
    """Per-frame vane counts: 4 vanes, ``n_three`` frames with the 3-vane variant."""
    counts = np.full(n_frames, 4)
    counts[_streams(seed)[3].permutation(n_frames)[:n_three]] = 3
    return [int(c) for c in counts]


def newtonian_spider_donuts(set_id, seed, split, n_frames=8, n_three=2):
    return _donut_set(set_id, "Z114-like Newtonian donuts: obstruction 0.25-0.35, spider vanes 0.5-1.5% of D "
                      "(4 vanes; 3-vane variant), one vane angle per frame, r_out 6-32 px, seeing sigma 1.0-1.8 px, "
                      "flux 5e4-1e6 ADU.", seed, split, n_frames, r_range=(6.0, 32.0),
                      obstruction_range=(0.25, 0.35), vanes=_vane_counts(seed, n_frames, n_three))


def refractor_disks(set_id, seed, split, n_frames=6):
    return _donut_set(set_id, "Defocused unobstructed refractor disks, r_out 6-24 px, seeing sigma 1.0-1.8 px, "
                      "flux 5e4-1e6 ADU.", seed, split, n_frames, r_range=(6.0, 24.0), obstruction_range=None,
                      vanes=None)


def comatic_donuts(set_id, seed, split, n_frames=6):
    return _donut_set(set_id, "Comatic donuts: obstruction 0.25-0.35, ring brightness modulated +-40% in azimuth, "
                      "obstruction offset up to 0.2 r_out, r_out 6-32 px.", seed, split, n_frames,
                      r_range=(6.0, 32.0), obstruction_range=(0.25, 0.35), vanes=None, comatic=True)


def close_doubles(set_id, seed, split, n_frames=6, n_pairs=10, n_singles=4, width=640, height=480):
    """Moffat (beta 3) pairs: FWHM 3-6 px (one per frame), separation 3-10 px,
    flux ratio 0.2-1.0, plus isolated singles as controls. 10 pairs + 4
    singles per frame: at most 24 candidates and 14 acceptable sources, so the
    gates' 48-examined / 16-used caps never bind."""
    setup = _streams(seed)[4]
    fwhms = _stratified(setup, 3.0, 6.0, n_frames)
    frames = []
    for idx in range(n_frames):
        rngs = _frame_rngs(seed, idx)
        fwhm = float(fwhms[idx])
        psf = {"model": "moffat", "fwhm_px": fwhm, "beta": 3.0}
        centres = place_uniform(rngs[0], n_pairs + n_singles, width, height, margin=24, min_sep=48.0)
        stars, extras = [], []
        for k, (x, y) in enumerate(centres):
            f1 = float(log_uniform(rngs[1], 2e4, 3e5, 1)[0])
            if k < n_pairs:
                sep_px = float(rngs[4].uniform(3.0, 10.0))
                ratio = float(rngs[4].uniform(0.2, 1.0))
                angle = float(rngs[4].uniform(0.0, 2.0 * math.pi))
                dx, dy = 0.5 * sep_px * math.cos(angle), 0.5 * sep_px * math.sin(angle)
                pair = {"pair_id": k, "separation_px": sep_px, "flux_ratio": ratio, "pair_mid_x": x, "pair_mid_y": y}
                stars.append((x - dx, y - dy, f1, psf))
                extras.append(dict(pair, role="primary"))
                stars.append((x + dx, y + dy, f1 * ratio, psf))
                extras.append(dict(pair, role="secondary"))
            else:
                stars.append((x, y, f1, psf))
                extras.append({"pair_id": None, "role": "single"})
        params = {"psf": psf, "n_pairs": n_pairs, "n_singles": n_singles, "separation_px": [3.0, 10.0],
                  "flux_ratio": [0.2, 1.0], "primary_flux_adu": [2e4, 3e5], "sky_adu": SKY_ADU,
                  "sigma_adu": SKY_SIGMA_ADU}
        frame_id = f"{set_id}_f{idx:02d}"
        frames.append(_v2_frame(frame_id, f"Close doubles, Moffat FWHM {fwhm:.2f} px. Frame {idx}.", seed, idx,
                                split, set_id, stars, params, width, height, extras=extras, link_px=12.0,
                                rngs=rngs))
    return ScenarioSet(set_id, "Close doubles: Moffat FWHM 3-6 px, separation 3-10 px, flux ratio 0.2-1.0.",
                       frames, split, {"seed": seed, "n_frames": n_frames, "generator_version": GENERATOR_VERSION})


def amp_glow_ridge(set_id, seed, split, n_amplitudes=6, width=512, height=512):
    """Level ridges of 150-900 ADU (Gaussian profile, 45-75 px 1/e half-width,
    vertical or horizontal) on 1000 ADU with 5 ADU noise; each amplitude once
    starless and once with 30 Moffat stars."""
    setup = _streams(seed)[4]
    amplitudes = _stratified(setup, 150.0, 900.0, n_amplitudes)
    frames = []
    for idx in range(2 * n_amplitudes):
        starless = idx < n_amplitudes
        amplitude = float(amplitudes[idx % n_amplitudes])
        rngs = _frame_rngs(seed, idx)
        vertical = bool(rngs[4].integers(0, 2))
        centre = float(rngs[4].uniform(0.35, 0.65)) * (width if vertical else height)
        half_width = float(rngs[4].uniform(45.0, 75.0))
        xx, yy = _grid(width, height)
        coord = xx if vertical else yy
        sky = 1000.0 + amplitude * np.exp(-(((coord - centre) / half_width) ** 2))
        stars = []
        psf = {"model": "moffat", "fwhm_px": 4.0, "beta": 3.0}
        if not starless:
            stars = _star_list(rngs[0], rngs[1], 30, width, height, 5e3, 2e5, psf, margin=16, min_sep=25.0)
        params = {"ridge_amplitude_adu": amplitude, "ridge_orientation": "vertical" if vertical else "horizontal",
                  "ridge_centre_px": centre, "ridge_half_width_px": half_width, "sky_adu": 1000.0, "sigma_adu": 5.0,
                  "n_stars": len(stars), "psf": psf if stars else None}
        frame_id = f"{set_id}_f{idx:02d}"
        frames.append(_v2_frame(frame_id, f"Sky ridge +{amplitude:.0f} ADU ({'starless' if starless else '30 stars'}). "
                                f"Frame {idx}.", seed, idx, split, set_id, stars, params, width, height, sky=sky,
                                sigma_adu=5.0, starless=starless, rngs=rngs))
    return ScenarioSet(set_id, "Amplifier-glow-like level ridges of 150-900 ADU on starless and starred frames.",
                       frames, split, {"seed": seed, "n_frames": 2 * n_amplitudes,
                                       "generator_version": GENERATOR_VERSION})


def dense_defocused_field(set_id, seed, split, n_frames=4, n_stars=80, width=640, height=480):
    """Many broad spider donuts (r_out 8-16 px, one per frame) close together,
    contaminating the background meshes; overlaps allowed (10 px minimum)."""
    setup = _streams(seed)[4]
    r_outs = _stratified(setup, 8.0, 16.0, n_frames)
    frames = []
    for idx in range(n_frames):
        rngs = _frame_rngs(seed, idx)
        r_out = float(r_outs[idx])
        angle = float(rngs[4].uniform(0.0, 0.5 * math.pi))
        psf = _pupil_psf(r_out, 0.3, 1.4, 4, 0.01, angle)
        xy = place_uniform(rngs[0], n_stars, width, height, margin=r_out + 10.0, min_sep=10.0)
        fluxes = log_uniform(rngs[1], 2e4, 5e5, n_stars)
        stars = [(x, y, float(f), psf) for (x, y), f in zip(xy, fluxes)]
        params = {"r_out_px": r_out, "obstruction": 0.3, "n_vanes": 4, "vane_width_frac": 0.01,
                  "sigma_seeing_px": 1.4, "n_stars": n_stars, "flux_adu": [2e4, 5e5], "sky_adu": SKY_ADU,
                  "sigma_adu": SKY_SIGMA_ADU}
        frame_id = f"{set_id}_f{idx:02d}"
        frames.append(_v2_frame(frame_id, f"Dense defocused field, {n_stars} donuts r_out {r_out:.1f} px. "
                                f"Frame {idx}.", seed, idx, split, set_id, stars, params, width, height,
                                sky=_sloped_sky(width, height, 0.2, 0.1), link_px=2.0 * r_out, rngs=rngs))
    return ScenarioSet(set_id, "Dense field of broad defocused donuts contaminating the meshes.", frames, split,
                       {"seed": seed, "n_frames": n_frames, "generator_version": GENERATOR_VERSION})


SWEEP_STEP = 100


def newtonian_sweep(sweep_id, seed, split, p0=4585.0, c=0.03, n_stars=20, width=640, height=480):
    """Focus sweep through spider donuts on both sides of focus: r_out =
    c |p - p0| px (c = 0.03 px/step), obstruction 0.3, 4 vanes of 1% D, one
    vane angle for the sweep, seeing sigma 1.4 px; 20 bright stars (2e5-1e6
    ADU) >= 70 px apart on a gradient sky."""
    rngs = _streams(seed)
    xy = place_uniform(rngs[0], n_stars, width, height, margin=45, min_sep=70.0)
    fluxes = log_uniform(rngs[1], 2e5, 1e6, n_stars)
    angle = float(rngs[4].uniform(0.0, 0.5 * math.pi))
    sky = _sloped_sky(width, height, 0.3, 0.2)
    frames = []
    for idx, pos in enumerate(SWEEP_POSITIONS):
        r_out = c * abs(pos - p0)
        psf = _pupil_psf(r_out, 0.3, 1.4, 4, 0.01, angle)
        stars = [(x, y, float(f), psf) for (x, y), f in zip(xy, fluxes)]
        params = {"position": pos, "p0": p0, "r_out_px": r_out, "psf": psf, "c_px_per_step": c}
        frame_id = f"{sweep_id}_p{pos}"
        frame = _v2_frame(frame_id, f"Newtonian spider-donut sweep. Position {pos}.", seed, idx, split, sweep_id,
                          stars, params, width, height, sky=sky, link_px=40.0, rngs=_frame_rngs(seed, idx))
        frame.metadata.update(focus_sweep=sweep_id, focus_position=pos, p0_true=p0)
        frames.append(frame)
    sweep = FocusSweep(sweep_id, "Focus sweep through Newtonian spider donuts on both sides of focus, gradient sky.",
                       list(SWEEP_POSITIONS), p0, frames,
                       metadata={"seed": seed, "split": split, "step": SWEEP_STEP, "p0_true": p0, "c_px_per_step": c,
                                 "generator_version": GENERATOR_VERSION, "n_stars": n_stars})
    return sweep


def defocused_wings_sweep(sweep_id, seed, split, p0=4655.0):
    """The D-020 class: sweep_moffat_fixed's generator (Moffat beta 3, broad
    wings far from focus, 24 bright stars, gradient sky) with a new seed and
    optimum."""
    params = {"psf": "moffat beta=3", "fwhm_model": f"sqrt(3.2^2 + (0.040 (p-{p0}))^2) px", "p0": p0,
              "background": "1000 + 0.3(x-xc) + 0.2(y-yc) ADU, constant over the sweep", "sigma_adu": SKY_SIGMA_ADU,
              "stars": "24 bright, flux 2e5-1e6 ADU, >= 55 px apart", "class": "D-020 (broad Moffat wings)"}
    sweep = _sweep(sweep_id, "Moffat focus sweep with broad wings far from focus (D-020 class).", seed,
                   lambda p: {"model": "moffat", "fwhm_px": moffat_sweep_fwhm(p, p0=p0), "beta": 3.0}, 24, 0, params,
                   55.0, p0=p0)
    sweep.metadata.update(split=split, step=SWEEP_STEP, p0_true=p0)
    for fr in sweep.frames:
        fr.metadata.update(split=split, scenario_set=sweep_id, pre_existing=False)
    return sweep


def _dev(name):
    return DEV_V2_SEEDS[name]


def _ho(name):
    return HOLDOUT_SEEDS[name]


#: Development variants (split "dev"): the held-out generators with dev seeds
#: and fewer frames. Builders take no arguments.
DEV_V2_SETS = {
    "dev_newtonian_spider_donuts": lambda: newtonian_spider_donuts("dev_newtonian_spider_donuts",
                                                                   _dev("dev_newtonian_spider_donuts"), SPLIT_DEV,
                                                                   n_frames=4, n_three=1),
    "dev_refractor_disks": lambda: refractor_disks("dev_refractor_disks", _dev("dev_refractor_disks"), SPLIT_DEV,
                                                   n_frames=3),
    "dev_comatic_donuts": lambda: comatic_donuts("dev_comatic_donuts", _dev("dev_comatic_donuts"), SPLIT_DEV,
                                                 n_frames=3),
    "dev_close_doubles": lambda: close_doubles("dev_close_doubles", _dev("dev_close_doubles"), SPLIT_DEV, n_frames=3),
    "dev_amp_glow_ridge": lambda: amp_glow_ridge("dev_amp_glow_ridge", _dev("dev_amp_glow_ridge"), SPLIT_DEV,
                                                 n_amplitudes=3),
    "dev_dense_defocused_field": lambda: dense_defocused_field("dev_dense_defocused_field",
                                                               _dev("dev_dense_defocused_field"), SPLIT_DEV,
                                                               n_frames=2),
}
DEV_V2_SWEEPS = {
    "dev_sweep_newtonian": lambda: newtonian_sweep("dev_sweep_newtonian", _dev("dev_sweep_newtonian"), SPLIT_DEV,
                                                   p0=4615.0),
}
#: Held-out sets (split "holdout"), generated and evaluated once after the
#: acceptance criteria (ml/eval/bge/acceptance/synthetic-v2.json) were
#: committed. run_benchmark refuses this split unless that file is tracked
#: and unchanged.
HOLDOUT_SETS = {
    "ho_newtonian_spider_donuts": lambda: newtonian_spider_donuts("ho_newtonian_spider_donuts",
                                                                  _ho("ho_newtonian_spider_donuts"), SPLIT_HOLDOUT),
    "ho_refractor_disks": lambda: refractor_disks("ho_refractor_disks", _ho("ho_refractor_disks"), SPLIT_HOLDOUT),
    "ho_comatic_donuts": lambda: comatic_donuts("ho_comatic_donuts", _ho("ho_comatic_donuts"), SPLIT_HOLDOUT),
    "ho_close_doubles": lambda: close_doubles("ho_close_doubles", _ho("ho_close_doubles"), SPLIT_HOLDOUT),
    "ho_amp_glow_ridge": lambda: amp_glow_ridge("ho_amp_glow_ridge", _ho("ho_amp_glow_ridge"), SPLIT_HOLDOUT),
    "ho_dense_defocused_field": lambda: dense_defocused_field("ho_dense_defocused_field",
                                                              _ho("ho_dense_defocused_field"), SPLIT_HOLDOUT),
}
HOLDOUT_SWEEPS = {
    "ho_sweep_newtonian": lambda: newtonian_sweep("ho_sweep_newtonian", _ho("ho_sweep_newtonian"), SPLIT_HOLDOUT),
    "ho_sweep_defocused_wings": lambda: defocused_wings_sweep("ho_sweep_defocused_wings",
                                                              _ho("ho_sweep_defocused_wings"), SPLIT_HOLDOUT),
}
HOLDOUT_IDS = tuple(HOLDOUT_SETS) + tuple(HOLDOUT_SWEEPS)


# --------------------------------------------------------------------------
# Mesh-size / source-footprint study fields (BGE-09)
# --------------------------------------------------------------------------

MESH_STUDY_FOOTPRINTS = {
    "focused_moffat_fwhm3": {"model": "moffat", "fwhm_px": 3.0, "beta": 3.0},
    "soft_moffat_fwhm8": {"model": "moffat", "fwhm_px": 8.0, "beta": 3.0},
    "donut_r12": {"model": "annulus", "r_out_px": 12.0, "r_in_px": 4.2, "sigma_seeing_px": 1.3},
    "donut_r24": {"model": "annulus", "r_out_px": 24.0, "r_in_px": 8.4, "sigma_seeing_px": 1.3},
    "donut_r40": {"model": "annulus", "r_out_px": 40.0, "r_in_px": 14.0, "sigma_seeing_px": 1.3},
}
MESH_STUDY_SIZES = (16, 32, 64, 128)


def mesh_study_fields(seed=301):
    """One fixed 5x4 jittered grid of 20 equal stars, re-rendered per footprint.

    Same positions, fluxes and noise realisation for every footprint (common
    random numbers), on the curved vignetting background, so differences are
    due to footprint size and mesh size only.
    """
    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    rngs = _streams(seed)
    cell_x, cell_y = width / 5, height / 4
    xy = [((i + 0.5) * cell_x + rngs[0].uniform(-10, 10), (j + 0.5) * cell_y + rngs[0].uniform(-10, 10))
          for j in range(4) for i in range(5)]
    flux = 3e5
    bg = vignetting_background(width, height)
    sig = np.full_like(bg, SKY_SIGMA_ADU)
    noise = gaussian_noise(rngs[2], sig)
    fields = []
    for name, psf in MESH_STUDY_FOOTPRINTS.items():
        stars = [(x, y, flux, psf) for x, y in xy]
        src, sources = add_stars((height, width), stars)
        assign_blend_groups(sources, 10.0)
        params = {"footprint": name, "psf": psf, "flux_adu": flux, "n_stars": 20,
                  "background": "vignetting 1000(1-0.25rho^2-0.05rho^4)", "sigma_adu": SKY_SIGMA_ADU}
        meta = _base_metadata(f"mesh_{name}", seed, params, noise_model="gaussian", footprint=name)
        fields.append(_finish(f"mesh_{name}", f"Mesh study field, footprint {name}.",
                              bg + src + noise, bg, sig, sig, src, sources, meta))
    return fields


# --------------------------------------------------------------------------
# Timing frame (BGE-23)
# --------------------------------------------------------------------------


def timing_frame(seed=401, width=6000, height=4000, n_stars=2000):
    """6000x4000 frame for host timing only; truth maps are not retained."""
    rngs = _streams(seed)
    xx = np.arange(width, dtype=np.float32)[None, :]
    yy = np.arange(height, dtype=np.float32)[:, None]
    image = (SKY_ADU + 0.04 * (xx - width / 2) + 0.03 * (yy - height / 2)).astype(np.float32)
    image = image + (rngs[2].standard_normal((height, width), dtype=np.float32) * np.float32(SKY_SIGMA_ADU))
    xy = place_uniform(rngs[0], n_stars, width, height, margin=16)
    fluxes = log_uniform(rngs[1], 2e3, 2e5, n_stars)
    for (x, y), f in zip(xy, fluxes):
        stamp = render_gaussian(x, y, 2.0, float(f))
        ny, nx = stamp.data.shape
        image[stamp.iy0:stamp.iy0 + ny, stamp.ix0:stamp.ix0 + nx] += stamp.data.astype(np.float32)
    params = {"width": width, "height": height, "n_stars": n_stars, "psf": GAUSS_PSF,
              "background": "1000 + 0.04(x-W/2) + 0.03(y-H/2) ADU", "sigma_adu": SKY_SIGMA_ADU,
              "purpose": "timing/memory only; no accuracy metrics"}
    meta = _base_metadata("timing_6000x4000", seed, params, noise_model="gaussian")
    return Scenario("timing_6000x4000", "6000x4000 timing frame (no truth retained).",
                    np.ascontiguousarray(image), None, None, None, None, {}, [], meta)
