"""
psf_measure — reference PSF/quality labels for the FQE dataset (Section 7.7.5).

This module produces the *ground-truth labels* that replace the hand-wavy
reciprocal focus scores in the daemon (``achernar-daemon/src/vision/quality.rs``).
It follows the source-detection / background / half-light-radius foundation of
SExtractor (Bertin & Arnouts 1996), SEP (Barbary 2016) and DAOPHOT
(Stetson 1987), and the PSF-width conventions of Law et al. (2006).

Two families of code live here and must not be confused:

  * **The SEP labeler** (:func:`estimate_spatial_background`,
    :func:`measure_frame_sep`). It calls the pinned SEP 1.4.1 library for the
    spatial mesh background B(x,y) and RMS R(x,y), for detection/deblending
    (``sep.extract``) and for aperture measurements (``sep.flux_radius``,
    ``sep.sum_circle``). Agreement between SEP and a SExtractor executable is
    a third, separate claim that is not tested here (Barbary 2016 p.1: SEP is a
    fork that has "diverged significantly" from the original code base).
  * **Custom crop helpers** (:func:`estimate_background`,
    :func:`detect_sources`, :func:`curve_of_growth_hfr`, :func:`measure_star`).
    These are Achernar's own NumPy/SciPy implementations, global within one
    crop, with no mesh, no matched filter and no deblending. Their method ids
    (``python_crop_*``) say so.

Key facts this module encodes:

  * **HFR** (half-flux radius) is measured by a curve of growth after *local*
    background subtraction. HFD = 2 * HFR by definition.
  * **FWHM** is a *fitted* PSF width. ``FWHM = 2 * HFR`` holds only for a
    circular Gaussian, not for a Moffat PSF, so FWHM is never derived from HFR.
    A Gaussian-moment fallback is labelled as such and never pooled with
    Moffat fits.
  * Missing / invalid measurements are ``None`` (never ``0.0``), because 0 is the
    mathematically *best* value for a lower-is-better width metric and would let
    an invalid frame win a focus comparison. The same applies to background
    level and noise: an unavailable or unresolved estimate is ``None`` with a
    status and reason, never a fabricated number.

All widths are in **pixels**; levels and noise are in the caller's data units
(``units``; ADU for native camera frames). Conversion to arc-seconds requires a
known pixel scale and is intentionally *not* done here.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Optional

import numpy as np

try:
    from scipy.optimize import least_squares

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - scipy is a hard dep for fitting
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# Identity and provenance constants. These strings are part of the frozen
# cross-language contract with achernar-daemon/src/vision/background_model.rs.
# --------------------------------------------------------------------------- #
SEP_PINNED_VERSION = "1.4.1"

# Version of the per-label/per-frame metadata emitted by measure_frame_sep.
# Bump on any change to the meaning of a field, to the detection/measurement
# policy, or to the background model; frozen datasets carry the old value.
LABEL_SCHEMA_VERSION = "fqe-sep-label-v2"

BACKGROUND_METHOD_SEP = "sep_spatial_mesh"
BACKGROUND_METHOD_SEP_VERSION = 1
DETECTION_METHOD_SEP = "sep_extract"

# Custom Python helpers. Named for what they are: not SEP.
BACKGROUND_METHOD_CROP = "python_crop_iterative_clip"
DETECTION_METHOD_CROP = "python_crop_connected_components"
HFR_METHOD_CROP = "python_signed_curve_of_growth"
HFR_METHOD_SEP = "sep_flux_radius"

NOISE_EMPIRICAL_MESH_RMS = "empirical_mesh_rms"
NOISE_PIPELINE_ERROR = "pipeline_error"
NOISE_CALIBRATED_DETECTOR_MODEL = "calibrated_detector_model"
NOISE_UNAVAILABLE = "unavailable"

STATUS_MEASURED = "measured"
STATUS_UNAVAILABLE = "unavailable"
STATUS_UNRESOLVED = "unresolved"

# Reason codes (snake_case, identical to the Rust BackgroundReason serde names).
REASON_MALFORMED_GEOMETRY = "malformed_geometry"
REASON_NO_USABLE_PIXELS = "no_usable_pixels"
REASON_INSUFFICIENT_SUPPORT = "insufficient_support"
REASON_FRAME_SMALLER_THAN_MESH_GRID = "frame_smaller_than_mesh_grid"
REASON_ZERO_EMPIRICAL_SCATTER = "zero_empirical_scatter"
REASON_BACKEND_FAILURE = "backend_failure"
REASON_FALLBACK_USED = "fallback_used"
REASON_NO_SPATIAL_MODEL = "no_spatial_model"
REASON_INSUFFICIENT_MESHES_FOR_PLANE = "insufficient_meshes_for_plane"
REASON_MESHES_INTERPOLATED = "meshes_interpolated"
# SEP's bicubic RMS spline can undershoot below zero between meshes whose
# sigmas differ sharply (e.g. a narrow bright ridge inflating some mesh
# sigmas). A negative or non-finite R(x,y) on usable support cannot serve as
# a detection noise map (a negative per-pixel threshold detects everything).
REASON_INVALID_RMS_MAP = "invalid_rms_map"

# SEP background.c: a mesh with fewer than this fraction of usable pixels is
# discarded and replaced by the mean of its nearest valid meshes
# (BACK_MINGOODFRAC, compared as npix < area * 0.5).
SEP_BACK_MINGOODFRAC = 0.5

# SEP Python wrapper default kernel (sep.pyx `default_kernel`).
SEP_DEFAULT_KERNEL = ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0))

LEVEL_DEFINITION_SEP = (
    "global_background = sep.Background.globalback = sep_bkg_global: the median "
    "of the median-filtered per-mesh background values (SEP 1.4.1 background.c, "
    "fqmedian over all nx*ny meshes, including meshes SEP filled from neighbours). "
    "Local values B(x,y) are SEP's bicubic-spline map (sep_bkg_array)."
)
RMS_DEFINITION_SEP = (
    "global_rms = sep.Background.globalrms = sep_bkg_globalrms: the median of the "
    "median-filtered per-mesh clipped-histogram sigma (no truncation correction). "
    "Local values R(x,y) are SEP's bicubic-spline map (sep_bkg_rmsarray). Empirical: "
    "already contains sky shot noise, dark and read noise, and quantization."
)


class MissingPrerequisite(RuntimeError):
    """A required library is absent or is not the pinned version."""


def require_sep():
    """Import SEP, insisting on the pinned version.

    Labels claim SEP 1.4.1 semantics (verified threshold/filter/mask behaviour
    of that release), so any other version is a missing prerequisite rather
    than a silently different label.
    """
    try:
        import sep
    except ImportError as exc:
        raise MissingPrerequisite(
            f"missing prerequisite: sep=={SEP_PINNED_VERSION} is not installed "
            f"(pip install sep=={SEP_PINNED_VERSION})"
        ) from exc
    version = getattr(sep, "__version__", None)
    if version != SEP_PINNED_VERSION:
        raise MissingPrerequisite(
            f"missing prerequisite: sep=={SEP_PINNED_VERSION} (found sep {version})"
        )
    return sep


def canonical_json(obj) -> str:
    """Canonical JSON used for configuration hashes (matches Rust canonical_json)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def config_sha256(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Explicit SEP configuration.
# --------------------------------------------------------------------------- #
def _require_int(name: str, value, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _require_finite(name: str, value, *, minimum: Optional[float] = None, strict: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None and (value <= minimum if strict else value < minimum):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be {relation} {minimum}, got {value!r}")


@dataclass(frozen=True)
class SepBackgroundParams:
    """``sep.Background`` mesh settings, in pixels.

    Starting values, not qualified camera defaults: 64x64 px meshes and a 3x3
    median filter of the mesh grid are the SEP Python defaults (sep.pyx
    ``Background``) and the SExtractor defaults quoted in ``sep.h``. Bertin &
    Arnouts (1996, §2, p.394) report 32-128 px meshes "work fine" on most
    images; the right size depends on source footprints and on the spatial
    scale of the real background (BGE-09).

    ``min_meshes_per_axis``: the frame must hold at least this many full meshes
    along each axis (``width >= n*bw`` and ``height >= n*bh``); below that a
    "mesh" model is one global estimate wearing a spatial label, so the
    background is reported unavailable instead.
    """

    bw: int = 64
    bh: int = 64
    fw: int = 3
    fh: int = 3
    fthresh: float = 0.0
    min_meshes_per_axis: int = 2

    def __post_init__(self):
        for name in ("bw", "bh", "fw", "fh", "min_meshes_per_axis"):
            _require_int(name, getattr(self, name), 1)
        _require_finite("fthresh", self.fthresh, minimum=0.0)

    def config(self, library_version: str = SEP_PINNED_VERSION) -> dict:
        """Canonical configuration object (keys frozen by the cross-language contract)."""
        return {
            "method": BACKGROUND_METHOD_SEP,
            "method_version": BACKGROUND_METHOD_SEP_VERSION,
            "library": "sep",
            "library_version": library_version,
            "bw": int(self.bw),
            "bh": int(self.bh),
            "fw": int(self.fw),
            "fh": int(self.fh),
            "fthresh": float(self.fthresh),
            "maskthresh": 0.0,
            "min_meshes_per_axis": int(self.min_meshes_per_axis),
        }

    def config_sha256(self, library_version: str = SEP_PINNED_VERSION) -> str:
        return config_sha256(self.config(library_version))


@dataclass(frozen=True)
class SepDetectionParams:
    """``sep.extract`` settings for the full-frame labeler.

    ``thresh_sigma`` is relative to the local background RMS map R(x,y).
    ``filter_type="conv"`` is deliberate: in SEP 1.4.1 a CONV-filtered
    residual is compared with ``thresh_sigma * R(x,y)``, whereas MATCHED
    compares a unit-variance S/N with ``thresh_sigma`` itself (extract.c
    L445-448, L604-605). For the default 3x3 kernel MATCHED is therefore
    ~2.67x more sensitive at the same number (sum|k| = 16, sqrt(sum k^2) = 6).
    The pre-v2 labeler passed a scalar ``err=globalrms``, which SEP silently
    downgrades to CONV, so ``thresh_sigma=5.0`` with ``conv`` keeps the old
    detection sensitivity while the noise becomes local. MATCHED remains
    available, but its threshold must then be re-derived, not copied.

    ``accept_deblended_for_focus``: SEP deblending (splitting a blend, which
    always runs with ``deblend_cont < 1``) and accepting a deblended child as a
    focus star (a policy) are separate decisions; the default rejects
    ``OBJ_MERGED`` children, as the pre-v2 labeler did.

    ``pixstack`` is SEP's process-global active-pixel stack; the labeler sets
    it for the duration of one extraction and restores the previous value.
    """

    thresh_sigma: float = 5.0
    minarea: int = 5
    filter_type: str = "conv"
    filter_kernel: tuple = SEP_DEFAULT_KERNEL
    deblend_nthresh: int = 32
    deblend_cont: float = 0.005
    clean: bool = True
    clean_param: float = 1.0
    pixstack: int = 300_000
    accept_deblended_for_focus: bool = False

    def __post_init__(self):
        _require_finite("thresh_sigma", self.thresh_sigma, minimum=0.0, strict=True)
        _require_int("minarea", self.minarea, 1)
        _require_int("deblend_nthresh", self.deblend_nthresh, 1)
        _require_int("pixstack", self.pixstack, 1)
        _require_finite("deblend_cont", self.deblend_cont, minimum=0.0, strict=True)
        if self.deblend_cont > 1.0:
            raise ValueError(f"deblend_cont must be <= 1, got {self.deblend_cont!r}")
        _require_finite("clean_param", self.clean_param, minimum=0.0, strict=True)
        if self.filter_type not in ("conv", "matched"):
            raise ValueError(f"filter_type must be 'conv' or 'matched', got {self.filter_type!r}")
        if not isinstance(self.clean, bool) or not isinstance(self.accept_deblended_for_focus, bool):
            raise ValueError("clean and accept_deblended_for_focus must be bool")
        kernel = tuple(tuple(float(v) for v in row) for row in self.filter_kernel)
        if not kernel or not kernel[0] or len({len(row) for row in kernel}) != 1:
            raise ValueError("filter_kernel must be a non-empty rectangular 2-D array")
        if len(kernel) % 2 == 0 or len(kernel[0]) % 2 == 0:
            raise ValueError("filter_kernel dimensions must be odd (SEP centres at size//2)")
        if not all(np.isfinite(v) for row in kernel for v in row) or sum(abs(v) for row in kernel for v in row) == 0:
            raise ValueError("filter_kernel must be finite and not all zero")
        object.__setattr__(self, "filter_kernel", kernel)

    def kernel_array(self) -> np.ndarray:
        return np.ascontiguousarray(self.filter_kernel, dtype=np.float32)

    def config(self, library_version: str = SEP_PINNED_VERSION) -> dict:
        """Canonical configuration object (keys frozen by the cross-language contract)."""
        return {
            "method": DETECTION_METHOD_SEP,
            "library": "sep",
            "library_version": library_version,
            "thresh_sigma": float(self.thresh_sigma),
            "noise": "background_rms_map",
            "minarea": int(self.minarea),
            "filter_type": self.filter_type,
            "filter_kernel": [list(row) for row in self.filter_kernel],
            "deblend_nthresh": int(self.deblend_nthresh),
            "deblend_cont": float(self.deblend_cont),
            "clean": bool(self.clean),
            "clean_param": float(self.clean_param),
            "pixstack": int(self.pixstack),
            "accept_deblended_for_focus": bool(self.accept_deblended_for_focus),
        }

    def config_sha256(self, library_version: str = SEP_PINNED_VERSION) -> str:
        return config_sha256(self.config(library_version))


# --------------------------------------------------------------------------- #
# Spatial SEP background.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class SpatialBackground:
    """SEP mesh background B(x,y) and RMS R(x,y) for one full frame.

    ``background_map`` / ``rms_map`` are evaluated once (float32, same shape as
    the frame) and must be reused by every later stage rather than
    re-evaluated or replaced by a scalar. A map is ``None`` when its status is
    not ``measured``: an unavailable level or an unresolved noise scale never
    carries a fabricated array.

    ``global_background`` and ``global_rms`` are SEP's own library-defined
    summaries, not a mean or a pixel median (see :data:`LEVEL_DEFINITION_SEP`
    and :data:`RMS_DEFINITION_SEP`; SEP 1.4.1 background.c L639-653).

    ``detector_mask`` is the single detector-validity mask (non-finite | bad |
    invalid | native saturation) that every SEP call in the same measurement
    must receive. ``source_mask_applied`` reports an optional, separate
    source-contamination mask that was added for the background estimate only.
    """

    background_map: Optional[np.ndarray]
    rms_map: Optional[np.ndarray]
    global_background: Optional[float]
    global_rms: Optional[float]
    params: SepBackgroundParams
    config: dict
    config_sha256: str
    library_version: Optional[str]
    level_status: str
    noise_status: str
    reasons: tuple
    noise_source: str
    usable_fraction: Optional[float]
    units: str = "adu"
    shape: Optional[tuple] = None
    mesh_grid: Optional[tuple] = None  # (nx, ny) as SEP computes it
    meshes_supported: Optional[int] = None
    meshes_total: Optional[int] = None
    detector_mask: Optional[np.ndarray] = None
    source_mask_applied: bool = False
    source_masked_fraction: Optional[float] = None
    backend_error: Optional[str] = None
    method: str = BACKGROUND_METHOD_SEP

    @property
    def level_measured(self) -> bool:
        return self.level_status == STATUS_MEASURED

    @property
    def noise_measured(self) -> bool:
        return self.noise_status == STATUS_MEASURED

    def provenance(self) -> dict:
        """JSON-serialisable summary (no maps) for label metadata and manifests."""
        return {
            "method": self.method,
            "method_version": BACKGROUND_METHOD_SEP_VERSION,
            "library": "sep",
            "library_version": self.library_version,
            "config": dict(self.config),
            "config_sha256": self.config_sha256,
            "units": self.units,
            "level_status": self.level_status,
            "noise_status": self.noise_status,
            "reasons": list(self.reasons),
            "noise_source": self.noise_source,
            "global_background": self.global_background,
            "global_rms": self.global_rms,
            "level_definition": LEVEL_DEFINITION_SEP,
            "rms_definition": RMS_DEFINITION_SEP,
            "usable_fraction": self.usable_fraction,
            "mesh_grid": list(self.mesh_grid) if self.mesh_grid else None,
            "meshes_supported": self.meshes_supported,
            "meshes_total": self.meshes_total,
            "source_mask_applied": self.source_mask_applied,
            "source_masked_fraction": self.source_masked_fraction,
            "backend_error": self.backend_error,
        }


def _unavailable_background(params, library_version, units, shape, reasons, *, usable_fraction=None,
                            detector_mask=None, mesh_grid=None, meshes_supported=None,
                            meshes_total=None, backend_error=None, source_mask_applied=False,
                            source_masked_fraction=None) -> SpatialBackground:
    version = library_version or SEP_PINNED_VERSION
    return SpatialBackground(
        background_map=None,
        rms_map=None,
        global_background=None,
        global_rms=None,
        params=params,
        config=params.config(version),
        config_sha256=params.config_sha256(version),
        library_version=library_version,
        level_status=STATUS_UNAVAILABLE,
        noise_status=STATUS_UNAVAILABLE,
        reasons=tuple(reasons),
        noise_source=NOISE_UNAVAILABLE,
        usable_fraction=usable_fraction,
        units=units,
        shape=shape,
        mesh_grid=mesh_grid,
        meshes_supported=meshes_supported,
        meshes_total=meshes_total,
        detector_mask=detector_mask,
        source_mask_applied=source_mask_applied,
        source_masked_fraction=source_masked_fraction,
        backend_error=backend_error,
    )


def _mesh_support(excluded: np.ndarray, bw: int, bh: int) -> tuple[int, int, int, int]:
    """Replicate SEP's per-mesh support rule on the exclusion mask.

    Returns ``(nx, ny, supported, total)``. SEP's grid is
    ``nx = (w-1)//bw + 1``; the last column/row of meshes is partial and uses
    its actual size; a mesh is discarded when its usable count is below half
    of that actual area (background.c backstat, BACK_MINGOODFRAC).
    """
    h, w = excluded.shape
    nx = (w - 1) // bw + 1
    ny = (h - 1) // bh + 1
    usable = ~excluded
    supported = 0
    for j in range(ny):
        y0, y1 = j * bh, min(h, (j + 1) * bh)
        row = usable[y0:y1]
        for i in range(nx):
            x0, x1 = i * bw, min(w, (i + 1) * bw)
            area = (y1 - y0) * (x1 - x0)
            if float(row[:, x0:x1].sum()) >= float(area) * SEP_BACK_MINGOODFRAC:
                supported += 1
    return nx, ny, supported, nx * ny


def estimate_spatial_background(
    image: np.ndarray,
    *,
    detector_mask: Optional[np.ndarray] = None,
    params: SepBackgroundParams = SepBackgroundParams(),
    source_mask: Optional[np.ndarray] = None,
    units: str = "adu",
) -> SpatialBackground:
    """SEP spatial background and RMS maps of a full frame, with explicit status.

    ``detector_mask`` (bool, ``True`` = do not use) marks bad, invalid and
    native-saturated pixels; non-finite pixels are always added to it. The
    combined mask is returned on the result so that extraction and aperture
    calls use exactly the same support.

    ``source_mask`` is an optional *source-contamination* mask (for example
    dilated detections from a first pass). It is excluded from the background
    estimate only and is never merged into ``detector_mask``.

    Unavailable / unresolved outcomes (never silently replaced by a number):

    * ``malformed_geometry``: not 2-D, an empty axis, or a mask of another shape;
    * ``no_usable_pixels``: every pixel is non-finite or detector-masked;
    * ``frame_smaller_than_mesh_grid``: fewer than ``min_meshes_per_axis``
      full meshes on an axis;
    * ``insufficient_support``: no mesh has the >= 50% usable pixels SEP
      requires. SEP 1.4.1 would then return background 0.0 and RMS 1.0 with
      ``RETURN_OK`` (background.c filterback), which are fabricated values;
    * ``backend_failure``: SEP raised;
    * ``zero_empirical_scatter`` (noise ``unresolved``, level still measured):
      R(x,y) is exactly zero on a usable pixel (e.g. a quantized plateau), or
      SEP's global RMS is not positive. SEP replaces a non-positive global RMS
      by the median positive mesh sigma or by 1.0 (background.c L645-653);
      neither value is consumed here;
    * ``invalid_rms_map`` (noise ``unresolved``): R(x,y) is negative or
      non-finite on a usable pixel — SEP's bicubic spline undershooting between
      meshes of very different sigma.

    ``meshes_interpolated`` is informational: some meshes lacked support and
    SEP filled them from their nearest valid neighbours.
    """
    image = np.asarray(image)
    if image.ndim != 2 or image.shape[0] == 0 or image.shape[1] == 0:
        return _unavailable_background(params, None, units, tuple(image.shape), [REASON_MALFORMED_GEOMETRY])
    shape = tuple(int(v) for v in image.shape)
    for supplied in (detector_mask, source_mask):
        if supplied is not None and np.asarray(supplied).shape != image.shape:
            return _unavailable_background(params, None, units, shape, [REASON_MALFORMED_GEOMETRY])

    sep = require_sep()
    version = sep.__version__
    data = np.asarray(image, dtype=np.float32)
    mask = ~np.isfinite(data)
    if detector_mask is not None:
        mask |= np.asarray(detector_mask, dtype=bool)
    usable_count = int((~mask).sum())
    usable_fraction = usable_count / float(mask.size)
    excluded = mask.copy()
    source_masked_fraction = None
    if source_mask is not None:
        source = np.asarray(source_mask, dtype=bool) & ~mask
        excluded |= source
        source_masked_fraction = float(source.sum()) / float(max(usable_count, 1))
    common = dict(
        usable_fraction=usable_fraction,
        detector_mask=mask,
        source_mask_applied=source_mask is not None,
        source_masked_fraction=source_masked_fraction,
    )

    if usable_count == 0:
        return _unavailable_background(params, version, units, shape, [REASON_NO_USABLE_PIXELS], **common)
    h, w = shape
    n = params.min_meshes_per_axis
    if w < n * params.bw or h < n * params.bh:
        return _unavailable_background(
            params, version, units, shape, [REASON_FRAME_SMALLER_THAN_MESH_GRID], **common
        )
    nx, ny, supported, total = _mesh_support(excluded, params.bw, params.bh)
    mesh = dict(mesh_grid=(nx, ny), meshes_supported=supported, meshes_total=total)
    if supported == 0:
        return _unavailable_background(
            params, version, units, shape, [REASON_INSUFFICIENT_SUPPORT], **common, **mesh
        )

    # Masked pixels are excluded by SEP, but NaN/inf must not reach the
    # histogram code at all (backhisto has no NaN guard), so they are zeroed.
    clean = np.ascontiguousarray(np.where(excluded, np.float32(0.0), data), dtype=np.float32)
    try:
        bkg = sep.Background(
            clean,
            mask=np.ascontiguousarray(excluded, dtype=np.uint8),
            maskthresh=0.0,
            bw=int(params.bw),
            bh=int(params.bh),
            fw=int(params.fw),
            fh=int(params.fh),
            fthresh=float(params.fthresh),
        )
        back = np.ascontiguousarray(bkg.back(dtype=np.float32))
        rms = np.ascontiguousarray(bkg.rms(dtype=np.float32))
        globalback = float(bkg.globalback)
        globalrms = float(bkg.globalrms)
    except Exception as exc:  # SEP raises plain Exception / MemoryError
        return _unavailable_background(
            params, version, units, shape, [REASON_BACKEND_FAILURE], backend_error=str(exc), **common, **mesh
        )

    reasons = []
    if supported < total:
        reasons.append(REASON_MESHES_INTERPOLATED)
    usable = ~mask
    level_ok = bool(np.isfinite(back[usable]).all()) and np.isfinite(globalback)
    if not level_ok:
        return _unavailable_background(
            params, version, units, shape, [REASON_BACKEND_FAILURE], backend_error="non-finite background map",
            **common, **mesh,
        )
    rms_usable = rms[usable]
    finite_rms = np.isfinite(rms_usable)
    if (finite_rms & (rms_usable == 0)).any() or not (np.isfinite(globalrms) and globalrms > 0):
        reasons.append(REASON_ZERO_EMPIRICAL_SCATTER)
    if (~finite_rms).any() or (finite_rms & (rms_usable < 0)).any():
        reasons.append(REASON_INVALID_RMS_MAP)
    noise_ok = REASON_ZERO_EMPIRICAL_SCATTER not in reasons and REASON_INVALID_RMS_MAP not in reasons
    return SpatialBackground(
        background_map=back,
        rms_map=rms if noise_ok else None,
        global_background=globalback,
        global_rms=globalrms if noise_ok else None,
        params=params,
        config=params.config(version),
        config_sha256=params.config_sha256(version),
        library_version=version,
        level_status=STATUS_MEASURED,
        noise_status=STATUS_MEASURED if noise_ok else STATUS_UNRESOLVED,
        reasons=tuple(reasons),
        noise_source=NOISE_EMPIRICAL_MESH_RMS if noise_ok else NOISE_UNAVAILABLE,
        units=units,
        shape=shape,
        **common,
        **mesh,
    )


# --------------------------------------------------------------------------- #
# Rejection taxonomy — why a candidate star is not usable for a width label.
# --------------------------------------------------------------------------- #
class Rejection(str, Enum):
    SATURATED = "saturated"
    MASKED_SUPPORT = "masked_support"
    EDGE_TRUNCATED = "edge_truncated"
    LOW_SNR = "low_snr"
    NOISE_UNRESOLVED = "noise_unresolved"
    TOO_FEW_PIXELS = "too_few_pixels"
    ELONGATED = "elongated"
    BLENDED = "blended"
    FIT_FAILED = "fit_failed"


@dataclass
class StarMeasurement:
    """Per-star measurement from the custom crop path. ``valid`` gates whether
    it contributes a label. ``snr`` is ``None`` when the crop noise is
    unresolved."""

    x: float
    y: float
    flux: float
    peak: float
    snr: Optional[float]
    hfr_px: Optional[float]
    fwhm_px: Optional[float]  # model given by fwhm_model; None if not run / failed
    moffat_beta: Optional[float]
    ellipticity: float
    valid: bool
    rejections: list[str]
    fwhm_model: Optional[str] = None
    fwhm_fallback_reason: Optional[str] = None
    fit_converged: bool = False


@dataclass
class FrameLabel:
    """Robust frame-level label aggregated over accepted stars.

    ``sky_median_adu`` / ``sky_rms_adu`` are legacy names: on the SEP path they
    hold SEP's ``globalback`` / ``globalrms`` (medians of mesh values, see
    ``background['level_definition']``), on the crop path the caller's crop
    estimate. ``None`` means unavailable or unresolved, never zero.

    ``fwhm_px_median`` pools **only** Moffat fits; Gaussian-moment fallbacks
    are counted (``fwhm_fallback_count``) and summarised separately.
    """

    hfr_px_median: Optional[float]
    hfr_px_mad: Optional[float]
    fwhm_px_median: Optional[float]
    fwhm_px_mad: Optional[float]
    ellipticity_median: Optional[float]
    star_count_detected: int
    star_count_used: int
    sky_median_adu: Optional[float]
    sky_rms_adu: Optional[float]
    saturated_star_fraction: float
    label_schema_version: str = LABEL_SCHEMA_VERSION
    label_method: str = ""
    status: str = STATUS_MEASURED
    reasons: list = field(default_factory=list)
    hfd_px_median: Optional[float] = None
    fwhm_moffat_count: int = 0
    fwhm_fallback_count: int = 0
    fwhm_fallback_px_median: Optional[float] = None
    units: str = "adu"
    width_definition: dict = field(default_factory=dict)
    snr_definition: Optional[str] = None
    gain_e_per_adu: Optional[float] = None
    saturation_gate: Optional[str] = None
    saturation_threshold: Optional[float] = None
    rejection_counts: dict = field(default_factory=dict)
    sep_flag_counts: dict = field(default_factory=dict)
    background: dict = field(default_factory=dict)
    detection: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Custom crop background (NOT SEP).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CropBackground:
    """Result of the custom crop estimator with explicit status.

    ``level`` is the median of the retained sample; ``rms`` the population
    standard deviation of the retained sample. ``noise_status`` is
    ``unresolved`` (reason ``zero_empirical_scatter``) when the robust scale
    is zero, i.e. a quantized plateau where more than half of the usable
    pixels share one value; ``rms`` is then ``None``.
    """

    level: Optional[float]
    rms: Optional[float]
    level_status: str
    noise_status: str
    reasons: tuple
    n_used: int
    method: str = BACKGROUND_METHOD_CROP


def estimate_background_detailed(image: np.ndarray, sigma: float = 3.0, iters: int = 5) -> CropBackground:
    """Custom iterative MAD clip over one crop (``python_crop_iterative_clip``).

    Not SEP and not a mesh model: one global estimate for the whole crop, for
    small, fairly uniform autofocus crops only. Each pass keeps pixels within
    ``sigma * 1.4826 * MAD`` of the median; it stops when nothing is clipped,
    when fewer than 8 samples would remain, after ``iters`` passes, or when
    the MAD is zero.

    A zero MAD means the robust scale is unresolved. The previous code then
    returned the standard deviation of the *unclipped* crop, stars included —
    an inflated number with no defined meaning. It is now reported as
    unresolved instead.
    """
    data = np.asarray(image)
    data = data[np.isfinite(data)].astype(np.float64).ravel()
    if data.size == 0:
        return CropBackground(None, None, STATUS_UNAVAILABLE, STATUS_UNAVAILABLE, (REASON_NO_USABLE_PIXELS,), 0)
    for _ in range(iters):
        med = np.median(data)
        mad = np.median(np.abs(data - med)) * 1.4826
        if mad <= 0:
            return CropBackground(
                float(med), None, STATUS_MEASURED, STATUS_UNRESOLVED, (REASON_ZERO_EMPIRICAL_SCATTER,), int(data.size)
            )
        keep = np.abs(data - med) <= sigma * mad
        if keep.all() or keep.sum() < 8:
            break
        data = data[keep]
    med = float(np.median(data))
    rms = float(np.std(data)) if data.size > 1 else 0.0
    if not rms > 0:
        return CropBackground(med, None, STATUS_MEASURED, STATUS_UNRESOLVED, (REASON_ZERO_EMPIRICAL_SCATTER,), int(data.size))
    return CropBackground(med, rms, STATUS_MEASURED, STATUS_MEASURED, (), int(data.size))


def estimate_background(image: np.ndarray, sigma: float = 3.0, iters: int = 5) -> tuple[float, float]:
    """Scalar form of :func:`estimate_background_detailed` (custom, not SEP).

    Returns ``(level, rms)``. When the noise is unresolved (zero MAD plateau)
    ``rms`` is ``0.0`` — the empirical scatter actually measured, as the Rust
    helper reports it — never the inflated std of the unclipped crop. When no
    pixel is finite both values are ``nan``. Callers that must distinguish
    these states use the detailed form; the crop measurement functions here do.
    """
    result = estimate_background_detailed(image, sigma, iters)
    if result.level is None:
        return float("nan"), float("nan")
    return result.level, (result.rms if result.rms is not None else 0.0)


# --------------------------------------------------------------------------- #
# Source detection on a crop or frame (custom, for the crop path only).
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
    """Custom connected-component detector (``python_crop_connected_components``).

    Pixels above one global threshold ``sky + threshold_sigma * sky_rms`` are
    grouped with plain scipy.ndimage labelling; returns ``(crop, cx, cy)`` per
    component. NO mesh background, deblending, matched filter, or cosmic-ray
    rejection — do not mistake it for SEP; the SEP path is
    :func:`measure_frame_sep`. Crops truncated by the frame edge are dropped.

    When the noise scale is unresolved (``sky_rms <= 0`` or the crop estimator
    reports a zero-MAD plateau) no threshold can be stated in sigma units, so
    nothing is detected (empty list) rather than thresholding at an invented
    epsilon.
    """
    from scipy import ndimage

    image = np.asarray(image, dtype=np.float64)
    if sky is None or sky_rms is None:
        estimate = estimate_background_detailed(image)
        if estimate.noise_status != STATUS_MEASURED:
            return []
        sky, sky_rms = estimate.level, estimate.rms
    if not (np.isfinite(sky) and np.isfinite(sky_rms) and sky_rms > 0):
        return []
    thresh = sky + threshold_sigma * sky_rms
    labels, n = ndimage.label(image > thresh)
    if n == 0:
        return []

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
# Curve-of-growth half-flux radius (custom crop path).
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
    """Aperture-bounded half-flux radius (px) for a caller-supplied scalar sky
    (``python_signed_curve_of_growth``; custom, not ``sep.flux_radius``).

    Two properties that the previous implementation got wrong and that this one
    guarantees (see the astronomical-quality-metrics research note):

    * **No noise pedestal.** The sky-subtracted flux is integrated *without*
      clipping negatives to zero, so background noise cancels in the mean.
    * **Bounded aperture.** Integration is limited to a finite radius adapted to
      the star (~4x an initial HFR estimate), so enlarging the crop cannot
      accumulate more noise-only annuli.

    ``sky`` is used as given: this function never re-estimates it. ``sky_rms``
    only sets the 3-sigma core threshold used to *locate* the centroid; when it
    is ``None`` or not positive the centroid uses all positive residual.

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
        thr = 3.0 * sky_rms if sky_rms is not None and np.isfinite(sky_rms) and sky_rms > 0 else 0.0
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
# Circular Moffat fit -> fitted FWHM + beta, with explicit model/fallback.
# --------------------------------------------------------------------------- #
FWHM_MODEL_MOFFAT = "moffat"
FWHM_MODEL_GAUSSIAN_FALLBACK = "gaussian_moment_fallback"


@dataclass(frozen=True)
class MoffatFit:
    """Outcome of :func:`fit_moffat` on a background-subtracted residual.

    ``fwhm_model`` is ``"moffat"`` for a converged fit, ``"gaussian_moment_fallback"``
    when the fit failed and the flux-weighted second moment was used instead
    (``fallback_reason`` says why), or ``None`` when neither produced a width.
    ``residual_offset`` is the fitted constant when ``offset_mode == "free"``;
    with ``"fixed_zero"`` the model has no offset term and it is ``None``.
    """

    fwhm_px: Optional[float]
    beta: Optional[float]
    alpha_px: Optional[float]
    amplitude: Optional[float]
    residual_offset: Optional[float]
    offset_mode: str
    fwhm_model: Optional[str]
    fit_converged: bool
    fallback_reason: Optional[str]


def _moffat_radial(r, amp, alpha, beta, offset):
    return offset + amp * (1.0 + (r / alpha) ** 2) ** (-beta)


def _gaussian_moment_fwhm_residual(residual: np.ndarray, valid: np.ndarray) -> Optional[float]:
    h, w = residual.shape
    ys, xs = np.mgrid[0:h, 0:w]
    f = np.where(valid, np.clip(residual, 0.0, None), 0.0)
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


def fit_moffat(
    residual: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    fit_offset: bool = False,
) -> MoffatFit:
    """Fit a circular Moffat to an already background-subtracted crop.

    ``FWHM = 2 * alpha * sqrt(2**(1/beta) - 1)``. Pixels that are masked or
    non-finite are excluded from both the centroid and the fit. With
    ``fit_offset=False`` the model has no constant term: the caller's local
    background (e.g. B(x,y)) is taken as correct. With ``fit_offset=True`` a
    constant residual offset is fitted (bounded to +-peak) and reported.
    """
    residual = np.asarray(residual, dtype=np.float64)
    offset_mode = "free" if fit_offset else "fixed_zero"
    valid = np.isfinite(residual)
    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool)

    def fallback(reason: str) -> MoffatFit:
        fwhm = _gaussian_moment_fwhm_residual(residual, valid)
        return MoffatFit(
            fwhm, None, None, None, None, offset_mode,
            FWHM_MODEL_GAUSSIAN_FALLBACK if fwhm is not None else None, False, reason,
        )

    h, w = residual.shape
    ys, xs = np.mgrid[0:h, 0:w]
    f = np.where(valid, np.clip(residual, 0.0, None), 0.0)
    total = f.sum()
    if total <= 0:
        return MoffatFit(None, None, None, None, None, offset_mode, None, False, "no_positive_flux")
    cx = (xs * f).sum() / total
    cy = (ys * f).sum() / total
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)[valid]
    z = residual[valid]

    if not _HAVE_SCIPY:
        return fallback("scipy_unavailable")

    amp0 = float(z.max())
    if amp0 <= 0:
        return MoffatFit(None, None, None, None, None, offset_mode, None, False, "no_positive_flux")
    if fit_offset:
        p0 = np.array([amp0, 2.0, 2.5, 0.0])
        bounds = ([0.0, 1e-3, 0.6, -abs(amp0)], [np.inf, max(h, w), 20.0, abs(amp0)])
    else:
        p0 = np.array([amp0, 2.0, 2.5])
        bounds = ([0.0, 1e-3, 0.6], [np.inf, max(h, w), 20.0])

    def resid(p):
        amp, alpha, beta = p[0], max(p[1], 1e-3), max(p[2], 0.5)
        offset = p[3] if fit_offset else 0.0
        return _moffat_radial(r, amp, alpha, beta, offset) - z

    try:
        sol = least_squares(resid, p0, bounds=bounds, max_nfev=200)
    except Exception:
        return fallback("least_squares_exception")
    if not sol.success:
        return fallback("least_squares_not_converged")

    amp, alpha, beta = (float(v) for v in sol.x[:3])
    if beta <= 0.5 or alpha <= 0 or not np.isfinite(alpha) or not np.isfinite(beta):
        return fallback("invalid_parameters")
    fwhm = float(2.0 * alpha * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    offset = float(sol.x[3]) if fit_offset else None
    return MoffatFit(fwhm, beta, alpha, amp, offset, offset_mode, FWHM_MODEL_MOFFAT, True, None)


def fit_moffat_fwhm(crop: np.ndarray, sky: float) -> tuple[Optional[float], Optional[float]]:
    """Legacy crop-path wrapper: fit ``crop - sky`` with a free constant offset
    and return ``(fwhm_px, beta)``.

    On a failed fit this returns the Gaussian-moment width with ``beta=None``;
    callers that must tell the two apart use :func:`fit_moffat`. Note this is a
    genuinely different quantity from ``2 * HFR`` unless the PSF is Gaussian
    (beta -> inf)."""
    result = fit_moffat(np.asarray(crop, dtype=np.float64) - sky, fit_offset=True)
    return result.fwhm_px, (result.beta if result.fwhm_model == FWHM_MODEL_MOFFAT else None)


def _gaussian_moment_fwhm(crop: np.ndarray, sky: float) -> Optional[float]:
    """FWHM from the flux-weighted radial second moment assuming a Gaussian
    core: FWHM = 2*sqrt(2 ln 2) * sigma."""
    residual = np.asarray(crop, dtype=np.float64) - sky
    return _gaussian_moment_fwhm_residual(residual, np.isfinite(residual))


def _count_core_peaks(crop: np.ndarray, sky: float, sky_rms: float, min_sep: int = 3) -> int:
    """Count distinct local maxima above sky + 5*rms — a cheap blend *rejection
    heuristic* for the crop path, not a deblender. Requires a resolved
    ``sky_rms > 0``."""
    try:
        from scipy.ndimage import maximum_filter
    except ImportError:  # pragma: no cover
        return 1
    crop = np.asarray(crop, dtype=np.float64)
    thresh = sky + 5.0 * sky_rms
    footprint = 2 * min_sep + 1
    is_peak = (crop == maximum_filter(crop, size=footprint)) & (crop > thresh)
    # Collapse plateau peaks that touch into a single detection.
    from scipy.ndimage import label as _label

    _, n = _label(is_peak)
    return int(n)


# --------------------------------------------------------------------------- #
# Per-star measurement with source-validity gates (custom crop path).
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

    Custom crop path: one scalar sky for the crop (the caller's, or the
    ``python_crop_iterative_clip`` estimate). The gates reject saturated,
    edge-truncated, low-SNR, blended/elongated stars *before* they can bias the
    aggregate HFR. When the crop noise is unresolved (zero-MAD plateau, or a
    caller-supplied ``sky_rms <= 0``) the SNR, core-count and blend gates
    cannot be evaluated in sigma units: the star is rejected with
    ``noise_unresolved`` and ``snr`` is ``None`` instead of dividing by an
    invented epsilon."""
    crop = np.asarray(crop, dtype=np.float64)
    h, w = crop.shape
    rejections: list[str] = []
    if sky is None or sky_rms is None:
        estimate = estimate_background_detailed(crop)
        if estimate.level is None:
            return StarMeasurement(
                w / 2.0, h / 2.0, 0.0, float("nan"), None, None, None, None, 0.0, False,
                [Rejection.NOISE_UNRESOLVED.value],
            )
        sky, sky_rms = estimate.level, estimate.rms
    noise_resolved = sky_rms is not None and np.isfinite(sky_rms) and sky_rms > 0

    signal = np.clip(crop - sky, 0.0, None)
    total = float(signal.sum())
    peak = float(crop.max())

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

    snr: Optional[float] = None
    if noise_resolved:
        # Aperture SNR proxy (sky-noise only; no gain/read-noise model — hence a
        # *proxy*): sky-subtracted flux summed inside a fixed core aperture, over
        # the propagated sky noise. The sum is *unclipped* so that zero-mean sky
        # noise cancels.
        r_ap = max(4.0, min(h, w) / 8.0)
        aperture = ((xs - cx) ** 2 + (ys - cy) ** 2) <= r_ap**2
        ap_signal = float((crop[aperture] - sky).sum())
        n_ap = float(aperture.sum())
        core_pixels = int((signal[aperture] > 3.0 * sky_rms).sum())
        snr = ap_signal / (sky_rms * float(np.sqrt(max(n_ap, 1.0))))
        if snr < min_snr:
            rejections.append(Rejection.LOW_SNR.value)
        if core_pixels < 5:
            rejections.append(Rejection.TOO_FEW_PIXELS.value)
    else:
        rejections.append(Rejection.NOISE_UNRESOLVED.value)

    ellipticity, elongation = second_moment_shape(crop, sky)
    if ellipticity > max_ellipticity:
        rejections.append(Rejection.ELONGATED.value)

    # Blend gate: a clean star has one core peak. Two or more well-separated
    # peaks above 5-sigma mean a close double whose blended profile would give a
    # spuriously broad HFR — reject it (SEP deblends; the crop path detects and
    # drops it).
    if noise_resolved and _count_core_peaks(crop, sky, sky_rms) >= 2:
        rejections.append(Rejection.BLENDED.value)

    hfr = curve_of_growth_hfr(crop, sky, sky_rms=sky_rms if noise_resolved else None, centroid=(cx, cy))
    fit = None
    if fit_fwhm and not rejections:
        fit = fit_moffat(crop - sky, fit_offset=True)
        if fit.fwhm_px is None:
            rejections.append(Rejection.FIT_FAILED.value)

    valid = len(rejections) == 0 and hfr is not None and hfr > 0
    return StarMeasurement(
        x=cx,
        y=cy,
        flux=total,
        peak=peak,
        snr=None if snr is None else float(snr),
        hfr_px=hfr,
        fwhm_px=fit.fwhm_px if fit else None,
        moffat_beta=fit.beta if fit else None,
        ellipticity=float(ellipticity),
        valid=valid,
        rejections=rejections,
        fwhm_model=fit.fwhm_model if fit else None,
        fwhm_fallback_reason=fit.fallback_reason if fit else None,
        fit_converged=bool(fit.fit_converged) if fit else False,
    )


def _robust_median_mad(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)) * 1.4826)
    return med, mad


def _fwhm_summary(items) -> dict:
    """Pool Moffat FWHM only; summarise Gaussian-moment fallbacks separately."""
    moffat = [i.fwhm_px for i in items if i.fwhm_px is not None and i.fwhm_model == FWHM_MODEL_MOFFAT]
    fallback = [i.fwhm_px for i in items if i.fwhm_px is not None and i.fwhm_model == FWHM_MODEL_GAUSSIAN_FALLBACK]
    med, mad = _robust_median_mad(moffat)
    return {
        "fwhm_px_median": med,
        "fwhm_px_mad": mad,
        "fwhm_moffat_count": len(moffat),
        "fwhm_fallback_count": len(fallback),
        "fwhm_fallback_px_median": _robust_median_mad(fallback)[0],
    }


CROP_WIDTH_DEFINITION = {
    "hfr": "python_signed_curve_of_growth: radius enclosing half of the unclipped sky-subtracted flux "
    "inside a bounded aperture (clip(4*HFR_est, 6, crop/2-1) px), scalar crop sky",
    "hfd": "2 * HFR",
    "fwhm": "fitted circular Moffat with a free constant offset (fwhm_model='moffat'); "
    "Gaussian second-moment fallbacks are counted separately and not pooled",
}


def aggregate_frame(
    measurements: list[StarMeasurement],
    sky_median: Optional[float],
    sky_rms: Optional[float],
) -> FrameLabel:
    """Robustly pool accepted crop-path stars into a frame label (median + MAD).
    This is the transparent robust pooling that Bosse et al. (2018) recommend
    *before* any learned patch weighting."""
    used = [m for m in measurements if m.valid]
    hfrs = [m.hfr_px for m in used if m.hfr_px is not None]
    ells = [m.ellipticity for m in used]
    saturated = sum(1 for m in measurements if Rejection.SATURATED.value in m.rejections)

    hfr_med, hfr_mad = _robust_median_mad(hfrs)
    ell_med, _ = _robust_median_mad(ells)
    fwhm = _fwhm_summary(used)

    return FrameLabel(
        hfr_px_median=hfr_med,
        hfr_px_mad=hfr_mad,
        fwhm_px_median=fwhm["fwhm_px_median"],
        fwhm_px_mad=fwhm["fwhm_px_mad"],
        ellipticity_median=ell_med,
        star_count_detected=len(measurements),
        star_count_used=len(used),
        sky_median_adu=None if sky_median is None else float(sky_median),
        sky_rms_adu=None if sky_rms is None else float(sky_rms),
        saturated_star_fraction=(saturated / len(measurements)) if measurements else 0.0,
        label_method=f"{BACKGROUND_METHOD_CROP}+{HFR_METHOD_CROP}",
        status=STATUS_MEASURED if hfr_med is not None else STATUS_UNAVAILABLE,
        reasons=[] if hfr_med is not None else ["no_valid_sources"],
        hfd_px_median=None if hfr_med is None else 2.0 * hfr_med,
        fwhm_moffat_count=fwhm["fwhm_moffat_count"],
        fwhm_fallback_count=fwhm["fwhm_fallback_count"],
        fwhm_fallback_px_median=fwhm["fwhm_fallback_px_median"],
        width_definition=dict(CROP_WIDTH_DEFINITION),
        snr_definition="crop aperture flux over sky_rms*sqrt(npix): background-noise-only proxy",
        rejection_counts=dict(Counter(r for m in measurements for r in m.rejections)),
    )


# --------------------------------------------------------------------------- #
# Real SEP path: spatial mesh background + local-noise extraction with
# deblending + aperture-normalised flux_radius. This is the labeler the dataset
# builder and the session importer use on full frames. It requires the pinned
# SEP (MissingPrerequisite otherwise); there is no silent fallback to the crop
# helpers above.
# --------------------------------------------------------------------------- #
SEP_HFR_APERTURE_A_FACTOR = 6.0
SEP_HFR_APERTURE_MIN_PX = 6.0
SEP_FLUX_APERTURE_A_FACTOR = 3.0
SEP_FLUX_APERTURE_MIN_PX = 4.0
SEP_SUBPIX = 5
MOFFAT_CROP_HALF_PX = 24

SEP_WIDTH_DEFINITION = {
    "hfr": "sep.flux_radius(frac=0.5, normflux=None, subpix=5) on data - B(x,y): radius at which the "
    "cumulative flux reaches half of the flux inside rmax = max(6*a, 6) px (a = SEP isophotal "
    "semi-major axis); first crossing, linear interpolation over 64 annuli",
    "hfd": "2 * HFR",
    "fwhm": "fitted circular Moffat on the 48x48 px crop of data - B(x,y) with the constant offset fixed "
    "at zero by default (fwhm_model='moffat'); detector-masked pixels excluded. Gaussian second-moment "
    "fallbacks are labelled 'gaussian_moment_fallback', counted separately and never pooled",
}
SNR_DEFINITION_BACKGROUND_ONLY = (
    "background_noise_only_proxy: sep.sum_circle flux in r = max(3*a, 4) px of data - B(x,y) over its "
    "error from R(x,y) alone; gain unknown, so source Poisson noise is omitted"
)
SNR_DEFINITION_WITH_GAIN = (
    "aperture_snr_rms_map_plus_source_poisson: sep.sum_circle flux in r = max(3*a, 4) px of data - B(x,y) "
    "over sqrt(sum R(x,y)^2 + flux/gain); R(x,y) is empirical and already contains read noise, so no "
    "detector term is added"
)

SEP_FLAG_NAMES = (
    ("OBJ_MERGED", "obj_merged"),
    ("OBJ_TRUNC", "obj_trunc"),
    ("OBJ_SINGU", "obj_singu"),
)
SEP_APER_FLAG_NAMES = (
    ("APER_TRUNC", "aper_trunc"),
    ("APER_HASMASKED", "aper_hasmasked"),
    ("APER_ALLMASKED", "aper_allmasked"),
    ("APER_NONPOSITIVE", "aper_nonpositive"),
)


@dataclass
class SepSource:
    """One SEP detection and its focus-label decision.

    ``peak`` is the raw (not background-subtracted) peak value inside the
    measurement footprint over unmasked pixels. ``flux`` is the background-
    subtracted ``sep.sum_circle`` aperture flux. ``snr`` follows the frame's
    ``snr_definition``.
    """

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
    hfd_px: Optional[float] = None
    fwhm_model: Optional[str] = None
    fwhm_fallback_reason: Optional[str] = None
    fit_converged: bool = False
    moffat_beta: Optional[float] = None
    moffat_offset_mode: Optional[str] = None
    moffat_residual_offset: Optional[float] = None
    sep_flag: int = 0
    radius_flag: int = 0
    aperture_flag: int = 0
    deblended: bool = False
    saturation_evidence: list = field(default_factory=list)
    hfr_aperture_radius_px: Optional[float] = None
    flux_aperture_radius_px: Optional[float] = None
    local_background: Optional[float] = None
    local_rms: Optional[float] = None


def _disk_footprint(shape, cx: float, cy: float, radius: float):
    """Pixels whose square overlaps a disk of ``radius`` about (cx, cy).

    SEP coordinates: (0, 0) is the centre of the first pixel, so pixel (i, j)
    covers [i-0.5, i+0.5] x [j-0.5, j+0.5]. A pixel touches the disk when the
    nearest point of its square is closer than ``radius`` — the support SEP's
    aperture integrators can read.
    """
    h, w = shape
    x0 = max(0, int(np.floor(cx - radius - 0.5)))
    x1 = min(w, int(np.ceil(cx + radius + 0.5)) + 1)
    y0 = max(0, int(np.floor(cy - radius - 0.5)))
    y1 = min(h, int(np.ceil(cy + radius + 0.5)) + 1)
    if x1 <= x0 or y1 <= y0:
        return (slice(0, 0), slice(0, 0)), np.zeros((0, 0), dtype=bool)
    dx = np.maximum(np.abs(np.arange(x0, x1) - cx) - 0.5, 0.0)
    dy = np.maximum(np.abs(np.arange(y0, y1) - cy) - 0.5, 0.0)
    inside = (dy[:, None] ** 2 + dx[None, :] ** 2) < radius * radius
    return (slice(y0, y1), slice(x0, x1)), inside


def _combine_detector_mask(shape, data, detector_mask, native_saturation_mask):
    """Return ``(combined, native_saturation, malformed)`` boolean masks."""
    malformed = False
    combined = ~np.isfinite(data)
    saturated = np.zeros(shape, dtype=bool)
    if detector_mask is not None:
        dm = np.asarray(detector_mask)
        if dm.shape != shape:
            malformed = True
        else:
            combined |= dm.astype(bool)
    if native_saturation_mask is not None:
        sm = np.asarray(native_saturation_mask)
        if sm.shape != shape:
            malformed = True
        else:
            saturated = sm.astype(bool)
            combined |= saturated
    return combined, saturated, malformed


def _saturation_gate(saturation_level, native_saturation_mask) -> tuple[str, Optional[float]]:
    """Mirror of the Rust SaturationGate states. Unknown stays unknown."""
    threshold = None
    if saturation_level is not None and np.isfinite(saturation_level):
        threshold = float(saturation_level)
    if native_saturation_mask is not None:
        return ("native_mask_applied" if threshold is None else "native_mask_and_threshold_applied"), threshold
    if threshold is not None:
        return "applied", threshold
    return "not_performed", None


def _empty_frame_label(background: SpatialBackground, detection: dict, reasons, units, gate, threshold,
                       gain, label_method) -> FrameLabel:
    return FrameLabel(
        hfr_px_median=None,
        hfr_px_mad=None,
        fwhm_px_median=None,
        fwhm_px_mad=None,
        ellipticity_median=None,
        star_count_detected=0,
        star_count_used=0,
        sky_median_adu=background.global_background,
        sky_rms_adu=background.global_rms,
        saturated_star_fraction=0.0,
        label_method=label_method,
        status=STATUS_UNAVAILABLE,
        reasons=list(reasons),
        units=units,
        width_definition=dict(SEP_WIDTH_DEFINITION),
        snr_definition=SNR_DEFINITION_WITH_GAIN if gain is not None else SNR_DEFINITION_BACKGROUND_ONLY,
        gain_e_per_adu=gain,
        saturation_gate=gate,
        saturation_threshold=threshold,
        background=background.provenance(),
        detection=detection,
    )


def _sep_extract(sep, data_sub, rms, mask, params: SepDetectionParams):
    previous = sep.get_extract_pixstack()
    try:
        sep.set_extract_pixstack(int(params.pixstack))
        return sep.extract(
            data_sub,
            float(params.thresh_sigma),
            err=rms,
            mask=mask,
            maskthresh=0.0,
            minarea=int(params.minarea),
            filter_kernel=params.kernel_array(),
            filter_type=params.filter_type,
            deblend_nthresh=int(params.deblend_nthresh),
            deblend_cont=float(params.deblend_cont),
            clean=bool(params.clean),
            clean_param=float(params.clean_param),
            segmentation_map=True,
        )
    finally:
        sep.set_extract_pixstack(previous)


def measure_frame_sep(
    image: np.ndarray,
    *,
    saturation_level: Optional[float] = None,
    detector_mask: Optional[np.ndarray] = None,
    native_saturation_mask: Optional[np.ndarray] = None,
    background_params: SepBackgroundParams = SepBackgroundParams(),
    detection_params: SepDetectionParams = SepDetectionParams(),
    gain_e_per_adu: Optional[float] = None,
    max_ellipticity: float = 0.35,
    min_snr: float = 5.0,
    fit_fwhm: bool = True,
    moffat_fit_offset: bool = False,
    source_mask_background: bool = False,
    source_mask_dilation_px: int = 3,
    units: str = "adu",
    thresh_sigma: Optional[float] = None,
    deblend_cont: Optional[float] = None,
    min_area: Optional[int] = None,
) -> tuple[list[SepSource], FrameLabel]:
    """Extract and measure every source in a full frame with pinned SEP.

    Returns ``(sources, frame_label)``.

    Pipeline (one detector mask throughout):

    1. Detector mask = non-finite | ``detector_mask`` (bad/invalid pixels) |
       ``native_saturation_mask``. The same mask goes to ``sep.Background``,
       ``sep.extract``, ``sep.flux_radius`` and ``sep.sum_circle``.
    2. :func:`estimate_spatial_background` gives B(x,y) and R(x,y). If the
       level is unavailable or the noise unresolved, nothing is detected and
       the label is ``unavailable`` with the background's reasons.
    3. Optional (``source_mask_background``, off by default): a first
       extraction's segmentation, dilated by ``source_mask_dilation_px``, is
       used as a separate source-contamination mask for a second background
       estimate. It never enters the detector mask.
    4. ``sep.extract`` on data - B(x,y) with ``err=R(x,y)`` (local noise) and
       the explicit :class:`SepDetectionParams`.
    5. Per source: HFR from ``sep.flux_radius`` (see ``width_definition``),
       aperture flux/SNR from ``sep.sum_circle`` with ``err=R(x,y)`` and, when
       given, ``gain=gain_e_per_adu`` (e-/ADU) so source Poisson noise is
       included. Without a gain the SNR is a background-noise-only proxy.

    Rejections: a source whose measurement footprint (every pixel touching the
    HFR aperture of radius ``rmax``) contains native-saturated support is
    ``saturated`` and one touching other masked support is ``masked_support``,
    even when the masked core itself contributes no flux — a masked core does
    not make clipped wings a valid width. ``saturation_level`` is an extra
    legacy gate on the raw unmasked footprint peak; it is never inferred here,
    and ``None`` records the gate as ``not_performed``. SEP flags:
    ``OBJ_MERGED`` -> ``blended`` unless ``accept_deblended_for_focus``;
    ``OBJ_TRUNC`` / ``APER_TRUNC`` -> ``edge_truncated``.

    ``thresh_sigma``, ``deblend_cont`` and ``min_area`` are kept as overrides of
    ``detection_params`` for older callers.
    """
    overrides = {}
    if thresh_sigma is not None:
        overrides["thresh_sigma"] = float(thresh_sigma)
    if deblend_cont is not None:
        overrides["deblend_cont"] = float(deblend_cont)
    if min_area is not None:
        overrides["minarea"] = int(min_area)
    if overrides:
        detection_params = replace(detection_params, **overrides)
    if gain_e_per_adu is not None:
        _require_finite("gain_e_per_adu", gain_e_per_adu, minimum=0.0, strict=True)
        gain_e_per_adu = float(gain_e_per_adu)

    sep = require_sep()
    version = sep.__version__
    label_method = f"{BACKGROUND_METHOD_SEP}+{DETECTION_METHOD_SEP}+{HFR_METHOD_SEP}"
    gate, threshold = _saturation_gate(saturation_level, native_saturation_mask)
    detection = {
        "method": DETECTION_METHOD_SEP,
        "library": "sep",
        "library_version": version,
        "config": detection_params.config(version),
        "config_sha256": detection_params.config_sha256(version),
        "noise_source": NOISE_EMPIRICAL_MESH_RMS,
        "status": STATUS_UNAVAILABLE,
        "backend_error": None,
    }

    data = np.ascontiguousarray(image, dtype=np.float32)
    if data.ndim != 2 or 0 in data.shape:
        background = _unavailable_background(background_params, version, units, tuple(data.shape),
                                             [REASON_MALFORMED_GEOMETRY])
        return [], _empty_frame_label(background, detection, background.reasons, units, gate, threshold,
                                      gain_e_per_adu, label_method)
    combined, saturated_mask, malformed = _combine_detector_mask(
        data.shape, data, detector_mask, native_saturation_mask
    )
    if malformed:
        background = _unavailable_background(background_params, version, units, tuple(data.shape),
                                             [REASON_MALFORMED_GEOMETRY])
        return [], _empty_frame_label(background, detection, background.reasons, units, gate, threshold,
                                      gain_e_per_adu, label_method)

    background = estimate_spatial_background(data, detector_mask=combined, params=background_params, units=units)
    if not (background.level_measured and background.noise_measured):
        return [], _empty_frame_label(background, detection, background.reasons, units, gate, threshold,
                                      gain_e_per_adu, label_method)

    mask_u8 = np.ascontiguousarray(combined, dtype=np.uint8)

    def residual_of(bkg: SpatialBackground) -> np.ndarray:
        # SEP extraction has no background term; masked pixels are zeroed, as
        # SEP itself does inside extract (apply_mask_line).
        return np.ascontiguousarray(np.where(combined, np.float32(0.0), data - bkg.background_map), dtype=np.float32)

    try:
        data_sub = residual_of(background)
        objects, segmap = _sep_extract(sep, data_sub, background.rms_map, mask_u8, detection_params)
        if source_mask_background:
            from scipy.ndimage import binary_dilation

            _require_int("source_mask_dilation_px", source_mask_dilation_px, 0)
            source_mask = segmap > 0
            if source_mask_dilation_px > 0:
                source_mask = binary_dilation(source_mask, iterations=int(source_mask_dilation_px))
            refined = estimate_spatial_background(
                data, detector_mask=combined, params=background_params, source_mask=source_mask, units=units
            )
            if not (refined.level_measured and refined.noise_measured):
                return [], _empty_frame_label(refined, detection, refined.reasons, units, gate, threshold,
                                              gain_e_per_adu, label_method)
            background = refined
            data_sub = residual_of(background)
            objects, segmap = _sep_extract(sep, data_sub, background.rms_map, mask_u8, detection_params)
    except Exception as exc:
        detection["backend_error"] = str(exc)
        return [], _empty_frame_label(background, detection, ["detection_backend_failure"], units, gate,
                                      threshold, gain_e_per_adu, label_method)
    detection["status"] = STATUS_MEASURED
    detection["source_mask_background"] = {
        "enabled": bool(source_mask_background),
        "dilation_px": int(source_mask_dilation_px) if source_mask_background else None,
    }

    back_map, rms_map = background.background_map, background.rms_map
    h, w = data.shape
    sources: list[SepSource] = []
    flag_counts: Counter = Counter()

    if len(objects) > 0:
        x = np.asarray(objects["x"], dtype=np.float64)
        y = np.asarray(objects["y"], dtype=np.float64)
        a_axis = np.asarray(objects["a"], dtype=np.float64)
        # Half-light radius, bounded to 6x the isophotal semi-major axis. SEP
        # measures the normalisation flux inside that aperture; objects["flux"]
        # is isophotal and undercounts Moffat wings, biasing HFR low.
        rmax = np.maximum(SEP_HFR_APERTURE_A_FACTOR * a_axis, SEP_HFR_APERTURE_MIN_PX)
        rhalf, rflag = sep.flux_radius(
            data_sub, x, y, rmax, 0.5, normflux=None, mask=mask_u8, maskthresh=0.0, subpix=SEP_SUBPIX
        )
        aperture_radius = np.maximum(SEP_FLUX_APERTURE_A_FACTOR * a_axis, SEP_FLUX_APERTURE_MIN_PX)
        aperture_flux, aperture_fluxerr, aperture_flag = sep.sum_circle(
            data_sub,
            x,
            y,
            aperture_radius,
            err=rms_map,
            gain=gain_e_per_adu,
            mask=mask_u8,
            maskthresh=0.0,
            subpix=SEP_SUBPIX,
        )
        other_masked = combined & ~saturated_mask

        for i in range(len(objects)):
            rej: list[str] = []
            a = float(objects["a"][i])
            b = float(objects["b"][i])
            ellip = float(1.0 - b / a) if a > 0 else 0.0
            cx, cy = float(x[i]), float(y[i])
            ix = int(np.clip(np.floor(cx + 0.5), 0, w - 1))
            iy = int(np.clip(np.floor(cy + 0.5), 0, h - 1))
            fluxerr = float(aperture_fluxerr[i])
            snr = float(aperture_flux[i]) / fluxerr if np.isfinite(fluxerr) and fluxerr > 0 else float("nan")

            flag = int(objects["flag"][i])
            r_flag = int(rflag[i])
            a_flag = int(aperture_flag[i])
            for attr, name in SEP_FLAG_NAMES:
                if flag & int(getattr(sep, attr)):
                    flag_counts[name] += 1
            for attr, name in SEP_APER_FLAG_NAMES:
                if (r_flag | a_flag) & int(getattr(sep, attr)):
                    flag_counts[name] += 1

            deblended = bool(flag & sep.OBJ_MERGED)
            if deblended and not detection_params.accept_deblended_for_focus:
                rej.append(Rejection.BLENDED.value)
            if flag & sep.OBJ_TRUNC or (r_flag | a_flag) & sep.APER_TRUNC:
                rej.append(Rejection.EDGE_TRUNCATED.value)

            # Footprint support: the whole HFR aperture, not only the core.
            window, inside = _disk_footprint((h, w), cx, cy, float(rmax[i]))
            evidence = []
            if saturated_mask[window][inside].any():
                evidence.append("native_mask")
            footprint_valid = inside & ~combined[window]
            raw_values = data[window][footprint_valid]
            peak = float(raw_values.max()) if raw_values.size else float("nan")
            if threshold is not None and np.isfinite(peak) and peak >= threshold:
                evidence.append("scalar_threshold")
            if evidence:
                rej.append(Rejection.SATURATED.value)
            if other_masked[window][inside].any() or (
                not evidence and (r_flag | a_flag) & sep.APER_HASMASKED
            ):
                rej.append(Rejection.MASKED_SUPPORT.value)

            if not np.isfinite(snr) or snr < min_snr:
                rej.append(Rejection.LOW_SNR.value)
            if ellip > max_ellipticity:
                rej.append(Rejection.ELONGATED.value)

            hfr = float(rhalf[i]) if np.isfinite(rhalf[i]) and rhalf[i] > 0 and r_flag == 0 else None
            fit = None
            if fit_fwhm and not rej and hfr is not None:
                half = MOFFAT_CROP_HALF_PX
                x0, y0 = ix - half, iy - half
                if 0 <= x0 and 0 <= y0 and x0 + 2 * half <= w and y0 + 2 * half <= h:
                    sl = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
                    # Local residual: data - B(x,y) over this crop, never globalback.
                    fit = fit_moffat(data[sl] - back_map[sl], mask=combined[sl], fit_offset=moffat_fit_offset)

            valid = not rej and hfr is not None
            sources.append(
                SepSource(
                    x=cx,
                    y=cy,
                    flux=float(aperture_flux[i]),
                    peak=peak,
                    snr=snr,
                    hfr_px=hfr,
                    fwhm_px=fit.fwhm_px if fit else None,
                    ellipticity=ellip,
                    valid=valid,
                    rejections=rej,
                    hfd_px=None if hfr is None else 2.0 * hfr,
                    fwhm_model=fit.fwhm_model if fit else None,
                    fwhm_fallback_reason=fit.fallback_reason if fit else None,
                    fit_converged=bool(fit.fit_converged) if fit else False,
                    moffat_beta=fit.beta if fit else None,
                    moffat_offset_mode=fit.offset_mode if fit else None,
                    moffat_residual_offset=fit.residual_offset if fit else None,
                    sep_flag=flag,
                    radius_flag=r_flag,
                    aperture_flag=a_flag,
                    deblended=deblended,
                    saturation_evidence=evidence,
                    hfr_aperture_radius_px=float(rmax[i]),
                    flux_aperture_radius_px=float(aperture_radius[i]),
                    local_background=float(back_map[iy, ix]),
                    local_rms=float(rms_map[iy, ix]),
                )
            )

    used = [s for s in sources if s.valid]
    hfr_med, hfr_mad = _robust_median_mad([s.hfr_px for s in used if s.hfr_px is not None])
    ell_med, _ = _robust_median_mad([s.ellipticity for s in used])
    fwhm = _fwhm_summary(used)
    saturated = sum(1 for s in sources if Rejection.SATURATED.value in s.rejections)
    label = FrameLabel(
        hfr_px_median=hfr_med,
        hfr_px_mad=hfr_mad,
        fwhm_px_median=fwhm["fwhm_px_median"],
        fwhm_px_mad=fwhm["fwhm_px_mad"],
        ellipticity_median=ell_med,
        star_count_detected=len(sources),
        star_count_used=len(used),
        sky_median_adu=background.global_background,
        sky_rms_adu=background.global_rms,
        saturated_star_fraction=(saturated / len(sources)) if sources else 0.0,
        label_method=label_method,
        status=STATUS_MEASURED if hfr_med is not None else STATUS_UNAVAILABLE,
        reasons=list(background.reasons) + ([] if hfr_med is not None else ["no_valid_sources"]),
        hfd_px_median=None if hfr_med is None else 2.0 * hfr_med,
        fwhm_moffat_count=fwhm["fwhm_moffat_count"],
        fwhm_fallback_count=fwhm["fwhm_fallback_count"],
        fwhm_fallback_px_median=fwhm["fwhm_fallback_px_median"],
        units=units,
        width_definition=dict(SEP_WIDTH_DEFINITION),
        snr_definition=SNR_DEFINITION_WITH_GAIN if gain_e_per_adu is not None else SNR_DEFINITION_BACKGROUND_ONLY,
        gain_e_per_adu=gain_e_per_adu,
        saturation_gate=gate,
        saturation_threshold=threshold,
        rejection_counts=dict(Counter(r for s in sources for r in s.rejections)),
        sep_flag_counts=dict(flag_counts),
        background=background.provenance(),
        detection=detection,
    )
    return sources, label
