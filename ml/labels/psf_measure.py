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

BGE v2 (docs/bge-v2-spec.md; label version ``fqe_sep_spatial_v3``):

  * **Blend gate** ``dip_annulus_v2`` (:func:`blend_gate`, default; the old
    ``peak_count_v1`` stays selectable) on the crop path and on regrouped SEP
    families; its configuration record :class:`StarGateParams`.
  * **SEP deblend regrouping** (:func:`deblend_families`,
    :func:`focus_candidates_from_catalog`): children of one parent are
    measured and judged as one source.
  * **Noise map** ``masked_local_v2`` (``SepBackgroundParams.noise_policy``,
    default; ``frame_wide_v1`` keeps the D-018 rule and its config hash).
  * **Rust-parity focus gates** (:func:`focus_quality`): a port of the
    daemon's quality.rs gate pipeline, the Python reference the BGE harness
    compares with the daemon.

All widths are in **pixels**; levels and noise are in the caller's data units
(``units``; ADU for native camera frames). Conversion to arc-seconds requires a
known pixel scale and is intentionally *not* done here.
"""
from __future__ import annotations

import hashlib
import json
import math
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
# BGE v2 (docs/bge-v2-spec.md) only *adds* fields (star_gates,
# gate_diagnostics, background.refinement); the policy change is carried by
# LABEL_VERSION below and by the recorded star-gate/background config hashes.
LABEL_SCHEMA_VERSION = "fqe-sep-label-v2"

# Label version of the labeller's policy (docs/bge-v2-spec.md §3.9). The
# defaults (dip_annulus_v2 blend gate, SEP deblend regrouping, masked_local_v2
# noise map) are fqe_sep_spatial_v3; the v1 policies selected together
# (peak_count_v1, regroup_deblended=False, frame_wide_v1) reproduce the v2
# labels byte for byte and are recorded as fqe_sep_spatial_v2.
LABEL_VERSION = "fqe_sep_spatial_v3"
LABEL_VERSION_V2 = "fqe_sep_spatial_v2"
READABLE_LABEL_VERSIONS = (LABEL_VERSION_V2, LABEL_VERSION)

# SEP noise-map policies (spec §3). frame_wide_v1 is the D-018 behaviour.
NOISE_POLICY_FRAME_WIDE_V1 = "frame_wide_v1"
NOISE_POLICY_MASKED_LOCAL_V2 = "masked_local_v2"
NOISE_POLICIES = (NOISE_POLICY_FRAME_WIDE_V1, NOISE_POLICY_MASKED_LOCAL_V2)
# masked_local_v2 outcomes (spec §3.4-3.5).
PASS2_SKIP_SOURCE_MASK_FAILED = "source_mask_failed"
PASS2_SKIP_SOURCE_MASK_TOO_LARGE = "source_mask_too_large"
# Not in the spec (shared with the Rust implementation): pass 2's own
# sep.Background call failed (pass 1 kept), and no finite positive mesh
# sigma exists so no cell floor and no masking noise (pass 1 kept; the noise
# is then judged by the v1 rules). Neither is appended to the reasons.
PASS2_SKIP_BACKEND_FAILURE = "pass2_backend_failure"
PASS2_SKIP_NOISE_UNRESOLVED = "noise_unresolved"
RMS_VALIDITY_RULE_LOCAL = "local_cell_min"

BACKGROUND_METHOD_SEP = "sep_spatial_mesh"
# Discriminator of SpatialBackground.provenance(). quality_json["background"]
# can also hold the daemon's Rust BackgroundResult (it has "schema_version");
# this Python reference object says which shape it is instead.
BACKGROUND_PROVENANCE_PRODUCER = "python_sep_label"
BACKGROUND_PROVENANCE_SCHEMA = "achernar.python_sep_background_provenance.v1"
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
# Decision D-018 (mirrors the Rust rule): the natural-cubic RMS spline can also
# undershoot while staying positive. A usable R(x,y) below this fraction of the
# smallest positive filtered mesh sigma is invalid_rms_map too.
RMS_UNDERSHOOT_FRACTION = 0.5
# masked_local_v2 (spec §3.5): some usable pixels are invalid, at most
# RMS_MAX_INVALID_FRACTION of them; noise is measured, the pixels are masked.
REASON_RMS_MAP_LOCALLY_INVALID = "rms_map_locally_invalid"

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
    # Noise-map policy (docs/bge-v2-spec.md §3). ``frame_wide_v1`` is the
    # D-018 rule with a byte-identical configuration record (hash
    # 80eab0fb...0b84); ``masked_local_v2`` (the default) adds a source-masked
    # second background pass and the local cell-floor validity rule. The
    # values below are the spec's provisional parameters (§3.8).
    noise_policy: str = NOISE_POLICY_MASKED_LOCAL_V2
    source_mask_thresh_sigma: float = 2.0
    source_mask_minarea: int = 5
    source_mask_ellipse_scale: float = 3.0
    source_mask_max_fraction: float = 0.5
    rms_local_fraction: float = 0.5
    rms_max_invalid_fraction: float = 0.25

    def __post_init__(self):
        for name in ("bw", "bh", "fw", "fh", "min_meshes_per_axis", "source_mask_minarea"):
            _require_int(name, getattr(self, name), 1)
        _require_finite("fthresh", self.fthresh, minimum=0.0)
        if self.noise_policy not in NOISE_POLICIES:
            raise ValueError(f"noise_policy must be one of {NOISE_POLICIES}, got {self.noise_policy!r}")
        for name in ("source_mask_thresh_sigma", "source_mask_ellipse_scale", "rms_local_fraction"):
            _require_finite(name, getattr(self, name), minimum=0.0, strict=True)
        for name in ("source_mask_max_fraction", "rms_max_invalid_fraction"):
            _require_finite(name, getattr(self, name), minimum=0.0)
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must be <= 1, got {getattr(self, name)!r}")

    def config(self, library_version: str = SEP_PINNED_VERSION) -> dict:
        """Canonical configuration object (keys frozen by the cross-language contract).

        ``frame_wide_v1`` returns exactly the pre-v2 object. ``masked_local_v2``
        adds ``noise_policy``, ``source_mask`` and ``rms_validity`` (spec §3.8).
        """
        config = {
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
        if self.noise_policy == NOISE_POLICY_MASKED_LOCAL_V2:
            config["noise_policy"] = NOISE_POLICY_MASKED_LOCAL_V2
            config["source_mask"] = {
                "thresh_sigma": float(self.source_mask_thresh_sigma),
                "minarea": int(self.source_mask_minarea),
                "ellipse_scale": float(self.source_mask_ellipse_scale),
                "max_fraction": float(self.source_mask_max_fraction),
            }
            config["rms_validity"] = {
                "rule": RMS_VALIDITY_RULE_LOCAL,
                "fraction": float(self.rms_local_fraction),
                "max_invalid_fraction": float(self.rms_max_invalid_fraction),
            }
        return config

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
    rms_validity: Optional[dict] = None  # rms_map_validity diagnostics
    method: str = BACKGROUND_METHOD_SEP
    # masked_local_v2 (spec §3): the noise policy, the refinement record
    # (``None`` for frame_wide_v1), the locally invalid usable pixels M_inv
    # (bool map; they join the extraction and gate masks) and
    # (usable - n_inv) / usable.
    noise_policy: str = NOISE_POLICY_FRAME_WIDE_V1
    refinement: Optional[dict] = None
    invalid_mask: Optional[np.ndarray] = None
    detection_area_fraction: Optional[float] = None
    # The filtered mesh sigmas S of the maps used (ny x nx), when computed.
    mesh_sigmas: Optional[np.ndarray] = None
    # SEP's R(x,y) as returned, kept even when it is not a valid noise
    # measurement (cross-implementation parity only; never consumed).
    raw_rms_map: Optional[np.ndarray] = None
    # masked_local_v2: the cell floor of the S used (float32 map).
    rms_floor_map: Optional[np.ndarray] = None

    def extraction_mask(self) -> Optional[np.ndarray]:
        """The extraction/gate mask: ``detector_mask`` plus, under
        masked_local_v2, the locally invalid pixels ``invalid_mask``."""
        if self.detector_mask is None or self.invalid_mask is None:
            return self.detector_mask
        return self.detector_mask | self.invalid_mask

    def extraction_rms(self) -> Optional[np.ndarray]:
        """The ``err`` array for ``sep.extract``.

        ``frame_wide_v1``: ``rms_map`` unchanged. ``masked_local_v2``:
        ``rms_map`` with every extraction-masked pixel set to the cell floor.
        SEP replaces masked noise by BIG, except that SEP 1.4.1 (CONV filter)
        thresholds masked pixels of the FIRST image row against their own
        noise value: a negative R there is "detected" as an object of masked
        pixels with zero flux and a NaN centroid. Masked values carry no
        information, so this changes nothing else.
        """
        if self.rms_map is None or self.noise_policy != NOISE_POLICY_MASKED_LOCAL_V2 or self.rms_floor_map is None:
            return self.rms_map
        masked = self.extraction_mask()
        if masked is None:
            return self.rms_map
        return np.ascontiguousarray(np.where(masked, self.rms_floor_map, self.rms_map), dtype=np.float32)

    @property
    def level_measured(self) -> bool:
        return self.level_status == STATUS_MEASURED

    @property
    def noise_measured(self) -> bool:
        return self.noise_status == STATUS_MEASURED

    def provenance(self) -> dict:
        """JSON-serialisable summary (no maps) for label metadata and manifests.

        ``producer`` / ``schema`` identify this Python reference shape; it is
        not the Rust ``BackgroundResult`` written by the daemon.
        """
        return {
            "producer": BACKGROUND_PROVENANCE_PRODUCER,
            "schema": BACKGROUND_PROVENANCE_SCHEMA,
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
            "rms_validity": None if self.rms_validity is None else dict(self.rms_validity),
            "noise_policy": self.noise_policy,
            "refinement": None if self.refinement is None else dict(self.refinement),
            "detection_area_fraction": self.detection_area_fraction,
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
        noise_policy=params.noise_policy,
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
    row_starts = np.arange(ny) * bh
    col_starts = np.arange(nx) * bw
    usable = (~excluded).astype(np.int64)
    counts = np.add.reduceat(np.add.reduceat(usable, row_starts, axis=0), col_starts, axis=1)
    heights = np.minimum(row_starts + bh, h) - row_starts
    widths = np.minimum(col_starts + bw, w) - col_starts
    area = heights[:, None] * widths[None, :]
    supported = int((counts.astype(np.float64) >= area.astype(np.float64) * SEP_BACK_MINGOODFRAC).sum())
    return nx, ny, supported, nx * ny


def _sep_spline_d2_operator(n: int) -> np.ndarray:
    """Linear operator of SEP's spline "second derivative" pass over ``n`` nodes.

    A literal transcription of the recurrence in SEP 1.4.1 background.c
    (``makebackspline`` along y; the same code inline in
    ``bkg_line_flt_internal`` along x), applied to the identity so that
    ``d = D @ values``. SEP divides by 6 inside the back-substitution, which is
    not the textbook natural spline; transcribing it keeps the result linear in
    the node values and identical to what SEP evaluates.
    """
    eye = np.eye(n)
    d = np.zeros((n, n))
    if n < 3:
        return d  # both natural boundary conditions; no interior nodes
    u = np.zeros((n, n))
    coef = np.zeros(n)
    for i in range(1, n - 1):
        temp = -1.0 / (coef[i - 1] + 4.0)
        coef[i] = temp
        u[i] = temp * (u[i - 1] - 6.0 * (eye[i + 1] + eye[i - 1] - 2.0 * eye[i]))
    for i in range(n - 2, 0, -1):
        d[i] = (coef[i] * d[i + 1] + u[i]) / 6.0
    return d


def _sep_spline_basis_y(h: int, bh: int, ny: int) -> np.ndarray:
    """``Phi`` (h x ny): row y of SEP's y-interpolation of a node column.

    Node j sits at pixel row ``(j + 0.5) * bh`` (non-centred, as in SEP).
    """
    if ny < 2:
        return np.ones((h, 1))
    d = _sep_spline_d2_operator(ny)
    eye = np.eye(ny)
    phi = np.zeros((h, ny))
    for y in range(h):
        dy = float(np.float32(np.float32(y) / np.float32(bh)) - np.float32(0.5))
        yl = int(dy)  # C truncation toward zero
        dy -= yl
        if yl >= ny - 1:
            yl = ny - 2
            dy += 1.0
        cdy = 1.0 - dy
        phi[y] = (cdy * eye[yl] + dy * eye[yl + 1]
                  + (cdy ** 3 - cdy) * d[yl] + (dy ** 3 - dy) * d[yl + 1])
    return phi


def _sep_spline_basis_x(w: int, bw: int, nx: int) -> np.ndarray:
    """``Psi`` (w x nx): pixel x of SEP's x-interpolation of a node row.

    Node k sits at pixel ``(k + 0.5) * bw - 0.5`` (pixel-centred); the segment
    switches at mid-mesh, exactly as in ``bkg_line_flt_internal``.
    """
    if nx < 2:
        return np.ones((w, 1))
    d = _sep_spline_d2_operator(nx)
    eye = np.eye(nx)
    psi = np.zeros((w, nx))
    xstep = np.float32(1.0) / np.float32(bw)
    changepoint = bw // 2
    dx = (xstep - np.float32(1.0)) / np.float32(2.0)
    dx0 = np.float32(((bw + 1) % 2)) * xstep / np.float32(2.0)
    lo, mesh, i = 0, 0, 0
    for col in range(w):
        if i == changepoint and 0 < mesh < nx - 1:
            lo += 1
            dx = dx0
        fdx = float(dx)
        cdx = 1.0 - fdx
        psi[col] = (cdx * eye[lo] + fdx * eye[lo + 1]
                    + (cdx ** 3 - cdx) * d[lo] + (fdx ** 3 - fdx) * d[lo + 1])
        if i == bw:
            mesh += 1
            i = 0
        i += 1
        dx = np.float32(dx + xstep)
    return psi


def _mesh_size(mesh_grid) -> tuple[int, int]:
    if isinstance(mesh_grid, SepBackgroundParams):
        return int(mesh_grid.bw), int(mesh_grid.bh)
    if isinstance(mesh_grid, dict):
        return int(mesh_grid["bw"]), int(mesh_grid["bh"])
    bw, bh = mesh_grid
    return int(bw), int(bh)


def filtered_mesh_sigmas(rms_map: np.ndarray, mesh_grid) -> tuple[Optional[np.ndarray], dict]:
    """Recover SEP's filtered per-mesh sigma grid (ny x nx) from its RMS map.

    The Python wrapper does not expose ``sep_bkg.sigma`` (sep.pyx keeps the
    struct pointer private; only ``globalrms``/``rms()`` are public). The map is
    linear in the grid, ``R = Phi @ S @ Psi.T``, with ``Phi``/``Psi`` the exact
    transcription of SEP's spline evaluation, so ``S`` is recovered by least
    squares on a strided subset of rows/columns. The fit is verified by
    reconstruction: if the residual exceeds 1e-4 of max|R| (float32 round-off
    is ~1e-7) the grid is not trusted and ``None`` is returned with
    ``method="node_pixel_sample"`` diagnostics from :func:`rms_map_validity`.
    """
    bw, bh = _mesh_size(mesh_grid)
    rms = np.asarray(rms_map, dtype=np.float64)
    h, w = rms.shape
    nx, ny = (w - 1) // bw + 1, (h - 1) // bh + 1
    rows = np.unique(np.r_[np.arange(0, h, max(1, bh // 8)), h - 1])
    cols = np.unique(np.r_[np.arange(0, w, max(1, bw // 8)), w - 1])
    phi = _sep_spline_basis_y(h, bh, ny)[rows]
    psi = _sep_spline_basis_x(w, bw, nx)[cols]
    sample = rms[np.ix_(rows, cols)]
    diagnostics = {"mesh_sigma_method": "spline_inversion", "mesh_grid": [nx, ny],
                   "reconstruction_max_rel_error": None}
    if not np.isfinite(sample).all():
        return None, diagnostics
    left = np.linalg.lstsq(phi, sample, rcond=None)[0]             # ny x len(cols)
    grid = np.linalg.lstsq(psi, left.T, rcond=None)[0].T            # ny x nx
    scale = max(float(np.abs(sample).max()), 1e-30)
    error = float(np.abs(phi @ grid @ psi.T - sample).max()) / scale
    diagnostics["reconstruction_max_rel_error"] = error
    if not np.isfinite(error) or error > 1e-4:
        return None, diagnostics
    return grid, diagnostics


def mesh_sigma_grid(rms_map: np.ndarray, mesh_grid) -> tuple[np.ndarray, dict]:
    """SEP's filtered mesh sigmas (ny x nx), never ``None``.

    :func:`filtered_mesh_sigmas` when its reconstruction verifies
    (``mesh_sigma_method="spline_inversion"``), otherwise the documented
    approximation ``"node_pixel_sample"``: R at the pixel nearest each spline
    node (row ``(j+0.5)*bh``, column ``(k+0.5)*bw - 0.5``, clamped).
    """
    rms = np.asarray(rms_map, dtype=np.float64)
    grid, diagnostics = filtered_mesh_sigmas(rms, mesh_grid)
    if grid is None:
        bw, bh = _mesh_size(mesh_grid)
        h, w = rms.shape
        nx, ny = (w - 1) // bw + 1, (h - 1) // bh + 1
        ys = np.clip(np.floor((np.arange(ny) + 0.5) * bh + 0.5).astype(int), 0, h - 1)
        xs = np.clip(np.floor((np.arange(nx) + 0.5) * bw - 0.5 + 0.5).astype(int), 0, w - 1)
        grid = rms[np.ix_(ys, xs)].astype(np.float64)
        diagnostics["mesh_sigma_method"] = "node_pixel_sample"
    return grid, diagnostics


def _cell_lower_nodes(n_pixels: int, mesh: int, n_nodes: int) -> np.ndarray:
    """Lower spline node of each pixel coordinate (spec §3.2 step 1).

    Nodes sit at ``(j + 0.5) * mesh``: ``0`` if ``2c < mesh``, else
    ``(2c - mesh) div (2 mesh)``, clamped to ``[0, max(n_nodes - 2, 0)]``.
    """
    c = np.arange(n_pixels, dtype=np.int64)
    lower = np.where(2 * c < mesh, 0, (2 * c - mesh) // (2 * mesh))
    return np.clip(lower, 0, max(n_nodes - 2, 0))


def cell_floor_map(mesh_sigmas: np.ndarray, shape, mesh_grid) -> Optional[np.ndarray]:
    """Local reference sigma ``floor(x, y)`` of ``masked_local_v2`` (spec §3.2).

    For each pixel, the minimum finite, strictly positive filtered mesh sigma
    over the cell nodes ``{xl, xl+1} x {yl, yl+1}`` (``+1`` dropped on an axis
    with one node); a cell without such a node takes the global minimum
    positive sigma. Returns a float32 ``shape`` array (sigmas narrowed to
    float32 first, SEP's ``float``), or ``None`` when no positive sigma
    exists anywhere (the floor is undefined).
    """
    sigmas = np.asarray(mesh_sigmas, dtype=np.float64).astype(np.float32)
    ny, nx = sigmas.shape
    bw, bh = _mesh_size(mesh_grid)
    h, w = (int(v) for v in shape)
    positive = np.isfinite(sigmas) & (sigmas > 0)
    if not positive.any():
        return None
    padded = np.where(positive, sigmas, np.float32(np.inf))
    rows = np.minimum(padded[:-1], padded[1:]) if ny > 1 else padded
    cells = np.minimum(rows[:, :-1], rows[:, 1:]) if nx > 1 else rows
    cells = np.where(np.isfinite(cells), cells, padded.min()).astype(np.float32)
    return cells[np.ix_(_cell_lower_nodes(h, bh, ny), _cell_lower_nodes(w, bw, nx))]


def rms_map_validity(rms_map: np.ndarray, usable_mask: np.ndarray, mesh_grid) -> tuple[list, dict]:
    """Validity of SEP's R(x,y) on usable support: ``(reasons, diagnostics)``.

    ``mesh_grid`` is the mesh size: a :class:`SepBackgroundParams`, a dict with
    ``bw``/``bh``, or a ``(bw, bh)`` tuple (nx, ny follow SEP's grid rule).

    Rules (reason codes as in the Rust contract):

    * ``zero_empirical_scatter``: R is exactly 0 on a usable pixel;
    * ``invalid_rms_map``: R is negative or non-finite on a usable pixel, or
      (D-018) below ``RMS_UNDERSHOOT_FRACTION`` (0.5) x the smallest positive
      filtered mesh sigma — the natural-cubic spline undershooting while still
      positive (amp-600 ridge: map minimum 0.549 against a 4.865 mesh minimum,
      103 false detections on a starless frame).

    Exactness relative to the Rust rule: Rust reads ``sep_bkg.sigma`` directly.
    Here the filtered mesh sigmas come from :func:`filtered_mesh_sigmas`, an
    exact (to float32 round-off) inversion of SEP's spline, verified by
    reconstruction; ``diagnostics["mesh_sigma_method"]`` is then
    ``"spline_inversion"``. Only if that verification fails does it fall back
    to the documented approximation ``"node_pixel_sample"``: R at the pixel
    nearest each spline node (row ``(j+0.5)*bh``, column ``(k+0.5)*bw - 0.5``,
    clamped into the frame), minimum over meshes — approximate.

    Diagnostics: ``min_rms`` (over usable pixels), ``min_positive_mesh_sigma``,
    ``bound``, ``count_below_bound``, plus the counts per rule and the method.
    """
    rms = np.asarray(rms_map)
    usable = np.asarray(usable_mask, dtype=bool)
    values = rms[usable].astype(np.float64)
    finite = np.isfinite(values)
    reasons = []
    zero = int((finite & (values == 0)).sum())
    negative = int((finite & (values < 0)).sum())
    nonfinite = int((~finite).sum())
    if zero:
        reasons.append(REASON_ZERO_EMPIRICAL_SCATTER)
    if negative or nonfinite:
        reasons.append(REASON_INVALID_RMS_MAP)

    grid, diagnostics = mesh_sigma_grid(rms, mesh_grid)
    positive = grid[np.isfinite(grid) & (grid > 0)]
    min_positive = float(positive.min()) if positive.size else None
    bound = None if min_positive is None else RMS_UNDERSHOOT_FRACTION * min_positive
    below = int((finite & (values < bound)).sum()) if bound is not None else 0
    if below and REASON_INVALID_RMS_MAP not in reasons:
        reasons.append(REASON_INVALID_RMS_MAP)
    diagnostics.update({
        "min_rms": float(values[finite].min()) if finite.any() else None,
        "min_positive_mesh_sigma": min_positive,
        "bound": bound,
        "bound_fraction": RMS_UNDERSHOOT_FRACTION,
        "count_below_bound": below,
        "count_zero": zero,
        "count_negative": negative,
        "count_nonfinite": nonfinite,
    })
    return reasons, diagnostics


def estimate_spatial_background(
    image: np.ndarray,
    *,
    detector_mask: Optional[np.ndarray] = None,
    params: SepBackgroundParams = SepBackgroundParams(),
    source_mask: Optional[np.ndarray] = None,
    units: str = "adu",
    detection_params: Optional["SepDetectionParams"] = None,
) -> SpatialBackground:
    """SEP spatial background and RMS maps of a full frame, with explicit status.

    ``params.noise_policy`` selects the noise map (docs/bge-v2-spec.md §3):
    ``frame_wide_v1`` is the single pass with the D-018 frame-wide validity
    rule described below; ``masked_local_v2`` (default) is
    :func:`_masked_local_v2_background` (source-masked second pass, local
    cell-floor validity). ``detection_params`` supplies the kernel,
    ``filter_type``, ``deblend_nthresh`` and ``pixstack`` of the v2
    source-mask extraction (default :class:`SepDetectionParams`); it is not
    used by ``frame_wide_v1``.

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
      non-finite on a usable pixel, or below 0.5 x the smallest positive
      filtered mesh sigma (D-018) — SEP's bicubic spline undershooting between
      meshes of very different sigma. See :func:`rms_map_validity`; its
      diagnostics are kept in ``rms_validity``.

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

    if params.noise_policy == NOISE_POLICY_MASKED_LOCAL_V2:
        return _masked_local_v2_background(
            sep, version, data, mask, excluded, params,
            detection_params if detection_params is not None else SepDetectionParams(),
            units, shape, common, mesh,
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
    rms_reasons, rms_diagnostics = rms_map_validity(rms, usable, params)
    reasons.extend(rms_reasons)
    if not (np.isfinite(globalrms) and globalrms > 0) and REASON_ZERO_EMPIRICAL_SCATTER not in reasons:
        reasons.append(REASON_ZERO_EMPIRICAL_SCATTER)
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
        rms_validity=rms_diagnostics,
        raw_rms_map=rms,
        **common,
        **mesh,
    )


def _sep_background_maps(sep, data, excluded, params: SepBackgroundParams):
    """One ``sep.Background`` call: ``(back, rms, globalback, globalrms)``.

    Excluded pixels are zeroed first (SEP ignores them; NaN must not reach
    its histogram), exactly as the frame_wide_v1 path does.
    """
    clean = np.ascontiguousarray(np.where(excluded, np.float32(0.0), data), dtype=np.float32)
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
    return (np.ascontiguousarray(bkg.back(dtype=np.float32)), np.ascontiguousarray(bkg.rms(dtype=np.float32)),
            float(bkg.globalback), float(bkg.globalrms))


def _source_mask_extract(sep, data_sub, r_mask, mask_u8, params: SepBackgroundParams,
                         detection: "SepDetectionParams"):
    """Spec §3.3 step 2: the source-mask extraction (no deblending, no cleaning)."""
    previous = sep.get_extract_pixstack()
    try:
        sep.set_extract_pixstack(int(detection.pixstack))
        return sep.extract(
            data_sub,
            float(params.source_mask_thresh_sigma),
            err=r_mask,
            mask=mask_u8,
            maskthresh=0.0,
            minarea=int(params.source_mask_minarea),
            filter_kernel=detection.kernel_array(),
            filter_type=detection.filter_type,
            deblend_nthresh=int(detection.deblend_nthresh),
            deblend_cont=1.0,
            clean=False,
            segmentation_map=True,
        )
    finally:
        sep.set_extract_pixstack(previous)


def build_source_mask(sep, objects, segmap, ellipse_scale: float) -> np.ndarray:
    """``M_src`` (spec §3.3 step 3): segmentation pixels > 0 plus, per object,
    SEP's own ellipse ``sep.mask_ellipse(x, y, a, b, theta, r=ellipse_scale)``
    (``sep_ellipse_coeffs`` + ``sep_set_ellipse``)."""
    source = np.ascontiguousarray(np.asarray(segmap) > 0, dtype=np.uint8)
    if len(objects):
        sep.mask_ellipse(source, np.asarray(objects["x"], dtype=np.float64), np.asarray(objects["y"], dtype=np.float64),
                         np.asarray(objects["a"], dtype=np.float64), np.asarray(objects["b"], dtype=np.float64),
                         np.asarray(objects["theta"], dtype=np.float64), r=float(ellipse_scale))
    return source.astype(bool)


def local_rms_validity(rms_map, usable_mask, mesh_sigmas, mesh_grid, fraction: float) -> dict:
    """Per-pixel validity of R under ``masked_local_v2`` (spec §3.5).

    A usable pixel is invalid when R is non-finite, R < 0, or
    ``0 < R < fraction * floor(x, y)`` (compared in float64; ``floor`` from
    :func:`cell_floor_map`). ``R == 0`` is the zero rule, counted separately.
    Returns counts, the invalid mask and the floor (``None`` if undefined).
    """
    rms = np.asarray(rms_map, dtype=np.float32)
    usable = np.asarray(usable_mask, dtype=bool)
    floor = cell_floor_map(mesh_sigmas, rms.shape, mesh_grid)
    finite = np.isfinite(rms)
    zero = usable & finite & (rms == 0)
    nonfinite = usable & ~finite
    negative = usable & finite & (rms < 0)
    if floor is not None:
        # (R as f64) < fraction * (floor as f64), as the Rust rule.
        bound = float(fraction) * floor.astype(np.float64)
        below = usable & finite & (rms > 0) & (rms.astype(np.float64) < bound)
    else:
        below = np.zeros(rms.shape, dtype=bool)
    invalid = nonfinite | negative | below
    values = rms[usable & finite]
    positive = np.asarray(mesh_sigmas, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    return {
        "invalid_mask": invalid,
        "floor": floor,
        "count_zero": int(zero.sum()),
        "count_negative": int(negative.sum()),
        "count_nonfinite": int(nonfinite.sum()),
        "count_below_local_bound": int(below.sum()),
        "invalid_pixels": int(invalid.sum()),
        "min_rms": float(values.min()) if values.size else None,
        "min_positive_mesh_sigma": float(positive.min()) if positive.size else None,
        "floor_min": None if floor is None else float(floor.min()),
        "floor_max": None if floor is None else float(floor.max()),
    }


def _masked_local_v2_background(sep, version, data, mask, excluded, params: SepBackgroundParams,
                                detection: "SepDetectionParams", units, shape, common, mesh) -> SpatialBackground:
    """``masked_local_v2`` (docs/bge-v2-spec.md §3): two passes and local validity.

    1. Pass 1 = the frame_wide_v1 SEP call (``B1, R1``); ``S1`` is recovered
       by :func:`mesh_sigma_grid` (spline inversion, D-018).
    2. ``R_mask = R1`` where finite and above ``floor``, else ``floor``
       (never reported). Sources are extracted on ``I - B1`` (0 on M_det) at
       ``source_mask_thresh_sigma`` x R_mask with the detection kernel, no
       deblending and no cleaning; ``M_src`` = segmentation + SEP ellipses at
       ``source_mask_ellipse_scale``.
    3. Pass 2 = SEP with ``M_det | M_src``, unless the extraction failed or
       the floor is undefined (``source_mask_failed``), ``|M_src \\ M_det|``
       exceeds ``source_mask_max_fraction`` of the usable pixels or no mesh
       keeps 50% support (``source_mask_too_large``), or pass 2's own SEP call
       fails (``pass2_backend_failure``, not in the spec; pass 1 is kept).
    4. On the maps used: the v1 zero rule (R == 0 on a usable pixel, or
       globalrms <= 0) and the local rule (:func:`local_rms_validity`). More
       than ``rms_max_invalid_fraction`` invalid usable pixels is
       ``invalid_rms_map`` (noise unresolved); fewer is measured with
       ``rms_map_locally_invalid`` and the pixels returned as ``invalid_mask``.
    """
    usable = ~mask
    usable_count = int(usable.sum())
    mesh_grid = (int(params.bw), int(params.bh))

    def unavailable(reason, error):
        return _unavailable_background(params, version, units, shape, [reason], backend_error=error,
                                       **common, **mesh)

    try:
        back, rms, globalback, globalrms = _sep_background_maps(sep, data, excluded, params)
    except Exception as exc:  # SEP raises plain Exception / MemoryError
        return unavailable(REASON_BACKEND_FAILURE, str(exc))
    if not (bool(np.isfinite(back[usable]).all()) and np.isfinite(globalback)):
        return unavailable(REASON_BACKEND_FAILURE, "non-finite background map")
    sigmas, sigma_diag = mesh_sigma_grid(rms, mesh_grid)
    pass1_method = sigma_diag.get("mesh_sigma_method")
    floor1 = cell_floor_map(sigmas, shape, mesh_grid)

    skip, skip_detail, n_objects, n_source = None, None, None, None
    used_mesh = dict(mesh)
    pass2_used = False
    if floor1 is None:
        skip, skip_detail = PASS2_SKIP_NOISE_UNRESOLVED, "no finite positive filtered mesh sigma: cell floor undefined"
    else:
        # NaN-ignoring max(R1, floor) (np.fmax), never reported.
        r_mask = np.ascontiguousarray(np.fmax(rms, floor1), dtype=np.float32)
        data_sub = np.ascontiguousarray(np.where(mask, np.float32(0.0), data - back), dtype=np.float32)
        try:
            objects, segmap = _source_mask_extract(sep, data_sub, r_mask, np.ascontiguousarray(mask, dtype=np.uint8),
                                                   params, detection)
            source = build_source_mask(sep, objects, segmap, params.source_mask_ellipse_scale)
        except Exception as exc:  # e.g. pixstack overflow
            skip, skip_detail = PASS2_SKIP_SOURCE_MASK_FAILED, f"{type(exc).__name__}: {exc}"
        else:
            n_objects = int(len(objects))
            n_source = int((source & ~mask).sum())
            combined = excluded | source
            if n_source / float(usable_count) > params.source_mask_max_fraction:
                skip = PASS2_SKIP_SOURCE_MASK_TOO_LARGE
                skip_detail = f"{n_source} source-masked of {usable_count} usable pixels"
            else:
                nx2, ny2, supported2, total2 = _mesh_support(combined, int(params.bw), int(params.bh))
                if supported2 == 0:
                    skip = PASS2_SKIP_SOURCE_MASK_TOO_LARGE
                    skip_detail = "every mesh is below SEP's 50% support under the combined mask"
                else:
                    try:
                        back2, rms2, globalback2, globalrms2 = _sep_background_maps(sep, data, combined, params)
                    except Exception as exc:
                        skip, skip_detail = PASS2_SKIP_BACKEND_FAILURE, str(exc)
                    else:
                        if bool(np.isfinite(back2[usable]).all()) and np.isfinite(globalback2):
                            back, rms, globalback, globalrms = back2, rms2, globalback2, globalrms2
                            sigmas, sigma_diag = mesh_sigma_grid(rms, mesh_grid)
                            used_mesh = dict(mesh_grid=(nx2, ny2), meshes_supported=supported2, meshes_total=total2)
                            pass2_used = True
                        else:
                            skip, skip_detail = PASS2_SKIP_BACKEND_FAILURE, "non-finite pass-2 background map"

    reasons = []
    if used_mesh.get("meshes_supported") is not None and used_mesh["meshes_supported"] < used_mesh["meshes_total"]:
        reasons.append(REASON_MESHES_INTERPOLATED)
    if skip in (PASS2_SKIP_SOURCE_MASK_FAILED, PASS2_SKIP_SOURCE_MASK_TOO_LARGE):
        reasons.append(skip)
    validity = local_rms_validity(rms, usable, sigmas, mesh_grid, params.rms_local_fraction)
    if validity["floor"] is None:
        # No finite positive mesh sigma on the maps used: unresolved exactly as
        # v1 says (the v1 rules of this module), no local count.
        v1_reasons, v1_diag = rms_map_validity(rms, usable, mesh_grid)
        noise = list(v1_reasons)
        if not (np.isfinite(globalrms) and globalrms > 0) and REASON_ZERO_EMPIRICAL_SCATTER not in noise:
            noise.insert(0, REASON_ZERO_EMPIRICAL_SCATTER)
        if not noise:
            noise = [REASON_ZERO_EMPIRICAL_SCATTER]
        n_invalid = None
        validity.update({k: v1_diag.get(k) for k in ("count_below_bound", "bound")})
    else:
        # The zero rule and the mesh-level checks are v1's: globalrms == 0 or a
        # usable R == 0 is zero scatter; a non-finite or negative globalrms or
        # (recovered) mesh sigma is a frame-wide invalid map.
        zero = validity["count_zero"] > 0 or globalrms == 0.0
        mesh_invalid = (not np.isfinite(globalrms) or globalrms < 0
                        or not bool(np.isfinite(np.asarray(sigmas, dtype=np.float64)).all()))
        n_invalid = validity["invalid_pixels"]
        frame_wide = n_invalid / float(usable_count) > params.rms_max_invalid_fraction
        noise = []
        if zero:
            noise.append(REASON_ZERO_EMPIRICAL_SCATTER)
        if mesh_invalid or frame_wide:
            noise.append(REASON_INVALID_RMS_MAP)
        if n_invalid and not noise:
            noise.append(REASON_RMS_MAP_LOCALLY_INVALID)
    noise_ok = noise in ([], [REASON_RMS_MAP_LOCALLY_INVALID])
    reasons.extend(noise)
    invalid_fraction = None if n_invalid is None else n_invalid / float(usable_count)
    detection_area_fraction = None if n_invalid is None else (usable_count - n_invalid) / float(usable_count)
    refinement = {
        "noise_policy": NOISE_POLICY_MASKED_LOCAL_V2,
        "pass2_used": pass2_used,
        "pass2_skipped_reason": skip,
        "source_mask_objects": n_objects,
        "source_mask_pixels": n_source,
        "source_mask_fraction": None if n_source is None else n_source / float(usable_count),
        "invalid_rms_pixels": n_invalid,
        "invalid_rms_fraction": invalid_fraction,
        "detection_area_fraction": detection_area_fraction,
    }
    diagnostics = {
        "rule": RMS_VALIDITY_RULE_LOCAL,
        "fraction": float(params.rms_local_fraction),
        "max_invalid_fraction": float(params.rms_max_invalid_fraction),
        "mesh_sigma_method": sigma_diag.get("mesh_sigma_method"),
        "reconstruction_max_rel_error": sigma_diag.get("reconstruction_max_rel_error"),
        "pass1_mesh_sigma_method": pass1_method,
        "pass2_skipped_detail": skip_detail,
        "invalid_fraction": invalid_fraction,
        **{k: v for k, v in validity.items() if k not in ("invalid_mask", "floor")},
    }
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
        rms_validity=diagnostics,
        noise_policy=NOISE_POLICY_MASKED_LOCAL_V2,
        refinement=refinement,
        invalid_mask=validity["invalid_mask"] if noise_ok and n_invalid else None,
        detection_area_fraction=detection_area_fraction,
        mesh_sigmas=sigmas,
        raw_rms_map=rms,
        rms_floor_map=validity["floor"],
        **common,
        **used_mesh,
    )


# --------------------------------------------------------------------------- #
# Rejection taxonomy — why a candidate star is not usable for a width label.
# --------------------------------------------------------------------------- #
class Rejection(str, Enum):
    SATURATED = "saturated"
    MASKED_SUPPORT = "masked_support"
    NO_USABLE_PIXELS = "no_usable_pixels"
    EDGE_TRUNCATED = "edge_truncated"
    LOW_SNR = "low_snr"
    NOISE_UNRESOLVED = "noise_unresolved"
    TOO_FEW_PIXELS = "too_few_pixels"
    ELONGATED = "elongated"
    # Crop-path two-core-peak heuristic (Rust StarRejectionTallies.blended).
    BLENDED = "blended"
    # SEP-deblended child (OBJ_MERGED) rejected by the focus policy
    # (Rust StarRejectionTallies.deblended); not the same population as BLENDED.
    DEBLENDED = "deblended"
    FIT_FAILED = "fit_failed"
    # masked_local_v2: the measurement support holds locally invalid R(x,y)
    # (spec §3.6; Rust StarRejectionTallies.invalid_noise_support).
    INVALID_NOISE_SUPPORT = "invalid_noise_support"


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
    # Gate-5 outcome (BlendGateResult.to_json()) when the blend gate ran.
    blend: Optional[dict] = None


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
    saturated_star_fraction: Optional[float]  # None when no source was detected
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
    # BGE v2 (docs/bge-v2-spec.md): the labeller's policy version, the
    # star-gate configuration record and its sha256, and the per-frame
    # accepted-star diagnostics (annulus_exempt, regrouped_accepted).
    label_version: Optional[str] = None
    star_gates: Optional[dict] = None
    star_gates_config_sha256: Optional[str] = None
    gate_diagnostics: Optional[dict] = None

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

    Returns ``(level, rms)``. When the noise is unresolved (zero-MAD plateau)
    ``rms`` is ``nan``: an unresolved scale has no numeric value, and a 0.0
    would be indistinguishable from a measurement (and invites a silent floor
    such as ``max(rms, 1e-6)``). It is never the inflated std of the unclipped
    crop. When no pixel is finite both values are ``nan``. Callers that must
    distinguish these states use the detailed form; the crop measurement
    functions here do, and :func:`curve_of_growth_hfr` treats a non-finite
    ``sky_rms`` as "no core threshold".
    """
    result = estimate_background_detailed(image, sigma, iters)
    if result.level is None:
        return float("nan"), float("nan")
    return result.level, (result.rms if result.rms is not None else float("nan"))


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
    is ``None``, non-finite (unresolved) or not positive the centroid uses all
    positive residual.

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
# Star blend gates (docs/bge-v2-spec.md §1-§2) and their configuration record.
#
# Both the crop path (measure_star), the SEP labeller (measure_frame_sep,
# regrouped families) and the Rust-parity focus gates below call
# :func:`blend_gate`. Every numeric parameter is provisional and versioned in
# the configuration record (spec §1.5-§1.6).
# --------------------------------------------------------------------------- #
BLEND_TEST_PEAK_COUNT_V1 = "peak_count_v1"
BLEND_TEST_DIP_ANNULUS_V2 = "dip_annulus_v2"
BLEND_TESTS = (BLEND_TEST_PEAK_COUNT_V1, BLEND_TEST_DIP_ANNULUS_V2)
#: Peak window half-size and threshold (unchanged from v1).
BLEND_PEAK_WINDOW_HALF = 3
BLEND_THRESHOLD_SIGMA = 5.0
#: Neighbour visiting order (dx, dy) of the saddle merge (spec §1.3 step 4).
SADDLE_NEIGHBOURS = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


@dataclass(frozen=True)
class StarGateParams:
    """Star blend gate and deblend-regrouping policy (spec §1.6).

    ``blend_test``: ``"dip_annulus_v2"`` (default) or ``"peak_count_v1"``.
    ``regroup_deblended`` (default ``True``): SEP-deblended children are
    regrouped into their parent families, each judged by the blend gate.
    ``accept_deblended``: accept children individually (v1); regrouping does
    not run then. The ``dip_*``/``ann_*`` fields are the provisional
    ``dip_annulus_v2`` parameters of spec §1.5.
    """

    blend_test: str = BLEND_TEST_DIP_ANNULUS_V2
    regroup_deblended: bool = True
    accept_deblended: bool = False
    dip_fraction: float = 0.15
    dip_min_sigma: float = 4.0
    ann_min_radius_px: float = 2.5
    ann_radius_tol: float = 0.35
    ann_half_width: float = 0.25
    ann_bins: int = 16
    ann_bin_fraction: float = 0.5
    ann_min_coverage: float = 0.625
    ann_hole_radius: float = 0.35
    ann_hole_fraction: float = 0.6

    def __post_init__(self):
        if self.blend_test not in BLEND_TESTS:
            raise ValueError(f"blend_test must be one of {BLEND_TESTS}, got {self.blend_test!r}")
        if not isinstance(self.regroup_deblended, bool) or not isinstance(self.accept_deblended, bool):
            raise ValueError("regroup_deblended and accept_deblended must be bool")
        _require_int("ann_bins", self.ann_bins, 4)
        for name in ("dip_fraction", "dip_min_sigma", "ann_min_radius_px", "ann_radius_tol", "ann_half_width",
                     "ann_bin_fraction", "ann_min_coverage", "ann_hole_radius", "ann_hole_fraction"):
            _require_finite(name, getattr(self, name), minimum=0.0)

    @property
    def regroups(self) -> bool:
        """Regrouping runs only with ``regroup_deblended`` and not ``accept_deblended`` (spec §2.5)."""
        return self.regroup_deblended and not self.accept_deblended

    def config(self) -> dict:
        """Canonical star-gate configuration record (spec §1.6)."""
        if self.blend_test == BLEND_TEST_PEAK_COUNT_V1:
            return {
                "blend_test": BLEND_TEST_PEAK_COUNT_V1,
                "peak_window_half": BLEND_PEAK_WINDOW_HALF,
                "threshold_sigma": BLEND_THRESHOLD_SIGMA,
                "regroup_deblended": bool(self.regroup_deblended),
                "accept_deblended": bool(self.accept_deblended),
            }
        return {
            "blend_test": BLEND_TEST_DIP_ANNULUS_V2,
            "peak_window_half": BLEND_PEAK_WINDOW_HALF,
            "threshold_sigma": BLEND_THRESHOLD_SIGMA,
            "dip_fraction": float(self.dip_fraction),
            "dip_min_sigma": float(self.dip_min_sigma),
            "annulus": {
                "bins": int(self.ann_bins),
                "bin_fraction": float(self.ann_bin_fraction),
                "min_coverage": float(self.ann_min_coverage),
                "half_width": float(self.ann_half_width),
                "hole_radius": float(self.ann_hole_radius),
                "hole_fraction": float(self.ann_hole_fraction),
                "min_radius_px": float(self.ann_min_radius_px),
                "radius_tol": float(self.ann_radius_tol),
            },
            "regroup_deblended": bool(self.regroup_deblended),
            "accept_deblended": bool(self.accept_deblended),
        }

    def config_sha256(self) -> str:
        return config_sha256(self.config())


def star_gates(blend_test: str = BLEND_TEST_DIP_ANNULUS_V2, regroup_deblended: bool = True,
               accept_deblended: bool = False) -> StarGateParams:
    """:class:`StarGateParams` from the keyword names of spec §1.6."""
    return StarGateParams(blend_test=blend_test, regroup_deblended=regroup_deblended,
                          accept_deblended=accept_deblended)


@dataclass(frozen=True)
class BlendPeak:
    """One peak of spec §1.2: plateau maximum ``value``, its representative
    pixel ``rep`` (lowest linear index holding the maximum), ``sigma`` and
    ``threshold`` at ``rep`` (all float32 values)."""

    value: float
    rep: int
    sigma: float
    threshold: float


@dataclass
class BlendGateResult:
    """Outcome of :func:`blend_gate` on one measurement crop.

    ``flagged``: the gate fired (v1: >= 2 peaks; v2: >= 2 components).
    ``exempt``: v2 only, the annulus exemption accepted the flagged star.
    ``blended`` = ``flagged and not exempt`` (the rejection).
    """

    blend_test: str
    n_peaks: int
    n_components: int
    flagged: bool
    exempt: bool
    blended: bool
    resolved: list = field(default_factory=list)  # [{rep, value, dip, significant, kind}]
    annulus: Optional[dict] = None

    def to_json(self) -> dict:
        return {"blend_test": self.blend_test, "n_peaks": self.n_peaks, "n_components": self.n_components,
                "flagged": self.flagged, "exempt": self.exempt, "blended": self.blended,
                "resolved": list(self.resolved), "annulus": self.annulus}


def find_blend_peaks(values: np.ndarray, threshold: np.ndarray, *, strict_above: bool = True,
                     sigma: Optional[np.ndarray] = None) -> list:
    """Peaks of spec §1.2 (identical to v1's ``count_core_peaks``).

    A peak pixel is finite, above ``threshold`` and ``>=`` the maximum of the
    finite values in its ``(2*3+1)^2`` window clipped to the crop; 8-connected
    peak pixels collapse into one :class:`BlendPeak`. ``strict_above=False``
    reproduces the Rust v1 predicate ``!(value <= threshold)`` exactly (it
    differs only for a non-finite threshold). Peaks are ordered by ``rep``.
    """
    from scipy.ndimage import label, maximum_filter

    v = np.asarray(values, dtype=np.float32)
    t = np.broadcast_to(np.asarray(threshold, dtype=np.float32), v.shape)
    finite = np.isfinite(v)
    filled = np.where(finite, v, np.float32(-np.inf))
    size = 2 * BLEND_PEAK_WINDOW_HALF + 1
    window_max = maximum_filter(filled, size=size, mode="constant", cval=-np.inf)
    above = (v > t) if strict_above else ~(v <= t)
    is_peak = finite & above & (v >= window_max)
    labels, n = label(is_peak, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return []
    flat_labels = labels.ravel()
    index = np.flatnonzero(flat_labels)
    flat_v = v.ravel()
    flat_t = t.ravel()
    flat_s = None if sigma is None else np.broadcast_to(np.asarray(sigma, dtype=np.float32), v.shape).ravel()
    peaks = []
    for k in range(1, n + 1):
        members = index[flat_labels[index] == k]
        vals = flat_v[members]
        top = vals.max()
        rep = int(members[vals == top].min())
        peaks.append(BlendPeak(value=np.float32(top), rep=rep,
                               sigma=np.float32(flat_s[rep]) if flat_s is not None else np.float32(np.nan),
                               threshold=np.float32(flat_t[rep])))
    peaks.sort(key=lambda p: p.rep)
    return peaks


def _brighter(a: BlendPeak, b: BlendPeak) -> bool:
    """``a`` brighter than ``b``: higher value, ties to the lower rep index."""
    return a.value > b.value or (a.value == b.value and a.rep < b.rep)


def saddle_merge(residual: np.ndarray, threshold: np.ndarray, peaks: list) -> list:
    """Bottleneck merge of spec §1.3: ``(records, brightest)``, one record
    per resolved (non-brightest) peak and the index of the brightest peak.

    Pixels of ``A`` (finite, above threshold) are processed by value
    descending, ties by linear index ascending; union-find over 8-neighbours
    in :data:`SADDLE_NEIGHBOURS` order. When two components that both hold a
    peak merge at level ``v``, the fainter peak resolves with dip
    ``p - v``; after the sweep, every remaining non-brightest component
    resolves with the disjoint dip ``p - t_rep``. Values and thresholds are
    float32; dips are float64 differences of them.
    """
    res = np.asarray(residual, dtype=np.float32)
    t = np.broadcast_to(np.asarray(threshold, dtype=np.float32), res.shape)
    h, w = res.shape
    flat = res.ravel()
    flat_t = t.ravel()
    in_a = np.flatnonzero(np.isfinite(flat) & (flat > flat_t))
    order = in_a[np.lexsort((in_a, -flat[in_a].astype(np.float64)))].tolist()
    peak_at = {p.rep: k for k, p in enumerate(peaks)}
    parent = [-1] * (h * w)
    best = [-1] * (h * w)
    resolved_at = {}  # peak id -> pixel index whose level resolved it

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for q in order:
        parent[q] = q
        best[q] = peak_at.get(q, -1)
        qx, qy = q % w, q // w
        for dx, dy in SADDLE_NEIGHBOURS:
            nx_, ny_ = qx + dx, qy + dy
            if nx_ < 0 or ny_ < 0 or nx_ >= w or ny_ >= h:
                continue
            n = ny_ * w + nx_
            if parent[n] == -1:
                continue
            rq, rn = find(q), find(n)
            if rq == rn:
                continue
            bq, bn = best[rq], best[rn]
            if bq >= 0 and bn >= 0:
                if _brighter(peaks[bq], peaks[bn]):
                    winner, loser = bq, bn
                else:
                    winner, loser = bn, bq
                resolved_at[loser] = q
            else:
                winner = bq if bq >= 0 else bn
            parent[rn] = rq
            best[rq] = winner
    brightest = 0
    for k in range(1, len(peaks)):
        if _brighter(peaks[k], peaks[brightest]):
            brightest = k
    records = []
    for k, p in enumerate(peaks):
        if k == brightest:
            continue
        # Dips in float64 of the float32 values (as the Rust implementation).
        if k in resolved_at:
            dip, kind = float(p.value) - float(flat[resolved_at[k]]), "saddle"
        else:
            dip, kind = float(p.value) - float(p.threshold), "disjoint"
        records.append({"peak": k, "rep": p.rep, "value": float(p.value), "dip": dip, "kind": kind})
    return records, brightest


def annulus_exemption(residual: np.ndarray, threshold: np.ndarray, peaks_s: list,
                      gates: StarGateParams) -> dict:
    """Annulus exemption of spec §1.4 for the peaks ``S`` (brightest +
    significant). Geometry and bin statistics are float64; non-finite
    residuals are skipped. Returns diagnostics with ``passed`` and, on
    failure, the first failed ``step``."""
    from scipy.ndimage import label

    res = np.asarray(residual, dtype=np.float32)
    t = np.broadcast_to(np.asarray(threshold, dtype=np.float32), res.shape)
    h, w = res.shape
    finite = np.isfinite(res)
    in_a = finite & (res > t)
    labels, _ = label(in_a, structure=np.ones((3, 3), dtype=int))
    keep = sorted({int(labels.ravel()[p.rep]) for p in peaks_s})
    union = np.isin(labels, keep)
    ys, xs = np.nonzero(union)
    weight = res[ys, xs].astype(np.float64)
    total = float(weight.sum())
    out = {"passed": False, "step": None}
    if not total > 0:
        out["step"] = "centre"
        return out
    cx = float((xs * weight).sum() / total)
    cy = float((ys * weight).sum() / total)
    out["centre"] = [cx, cy]

    def dist_to_c(x, y):
        dx, dy = x - cx, y - cy
        return np.sqrt(dx * dx + dy * dy)

    distances = [float(dist_to_c(float(p.rep % w), float(p.rep // w))) for p in peaks_s]
    radius = float(np.median(distances))
    out.update(radius_px=radius, peak_distances_px=distances)
    if not radius >= gates.ann_min_radius_px:
        out["step"] = "radius"
        return out
    if any(abs(d - radius) > gates.ann_radius_tol * radius + 1.0 for d in distances):
        out["step"] = "common_circle"
        return out
    yy, xx = np.mgrid[0:h, 0:w]
    dx = xx.astype(np.float64) - cx
    dy = yy.astype(np.float64) - cy
    dist = np.sqrt(dx * dx + dy * dy)
    half = max(1.5, gates.ann_half_width * radius)
    ring = finite & (np.abs(dist - radius) <= half)
    angle = np.arctan2(dy, dx)
    bins = int(gates.ann_bins)
    index = np.clip(np.floor((angle + np.pi) / (2.0 * np.pi) * bins).astype(np.int64), 0, bins - 1)
    values = res.astype(np.float64)
    means = np.full(bins, -np.inf)
    for k in range(bins):
        sel = ring & (index == k)
        if sel.any():
            means[k] = float(values[sel].mean())
    top = float(np.mean(np.sort(means)[::-1][:4]))
    coverage = float(np.sum(means >= gates.ann_bin_fraction * top)) / bins
    out.update(bin_means=[float(m) if np.isfinite(m) else None for m in means], top=top, coverage=coverage)
    if not coverage >= gates.ann_min_coverage:
        out["step"] = "coverage"
        return out
    hole = finite & (dist <= max(1.0, gates.ann_hole_radius * radius))
    hole_mean = float(values[hole].mean()) if hole.any() else float("nan")
    out["hole_mean"] = hole_mean if np.isfinite(hole_mean) else None
    if not hole_mean <= gates.ann_hole_fraction * top:
        out["step"] = "hole"
        return out
    out["passed"] = True
    return out


def blend_gate(residual: np.ndarray, sigma, gates: StarGateParams = StarGateParams(), *,
               v1_values: Optional[np.ndarray] = None, v1_threshold=None) -> BlendGateResult:
    """Gate 5 on one measurement crop (spec §1.1-§1.4).

    ``residual``: crop minus sky (crop path) or minus B(x,y) (map path);
    ``sigma``: the crop noise (scalar) or R(x,y). The threshold is
    ``5 * sigma`` (float32). ``peak_count_v1`` flags >= 2 peaks; by default
    on ``residual > 5 sigma``; ``v1_values``/``v1_threshold`` let the Rust
    crop-path port count on the raw crop against ``sky + 5 noise`` exactly
    as v1 did. ``dip_annulus_v2`` flags >= 2 significant components and then
    tries the annulus exemption.
    """
    res = np.asarray(residual, dtype=np.float32)
    sig = np.broadcast_to(np.asarray(sigma, dtype=np.float32), res.shape)
    t = np.float32(BLEND_THRESHOLD_SIGMA) * sig
    if gates.blend_test == BLEND_TEST_PEAK_COUNT_V1:
        if v1_values is not None:
            peaks = find_blend_peaks(v1_values, v1_threshold, strict_above=False)
        else:
            peaks = find_blend_peaks(res, t, strict_above=False)
        flagged = len(peaks) >= 2
        return BlendGateResult(BLEND_TEST_PEAK_COUNT_V1, len(peaks), len(peaks), flagged, False, flagged)
    peaks = find_blend_peaks(res, t, sigma=sig)
    if len(peaks) <= 1:
        return BlendGateResult(BLEND_TEST_DIP_ANNULUS_V2, len(peaks), len(peaks), False, False, False)
    records, brightest = saddle_merge(res, t, peaks)
    significant = []
    for rec in records:
        p = peaks[rec["peak"]]
        bound = max(gates.dip_fraction * float(p.value), gates.dip_min_sigma * float(p.sigma))
        rec["significant"] = bool(rec["dip"] > bound)
        if rec["significant"]:
            significant.append(p)
    n_components = 1 + len(significant)
    if n_components < 2:
        return BlendGateResult(BLEND_TEST_DIP_ANNULUS_V2, len(peaks), n_components, False, False, False, records)
    annulus = annulus_exemption(res, t, [peaks[brightest]] + significant, gates)
    exempt = bool(annulus["passed"])
    return BlendGateResult(BLEND_TEST_DIP_ANNULUS_V2, len(peaks), n_components, True, exempt, not exempt,
                           records, annulus)


# --------------------------------------------------------------------------- #
# Rust-parity focus-star gates: a port of achernar-daemon
# src/vision/quality.rs::measure_stars (+ ClassicalFocusSolver::
# compute_hfr_with_sky), with the v2 blend gate, SEP deblend regrouping and
# invalid_noise_support. This is the Python reference the BGE harness
# compares with the daemon (python_sep__nXgY); arithmetic follows the Rust
# types (float32 gates with sequential sums, float64 HFR).
# --------------------------------------------------------------------------- #
FOCUS_MAX_STARS_USED = 16
FOCUS_MAX_STARS_EXAMINED = 48
FOCUS_SATURATION_FRACTION_OF_MAX = 0.98
FOCUS_EDGE_MARGIN_PX = 2
FOCUS_MIN_APERTURE_SNR = 5.0
FOCUS_MIN_CORE_PIXELS = 5
FOCUS_MAX_ELLIPTICITY = 0.35
FOCUS_CROP_MIN_PX = 32
FOCUS_CROP_MAX_PX = 128
FOCUS_MAX_SOURCE_AREA = 5_000
#: Rust StarRejectionTallies serde names (+ v2 invalid_noise_support).
FOCUS_TALLY_KEYS = ("saturated", "edge_truncated", "low_snr", "too_few_pixels", "elongated", "blended",
                    "invalid_hfr", "masked_support", "deblended", "invalid_noise_support")
_F32_EPS = np.float32(np.finfo(np.float32).eps)
_F32_MIN_POSITIVE = np.float32(np.finfo(np.float32).tiny)
_F64_EPS = float(np.finfo(np.float64).eps)


@dataclass
class FocusCandidate:
    """Rust ``StarCandidate`` (float32 position/flux/peak) plus the family
    fields of spec §2 (``family_size >= 2`` for a regrouped family)."""

    x: float
    y: float
    area: int
    flux: float
    peak: float
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    deblended: bool = False
    family_size: int = 1
    members: tuple = ()

    def to_json(self) -> dict:
        return {"x": float(self.x), "y": float(self.y), "xmin": int(self.min_x), "xmax": int(self.max_x),
                "ymin": int(self.min_y), "ymax": int(self.max_y), "npix": int(self.area), "flux": float(self.flux),
                "peak": float(self.peak), "deblended": bool(self.deblended), "family_size": int(self.family_size),
                "members": [int(m) for m in self.members]}


def _f32_sum(values: np.ndarray) -> np.float32:
    """Sequential float32 sum (Rust ``+=`` over the iteration order)."""
    values = np.asarray(values, dtype=np.float32).ravel()
    if values.size == 0:
        return np.float32(0.0)
    return np.cumsum(values, dtype=np.float32)[-1]


def _f64_sum(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0
    return float(np.cumsum(values)[-1])


def _median_sorted_f32(sorted_values: np.ndarray) -> np.float32:
    n = sorted_values.size
    if n % 2 == 1:
        return np.float32(sorted_values[n // 2])
    return np.float32(np.float32(0.5) * (np.float32(sorted_values[n // 2 - 1]) + np.float32(sorted_values[n // 2])))


def rust_local_sky_stats(crop: np.ndarray) -> tuple:
    """quality.rs ``local_sky_stats``: (median, 1.4826*MAD), float32, mean of
    the two middle values for even counts; ``(0, 0)`` without finite pixels."""
    values = np.asarray(crop, dtype=np.float32).ravel()
    finite = np.sort(values[np.isfinite(values)])
    if finite.size == 0:
        return np.float32(0.0), np.float32(0.0)
    sky = _median_sorted_f32(finite)
    deviations = np.sort(np.abs(finite - sky))
    return sky, np.float32(_median_sorted_f32(deviations) * np.float32(1.4826))


def rust_legacy_crop_sky(crop: np.ndarray) -> tuple:
    """classical_focus_solver.rs ``legacy_crop_sky``: nearest-rank median
    (rank ``round((n-1)/2)``) and upper-middle MAD x 1.4826, float64."""
    values = np.asarray(crop, dtype=np.float32).ravel()
    finite = np.sort(values[np.isfinite(values)])
    if finite.size == 0:
        return 0.0, 0.0
    idx = int(math.floor((finite.size - 1) * 0.5 + 0.5))
    level = float(finite[idx])
    deviations = np.sort(np.abs(finite.astype(np.float64) - level))
    return level, float(deviations[deviations.size // 2] * 1.4826)


def _cog_half_radius_rust(radii: np.ndarray, values: np.ndarray, max_radius: float):
    count = int(np.searchsorted(radii, max_radius, side="right"))
    if count < 4:
        return None
    cumulative = np.cumsum(values[:count])
    total = float(cumulative[-1])
    if total <= 0.0:
        return None
    half = 0.5 * total
    idx = int(np.argmax(cumulative >= half))
    if idx == 0:
        return float(radii[0]), total
    c0, c1 = float(cumulative[idx - 1]), float(cumulative[idx])
    r0, r1 = float(radii[idx - 1]), float(radii[idx])
    if abs(c1 - c0) < _F64_EPS:
        return r1, total
    return r0 + (half - c0) / (c1 - c0) * (r1 - r0), total


def rust_compute_hfr_with_sky(crop: np.ndarray, *, level=None, sigma=None) -> float:
    """``ClassicalFocusSolver::compute_hfr_with_sky``: scalar ``level``/``sigma``
    (``CropSky::Uniform``) or per-pixel float32 arrays (``PerPixel``).
    Centroid of the 3-sigma core, unclipped curve of growth, two passes
    (cap ``min(crop)/2 - 1``; aperture ``clip(4*hfr1, 6, cap)``). NaN when no
    positive core flux or a per-pixel sigma is not finite and positive."""
    data = np.asarray(crop, dtype=np.float32)
    h, w = data.shape
    values = data.astype(np.float64)
    if np.ndim(level) == 0:
        lev = float(level)
        core = np.maximum(values - lev - 3.0 * max(float(sigma), 1e-6), 0.0)
        resid = values - lev
    else:
        lev32 = np.asarray(level, dtype=np.float32)
        sig32 = np.asarray(sigma, dtype=np.float32)
        if lev32.shape != data.shape or sig32.shape != data.shape or not bool(
                (np.isfinite(sig32) & (sig32 > 0)).all()):
            return float("nan")
        lev64 = lev32.astype(np.float64)
        core = np.maximum(values - lev64 - 3.0 * sig32.astype(np.float64), 0.0)
        resid = values - lev64
    core = np.where(np.isnan(core), 0.0, core)
    ys, xs = np.mgrid[0:h, 0:w]
    core_total = _f64_sum(core)
    if core_total <= _F64_EPS:
        return float("nan")
    cx = _f64_sum(xs * core) / core_total
    cy = _f64_sum(ys * core) / core_total
    dx = xs.astype(np.float64) - cx
    dy = ys.astype(np.float64) - cy
    radii = np.sqrt(dx * dx + dy * dy).ravel()
    order = np.argsort(radii, kind="stable")
    radii, resid = radii[order], resid.ravel()[order]
    hard_cap = min(w, h) / 2.0 - 1.0
    first = _cog_half_radius_rust(radii, resid, min(hard_cap, 20.0))
    if first is None or not first[1] > 0.0:
        return float("nan")
    aperture = min(max(4.0 * first[0], 6.0), hard_cap)
    second = _cog_half_radius_rust(radii, resid, aperture)
    return float("nan") if second is None else float(second[0])


def rust_second_moment_ellipticity(crop: np.ndarray, sky) -> np.float32:
    """quality.rs ``second_moment_ellipticity`` (float32, sequential sums)."""
    data = np.asarray(crop, dtype=np.float32)
    h, w = data.shape
    sky = np.broadcast_to(np.asarray(sky, dtype=np.float32), data.shape)
    f = np.fmax(data - sky, np.float32(0.0)).ravel()
    ys, xs = np.mgrid[0:h, 0:w]
    xs = xs.ravel().astype(np.float32)
    ys = ys.ravel().astype(np.float32)
    total = _f32_sum(f)
    if total <= _F32_EPS:
        return np.float32(0.0)
    cx = np.float32(_f32_sum(xs * f) / total)
    cy = np.float32(_f32_sum(ys * f) / total)
    dx = xs - cx
    dy = ys - cy
    mxx = np.float32(_f32_sum(f * dx * dx) / total)
    myy = np.float32(_f32_sum(f * dy * dy) / total)
    mxy = np.float32(_f32_sum(f * dx * dy) / total)
    half_diff = np.float32((mxx - myy) * np.float32(0.5))
    common = np.float32(np.sqrt(half_diff * half_diff + mxy * mxy))
    mean = np.float32((mxx + myy) * np.float32(0.5))
    a2 = max(np.float32(mean + common), np.float32(0.0))
    b2 = max(np.float32(mean - common), np.float32(0.0))
    a = np.float32(np.sqrt(a2))
    if a <= _F32_EPS:
        return np.float32(0.0)
    return np.float32(min(max(np.float32(1.0) - np.float32(np.sqrt(b2)) / a, 0.0), 1.0))


def _rust_round(value: float) -> int:
    """f32::round (half away from zero) of a float32 value, as an int."""
    v = float(value)
    return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(-v + 0.5))


def rust_extract_crop(width: int, height: int, cx: float, cy: float, crop_size: int) -> tuple:
    """quality.rs ``extract_crop`` window: ``(start_x, start_y, cw, ch)``."""
    cw, ch = min(crop_size, width), min(crop_size, height)
    centre_x = int(min(max(_rust_round(cx), 0), max(width - 1, 0)))
    centre_y = int(min(max(_rust_round(cy), 0), max(height - 1, 0)))
    start_x = min(max(centre_x - cw // 2, 0), width - cw)
    start_y = min(max(centre_y - ch // 2, 0), height - ch)
    return start_x, start_y, cw, ch


@dataclass
class FocusGateResult:
    used: list
    tallies: dict
    gate_diagnostics: dict
    records: list
    gates: StarGateParams

    @property
    def star_gates_config(self) -> dict:
        return self.gates.config()

    @property
    def star_gates_config_sha256(self) -> str:
        return self.gates.config_sha256()


def focus_star_gates(pixels: np.ndarray, candidates: list, *, field_kind: str, level=None, sigma=None,
                     rms=None, masks: Optional[dict] = None, invalid_mask: Optional[np.ndarray] = None,
                     saturation_adu: Optional[float] = None, gates: StarGateParams = StarGateParams(),
                     has_analysis_descriptor: bool = False) -> FocusGateResult:
    """Port of quality.rs ``measure_stars`` for already sorted ``candidates``.

    ``field_kind="uniform"`` is the legacy path (gate sky = crop median and
    MAD with the 1e-6 floor; HFR on the legacy crop sky); ``"map"`` uses the
    parent maps ``level``/``rms`` over the crop. ``masks`` holds the
    bridge's ``bad_pixel``/``invalid``/``edge``/``native_saturation`` planes;
    ``invalid_mask`` is ``masked_local_v2``'s M_inv (tally
    ``invalid_noise_support``). Gate order, tallies and the 48 examined /
    16 used caps follow the Rust code; a regrouped family skips gate 2b and
    a blend-flagged, non-exempt family tallies ``deblended``.
    """
    data = np.asarray(pixels, dtype=np.float32)
    h, w = data.shape
    masks = masks or {}
    measurement_invalid = np.zeros((h, w), dtype=bool)
    for name in ("bad_pixel", "invalid", "edge"):
        if name in masks:
            measurement_invalid |= np.asarray(masks[name]).astype(bool)
    native = np.asarray(masks["native_saturation"]).astype(bool) if "native_saturation" in masks else None
    inv = None if invalid_mask is None else np.asarray(invalid_mask, dtype=bool)
    sat_threshold = None
    if not has_analysis_descriptor and saturation_adu is not None and np.isfinite(saturation_adu) \
            and saturation_adu > 0:
        sat_threshold = np.float32(np.float32(saturation_adu) * np.float32(FOCUS_SATURATION_FRACTION_OF_MAX))
    if field_kind == "map":
        level_map = np.asarray(level, dtype=np.float32)
        rms_map = np.asarray(rms, dtype=np.float32)
    elif field_kind != "uniform":
        raise ValueError(f"field_kind must be 'uniform' or 'map', got {field_kind!r}")
    tallies = {k: 0 for k in FOCUS_TALLY_KEYS}
    used, records = [], []
    for star in candidates[:FOCUS_MAX_STARS_EXAMINED]:
        if len(used) >= FOCUS_MAX_STARS_USED:
            break
        rec = {"x": float(star.x), "y": float(star.y), "family_size": int(star.family_size), "rejections": []}
        records.append(rec)
        family = star.family_size >= 2

        def reject(reason):
            tallies[reason] += 1
            rec["rejections"].append(reason)

        if sat_threshold is not None and np.float32(star.peak) >= sat_threshold:
            reject("saturated")
        if (star.min_x <= FOCUS_EDGE_MARGIN_PX or star.min_y <= FOCUS_EDGE_MARGIN_PX
                or star.max_x + FOCUS_EDGE_MARGIN_PX >= max(w - 1, 0)
                or star.max_y + FOCUS_EDGE_MARGIN_PX >= max(h - 1, 0)):
            reject("edge_truncated")
        if star.deblended and not gates.accept_deblended and not family:
            reject("deblended")
        if rec["rejections"]:
            continue
        extent = max(star.max_x - star.min_x + 1, star.max_y - star.min_y + 1)
        crop_size = int(min(max(2 * extent, FOCUS_CROP_MIN_PX), FOCUS_CROP_MAX_PX))
        x0, y0, cw, ch = rust_extract_crop(w, h, star.x, star.y, crop_size)
        window = (slice(y0, y0 + ch), slice(x0, x0 + cw))
        rec["crop"] = [x0, y0, cw, ch]
        masked = bool(measurement_invalid[window].any())
        saturated_support = bool(native[window].any()) if native is not None else False
        invalid_support = bool(inv[window].any()) if inv is not None else False
        if masked:
            reject("masked_support")
        if saturated_support:
            reject("saturated")
        if invalid_support:
            reject("invalid_noise_support")
        if masked or saturated_support or invalid_support:
            continue
        crop = data[window]
        if field_kind == "uniform":
            sky, sky_rms = rust_local_sky_stats(crop)
            noise = max(sky_rms, np.float32(1e-6))
            level_c = np.full(crop.shape, sky, dtype=np.float32)
            noise_c = np.full(crop.shape, noise, dtype=np.float32)
        else:
            level_c = level_map[window]
            noise_c = rms_map[window]
        cx = np.float32(min(max(np.float32(np.float32(star.x) - np.float32(x0)), 0.0), cw - 1))
        cy = np.float32(min(max(np.float32(np.float32(star.y) - np.float32(y0)), 0.0), ch - 1))
        r_ap = max(np.float32(np.float32(min(cw, ch)) / np.float32(8.0)), np.float32(4.0))
        yy, xx = np.mgrid[0:ch, 0:cw]
        ddx = xx.astype(np.float32) - cx
        ddy = yy.astype(np.float32) - cy
        aperture = (ddx * ddx + ddy * ddy) <= np.float32(r_ap * r_ap)
        resid_c = (crop - level_c).astype(np.float32)
        ap_signal = _f32_sum(resid_c[aperture])
        n_ap = int(aperture.sum())
        core_pixels = int(np.sum(resid_c[aperture] > np.float32(3.0) * noise_c[aperture]))
        if field_kind == "uniform":
            snr = np.float32(ap_signal / np.float32(noise * np.float32(np.sqrt(np.float32(max(n_ap, 1))))))
        else:
            ap_variance = _f32_sum(noise_c[aperture] * noise_c[aperture])
            snr = np.float32(ap_signal / np.float32(np.sqrt(max(ap_variance, _F32_MIN_POSITIVE))))
        rec["snr"] = float(snr)
        if snr < np.float32(FOCUS_MIN_APERTURE_SNR):
            reject("low_snr")
        if core_pixels < FOCUS_MIN_CORE_PIXELS:
            reject("too_few_pixels")
        ellipticity = rust_second_moment_ellipticity(crop, level_c)
        rec["ellipticity"] = float(ellipticity)
        if ellipticity > np.float32(FOCUS_MAX_ELLIPTICITY):
            reject("elongated")
        if field_kind == "uniform" and gates.blend_test == BLEND_TEST_PEAK_COUNT_V1:
            result = blend_gate(resid_c, noise_c, gates, v1_values=crop,
                                v1_threshold=np.float32(sky + np.float32(5.0) * noise))
        else:
            result = blend_gate(resid_c, noise_c, gates)
        rec["blend"] = result.to_json()
        if result.blended:
            reject("deblended" if family else "blended")
        if rec["rejections"]:
            continue
        if field_kind == "uniform":
            lev, sg = rust_legacy_crop_sky(crop)
            hfr = rust_compute_hfr_with_sky(crop, level=lev, sigma=sg)
        else:
            hfr = rust_compute_hfr_with_sky(crop, level=level_c, sigma=noise_c)
        if np.isfinite(hfr) and hfr > 0.0:
            used.append({"x": float(np.float32(star.x)), "y": float(np.float32(star.y)),
                         "hfr_px": float(np.float32(hfr)), "family_size": int(star.family_size),
                         "annulus_exempt": bool(result.exempt)})
            rec["used"] = True
        else:
            reject("invalid_hfr")
    diagnostics = {"annulus_exempt": sum(1 for u in used if u["annulus_exempt"]),
                   "regrouped_accepted": sum(1 for u in used if u["family_size"] >= 2)}
    return FocusGateResult(used, tallies, diagnostics, records, gates)


def sort_focus_candidates(candidates: list) -> list:
    """quality.rs ordering: flux, then peak, then area, all descending (stable)."""
    def key(c):
        peak = float(c.peak)
        return (-float(c.flux), -(math.inf if math.isnan(peak) else peak), -int(c.area))

    return sorted(candidates, key=key)


def deblend_families(segmap: np.ndarray, merged) -> list:
    """Port of ``achernar_sep::deblend_families`` from a SEP segmentation map.

    Pixel ``p`` belongs to object ``segmap[p] - 1``; only children
    (``merged[i]``, SEP ``OBJ_MERGED``) take part. Children whose footprints
    are 8-connected are united; families are listed by their lowest-index
    member, members ascending; single-member families are included.
    """
    seg = np.asarray(segmap)
    merged = np.asarray(merged, dtype=bool)
    if seg.size == 0 or merged.size == 0 or not merged.any():
        return []
    idx = seg.astype(np.int64) - 1
    valid = (idx >= 0) & (idx < merged.size)
    owner = np.full(seg.shape, -1, dtype=np.int64)
    owner[valid] = np.where(merged[idx[valid]], idx[valid], -1)
    parent = list(range(merged.size))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    h, w = seg.shape
    # Right, below-left, below and below-right cover all 8 neighbours once per pair.
    for dx, dy in ((1, 0), (-1, 1), (0, 1), (1, 1)):
        x_lo, x_hi = max(0, -dx), w - max(0, dx)
        if x_hi <= x_lo or h - dy <= 0:
            continue
        a = owner[0:h - dy, x_lo:x_hi].ravel()
        b = owner[dy:h, x_lo + dx:x_hi + dx].ravel()
        sel = (a >= 0) & (b >= 0) & (a != b)
        if not sel.any():
            continue
        for pa, pb in np.unique(np.stack([a[sel], b[sel]], axis=1), axis=0).tolist():
            ra, rb = root(pa), root(pb)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    families, family_of_root = [], {}
    for i in np.flatnonzero(merged).tolist():
        r = root(i)
        if r not in family_of_root:
            family_of_root[r] = len(families)
            families.append([])
        families[family_of_root[r]].append(i)
    return families


def sep_catalog_rows(objects, pixels: np.ndarray, sep) -> list:
    """SEP objects as Rust ``CatalogSource`` rows (float32 x/y/flux,
    ``peak_raw`` = raw input at SEP's unfiltered peak pixel)."""
    data = np.asarray(pixels, dtype=np.float32)
    rows = []
    for o in objects:
        flag = int(o["flag"])
        rows.append({
            "x": float(np.float32(o["x"])), "y": float(np.float32(o["y"])),
            "xmin": int(o["xmin"]), "xmax": int(o["xmax"]), "ymin": int(o["ymin"]), "ymax": int(o["ymax"]),
            "npix": int(o["npix"]), "flux": float(np.float32(o["flux"])),
            "peak_residual": float(np.float32(o["peak"])),
            "peak_raw": float(data[int(o["ypeak"]), int(o["xpeak"])]),
            "a": float(np.float32(o["a"])), "b": float(np.float32(o["b"])), "theta": float(np.float32(o["theta"])),
            "flags": flag, "deblended": bool(flag & sep.OBJ_MERGED), "truncated": bool(flag & sep.OBJ_TRUNC),
            "xpeak": int(o["xpeak"]), "ypeak": int(o["ypeak"]),
        })
    return rows


def family_candidate(rows: list, members: list) -> FocusCandidate:
    """One candidate per family (spec §2.2): union footprint/bbox, summed
    flux and npix, flux-weighted x/y (non-positive flux weight 0; all 0 ->
    unweighted mean), maximum raw peak."""
    xs = np.array([np.float32(rows[i]["x"]) for i in members], dtype=np.float64)
    ys = np.array([np.float32(rows[i]["y"]) for i in members], dtype=np.float64)
    flux = np.array([rows[i]["flux"] for i in members], dtype=np.float32)
    weight = np.where(flux > 0, flux, np.float32(0.0)).astype(np.float64)
    if _f64_sum(weight) > 0:
        x = _f64_sum(weight * xs) / _f64_sum(weight)
        y = _f64_sum(weight * ys) / _f64_sum(weight)
    else:
        x, y = _f64_sum(xs) / len(members), _f64_sum(ys) / len(members)
    return FocusCandidate(
        x=float(np.float32(x)), y=float(np.float32(y)),
        area=int(sum(rows[i]["npix"] for i in members)),
        flux=float(_f32_sum(flux)),
        peak=float(np.max(np.array([rows[i]["peak_raw"] for i in members], dtype=np.float32))),
        min_x=min(rows[i]["xmin"] for i in members), max_x=max(rows[i]["xmax"] for i in members),
        min_y=min(rows[i]["ymin"] for i in members), max_y=max(rows[i]["ymax"] for i in members),
        deblended=True, family_size=len(members), members=tuple(members),
    )


def focus_candidates_from_catalog(rows: list, segmap: Optional[np.ndarray], gates: StarGateParams) -> tuple:
    """quality.rs ``candidates_from_catalog`` plus spec §2 regrouping.

    Returns ``(candidates, families)``: with regrouping, each multi-member
    family replaces its children at the position of its lowest-index member;
    the area (<= 5000 px) and positive-finite-flux filter then applies to
    every candidate, families included. ``families`` lists every family
    (single-member ones included) as found by :func:`deblend_families`.
    """
    families = []
    member_of = {}
    if gates.regroups and segmap is not None:
        families = deblend_families(segmap, [r["deblended"] for r in rows])
        for fam in families:
            if len(fam) >= 2:
                for i in fam:
                    member_of[i] = fam
    out = []
    for i, r in enumerate(rows):
        if i in member_of:
            fam = member_of[i]
            if i != fam[0]:
                continue
            cand = family_candidate(rows, fam)
        else:
            cand = FocusCandidate(x=r["x"], y=r["y"], area=int(r["npix"]), flux=r["flux"], peak=r["peak_raw"],
                                  min_x=r["xmin"], max_x=r["xmax"], min_y=r["ymin"], max_y=r["ymax"],
                                  deblended=bool(r["deblended"]))
        if cand.area <= FOCUS_MAX_SOURCE_AREA and np.isfinite(cand.flux) and cand.flux > 0:
            out.append(cand)
    return out, families


def focus_quality(pixels: np.ndarray, candidates: list, *, gates: StarGateParams = StarGateParams(),
                  **gate_kwargs) -> dict:
    """quality.rs ``estimate_quality_with_policy`` after detection: sort the
    candidates, run :func:`focus_star_gates`, take the median HFR."""
    stars = sort_focus_candidates(candidates)
    out = {"star_count_detected": len(stars), "candidates": [c.to_json() for c in stars],
           "star_gates_config": gates.config(), "star_gates_config_sha256": gates.config_sha256()}
    if not stars:
        out.update(status="NoSources", hfr_px=None, star_count_used=0, used_stars=[],
                   star_rejections={k: 0 for k in FOCUS_TALLY_KEYS},
                   gate_diagnostics={"annulus_exempt": 0, "regrouped_accepted": 0}, records=[])
        return out
    result = focus_star_gates(pixels, stars, gates=gates, **gate_kwargs)
    hfrs = np.sort(np.array([u["hfr_px"] for u in result.used], dtype=np.float32))
    hfr = float(_median_sorted_f32(hfrs)) if hfrs.size else None
    valid = hfr is not None and np.isfinite(hfr) and hfr > 0
    out.update(status="Valid" if valid else "NoUsableStars", hfr_px=hfr if valid else None,
               star_count_used=len(result.used), used_stars=result.used, star_rejections=result.tallies,
               gate_diagnostics=result.gate_diagnostics, records=result.records)
    return out


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
    blend_test: str = BLEND_TEST_DIP_ANNULUS_V2,
) -> StarMeasurement:
    """Measure one star crop and decide whether it may contribute a width label.

    ``blend_test`` selects gate 5 (docs/bge-v2-spec.md §1): ``"dip_annulus_v2"``
    (default; saddle/bottleneck merge of the >5-sigma peaks plus the annulus
    exemption for obstructed, spider-cut donuts, on ``crop - sky`` with the
    crop noise) or ``"peak_count_v1"`` (the previous two-core-peak test,
    unchanged).

    Custom crop path: one scalar sky for the crop (the caller's, or the
    ``python_crop_iterative_clip`` estimate). The gates reject saturated,
    edge-truncated, low-SNR, blended/elongated stars *before* they can bias the
    aggregate HFR. When the crop noise is unresolved (zero-MAD plateau, or a
    caller-supplied ``sky_rms <= 0``) the SNR, core-count and blend gates
    cannot be evaluated in sigma units: the star is rejected with
    ``noise_unresolved`` and ``snr`` is ``None`` instead of dividing by an
    invented epsilon."""
    if blend_test not in BLEND_TESTS:
        raise ValueError(f"blend_test must be one of {BLEND_TESTS}, got {blend_test!r}")
    crop = np.asarray(crop, dtype=np.float64)
    h, w = crop.shape
    rejections: list[str] = []
    if sky is None or sky_rms is None:
        estimate = estimate_background_detailed(crop)
        if estimate.level is None:
            return StarMeasurement(
                w / 2.0, h / 2.0, 0.0, float("nan"), None, None, None, None, 0.0, False,
                [Rejection.NO_USABLE_PIXELS.value],
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
    # drops it). v2 separates the peaks of one ring (shallow dips along the
    # ridge, or arcs of an annulus) from a real saddle between two stars.
    blend = None
    if noise_resolved:
        if blend_test == BLEND_TEST_PEAK_COUNT_V1:
            if _count_core_peaks(crop, sky, sky_rms) >= 2:
                rejections.append(Rejection.BLENDED.value)
        else:
            result = blend_gate(crop - sky, sky_rms, StarGateParams(blend_test=blend_test))
            blend = result.to_json()
            if result.blended:
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
        blend=blend,
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
        saturated_star_fraction=(saturated / len(measurements)) if measurements else None,
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
    # Spec §2: a regrouped family of SEP children (family_size >= 2, the
    # catalog indices in family_members) measured as one source, and its
    # gate-5 outcome (BlendGateResult.to_json()).
    family_size: int = 1
    family_members: list = field(default_factory=list)
    blend: Optional[dict] = None


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


def labeller_label_version(background_params: SepBackgroundParams, gates: "StarGateParams") -> str:
    """``fqe_sep_spatial_v2`` when every v1 policy is selected (frame_wide_v1,
    peak_count_v1, no regrouping: the pre-v2 labels byte for byte), else
    :data:`LABEL_VERSION`."""
    if (background_params.noise_policy == NOISE_POLICY_FRAME_WIDE_V1
            and gates.blend_test == BLEND_TEST_PEAK_COUNT_V1 and not gates.regroups):
        return LABEL_VERSION_V2
    return LABEL_VERSION


def _empty_frame_label(background: SpatialBackground, detection: dict, reasons, units, gate, threshold,
                       gain, label_method, gates: Optional["StarGateParams"] = None) -> FrameLabel:
    gates = gates or StarGateParams()
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
        saturated_star_fraction=None,
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
        label_version=labeller_label_version(background.params, gates),
        star_gates=gates.config(),
        star_gates_config_sha256=gates.config_sha256(),
        gate_diagnostics={"annulus_exempt": 0, "regrouped_accepted": 0},
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
    blend_test: str = BLEND_TEST_DIP_ANNULUS_V2,
    regroup_deblended: bool = True,
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
    ``OBJ_MERGED`` -> ``deblended`` unless ``accept_deblended_for_focus``;
    ``OBJ_TRUNC`` / ``APER_TRUNC`` -> ``edge_truncated``.

    BGE v2 (docs/bge-v2-spec.md). ``background_params.noise_policy`` selects
    the noise map; under ``masked_local_v2`` locally invalid R(x,y) pixels
    join the extraction mask and a source whose footprint touches one is
    ``invalid_noise_support``. With ``regroup_deblended`` (default) and not
    ``accept_deblended_for_focus``, SEP children whose footprints are
    8-connected are regrouped into one family source (flux-weighted position,
    union footprint; ``a``/``b`` from the union footprint's moments) measured
    like any source except the deblending policy; ``blend_test``
    (``"dip_annulus_v2"`` by default, or ``"peak_count_v1"``) then judges the
    family on its Rust-style adaptive crop of data - B(x,y) against R(x,y),
    and a blend-flagged family that the annulus exemption does not accept is
    ``deblended``. Single-member families keep the v1 behaviour. On this SEP
    path the blend gate judges regrouped families only: unregrouped sources
    keep SEP's own deblending as their blend test, as in v1. With
    ``peak_count_v1``, ``regroup_deblended=False`` and ``frame_wide_v1`` the
    labels equal the pre-v2 labels (``label_version`` ``fqe_sep_spatial_v2``).

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
    if source_mask_background:
        _require_int("source_mask_dilation_px", source_mask_dilation_px, 0)
    gates = StarGateParams(blend_test=blend_test, regroup_deblended=bool(regroup_deblended),
                           accept_deblended=bool(detection_params.accept_deblended_for_focus))

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
                                      gain_e_per_adu, label_method, gates)
    combined, saturated_mask, malformed = _combine_detector_mask(
        data.shape, data, detector_mask, native_saturation_mask
    )
    if malformed:
        background = _unavailable_background(background_params, version, units, tuple(data.shape),
                                             [REASON_MALFORMED_GEOMETRY])
        return [], _empty_frame_label(background, detection, background.reasons, units, gate, threshold,
                                      gain_e_per_adu, label_method, gates)

    background = estimate_spatial_background(data, detector_mask=combined, params=background_params, units=units,
                                             detection_params=detection_params)
    if not (background.level_measured and background.noise_measured):
        return [], _empty_frame_label(background, detection, background.reasons, units, gate, threshold,
                                      gain_e_per_adu, label_method, gates)

    def extraction_mask(bkg: SpatialBackground) -> np.ndarray:
        # masked_local_v2: locally invalid R(x,y) pixels join the extraction mask.
        return combined if bkg.invalid_mask is None else (combined | bkg.invalid_mask)

    def residual_of(bkg: SpatialBackground) -> np.ndarray:
        # SEP extraction has no background term; masked pixels are zeroed, as
        # SEP itself does inside extract (apply_mask_line).
        return np.ascontiguousarray(np.where(extraction_mask(bkg), np.float32(0.0), data - bkg.background_map),
                                    dtype=np.float32)

    try:
        data_sub = residual_of(background)
        mask_u8 = np.ascontiguousarray(extraction_mask(background), dtype=np.uint8)
        objects, segmap = _sep_extract(sep, data_sub, background.extraction_rms(), mask_u8, detection_params)
        if source_mask_background:
            from scipy.ndimage import binary_dilation

            source_mask = segmap > 0
            if source_mask_dilation_px > 0:
                source_mask = binary_dilation(source_mask, iterations=int(source_mask_dilation_px))
            refined = estimate_spatial_background(
                data, detector_mask=combined, params=background_params, source_mask=source_mask, units=units,
                detection_params=detection_params,
            )
            if not (refined.level_measured and refined.noise_measured):
                return [], _empty_frame_label(refined, detection, refined.reasons, units, gate, threshold,
                                              gain_e_per_adu, label_method, gates)
            background = refined
            data_sub = residual_of(background)
            mask_u8 = np.ascontiguousarray(extraction_mask(background), dtype=np.uint8)
            objects, segmap = _sep_extract(sep, data_sub, background.extraction_rms(), mask_u8, detection_params)
    except Exception as exc:
        detection["backend_error"] = str(exc)
        return [], _empty_frame_label(background, detection, ["detection_backend_failure"], units, gate,
                                      threshold, gain_e_per_adu, label_method, gates)
    detection["status"] = STATUS_MEASURED
    detection["source_mask_background"] = {
        "enabled": bool(source_mask_background),
        "dilation_px": int(source_mask_dilation_px) if source_mask_background else None,
    }

    back_map, rms_map = background.background_map, background.rms_map
    invalid_mask = background.invalid_mask
    h, w = data.shape
    sources: list[SepSource] = []
    flag_counts: Counter = Counter()
    measure_mask = extraction_mask(background)
    measure_mask_u8 = np.ascontiguousarray(measure_mask, dtype=np.uint8)

    # Measurement units: one per SEP object, except that (spec §2) each
    # deblend family with >= 2 children becomes one unit at the position of
    # its lowest-index member.
    families = []
    if gates.regroups and len(objects) > 0:
        families = deblend_families(segmap, [bool(int(f) & int(sep.OBJ_MERGED)) for f in objects["flag"]])
    member_of = {i: fam for fam in families if len(fam) >= 2 for i in fam}
    detection["objects_detected"] = int(len(objects))
    detection["regrouped_families"] = sum(1 for fam in families if len(fam) >= 2)
    detection["single_member_families"] = sum(1 for fam in families if len(fam) == 1)
    measure_units = []
    nonfinite_centroids = 0
    for i in range(len(objects)):
        if i in member_of:
            fam = member_of[i]
            if i != fam[0]:
                continue
            unit = _family_unit(objects, segmap, data_sub, measure_mask, fam)
        else:
            o = objects[i]
            unit = {"members": [i], "x": float(o["x"]), "y": float(o["y"]), "a": float(o["a"]),
                    "b": float(o["b"]), "flag": int(o["flag"]),
                    "bbox": (int(o["xmin"]), int(o["xmax"]), int(o["ymin"]), int(o["ymax"]))}
        if not (np.isfinite(unit["x"]) and np.isfinite(unit["y"])):
            # A zero-flux SEP object has no centroid (x = y = NaN): nothing
            # to measure. Counted, never measured.
            nonfinite_centroids += 1
            continue
        measure_units.append(unit)
    detection["objects_nonfinite_centroid"] = nonfinite_centroids

    if measure_units:
        x = np.array([u["x"] for u in measure_units], dtype=np.float64)
        y = np.array([u["y"] for u in measure_units], dtype=np.float64)
        a_axis = np.array([u["a"] for u in measure_units], dtype=np.float64)
        # Half-light radius, bounded to 6x the isophotal semi-major axis. SEP
        # measures the normalisation flux inside that aperture; objects["flux"]
        # is isophotal and undercounts Moffat wings, biasing HFR low.
        rmax = np.maximum(SEP_HFR_APERTURE_A_FACTOR * a_axis, SEP_HFR_APERTURE_MIN_PX)
        rhalf, rflag = sep.flux_radius(
            data_sub, x, y, rmax, 0.5, normflux=None, mask=measure_mask_u8, maskthresh=0.0, subpix=SEP_SUBPIX
        )
        aperture_radius = np.maximum(SEP_FLUX_APERTURE_A_FACTOR * a_axis, SEP_FLUX_APERTURE_MIN_PX)
        aperture_flux, aperture_fluxerr, aperture_flag = sep.sum_circle(
            data_sub,
            x,
            y,
            aperture_radius,
            err=rms_map,
            gain=gain_e_per_adu,
            mask=measure_mask_u8,
            maskthresh=0.0,
            subpix=SEP_SUBPIX,
        )
        other_masked = combined & ~saturated_mask

        for i, unit in enumerate(measure_units):
            rej: list[str] = []
            family_size = len(unit["members"])
            a = float(unit["a"])
            b = float(unit["b"])
            ellip = float(1.0 - b / a) if a > 0 else 0.0
            cx, cy = float(x[i]), float(y[i])
            ix = int(np.clip(np.floor(cx + 0.5), 0, w - 1))
            iy = int(np.clip(np.floor(cy + 0.5), 0, h - 1))
            fluxerr = float(aperture_fluxerr[i])
            snr = float(aperture_flux[i]) / fluxerr if np.isfinite(fluxerr) and fluxerr > 0 else float("nan")

            flag = int(unit["flag"])
            r_flag = int(rflag[i])
            a_flag = int(aperture_flag[i])
            for member in unit["members"]:
                member_flag = int(objects["flag"][member])
                for attr, name in SEP_FLAG_NAMES:
                    if member_flag & int(getattr(sep, attr)):
                        flag_counts[name] += 1
            for attr, name in SEP_APER_FLAG_NAMES:
                if (r_flag | a_flag) & int(getattr(sep, attr)):
                    flag_counts[name] += 1

            deblended = bool(flag & sep.OBJ_MERGED)
            if deblended and not detection_params.accept_deblended_for_focus and family_size < 2:
                rej.append(Rejection.DEBLENDED.value)
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
            invalid_noise = invalid_mask is not None and bool(invalid_mask[window][inside].any())
            if other_masked[window][inside].any() or (
                not evidence and not invalid_noise and (r_flag | a_flag) & sep.APER_HASMASKED
            ):
                rej.append(Rejection.MASKED_SUPPORT.value)
            if invalid_noise:
                rej.append(Rejection.INVALID_NOISE_SUPPORT.value)

            if not np.isfinite(snr) or snr < min_snr:
                rej.append(Rejection.LOW_SNR.value)
            if ellip > max_ellipticity:
                rej.append(Rejection.ELONGATED.value)

            blend = None
            if family_size >= 2:
                residual, noise = _labeller_blend_crop(data, back_map, rms_map, measure_mask, unit["bbox"], cx, cy)
                result = blend_gate(residual, noise, gates)
                blend = result.to_json()
                if result.blended:
                    # The detector's split is confirmed (spec §2.3).
                    rej.append(Rejection.DEBLENDED.value)

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
                    family_size=family_size,
                    family_members=[int(m) for m in unit["members"]] if family_size >= 2 else [],
                    blend=blend,
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
        saturated_star_fraction=(saturated / len(sources)) if sources else None,
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
        label_version=labeller_label_version(background_params, gates),
        star_gates=gates.config(),
        star_gates_config_sha256=gates.config_sha256(),
        gate_diagnostics={
            "annulus_exempt": sum(1 for s in used if s.blend and s.blend.get("exempt")),
            "regrouped_accepted": sum(1 for s in used if s.family_size >= 2),
        },
    )
    return sources, label


def _family_unit(objects, segmap, data_sub, excluded, members) -> dict:
    """A regrouped family as one measurement unit (spec §2.2).

    Position: children's ``x, y`` weighted by their SEP ``flux`` (children
    with ``flux <= 0`` weigh 0; all zero -> unweighted mean). Footprint and
    bounding box: the union. ``a``/``b`` (which set the HFR and flux
    apertures and the ellipticity gate): second moments of the unfiltered
    residual, clipped at 0, over the union footprint — SEP's isophotal axes of
    one child do not describe the family. ``flag``: the OR of the children.
    """
    xs = np.asarray(objects["x"][members], dtype=np.float64)
    ys = np.asarray(objects["y"][members], dtype=np.float64)
    flux = np.asarray(objects["flux"][members], dtype=np.float64)
    weight = np.where(flux > 0, flux, 0.0)
    if weight.sum() > 0:
        x, y = float((weight * xs).sum() / weight.sum()), float((weight * ys).sum() / weight.sum())
    else:
        x, y = float(xs.mean()), float(ys.mean())
    footprint = np.isin(segmap, np.asarray(members) + 1) & ~excluded
    fy, fx = np.nonzero(footprint)
    values = np.clip(data_sub[fy, fx].astype(np.float64), 0.0, None)
    total = values.sum()
    a = b = 0.0
    if total > 0:
        mx, my = (fx * values).sum() / total, (fy * values).sum() / total
        x2 = (values * (fx - mx) ** 2).sum() / total
        y2 = (values * (fy - my) ** 2).sum() / total
        xy = (values * (fx - mx) * (fy - my)).sum() / total
        common = np.sqrt(((x2 - y2) / 2.0) ** 2 + xy * xy)
        a = float(np.sqrt(max((x2 + y2) / 2.0 + common, 0.0)))
        b = float(np.sqrt(max((x2 + y2) / 2.0 - common, 0.0)))
    flag = 0
    for m in members:
        flag |= int(objects["flag"][m])
    bbox = (int(min(objects["xmin"][m] for m in members)), int(max(objects["xmax"][m] for m in members)),
            int(min(objects["ymin"][m] for m in members)), int(max(objects["ymax"][m] for m in members)))
    return {"members": list(members), "x": x, "y": y, "a": a, "b": b, "flag": flag, "bbox": bbox}


def _labeller_blend_crop(data, back_map, rms_map, excluded, bbox, cx, cy):
    """The family's measurement crop for gate 5: the Rust adaptive crop
    (2x the bounding-box extent, clamped to 32..128 px) about ``(cx, cy)``;
    returns ``(data - B, R)`` over it, excluded pixels NaN."""
    h, w = data.shape
    min_x, max_x, min_y, max_y = bbox
    extent = max(max_x - min_x + 1, max_y - min_y + 1)
    size = int(min(max(2 * extent, FOCUS_CROP_MIN_PX), FOCUS_CROP_MAX_PX))
    x0, y0, cw, ch = rust_extract_crop(w, h, np.float32(cx), np.float32(cy), size)
    window = (slice(y0, y0 + ch), slice(x0, x0 + cw))
    residual = (data[window] - back_map[window]).astype(np.float32)
    residual[excluded[window]] = np.nan
    return residual, rms_map[window]
