"""Python background methods and the shared Python detection/measurement path.

Methods (identifiers used in results):

``python_sep``
    Reference: the pinned SEP 1.4.1 wheel called directly with the same
    canonical configuration as the Rust defaults. Background on the frame
    with mask = invalid | bad_pixel | edge | native_saturation | non-finite;
    extraction on ``I - B(x,y)`` with masked pixels set to 0, the same mask
    passed to ``sep.extract``, ``err = R(x,y)`` (as the daemon adapter does).
``python_global_scalar``
    Python comparator, *not* the Rust legacy method: one iteratively
    sigma-clipped median and standard deviation over all usable pixels,
    evaluated as a constant B and R everywhere. It shows what a scalar
    background does to the same detection and HFR path. The Rust legacy
    method (``legacy_global_clip``) is measured through the bridge.
``oracle_true_background``
    Oracle comparator: the simulated truth B(x,y) and sky noise R(x,y) fed to
    the same detection and HFR path. It separates background-model error
    from detection/estimator effects. It is not a method.

Detection, acceptance and HFR are identical for all three, so differences
between them come from the background and noise maps alone. The acceptance
policy (``ACCEPTANCE_POLICY``) is an evaluation-harness policy modelled on
the daemon's star gates; it is not the daemon's policy, which is measured
through the bridge. In particular its blend rule is a "neighbour" rule
(another catalog centroid inside the HFR aperture), not the daemon's gate 5.

``python_sep__n1g1``, ``__n2g1``, ``__n1g2``, ``__n2g2`` (BGE v2 matrix)
    The Python reference of the daemon's SEP path, for Rust-Python parity
    (docs/bge-v2-spec.md §4): the labeller's SEP background
    (``ml.labels.psf_measure.estimate_spatial_background``) with noise policy
    ``n1`` = ``frame_wide_v1`` or ``n2`` = ``masked_local_v2``; ``sep.extract``
    with the daemon's settings and segmentation map; then the labeller's
    Rust-parity focus gates (``psf_measure.focus_quality``, a port of
    quality.rs ``measure_stars``: crop gates, 48 examined / 16 used, gate 5,
    HFR by the daemon's curve of growth) with ``g1`` = ``peak_count_v1``
    without regrouping or ``g2`` = ``dip_annulus_v2`` with SEP deblend
    regrouping. Unlike ``python_sep`` these share no harness policy.

SEP fabricates values in two cases (preflight notes, SEP 1.4.1
``background.c``): with no mesh at >= 50% valid pixels the grid becomes
back = 0, sigma = 1 with ``RETURN_OK``; with every mesh sigma <= 0
``globalrms`` becomes 1.0. Both are detected here from the mask (the 50%
rule is re-applied on SEP's own mesh grid) and from ``R(x,y) <= 0``, and are
reported as unsupported/unresolved instead of being scored. The R(x,y)
validity rules (including D-018's positive-undershoot bound) come from
``ml.labels.psf_measure.rms_map_validity``, shared with the labeler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import config_sha256

SEP_VERSION_EXPECTED = "1.4.1"
#: SEP's ``BACK_MINGOODFRAC``: a mesh with fewer valid pixels is "bad".
SEP_BACK_MIN_GOOD_FRAC = 0.5

DEFAULT_BACKGROUND_PARAMS = {"bw": 64, "bh": 64, "fw": 3, "fh": 3, "fthresh": 0.0, "min_meshes_per_axis": 2}
DEFAULT_DETECTION_PARAMS = {
    "thresh_sigma": 5.0,
    "minarea": 5,
    "filter_type": "conv",
    "filter_kernel": [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
    "deblend_nthresh": 32,
    "deblend_cont": 0.005,
    "clean": True,
    "clean_param": 1.0,
    "pixstack": 300000,
    "accept_deblended_for_focus": False,
}
#: Contract hashes from the daemon (background_model.rs, integration 92d9bb47).
EXPECTED_BACKGROUND_CONFIG_SHA256 = "80eab0fbae4abf9f60f348adb7de3004067a3873cca29f996289e9dc14040b84"
EXPECTED_DETECTION_CONFIG_SHA256 = "4f4499cda7067107c0fe6d64319684c13d1ec62f590b4bee4b346c1459dc99df"

GLOBAL_SCALAR_PARAMS = {"clip_sigma": 3.0, "max_iters": 10, "center": "median", "scale": "std(ddof=0)"}

ACCEPTANCE_POLICY = {
    "policy": "bge_eval_python_acceptance",
    "version": 1,
    "edge_margin_px": 2,
    "reject_sep_truncated": True,
    "reject_deblended": "unless accept_deblended_for_focus",
    "max_ellipticity_1_minus_b_over_a": 0.35,
    "min_aperture_snr": 5.0,
    "snr_aperture_radius": "max(3, 2*hfr_pass1) px; err = R map (or scalar)",
    "hfr": {
        "estimator": "sep.flux_radius",
        "frac": 0.5,
        "subpix": 5,
        "data": "I - B_hat, masked pixels 0",
        "normalisation": "flux inside rmax",
        "pass1_rmax": "min(20, cap)",
        "pass2_rmax": "clip(4*hfr_pass1, 6, cap)",
        "cap_px": 63.0,
    },
    "support": "reject if any masked or non-finite pixel lies within rmax2 + 0.7072 px; native_saturation there tallies 'saturated'",
    "aperture_inside_frame": "reject (edge_truncated) if the pass-2 aperture leaves the frame",
    "neighbour": "reject (blended) if another catalog object's centroid lies within rmax2",
    "max_stars_used": None,
}

HFR_CAP_PX = 63.0
NOISE_REASONS = ("zero_empirical_scatter", "invalid_rms_map")
#: BGE v2 Python matrix: noise policy x star gates (docs/bge-v2-spec.md §4).
PYTHON_SEP_VARIANTS = {
    "python_sep__n1g1": {"noise_policy": "frame_wide_v1", "blend_test": "peak_count_v1", "regroup_deblended": False},
    "python_sep__n2g1": {"noise_policy": "masked_local_v2", "blend_test": "peak_count_v1", "regroup_deblended": False},
    "python_sep__n1g2": {"noise_policy": "frame_wide_v1", "blend_test": "dip_annulus_v2", "regroup_deblended": True},
    "python_sep__n2g2": {"noise_policy": "masked_local_v2", "blend_test": "dip_annulus_v2", "regroup_deblended": True},
}
#: SEP's BIG sentinel: the daemon also masks |value| >= 1e30 (usable_data_value).
SEP_BIG = 1e30
METHOD_LABELS = {
    "python_sep__n1g1": "Python reference of the daemon SEP path: frame_wide_v1 noise, peak_count_v1, no regrouping",
    "python_sep__n2g1": "Python reference of the daemon SEP path: masked_local_v2 noise, peak_count_v1, no regrouping",
    "python_sep__n1g2": "Python reference of the daemon SEP path: frame_wide_v1 noise, dip_annulus_v2 + regrouping",
    "python_sep__n2g2": "Python reference of the daemon SEP path: masked_local_v2 noise, dip_annulus_v2 + regrouping",
    "python_sep": "Python SEP 1.4.1 reference (spatial mesh background + RMS map)",
    "python_global_scalar": "Python comparator: global sigma-clipped scalar background (not the Rust legacy method)",
    "oracle_true_background": "Oracle comparator: simulated truth B and R (not a method)",
}


def sep_background_config_json(params: dict | None = None) -> dict:
    p = dict(DEFAULT_BACKGROUND_PARAMS, **(params or {}))
    return {
        "method": "sep_spatial_mesh",
        "method_version": 1,
        "library": "sep",
        "library_version": SEP_VERSION_EXPECTED,
        "bw": int(p["bw"]),
        "bh": int(p["bh"]),
        "fw": int(p["fw"]),
        "fh": int(p["fh"]),
        "fthresh": float(p["fthresh"]),
        "maskthresh": 0.0,
        "min_meshes_per_axis": int(p["min_meshes_per_axis"]),
    }


def sep_detection_config_json(params: dict | None = None) -> dict:
    p = dict(DEFAULT_DETECTION_PARAMS, **(params or {}))
    return {
        "method": "sep_extract",
        "library": "sep",
        "library_version": SEP_VERSION_EXPECTED,
        "thresh_sigma": float(p["thresh_sigma"]),
        "noise": "background_rms_map",
        "minarea": int(p["minarea"]),
        "filter_type": str(p["filter_type"]),
        "filter_kernel": [[float(v) for v in row] for row in p["filter_kernel"]],
        "deblend_nthresh": int(p["deblend_nthresh"]),
        "deblend_cont": float(p["deblend_cont"]),
        "clean": bool(p["clean"]),
        "clean_param": float(p["clean_param"]),
        "pixstack": int(p["pixstack"]),
        "accept_deblended_for_focus": bool(p["accept_deblended_for_focus"]),
    }


def global_scalar_config_json() -> dict:
    return dict({"method": "python_global_sigma_clip", "method_version": 1}, **GLOBAL_SCALAR_PARAMS)


class MissingPrerequisite(RuntimeError):
    """The pinned SEP wheel is absent or a different version. Tests that need
    it ERROR with this message instead of skipping (prompt section 7)."""


def sep_module():
    """The pinned SEP module, imported only where SEP actually runs."""
    try:
        import sep
    except ImportError as exc:
        raise MissingPrerequisite(f"missing prerequisite: sep=={SEP_VERSION_EXPECTED} ({exc})") from exc
    if sep.__version__ != SEP_VERSION_EXPECTED:
        raise MissingPrerequisite(f"missing prerequisite: sep=={SEP_VERSION_EXPECTED} (found sep {sep.__version__})")
    return sep


# --------------------------------------------------------------------------
# Background estimates
# --------------------------------------------------------------------------


@dataclass
class BackgroundEstimate:
    method: str
    status: str  # completed | unsupported | failed
    level_status: str  # measured | unavailable
    noise_status: str  # measured | unresolved | unavailable
    reasons: list
    config: dict
    config_sha256: str
    spatial: bool = False
    back: object = None  # 2-d float32 map (spatial) or float (scalar)
    rms: object = None
    level_summary: float | None = None
    rms_summary: float | None = None
    level_definition: str = ""
    rms_definition: str = ""
    noise_source: str = "unavailable"
    diagnostics: dict = field(default_factory=dict)
    detail: str | None = None
    #: SEP's RMS map as returned, kept for cross-implementation parity even
    #: when it is not a valid noise measurement. Never scored.
    raw_rms: object = None
    #: BGE v2 variants: the labeller's SpatialBackground behind this estimate
    #: (invalid mask, refinement record, extraction noise).
    spatial_background: object = None
    #: masked_local_v2 refinement record (spec §3.7); None for frame_wide_v1.
    refinement: dict | None = None

    def back_map(self, shape):
        if self.back is None:
            return None
        if np.isscalar(self.back):
            return np.full(shape, self.back, dtype=np.float32)
        return self.back

    def rms_map(self, shape):
        if self.rms is None:
            return None
        if np.isscalar(self.rms):
            return np.full(shape, self.rms, dtype=np.float32)
        return self.rms

    def summary(self) -> dict:
        return {
            "method": self.method,
            "status": self.status,
            "level_status": self.level_status,
            "noise_status": self.noise_status,
            "reasons": list(self.reasons),
            "config_sha256": self.config_sha256,
            "spatial": self.spatial,
            "level": self.level_summary,
            "level_definition": self.level_definition,
            "rms": self.rms_summary,
            "rms_definition": self.rms_definition,
            "noise_source": self.noise_source,
            "diagnostics": self.diagnostics,
            "detail": self.detail,
            "refinement": self.refinement,
        }


def unusable_mask(image: np.ndarray, masks: dict) -> np.ndarray:
    """invalid | bad_pixel | edge | native_saturation | non-finite."""
    bad = ~np.isfinite(image)
    for name in ("invalid", "bad_pixel", "edge", "native_saturation"):
        if name in masks:
            bad = bad | masks[name].astype(bool)
    return bad


def _unavailable(method, config, reasons, detail=None, diagnostics=None) -> BackgroundEstimate:
    return BackgroundEstimate(
        method=method,
        status="unsupported",
        level_status="unavailable",
        noise_status="unavailable",
        reasons=list(reasons),
        config=config,
        config_sha256=config_sha256(config),
        detail=detail,
        diagnostics=diagnostics or {},
    )


def sep_mesh_support(usable: np.ndarray, bw: int, bh: int):
    """SEP's mesh grid (``nx = (w-1)//bw + 1``, partial last meshes) and the
    fraction of usable pixels per mesh; a mesh below 50% is "bad" in SEP."""
    h, w = usable.shape
    nx, ny = (w - 1) // bw + 1, (h - 1) // bh + 1
    frac = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            block = usable[j * bh : min(h, (j + 1) * bh), i * bw : min(w, (i + 1) * bw)]
            frac[j, i] = block.mean()
    return nx, ny, frac


def rms_map_validity(rms: np.ndarray, usable: np.ndarray, bw: int, bh: int):
    """Noise validity of SEP's R(x,y) on usable pixels: ``(reasons, diagnostics)``.

    Delegates to W1's ``ml.labels.psf_measure.rms_map_validity`` so the Python
    reference, the labeler and the daemon apply one rule set (contract and
    D-018): any usable R == 0 -> ``zero_empirical_scatter``; R < 0, non-finite,
    or below 0.5 x the smallest positive filtered mesh sigma (the spline
    undershooting while still positive) -> ``invalid_rms_map``. Either makes
    the noise unresolved and no extraction runs.
    """
    from ml.labels.psf_measure import rms_map_validity as _validity

    reasons, diagnostics = _validity(rms, usable, (bw, bh))
    return list(reasons), dict(diagnostics)


def python_sep_background(image: np.ndarray, masks: dict, params: dict | None = None) -> BackgroundEstimate:
    config = sep_background_config_json(params)
    method = "python_sep"
    if image.ndim != 2 or image.size == 0:
        return _unavailable(method, config, ["malformed_geometry"])
    h, w = image.shape
    bad = unusable_mask(image, masks)
    usable = ~bad
    n_usable = int(usable.sum())
    if n_usable == 0:
        return _unavailable(method, config, ["no_usable_pixels"])
    bw, bh, mm = config["bw"], config["bh"], config["min_meshes_per_axis"]
    if w < mm * bw or h < mm * bh:
        return _unavailable(method, config, ["frame_smaller_than_mesh_grid"],
                            detail=f"{w}x{h} < {mm} meshes of {bw}x{bh} per axis")
    nx, ny, frac = sep_mesh_support(usable, bw, bh)
    good = frac >= SEP_BACK_MIN_GOOD_FRAC
    diag = {
        "mesh_nx": nx,
        "mesh_ny": ny,
        "meshes_below_support": int((~good).sum()),
        "usable_pixels": n_usable,
        "total_pixels": int(image.size),
    }
    if not good.any():
        return _unavailable(method, config, ["insufficient_support"],
                            detail="no mesh has >= 50% usable pixels; SEP would return back=0, rms=1 with no error",
                            diagnostics=diag)
    sep = sep_module()
    # SEP's histogram does not check NaN: sanitize and mask (preflight H1).
    data = np.where(bad, 0.0, image).astype(np.float32)
    mask = bad.astype(np.uint8)
    try:
        bkg = sep.Background(np.ascontiguousarray(data), mask=np.ascontiguousarray(mask), maskthresh=0.0,
                             bw=bw, bh=bh, fw=config["fw"], fh=config["fh"], fthresh=config["fthresh"])
        back = np.ascontiguousarray(bkg.back(dtype=np.float32))
        rms = np.ascontiguousarray(bkg.rms(dtype=np.float32))
        diag["sep_globalback_raw"] = float(bkg.globalback)
        diag["sep_globalrms_raw"] = float(bkg.globalrms)
    except Exception as exc:  # noqa: BLE001 - SEP raises plain Exception subclasses
        est = _unavailable(method, config, ["backend_failure"], detail=f"{type(exc).__name__}: {exc}", diagnostics=diag)
        est.status = "failed"
        return est
    reasons = ["meshes_interpolated"] if diag["meshes_below_support"] else []
    r_usable = rms[usable]
    noise_reasons, validity = rms_map_validity(rms, usable, bw, bh)
    diag["rms_map_validity"] = validity
    noise_ok = not noise_reasons
    est = BackgroundEstimate(
        method=method,
        status="completed",
        level_status="measured",
        noise_status="measured" if noise_ok else "unresolved",
        reasons=reasons + noise_reasons,
        config=config,
        config_sha256=config_sha256(config),
        spatial=True,
        back=back,
        rms=rms if noise_ok else None,
        raw_rms=rms,
        level_summary=float(np.median(back[usable])),
        rms_summary=float(np.median(r_usable)) if noise_ok else None,
        level_definition="median of the SEP background map over usable pixels",
        rms_definition="median of the SEP RMS map over usable pixels",
        noise_source="empirical_mesh_rms" if noise_ok else "unavailable",
        diagnostics=diag,
    )
    if not noise_ok:
        est.diagnostics["sep_globalrms_note"] = "not scored: SEP substitutes 1.0 or a positive-mesh median"
    return est


def python_global_scalar(image: np.ndarray, masks: dict) -> BackgroundEstimate:
    config = global_scalar_config_json()
    method = "python_global_scalar"
    if image.ndim != 2 or image.size == 0:
        return _unavailable(method, config, ["malformed_geometry"])
    usable = ~unusable_mask(image, masks)
    values = image[usable].astype(np.float64)
    if values.size == 0:
        return _unavailable(method, config, ["no_usable_pixels"])
    keep = np.ones(values.size, bool)
    iters = 0
    for iters in range(1, GLOBAL_SCALAR_PARAMS["max_iters"] + 1):
        sample = values[keep]
        center, scale = float(np.median(sample)), float(np.std(sample))
        new_keep = np.abs(values - center) <= GLOBAL_SCALAR_PARAMS["clip_sigma"] * scale
        if scale == 0.0 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    sample = values[keep]
    center, scale = float(np.median(sample)), float(np.std(sample))
    noise_ok = scale > 0.0
    return BackgroundEstimate(
        method=method,
        status="completed",
        level_status="measured",
        noise_status="measured" if noise_ok else "unresolved",
        reasons=["no_spatial_model"] + ([] if noise_ok else ["zero_empirical_scatter"]),
        config=config,
        config_sha256=config_sha256(config),
        spatial=False,
        back=np.float32(center),
        rms=np.float32(scale) if noise_ok else None,
        level_summary=center,
        rms_summary=scale if noise_ok else None,
        level_definition="median of the iteratively 3-sigma-clipped usable pixels",
        rms_definition="population std of the same clipped sample",
        noise_source="empirical_global_clip" if noise_ok else "unavailable",
        diagnostics={"iterations": iters, "clipped_sample_pixels": int(keep.sum()), "usable_pixels": int(values.size)},
    )


def oracle_background(scenario) -> BackgroundEstimate:
    """Truth B and R. For digitized scenarios R is the effective digitized
    noise and, where the analog level is not recoverable from the digitized
    data, B is the expected digitized level (``scoring_background``)."""
    config = {"method": "oracle_true_background",
              "source": "simulated truth: scoring background, effective sky noise std"}
    if scenario.background is None or scenario.image.size == 0:
        return _unavailable("oracle_true_background", config, ["no_truth"])
    usable = ~unusable_mask(scenario.image, scenario.masks)
    if not usable.any():
        return _unavailable("oracle_true_background", config, ["no_usable_pixels"])
    r = scenario.noise_std_sky
    scoreable = scenario.metadata.get("noise_scoreable", True)
    noise_ok = r is not None and scoreable and bool(np.all(r[usable] > 0))
    back = scenario.scoring_background()
    reasons = [] if noise_ok else ["zero_empirical_scatter"] if not scoreable else ["no_truth_noise"]
    return BackgroundEstimate(
        method="oracle_true_background",
        status="completed",
        level_status="measured",
        noise_status="measured" if noise_ok else "unresolved",
        reasons=reasons,
        config=config,
        config_sha256=config_sha256(config),
        spatial=True,
        back=back,
        rms=r if noise_ok else None,
        detail=None if noise_ok or scoreable else "effective digitized noise is below 0.01 ADU step",
        level_summary=float(np.median(back[usable])),
        rms_summary=float(np.median(r[usable])) if noise_ok else None,
        level_definition="median of truth B over usable pixels",
        rms_definition="median of truth sky noise std over usable pixels",
        noise_source="truth",
    )


# --------------------------------------------------------------------------
# Detection, acceptance and HFR (identical for every Python background)
# --------------------------------------------------------------------------


def _catalog_rows(objects, sep) -> list:
    rows = []
    for o in objects:
        flag = int(o["flag"])
        rows.append({
            "x": float(o["x"]), "y": float(o["y"]),
            "xmin": int(o["xmin"]), "xmax": int(o["xmax"]), "ymin": int(o["ymin"]), "ymax": int(o["ymax"]),
            "npix": int(o["npix"]), "flux": float(o["flux"]), "peak": float(o["peak"]),
            "a": float(o["a"]), "b": float(o["b"]), "theta": float(o["theta"]), "flags": flag,
            "deblended": bool(flag & sep.OBJ_MERGED), "truncated": bool(flag & sep.OBJ_TRUNC),
        })
    return rows


def detect_and_measure(image: np.ndarray, masks: dict, estimate: BackgroundEstimate,
                       det_params: dict | None = None, policy: dict | None = None) -> dict:
    """SEP extraction on I - B_hat and harness acceptance/HFR.

    Never raises on bad data; raises MissingPrerequisite only when SEP must
    run and the pinned wheel is absent."""
    det = sep_detection_config_json(det_params)
    policy = policy or ACCEPTANCE_POLICY
    out = {
        "detection_config_sha256": config_sha256(det),
        "acceptance_policy_sha256": config_sha256(policy),
        "detection": {"status": "skipped", "reason": None},
        "candidates": None,
        "accepted": None,
        "rejections": None,
        "frame_hfr_median_px": None,
        "n_accepted": None,
    }
    if estimate.level_status != "measured":
        out["detection"] = {"status": "unsupported", "reason": "background level unavailable"}
        return out
    if estimate.noise_status != "measured":
        out["detection"] = {"status": "unsupported",
                            "reason": f"noise {estimate.noise_status} ({','.join(r for r in estimate.reasons if r in NOISE_REASONS)}): "
                                      "a relative detection threshold is undefined"}
        return out
    sep = sep_module()
    h, w = image.shape
    bad = unusable_mask(image, masks)
    sat = masks.get("native_saturation")
    sat = sat.astype(bool) if sat is not None else np.zeros_like(bad)
    back = estimate.back_map(image.shape)
    resid = np.where(bad, np.float32(0.0), image.astype(np.float32) - back.astype(np.float32))
    resid = np.ascontiguousarray(resid, dtype=np.float32)
    err = estimate.rms
    err = float(err) if np.isscalar(err) else np.ascontiguousarray(err, dtype=np.float32)
    try:
        sep.set_extract_pixstack(det["pixstack"])
        # The detector mask also goes to sep.extract, as in the daemon adapter:
        # under CONV with a noise array a masked pixel's threshold becomes
        # thresh*1e30, so a masked (e.g. clipped) core cannot join a footprint
        # through its neighbours' filtered signal.
        objects = sep.extract(
            resid, det["thresh_sigma"], err=err, mask=np.ascontiguousarray(bad.astype(np.uint8)), maskthresh=0.0,
            minarea=det["minarea"],
            filter_kernel=np.asarray(det["filter_kernel"], dtype=np.float32), filter_type=det["filter_type"],
            deblend_nthresh=det["deblend_nthresh"], deblend_cont=det["deblend_cont"],
            clean=det["clean"], clean_param=det["clean_param"],
        )
    except Exception as exc:  # noqa: BLE001
        out["detection"] = {"status": "failed", "reason": f"sep.extract: {type(exc).__name__}: {exc}"}
        return out
    catalog = _catalog_rows(objects, sep)
    out["candidates"] = catalog
    out["detection"] = {"status": "completed", "reason": None, "n_objects": len(catalog)}
    tallies = {k: 0 for k in ("edge_truncated", "deblended", "masked_support", "saturated", "low_snr",
                              "elongated", "blended", "invalid_hfr")}
    accepted = []
    if not catalog:
        out.update(accepted=[], rejections=tallies, n_accepted=0)
        return out
    xs = np.array([c["x"] for c in catalog])
    ys = np.array([c["y"] for c in catalog])
    cap = float(policy["hfr"]["cap_px"]) if "hfr" in policy else HFR_CAP_PX
    r1, f1 = sep.flux_radius(resid, xs, ys, np.full(xs.size, min(20.0, cap)), 0.5, subpix=5)
    r1 = np.asarray(r1, dtype=np.float64)
    rmax2 = np.clip(4.0 * np.nan_to_num(r1, nan=1.5), 6.0, cap)
    r2, f2 = sep.flux_radius(resid, xs, ys, rmax2, 0.5, subpix=5)
    r_snr = np.maximum(3.0, 2.0 * np.nan_to_num(r1, nan=1.5))
    flux, fluxerr, _ = sep.sum_circle(resid, xs, ys, r_snr, err=err, subpix=5)
    margin = policy.get("edge_margin_px", 2)
    for k, c in enumerate(catalog):
        reasons = []
        x, y, rr = c["x"], c["y"], float(rmax2[k])
        if c["truncated"] or c["xmin"] <= margin or c["ymin"] <= margin or c["xmax"] >= w - 1 - margin \
                or c["ymax"] >= h - 1 - margin or x - rr < -0.5 or y - rr < -0.5 or x + rr > w - 0.5 \
                or y + rr > h - 0.5:
            reasons.append("edge_truncated")
        if c["deblended"] and not det["accept_deblended_for_focus"]:
            reasons.append("deblended")
        x_lo, x_hi = max(0, int(math.floor(x - rr - 1))), min(w, int(math.ceil(x + rr + 1)) + 1)
        y_lo, y_hi = max(0, int(math.floor(y - rr - 1))), min(h, int(math.ceil(y + rr + 1)) + 1)
        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        inside = (xx - x) ** 2 + (yy - y) ** 2 <= (rr + 0.7072) ** 2
        if np.any(sat[y_lo:y_hi, x_lo:x_hi][inside]):
            reasons.append("saturated")
        if np.any(bad[y_lo:y_hi, x_lo:x_hi][inside] & ~sat[y_lo:y_hi, x_lo:x_hi][inside]):
            reasons.append("masked_support")
        snr = flux[k] / fluxerr[k] if fluxerr[k] > 0 else float("nan")
        if not (snr >= policy["min_aperture_snr"]):
            reasons.append("low_snr")
        if c["a"] > 0 and 1.0 - c["b"] / c["a"] > policy["max_ellipticity_1_minus_b_over_a"]:
            reasons.append("elongated")
        others = np.hypot(xs - x, ys - y)
        others[k] = np.inf
        if np.any(others < rr):
            reasons.append("blended")
        hfr = float(r2[k])
        if not (np.isfinite(hfr) and hfr > 0) or (int(f2[k]) & sep.APER_NONPOSITIVE):
            reasons.append("invalid_hfr")
        for reason in reasons:
            tallies[reason] += 1
        c["hfr_px"] = hfr if np.isfinite(hfr) else None
        c["aperture_snr"] = float(snr) if np.isfinite(snr) else None
        c["rejections"] = reasons
        if not reasons:
            accepted.append({"x": x, "y": y, "hfr_px": hfr})
    out["accepted"] = accepted
    out["rejections"] = tallies
    out["n_accepted"] = len(accepted)
    out["frame_hfr_median_px"] = float(np.median([a["hfr_px"] for a in accepted])) if accepted else None
    return out


def run_python_method(method: str, scenario, bg_params: dict | None = None, det_params: dict | None = None) -> dict:
    """Run one Python method on a scenario; returns estimate and measurement."""
    if method in PYTHON_SEP_VARIANTS:
        variant = PYTHON_SEP_VARIANTS[method]
        est = python_sep_variant_background(scenario.image, scenario.masks, variant["noise_policy"], bg_params,
                                            det_params, method=method)
        meas = detect_and_gate(scenario.image, scenario.masks, est, variant_star_gates(method), det_params,
                               saturation_adu=scenario.metadata.get("saturation_adu"))
        return {"estimate": est, "measurement": meas}
    if method == "python_sep":
        est = python_sep_background(scenario.image, scenario.masks, bg_params)
    elif method == "python_global_scalar":
        est = python_global_scalar(scenario.image, scenario.masks)
    elif method == "oracle_true_background":
        est = oracle_background(scenario)
    else:
        raise KeyError(method)
    meas = detect_and_measure(scenario.image, scenario.masks, est, det_params) if scenario.image.size else {
        "detection": {"status": "unsupported", "reason": "malformed geometry"}, "candidates": None,
        "accepted": None, "rejections": None, "frame_hfr_median_px": None, "n_accepted": None}
    return {"estimate": est, "measurement": meas}


# --------------------------------------------------------------------------
# BGE v2 Python matrix (python_sep__nXgY)
# --------------------------------------------------------------------------


def _labeller():
    from ml.labels import psf_measure

    return psf_measure


def variant_star_gates(method: str):
    """The :class:`psf_measure.StarGateParams` of a ``python_sep__nXgY`` /
    ``rust_*__*`` method id (``g1``: peak_count_v1 without regrouping;
    ``g2``: dip_annulus_v2 with regrouping)."""
    LM = _labeller()
    suffix = method.rsplit("__", 1)[-1]
    g = suffix[-2:]
    if g == "g1":
        return LM.StarGateParams(blend_test="peak_count_v1", regroup_deblended=False)
    if g == "g2":
        return LM.StarGateParams(blend_test="dip_annulus_v2", regroup_deblended=True)
    raise KeyError(method)


def variant_background_params(noise_policy: str, params: dict | None = None):
    LM = _labeller()
    p = dict(DEFAULT_BACKGROUND_PARAMS, **(params or {}))
    return LM.SepBackgroundParams(bw=int(p["bw"]), bh=int(p["bh"]), fw=int(p["fw"]), fh=int(p["fh"]),
                                  fthresh=float(p["fthresh"]), min_meshes_per_axis=int(p["min_meshes_per_axis"]),
                                  noise_policy=noise_policy)


def variant_detection_params(det_params: dict | None = None):
    LM = _labeller()
    p = dict(DEFAULT_DETECTION_PARAMS, **(det_params or {}))
    return LM.SepDetectionParams(thresh_sigma=float(p["thresh_sigma"]), minarea=int(p["minarea"]),
                                 filter_type=str(p["filter_type"]),
                                 filter_kernel=tuple(tuple(r) for r in p["filter_kernel"]),
                                 deblend_nthresh=int(p["deblend_nthresh"]), deblend_cont=float(p["deblend_cont"]),
                                 clean=bool(p["clean"]), clean_param=float(p["clean_param"]),
                                 pixstack=int(p["pixstack"]),
                                 accept_deblended_for_focus=bool(p["accept_deblended_for_focus"]))


def sep_background_config_for(noise_policy: str, params: dict | None = None) -> dict:
    """Canonical background config of a noise policy (frame_wide_v1 equals
    :func:`sep_background_config_json`, hash 80eab0fb...)."""
    return variant_background_params(noise_policy, params).config(SEP_VERSION_EXPECTED)


def variant_unusable_mask(image: np.ndarray, masks: dict) -> np.ndarray:
    """The daemon SEP adapter's detector mask: :func:`unusable_mask` plus
    |value| >= 1e30 (SEP's BIG sentinel)."""
    bad = unusable_mask(image, masks)
    with np.errstate(invalid="ignore"):
        bad |= np.abs(image) >= SEP_BIG
    return bad


def python_sep_variant_background(image: np.ndarray, masks: dict, noise_policy: str, params: dict | None = None,
                                  det_params: dict | None = None, method: str = "python_sep__n2g2") -> BackgroundEstimate:
    """The labeller's SEP background under ``noise_policy`` as a BackgroundEstimate."""
    LM = _labeller()
    bg_params = variant_background_params(noise_policy, params)
    config = bg_params.config(SEP_VERSION_EXPECTED)
    image = np.asarray(image)
    if image.ndim != 2 or image.size == 0:
        return _unavailable(method, config, ["malformed_geometry"])
    bad = variant_unusable_mask(image, masks)
    sb = LM.estimate_spatial_background(image, detector_mask=bad, params=bg_params,
                                        detection_params=variant_detection_params(det_params))
    usable = ~bad
    diag = {"mesh_nx": (sb.mesh_grid or (None, None))[0], "mesh_ny": (sb.mesh_grid or (None, None))[1],
            "meshes_below_support": None if sb.meshes_total is None else sb.meshes_total - sb.meshes_supported,
            "usable_pixels": int(usable.sum()), "total_pixels": int(image.size),
            "rms_map_validity": sb.rms_validity, "detection_area_fraction": sb.detection_area_fraction}
    if sb.global_background is not None:
        diag["sep_globalback_raw"] = sb.global_background
    if sb.global_rms is not None:
        diag["sep_globalrms_raw"] = sb.global_rms
    if not sb.level_measured:
        est = _unavailable(method, config, list(sb.reasons), detail=sb.backend_error, diagnostics=diag)
        if "backend_failure" in sb.reasons:
            est.status = "failed"
        est.spatial_background = sb
        return est
    noise_ok = sb.noise_measured
    valid_noise = usable if sb.invalid_mask is None else usable & ~sb.invalid_mask
    est = BackgroundEstimate(
        method=method,
        status="completed",
        level_status="measured",
        noise_status="measured" if noise_ok else "unresolved",
        reasons=list(sb.reasons),
        config=config,
        config_sha256=config_sha256(config),
        spatial=True,
        back=sb.background_map,
        rms=sb.rms_map if noise_ok else None,
        raw_rms=sb.raw_rms_map,
        level_summary=float(np.median(sb.background_map[usable])),
        rms_summary=float(np.median(sb.rms_map[valid_noise])) if noise_ok and valid_noise.any() else None,
        level_definition="median of the SEP background map over usable pixels",
        rms_definition="median of the SEP RMS map over usable pixels outside the locally invalid mask",
        noise_source="empirical_mesh_rms" if noise_ok else "unavailable",
        diagnostics=diag,
        spatial_background=sb,
        refinement=sb.refinement,
    )
    return est


def detect_and_gate(image: np.ndarray, masks: dict, estimate: BackgroundEstimate, gates,
                    det_params: dict | None = None, saturation_adu=None) -> dict:
    """SEP extraction as the daemon adapter runs it, then the labeller's
    Rust-parity focus gates (``psf_measure.focus_quality``)."""
    LM = _labeller()
    det = sep_detection_config_json(det_params)
    out = {
        "detection_config_sha256": config_sha256(det),
        "acceptance_policy_sha256": gates.config_sha256(),
        "star_gates_config": gates.config(),
        "star_gates_config_sha256": gates.config_sha256(),
        "detection": {"status": "skipped", "reason": None},
        "candidates": None,
        "accepted": None,
        "rejections": None,
        "frame_hfr_median_px": None,
        "n_accepted": None,
        "gate_diagnostics": None,
        "refinement": estimate.refinement,
        "quality_status": None,
    }
    if estimate.level_status != "measured":
        out["detection"] = {"status": "unsupported", "reason": "background level unavailable"}
        out["quality_status"] = "BackgroundUnavailable"
        return out
    if estimate.noise_status != "measured":
        out["detection"] = {"status": "unsupported",
                            "reason": f"noise {estimate.noise_status} ({','.join(r for r in estimate.reasons if r in NOISE_REASONS)}): "
                                      "a relative detection threshold is undefined"}
        out["quality_status"] = "DetectionUnavailable"
        return out
    sep = sep_module()
    sb = estimate.spatial_background
    data = np.asarray(image, dtype=np.float32)
    bad = variant_unusable_mask(image, masks)
    invalid = None if sb is None else sb.invalid_mask
    ext_mask = bad if invalid is None else (bad | invalid)
    back = estimate.back_map(image.shape).astype(np.float32)
    resid = np.ascontiguousarray(np.where(ext_mask, np.float32(0.0), data - back), dtype=np.float32)
    err = sb.extraction_rms() if sb is not None else np.ascontiguousarray(estimate.rms, dtype=np.float32)
    previous = sep.get_extract_pixstack()
    try:
        sep.set_extract_pixstack(det["pixstack"])
        objects, segmap = sep.extract(
            resid, det["thresh_sigma"], err=err, mask=np.ascontiguousarray(ext_mask, dtype=np.uint8),
            maskthresh=0.0, minarea=det["minarea"],
            filter_kernel=np.asarray(det["filter_kernel"], dtype=np.float32), filter_type=det["filter_type"],
            deblend_nthresh=det["deblend_nthresh"], deblend_cont=det["deblend_cont"],
            clean=det["clean"], clean_param=det["clean_param"], segmentation_map=True,
        )
    except Exception as exc:  # noqa: BLE001
        out["detection"] = {"status": "failed", "reason": f"sep.extract: {type(exc).__name__}: {exc}"}
        out["quality_status"] = "DetectionUnavailable"
        return out
    finally:
        sep.set_extract_pixstack(previous)
    rows = LM.sep_catalog_rows(objects, data, sep)
    candidates, families = LM.focus_candidates_from_catalog(rows, segmap, gates)
    quality = LM.focus_quality(data, candidates, gates=gates, field_kind="map", level=back,
                               rms=np.asarray(estimate.rms, dtype=np.float32), masks=masks, invalid_mask=invalid,
                               saturation_adu=saturation_adu)
    out.update(
        candidates=rows,
        accepted=quality["used_stars"],
        rejections=quality["star_rejections"],
        frame_hfr_median_px=quality["hfr_px"],
        n_accepted=quality["star_count_used"],
        gate_diagnostics=quality["gate_diagnostics"],
        quality_status=quality["status"],
        focus_candidates=quality["candidates"],
        star_count_detected=quality["star_count_detected"],
        families={"multi_member": sum(1 for f in families if len(f) >= 2),
                  "single_member": sum(1 for f in families if len(f) == 1)},
        detection={"status": "completed", "reason": None, "n_objects": len(rows),
                   "objects_deblended": sum(1 for r in rows if r["deblended"]),
                   "objects_truncated": sum(1 for r in rows if r["truncated"])},
    )
    return out
