"""One command for the background-estimation benchmark (BGE-09, BGE-14, BGE-22, BGE-23).

    python -m ml.eval.bge.run_benchmark --out <dir> [--rust] [--mesh-study] [--timing]
        [--split dev|holdout] [--methods default|matrix|<id> ...]

``--split dev`` (default) runs the development set: the pre-existing
scenarios and sweeps plus the BGE v2 development variants (``dev_*``).
``--split holdout`` additionally runs the held-out sets (``ho_*``) and refuses
to start unless ml/eval/bge/acceptance/synthetic-v2.json is tracked in git and
unchanged versus HEAD; that file's last commit is recorded in results.
``--methods`` selects method ids (``default`` = the five pre-v2 rows,
``matrix`` = every Python and Rust variant of docs/bge-v2-spec.md §4).

Writes ``results.json`` (every scenario x method with status completed |
unsupported | skipped | failed and a reason), ``summary.md``, effective
configurations, input hashes, bridge inputs/outputs and catalogs. See
README.md for metric definitions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import GENERATOR_VERSION, HARNESS_VERSION, RESULTS_SCHEMA, bridge, config_sha256
from . import metrics as MT
from . import python_methods as PM
from . import scenarios as SC

PY_METHODS = ("python_sep", "python_global_scalar", "oracle_true_background")
RUST_METHODS = {"rust_legacy_global_clip": "legacy_global_clip", "rust_sep": "sep"}
#: BGE v2 method matrix (docs/bge-v2-spec.md §4).
PY_VARIANTS = tuple(PM.PYTHON_SEP_VARIANTS)
RUST_VARIANT_IDS = tuple(bridge.RUST_VARIANTS)
#: The pre-v2 default rows (backward compatible).
DEFAULT_METHODS = PY_METHODS + tuple(RUST_METHODS)
MATRIX_METHODS = PY_METHODS + PY_VARIANTS + RUST_VARIANT_IDS
METHOD_GROUPS = {"default": DEFAULT_METHODS, "matrix": MATRIX_METHODS, "python": PY_METHODS + PY_VARIANTS,
                 "python_variants": PY_VARIANTS, "rust": tuple(RUST_METHODS) + RUST_VARIANT_IDS,
                 "rust_variants": RUST_VARIANT_IDS}
ACCEPTANCE_RELPATH = "ml/eval/bge/acceptance/synthetic-v2.json"
METHOD_INFO = {
    "python_sep": {"kind": "python_reference", "label": PM.METHOD_LABELS["python_sep"]},
    "python_global_scalar": {"kind": "python_comparator", "label": PM.METHOD_LABELS["python_global_scalar"]},
    "oracle_true_background": {"kind": "oracle", "label": PM.METHOD_LABELS["oracle_true_background"]},
    "rust_legacy_global_clip": {"kind": "rust_bridge",
                                "label": "Rust daemon legacy_global_clip via bge_eval_bridge (current default)"},
    "rust_sep": {"kind": "rust_bridge", "label": "Rust daemon SEP backend via bge_eval_bridge (opt-in)"},
}
METHOD_INFO["rust_legacy_global_clip"]["label"] += "; = rust_legacy__g1 (peak_count_v1 passed explicitly)"
METHOD_INFO["rust_legacy_global_clip"]["alias_of"] = "rust_legacy__g1"
METHOD_INFO["rust_sep"]["label"] += "; = rust_sep__n1g1 (frame_wide_v1, peak_count_v1, no regrouping passed explicitly)"
METHOD_INFO["rust_sep"]["alias_of"] = "rust_sep__n1g1"
for _m in PY_VARIANTS:
    METHOD_INFO[_m] = {"kind": "python_reference", "label": PM.METHOD_LABELS[_m]}
for _m, _spec in bridge.RUST_VARIANTS.items():
    METHOD_INFO[_m] = {"kind": "rust_bridge",
                       "label": f"Rust daemon {_spec['backend']} via bge_eval_bridge; settings {json.dumps(_spec['settings'], sort_keys=True)}, "
                                f"star_gates {json.dumps(_spec['star_gates'], sort_keys=True)}"}
SECTION_TRUTH = "Simulated truth (correctness under controlled assumptions; not real-camera accuracy)"
SECTION_PARITY = ("Cross-implementation agreement (Python SEP vs Rust SEP; implementation consistency, "
                  "not physical validation)")
SECTION_REAL = "Real-frame reference: not run (no predeclared thresholds; hold-out not consumed)"
#: D-006 (decisions.md), predeclared by the coordinator before any candidate result.
D006_MAP_REL = 1e-5
REPO_ROOT = Path(__file__).resolve().parents[3]


def _sysctl(name):
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def host_label() -> str:
    try:
        mem = int(_sysctl("hw.memsize") or "")
    except ValueError:
        mem = None
    cpu = _sysctl("machdep.cpu.brand_string") or platform.processor() or platform.system()
    mem_s = f", {round(mem / 2**30)} GB" if mem else ""
    return f"available-host software timing ({cpu}, {platform.machine()}, {os.cpu_count()} CPU{mem_s}), " \
           f"not a deployment target"


def rust_build_profile(repo: Path) -> dict:
    """The daemon's [profile.release] (the bridge is built with --release)."""
    try:
        import tomllib

        with open(Path(repo) / "achernar-daemon" / "Cargo.toml", "rb") as fh:
            return {"profile": "release", "settings": tomllib.load(fh).get("profile", {}).get("release", {})}
    except (OSError, ValueError) as exc:
        return {"profile": "release", "settings": None, "error": str(exc)}


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _clean(o):
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(json.loads(json.dumps(obj, default=_json_default))), indent=1,
                               sort_keys=True))


def git_info(repo: Path) -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_eval_files": git("status", "--porcelain", "--", "ml/eval/bge", "ml/tests/test_bge_fixtures.py")}


# --------------------------------------------------------------------------
# Scoring one scenario x method
# --------------------------------------------------------------------------

#: Stages of one scenario x method row, in pipeline order, and the metric
#: sections each one owns. A section is computed only when its stage
#: completed; ``scored_stages`` lists the sections actually present.
STAGES = ("background", "noise", "detection", "hfr")
STAGE_METRICS = {"background": ("background", "background_vs_analog"), "noise": ("noise", "noise_vs_analog"),
                 "detection": ("detection", "blends"), "hfr": ("hfr",)}
NOT_APPLICABLE = "not_applicable"


def overall_status(stages: dict):
    """Row status from its stages. ``not_applicable`` stages are ignored."""
    reasons = []
    for name, st in stages.items():
        if st["status"] == "failed":
            return "failed", f"{name}: {st.get('reason')}"
    for name, st in stages.items():
        if st["status"] in ("unsupported", "skipped"):
            reasons.append(f"{name}: {st.get('reason')}")
    if reasons:
        return "unsupported", "; ".join(reasons)
    return "completed", None


def hfr_stage(sc, detection_completed, n_accepted, frame_hfr, why_none) -> dict:
    """The width-measurement stage: completed only when an HFR was produced."""
    has_stars = any(s.kind == "star" for s in sc.sources)
    if not has_stars and not n_accepted:
        return {"status": NOT_APPLICABLE, "reason": "scenario has no truth stars"}
    if not detection_completed:
        return {"status": "unsupported", "reason": "detection not run"}
    if frame_hfr is not None and np.isfinite(frame_hfr) and (n_accepted or 0) > 0:
        return {"status": "completed", "reason": None}
    return {"status": "unsupported", "reason": f"no usable stars ({why_none})"}


def score(sc, stages, b_hat, r_hat, candidates, accepted, frame_hfr) -> dict:
    """Metrics for the completed stages only; marks each stage scored or not."""
    out = {}
    done = {k: v.get("status") == "completed" for k, v in stages.items()}
    level_measured = done.get("background", False)
    if sc.image.size == 0:
        out["honest_absence"] = not level_measured
        return out
    usable = ~PM.unusable_mask(sc.image, sc.masks)
    if not usable.any():
        out["honest_absence"] = not level_measured
    truth = [s.to_json() for s in sc.sources]
    fp_noise = sc.footprint_noise()
    if level_measured and b_hat is not None and sc.background is not None and usable.any():
        under, far = MT.source_masks(sc.source_image, fp_noise, usable)
        star_xy = [(s.x, s.y) for s in sc.sources if s.kind == "star"]
        truth_b = sc.scoring_background()
        bg = MT.background_map_metrics(b_hat, truth_b, usable, under, far, star_xy)
        if truth_b is sc.background:
            bg["truth"] = "analog B(x,y)"
        else:
            bg["truth"] = ("expected digitized level (quantization-limited: the analog level "
                           f"{sc.metadata['params'].get('background_adu')} ADU is not recoverable from the data)")
            out["background_vs_analog"] = {"bias_median_adu": MT.background_map_metrics(
                b_hat, sc.background, usable)["bias_median_adu"], "truth": "analog B (secondary)"}
        out["background"] = bg
    if done.get("noise") and r_hat is not None and sc.noise_std_sky is not None and usable.any():
        if not sc.metadata.get("noise_scoreable", True):
            stages["noise"]["score_note"] = (f"not scoreable: effective digitized noise "
                                             f"{sc.metadata.get('effective_sky_std_adu'):.2e} ADU < 0.01 step")
        else:
            nz = MT.noise_map_metrics(r_hat, sc.noise_std_sky, usable)
            nz["truth"] = sc.metadata.get("noise_truth", "expected noise std of the sky")
            out["noise"] = nz
            if sc.noise_std_sky_analog is not None:
                sec = MT.noise_map_metrics(r_hat, sc.noise_std_sky_analog, usable)
                out["noise_vs_analog"] = {"rel_error_median": sec.get("rel_error_median"),
                                          "truth": "analog sigma (secondary)"}
    if done.get("detection") and candidates is not None:
        det = MT.detection_metrics(candidates, truth, sc.width, sc.height, sc.source_image, fp_noise)
        if any(c > 1 for c in SC.group_sizes(sc.sources).values()):
            out["blends"] = MT.blend_metrics(candidates, truth, det["_matches"])
        out["detection"] = MT.strip_private(det)
    if done.get("hfr") and accepted is not None:
        hfr = MT.hfr_metrics(accepted, truth)
        hfr["frame_hfr_median_px"] = frame_hfr
        out["hfr"] = hfr
    for name, st in stages.items():
        st["scored"] = any(k in out for k in STAGE_METRICS.get(name, ()))
    return out


def finish_record(rec: dict, stages: dict) -> dict:
    """Status, reason and the scored/not-scored markers of a row."""
    status, reason = overall_status(stages)
    scored = [k for k in STAGES if stages.get(k, {}).get("scored")]
    unscored = [k for k in STAGES if k in stages and not stages[k].get("scored")
                and stages[k]["status"] != NOT_APPLICABLE]
    rec.update(status=status, reason=reason, stages=stages, scored_stages=scored, unscored_stages=unscored,
               partial=bool(status == "unsupported" and scored))
    return rec


def run_python(sc, method, bg_params=None) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    try:
        res = PM.run_python_method(method, sc, bg_params=bg_params)
    except PM.MissingPrerequisite:
        raise
    except Exception as exc:  # noqa: BLE001 - recorded as a failed case, never as a pass
        rec = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "stages": {}, "metrics": {},
               "scored_stages": [], "unscored_stages": list(STAGES), "partial": False}
        return rec, {}
    est, meas = res["estimate"], res["measurement"]
    level_ok = est.level_status == "measured"
    stages = {
        "background": {"status": "completed" if level_ok else ("failed" if est.status == "failed" else "unsupported"),
                       "reason": None if level_ok else ",".join(est.reasons) + (f" ({est.detail})" if est.detail else "")},
    }
    if level_ok:
        stages["noise"] = {"status": "completed" if est.noise_status == "measured" else "unsupported",
                           "reason": None if est.noise_status == "measured" else
                           f"noise {est.noise_status}: {','.join(est.reasons)}"}
    stages["detection"] = {"status": meas["detection"]["status"], "reason": meas["detection"].get("reason")}
    if level_ok:
        rej = {k: v for k, v in (meas.get("rejections") or {}).items() if v}
        why = "no candidates detected" if not meas.get("candidates") else f"rejections {rej}"
        stages["hfr"] = hfr_stage(sc, stages["detection"]["status"] == "completed", meas.get("n_accepted"),
                                  meas.get("frame_hfr_median_px"), why)
    shape = sc.image.shape
    b_hat = est.back_map(shape) if level_ok else None
    r_hat = est.rms_map(shape) if est.noise_status == "measured" else None
    rec = {
        "background_result": est.summary(),
        "detection_config_sha256": meas.get("detection_config_sha256"),
        "acceptance_policy_sha256": meas.get("acceptance_policy_sha256"),
        "rejections": meas.get("rejections"),
        "n_candidates": None if meas.get("candidates") is None else len(meas["candidates"]),
        "n_accepted": meas.get("n_accepted"),
        "frame_hfr_median_px": meas.get("frame_hfr_median_px"),
        "wall_ms": 1e3 * (time.perf_counter() - t0),
        "accepted_stars": meas.get("accepted"),
        "detection_ran": meas["detection"]["status"] == "completed",
    }
    for key in ("star_gates_config_sha256", "gate_diagnostics", "refinement", "quality_status",
                "star_count_detected", "families"):
        if key in meas:
            rec[key] = meas[key]
    rec["metrics"] = score(sc, stages, b_hat, r_hat, meas.get("candidates"), meas.get("accepted"),
                           meas.get("frame_hfr_median_px"))
    finish_record(rec, stages)
    raw_rms = est.raw_rms if isinstance(est.raw_rms, np.ndarray) else r_hat
    artifacts = {"candidates": meas.get("candidates"), "accepted": meas.get("accepted"), "back": b_hat,
                 "rms": raw_rms, "globalback": est.diagnostics.get("sep_globalback_raw"),
                 "globalrms": est.diagnostics.get("sep_globalrms_raw"),
                 "focus_candidates": meas.get("focus_candidates"),
                 "usable_pixels": est.diagnostics.get("usable_pixels")}
    return rec, artifacts


#: backend_error text of a daemon revision that has the bridge but not the SEP adapter.
SEP_NOT_BUILT_MARKER = "not built into this revision"
#: Daemon quality statuses meaning the detector never ran on the frame.
DETECTION_NOT_RUN = ("BackgroundUnavailable", "DetectionUnavailable")


def run_rust_parse(sc, out_dir, backend, revision=None) -> tuple[dict, dict]:
    out = bridge.read_output(out_dir, sc.id, backend, sc.width, sc.height)
    if out["status"] != "completed":
        return {"status": "failed", "reason": out["reason"], "stages": {}, "metrics": {}, "scored_stages": [],
                "unscored_stages": list(STAGES), "partial": False}, {}
    j = out["json"]
    bge = (j.get("background") or {}).get("backend_error") or ""
    if bridge.backend_of(backend) == "sep" and SEP_NOT_BUILT_MARKER in bge:
        rev = (revision or "unknown revision")[:12]
        return {"status": "skipped", "reason": f"skipped: rust sep backend not integrated at {rev} ({bge})",
                "stages": {}, "metrics": {}, "background_result": j.get("background"), "scored_stages": [],
                "unscored_stages": list(STAGES), "partial": False}, {}
    bgr = j.get("background") or {}
    level_ok = bgr.get("level_status") == "measured"
    noise_ok = bgr.get("noise_status") == "measured"
    spatial = bgr.get("method") == "sep_spatial_mesh"
    reasons = ",".join(bgr.get("reasons") or [])
    if spatial:
        b_hat, r_hat = out["back"], out["rms"] if noise_ok else None
    else:
        b_hat = np.float64(bgr["level"]) if level_ok and bgr.get("level") is not None else None
        r_hat = np.float64(bgr["rms"]) if noise_ok and bgr.get("rms") is not None else None
    quality = j.get("quality") or {}
    q_status = quality.get("status")
    # The bridge emits empty lists when detection never ran; those are not
    # zero-completeness measurements.
    detection_ran = j.get("detection") is not None and q_status not in DETECTION_NOT_RUN
    # Detection metrics score the detector's output: the SEP catalog when the
    # bridge emits one (gate candidates hold regrouped families under BGE v2),
    # else the legacy detector's candidates.
    catalog = j.get("catalog")
    candidates = (catalog if catalog is not None else j.get("candidates")) if detection_ran else None
    accepted = quality.get("used_stars") if detection_ran else None
    stages = {"background": {"status": "completed" if level_ok else "unsupported",
                             "reason": None if level_ok else reasons}}
    if level_ok:
        stages["noise"] = {"status": "completed" if noise_ok else "unsupported",
                           "reason": None if noise_ok else f"noise {bgr.get('noise_status')}: {reasons}"}
    stages["detection"] = {"status": "completed" if detection_ran else "unsupported",
                           "reason": None if detection_ran else
                           f"detection not run (quality status {q_status}; catalog_error {j.get('catalog_error')})"}
    if (j.get("detection") or {}).get("noise_floor_applied"):
        stages["detection"]["note"] = "ran with the legacy 1 ADU noise floor (noise_floor_applied), a labelled heuristic"
    if level_ok:
        rej = {k: v for k, v in (quality.get("star_rejections") or {}).items() if v}
        why = f"quality status {q_status}; " + ("no candidates detected" if not candidates else f"rejections {rej}")
        stages["hfr"] = hfr_stage(sc, detection_ran, None if accepted is None else len(accepted), quality.get("hfr_px"),
                                  why)
    if level_ok and spatial and b_hat is None:
        stages["background"] = {"status": "unsupported", "reason": "spatial field but no .back.f32 emitted"}
    bg_summary = {k: bgr.get(k) for k in ("method", "method_version", "requested_method", "config_sha256",
                                          "level_status", "noise_status", "reasons", "backend_error", "units",
                                          "noise_source", "level", "level_definition", "rms", "rms_definition",
                                          "mesh", "plane", "support", "library_version")}
    rec = {
        "background_result": bg_summary,
        "background_config": bgr.get("config"),
        "detection": j.get("detection"),
        "quality": {k: quality.get(k) for k in ("status", "rejection_reasons", "hfr_px", "hfd_px",
                                                "star_count_detected", "star_count_used", "star_rejections")},
        "trailing": j.get("trailing"),
        "timing_ms": j.get("timing_ms"),
        "n_candidates": None if candidates is None else len(candidates),
        "n_accepted": None if accepted is None else len(accepted),
        "frame_hfr_median_px": quality.get("hfr_px"),
        "accepted_stars": accepted,
        "detection_ran": bool(detection_ran),
        "refinement": bridge.output_refinement(j),
        "gate_diagnostics": bridge.output_gate_diagnostics(j),
        "star_gates_config_sha256": bridge.output_star_gates_sha256(j),
        "quality_status": q_status,
        "star_count_detected": quality.get("star_count_detected"),
    }
    rec["metrics"] = score(sc, stages, b_hat, r_hat, candidates, accepted, quality.get("hfr_px"))
    finish_record(rec, stages)
    support = bgr.get("support") or {}
    return rec, {"candidates": candidates, "accepted": accepted, "catalog": j.get("catalog"), "back": out["back"],
                 "rms": out["rms"], "globalback": bgr.get("level") if level_ok else None,
                 "globalrms": bgr.get("rms") if noise_ok else None,
                 "focus_candidates": j.get("candidates"), "usable_pixels": support.get("usable_pixels")}


def skipped_record(reason) -> dict:
    return {"status": "skipped", "reason": reason, "stages": {}, "metrics": {}, "scored_stages": [],
            "unscored_stages": list(STAGES), "partial": False}


# --------------------------------------------------------------------------
# Cross-implementation agreement
# --------------------------------------------------------------------------

PARITY_NOT_COMPARED = ("segmentation maps (the bridge does not export them); globalrms when the daemon reports "
                       "noise unresolved (it publishes no rms then)")


def _within(a: float, b: float) -> bool:
    return abs(a - b) <= D006_MAP_REL * max(1.0, abs(a))


def parity(py_art, rs_art, py_rec, rs_rec) -> dict:
    """D-006 comparison of every item both implementations produced: B and
    raw R maps (whatever the noise status), globalback/globalrms, catalogs."""
    out = {"python_status": py_rec["status"], "rust_status": rs_rec["status"]}
    for key in ("back", "rms"):
        a, b = py_art.get(key), rs_art.get(key)
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape:
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            bound = D006_MAP_REL * np.maximum(1.0, np.abs(a.astype(np.float64)))
            out[f"{key}_max_abs_diff"] = float(np.max(d))
            out[f"{key}_within_d006_bound"] = bool(np.all(d <= bound))
        else:
            out[f"{key}_max_abs_diff"] = None
    for key in ("globalback", "globalrms"):
        a, b = py_art.get(key), rs_art.get(key)
        if a is not None and b is not None:
            out[f"{key}_abs_diff"] = abs(float(a) - float(b))
            out[f"{key}_within_d006_bound"] = _within(float(a), float(b))
    pl = (py_rec.get("background_result") or {})
    rl = (rs_rec.get("background_result") or {})
    out["python_level_status"], out["rust_level_status"] = pl.get("level_status"), rl.get("level_status")
    out["python_noise_status"], out["rust_noise_status"] = pl.get("noise_status"), rl.get("noise_status")
    pc = py_art.get("candidates")
    rc = rs_art.get("catalog") if rs_art.get("catalog") is not None else rs_art.get("candidates")
    if pc is not None and rc is not None:
        out.update(catalog_parity(pc, rc))
    out["not_compared"] = PARITY_NOT_COMPARED
    return out


def parity_tally(rows) -> dict:
    def tally(key):
        vals = [p[key] for p in rows if p.get(key) is not None]
        return {"within": sum(bool(v) for v in vals), "compared": len(vals)}

    return {"cases": len(rows),
            "statuses_identical": sum(1 for p in rows if (p["python_level_status"], p["python_noise_status"]) ==
                                      (p["rust_level_status"], p["rust_noise_status"])),
            **{k: tally(f"{k}_within_d006_bound") for k in ("back", "rms", "globalback", "globalrms")},
            "catalog": tally("catalog_within_d006")}


def catalog_parity(pc, rc) -> dict:
    """SEP catalogs compared object by object against the D-006 bounds."""
    out = {"n_candidates_python": len(pc), "n_candidates_rust": len(rc)}
    m = MT.match_greedy([(c["x"], c["y"]) for c in pc], [(c["x"], c["y"]) for c in rc], [0.5] * len(rc))
    out["n_matched_within_0p5px"] = len(m)
    out["max_centroid_diff_px"] = max((d for _, _, d in m), default=None)
    int_keys = ("npix", "xmin", "xmax", "ymin", "ymax", "flags")
    mism = {k: 0 for k in int_keys}
    worst = {"x": 0.0, "y": 0.0, "flux_rel": 0.0, "a_rel": 0.0, "b_rel": 0.0, "theta": 0.0}
    for i, j, _ in m:
        a, b = pc[i], rc[j]
        for k in int_keys:
            if k in a and k in b and int(a[k]) != int(b[k]):
                mism[k] += 1
        for k in ("x", "y"):
            worst[k] = max(worst[k], abs(a[k] - b[k]))
        for k in ("flux", "a", "b"):
            if k in a and k in b:
                worst[f"{k}_rel"] = max(worst[f"{k}_rel"], abs(a[k] - b[k]) / max(abs(a[k]), 1e-5))
        if "theta" in a and "theta" in b:
            worst["theta"] = max(worst["theta"], abs(a["theta"] - b["theta"]))
    out["integer_field_mismatches"] = mism
    out["max_abs_diff"] = worst
    out["catalog_within_d006"] = bool(
        len(pc) == len(rc) == len(m) and not any(mism.values()) and worst["x"] <= 1e-4 and worst["y"] <= 1e-4
        and worst["flux_rel"] <= 1e-5 and worst["a_rel"] <= 1e-5 and worst["b_rel"] <= 1e-5 and worst["theta"] <= 1e-5)
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# BGE v2: method resolution, held-out guard, parity, set metrics
# --------------------------------------------------------------------------


def resolve_methods(tokens) -> list:
    """Method ids from ``--methods`` tokens (group names expand), in order, unique."""
    out = []
    for token in tokens or ["default"]:
        names = METHOD_GROUPS.get(token, (token,))
        for name in names:
            if name not in METHOD_INFO:
                raise SystemExit(f"unknown method {name!r} (known: {', '.join(METHOD_INFO)}; "
                                 f"groups: {', '.join(METHOD_GROUPS)})")
            if name not in out:
                out.append(name)
    return out


def is_rust_method(method: str) -> bool:
    return method in RUST_METHODS or method in bridge.RUST_VARIANTS


def _git(repo: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)


def holdout_guard(repo: Path, relpath: str = ACCEPTANCE_RELPATH) -> dict:
    """The held-out split may run only on predeclared, committed criteria:
    ``relpath`` tracked (``git ls-files --error-unmatch``) and unchanged
    versus HEAD (``git diff --quiet HEAD --``). Returns ``ok`` plus the
    file's last commit and sha256 for the results."""
    path = Path(repo) / relpath
    out = {"file": relpath, "ok": False, "reason": None, "commit": None, "sha256": None}
    try:
        tracked = _git(repo, "ls-files", "--error-unmatch", "--", relpath)
        if tracked.returncode != 0:
            out["reason"] = f"{relpath} is not tracked in git: {tracked.stderr.strip()}"
            return out
        diff = _git(repo, "diff", "--quiet", "HEAD", "--", relpath)
        if diff.returncode != 0:
            out["reason"] = f"{relpath} differs from HEAD (git diff --quiet HEAD exit {diff.returncode})"
            return out
        commit = _git(repo, "log", "-1", "--format=%H", "--", relpath).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        out["reason"] = f"git unavailable: {exc}"
        return out
    if not commit:
        out["reason"] = f"no commit found for {relpath}"
        return out
    import hashlib

    out.update(ok=True, commit=commit, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return out


#: Spec §5: invalid_rms_pixels may differ by at most 0.1% of usable pixels.
PARITY_INVALID_RMS_FRACTION = 1e-3
#: Accepted stars are the same star when their positions agree within this.
PARITY_STAR_POSITION_PX = 0.01
REFINEMENT_EXACT_KEYS = ("noise_policy", "pass2_used", "pass2_skipped_reason", "source_mask_objects",
                         "source_mask_pixels")


def accepted_set_parity(py_used, rs_used) -> dict:
    py_used, rs_used = py_used or [], rs_used or []
    m = MT.match_greedy([(a["x"], a["y"]) for a in py_used], [(a["x"], a["y"]) for a in rs_used],
                        [PARITY_STAR_POSITION_PX] * len(rs_used))
    hfr_rel = [abs(py_used[i]["hfr_px"] - rs_used[j]["hfr_px"]) / max(abs(rs_used[j]["hfr_px"]), 1e-12)
               for i, j, _ in m]
    return {"n_python": len(py_used), "n_rust": len(rs_used), "n_matched": len(m),
            "max_position_diff_px": max((d for _, _, d in m), default=None),
            "max_hfr_rel_diff": max(hfr_rel, default=None),
            "agree": len(py_used) == len(rs_used) == len(m)}


def parity_v2(py_rec, rs_rec, py_art, rs_art) -> dict:
    """Spec §5 parity of one case for one SEP variant: statuses and reasons,
    refinement counts (invalid_rms_pixels within 0.1% of usable), catalog
    sizes (exact, and platform-aware: non-deblended objects exact plus the
    deblended totals), gate tallies, gate diagnostics and accepted-star sets."""
    pb = py_rec.get("background_result") or {}
    rb = rs_rec.get("background_result") or {}
    out = {"python_status": py_rec["status"], "rust_status": rs_rec["status"]}
    out["statuses_agree"] = (pb.get("level_status"), pb.get("noise_status")) == \
        (rb.get("level_status"), rb.get("noise_status"))
    out["reasons_agree"] = sorted(pb.get("reasons") or []) == sorted(rb.get("reasons") or [])
    pr, rr = py_rec.get("refinement"), rs_rec.get("refinement")
    if pr is None and rr is None:
        out["refinement_agree"] = True
    elif pr is None or rr is None:
        out["refinement_agree"] = False
        out["refinement_note"] = "only one side reports a refinement record"
    else:
        usable = py_art.get("usable_pixels") or rs_art.get("usable_pixels")
        exact = {k: pr.get(k) == rr.get(k) for k in REFINEMENT_EXACT_KEYS}
        d_inv = abs((pr.get("invalid_rms_pixels") or 0) - (rr.get("invalid_rms_pixels") or 0))
        within = usable is not None and d_inv <= PARITY_INVALID_RMS_FRACTION * usable
        out["refinement_mismatches"] = sorted(k for k, ok in exact.items() if not ok)
        out["invalid_rms_pixels_diff"] = d_inv
        out["refinement_agree"] = all(exact.values()) and bool(within)
    pc = py_art.get("candidates")
    rc = rs_art.get("catalog")
    if pc is not None and rc is not None:
        out["n_objects_python"], out["n_objects_rust"] = len(pc), len(rc)
        out["catalog_size_agree"] = len(pc) == len(rc)

        def split(rows):
            return (sum(1 for r in rows if not r.get("deblended")), sum(1 for r in rows if r.get("deblended")))

        out["catalog_platform_aware_agree"] = split(pc)[0] == split(rc)[0]
    else:
        out["catalog_size_agree"] = pc is None and rc is None
        out["catalog_platform_aware_agree"] = out["catalog_size_agree"]
    pt = py_rec.get("rejections") or {}
    rt = (rs_rec.get("quality") or {}).get("star_rejections") or {}
    keys = sorted(set(pt) | set(rt))
    out["tally_mismatches"] = {k: [pt.get(k, 0), rt.get(k, 0)] for k in keys if pt.get(k, 0) != rt.get(k, 0)}
    out["tallies_agree"] = not out["tally_mismatches"]
    pg, rg = py_rec.get("gate_diagnostics") or {}, rs_rec.get("gate_diagnostics") or {}
    out["gate_diagnostics_agree"] = all(pg.get(k, 0) == rg.get(k, 0) for k in ("annulus_exempt", "regrouped_accepted"))
    out["accepted"] = accepted_set_parity(py_rec.get("accepted_stars"), rs_rec.get("accepted_stars"))
    out["star_gates_sha_agree"] = (py_rec.get("star_gates_config_sha256") == rs_rec.get("star_gates_config_sha256")
                                   if rs_rec.get("star_gates_config_sha256") else None)
    out["all_agree"] = bool(out["statuses_agree"] and out["reasons_agree"] and out["refinement_agree"]
                            and out["catalog_platform_aware_agree"] and out["tallies_agree"]
                            and out["gate_diagnostics_agree"] and out["accepted"]["agree"])
    return out


def parity_v2_tally(rows: dict) -> dict:
    keys = ("statuses_agree", "reasons_agree", "refinement_agree", "catalog_size_agree",
            "catalog_platform_aware_agree", "tallies_agree", "gate_diagnostics_agree", "all_agree")
    tally = {k: sum(1 for r in rows.values() if r.get(k)) for k in keys}
    tally["accepted_sets_agree"] = sum(1 for r in rows.values() if r["accepted"]["agree"])
    tally["cases"] = len(rows)
    tally["disagreeing_cases"] = sorted(cid for cid, r in rows.items() if not r.get("all_agree"))
    return tally


def frame_view(doc_or_scenarios, rows_by_key, frame_ids, method) -> list:
    """Frames of one method for the acceptance metrics (see metrics.py)."""
    frames = []
    for fid in frame_ids:
        rec = rows_by_key.get((fid, method))
        truth = doc_or_scenarios[fid]["sources"]
        if rec is None or rec["status"] in ("skipped", "failed"):
            frames.append({"truth": truth, "accepted": None, "false_detections": None, "detection_ran": False,
                           "status": None if rec is None else rec["status"]})
            continue
        det = (rec.get("metrics") or {}).get("detection") or {}
        frames.append({"truth": truth, "accepted": rec.get("accepted_stars"),
                       "false_detections": det.get("false_detections") if rec.get("detection_ran") else None,
                       "detection_ran": bool(rec.get("detection_ran")), "status": rec["status"]})
    return frames


def _isolated_donut(s):
    return bool(s.get("isolated")) and (s.get("r_out_px") or 0.0) >= 6.0


def set_metrics(scenarios_doc: dict, rows_by_key: dict, frame_ids: list, method: str) -> dict:
    """Every v2 acceptance metric that applies to one scenario set and method."""
    frames = frame_view(scenarios_doc, rows_by_key, frame_ids, method)
    truth = [s for fr in frames for s in fr["truth"]]
    not_run = sum(1 for fr in frames if fr["status"] in (None, "skipped"))
    out = {"n_frames": len(frames), "frames_completed": sum(1 for fr in frames if fr["status"] == "completed"),
           "frames_detection_ran": sum(1 for fr in frames if fr["detection_ran"]),
           "frames_not_run": not_run, "frames_failed": sum(1 for fr in frames if fr["status"] == "failed"),
           "frames_at_used_star_cap": sum(1 for fr in frames if len(fr["accepted"] or []) >= SC.GATE_MAX_STARS_USED)}
    # Skipped frames are not measurements: the set is not evaluable for this
    # method. (A failed or unsupported frame accepted nothing and counts.)
    out["status"] = "completed" if not_run == 0 else ("not_run" if not_run == len(frames) else "incomplete")
    if out["status"] == "not_run":
        return out
    if any("r_out_px" in s for s in truth):
        out["donut_acceptance_isolated_r_out_ge_6"] = MT.acceptance_rate(frames, _isolated_donut)
        out["donut_acceptance_all"] = MT.acceptance_rate(frames, lambda s: "r_out_px" in s)
        if any((s.get("n_vanes") or 0) > 0 for s in truth):
            out["donut_acceptance_vane_width_le_0p015"] = MT.acceptance_rate(
                frames, lambda s: s.get("vane_width_frac") is not None and s["vane_width_frac"] <= 0.015
                and (s.get("n_vanes") or 0) > 0)
        out["accepted_donut_hfr_bias"] = MT.accepted_rel_hfr_bias(frames, lambda s: "r_out_px" in s)
    if any(s.get("pair_id") is not None for s in truth):
        out["close_double_rejection_sep_ge_5_ratio_ge_0p5"] = MT.close_double_rejection(frames, 5.0, 0.5)
        out["close_double_rejection_all"] = MT.close_double_rejection(frames, 0.0, 0.0)
        out["single_acceptance"] = MT.acceptance_rate(frames, lambda s: s.get("role") == "single")
    starless = [fr for fid, fr in zip(frame_ids, frames) if scenarios_doc[fid]["metadata"].get("starless")]
    if starless:
        out["starless_false_detections"] = MT.false_detections_per_frame(starless)
    out["false_detections"] = MT.false_detections_per_frame(frames)
    return out


def build_cases(args):
    """Scenarios of the requested split: ``(core, sweeps, sets)``.

    ``--scenarios`` restricts the core scenarios (and may name v2 set or sweep
    ids); without it the dev split is every pre-existing scenario and sweep
    plus the BGE v2 development variants, and the holdout split adds the
    held-out sets (``--holdout-only`` drops the development set).
    """
    core, sweeps, sets = [], [], []
    v2_sets = dict(SC.DEV_V2_SETS)
    v2_sweeps = dict(SC.DEV_V2_SWEEPS)
    if args.split == "holdout":
        v2_sets.update(SC.HOLDOUT_SETS)
        v2_sweeps.update(SC.HOLDOUT_SWEEPS)
    if args.scenarios is not None:
        for name in args.scenarios:
            if name in SC.SCENARIOS:
                core.append(SC.build_scenario(name))
            elif name in v2_sets:
                sets.append(v2_sets[name]())
            elif name in v2_sweeps:
                sweeps.append(v2_sweeps[name]())
            elif name in SC.SWEEPS:
                sweeps.append(SC.SWEEPS[name]())
            elif name in SC.HOLDOUT_SETS or name in SC.HOLDOUT_SWEEPS:
                raise SystemExit(f"{name} is a held-out scenario: run with --split holdout")
            else:
                raise SystemExit(f"unknown scenario {name!r}")
        return core, sweeps, sets
    dev = not (args.split == "holdout" and args.holdout_only)
    if dev:
        core = [SC.build_scenario(n) for n in SC.SCENARIOS]
        sweeps = [f() for f in SC.SWEEPS.values()] + [f() for f in SC.DEV_V2_SWEEPS.values()]
        sets = [f() for f in SC.DEV_V2_SETS.values()]
    if args.split == "holdout":
        sets += [f() for f in SC.HOLDOUT_SETS.values()]
        sweeps += [f() for f in SC.HOLDOUT_SWEEPS.values()]
    return core, sweeps, sets


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rust", action="store_true", help="run the daemon's bge_eval_bridge (skipped if unavailable)")
    ap.add_argument("--mesh-study", action="store_true", help="SEP bw in {16,32,64,128} x footprint study")
    ap.add_argument("--timing", action="store_true", help="6000x4000 host timing/memory")
    ap.add_argument("--repo", type=Path, default=REPO_ROOT, help="repository root holding achernar-daemon/")
    ap.add_argument("--cargo-target-dir", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timing-repeats", type=int, default=5)
    ap.add_argument("--save-maps", action="store_true", help="save Python SEP B/R maps as float32")
    ap.add_argument("--scenarios", nargs="*", default=None, help="subset of core scenarios (or v2 set/sweep ids)")
    ap.add_argument("--split", choices=("dev", "holdout"), default="dev",
                    help="dev (default) or holdout (requires the committed acceptance criteria)")
    ap.add_argument("--holdout-only", action="store_true", help="with --split holdout: skip the development set")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="method ids or groups (default, matrix, python, python_variants, rust, rust_variants); "
                         "default: 'default' for dev, 'matrix' for holdout")
    args = ap.parse_args(argv)
    methods = resolve_methods(args.methods or (["matrix"] if args.split == "holdout" else ["default"]))
    py_methods = [m for m in methods if not is_rust_method(m)]
    rust_methods = [m for m in methods if is_rust_method(m)]
    acceptance = {"split": args.split, "status": "not_required"}
    if args.split == "holdout":
        acceptance = dict(holdout_guard(REPO_ROOT), split="holdout")
        if not acceptance["ok"]:
            print(f"refusing the held-out split: {acceptance['reason']}", file=sys.stderr)
            return 2
        acceptance["status"] = "verified"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    cargo_target = (args.cargo_target_dir or Path(os.environ.get("ACHERNAR_BGE_CARGO_TARGET_DIR", out / "cargo-target"))).resolve()
    t_start = time.perf_counter()
    log = []

    def note(msg):
        log.append(msg)
        print(msg, flush=True)

    # ---- fixtures
    core, sweeps, sets = build_cases(args)
    cases = core + [fr for sw in sweeps for fr in sw.frames] + [fr for st in sets for fr in st.frames]
    group_of = {sc.id: "core" for sc in core}
    split_of = {sc.id: SC.SPLIT_DEV for sc in core}
    pre_existing = {sc.id: sc.id in SC.PRE_EXISTING_SCENARIOS for sc in core}
    for sw in sweeps:
        for fr in sw.frames:
            group_of[fr.id] = f"focus_sweep:{sw.id}"
            split_of[fr.id] = sw.metadata.get("split", SC.SPLIT_DEV)
            pre_existing[fr.id] = sw.id in SC.PRE_EXISTING_SWEEPS
    for st in sets:
        for fr in st.frames:
            group_of[fr.id] = f"set:{st.id}"
            split_of[fr.id] = st.split
            pre_existing[fr.id] = False
    note(f"split {args.split}: {len(core)} core scenarios, {len(sweeps)} sweeps, {len(sets)} v2 sets "
         f"({len(cases)} frames); methods {', '.join(methods)}")

    bridge_dir = out / "bridge_inputs"
    bridge_cases = [sc for sc in cases if sc.image.size]
    excluded = [sc.id for sc in cases if not sc.image.size]
    case_entries = bridge.write_inputs(bridge_cases, bridge_dir)
    manifest_path = bridge.write_manifest(case_entries, bridge_dir, write_maps=True,
                                          extras={"excluded_cases": excluded})
    input_hashes = {sc.id: sc.input_hashes() for sc in cases}
    write_json(out / "input_hashes.json", {"generator_version": GENERATOR_VERSION, "cases": input_hashes,
                                           "bridge_files": json.loads(manifest_path.read_text())["extras"]["input_sha256"]})

    # ---- configs
    from ml.labels import psf_measure as LM

    configs = {
        "python_sep_background": PM.sep_background_config_json(),
        "python_sep_detection": PM.sep_detection_config_json(),
        "python_global_scalar": PM.global_scalar_config_json(),
        "acceptance_policy": PM.ACCEPTANCE_POLICY,
        "sep_background_frame_wide_v1": PM.sep_background_config_for("frame_wide_v1"),
        "sep_background_masked_local_v2": PM.sep_background_config_for("masked_local_v2"),
        "star_gates_peak_count_v1": LM.StarGateParams(blend_test="peak_count_v1", regroup_deblended=False).config(),
        "star_gates_dip_annulus_v2": LM.StarGateParams().config(),
    }
    config_hashes = {k: config_sha256(v) for k, v in configs.items()}
    contract = {
        "background_hash_matches_daemon_default": config_hashes["python_sep_background"] == PM.EXPECTED_BACKGROUND_CONFIG_SHA256,
        "detection_hash_matches_daemon_default": config_hashes["python_sep_detection"] == PM.EXPECTED_DETECTION_CONFIG_SHA256,
        "expected_background_sha256": PM.EXPECTED_BACKGROUND_CONFIG_SHA256,
        "expected_detection_sha256": PM.EXPECTED_DETECTION_CONFIG_SHA256,
        "frame_wide_v1_hash_unchanged": config_hashes["sep_background_frame_wide_v1"] == PM.EXPECTED_BACKGROUND_CONFIG_SHA256,
    }
    for k, v in configs.items():
        write_json(out / "configs" / f"{k}.json", {"config": v, "sha256": config_hashes[k]})

    # ---- Python methods
    results = []
    artifacts = {}

    def tag(rec, sc, m):
        rec.update(scenario=sc.id, group=group_of[sc.id], split=split_of[sc.id], pre_existing=pre_existing[sc.id],
                   method=m, method_kind=METHOD_INFO[m]["kind"])
        return rec

    for sc in cases:
        for m in py_methods:
            rec, art = run_python(sc, m)
            results.append(tag(rec, sc, m))
            artifacts[(sc.id, m)] = art
    note(f"python methods done ({time.perf_counter() - t_start:.1f} s)")

    # ---- Rust bridge: one run; every canonical method id is a backend
    rust_runs = {}
    canonical_ids = []
    for m in rust_methods:
        c, _ = bridge.rust_variant(m)
        if c not in canonical_ids:
            canonical_ids.append(c)
    if canonical_ids:
        rust_dir = out / "bridge_outputs"
        rust_manifest = bridge.write_methods_manifest(case_entries, bridge_dir, canonical_ids,
                                                      name="manifest_rust.json", extras={"excluded_cases": excluded})
        if args.rust:
            info = bridge.run_bridge(rust_manifest, rust_dir, args.repo, cargo_target, out / "logs" / "bridge.log",
                                     jobs=args.jobs)
            note(f"rust bridge ({', '.join(canonical_ids)}): {info['status']} {info.get('reason') or ''}")
        else:
            info = {"status": "skipped", "reason": "skipped: --rust not requested"}
        info.update(out_dir=str(rust_dir), manifest=str(rust_manifest), methods=canonical_ids)
        rust_runs = {c: info for c in canonical_ids}
    rust_info = next(iter(rust_runs.values()), {"status": "skipped", "reason": "skipped: no Rust method selected"})
    for sc in cases:
        for m in rust_methods:
            canonical, spec = bridge.rust_variant(m)
            info = rust_runs[canonical]
            if not sc.image.size:
                rec, art = skipped_record("skipped: zero-size frame excluded from bridge v1 manifest"), {}
            elif info["status"] == "completed":
                rec, art = run_rust_parse(sc, Path(info["out_dir"]), canonical,
                                          (info.get("revision") or {}).get("commit"))
            elif info["status"] == "failed":
                rec, art = {"status": "failed", "reason": info["reason"], "stages": {}, "metrics": {}}, {}
            else:
                rec, art = skipped_record(info["reason"]), {}
            rec["variant"] = canonical
            results.append(tag(rec, sc, m))
            artifacts[(sc.id, m)] = art

    # ---- catalogs and maps
    for (case_id, m), art in artifacts.items():
        if art.get("candidates") is not None or art.get("accepted") is not None:
            write_json(out / "catalogs" / f"{case_id}__{m}.json",
                       {"candidates": art.get("candidates"), "accepted": art.get("accepted"),
                        "focus_candidates": art.get("focus_candidates")})
        if args.save_maps and m in ("python_sep",) + PY_VARIANTS:
            for key in ("back", "rms"):
                if isinstance(art.get(key), np.ndarray):
                    (out / "maps").mkdir(exist_ok=True)
                    np.ascontiguousarray(art[key], dtype="<f4").tofile(out / "maps" / f"{case_id}__{m}.{key}.f32")

    # ---- focus sweeps
    by_key = {(r["scenario"], r["method"]): r for r in results}
    sweep_results = []
    for sw in sweeps:
        truth_curve = [float(np.median([s.hfr_true_px for s in fr.sources])) for fr in sw.frames]
        step = float(sw.metadata.get("step", SC.SWEEP_STEP))
        entry = {"sweep": sw.id, "description": sw.description, "positions": sw.positions, "p0_true": sw.p0_true,
                 "split": sw.metadata.get("split", SC.SPLIT_DEV), "pre_existing": sw.id in SC.PRE_EXISTING_SWEEPS,
                 "step": step, "truth_median_hfr_px": truth_curve,
                 "truth_fit": MT.focus_shift(sw.positions, truth_curve, sw.p0_true), "params": sw.metadata,
                 "methods": {}}
        ref_idx = int(np.argmin(np.abs(np.asarray(sw.positions) - sw.p0_true)))
        for m in methods:
            entry["methods"][m] = summarize_sweep_method([by_key[(fr.id, m)] for fr in sw.frames], sw.positions,
                                                         sw.p0_true, ref_idx, truth_curve, step=step)
        sweep_results.append(entry)

    # ---- v2 set metrics (scenario sets and the pre-v2 donut/double scenarios)
    scenarios_doc = {sc.id: {"description": sc.description, "group": group_of[sc.id], "split": split_of[sc.id],
                             "pre_existing": pre_existing[sc.id], "metadata": sc.metadata, "width": sc.width,
                             "height": sc.height, "truth": sc.truth_summary(), "input_sha256": input_hashes[sc.id],
                             "sources": [s.to_json() for s in sc.sources]} for sc in cases}
    set_frames = {st.id: [fr.id for fr in st.frames] for st in sets}
    set_frames.update({sw.id: [fr.id for fr in sw.frames] for sw in sweeps if not sw.id in SC.PRE_EXISTING_SWEEPS})
    v2_metrics = {sid: {"split": scenarios_doc[fids[0]]["split"],
                        "methods": {m: set_metrics(scenarios_doc, by_key, fids, m) for m in methods}}
                  for sid, fids in set_frames.items() if fids}
    headline = headline_metrics(scenarios_doc, by_key, core, sweeps, methods)

    # ---- mesh study
    mesh = {"status": "skipped", "reason": "skipped: --mesh-study not requested"}
    if args.mesh_study:
        mesh = run_mesh_study(args, out, cargo_target, note, methods)

    # ---- timing
    timing = {"status": "skipped", "reason": "skipped: --timing not requested"}
    if args.timing:
        timing = run_timing(args, out, cargo_target, rust_runs, note, methods)

    # ---- cross-implementation
    parity_rows = {}
    parity_reason = rust_info["reason"]
    mesh_parity = mesh.get("parity") or {}
    if "python_sep" in methods and "rust_sep" in methods and rust_runs.get("rust_sep__n1g1", {}).get("status") == "completed":
        for sc in cases:
            rs = by_key[(sc.id, "rust_sep")]
            if sc.image.size and rs["status"] != "skipped":
                parity_rows[sc.id] = parity(artifacts[(sc.id, "python_sep")], artifacts[(sc.id, "rust_sep")],
                                            by_key[(sc.id, "python_sep")], rs)
            elif sc.image.size:
                parity_reason = rs["reason"]
    v2_parity = {}
    for py_m in PY_VARIANTS:
        rs_m = "rust_sep__" + py_m.split("__")[1]
        if py_m not in methods or rs_m not in methods:
            continue
        run = rust_runs.get(rs_m, {})
        if run.get("status") != "completed":
            v2_parity[py_m.split("__")[1]] = {"status": "skipped", "reason": run.get("reason")}
            continue
        rows = {}
        for sc in cases:
            rs = by_key[(sc.id, rs_m)]
            if sc.image.size and rs["status"] != "skipped":
                rows[sc.id] = parity_v2(by_key[(sc.id, py_m)], rs, artifacts[(sc.id, py_m)], artifacts[(sc.id, rs_m)])
        v2_parity[py_m.split("__")[1]] = {"status": "completed" if rows else "skipped", "python": py_m, "rust": rs_m,
                                          "cases": rows, "tally": parity_v2_tally(rows) if rows else None}
    cross = {"status": "completed" if parity_rows else "skipped",
             "reason": None if parity_rows else parity_reason,
             "interpretation": SECTION_PARITY,
             "bound": ("D-006 (decisions.md): maps and globalback/globalrms |delta| <= 1e-5*max(1,|ref|); catalogs: "
                       "identical count, npix, bbox, flags; x, y <= 1e-4 px; flux, a, b relative <= 1e-5; "
                       "theta <= 1e-5 rad. Reported as diagnostics."),
             "not_compared": PARITY_NOT_COMPARED,
             "cases": parity_rows,
             "tally": parity_tally(list(parity_rows.values())) if parity_rows else None,
             "mesh_study_cases": mesh_parity,
             "mesh_study_tally": parity_tally(list(mesh_parity.values())) if mesh_parity else None,
             "v2": v2_parity,
             "v2_rule": (f"spec §5: statuses, reasons (as sets), refinement (exact {', '.join(REFINEMENT_EXACT_KEYS)}; "
                         f"invalid_rms_pixels within {PARITY_INVALID_RMS_FRACTION:.1%} of usable), catalog sizes "
                         "(exact and platform-aware: non-deblended objects), gate tallies, gate_diagnostics, "
                         f"accepted-star sets (positions within {PARITY_STAR_POSITION_PX} px)")}

    real = {"status": "not_run", "reason": SECTION_REAL,
            "adapter": "ml/eval/bge/real_frames.py (refuses accuracy/pass-fail without a predeclared thresholds file)"}

    doc = {
        "schema": RESULTS_SCHEMA,
        "harness_version": HARNESS_VERSION,
        "generator_version": GENERATOR_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.executable).name, "-m", "ml.eval.bge.run_benchmark", *(argv or sys.argv[1:])]),
        "git": git_info(REPO_ROOT),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "scipy": __import__("scipy").__version__, "sep": PM.sep_module().__version__,
                        "platform": platform.platform()},
        "host": host_label(),
        "split": args.split,
        "acceptance_criteria": acceptance,
        "interpretation": {"simulated_truth": SECTION_TRUTH, "cross_implementation": SECTION_PARITY,
                           "real_frames": SECTION_REAL},
        "methods": {m: METHOD_INFO[m] for m in methods},
        "methods_run": methods,
        "configs": {k: {"config": v, "sha256": config_hashes[k]} for k, v in configs.items()},
        "contract_check": contract,
        "rust_bridge": rust_info,
        "rust_bridge_runs": rust_runs,
        "scenarios": scenarios_doc,
        "scenario_sets": {st.id: {"description": st.description, "split": st.split, "frames": [fr.id for fr in st.frames],
                                  "metadata": st.metadata} for st in sets},
        "results": results,
        "focus_sweeps": sweep_results,
        "v2_set_metrics": v2_metrics,
        "headline": headline,
        "mesh_study": mesh,
        "timing": timing,
        "cross_implementation": cross,
        "real_frames": real,
        "wall_s": time.perf_counter() - t_start,
        "log": log,
    }
    write_json(out / "results.json", doc)
    (out / "summary.md").write_text(render_summary(doc, core, sweeps))
    counts = {}
    for r in results:
        counts.setdefault(r["method"], {}).setdefault(r["status"], 0)
        counts[r["method"]][r["status"]] += 1
    note(f"status counts: {json.dumps(counts, sort_keys=True)}")
    note(f"wrote {out / 'results.json'} and {out / 'summary.md'} in {time.perf_counter() - t_start:.1f} s")
    return 0


def headline_metrics(scenarios_doc, by_key, core, sweeps, methods) -> dict:
    """Development headline numbers: donut acceptance on the pre-v2 donut
    scenario and sweep, close-double handling in crowded_field, noise status
    of sweep_moffat_fixed."""
    out = {}
    core_ids = {sc.id for sc in core}
    sweep_frames = {sw.id: [fr.id for fr in sw.frames] for sw in sweeps}
    for m in methods:
        h = {}
        if "defocused_donut_stars" in core_ids:
            h["defocused_donut_stars_acceptance"] = MT.acceptance_rate(
                frame_view(scenarios_doc, by_key, ["defocused_donut_stars"], m), lambda s: True)
        if "sweep_donut" in sweep_frames:
            h["sweep_donut_acceptance"] = MT.acceptance_rate(
                frame_view(scenarios_doc, by_key, sweep_frames["sweep_donut"], m),
                lambda s: (s.get("psf") or {}).get("r_out_px", 0.0) >= 6.0)
        if "crowded_field" in core_ids:
            rec = by_key.get(("crowded_field", m)) or {}
            hfr = (rec.get("metrics") or {}).get("hfr") or {}
            h["crowded_field"] = {"n_accepted": rec.get("n_accepted"), "blend_leakage": hfr.get("blend_leakage"),
                                  "n_clean": hfr.get("n_clean"), "clean_rel_bias_median": hfr.get("clean_rel_bias_median"),
                                  "rejections": rec.get("rejections") or (rec.get("quality") or {}).get("star_rejections")}
        if "sweep_moffat_fixed" in sweep_frames:
            recs = [by_key.get((fid, m)) or {} for fid in sweep_frames["sweep_moffat_fixed"]]
            h["sweep_moffat_fixed_noise_measured_frames"] = sum(
                1 for r in recs if (r.get("background_result") or {}).get("noise_status") == "measured")
        out[m] = h
    return out


def summarize_sweep_method(recs, positions, p0_true, ref_idx, truth_curve, step: float = 100.0) -> dict:
    """One method's focus sweep: fit, frame counts and status (completed only
    when the fit succeeded and every frame produced an HFR)."""
    statuses = [r["status"] for r in recs]
    if all(s == "skipped" for s in statuses):
        return {"status": "skipped", "reason": recs[0]["reason"]}
    vals = [r.get("frame_hfr_median_px") for r in recs]
    curve = [float(v) if v is not None else float("nan") for v in vals]
    fit = MT.focus_shift(positions, curve, p0_true)
    sets = [(r.get("metrics", {}).get("hfr") or {}).get("matched_truth_ids") or [] for r in recs]
    jac = [MT.jaccard(s, sets[ref_idx]) for s in sets]
    hyper_ok = fit["hyperbola"]["status"] == "completed"
    hfr_status = [(r.get("stages", {}).get("hfr") or {}).get("status", r["status"]) for r in recs]
    counts = {"frames": len(recs), "frames_with_hfr": sum(1 for h in hfr_status if h == "completed"),
              "frames_failed": sum(1 for r in recs if r["status"] == "failed"),
              "frames_without_hfr": sum(1 for h in hfr_status if h != "completed")}
    # A sweep is completed only if every frame produced an HFR and the
    # fit succeeded; a successful fit over missing frames is partial.
    if not hyper_ok:
        sweep_status = "failed"
        sweep_reason = f"sweep fit failed: {fit['hyperbola'].get('reason')}"
    elif counts["frames_without_hfr"] or counts["frames_failed"]:
        sweep_status = "partial"
        sweep_reason = (f"fit over {counts['frames_with_hfr']}/{counts['frames']} frames; "
                        f"{counts['frames_without_hfr']} without HFR, {counts['frames_failed']} failed")
    else:
        sweep_status, sweep_reason = "completed", None
    shift = fit["hyperbola"].get("p0_shift") if hyper_ok else None
    noise_measured = sum(1 for r in recs if (r.get("background_result") or {}).get("noise_status") == "measured")
    return {
        "status": sweep_status,
        "reason": sweep_reason,
        "step": step,
        "frames_noise_measured": noise_measured,
        "valid_frames_left_right": fit["hyperbola"].get("valid_frames_left_right"),
        "hyperbola_p0_shift_fraction_of_step": None if shift is None else shift / step,
        "frame_counts": counts,
        "frame_statuses": statuses,
        "frame_hfr_stage_statuses": hfr_status,
        "median_hfr_px": curve,
        "n_accepted": [r.get("n_accepted") for r in recs],
        "fit": fit,
        "accepted_set_jaccard_vs_frame_nearest_p0": jac,
        "frame_hfr_rel_error_vs_truth_median": [
            (c / t - 1.0) if np.isfinite(c) else None for c, t in zip(curve, truth_curve)],
    }


def run_mesh_study(args, out, cargo_target, note, methods=DEFAULT_METHODS) -> dict:
    """SEP mesh size x source footprint (BGE-09). Python: ``python_sep`` and
    every selected ``python_sep__*`` variant per mesh size; Rust: every
    selected SEP method (one bridge run per mesh size, method-id backends)."""
    fields = SC.mesh_study_fields()
    rows = []
    mesh_dir = out / "mesh_study"
    entries = bridge.write_inputs(fields, mesh_dir)
    py_sep = ["python_sep"] + [m for m in methods if m in PY_VARIANTS]
    rust_sep = [m for m in methods if is_rust_method(m) and bridge.backend_of(bridge.rust_variant(m)[0]) == "sep"]
    rust_runs = {}
    for bw in SC.MESH_STUDY_SIZES:
        if not rust_sep:
            rust_runs[bw] = {"status": "skipped", "reason": "skipped: no Rust SEP method selected"}
            continue
        mpath = bridge.write_methods_manifest(entries, mesh_dir, rust_sep, name=f"manifest_bw{bw}.json",
                                              settings={"sep": {"bw": bw, "bh": bw}}, write_maps=True)
        if args.rust:
            (out / "logs").mkdir(exist_ok=True)
            rust_runs[bw] = bridge.run_bridge(mpath, mesh_dir / f"out_bw{bw}", args.repo, cargo_target,
                                              out / "logs" / f"bridge_mesh_bw{bw}.log", jobs=args.jobs)
        else:
            rust_runs[bw] = {"status": "skipped", "reason": "skipped: --rust not requested"}
    mesh_parity, mesh_parity_v2 = {}, {}
    for sc in fields:
        for m in ("python_global_scalar", "oracle_true_background"):
            rec, _ = run_python(sc, m)
            rows.append(dict(_mesh_row(rec), field=sc.id, footprint=sc.metadata["footprint"], method=m, bw=None))
        for bw in SC.MESH_STUDY_SIZES:
            py = {}
            for m in py_sep:
                rec, art = run_python(sc, m, bg_params={"bw": bw, "bh": bw})
                py[m] = (rec, art)
                rows.append(dict(_mesh_row(rec), field=sc.id, footprint=sc.metadata["footprint"], method=m, bw=bw))
            rr = rust_runs[bw]
            for m in rust_sep:
                canonical, _ = bridge.rust_variant(m)
                if rr["status"] == "completed":
                    rrec, rart = run_rust_parse(sc, mesh_dir / f"out_bw{bw}", canonical,
                                                (rr.get("revision") or {}).get("commit"))
                    if rrec["status"] != "skipped":
                        if m == "rust_sep":
                            mesh_parity[f"{sc.id}@bw{bw}"] = parity(py["python_sep"][1], rart, py["python_sep"][0], rrec)
                        py_m = "python_sep__" + canonical.split("__")[1]
                        if py_m in py:
                            mesh_parity_v2.setdefault(canonical.split("__")[1], {})[f"{sc.id}@bw{bw}"] = \
                                parity_v2(py[py_m][0], rrec, py[py_m][1], rart)
                else:
                    rrec = {"status": rr["status"], "reason": rr["reason"], "metrics": {}}
                rows.append(dict(_mesh_row(rrec), field=sc.id, footprint=sc.metadata["footprint"], method=m, bw=bw))
    note("mesh study done")
    return {"status": "completed", "sizes": list(SC.MESH_STUDY_SIZES), "footprints": SC.MESH_STUDY_FOOTPRINTS,
            "default_bw": PM.DEFAULT_BACKGROUND_PARAMS["bw"],
            "fields": {sc.id: {"truth_hfr_median_px": float(np.median([s.hfr_true_px for s in sc.sources])),
                               "input_sha256": sc.input_hashes(), "metadata": sc.metadata} for sc in fields},
            "rust_runs": rust_runs, "rows": rows, "parity": mesh_parity,
            "parity_v2": {k: {"cases": v, "tally": parity_v2_tally(v)} for k, v in mesh_parity_v2.items()}}


def _mesh_row(rec) -> dict:
    m = rec.get("metrics", {})
    bg, hfr, det = m.get("background") or {}, m.get("hfr") or {}, m.get("detection") or {}
    return {
        "status": rec["status"], "reason": rec.get("reason"), "partial": rec.get("partial", False),
        "scored_stages": rec.get("scored_stages", []), "unscored_stages": rec.get("unscored_stages", []),
        "bias_median_adu": bg.get("bias_median_adu"), "rmse_adu": bg.get("rmse_adu"),
        "bias_median_under_sources_adu": bg.get("bias_median_under_sources_adu"),
        "bias_median_far_from_sources_adu": bg.get("bias_median_far_from_sources_adu"),
        "bias_median_at_star_centres_adu": bg.get("bias_median_at_star_centres_adu"),
        "max_abs_error_at_star_centres_adu": bg.get("max_abs_error_at_star_centres_adu"),
        "large_scale_rms_adu": bg.get("large_scale_rms_adu"),
        "hfr_rel_bias_median": hfr.get("clean_rel_bias_median"), "n_accepted": hfr.get("n_accepted"),
        "completeness_stars": det.get("completeness_stars"),
        "false_detections": det.get("false_detections"),
        "fragments_on_sources": det.get("fragments_on_sources"),
    }


def run_timing(args, out, cargo_target, rust_runs, note, methods=DEFAULT_METHODS) -> dict:
    """6000x4000 host timing. Python: the timing worker's two methods. Rust:
    every selected Rust method (canonical ids; the aliases also fill the
    pre-v2 keys ``legacy_global_clip`` and ``sep``)."""
    tdir = out / "timing"
    frame = SC.timing_frame()
    entries = bridge.write_inputs([frame], tdir)
    rel = entries[0]["pixels"]
    result = {"status": "completed", "label": host_label(), "python_repeat_semantics": PY_REPEAT_SEMANTICS,
              "frame": {"id": frame.id, "width": frame.width,
              "height": frame.height, "input_sha256": entries[0]["extras"]["sha256"], "metadata": frame.metadata},
              "python": {}, "rust": {}}
    for m in ("python_sep", "python_global_scalar"):
        cmd = [sys.executable, "-m", "ml.eval.bge.timing_worker", "--pixels", str(tdir / rel), "--width",
               str(frame.width), "--height", str(frame.height), "--method", m, "--repeats", str(args.timing_repeats)]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600)
        if proc.returncode == 0:
            result["python"][m] = dict(json.loads(proc.stdout.strip().splitlines()[-1]), status="completed",
                                       command=" ".join(cmd))
        else:
            result["python"][m] = {"status": "failed", "reason": proc.stderr[-2000:], "command": " ".join(cmd)}
        note(f"timing {m}: {result['python'][m].get('total_ms_median')} ms")
    completed = [info for info in (rust_runs or {}).values() if info.get("status") == "completed"]
    if not completed:
        first = next(iter((rust_runs or {}).values()), {"reason": "skipped: no Rust method selected"})
        result["rust"] = {"status": "skipped", "reason": first.get("reason")}
        return result
    binary = bridge.bridge_test_binary(args.repo, cargo_target, args.jobs)
    if binary is None:
        result["rust"] = {"status": "skipped", "reason": f"{bridge.UNAVAILABLE}: test binary not found"}
        return result
    result["rust_build"] = rust_build_profile(args.repo)
    legacy_keys = {"rust_legacy__g1": "legacy_global_clip", "rust_sep__n1g1": "sep"}
    canonical_ids = []
    for m in methods:
        if is_rust_method(m):
            c, _ = bridge.rust_variant(m)
            if c not in canonical_ids:
                canonical_ids.append(c)
    revision = ((completed[0].get("revision") or {}).get("commit") or "unknown")[:12]
    for canonical in canonical_ids:
        mpath = bridge.write_variant_manifest(entries, tdir, canonical, name=f"manifest_{canonical}.json",
                                              write_maps=False)
        runs = rust_timing_runs(binary, mpath, tdir, canonical, frame, args.timing_repeats, out / "logs")
        if any(SEP_NOT_BUILT_MARKER in (r.get("backend_error") or "") for r in runs):
            entry = {"status": "skipped", "runs": runs, "reason": f"skipped: rust sep backend not integrated at {revision}"}
        else:
            entry = dict(summarize_rust_timing(runs), binary=binary)
        result["rust"][canonical] = entry
        if canonical in legacy_keys and any(bridge.RUST_ALIASES.get(m) == canonical for m in methods):
            result["rust"][legacy_keys[canonical]] = entry
        note(f"timing rust {canonical}: {entry.get('bridge_total_ms_median')} ms")
    return result


PY_REPEAT_SEMANTICS = ("one fresh Python process per method; one discarded in-process warm-up call, then the "
                       "repeats in the same process (warm); peak RSS from getrusage covers the whole process")
RUST_REPEAT_SEMANTICS = ("each repeat is a fresh test-binary process (cold process): /usr/bin/time wall time and peak "
                         "RSS include process start, manifest and frame read and result write; bridge timing_ms "
                         "covers only the in-process background, quality and trailing stages of that one call; one "
                         "discarded warm-up process runs first (file cache)")


def rust_timing_runs(binary, manifest, tdir, backend, frame, repeats, log_dir) -> list:
    """One discarded warm-up process, then ``repeats`` timed processes. Each
    output directory is cleared first so a stale result is never read."""
    import shutil

    runs = []
    for rep in ["warmup", *range(repeats)]:
        odir = Path(tdir) / f"out_{backend}_{rep}"
        if odir.exists():
            shutil.rmtree(odir)
        run = bridge.run_binary_timed(binary, manifest, odir, Path(log_dir) / f"timing_{backend}_{rep}.log")
        o = bridge.read_output(odir, frame.id, backend, frame.width, frame.height)
        payload = o.get("json") or {}
        run.update(repeat=rep, warmup=rep == "warmup", output_status=o["status"],
                   timing_ms=payload.get("timing_ms"),
                   backend_error=(payload.get("background") or {}).get("backend_error"))
        run["ok"] = run["returncode"] == 0 and o["status"] == "completed" and bool(run["timing_ms"])
        runs.append(run)
    return runs


def summarize_rust_timing(runs) -> dict:
    measured = [r for r in runs if not r["warmup"]]
    ok = [r for r in measured if r["ok"]]
    tm = [sum(r["timing_ms"].values()) for r in ok]
    bg_ms = [r["timing_ms"]["background"] for r in ok]
    rss = [r["peak_rss_bytes"] for r in ok if r.get("peak_rss_bytes")]
    failed = len(measured) - len(ok)
    return {
        "status": "completed" if measured and not failed else "failed",
        "reason": None if measured and not failed else
        f"{failed}/{len(measured)} timed repeats failed (non-zero return code or no output); see runs",
        "runs": runs,
        "repeats_ok": len(ok),
        "repeat_semantics": RUST_REPEAT_SEMANTICS,
        "bridge_total_ms_median": float(np.median(tm)) if tm else None,
        "bridge_total_ms_p95": float(np.percentile(tm, 95)) if tm else None,
        "bridge_background_ms_median": float(np.median(bg_ms)) if bg_ms else None,
        "process_wall_s_median": float(np.median([r["wall_s"] for r in ok])) if ok else None,
        "process_peak_rss_bytes_max": max(rss) if rss else None,
    }


# --------------------------------------------------------------------------
# summary.md
# --------------------------------------------------------------------------


def _status(status, reason) -> str:
    """'completed', or 'status: reason' without repeating the status word."""
    if status == "completed":
        return status
    reason = reason or ""
    return reason if reason.startswith(f"{status}:") else f"{status}: {reason}"


def _row_status(r) -> str:
    """Status cell. A row with some scored stages but an unsupported stage is
    labelled ``partial: <reason>`` and names the stages its numbers cover."""
    if r["status"] == "unsupported" and r.get("partial"):
        text = (f"partial: {r.get('reason')} [scored: {', '.join(r.get('scored_stages') or [])}; "
                f"not scored: {', '.join(r.get('unscored_stages') or [])}]")
    else:
        text = _status(r["status"], r.get("reason"))
    for name, st in (r.get("stages") or {}).items():
        if st.get("score_note"):
            text += f" [{name} {st['score_note']}]"
    return text


def _fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        if not np.isfinite(v):
            return "-"
        if abs(v) >= 1e4 or (0 < abs(v) < 10 ** (-nd)):
            return f"{v:.{max(nd - 1, 1)}e}"
        return f"{v:.{nd}f}"
    return str(v)


def _table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return "\n".join(lines)


def _rate(m):
    if not m or m.get("rate") is None:
        return "-"
    num = m.get("n_accepted", m.get("n_rejected"))
    den = m.get("n_population", m.get("n_pairs"))
    return f"{m['rate']:.3f} ({num}/{den})"


def render_v2_sections(doc) -> list:
    """BGE v2 tables: development headline, scenario-set metrics, parity tallies."""
    methods = doc.get("methods_run") or []
    L = ["### BGE v2 headline (development scenarios)", "",
         "Donut acceptance = truth stars matched by an accepted star (sweep_donut: frames with r_out >= 6 px); "
         "crowded_field: accepted stars, accepted stars matched to a multi-member blend group (leakage), clean "
         "relative HFR bias; sweep_moffat_fixed: frames with measured noise (of 13).", ""]
    rows = []
    for m in methods:
        h = (doc.get("headline") or {}).get(m) or {}
        cf = h.get("crowded_field") or {}
        rows.append([f"`{m}`", _rate(h.get("defocused_donut_stars_acceptance")), _rate(h.get("sweep_donut_acceptance")),
                     f"{_fmt(cf.get('n_accepted'))} / {_fmt(cf.get('blend_leakage'))} / "
                     f"{_fmt(cf.get('clean_rel_bias_median'), 3)}",
                     _fmt(h.get("sweep_moffat_fixed_noise_measured_frames"))])
    L.append(_table(["method", "defocused_donut_stars", "sweep_donut", "crowded_field acc / leak / HFR rel",
                     "sweep_moffat_fixed noise measured"], rows))
    L += ["", "### BGE v2 scenario sets", "",
          "Per set and method: donut acceptance of isolated truth donuts with r_out >= 6 px (all donuts), of spider "
          "donuts with vane width <= 1.5% D; median relative HFR bias of accepted donuts; close-double rejection "
          "(separation >= 5 px and flux ratio >= 0.5; all pairs); false detections per starless frame. Development "
          "sets are `dev_*`; held-out sets `ho_*` appear only in a `--split holdout` run.", ""]
    for sid, entry in (doc.get("v2_set_metrics") or {}).items():
        L += [f"#### `{sid}` ({entry['split']})", ""]
        rows = []
        for m, mt in entry["methods"].items():
            bias = mt.get("accepted_donut_hfr_bias") or {}
            starless = mt.get("starless_false_detections") or {}
            rows.append([f"`{m}`", f"{mt['frames_completed']}/{mt['n_frames']}",
                         f"{_rate(mt.get('donut_acceptance_isolated_r_out_ge_6'))} [{_rate(mt.get('donut_acceptance_all'))}]",
                         _rate(mt.get("donut_acceptance_vane_width_le_0p015")), _fmt(bias.get("median_rel_bias"), 3),
                         f"{_rate(mt.get('close_double_rejection_sep_ge_5_ratio_ge_0p5'))} [{_rate(mt.get('close_double_rejection_all'))}]",
                         _fmt(starless.get("per_frame"), 2)])
        L.append(_table(["method", "frames completed", "donut acc (all)", "spider acc", "donut HFR rel bias",
                         "double rejection (all)", "false / starless frame"], rows))
        L.append("")
    v2 = (doc.get("cross_implementation") or {}).get("v2") or {}
    L += ["### BGE v2 Rust-Python parity", "", (doc.get("cross_implementation") or {}).get("v2_rule", ""), ""]
    if not v2:
        L.append("not run (needs matching python_sep__* and rust_sep__* methods and a completed bridge)")
    rows = []
    for variant, entry in v2.items():
        t = entry.get("tally")
        if not t:
            rows.append([variant, _status(entry.get("status", "skipped"), entry.get("reason")), "-", "-", "-", "-", "-", "-"])
            continue
        n = t["cases"]
        rows.append([variant, f"{t['all_agree']}/{n}", f"{t['statuses_agree']}/{n}", f"{t['reasons_agree']}/{n}",
                     f"{t['refinement_agree']}/{n}", f"{t['catalog_platform_aware_agree']}/{n}",
                     f"{t['tallies_agree']}/{n}", f"{t['accepted_sets_agree']}/{n}"])
    if rows:
        L.append(_table(["variant", "all agree", "statuses", "reasons", "refinement", "catalog (platform-aware)",
                         "tallies", "accepted sets"], rows))
    L.append("")
    return L


def render_summary(doc, core, sweeps) -> str:
    methods = doc.get("methods_run") or (list(PY_METHODS) + list(RUST_METHODS))
    by = {(r["scenario"], r["method"]): r for r in doc["results"]}
    L = [f"# BGE evaluation summary", "",
         f"- Command: `{doc['command']}`",
         f"- Revision: `{doc['git']['commit']}` ({doc['git']['branch']}); generator `{doc['generator_version']}`, "
         f"harness `{doc['harness_version']}`; created {doc['created_utc']}",
         f"- Environment: Python {doc['environment']['python']}, numpy {doc['environment']['numpy']}, "
         f"scipy {doc['environment']['scipy']}, sep {doc['environment']['sep']}",
         f"- Rust bridge: **{doc['rust_bridge']['status']}** {doc['rust_bridge'].get('reason') or ''} "
         f"(daemon revision `{(doc['rust_bridge'].get('revision') or {}).get('commit')}`, "
         f"binary `{doc['rust_bridge'].get('binary')}` sha256 `{(doc['rust_bridge'].get('binary_sha256') or '')[:16]}`)",
         f"- Config hashes: SEP background `{doc['configs']['python_sep_background']['sha256'][:12]}...` "
         f"(matches daemon default: {_fmt(doc['contract_check']['background_hash_matches_daemon_default'])}), "
         f"SEP detection `{doc['configs']['python_sep_detection']['sha256'][:12]}...` "
         f"(matches: {_fmt(doc['contract_check']['detection_hash_matches_daemon_default'])})",
         "- Metric definitions and units: `ml/eval/bge/README.md`. Every scenario x method row has a status: "
         "`completed` = every applicable stage (background, noise, detection, hfr) ran and is scored; "
         "`unsupported` = at least one stage reported that it could not measure. An unsupported row whose other "
         "stages completed is shown as `partial: <reason>` with the stages it scored; its numbers cover only those "
         "stages and must not be aggregated with completed rows. `skipped` / `failed` rows carry no numbers. "
         "results.json marks each row with `partial`, `scored_stages`, `unscored_stages` and a `scored` flag per stage.",
         "",
         f"- Split: `{doc.get('split', 'dev')}`; acceptance criteria: {json.dumps(doc.get('acceptance_criteria'))}",
         "",
         "Methods:", ""]
    for m in methods:
        L.append(f"- `{m}` ({METHOD_INFO[m]['kind']}): {METHOD_INFO[m]['label']}")
    L += ["", f"## 1. {SECTION_TRUTH}", ""]
    L += ["### Status overview", ""]
    counts = {}
    for r in doc["results"]:
        counts.setdefault(r["method"], {}).setdefault(r["status"], 0)
        counts[r["method"]][r["status"]] += 1
        if r.get("partial"):
            counts[r["method"]]["partial"] = counts[r["method"]].get("partial", 0) + 1
    L.append(_table(["method", "completed", "unsupported", "of which partial", "skipped", "failed"],
                    [[m] + [counts.get(m, {}).get(s, 0) for s in ("completed", "unsupported", "partial", "skipped",
                                                                   "failed")] for m in methods]))
    L += ["", "### Per-scenario tables", "",
          "Columns: background bias = median(B_hat - B_true) [ADU]; RMSE [ADU]; LS = large-scale residual RMS "
          "(32 px blocks) [ADU]; under-src = median error under source footprints [ADU]; R err = median(R_hat/R_true - 1) "
          "against the expected sky noise (for digitized scenarios the effective std of the digitized values); "
          "R err analog = the same against the analog sigma (digitized scenarios only, secondary); "
          "compl = completeness of bright truth stars (all stars in parentheses); false = unmatched detections off "
          "every source footprint; frag = unmatched detections on source footprints; acc = accepted stars; "
          "HFR bias = median signed (HFR - HFR_true) of clean matched accepted stars [px] (relative in parentheses); "
          "sat leak = accepted stars matched to saturated truth.", ""]
    for sc in core:
        L += [f"#### `{sc.id}`", "", sc.description, ""]
        rows = []
        for m in methods:
            r = by[(sc.id, m)]
            mt = r.get("metrics", {})
            bg, nz, det, hf, bl = (mt.get(k) or {} for k in ("background", "noise", "detection", "hfr", "blends"))
            status = _row_status(r)
            if (r.get("stages", {}).get("detection") or {}).get("note"):
                status += f" [detection {r['stages']['detection']['note']}]"
            if "honest_absence" in mt:
                status += f" (honest absence: {_fmt(mt['honest_absence'])})"
            compl = "-" if not det else f"{_fmt(det.get('completeness_bright_stars'))} ({_fmt(det.get('completeness_stars'))})"
            hfr = "-" if hf.get("clean_bias_median_px") is None else \
                f"{_fmt(hf.get('clean_bias_median_px'), 3)} ({_fmt(hf.get('clean_rel_bias_median'), 3)})"
            blend = "" if not bl else f"; merged {bl.get('merged_groups')}/{bl.get('n_multi_member_groups')}, split {bl.get('split_sources')}"
            nz_an = mt.get("noise_vs_analog") or {}
            rows.append([f"`{m}`", status, _fmt(bg.get("bias_median_adu")), _fmt(bg.get("rmse_adu")),
                         _fmt(bg.get("large_scale_rms_adu")), _fmt(bg.get("bias_median_under_sources_adu")),
                         _fmt(nz.get("rel_error_median"), 3), _fmt(nz_an.get("rel_error_median"), 3), compl,
                         f"{_fmt(det.get('false_detections'))} / {_fmt(det.get('fragments_on_sources'))}{blend}",
                         _fmt(r.get("n_accepted")), hfr, _fmt(hf.get("saturated_leakage"))])
        L.append(_table(["method", "status", "bias", "RMSE", "LS", "under-src", "R err", "R err analog", "compl",
                         "false / frag", "acc", "HFR bias", "sat leak"], rows))
        meta = doc["scenarios"].get(sc.id, {}).get("metadata", {}) if isinstance(doc.get("scenarios"), dict) else {}
        if meta.get("quantization_step_adu"):
            L.append("")
            L.append(f"Digitized scenario: analog sigma {meta.get('analog_sigma_adu')} ADU, effective digitized sky "
                     f"std {_fmt(meta.get('effective_sky_std_adu'), 5)} ADU (primary R truth), expected digitized "
                     f"level {_fmt(meta.get('expected_digitized_level_adu'), 4)} ADU. "
                     + ("Level is quantization-limited: bias is scored against the expected digitized level (the "
                        "analog level is not recoverable). " if meta.get("quantization_limited_level") else "")
                     + ("" if meta.get("noise_scoreable", True) else
                        "Noise is not scoreable: the ADC has swallowed it (effective std < 0.01 step)."))
        L.append("")
    L += ["### Focus sweeps (BGE-14)", "",
          "Per-frame median HFR of accepted stars vs focuser position; hyperbola a*sqrt(1+((p-p0)/b)^2) and "
          "parabola fitted with numpy. Shift = fitted p0 - true p0 (focuser steps; sweep step 100). "
          "Jaccard = mean over frames of the accepted-truth-star set overlap with the frame nearest p0. "
          "A sweep is `completed` only if the fit succeeded and every frame produced an HFR; `partial` when the fit "
          "succeeded over a subset (frame counts given); `failed` when the fit failed.", ""]
    for sw in doc["focus_sweeps"]:
        tf = sw["truth_fit"]
        L += [f"#### `{sw['sweep']}`", "", f"{sw['description']} True p0 = {sw['p0_true']}; noiseless truth-curve "
              f"hyperbola p0 shift = {_fmt(tf['hyperbola'].get('p0_shift'))}, parabola {_fmt(tf['parabola'].get('p0_shift'))}.", ""]
        rows = []
        for m in methods:
            e = sw["methods"][m]
            if e["status"] == "skipped":
                rows.append([f"`{m}`", _status("skipped", e["reason"]), "-", "-", "-", "-", "-"])
                continue
            f = e["fit"]
            jac = [j for j in e["accepted_set_jaccard_vs_frame_nearest_p0"] if j is not None]
            hstat = _fmt(f["hyperbola"].get("p0_shift"), 1) if f["hyperbola"]["status"] == "completed" else \
                f"failed: {f['hyperbola'].get('reason')}"
            pstat = _fmt(f["parabola"].get("p0_shift"), 1) if f["parabola"]["status"] == "completed" else \
                f"failed: {f['parabola'].get('reason')}"
            rel = [x for x in e["frame_hfr_rel_error_vs_truth_median"] if x is not None]
            rows.append([f"`{m}`", _status(e["status"], e["reason"]) if e["status"] != "partial" else
                         f"partial: {e['reason']}",
                         f"{f['n_frames_valid']}/{f['n_frames']}", hstat, pstat,
                         _fmt(float(np.mean(jac)) if jac else None, 3),
                         _fmt(float(np.median(rel)) if rel else None, 3)])
        L.append(_table(["method", "status", "valid frames", "hyperbola shift", "parabola shift", "Jaccard",
                         "median rel HFR err"], rows))
        L.append("")
    L += render_v2_sections(doc)
    mesh = doc["mesh_study"]
    L += ["### Mesh-size / source-footprint study (BGE-09)", ""]
    if mesh.get("status") != "completed":
        L.append(_status(mesh.get("status"), mesh.get("reason")))
    else:
        L.append("Same 20-star field (flux 3e5 ADU) and noise realisation on the curved vignetting background, "
                 "re-rendered per footprint; SEP bw = bh in {16, 32, 64, 128}, fw = fh = 3. Bias columns are "
                 "median(B_hat - B_true) [ADU]; centre max = largest |error| at a star centre [ADU]; HFR rel = median "
                 "relative HFR bias of clean accepted stars.")
        L.append("")
        rows = []
        for r in mesh["rows"]:
            rows.append([r["footprint"], f"`{r['method']}`", _fmt(r["bw"]),
                         _row_status(r),
                         _fmt(r["bias_median_adu"]), _fmt(r["bias_median_under_sources_adu"]),
                         _fmt(r["bias_median_at_star_centres_adu"]), _fmt(r["max_abs_error_at_star_centres_adu"]),
                         _fmt(r["bias_median_far_from_sources_adu"]), _fmt(r["rmse_adu"]),
                         _fmt(r["hfr_rel_bias_median"], 3), _fmt(r["n_accepted"]), _fmt(r["completeness_stars"]),
                         f"{_fmt(r['false_detections'])} / {_fmt(r['fragments_on_sources'])}"])
        L.append(_table(["footprint", "method", "bw", "status", "global bias", "under-src", "centre median",
                         "centre max", "far", "RMSE", "HFR rel", "acc", "compl", "false / frag"], rows))
    L += ["", f"## 2. {SECTION_PARITY}", ""]
    cross = doc["cross_implementation"]
    if cross["status"] != "completed":
        L.append(_status(cross["status"], cross["reason"]))
    else:
        L += [f"Bound: {cross['bound']}", "", f"Not compared: {cross.get('not_compared')}.", ""]

        def tally_text(t):
            parts = [f"{k} {t[k]['within']}/{t[k]['compared']}" for k in ("back", "rms", "globalback", "globalrms",
                                                                            "catalog")]
            return (f"{t['cases']} cases; level/noise statuses identical in {t['statuses_identical']}/{t['cases']}. "
                    f"Within D-006 / compared (only items both sides produced; R maps compared as raw arrays "
                    f"whatever the noise status): {', '.join(parts)}.")

        L += [f"Benchmark frames: {tally_text(cross['tally'])}", ""]
        if cross.get("mesh_study_tally"):
            L += [f"Mesh-study runs (bw 16/32/64/128): {tally_text(cross['mesh_study_tally'])}", ""]
        rows = []
        for cid, p in list(cross["cases"].items()) + list((cross.get("mesh_study_cases") or {}).items()):
            rows.append([cid, f"{p['python_level_status']}/{p['rust_level_status']}",
                         f"{p['python_noise_status']}/{p['rust_noise_status']}", _fmt(p.get("back_max_abs_diff"), 6),
                         _fmt(p.get("back_within_d006_bound")), _fmt(p.get("rms_max_abs_diff"), 6),
                         _fmt(p.get("rms_within_d006_bound")), _fmt(p.get("globalback_abs_diff"), 6),
                         _fmt(p.get("globalrms_abs_diff"), 6),
                         f"{_fmt(p.get('n_candidates_python'))}/{_fmt(p.get('n_candidates_rust'))}",
                         _fmt(p.get("n_matched_within_0p5px")), _fmt(p.get("max_centroid_diff_px"), 6),
                         _fmt(p.get("catalog_within_d006"))])
        L.append(_table(["case", "level py/rs", "noise py/rs", "max |dB|", "B in bound", "max |dR| (raw)",
                         "R in bound", "|d globalback|", "|d globalrms|", "objects py/rs", "matched <=0.5 px",
                         "max centroid diff", "catalog in bound"], rows))
    L += ["", f"## 3. {SECTION_REAL}", "",
          "`ml/eval/bge/real_frames.py` can load a real-frame manifest for a future run; it refuses accuracy or "
          "pass/fail output without a predeclared thresholds file and never reads hold-out frames in this run.", ""]
    t = doc["timing"]
    L += ["## 4. Timing and memory", ""]
    if t.get("status") != "completed":
        L.append(_status(t.get("status"), t.get("reason")))
    else:
        L += [f"**{t['label']}.** Workload: {t['frame']['width']}x{t['frame']['height']} float32 frame, "
              f"{t['frame']['metadata']['params']['n_stars']} Gaussian stars, gradient sky; no masks. "
              "Python: background + extraction + harness acceptance/HFR; " + t.get("python_repeat_semantics", "") +
              ". Rust: " + ((t.get("rust") or {}).get("sep", {}) or {}).get("repeat_semantics", RUST_REPEAT_SEMANTICS) +
              (f". Rust build: {json.dumps(t.get('rust_build'))}" if t.get("rust_build") else "") +
              ". Python and Rust columns are therefore warm-call vs cold-process numbers.", ""]
        rows = []
        for m, r in t["python"].items():
            rows.append([f"`{m}`", r.get("status"), _fmt(r.get("total_ms_median"), 1), _fmt(r.get("total_ms_p95"), 1),
                         _fmt(r.get("background_ms_median"), 1),
                         _fmt((r.get("peak_rss_bytes") or 0) / 2**20, 0) if r.get("peak_rss_bytes") else "-",
                         _fmt((r.get("peak_rss_after_load_bytes") or 0) / 2**20, 0) if r.get("peak_rss_after_load_bytes") else "-"])
        rust = t.get("rust", {})
        if rust.get("status") == "skipped":
            rows.append(["`rust_*`", _status("skipped", rust.get("reason")), "-", "-", "-", "-", "-"])
        else:
            for b, r in rust.items():
                if not isinstance(r, dict):
                    continue
                rss = r.get("process_peak_rss_bytes_max")
                rows.append([f"`rust_{b}`", _status(r.get("status"), r.get("reason")),
                             _fmt(r.get("bridge_total_ms_median"), 1), _fmt(r.get("bridge_total_ms_p95"), 1),
                             _fmt(r.get("bridge_background_ms_median"), 1), _fmt(rss / 2**20, 0) if rss else "-", "-"])
        L.append(_table(["method", "status", "median ms", "p95 ms", "background ms", "peak RSS MiB",
                         "RSS after load MiB"], rows))
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
