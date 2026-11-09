"""One command for the background-estimation benchmark (BGE-09, BGE-14, BGE-22, BGE-23).

    python -m ml.eval.bge.run_benchmark --out <dir> [--rust] [--mesh-study] [--timing]

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
METHOD_INFO = {
    "python_sep": {"kind": "python_reference", "label": PM.METHOD_LABELS["python_sep"]},
    "python_global_scalar": {"kind": "python_comparator", "label": PM.METHOD_LABELS["python_global_scalar"]},
    "oracle_true_background": {"kind": "oracle", "label": PM.METHOD_LABELS["oracle_true_background"]},
    "rust_legacy_global_clip": {"kind": "rust_bridge",
                                "label": "Rust daemon legacy_global_clip via bge_eval_bridge (current default)"},
    "rust_sep": {"kind": "rust_bridge", "label": "Rust daemon SEP backend via bge_eval_bridge (opt-in)"},
}
SECTION_TRUTH = "Simulated truth (correctness under controlled assumptions; not real-camera accuracy)"
SECTION_PARITY = ("Cross-implementation agreement (Python SEP vs Rust SEP; implementation consistency, "
                  "not physical validation)")
SECTION_REAL = "Real-frame reference: not run (no predeclared thresholds; hold-out not consumed)"
#: D-006 (decisions.md), predeclared by the coordinator before any candidate result.
D006_MAP_REL = 1e-5
REPO_ROOT = Path(__file__).resolve().parents[3]


def host_label() -> str:
    mem = None
    try:
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())
    except (OSError, ValueError):
        pass
    vendor = "Apple" if sys.platform == "darwin" else platform.system()
    mem_s = f", {round(mem / 2**30)} GB" if mem else ""
    return f"available-host software timing ({vendor} {platform.machine()}, {os.cpu_count()} CPU{mem_s}), " \
           f"not a deployment target"


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


def overall_status(stages: dict):
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


def score(sc, level_measured, noise_measured, b_hat, r_hat, candidates, accepted, frame_hfr) -> dict:
    out = {}
    if sc.image.size == 0:
        out["honest_absence"] = not level_measured
        return out
    usable = ~PM.unusable_mask(sc.image, sc.masks)
    if not usable.any():
        out["honest_absence"] = not level_measured
    truth = [s.to_json() for s in sc.sources]
    if level_measured and b_hat is not None and sc.background is not None and usable.any():
        under, far = MT.source_masks(sc.source_image, sc.noise_std_sky, usable)
        star_xy = [(s.x, s.y) for s in sc.sources if s.kind == "star"]
        out["background"] = MT.background_map_metrics(b_hat, sc.background, usable, under, far, star_xy)
    if noise_measured and r_hat is not None and sc.noise_std_sky is not None and usable.any():
        out["noise"] = MT.noise_map_metrics(r_hat, sc.noise_std_sky, usable)
    if candidates is not None:
        det = MT.detection_metrics(candidates, truth, sc.width, sc.height, sc.source_image, sc.noise_std_sky)
        if any(c > 1 for c in SC.group_sizes(sc.sources).values()):
            out["blends"] = MT.blend_metrics(candidates, truth, det["_matches"])
        out["detection"] = MT.strip_private(det)
    if accepted is not None:
        hfr = MT.hfr_metrics(accepted, truth)
        hfr["frame_hfr_median_px"] = frame_hfr
        out["hfr"] = hfr
    return out


def run_python(sc, method, bg_params=None) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    try:
        res = PM.run_python_method(method, sc, bg_params=bg_params)
    except Exception as exc:  # noqa: BLE001 - recorded as a failed case, never as a pass
        rec = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "stages": {}, "metrics": {}}
        return rec, {}
    est, meas = res["estimate"], res["measurement"]
    level_ok = est.level_status == "measured"
    stages = {
        "background": {"status": "completed" if level_ok else ("failed" if est.status == "failed" else "unsupported"),
                       "reason": None if level_ok else ",".join(est.reasons) + (f" ({est.detail})" if est.detail else "")},
        "noise": {"status": "completed" if est.noise_status == "measured" else "unsupported",
                  "reason": None if est.noise_status == "measured" else f"noise {est.noise_status}: {','.join(est.reasons)}"},
        "detection": {"status": meas["detection"]["status"], "reason": meas["detection"].get("reason")},
    }
    if not level_ok:
        stages.pop("noise")
    status, reason = overall_status(stages)
    shape = sc.image.shape
    b_hat = est.back_map(shape) if level_ok else None
    r_hat = est.rms_map(shape) if est.noise_status == "measured" else None
    rec = {
        "status": status,
        "reason": reason,
        "stages": stages,
        "background_result": est.summary(),
        "detection_config_sha256": meas.get("detection_config_sha256"),
        "acceptance_policy_sha256": meas.get("acceptance_policy_sha256"),
        "rejections": meas.get("rejections"),
        "n_candidates": None if meas.get("candidates") is None else len(meas["candidates"]),
        "n_accepted": meas.get("n_accepted"),
        "frame_hfr_median_px": meas.get("frame_hfr_median_px"),
        "wall_ms": 1e3 * (time.perf_counter() - t0),
        "metrics": score(sc, level_ok, est.noise_status == "measured", b_hat, r_hat, meas.get("candidates"),
                         meas.get("accepted"), meas.get("frame_hfr_median_px")),
    }
    artifacts = {"candidates": meas.get("candidates"), "accepted": meas.get("accepted"), "back": b_hat, "rms": r_hat}
    return rec, artifacts


#: backend_error text of a daemon revision that has the bridge but not the SEP adapter.
SEP_NOT_BUILT_MARKER = "not built into this revision"
#: Daemon quality statuses meaning the detector never ran on the frame.
DETECTION_NOT_RUN = ("BackgroundUnavailable", "DetectionUnavailable")


def run_rust_parse(sc, out_dir, backend, revision=None) -> tuple[dict, dict]:
    out = bridge.read_output(out_dir, sc.id, backend, sc.width, sc.height)
    if out["status"] != "completed":
        return {"status": "failed", "reason": out["reason"], "stages": {}, "metrics": {}}, {}
    j = out["json"]
    bge = (j.get("background") or {}).get("backend_error") or ""
    if backend == "sep" and SEP_NOT_BUILT_MARKER in bge:
        rev = (revision or "unknown revision")[:12]
        return {"status": "skipped", "reason": f"skipped: rust sep backend not integrated at {rev} ({bge})",
                "stages": {}, "metrics": {}, "background_result": j.get("background")}, {}
    bgr = j.get("background") or {}
    level_ok = bgr.get("level_status") == "measured"
    noise_ok = bgr.get("noise_status") == "measured"
    spatial = bgr.get("method") == "sep_spatial_mesh"
    reasons = ",".join(bgr.get("reasons") or [])
    if spatial:
        b_hat, r_hat = out["back"], out["rms"]
    else:
        b_hat = np.float64(bgr["level"]) if level_ok and bgr.get("level") is not None else None
        r_hat = np.float64(bgr["rms"]) if noise_ok and bgr.get("rms") is not None else None
    quality = j.get("quality") or {}
    q_status = quality.get("status")
    # The bridge emits empty lists when detection never ran; those are not
    # zero-completeness measurements.
    detection_ran = j.get("detection") is not None and q_status not in DETECTION_NOT_RUN
    candidates = j.get("candidates") if detection_ran else None
    accepted = quality.get("used_stars") if detection_ran else None
    stages = {
        "background": {"status": "completed" if level_ok else "unsupported", "reason": None if level_ok else reasons},
        "noise": {"status": "completed" if noise_ok else "unsupported",
                  "reason": None if noise_ok else f"noise {bgr.get('noise_status')}: {reasons}"},
        "detection": {"status": "completed" if detection_ran else "unsupported",
                      "reason": None if detection_ran else
                      f"detection not run (quality status {q_status}; catalog_error {j.get('catalog_error')})"},
    }
    if (j.get("detection") or {}).get("noise_floor_applied"):
        stages["detection"]["note"] = "ran with the legacy 1 ADU noise floor (noise_floor_applied), a labelled heuristic"
    if level_ok and spatial and b_hat is None:
        stages["background_map"] = {"status": "unsupported", "reason": "spatial field but no .back.f32 emitted"}
    if not level_ok:
        stages.pop("noise")
    status, reason = overall_status(stages)
    bg_summary = {k: bgr.get(k) for k in ("method", "method_version", "requested_method", "config_sha256",
                                          "level_status", "noise_status", "reasons", "backend_error", "units",
                                          "noise_source", "level", "level_definition", "rms", "rms_definition",
                                          "mesh", "plane", "support", "library_version")}
    rec = {
        "status": status,
        "reason": reason,
        "stages": stages,
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
        "metrics": score(sc, level_ok, noise_ok, b_hat, r_hat, candidates, accepted, quality.get("hfr_px")),
    }
    return rec, {"candidates": candidates, "accepted": accepted, "catalog": j.get("catalog"), "back": out["back"],
                 "rms": out["rms"]}


def skipped_record(reason) -> dict:
    return {"status": "skipped", "reason": reason, "stages": {}, "metrics": {}}


# --------------------------------------------------------------------------
# Cross-implementation agreement
# --------------------------------------------------------------------------


def parity(py_art, rs_art, py_rec, rs_rec) -> dict:
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
    pl = (py_rec.get("background_result") or {})
    rl = (rs_rec.get("background_result") or {})
    out["python_level_status"], out["rust_level_status"] = pl.get("level_status"), rl.get("level_status")
    out["python_noise_status"], out["rust_noise_status"] = pl.get("noise_status"), rl.get("noise_status")
    pc = py_art.get("candidates")
    rc = rs_art.get("catalog") if rs_art.get("catalog") is not None else rs_art.get("candidates")
    if pc is not None and rc is not None:
        out.update(catalog_parity(pc, rc))
    return out


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
    ap.add_argument("--scenarios", nargs="*", default=None, help="subset of core scenarios")
    args = ap.parse_args(argv)
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
    names = args.scenarios or list(SC.SCENARIOS)
    core = [SC.build_scenario(n) for n in names]
    sweeps = [f() for f in SC.SWEEPS.values()] if args.scenarios is None else []
    cases = core + [fr for sw in sweeps for fr in sw.frames]
    group_of = {sc.id: "core" for sc in core}
    for sw in sweeps:
        for fr in sw.frames:
            group_of[fr.id] = f"focus_sweep:{sw.id}"
    note(f"built {len(core)} core scenarios and {len(sweeps)} sweeps ({len(cases)} frames)")

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
    configs = {
        "python_sep_background": PM.sep_background_config_json(),
        "python_sep_detection": PM.sep_detection_config_json(),
        "python_global_scalar": PM.global_scalar_config_json(),
        "acceptance_policy": PM.ACCEPTANCE_POLICY,
    }
    config_hashes = {k: config_sha256(v) for k, v in configs.items()}
    contract = {
        "background_hash_matches_daemon_default": config_hashes["python_sep_background"] == PM.EXPECTED_BACKGROUND_CONFIG_SHA256,
        "detection_hash_matches_daemon_default": config_hashes["python_sep_detection"] == PM.EXPECTED_DETECTION_CONFIG_SHA256,
        "expected_background_sha256": PM.EXPECTED_BACKGROUND_CONFIG_SHA256,
        "expected_detection_sha256": PM.EXPECTED_DETECTION_CONFIG_SHA256,
    }
    for k, v in configs.items():
        write_json(out / "configs" / f"{k}.json", {"config": v, "sha256": config_hashes[k]})

    # ---- Python methods
    results = []
    artifacts = {}
    for sc in cases:
        for m in PY_METHODS:
            rec, art = run_python(sc, m)
            rec.update(scenario=sc.id, group=group_of[sc.id], method=m, method_kind=METHOD_INFO[m]["kind"])
            results.append(rec)
            artifacts[(sc.id, m)] = art
    note(f"python methods done ({time.perf_counter() - t_start:.1f} s)")

    # ---- Rust bridge
    rust_info = {"status": "skipped", "reason": "skipped: --rust not requested"}
    if args.rust:
        bridge_out = out / "bridge_outputs"
        rust_info = bridge.run_bridge(manifest_path, bridge_out, args.repo, cargo_target, out / "logs" / "bridge.log",
                                      jobs=args.jobs)
        note(f"rust bridge: {rust_info['status']} {rust_info.get('reason') or ''}")
    for sc in cases:
        for m, backend in RUST_METHODS.items():
            if not sc.image.size:
                rec, art = skipped_record("skipped: zero-size frame excluded from bridge v1 manifest"), {}
            elif rust_info["status"] == "completed":
                rec, art = run_rust_parse(sc, out / "bridge_outputs", backend,
                                          (rust_info.get("revision") or {}).get("commit"))
            elif rust_info["status"] == "failed":
                rec, art = {"status": "failed", "reason": rust_info["reason"], "stages": {}, "metrics": {}}, {}
            else:
                rec, art = skipped_record(rust_info["reason"]), {}
            rec.update(scenario=sc.id, group=group_of[sc.id], method=m, method_kind=METHOD_INFO[m]["kind"])
            results.append(rec)
            artifacts[(sc.id, m)] = art

    # ---- catalogs and maps
    for (case_id, m), art in artifacts.items():
        if art.get("candidates") is not None or art.get("accepted") is not None:
            write_json(out / "catalogs" / f"{case_id}__{m}.json",
                       {"candidates": art.get("candidates"), "accepted": art.get("accepted")})
        if args.save_maps and m == "python_sep":
            for key in ("back", "rms"):
                if isinstance(art.get(key), np.ndarray):
                    (out / "maps").mkdir(exist_ok=True)
                    np.ascontiguousarray(art[key], dtype="<f4").tofile(out / "maps" / f"{case_id}__python_sep.{key}.f32")

    # ---- focus sweeps
    by_key = {(r["scenario"], r["method"]): r for r in results}
    sweep_results = []
    for sw in sweeps:
        truth_curve = [float(np.median([s.hfr_true_px for s in fr.sources])) for fr in sw.frames]
        entry = {"sweep": sw.id, "description": sw.description, "positions": sw.positions, "p0_true": sw.p0_true,
                 "truth_median_hfr_px": truth_curve, "truth_fit": MT.focus_shift(sw.positions, truth_curve, sw.p0_true),
                 "params": sw.metadata, "methods": {}}
        ref_idx = int(np.argmin(np.abs(np.asarray(sw.positions) - sw.p0_true)))
        for m in list(PY_METHODS) + list(RUST_METHODS):
            recs = [by_key[(fr.id, m)] for fr in sw.frames]
            statuses = [r["status"] for r in recs]
            if all(s == "skipped" for s in statuses):
                entry["methods"][m] = {"status": "skipped", "reason": recs[0]["reason"]}
                continue
            vals = [r.get("frame_hfr_median_px") for r in recs]
            curve = [float(v) if v is not None else float("nan") for v in vals]
            fit = MT.focus_shift(sw.positions, curve, sw.p0_true)
            sets = [(r.get("metrics", {}).get("hfr") or {}).get("matched_truth_ids") or [] for r in recs]
            jac = [MT.jaccard(s, sets[ref_idx]) for s in sets]
            hyper_ok = fit["hyperbola"]["status"] == "completed"
            entry["methods"][m] = {
                "status": "completed" if hyper_ok else "failed",
                "reason": None if hyper_ok else f"sweep fit failed: {fit['hyperbola'].get('reason')}",
                "frame_statuses": statuses,
                "median_hfr_px": curve,
                "n_accepted": [r.get("n_accepted") for r in recs],
                "fit": fit,
                "accepted_set_jaccard_vs_frame_nearest_p0": jac,
                "frame_hfr_rel_error_vs_truth_median": [
                    (c / t - 1.0) if np.isfinite(c) else None for c, t in zip(curve, truth_curve)],
            }
        sweep_results.append(entry)

    # ---- mesh study
    mesh = {"status": "skipped", "reason": "skipped: --mesh-study not requested"}
    if args.mesh_study:
        mesh = run_mesh_study(args, out, cargo_target, note)

    # ---- timing
    timing = {"status": "skipped", "reason": "skipped: --timing not requested"}
    if args.timing:
        timing = run_timing(args, out, cargo_target, rust_info, note)

    # ---- cross-implementation
    parity_rows = {}
    parity_reason = rust_info["reason"]
    if rust_info["status"] == "completed":
        for sc in cases:
            rs = by_key[(sc.id, "rust_sep")]
            if sc.image.size and rs["status"] != "skipped":
                parity_rows[sc.id] = parity(artifacts[(sc.id, "python_sep")], artifacts[(sc.id, "rust_sep")],
                                            by_key[(sc.id, "python_sep")], rs)
            elif sc.image.size:
                parity_reason = rs["reason"]
    cross = {"status": "completed" if parity_rows else "skipped",
             "reason": None if parity_rows else parity_reason,
             "interpretation": SECTION_PARITY,
             "bound": "D-006 (decisions.md): maps |delta| <= 1e-5*max(1,|ref|) per pixel; reported as diagnostic here",
             "cases": parity_rows}

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
        "interpretation": {"simulated_truth": SECTION_TRUTH, "cross_implementation": SECTION_PARITY,
                           "real_frames": SECTION_REAL},
        "methods": METHOD_INFO,
        "configs": {k: {"config": v, "sha256": config_hashes[k]} for k, v in configs.items()},
        "contract_check": contract,
        "rust_bridge": rust_info,
        "scenarios": {sc.id: {"description": sc.description, "group": group_of[sc.id], "metadata": sc.metadata,
                              "width": sc.width, "height": sc.height, "truth": sc.truth_summary(),
                              "input_sha256": input_hashes[sc.id],
                              "sources": [s.to_json() for s in sc.sources]} for sc in cases},
        "results": results,
        "focus_sweeps": sweep_results,
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


def run_mesh_study(args, out, cargo_target, note) -> dict:
    fields = SC.mesh_study_fields()
    rows = []
    mesh_dir = out / "mesh_study"
    entries = bridge.write_inputs(fields, mesh_dir)
    rust_runs = {}
    for bw in SC.MESH_STUDY_SIZES:
        mpath = bridge.write_manifest(entries, mesh_dir, name=f"manifest_bw{bw}.json", backends=("sep",),
                                      settings={"sep": {"bw": bw, "bh": bw}}, write_maps=True)
        if args.rust:
            (out / "logs").mkdir(exist_ok=True)
            rust_runs[bw] = bridge.run_bridge(mpath, mesh_dir / f"out_bw{bw}", args.repo, cargo_target,
                                              out / "logs" / f"bridge_mesh_bw{bw}.log", jobs=args.jobs)
        else:
            rust_runs[bw] = {"status": "skipped", "reason": "skipped: --rust not requested"}
    for sc in fields:
        for m in ("python_global_scalar", "oracle_true_background"):
            rec, _ = run_python(sc, m)
            rows.append(dict(_mesh_row(rec), field=sc.id, footprint=sc.metadata["footprint"], method=m, bw=None))
        for bw in SC.MESH_STUDY_SIZES:
            rec, _ = run_python(sc, "python_sep", bg_params={"bw": bw, "bh": bw})
            rows.append(dict(_mesh_row(rec), field=sc.id, footprint=sc.metadata["footprint"], method="python_sep", bw=bw))
            rr = rust_runs[bw]
            if rr["status"] == "completed":
                rrec, _ = run_rust_parse(sc, mesh_dir / f"out_bw{bw}", "sep", (rr.get("revision") or {}).get("commit"))
            else:
                rrec = {"status": rr["status"], "reason": rr["reason"], "metrics": {}}
            rows.append(dict(_mesh_row(rrec), field=sc.id, footprint=sc.metadata["footprint"], method="rust_sep", bw=bw))
    note("mesh study done")
    return {"status": "completed", "sizes": list(SC.MESH_STUDY_SIZES), "footprints": SC.MESH_STUDY_FOOTPRINTS,
            "fields": {sc.id: {"truth_hfr_median_px": float(np.median([s.hfr_true_px for s in sc.sources])),
                               "input_sha256": sc.input_hashes(), "metadata": sc.metadata} for sc in fields},
            "rust_runs": rust_runs, "rows": rows}


def _mesh_row(rec) -> dict:
    m = rec.get("metrics", {})
    bg, hfr, det = m.get("background") or {}, m.get("hfr") or {}, m.get("detection") or {}
    return {
        "status": rec["status"], "reason": rec.get("reason"),
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


def run_timing(args, out, cargo_target, rust_info, note) -> dict:
    tdir = out / "timing"
    frame = SC.timing_frame()
    entries = bridge.write_inputs([frame], tdir)
    rel = entries[0]["pixels"]
    result = {"status": "completed", "label": host_label(), "frame": {"id": frame.id, "width": frame.width,
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
    if rust_info.get("status") != "completed":
        result["rust"] = {"status": "skipped", "reason": rust_info.get("reason")}
        return result
    binary = bridge.bridge_test_binary(args.repo, cargo_target, args.jobs)
    if binary is None:
        result["rust"] = {"status": "skipped", "reason": f"{bridge.UNAVAILABLE}: test binary not found"}
        return result
    for backend in bridge.RUST_BACKENDS:
        mpath = bridge.write_manifest(entries, tdir, name=f"manifest_{backend}.json", backends=(backend,),
                                      write_maps=False)
        rows = []
        for rep in range(args.timing_repeats):
            odir = tdir / f"out_{backend}_{rep}"
            run = bridge.run_binary_timed(binary, mpath, odir, out / "logs" / f"timing_{backend}_{rep}.log")
            o = bridge.read_output(odir, frame.id, backend, frame.width, frame.height)
            run["timing_ms"] = (o.get("json") or {}).get("timing_ms")
            run["output_status"] = o["status"]
            run["backend_error"] = ((o.get("json") or {}).get("background") or {}).get("backend_error")
            rows.append(run)
        if any(SEP_NOT_BUILT_MARKER in (r.get("backend_error") or "") for r in rows):
            rev = ((rust_info.get("revision") or {}).get("commit") or "unknown")[:12]
            result["rust"][backend] = {"status": "skipped", "runs": rows,
                                       "reason": f"skipped: rust sep backend not integrated at {rev}"}
            continue
        tm = [sum((r["timing_ms"] or {}).values()) for r in rows if r.get("timing_ms")]
        bg_ms = [r["timing_ms"]["background"] for r in rows if r.get("timing_ms")]
        result["rust"][backend] = {
            "status": "completed" if tm else "failed", "binary": binary, "runs": rows,
            "bridge_total_ms_median": float(np.median(tm)) if tm else None,
            "bridge_total_ms_p95": float(np.percentile(tm, 95)) if tm else None,
            "bridge_background_ms_median": float(np.median(bg_ms)) if bg_ms else None,
            "process_peak_rss_bytes_max": max((r["peak_rss_bytes"] or 0) for r in rows) or None,
            "note": "peak RSS of the whole test process (frame load + one backend), from /usr/bin/time -l",
        }
        note(f"timing rust {backend}: {result['rust'][backend].get('bridge_total_ms_median')} ms")
    return result


# --------------------------------------------------------------------------
# summary.md
# --------------------------------------------------------------------------


def _status(status, reason) -> str:
    """'completed', or 'status: reason' without repeating the status word."""
    if status == "completed":
        return status
    reason = reason or ""
    return reason if reason.startswith(f"{status}:") else f"{status}: {reason}"


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


def render_summary(doc, core, sweeps) -> str:
    methods = list(PY_METHODS) + list(RUST_METHODS)
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
         "- Metric definitions and units: `ml/eval/bge/README.md`. Every row has an explicit status; "
         "`unsupported` means the method reported that it could not measure (no number was scored).", "",
         "Methods:", ""]
    for m in methods:
        L.append(f"- `{m}` ({METHOD_INFO[m]['kind']}): {METHOD_INFO[m]['label']}")
    L += ["", f"## 1. {SECTION_TRUTH}", ""]
    L += ["### Status overview", ""]
    counts = {}
    for r in doc["results"]:
        counts.setdefault(r["method"], {}).setdefault(r["status"], 0)
        counts[r["method"]][r["status"]] += 1
    L.append(_table(["method", "completed", "unsupported", "skipped", "failed"],
                    [[m] + [counts.get(m, {}).get(s, 0) for s in ("completed", "unsupported", "skipped", "failed")]
                     for m in methods]))
    L += ["", "### Per-scenario tables", "",
          "Columns: background bias = median(B_hat - B_true) [ADU]; RMSE [ADU]; LS = large-scale residual RMS "
          "(32 px blocks) [ADU]; under-src = median error under source footprints [ADU]; R err = median(R_hat/R_true - 1); "
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
            status = _status(r["status"], r.get("reason"))
            if (r.get("stages", {}).get("detection") or {}).get("note"):
                status += f" [detection {r['stages']['detection']['note']}]"
            if "honest_absence" in mt:
                status += f" (honest absence: {_fmt(mt['honest_absence'])})"
            compl = "-" if not det else f"{_fmt(det.get('completeness_bright_stars'))} ({_fmt(det.get('completeness_stars'))})"
            hfr = "-" if hf.get("clean_bias_median_px") is None else \
                f"{_fmt(hf.get('clean_bias_median_px'), 3)} ({_fmt(hf.get('clean_rel_bias_median'), 3)})"
            blend = "" if not bl else f"; merged {bl.get('merged_groups')}/{bl.get('n_multi_member_groups')}, split {bl.get('split_sources')}"
            rows.append([f"`{m}`", status, _fmt(bg.get("bias_median_adu")), _fmt(bg.get("rmse_adu")),
                         _fmt(bg.get("large_scale_rms_adu")), _fmt(bg.get("bias_median_under_sources_adu")),
                         _fmt(nz.get("rel_error_median"), 3), compl,
                         f"{_fmt(det.get('false_detections'))} / {_fmt(det.get('fragments_on_sources'))}{blend}",
                         _fmt(hf.get("n_accepted")), hfr, _fmt(hf.get("saturated_leakage"))])
        L.append(_table(["method", "status", "bias", "RMSE", "LS", "under-src", "R err", "compl", "false / frag",
                         "acc", "HFR bias", "sat leak"], rows))
        L.append("")
    L += ["### Focus sweeps (BGE-14)", "",
          "Per-frame median HFR of accepted stars vs focuser position; hyperbola a*sqrt(1+((p-p0)/b)^2) and "
          "parabola fitted with numpy. Shift = fitted p0 - true p0 (focuser steps; sweep step 100). "
          "Jaccard = mean over frames of the accepted-truth-star set overlap with the frame nearest p0.", ""]
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
            rows.append([f"`{m}`", _status(e["status"], e["reason"]),
                         f"{f['n_frames_valid']}/{f['n_frames']}", hstat, pstat,
                         _fmt(float(np.mean(jac)) if jac else None, 3),
                         _fmt(float(np.median(rel)) if rel else None, 3)])
        L.append(_table(["method", "status", "valid frames", "hyperbola shift", "parabola shift", "Jaccard",
                         "median rel HFR err"], rows))
        L.append("")
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
                         _status(r["status"], r["reason"]),
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
        L.append(f"Bound: {cross['bound']}.")
        L.append("")
        rows = []
        for cid, p in cross["cases"].items():
            rows.append([cid, f"{p['python_level_status']}/{p['rust_level_status']}",
                         f"{p['python_noise_status']}/{p['rust_noise_status']}", _fmt(p.get("back_max_abs_diff"), 6),
                         _fmt(p.get("back_within_d006_bound")), _fmt(p.get("rms_max_abs_diff"), 6),
                         _fmt(p.get("rms_within_d006_bound")),
                         f"{_fmt(p.get('n_candidates_python'))}/{_fmt(p.get('n_candidates_rust'))}",
                         _fmt(p.get("n_matched_within_0p5px")), _fmt(p.get("max_centroid_diff_px"), 6),
                         _fmt(p.get("catalog_within_d006"))])
        L.append(_table(["case", "level py/rs", "noise py/rs", "max |dB|", "B in bound", "max |dR|", "R in bound",
                         "objects py/rs", "matched <=0.5 px", "max centroid diff", "catalog in bound"], rows))
        cases = list(cross["cases"].values())

        def tally(key):
            vals = [p[key] for p in cases if p.get(key) is not None]
            return f"{sum(bool(v) for v in vals)}/{len(vals)}"

        statuses_equal = sum(1 for p in cases if (p["python_level_status"], p["python_noise_status"]) ==
                             (p["rust_level_status"], p["rust_noise_status"]))
        L.insert(len(L) - 1, f"{len(cases)} cases. Level/noise statuses identical in {statuses_equal}/{len(cases)}. "
                             f"Within D-006 where both sides produced the item: B maps {tally('back_within_d006_bound')}, "
                             f"R maps {tally('rms_within_d006_bound')}, catalogs {tally('catalog_within_d006')}.")
        L.insert(len(L) - 1, "")
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
              "Python: background + extraction + harness acceptance/HFR, one subprocess per method, median of repeats "
              "after one warm-up, peak RSS from getrusage. Rust: bridge timing_ms (background + quality + trailing) and "
              "/usr/bin/time -l around the test binary.", ""]
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
