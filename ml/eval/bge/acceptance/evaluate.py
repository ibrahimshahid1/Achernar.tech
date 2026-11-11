"""Evaluate the predeclared BGE v2 acceptance criteria against benchmark results.

    python -m ml.eval.bge.acceptance.evaluate --results <results.json> [<results.json> ...]
        [--criteria ml/eval/bge/acceptance/synthetic-v2.json] [--out <verdict.json>]

Reads the criteria file (never written here) and one or more ``results.json``
files of ``ml.eval.bge.run_benchmark`` (merged: e.g. a held-out run plus the
development run with the mesh study and timing). Prints one line per criterion
and method: ``pass``, ``fail`` or ``not_evaluable`` with the measured value;
thresholds and comparisons are taken from the criteria file as written. A
criterion passes only if every implementation it applies to was evaluated and
passed; a failed criterion is reported as failed.

Interpretations (fixed here, before any held-out run; see README.md):

* Implementations: ``"v2"``/``"v1"`` map to the method suffixes of the file's
  ``variants`` (``__n2g2``/``__n1g1`` for SEP, ``__g2``/``__g1`` for legacy).
  SEP criteria apply to ``python_sep*`` and ``rust_sep*``; criteria without a
  ``backend`` also apply to ``rust_legacy*``.
* Donut acceptance: truth stars of the listed sets matched by an accepted star
  (radius max(2, 0.5 HFR_true)), pooled over the sets; a frame where
  detection did not run accepted nothing; a set missing from the results, or
  with skipped frames, makes the criterion not evaluable.
* Close-double rejection: a pair is accepted when an accepted star lies within
  separation/2 + max(2, 0.5 HFR_true) of its midpoint, else rejected.
* Sweeps: the hyperbola fit of the per-frame median HFR; ``p0 shift`` is the
  fitted minus true optimum, compared with the value times the sweep step.
  The ``requires`` clause must hold for a pass.
* False detections per frame: detections off every truth footprint; a frame
  whose detection did not run (noise unresolved) detected nothing and counts 0.
* A8: per pre-existing development frame row; completeness = detection
  completeness of truth stars, false detections per frame, |median relative
  HFR bias| of clean accepted stars. A value v1 has and v2 lacks fails; a value
  v1 lacks is not compared.
* A9: |median(B_hat - B_true) under source footprints| of the mesh study; the
  binding comparison is at the default mesh size (64 px), the strict reading
  over every mesh size is reported beside it.
* A10: Rust bridge total time, median of the timed repeats.
* A11: the per-case Rust-Python parity of results.json (spec §5), every SEP
  variant, benchmark frames and mesh-study runs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .. import metrics as MT
from .. import run_benchmark as RB

DEFAULT_CRITERIA = Path(__file__).resolve().parent / "synthetic-v2.json"
PASS, FAIL, NOT_EVALUABLE = "pass", "fail", "not_evaluable"
_COMPARE = re.compile(
    r"^(?:(?P<metric>[a-z_]+)\s+)?(?P<lhs>abs\(v2\)|v2)\s*(?P<op><=|>=)\s*"
    r"(?:(?P<mult>[0-9.]+)\s*\*\s*)?(?P<rhs>abs\(v1\)|abs_v1|v1)\s*(?:(?P<sign>[+-])\s*(?P<off>[0-9.]+))?(?P<rest>.*)$")


def parse_compare(text: str) -> dict:
    """``"v2 >= v1 - 0.05"`` -> {op, mult, offset, abs, metric}; numbers come from the text."""
    m = _COMPARE.match(text.strip())
    if not m:
        raise ValueError(f"unparsed comparison {text!r}")
    offset = float(m["off"]) * (-1.0 if m["sign"] == "-" else 1.0) if m["off"] else 0.0
    return {"metric": m["metric"], "op": m["op"], "mult": float(m["mult"]) if m["mult"] else 1.0, "offset": offset,
            "abs": m["lhs"].startswith("abs") or m["rhs"].startswith("abs"), "rest": m["rest"].strip(), "text": text}


def compare(spec: dict, v2, v1) -> bool:
    a, b = (abs(v2), abs(v1)) if spec["abs"] else (v2, v1)
    bound = spec["mult"] * b + spec["offset"]
    return a <= bound if spec["op"] == "<=" else a >= bound


def apply_op(op: str, value, threshold) -> bool:
    return value <= threshold if op == "<=" else value >= threshold


def merge_results(docs: list) -> dict:
    """One view over several results.json files (rows, scenarios, sets, sweeps,
    mesh study, timing, parity)."""
    merged = {"results": [], "scenarios": {}, "scenario_sets": {}, "focus_sweeps": [], "mesh_study": None,
              "timing": None, "parity_v2": {}, "mesh_parity_v2": {}, "sources": []}
    seen_rows = set()
    for doc in docs:
        merged["sources"].append({"command": doc.get("command"), "git": doc.get("git"), "split": doc.get("split"),
                                  "acceptance_criteria": doc.get("acceptance_criteria")})
        for r in doc.get("results", []):
            key = (r["scenario"], r["method"])
            if key not in seen_rows:
                seen_rows.add(key)
                merged["results"].append(r)
        merged["scenarios"].update(doc.get("scenarios") or {})
        merged["scenario_sets"].update(doc.get("scenario_sets") or {})
        known = {sw["sweep"] for sw in merged["focus_sweeps"]}
        merged["focus_sweeps"] += [sw for sw in doc.get("focus_sweeps") or [] if sw["sweep"] not in known]
        if (doc.get("mesh_study") or {}).get("status") == "completed" and merged["mesh_study"] is None:
            merged["mesh_study"] = doc["mesh_study"]
        if (doc.get("timing") or {}).get("status") == "completed" and merged["timing"] is None:
            merged["timing"] = doc["timing"]
        for variant, entry in ((doc.get("cross_implementation") or {}).get("v2") or {}).items():
            if entry.get("status") == "completed":
                merged["parity_v2"].setdefault(variant, {}).update(entry["cases"])
        for variant, entry in (((doc.get("mesh_study") or {}).get("parity_v2")) or {}).items():
            merged["mesh_parity_v2"].setdefault(variant, {}).update(entry.get("cases") or {})
    merged["by_key"] = {(r["scenario"], r["method"]): r for r in merged["results"]}
    return merged


def implementations(criterion: dict, variants: dict, which: str) -> list:
    """Method ids a criterion applies to for ``which`` in {"v2", "v1"}."""
    sep, legacy = variants[which]["sep"], variants[which]["legacy"]
    methods = [f"python_sep{sep}", f"rust_sep{sep}"]
    if criterion.get("backend") != "sep":
        methods.append(f"rust_legacy{legacy}")
    return methods


def paired(criterion, variants) -> list:
    return list(zip(implementations(criterion, variants, "v2"), implementations(criterion, variants, "v1")))


def method_present(merged, method) -> bool:
    return any(m == method for (_, m) in merged["by_key"])


def set_frames(merged, set_ids):
    frames, missing = [], []
    for sid in set_ids:
        if sid in merged["scenario_sets"]:
            frames += merged["scenario_sets"][sid]["frames"]
        else:
            sweep = next((sw for sw in merged["focus_sweeps"] if sw["sweep"] == sid), None)
            if sweep is None:
                missing.append(sid)
            else:
                frames += [fid for fid in merged["scenarios"] if merged["scenarios"][fid]["group"] == f"focus_sweep:{sid}"]
    return frames, missing


def frames_for(merged, frame_ids, method):
    frames = RB.frame_view(merged["scenarios"], merged["by_key"], frame_ids, method)
    not_run = sum(1 for fr in frames if fr["status"] in (None, "skipped"))
    return frames, not_run


def _verdict(entries: list) -> str:
    statuses = [e["status"] for e in entries]
    if not statuses:
        return NOT_EVALUABLE
    if FAIL in statuses:
        return FAIL
    if NOT_EVALUABLE in statuses:
        return NOT_EVALUABLE
    return PASS


def _population(criterion):
    text = criterion.get("population", "")
    if "vane_width_frac" in text:
        limit = float(re.search(r"vane_width_frac\s*<=\s*([0-9.]+)", text).group(1))
        return lambda s: s.get("vane_width_frac") is not None and (s.get("n_vanes") or 0) > 0 \
            and s["vane_width_frac"] <= limit
    if "r_out_px" in text:
        limit = float(re.search(r"r_out_px\s*>=\s*([0-9.]+)", text).group(1))
        isolated = "isolated" in text
        return lambda s: (s.get("r_out_px") or 0.0) >= limit and (bool(s.get("isolated")) or not isolated)
    return lambda s: "r_out_px" in s


def eval_donut_acceptance(c, merged, variants):
    frame_ids, missing = set_frames(merged, c["scenarios"])
    entries = []
    for m in implementations(c, variants, c["variant"]):
        if missing or not method_present(merged, m):
            entries.append({"method": m, "status": NOT_EVALUABLE, "detail": f"missing: {missing or m}"})
            continue
        frames, not_run = frames_for(merged, frame_ids, m)
        if not_run:
            entries.append({"method": m, "status": NOT_EVALUABLE, "detail": f"{not_run} frames not run"})
            continue
        res = MT.acceptance_rate(frames, _population(c))
        if res["rate"] is None:
            entries.append({"method": m, "status": NOT_EVALUABLE, "detail": "empty population", **res})
            continue
        entries.append({"method": m, "status": PASS if apply_op(c["op"], res["rate"], c["value"]) else FAIL,
                        "value": res["rate"], "detail": f"{res['n_accepted']}/{res['n_population']}", **res})
    return entries


def eval_close_doubles(c, merged, variants):
    frame_ids, missing = set_frames(merged, c["scenarios"])
    text = c.get("population", "")
    min_sep = float(re.search(r"separation_px\s*>=\s*([0-9.]+)", text).group(1))
    min_ratio = float(re.search(r"flux_ratio\s*>=\s*([0-9.]+)", text).group(1))
    spec = parse_compare(c["compare"])
    entries = []
    for m2, m1 in paired(c, variants):
        if missing or not (method_present(merged, m2) and method_present(merged, m1)):
            entries.append({"method": m2, "status": NOT_EVALUABLE, "detail": f"missing: {missing or [m2, m1]}"})
            continue
        (f2, n2), (f1, n1) = frames_for(merged, frame_ids, m2), frames_for(merged, frame_ids, m1)
        if n2 or n1:
            entries.append({"method": m2, "status": NOT_EVALUABLE, "detail": "frames not run"})
            continue
        r2, r1 = MT.close_double_rejection(f2, min_sep, min_ratio), MT.close_double_rejection(f1, min_sep, min_ratio)
        if r2["rate"] is None or r1["rate"] is None:
            entries.append({"method": m2, "status": NOT_EVALUABLE, "detail": "no pair in the population"})
            continue
        ok = compare(spec, r2["rate"], r1["rate"])
        entries.append({"method": m2, "baseline": m1, "status": PASS if ok else FAIL, "value": r2["rate"],
                        "baseline_value": r1["rate"],
                        "detail": f"v2 {r2['n_rejected']}/{r2['n_pairs']} vs v1 {r1['n_rejected']}/{r1['n_pairs']}"})
    return entries


def eval_hfr_bias(c, merged, variants):
    frame_ids, missing = set_frames(merged, c["scenarios"])
    entries = []
    for m in implementations(c, variants, c["variant"]):
        if missing or not method_present(merged, m):
            entries.append({"method": m, "status": NOT_EVALUABLE, "detail": f"missing: {missing or m}"})
            continue
        frames, not_run = frames_for(merged, frame_ids, m)
        res = MT.accepted_rel_hfr_bias(frames, lambda s: "r_out_px" in s)
        if not_run or res["abs_median_rel_bias"] is None:
            entries.append({"method": m, "status": NOT_EVALUABLE,
                            "detail": f"{not_run} frames not run; {res['n_matched']} matched accepted donuts"})
            continue
        entries.append({"method": m, "status": PASS if apply_op(c["op"], res["abs_median_rel_bias"], c["value"]) else FAIL,
                        "value": res["abs_median_rel_bias"], "detail": f"median rel bias {res['median_rel_bias']:+.4f} "
                                                                     f"over {res['n_matched']} accepted donuts"})
    return entries


def eval_sweep(c, merged, variants):
    entries = []
    sweep = next((sw for sw in merged["focus_sweeps"] if sw["sweep"] == c["scenarios"][0]), None)
    for m in implementations(c, variants, c["variant"]):
        e = (sweep or {}).get("methods", {}).get(m)
        if sweep is None or e is None or e.get("status") == "skipped":
            entries.append({"method": m, "status": NOT_EVALUABLE, "detail": "sweep or method not in results"})
            continue
        step = float(sweep.get("step") or 100.0)
        req = c.get("requires", "")
        hyper = (e.get("fit") or {}).get("hyperbola") or {}
        left_right = hyper.get("valid_frames_left_right") or [0, 0]
        problems = []
        if "noise measured" in req:
            need = int(re.search(r">=\s*([0-9]+)\s*of", req).group(1))
            if (e.get("frames_noise_measured") or 0) < need:
                problems.append(f"noise measured on {e.get('frames_noise_measured')}/{len(sweep['positions'])} < {need}")
        if "each side" in req:
            need = int(re.search(r">=\s*([0-9]+)\s*valid frames", req).group(1))
            if hyper.get("status") != "completed" or min(left_right) < need:
                problems.append(f"fit {hyper.get('status')} ({hyper.get('reason')}); valid frames left/right {left_right}")
        shift = hyper.get("p0_shift")
        if hyper.get("status") != "completed" or shift is None:
            problems.append(f"hyperbola fit {hyper.get('status')}: {hyper.get('reason')}")
            value = None
        else:
            value = abs(shift) / step
            if not value <= c["value"]:
                problems.append(f"|p0 shift| {abs(shift):.1f} steps = {value:.3f} step > {c['value']}")
        entries.append({"method": m, "status": FAIL if problems else PASS, "value": value,
                        "detail": "; ".join(problems) or f"|p0 shift| = {value:.3f} step; left/right {left_right}; "
                                                         f"noise measured {e.get('frames_noise_measured')}"})
    return entries


def eval_false_detections(c, merged, variants):
    frame_ids, missing = set_frames(merged, c["scenarios"])
    frame_ids = [f for f in frame_ids if merged["scenarios"][f]["metadata"].get("starless")]
    spec = parse_compare(c["compare"]) if c.get("compare") else None
    entries = []
    for m2, m1 in paired(c, variants):
        if missing or not frame_ids or not method_present(merged, m2):
            entries.append({"method": m2, "status": NOT_EVALUABLE, "detail": f"missing: {missing or m2}"})
            continue
        f2, n2 = frames_for(merged, frame_ids, m2)
        r2 = MT.false_detections_per_frame(f2)
        problems = []
        if n2:
            entries.append({"method": m2, "status": NOT_EVALUABLE, "detail": f"{n2} frames not run"})
            continue
        if not apply_op(c["op"], r2["per_frame"], c["value"]):
            problems.append(f"{r2['per_frame']:.2f} per frame not {c['op']} {c['value']}")
        detail = f"v2 {r2['per_frame']:.2f}/frame (values {r2['values']}, {r2['frames_without_detection']} frames without detection)"
        status = None
        if spec:
            if not method_present(merged, m1):
                status = NOT_EVALUABLE
                detail += f"; baseline {m1} missing"
            else:
                f1, n1 = frames_for(merged, frame_ids, m1)
                r1 = MT.false_detections_per_frame(f1)
                detail += f"; v1 {r1['per_frame']:.2f}/frame (values {r1['values']})"
                if n1:
                    status = NOT_EVALUABLE
                elif not compare(spec, r2["per_frame"], r1["per_frame"]):
                    problems.append(f"v2 {r2['per_frame']:.2f} not {spec['text']} (v1 {r1['per_frame']:.2f})")
        entries.append({"method": m2, "baseline": m1, "status": status or (FAIL if problems else PASS),
                        "value": r2["per_frame"], "detail": "; ".join(problems + [detail])})
    return entries


def _a8_values(rec):
    if rec is None:
        return {}
    mt = rec.get("metrics") or {}
    det = mt.get("detection") or {}
    hfr = mt.get("hfr") or {}
    out = {}
    if rec.get("detection_ran") or det:
        out["completeness"] = det.get("completeness_stars", det.get("completeness"))
        out["false_detections_per_frame"] = det.get("false_detections")
    elif rec.get("status") in ("completed", "unsupported"):
        out["false_detections_per_frame"] = 0
    out["abs_median_hfr_bias"] = hfr.get("clean_rel_bias_median")
    return {k: v for k, v in out.items() if v is not None}


def eval_no_regression(c, merged, variants):
    specs = [parse_compare(t) for t in c["compare"]]
    rows = sorted({sid for sid, s in merged["scenarios"].items() if s.get("pre_existing")})
    entries = []
    for m2, m1 in paired(c, variants):
        if not (method_present(merged, m2) and method_present(merged, m1)) or not rows:
            entries.append({"method": m2, "status": NOT_EVALUABLE, "detail": f"missing rows or methods {m2}/{m1}"})
            continue
        failures, compared, not_run = [], 0, []
        for sid in rows:
            r2, r1 = merged["by_key"].get((sid, m2)), merged["by_key"].get((sid, m1))
            if r2 is not None and r1 is not None and r2["status"] == r1["status"] == "skipped":
                continue  # excluded for both (e.g. zero_size is not in the bridge manifest)
            if r2 is None or r1 is None or r2["status"] == "skipped" or r1["status"] == "skipped":
                not_run.append(sid)
                continue
            v2, v1 = _a8_values(r2), _a8_values(r1)
            for spec in specs:
                key = spec["metric"]
                if key not in v1:
                    continue
                compared += 1
                if key not in v2:
                    failures.append(f"{sid}: {key} v1 {v1[key]} but v2 none")
                elif not compare(spec, v2[key], v1[key]):
                    failures.append(f"{sid}: {key} v2 {v2[key]:.4g} vs v1 {v1[key]:.4g} ({spec['text']})")
        status = FAIL if failures else (NOT_EVALUABLE if not_run else PASS)
        entries.append({"method": m2, "baseline": m1, "status": status, "value": len(failures),
                        "detail": (f"{len(failures)} failing of {compared} comparisons over {len(rows)} rows"
                                   + (f"; not run: {len(not_run)}" if not_run else "")),
                        "failures": failures})
    return entries


def eval_mesh_bias(c, merged, variants):
    mesh = merged["mesh_study"]
    entries = []
    for m2, m1 in paired(c, variants):
        if mesh is None:
            entries.append({"method": m2, "status": NOT_EVALUABLE, "detail": "no completed mesh study"})
            continue
        rows = {(r["method"], r["footprint"], r["bw"]): r for r in mesh["rows"]}
        footprints = sorted({r["footprint"] for r in mesh["rows"]})
        default_bw = mesh.get("default_bw", 64)
        if not any(k[0] == m2 for k in rows) or not any(k[0] == m1 for k in rows):
            entries.append({"method": m2, "status": NOT_EVALUABLE, "detail": f"mesh rows for {m2}/{m1} missing"})
            continue

        def check(bws):
            fails, missing = [], []
            for fp in footprints:
                for bw in bws:
                    a = (rows.get((m2, fp, bw)) or {}).get("bias_median_under_sources_adu")
                    b = (rows.get((m1, fp, bw)) or {}).get("bias_median_under_sources_adu")
                    if b is None:
                        missing.append(f"{fp}@{bw}")
                        continue
                    if a is None or not abs(a) <= abs(b):
                        fails.append(f"{fp}@bw{bw}: |v2| {abs(a) if a is not None else 'none'} > |v1| {abs(b):.3f}")
            return fails, missing

        fails, missing = check([default_bw])
        strict_fails, _ = check(mesh.get("sizes") or [default_bw])
        status = FAIL if fails else (NOT_EVALUABLE if missing else PASS)
        entries.append({"method": m2, "baseline": m1, "status": status, "value": len(fails),
                        "detail": f"default bw {default_bw}: {len(fails)} footprints worse"
                                  + (f" ({'; '.join(fails)})" if fails else "")
                                  + f"; strict reading over every mesh size: {len(strict_fails)} worse"
                                  + (f" ({'; '.join(strict_fails)})" if strict_fails else ""),
                        "strict_all_mesh_sizes_pass": not strict_fails})
    return entries


def eval_timing(c, merged, variants):
    spec = parse_compare(c["compare"])
    timing = merged["timing"]
    entries = []
    v2 = "rust_sep" + variants["v2"]["sep"]
    v1 = "rust_sep" + variants["v1"]["sep"]
    rust = (timing or {}).get("rust") or {}
    a, b = rust.get(v2) or {}, rust.get(v1) or {}
    if a.get("status") != "completed" or b.get("status") != "completed":
        return [{"method": v2, "status": NOT_EVALUABLE, "detail": "Rust timing of both variants not in results"}]
    t2, t1 = a["bridge_total_ms_median"], b["bridge_total_ms_median"]
    entries.append({"method": v2, "baseline": v1, "status": PASS if compare(spec, t2, t1) else FAIL,
                    "value": t2 / t1, "detail": f"{t2:.0f} ms vs {t1:.0f} ms (ratio {t2 / t1:.2f}; {spec['text']})"})
    return entries


def eval_parity(c, merged, variants):
    entries = []
    for variant in ("n1g1", "n2g1", "n1g2", "n2g2"):
        cases = merged["parity_v2"].get(variant)
        mesh_cases = merged["mesh_parity_v2"].get(variant, {})
        if not cases:
            entries.append({"method": f"sep__{variant}", "status": NOT_EVALUABLE, "detail": "no parity rows"})
            continue
        all_cases = dict(cases, **{f"mesh:{k}": v for k, v in mesh_cases.items()})
        bad = sorted(cid for cid, r in all_cases.items() if not r.get("all_agree"))
        entries.append({"method": f"sep__{variant}", "status": FAIL if bad else PASS, "value": len(bad),
                        "detail": f"{len(all_cases) - len(bad)}/{len(all_cases)} cases agree"
                                  + (f"; disagree: {', '.join(bad[:20])}" if bad else "")})
    return entries


EVALUATORS = {
    "donut_acceptance_rate": eval_donut_acceptance,
    "close_double_rejection_rate": eval_close_doubles,
    "abs_median_rel_hfr_bias_accepted_donuts": eval_hfr_bias,
    "sweep": eval_sweep,
    "false_detections_per_frame": eval_false_detections,
    "no_regression": eval_no_regression,
    "background_bias_under_stars": eval_mesh_bias,
    "timing_6000x4000_median_of_5_host_only": eval_timing,
    "rust_python_parity": eval_parity,
}


def evaluate(criteria: dict, docs: list) -> dict:
    merged = merge_results(docs)
    variants = criteria["variants"]
    out = {"criteria_name": criteria.get("name"), "schema": criteria.get("schema"), "results_sources": merged["sources"],
           "criteria": []}
    for c in criteria["criteria"]:
        evaluator = EVALUATORS.get(c["metric"])
        if evaluator is None:
            entries = [{"method": None, "status": NOT_EVALUABLE, "detail": f"no evaluator for metric {c['metric']}"}]
        else:
            try:
                entries = evaluator(c, merged, variants)
            except Exception as exc:  # noqa: BLE001 - reported, never a pass
                entries = [{"method": None, "status": NOT_EVALUABLE, "detail": f"{type(exc).__name__}: {exc}"}]
        out["criteria"].append({"id": c["id"], "metric": c["metric"], "status": _verdict(entries), "entries": entries,
                                "criterion": c})
    out["summary"] = {s: sum(1 for c in out["criteria"] if c["status"] == s) for s in (PASS, FAIL, NOT_EVALUABLE)}
    return out


def render(verdict: dict) -> str:
    lines = [f"{verdict['criteria_name']}: " + ", ".join(f"{k} {v}" for k, v in verdict["summary"].items()), ""]
    for c in verdict["criteria"]:
        lines.append(f"{c['id']:4s} {c['status'].upper():14s} {c['metric']}")
        for e in c["entries"]:
            value = "" if e.get("value") is None else (f" value={e['value']:.4g}" if isinstance(e["value"], float)
                                                       else f" value={e['value']}")
            lines.append(f"       {e['status']:14s} {e.get('method') or '-'}{value}: {e.get('detail', '')}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True, type=Path)
    ap.add_argument("--criteria", type=Path, default=DEFAULT_CRITERIA)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    criteria = json.loads(args.criteria.read_text())
    if criteria.get("schema") != "bge-synthetic-acceptance-v1":
        print(f"unexpected criteria schema {criteria.get('schema')!r}", file=sys.stderr)
        return 2
    docs = [json.loads(p.read_text()) for p in args.results]
    verdict = evaluate(criteria, docs)
    verdict["criteria_file"] = str(args.criteria)
    print(render(verdict))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(verdict, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
