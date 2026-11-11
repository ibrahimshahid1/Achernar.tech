"""Real-frame qualification runner.

    python -m ml.eval.bge.qualification.run_qualification \\
        --manifest <bge-real-frames-v2.json> --split development|holdout \\
        [--thresholds ml/eval/bge/qualification/thresholds-real-v1.json] \\
        [--methods rust_legacy__g1,rust_legacy__g2,rust_sep__n2g2,...] [--rust] --out <dir>

Development: observations on development nights; with ``--thresholds`` the
criteria are also evaluated, labelled informational (never a qualification
claim). Hold-out: refused unless the thresholds file and split rule are
tracked and unchanged vs HEAD, the code is clean, the manifest's split matches
the rule and the hold-out has not been consumed (``--acknowledge-repeat``
overrides the last check and is recorded). A ``started`` ledger entry is
appended before any hold-out pixel is read, a ``completed`` entry after.

Outputs in ``--out``: ``results.json`` (schema bge-real-qualification-results-v1),
``frames.jsonl`` (per frame x method rows), ``summary.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path

import numpy as np

from . import (DEFAULT_LEDGER, DEFAULT_SPLIT_RULE, DEFAULT_THRESHOLDS, QUALIFICATION_VERSION, REPO_ROOT,
               RESULTS_SCHEMA, config_sha256, sha256_file)
from . import holdout as HO
from . import split as SPLIT
from . import thresholds as TH
from .engine import Engine, EngineConfig, s0_from_calibration
from .manifest import frames_by_night, load_manifest, representation
from .methods import DEFAULT_METHODS, MethodSuite, RustBridgeRunner, canonical, method_config_sha256

SECTION = {
    "development": "Development nights: observations; criteria (if thresholds given) are informational, not a "
                   "qualification claim, and may guide changes before the hold-out.",
    "holdout": "Hold-out nights, one-shot: thresholds, split rule and code frozen in git before this run.",
}


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--split", required=True, choices=("development", "holdout"))
    ap.add_argument("--thresholds", type=Path, default=None,
                    help=f"bge-real-thresholds-v2 file (required for holdout; default file: {DEFAULT_THRESHOLDS})")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rust", action="store_true", help="run Rust variants through the bge_eval_bridge test")
    ap.add_argument("--repo", type=Path, default=REPO_ROOT)
    ap.add_argument("--cargo-target-dir", type=Path, default=None)
    ap.add_argument("--keep-bridge-io", action="store_true")
    ap.add_argument("--split-rule", type=Path, default=DEFAULT_SPLIT_RULE)
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--acknowledge-repeat", action="store_true")
    ap.add_argument("--role", action="append", default=[],
                    help="development only: override a threshold role, e.g. sep_v2=python_sep__n2g2")
    ap.add_argument("--no-injection", action="store_true")
    ap.add_argument("--injection-rounds", type=int, default=EngineConfig.injection_rounds)
    ap.add_argument("--injections-per-round", type=int, default=EngineConfig.injections_per_round)
    ap.add_argument("--bright-rounds", type=int, default=EngineConfig.bright_rounds,
                    help="injection rounds per frame bright enough for the daemon's 48-star HFR cap (R7)")
    ap.add_argument("--inject-sweep-frames", action="store_true")
    ap.add_argument("--dark-trials", type=int, default=EngineConfig.dark_trials_per_night)
    ap.add_argument("--seed", type=int, default=EngineConfig.seed)
    ap.add_argument("--max-frames-per-night", type=int, default=None, help="development only")
    ap.add_argument("--no-verify", action="store_true", help="development only: skip sha256 checks on load")
    return ap.parse_args(argv)


def run(args, suite=None, git_fn=HO.git, log=None) -> dict:
    log = log or (lambda msg: print(msg, file=sys.stderr))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    manifest_sha = sha256_file(args.manifest)
    rule = SPLIT.load_rule(args.split_rule)
    split_problems = SPLIT.check_manifest(manifest["frames"], rule)
    thresholds = TH.load_thresholds(args.thresholds) if args.thresholds else None
    holdout = args.split == "holdout"
    if holdout:
        if thresholds is None:
            raise HO.HoldoutRefusal("a hold-out run requires --thresholds")
        if args.role or args.max_frames_per_night or args.no_verify or args.no_injection:
            raise HO.HoldoutRefusal("--role, --max-frames-per-night, --no-verify and --no-injection are development-only")
        if manifest.get("upgraded_from"):
            raise HO.HoldoutRefusal("a v1 manifest has no nights: build a bge-real-frames-v2 manifest for the hold-out")
    roles = dict(thresholds["variants"]) if thresholds else {"sep_v2": "rust_sep__n2g2",
                                                             "legacy_background": "rust_legacy__g2",
                                                             "legacy_gate": "rust_legacy__g1"}
    for item in args.role:
        key, _, value = item.partition("=")
        if key not in roles:
            raise ValueError(f"--role {item!r}: unknown role {key!r}")
        roles[key] = canonical(value)
    methods = [canonical(m) for m in args.methods.split(",") if m.strip()]
    for m in roles.values():
        if m not in methods:
            methods.append(m)
    cfg = EngineConfig(injection=not args.no_injection, injection_rounds=args.injection_rounds,
                       injections_per_round=args.injections_per_round, bright_rounds=args.bright_rounds,
                       inject_sweep_frames=args.inject_sweep_frames,
                       dark_trials_per_night=args.dark_trials, seed=args.seed,
                       max_frames_per_night=args.max_frames_per_night, verify_sha=not args.no_verify)
    engine_cfg = {k: getattr(cfg, k) for k in ("injection", "injection_rounds", "injections_per_round",
                                                "bright_rounds", "inject_sweep_frames", "dark_trials_per_night",
                                                "seed")}
    config_hashes = {m: method_config_sha256(m) for m in methods}
    config_hashes["__engine__"] = config_sha256(engine_cfg)
    config_hashes["__roles__"] = config_sha256(roles)
    if suite is None:
        rust = None
        if args.rust:
            target = args.cargo_target_dir or (out_dir / "cargo-target")
            rust = RustBridgeRunner(args.repo, out_dir / "bridge_io", target, keep_io=args.keep_bridge_io)
        suite = MethodSuite(methods, rust_runner=rust)
    nights_all = frames_by_night(manifest)
    selected = frames_by_night(manifest, args.split)
    holdout_nights = sorted(frames_by_night(manifest, "holdout"))
    started = None
    if holdout:
        started = HO.guard(thresholds_path=args.thresholds, split_rule_path=args.split_rule, ledger_path=args.ledger,
                           manifest_sha=manifest_sha, config_hashes=config_hashes, holdout_nights=holdout_nights,
                           split_problems=split_problems, acknowledge_repeat=args.acknowledge_repeat,
                           include_rust=args.rust, git_fn=git_fn,
                           command=("python -m ml.eval.bge.qualification.run_qualification " +
                                    " ".join(sys.argv[1:])) if getattr(args, "_cli", False) else None)
    # Representation filter.
    required = thresholds.get("analysis_input_required") if thresholds else None
    excluded_rep = []
    if required:
        for night in list(selected):
            keep = [f for f in selected[night] if representation(f) == required]
            excluded_rep += [{"frame": f["id"], "night": night, "representation": representation(f)}
                             for f in selected[night] if representation(f) != required]
            selected[night] = keep
            if not keep:
                del selected[night]
    engine = Engine(manifest, Path(args.manifest).resolve().parent, suite, cfg, log)
    night_results = {}
    for night in sorted(selected):
        log(f"evaluating {args.split} night {night} ({len(selected[night])} frames)")
        night_results[night] = engine.evaluate_night(night, selected[night])
    # s0 from development darks/bias only.
    if holdout:
        dev = frames_by_night(manifest, "development")
        calib = {}
        for night in sorted(dev):
            frames = [f for f in dev[night] if f["frame_type"] in ("dark", "bias")
                      and (not required or representation(f) == required)]
            if frames:
                calib[night] = engine.calibrate_night(night, frames)
        s0 = s0_from_calibration(calib, suite.method_ids)
    else:
        s0 = s0_from_calibration(night_results, suite.method_ids)
    night_metrics = {n: r["methods"] for n, r in night_results.items()}
    criteria = (TH.evaluate(thresholds, night_metrics, s0, roles, "hold-out" if holdout else "development")
                if thresholds else None)
    observations = TH.observations(thresholds, night_metrics, roles) if thresholds else None
    comparisons = TH.paired_comparisons(night_metrics, suite.method_ids, roles["legacy_background"],
                                        roles["legacy_gate"])
    result = {
        "schema": RESULTS_SCHEMA, "qualification_version": QUALIFICATION_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "split": args.split, "interpretation": SECTION[args.split],
        "status": ("holdout_evaluated" if holdout else
                   "development_observations_with_informational_criteria" if criteria else "development_observations"),
        "repeat_holdout": bool(started and started["repeat"]),
        "manifest": {"path": str(args.manifest), "sha256": manifest_sha, "schema": manifest["schema"],
                     "upgraded_from": manifest.get("upgraded_from"), "split_rule_problems": split_problems,
                     "nights": {n: len(fs) for n, fs in sorted(nights_all.items())},
                     "evaluated_nights": sorted(night_results),
                     "excluded_by_representation": excluded_rep, "excluded_at_build": len(manifest.get("excluded", []))},
        "thresholds": None if not thresholds else {"path": thresholds["_path"], "sha256": thresholds["_sha256"],
                                                   "predeclared_utc": thresholds["predeclared_utc"]},
        "roles": roles, "methods": suite.availability(), "method_config_sha256": config_hashes,
        "engine_config": engine_cfg, "s0": s0, "criteria": criteria, "observations": observations,
        "paired_comparisons": comparisons,
        "nights": night_results, "ledger_entry": started,
        "rust_bridge_log": getattr(suite.rust_runner, "log", None) if suite.rust_runner else None,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
        "limitations": [
            "no pixel truth on real frames: every claim is a consistency/margin claim, not an accuracy claim",
            "frames within a night are correlated; the night is the independent unit",
            "s0 is calibrated on darks/bias whose CFA-phase noise mix differs from a coloured sky",
            "negated-residual detections use one common reference detector (SEP extract, daemon default config)",
        ],
    }
    path = out_dir / "results.json"
    path.write_text(json.dumps(result, indent=1, default=_json_default, sort_keys=True))
    with open(out_dir / "frames.jsonl", "w") as fh:
        for row in engine.frame_rows:
            fh.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")
    (out_dir / "summary.md").write_text(render_summary(result))
    if started:
        summary = {cid: c["status"] for cid, c in (criteria or {}).items()}
        result["ledger_completed"] = HO.complete(args.ledger, started, sha256_file(path), summary)
    return result


def _fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def render_summary(r) -> str:
    lines = [f"# BGE real-frame qualification: {r['split']}", "", r["interpretation"], "",
             f"Status: **{r['status']}**" + (" (REPEAT hold-out run: not one-shot)" if r.get("repeat_holdout") else ""),
             f"Manifest sha256 `{r['manifest']['sha256'][:16]}`; nights evaluated: "
             f"{', '.join(r['manifest']['evaluated_nights']) or 'none'}", "", "## Methods", "",
             "| method | availability | s0 |", "| --- | --- | --- |"]
    for m, a in r["methods"].items():
        s0 = r["s0"].get(m, {})
        lines.append(f"| `{m}` | {a} | {_fmt(s0.get('value'))}{'' if s0.get('calibrated') else ' (uncalibrated)'} |")
    if r["criteria"]:
        lines += ["", "## Criteria", "", "| id | criterion | status | nights evaluable | statistical power |",
                  "| --- | --- | --- | --- | --- |"]
        for cid, c in r["criteria"].items():
            lines.append(f"| {cid} | {c['title']} | **{c['status']}** | {c['n_nights_evaluable']}/{c['n_nights']} | "
                         f"{c['statistical_power']} |")
        lines += ["", "Per-night values and reasons are in results.json (`criteria.<id>.per_night`)."]
    for oid, o in (r.get("observations") or {}).items():
        lines += ["", f"Observation {oid} ({o['note']}): {o['title']}", ""]
        for role, v in o["values"].items():
            vals = ", ".join(f"{n}: {_fmt(x)}" for n, x in v["per_night"].items())
            lines.append(f"- {role} (`{v['method']}`): {vals or '-'}")
    lines += ["", "## Per-night observations (subject roles)", ""]
    for night, res in r["nights"].items():
        lines.append(f"### {night}")
        lines.append(f"inputs: {json.dumps(res['inputs']['frame_types'])}")
        lines += ["", "| method | lights | fail rate | pull |mean| | pull rstd | P(|z|>5) | neg /Mpx | HFR CV | sweeps fit |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for m, x in res["methods"].items():
            lines.append(f"| `{m}` | {x['n_lights']} | {_fmt(x['status_failure_rate'])} | "
                         f"{_fmt(x['pull']['median_abs_clipped_mean'])} | {_fmt(x['pull']['median_robust_std'])} | "
                         f"{_fmt(x['pull']['median_frac_abs_gt5'])} | {_fmt(x['negated']['rate_per_mpx'])} | "
                         f"{_fmt(x['repeatability']['median_hfr_cv'])} | {x['focus']['n_fits']}/{x['focus']['n_sweeps']} |")
        lines.append("")
    lines += ["## Limitations", ""] + [f"- {x}" for x in r["limitations"]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)
    args._cli = argv is None
    try:
        result = run(args)
    except (HO.HoldoutRefusal, TH.ThresholdsError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(result["status"])
    if result["criteria"]:
        for cid, c in result["criteria"].items():
            print(f"{cid:>4} {c['status']:<14} {c['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
