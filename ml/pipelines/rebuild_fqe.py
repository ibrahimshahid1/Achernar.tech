"""
rebuild_fqe — ONE command from training to a publishable shadow-mode FQE.

Resolves the audit's P1 "three incompatible FQE contracts": after this runs,
`models/focus_quality/<version>/` contains a two-head ONNX (`mu_log_hfr`,
`log_var`), a metadata.json that describes THAT graph (not the old
sigmoid/MSE reference), a fitted calibration.json, and a norm_params.json
documenting the per-crop normalization contract. The daemon picks the highest
runtime-eligible semver automatically (`fqe_runner::resolve_latest_model`), so
a real-data 1.1.0 can supersede the non-runnable 1.0.0 reference without
letting a synthetic integration artifact silently enter the live path.

Steps:
  1. train      train.train_focus_quality.train (heteroscedastic NLL + RankNet,
                grouped-by-run splits)
  2. calibrate  train.calibrate_uncertainty (quantile scaling: fit on val, report on
                test — fixes the measured 0.54 coverage at nominal 95%)
  3. baselines  eval.baselines.freeze_baselines (the promotion-gate numbers)
  4. export     export.export_quality_onnx.export (FP32 parity-gated; optional
                INT8 attempt is non-fatal)
  5. publish    metadata.json + calibration.json + norm_params.json
  6. parity     PARITY_PY on the same deterministic ramp crop that
                fqe_runner.rs's `runs_real_onnx_model_when_available` prints as
                PARITY_RUST — diff the two lines to prove Rust==Python.

Eligibility follows the DATA ROLE, never "an NPZ was supplied":

  * ``local_verified_run`` (``--dataset``: a local dataset whose every row passes
    ``build_focus_quality_dataset.load_qualified_dataset``): the genuine-run path,
    unchanged — test evaluated, calibration fitted on val and reported on test,
    frozen baselines on test. ``real_multi_run_data`` and ``runtime_eligible``
    are true only when the dataset has multi-row runs.
  * generated synthetic runs (``--synthetic``): role ``fixture_only``; the
    integration path is unchanged (its test split is regenerated in-process from
    the seed) and stays explicit-path-only (``runtime_eligible`` false, always).
  * public contract sources (``--source-id`` trains through
    ``data_pipeline.fqe_sources``; ``--from-run`` exports a frozen
    ``train.fqe_dev_experiment`` candidate without retraining): pointwise
    batching, ``lambda_rank`` 0, the sealed test partition is never evaluated,
    calibration is development-only, baselines are not run here. Never runtime
    eligible — except through the explicit prototype override
    ``--from-run RUN --prototype-runtime-eligible --prototype-note ...
    --prototype-gate-evidence EVIDENCE.json`` (owner authorization F-006). It is
    refused for ``--source-id`` (a fresh epoch/val-selected retrain), for runs
    trained under any contract other than the tracked
    ``fqe_source_contracts_v1.json`` or with different pinned file hashes, and
    unless the evidence (bound to the candidate's weights/config sha256) shows
    the F-006 gate passed on sim val; ONNX parity on the dev crops is checked
    here after export (a failure writes the bundle NOT runtime eligible and
    exits refused). It stamps a provenance block with the evidence sha256,
    keeps ``promotion_status`` shadow, writes an uncalibrated sigma (public
    calibration does not transfer to the rig), and never sets
    ``ready_for_promotion_review``.

Public archive NPZs given by ``--dataset`` and unqualified local datasets are
refused with a message naming the right route.

This pipeline ALWAYS stamps promotion_status="shadow". Promotion is a human
decision after the model beats the frozen baselines on REAL data (spec
Section 7.10); nothing here can flip that switch.

Run:
    python -m pipelines.rebuild_fqe --synthetic --epochs 30 --version 1.1.0
    # then: cd achernar-daemon && FQE_ONNX_MODEL=../models/focus_quality/1.1.0/model.onnx \
    #       cargo test runs_real_onnx_model_when_available -- --nocapture
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROLE_LOCAL = "local_verified_run"
ROLE_FIXTURE = "fixture_only"
PUBLIC_ROLES = ("synthetic_reference", "archive_measurement_reference")
#: Data origins whose test partition this pipeline may evaluate: genuine local
#: runs, and the synthetic generator whose "test" is regenerated in-process.
TEST_EVALUATION_ORIGINS = ("local_dataset", "generated_synthetic_runs")
PROTOTYPE_DOMAIN = (
    "public archive development data (stellar_sim_v1 synthetic reference / hst_acs_corpus_v1 "
    "SEP measurement reference); not qualified for R50/Z114"
)
PROTOTYPE_AUTHORIZATION = (
    "owner authorization F-006 (FQE public-data run 20261111T0038, decisions.md, 2026-11-11): a gated "
    "shadow prototype in the production model folder; explicit opt-in, never inferred from the data"
)
ORIGIN_RUN = "fqe_dev_experiment_run"
#: Schema of the --prototype-gate-evidence JSON (see verify_prototype_gate).
PROTOTYPE_GATE_SCHEMA = "fqe-prototype-gate-evidence-v1"
#: F-006 / contract-v1 production_prototype_gate.
PROTOTYPE_GATE_CRITERIA = (
    "F-006: finite outputs; ONNX FP32 parity max |delta mu| <= 1e-4 on fixtures and dev crops; on the "
    "synthetic dev population (sim val) HFR MAE no worse than the normalized-input classical HFR at equal "
    "coverage (common valid rows), with candidate coverage >= classical coverage"
)
PROTOTYPE_CLASSICAL_COMPARATOR = "normalized_input__direct_hfr"
PROTOTYPE_PARITY_TOLERANCE = 1e-4


class RebuildRefusal(ValueError):
    """The requested rebuild contradicts the data's role."""


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset(args) -> tuple[dict, str]:
    if args.synthetic or not args.dataset:
        from data_pipeline.synthetic_psf import make_synthetic_dataset

        return (
            make_synthetic_dataset(n_runs=args.synthetic_runs, seed=args.seed),
            f"synthetic(n_runs={args.synthetic_runs}, seed={args.seed})",
        )
    npz = np.load(args.dataset, allow_pickle=False)
    return {k: npz[k] for k in npz.files}, args.dataset


def _multi_row_runs(run_id) -> bool:
    counts = np.unique(np.asarray(run_id), return_counts=True)[1]
    return bool(counts.size and int(counts.max()) >= 2)


def _contract(args):
    from data_pipeline import fqe_sources

    return fqe_sources.load_contract(getattr(args, "contract", None))


def resolve_data_role(args) -> dict:
    """Classify the rebuild input. Returns roles, origin, batching facts and
    whether the test partition may be evaluated. Raises :class:`RebuildRefusal`
    for inputs that must take a different route."""
    from_run = getattr(args, "from_run", None)
    source_ids = list(getattr(args, "source_id", None) or [])
    if sum(bool(x) for x in (from_run, source_ids, args.dataset, args.synthetic)) > 1:
        raise RebuildRefusal("choose exactly one of --synthetic, --dataset, --source-id, --from-run")

    if from_run:
        from train.fqe_dev_experiment import load_completed_run

        config, record, model_path = load_completed_run(from_run)
        roles = sorted(set((record.get("data_roles") or {}).values()))
        return {
            "origin": "fqe_dev_experiment_run",
            "roles": roles,
            "source_ids": sorted(record.get("data_roles") or {}),
            "batching": "pointwise",
            "multi_row_runs": False,
            "evaluate_test": False,
            "contract_sha256": config.get("contract_sha256"),
            "run_dir": str(Path(from_run).resolve()),
            "run_config_sha256": record["config_sha256"],
            "model_path": str(model_path),
            "model_sha256": record["final_model_sha256"],
            "run_record": record,
            "run_config": config,
        }

    if source_ids:
        contract = _contract(args)
        roles, batching = [], set()
        for source_id in source_ids:
            entry = contract.source(source_id)
            roles.append(entry["role"])
            batching.add(entry.get("batching"))
        if ROLE_LOCAL in roles:
            raise RebuildRefusal("local runs are rebuilt with --dataset (genuine-run path), not --source-id")
        return {
            "origin": "contract_sources",
            "roles": sorted(set(roles)),
            "source_ids": sorted(source_ids),
            "batching": "pointwise" if batching != {"grouped_run"} else "grouped_run",
            "multi_row_runs": False,
            "evaluate_test": False,
            "contract_sha256": contract.sha256,
        }

    if args.synthetic or not args.dataset:
        return {
            "origin": "generated_synthetic_runs",
            "roles": [ROLE_FIXTURE],
            "source_ids": [],
            "batching": "grouped_run",
            "multi_row_runs": True,
            "evaluate_test": True,
            "contract_sha256": None,
        }

    from data_pipeline import build_focus_quality_dataset as builder
    from train.train_focus_quality import (
        is_archive_singleton_dataset,
        pinned_public_source,
        read_run_key_semantics,
    )

    # Screen before reading any array: only the lazy run_key_semantics member
    # and the file hash, so a public NPZ's crops/targets/test rows are never loaded.
    semantics = read_run_key_semantics(args.dataset)
    if semantics is not None and is_archive_singleton_dataset({"run_key_semantics": semantics}):
        raise RebuildRefusal(
            f"{args.dataset} is a public archive/simulation NPZ (singleton samples, no focus runs). "
            "Use --source-id (contract loader) or --from-run (a frozen fqe_dev_experiment candidate)."
        )
    dataset_sha = _sha256(args.dataset)
    pinned = pinned_public_source(args.dataset, dataset_sha)
    if pinned is None and getattr(args, "contract", None):
        try:
            contract = _contract(args)
            for source_id, entry in contract.sources.items():
                if entry.get("role") != ROLE_LOCAL and dataset_sha in (entry.get("files") or {}).values():
                    pinned = source_id
        except Exception:  # an unreadable contract does not make a local dataset public
            pass
    if pinned is not None:
        raise RebuildRefusal(f"{args.dataset} is the pinned contract file of {pinned}; use --source-id {pinned}")
    try:
        _, report = builder.load_qualified_dataset(args.dataset)
    except builder.LegacyUnverifiedLabels as exc:
        raise RebuildRefusal(
            f"{args.dataset} is not a qualified local dataset ({exc}). Only rows qualified by "
            "build_focus_quality_dataset.load_qualified_dataset may produce a runtime-eligible model."
        ) from exc
    if report["rows_qualified"] != report["rows_total"]:
        raise RebuildRefusal(
            f"{args.dataset}: {report['rows_excluded']} of {report['rows_total']} rows are not qualified; "
            "rebuild from a dataset whose every row qualifies"
        )
    with np.load(args.dataset, allow_pickle=False) as npz:
        run_id = np.asarray(npz["run_id"])
    multi_row = _multi_row_runs(run_id)
    return {
        "origin": "local_dataset",
        "roles": [ROLE_LOCAL],
        "source_ids": [],
        "batching": "grouped_run" if multi_row else "pointwise",
        "multi_row_runs": multi_row,
        "evaluate_test": True,
        "contract_sha256": None,
        "dataset_sha256": dataset_sha,
        "qualification": report,
    }


def _check_prototype_request(role_info: dict, prototype_note: str | None) -> None:
    """The parts of the F-006 override that need no files: a note, public roles
    only, and a frozen ``--from-run`` candidate (never a fresh retrain)."""
    roles = list(role_info["roles"])
    if not (prototype_note and prototype_note.strip()):
        raise RebuildRefusal("--prototype-runtime-eligible requires a non-empty --prototype-note")
    if not roles or any(role not in PUBLIC_ROLES for role in roles):
        raise RebuildRefusal(
            f"the prototype override applies only to public contract roles {PUBLIC_ROLES}; got {roles} "
            "(synthetic/fixture artifacts stay explicit-path-only; local runs need no override)"
        )
    if role_info.get("origin") != ORIGIN_RUN:
        raise RebuildRefusal(
            "--prototype-runtime-eligible applies only with --from-run: a frozen train.fqe_dev_experiment "
            "candidate (fixed update budget, final weights, contract source-balanced sampling). --source-id "
            "retrains here with epoch-based best-by-val selection, which is not a frozen candidate."
        )


def _tracked_contract():
    """The tracked source contract (``ml/data_pipeline/fqe_source_contracts_v1.json``,
    the repository mirror of contract-v1). Tests substitute a fixture stand-in."""
    from data_pipeline import fqe_sources

    return fqe_sources.load_contract()


def _gate_number(value, what: str) -> float:
    if not _finite_number(value):
        raise RebuildRefusal(f"prototype gate evidence: {what} is {value!r}, not a finite number")
    return float(value)


def verify_prototype_gate(role_info: dict, evidence_path, *, contract=None) -> dict:
    """Check the F-006 evidence gate for a ``--from-run`` candidate before
    anything is written. Returns the verified gate block (``status: passed``;
    the dev-crop ONNX parity is added after export) or raises :class:`RebuildRefusal`.

    1. The run was trained under the tracked source contract (same canonical
       sha256), every training source has the tracked role, and its recorded
       pinned file sha256s equal the tracked pins (so fixture contracts and
       re-pinned data are refused).
    2. ``evidence_path`` (schema ``fqe-prototype-gate-evidence-v1``) binds to this
       candidate (``candidate_model_sha256``, ``run_config_sha256``) and the
       tracked contract, names a ``synthetic_reference`` source's ``val``
       population, states ``test_partition_used: false`` and
       ``candidate_outputs_finite: true``, and carries a
       ``paired_comparison`` (``eval.fqe_benchmark.paired_comparison``) of
       ``candidate_comparator`` against ``normalized_input__direct_hfr`` in
       which the candidate's MAE on the common valid rows is no worse and its
       full-population coverage is at least the classical coverage.
    """
    if role_info.get("origin") != ORIGIN_RUN:
        raise RebuildRefusal("the prototype gate applies only to --from-run candidates")
    if not evidence_path:
        raise RebuildRefusal(
            "--prototype-runtime-eligible requires --prototype-gate-evidence (an "
            f"{PROTOTYPE_GATE_SCHEMA} JSON stating the F-006 criteria for this candidate)"
        )
    from data_pipeline import fqe_sources

    tracked = contract if contract is not None else _tracked_contract()
    if role_info.get("contract_sha256") != tracked.sha256:
        raise RebuildRefusal(
            f"the candidate was trained under source contract {role_info.get('contract_sha256')}, not the "
            f"tracked contract {tracked.sha256} (fqe_source_contracts_v1.json); fixture or edited contracts "
            "cannot produce a runtime-eligible prototype"
        )
    identity = json.loads((Path(role_info["run_dir"]) / "dataset_identity.json").read_text())
    run_roles = (role_info.get("run_record") or {}).get("data_roles") or {}
    if not run_roles:
        raise RebuildRefusal("the candidate run records no training data roles")
    for source_id, role in sorted(run_roles.items()):
        try:
            entry = tracked.source(source_id)
        except fqe_sources.ContractViolation as exc:
            raise RebuildRefusal(f"training source {source_id!r} is not declared by the tracked contract") from exc
        if entry.get("role") != role or role not in PUBLIC_ROLES:
            raise RebuildRefusal(
                f"training source {source_id!r} has role {role!r}; the tracked contract says {entry.get('role')!r}"
            )
        stored = (identity.get("sources") or {}).get(source_id) or {}
        if stored.get("contract_sha256") != tracked.sha256 or stored.get("files") != entry.get("files"):
            raise RebuildRefusal(
                f"training source {source_id!r}: the run's recorded file sha256s / contract do not match the "
                "tracked contract's pins"
            )

    raw = Path(evidence_path).read_bytes()
    try:
        evidence = json.loads(raw)
    except ValueError as exc:
        raise RebuildRefusal(f"prototype gate evidence {evidence_path} is not valid JSON: {exc}") from exc
    if not isinstance(evidence, dict) or evidence.get("schema") != PROTOTYPE_GATE_SCHEMA:
        raise RebuildRefusal(f"prototype gate evidence must have schema {PROTOTYPE_GATE_SCHEMA!r}")
    for key, expected in (("candidate_model_sha256", role_info["model_sha256"]),
                          ("run_config_sha256", role_info["run_config_sha256"]),
                          ("contract_sha256", tracked.sha256)):
        if evidence.get(key) != expected:
            raise RebuildRefusal(f"prototype gate evidence {key} is {evidence.get(key)!r}, expected {expected}")
    if evidence.get("test_partition_used") is not False:
        raise RebuildRefusal("prototype gate evidence must state test_partition_used: false (the test is sealed)")
    if evidence.get("candidate_outputs_finite") is not True:
        raise RebuildRefusal("prototype gate evidence does not state candidate_outputs_finite: true")
    population = evidence.get("population") or {}
    population_source = population.get("source_id")
    if population.get("partition") != "val":
        raise RebuildRefusal(f"the F-006 population is a val partition, got {population.get('partition')!r}")
    try:
        population_role = tracked.source(population_source).get("role")
    except fqe_sources.ContractViolation as exc:
        raise RebuildRefusal(f"population source {population_source!r} is not in the tracked contract") from exc
    if population_role != "synthetic_reference":
        raise RebuildRefusal(
            f"the F-006 population is the synthetic dev population; {population_source!r} is {population_role!r}"
        )
    candidate_id = evidence.get("candidate_comparator")
    classical_id = evidence.get("classical_comparator")
    if classical_id != PROTOTYPE_CLASSICAL_COMPARATOR or not candidate_id or candidate_id == classical_id:
        raise RebuildRefusal(
            f"prototype gate evidence must compare a candidate_comparator against {PROTOTYPE_CLASSICAL_COMPARATOR!r}"
        )
    comparison = evidence.get("paired_comparison") or {}
    per = comparison.get("per_comparator") or {}
    if candidate_id not in per or classical_id not in per or \
            set(comparison.get("comparators") or []) != {candidate_id, classical_id}:
        raise RebuildRefusal("prototype gate evidence paired_comparison does not cover exactly the two comparators")
    common = comparison.get("common_valid")
    if not isinstance(common, int) or isinstance(common, bool) or common <= 0:
        raise RebuildRefusal(f"prototype gate evidence has no common valid rows ({common!r})")
    metrics = {}
    for cid in (candidate_id, classical_id):
        block = per[cid].get("common_valid_metrics") or {}
        if block.get("n") != common:
            raise RebuildRefusal(f"{cid}: common-row metrics are on {block.get('n')!r} rows, not {common}")
        metrics[cid] = {"hfr_mae_px": _gate_number(block.get("hfr_mae_px"), f"{cid} hfr_mae_px"),
                        "coverage": _gate_number(per[cid].get("coverage_full_population"), f"{cid} coverage")}
    candidate, classical = metrics[candidate_id], metrics[classical_id]
    failures = []
    if candidate["hfr_mae_px"] > classical["hfr_mae_px"]:
        failures.append(f"candidate MAE {candidate['hfr_mae_px']:.6g} px > classical {classical['hfr_mae_px']:.6g} px")
    if candidate["coverage"] < classical["coverage"]:
        failures.append(f"candidate coverage {candidate['coverage']:.6g} < classical {classical['coverage']:.6g}")
    if failures:
        raise RebuildRefusal("the F-006 evidence gate is not met: " + "; ".join(failures))
    return {
        "status": "passed",
        "criteria": PROTOTYPE_GATE_CRITERIA,
        "evidence_schema": PROTOTYPE_GATE_SCHEMA,
        "evidence_name": Path(evidence_path).name,
        "evidence_sha256": hashlib.sha256(raw).hexdigest(),
        "contract_sha256": tracked.sha256,
        "contract_file_sha256": tracked.file_sha256,
        "experiment_contract_sha256": tracked.experiment_contract_sha256,
        "population": {"source_id": population_source, "partition": "val"},
        "candidate_comparator": candidate_id,
        "classical_comparator": classical_id,
        "common_valid_rows": common,
        "common_membership_sha256": comparison.get("common_membership_sha256"),
        "candidate": candidate,
        "classical": classical,
        "test_partition_used": False,
    }


def check_dev_crop_parity(pt_path, onnx_path, gate: dict, *, data_root=None, contract=None) -> dict:
    """F-006 dev-crop parity: torch vs the exported ONNX on the gate
    population's val crops (development rows only; loaded through the tracked
    contract, which verifies every pinned hash)."""
    import onnxruntime as ort
    import torch

    from data_pipeline import fqe_sources
    from models.focus_quality_cnn import FocusQualityEstimator
    from train.calibrate_uncertainty import predict_mu_log_var

    tracked = contract if contract is not None else _tracked_contract()
    source_id = gate["population"]["source_id"]
    arrays = fqe_sources.load_source(source_id, partitions=("val",), contract=tracked, data_root=data_root)
    crops = np.asarray(arrays.crops, dtype=np.float32)
    model = FocusQualityEstimator(pretrained=False)
    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    t_mu, t_lv = predict_mu_log_var(model, crops, "cpu")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    o_mu, o_lv = [], []
    for start in range(0, len(crops), 256):
        mu, lv = sess.run(None, {"image": np.ascontiguousarray(crops[start : start + 256][:, None])})
        o_mu.append(np.ravel(mu))
        o_lv.append(np.ravel(lv))
    o_mu = np.concatenate(o_mu).astype(np.float64) if o_mu else np.zeros(0)
    o_lv = np.concatenate(o_lv).astype(np.float64) if o_lv else np.zeros(0)
    finite = bool(np.all(np.isfinite(t_mu)) and np.all(np.isfinite(t_lv))
                  and np.all(np.isfinite(o_mu)) and np.all(np.isfinite(o_lv)))
    d_mu = float(np.max(np.abs(t_mu - o_mu))) if len(crops) else float("nan")
    d_lv = float(np.max(np.abs(t_lv - o_lv))) if len(crops) else float("nan")
    return {
        "source_id": source_id,
        "partition": "val",
        "rows": int(len(crops)),
        "max_abs_delta_mu": d_mu,
        "max_abs_delta_log_var": d_lv,
        "tolerance": PROTOTYPE_PARITY_TOLERANCE,
        "finite": finite,
        "passed": bool(len(crops) and finite and d_mu <= PROTOTYPE_PARITY_TOLERANCE),
    }


def derive_eligibility(role_info: dict, *, prototype_runtime_eligible: bool = False,
                       prototype_note: str | None = None, prototype_gate: dict | None = None) -> dict:
    """``real_multi_run_data`` / ``runtime_eligible`` from the data role.

    True only for ``local_verified_run`` with multi-row runs. Public roles become
    runtime eligible only through the explicit prototype override on a frozen
    ``--from-run`` candidate whose F-006 evidence gate passed
    (:func:`verify_prototype_gate`); it returns a provenance block.
    Fixture/synthetic data never do."""
    roles = list(role_info["roles"])
    real = bool(roles == [ROLE_LOCAL] and role_info.get("multi_row_runs"))
    prototype = None
    if prototype_runtime_eligible:
        _check_prototype_request(role_info, prototype_note)
        if not prototype_gate or prototype_gate.get("status") != "passed":
            raise RebuildRefusal("the prototype override requires a verified F-006 evidence gate")
        prototype = {
            "runtime_eligible_override": True,
            "authorization": PROTOTYPE_AUTHORIZATION,
            "note": prototype_note.strip(),
            "domain": PROTOTYPE_DOMAIN,
            "data_roles": roles,
            "source_ids": list(role_info.get("source_ids") or []),
            "contract_sha256": role_info.get("contract_sha256"),
            "qualified_for": [],
            "not_qualified_for": ["R50", "Z114", "autofocus control authority", "promotion review"],
            "real_multi_run_data": False,
            "ready_for_promotion_review": False,
            "gate": dict(prototype_gate),
        }
        for key in ("run_config_sha256", "model_sha256"):
            if role_info.get(key):
                prototype[key] = role_info[key]
        if role_info.get("run_dir"):
            prototype["run_name"] = Path(role_info["run_dir"]).name  # no machine path in runtime metadata
    return {
        "real_multi_run_data": real,
        "runtime_eligible": bool(real or prototype is not None),
        "prototype": prototype,
    }


def resolve_lambda_rank(requested, role_info: dict) -> float:
    """Default 0.2 for genuine grouped runs, 0 for pointwise sources. An explicit
    nonzero value on pointwise data is passed through and refused by train()."""
    if requested is None:
        return 0.2 if role_info["batching"] == "grouped_run" else 0.0
    return float(requested)


def _bundle_calibration(per_source: dict, role_info: dict) -> dict:
    """calibration.json for a public-role bundle: development results are kept
    for provenance, but the runtime scales are null (uncalibrated) because a
    scale fitted on public sources does not transfer to the rig's crops."""
    return {
        "status": "uncalibrated_for_runtime",
        "calibration_method": "quantile_interval_scaling",
        "sigma_scale": None,
        "frame_sigma_scale": None,
        "reason": (
            "public-archive development calibration (fit on contract calibration groups, assessed on val) "
            "is not evidence for R50/Z114 crops; the runtime sigma stays uncalibrated"
        ),
        "development_only": True,
        "test_reported": False,
        "data_roles": role_info["roles"],
        "development_calibration": per_source,
    }


def _baselines_not_run(reason: str) -> dict:
    return {"status": "not_run", "reason": reason, "constant": {"hfr_mae_px": None},
            "tenengrad": {"within_run_rho": None}, "direct_hfr": {"within_run_rho": None}}


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and bool(np.isfinite(value))


def main(args) -> dict:
    role_info = resolve_data_role(args)
    prototype_requested = bool(getattr(args, "prototype_runtime_eligible", False))
    prototype_gate = None
    if prototype_requested:
        # Every F-006 check that needs no export runs before anything is written.
        _check_prototype_request(role_info, getattr(args, "prototype_note", None))
        prototype_gate = verify_prototype_gate(role_info, getattr(args, "prototype_gate_evidence", None))
    eligibility = derive_eligibility(
        role_info,
        prototype_runtime_eligible=prototype_requested,
        prototype_note=getattr(args, "prototype_note", None),
        prototype_gate=prototype_gate,
    )
    evaluate_test = role_info["origin"] in TEST_EVALUATION_ORIGINS
    ml_dir = Path(__file__).resolve().parents[1]
    repo_root = ml_dir.parent
    models_root = Path(args.models_root) if args.models_root else repo_root / "models"
    out_dir = models_root / "focus_quality" / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    lambda_rank = resolve_lambda_rank(args.lambda_rank, role_info)
    device = getattr(args, "device", None)
    pt_path = str(out_dir / "focus_quality_fqe.pt")

    # ---- 1. Train (or take the frozen development candidate) ----------------
    if role_info["origin"] == "fqe_dev_experiment_run":
        shutil.copyfile(role_info["model_path"], pt_path)
        if _sha256(pt_path) != role_info["model_sha256"]:
            raise RebuildRefusal("copied candidate weights do not match the run's final_model_sha256")
        run_config = role_info["run_config"]
        lambda_rank = float(run_config.get("lambda_rank", 0.0))
        train_result = {
            "best_val": {},
            "test": {},
            "test_status": "sealed (not evaluated)",
            "batching": "pointwise",
            "lambda_rank": lambda_rank,
            "device": role_info["run_record"].get("device"),
            "data_roles": role_info["roles"],
            "candidate": {"run_dir": role_info["run_dir"], "config_sha256": role_info["run_config_sha256"],
                          "model_sha256": role_info["model_sha256"],
                          "updates": run_config.get("updates"), "arm": run_config.get("arm"),
                          "seed": run_config.get("seed"), "selection": "final weights (no validation)"},
        }
    else:
        from train.train_focus_quality import train

        train_config = dict(
            synthetic=args.synthetic,
            dataset=args.dataset,
            synthetic_runs=args.synthetic_runs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_rank=lambda_rank,
            variance_warmup_epochs=args.variance_warmup_epochs,
            pretrained=args.pretrained,
            seed=args.seed,
            split_registry=args.split_registry,
            out=pt_path,
            device=device,
            data_role=role_info["roles"][0] if len(role_info["roles"]) == 1 else None,
        )
        if role_info["origin"] == "contract_sources":
            train_config.update(source_ids=role_info["source_ids"], data_root=getattr(args, "data_root", None),
                                contract_path=getattr(args, "contract", None), dataset=None, synthetic=False)
        train_result = train(train_config, evaluate_test=evaluate_test)
    test_metrics = train_result.get("test", {}) or {}

    # ---- 2. Calibrate ---------------------------------------------------------
    if evaluate_test:
        from train.calibrate_uncertainty import calibrate

        calib = calibrate(
            pt_path,
            args.dataset,
            str(out_dir / "calibration.json"),
            args.synthetic,
            args.seed,
            args.split_registry,
            args.synthetic_runs,
        )
    elif role_info["origin"] == "contract_sources":
        from train.calibrate_uncertainty import calibrate_sources

        dev = calibrate_sources(
            pt_path,
            role_info["source_ids"],
            None,
            contract=_contract(args),
            data_root=getattr(args, "data_root", None),
            device=device,
        )
        calib = _bundle_calibration(dev["sources"], role_info)
    else:
        run_calibration = Path(role_info["run_dir"]) / "calibration" / "calibration.json"
        dev = json.loads(run_calibration.read_text()) if run_calibration.exists() else None
        if dev is not None and dev.get("model_sha256") != role_info["model_sha256"]:
            raise RebuildRefusal("the run's calibration.json belongs to different weights")
        calib = _bundle_calibration(
            dev["sources"] if dev else {"status": "not_run", "reason": "fqe_dev_experiment calibrate not run"},
            role_info,
        )

    # ---- 3. Frozen baselines (the gate the FQE must beat) -------------------
    if evaluate_test:
        from eval.baselines import freeze_baselines

        dataset, dataset_name = _load_dataset(args)
        from data_pipeline.split_registry import create_group_splits, splits_for_dataset

        if args.dataset and not args.synthetic:
            baseline_splits = splits_for_dataset(
                args.dataset,
                dataset,
                seed=args.seed,
                registry_path=args.split_registry,
            )
        else:
            baseline_splits = create_group_splits(
                dataset.get("split_group", dataset["run_id"]), seed=args.seed
            )
        # Explicit final evaluation of genuine local runs (or regenerated
        # synthetic integration data); eval.baselines has no default partition.
        baselines = freeze_baselines(
            dataset, split_seed=args.seed, splits=baseline_splits, partition="test"
        )
    else:
        dataset_name = (
            f"fqe_dev_experiment run {role_info.get('run_dir')}"
            if role_info["origin"] == "fqe_dev_experiment_run"
            else f"contract sources {role_info['source_ids']}"
        )
        baselines = _baselines_not_run(
            "public contract sources: eval.baselines evaluates the test partition, which is sealed; the "
            "development comparison comes from the classical benchmark package"
        )

    # ---- 4. Export (FP32 parity-gated) ---------------------------------------
    from export.export_quality_onnx import export as export_onnx

    onnx_path = str(out_dir / "model.onnx")
    export_result = export_onnx(pt_path, onnx_path, crop_size=64, quantize=args.quantize)

    # ---- 4b. F-006 dev-crop parity (prototype override only) -----------------
    prototype_refusal = None
    if eligibility["prototype"] is not None:
        run_root = getattr(args, "data_root", None) or role_info["run_config"].get("data_root")
        parity = check_dev_crop_parity(pt_path, onnx_path, prototype_gate, data_root=run_root)
        eligibility["prototype"]["gate"]["onnx_parity_dev_crops"] = parity
        eligibility["prototype"]["gate"]["onnx_parity_fixtures"] = {
            "max_abs_delta_mu": export_result["mu_max_abs_err"],
            "max_abs_delta_log_var": export_result["log_var_max_abs_err"],
            "tolerance": PROTOTYPE_PARITY_TOLERANCE, "passed": bool(export_result["parity_ok"])}
        if not parity["passed"]:
            # Demote: the bundle is written as an ordinary explicit-path-only
            # public artifact, the refusal is recorded, and the command fails.
            prototype_refusal = {"status": "refused", "reason": "F-006 dev-crop ONNX parity/finite check failed",
                                 "gate": eligibility["prototype"]["gate"]}
            eligibility = {"real_multi_run_data": eligibility["real_multi_run_data"],
                           "runtime_eligible": bool(eligibility["real_multi_run_data"]), "prototype": None}

    # ---- 5. Publish honest metadata ------------------------------------------
    from export.write_model_metadata import write_metadata

    mae = test_metrics.get("hfr_mae_px")
    rho = test_metrics.get("within_run_rank_rho")
    tenengrad_rho = baselines["tenengrad"]["within_run_rho"]
    direct_rho = baselines["direct_hfr"]["within_run_rho"]
    finite_baseline_rhos = [
        value for value in (tenengrad_rho, direct_rho) if _finite_number(value)
    ]
    rank_floor = max(finite_baseline_rhos) - 0.05 if finite_baseline_rhos else None
    calibrated_coverage = calib.get("coverage_95_test_after")
    frame_calibrated_coverage = calib.get("frame_coverage_95_test_after")
    constant_mae = baselines["constant"]["hfr_mae_px"]
    gate = {
        "policy": (
            "SHADOW ONLY. Classical curve-of-growth HFR remains the control "
            "authority. Promote by hand only after beating the frozen "
            "baselines on real multi-rig data (spec Section 7.10)."
        ),
        "beats_constant_mae": bool(
            mae is not None and constant_mae is not None and mae < constant_mae
        ),
        "real_multi_run_data": eligibility["real_multi_run_data"],
        "meets_classical_rank_floor": bool(
            rank_floor is not None
            and _finite_number(rho)
            and rho >= rank_floor
        ),
        "classical_rank_floor": rank_floor,
        "coverage_95_in_tolerance": bool(
            calibrated_coverage is not None and 0.90 <= calibrated_coverage <= 0.98
        ),
        "frame_coverage_95_in_tolerance": bool(
            frame_calibrated_coverage is not None
            and 0.90 <= frame_calibrated_coverage <= 0.98
        ),
        "test_hfr_mae_px": mae,
        "constant_baseline_mae_px": constant_mae,
        "test_within_run_rho": rho,
        "tenengrad_within_run_rho": tenengrad_rho,
    }
    gate["ready_for_promotion_review"] = bool(
        gate["real_multi_run_data"]
        and gate["beats_constant_mae"]
        and gate["meets_classical_rank_floor"]
        and gate["coverage_95_in_tolerance"]
        and gate["frame_coverage_95_in_tolerance"]
    )
    if eligibility["prototype"] is not None:
        gate["ready_for_promotion_review"] = False  # a prototype is never promotion evidence
    data_provenance = {
        "origin": role_info["origin"],
        "data_roles": role_info["roles"],
        "source_ids": role_info.get("source_ids") or [],
        "contract_sha256": role_info.get("contract_sha256"),
        "dataset_sha256": role_info.get("dataset_sha256"),
        "qualification": role_info.get("qualification"),
        "batching": train_result.get("batching", role_info["batching"]),
        "lambda_rank": lambda_rank,
        "test_evaluated": bool(evaluate_test and test_metrics),
        "test_policy": (
            "evaluated (genuine local runs / regenerated synthetic integration data)"
            if evaluate_test
            else "sealed: never evaluated by this pipeline"
        ),
        "multi_row_runs": role_info["multi_row_runs"],
    }
    training_info = {
        "status": "trained",
        "promotion_status": "shadow",
        # Synthetic-only and public-archive artifacts remain available by
        # explicit path, but the daemon's semver resolver will not load them
        # automatically as live shadow evidence (unless the recorded F-006
        # prototype override was given).
        "runtime_eligible": eligibility["runtime_eligible"],
        "promotion_gate": gate,
        "data_provenance": data_provenance,
        "architecture": "MobileNetV3-Small (grayscale stem) + heteroscedastic head",
        "output_semantics": (
            "mu_log_hfr = E[ln HFR_px]; sigma_log_hfr = exp(0.5*log_var) * "
            "sigma_scale per star; robust pooled frame sigma is then multiplied "
            "by frame_sigma_scale (calibration.json)"
        ),
        "loss": f"heteroscedastic_nll + {lambda_rank} * ranknet_within_run",
        "optimizer": "AdamW, cosine annealing",
        "dataset": dataset_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "device": train_result.get("device"),
        "best_val": train_result.get("best_val", {}),
        "test": test_metrics,
        "baselines": baselines,
        "calibration": {
            "sigma_scale": calib.get("sigma_scale"),
            "frame_sigma_scale": calib.get("frame_sigma_scale"),
            "coverage_95_test_before": calib.get("coverage_95_test_before"),
            "coverage_95_test_after": calib.get("coverage_95_test_after"),
            "coverage_test_after": calib.get("coverage_test_after"),
            "nll_test_after": calib.get("nll_test_after"),
            "frame_coverage_95_test_after": calib.get(
                "frame_coverage_95_test_after"
            ),
            "frame_coverage_test_after": calib.get("frame_coverage_test_after"),
            "frame_nll_test_after": calib.get("frame_nll_test_after"),
        },
        "preprocessing": "per-crop robust_normalize p1/p99 -> [0,1] (norm_params.json)",
        "export_parity": {
            "mu_max_abs_err": export_result["mu_max_abs_err"],
            "log_var_max_abs_err": export_result["log_var_max_abs_err"],
        },
    }
    if "candidate" in train_result:
        training_info["candidate"] = train_result["candidate"]
        training_info["epochs"] = None
        training_info["optimizer"] = "AdamW (fqe_dev_experiment fixed-update recipe; see candidate config)"
    if eligibility["prototype"] is not None:
        training_info["prototype"] = eligibility["prototype"]
        training_info["domain"] = PROTOTYPE_DOMAIN
    if prototype_refusal is not None:
        training_info["prototype_refused"] = prototype_refusal
    metadata = write_metadata(
        onnx_path,
        "focus_quality",
        args.version,
        "focus_quality",
        framework="onnx",
        input_format={"image": ["batch", 1, 64, 64]},
        output_format={"mu_log_hfr": ["batch"], "log_var": ["batch"]},
        training_info=training_info,
    )

    # Bind calibration to the exact ONNX graph consumed at the edge. The Rust
    # runner refuses to call a sigma calibrated if this hash and metadata disagree.
    calib["model_sha256"] = metadata["sha256"]
    calib["model_filename"] = Path(onnx_path).name
    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=2, default=str))

    norm_params = {
        "scheme": "per_crop_percentile",
        "p_lo": 1.0,
        "p_hi": 99.0,
        "scale_to": [0.0, 1.0],
        "contract": (
            "clip((x - p1) / (p99 - p1), 0, 1), percentiles computed PER CROP "
            "over finite pixels with NumPy linear interpolation "
            "(ml/preprocess/normalize.py::robust_normalize). Edge parity "
            "implementation: achernar-daemon/src/vision/preprocess.rs. There "
            "are no fixed per-rig parameters while training normalizes per "
            "crop — keep the two paths in lockstep if that ever changes."
        ),
        "spec_section": "7.6.1",
    }
    (out_dir / "norm_params.json").write_text(json.dumps(norm_params, indent=2))
    print(f"wrote {out_dir / 'norm_params.json'}")

    # ---- 6. Cross-language parity line ---------------------------------------
    import onnxruntime as ort

    ramp = (np.arange(64 * 64, dtype=np.float32) / (64 * 64)).reshape(1, 1, 64, 64)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    mu, lv = sess.run(None, {"image": ramp})
    mu = float(np.ravel(mu)[0])
    lv = float(np.clip(np.ravel(lv)[0], -8.0, 8.0))
    sigma_scale = calib.get("sigma_scale") or 1.0
    sigma_cal = float(np.exp(0.5 * lv) * sigma_scale)
    # Same fields/format as fqe_runner.rs test's PARITY_RUST line (the runner
    # applies sigma_scale from calibration.json, so calibrated sigma matches).
    print(
        f"PARITY_PY mu_log_hfr={mu:.6f} hfr_px={np.exp(mu):.6f} sigma_log_hfr={sigma_cal:.6f}"
    )

    verdict = "PASSES" if gate["beats_constant_mae"] else "DOES NOT PASS"
    constant_text = f"{constant_mae:.3f}" if _finite_number(constant_mae) else "n/a"
    tenengrad_text = f"{tenengrad_rho:+.3f}" if _finite_number(tenengrad_rho) else "n/a"
    print(
        f"\n=== FQE {args.version}: published to {out_dir} ===\n"
        f"    data roles {role_info['roles']} ({role_info['origin']}); runtime_eligible="
        f"{eligibility['runtime_eligible']}"
        f"{' (PROTOTYPE OVERRIDE F-006)' if eligibility['prototype'] else ''}\n"
        f"    test MAE {mae if mae is not None else float('nan')} px vs constant "
        f"{constant_text} px -> {verdict} the sanity floor\n"
        f"    rank rho {rho} vs tenengrad {tenengrad_text}\n"
        f"    coverage95 (test) {calib.get('coverage_95_test_before')} -> "
        f"{calib.get('coverage_95_test_after')} after sigma_scale={calib.get('sigma_scale')}\n"
        f"    frame coverage95 (test) {calib.get('frame_coverage_95_test_after')} "
        f"after frame_sigma_scale={calib.get('frame_sigma_scale')}\n"
        f"    promotion_status = SHADOW (daemon publishes fqe_* fields alongside "
        f"classical; classical stays authority)"
    )
    if prototype_refusal is not None:
        raise RebuildRefusal(
            f"F-006 dev-crop ONNX parity failed ({prototype_refusal['gate']['onnx_parity_dev_crops']}); "
            f"{out_dir} was written NOT runtime eligible"
        )
    return {"out_dir": str(out_dir), "gate": gate, "calibration": calib, "eligibility": eligibility,
            "data_provenance": data_provenance}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="1.1.0")
    parser.add_argument("--models-root", default=None, help="default: <repo>/models")
    parser.add_argument("--dataset", help="qualified local .npz dataset; omit with --synthetic")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-runs", type=int, default=48)
    parser.add_argument(
        "--source-id",
        action="append",
        default=None,
        help="public contract source (data_pipeline.fqe_sources; repeatable); sealed test never evaluated",
    )
    parser.add_argument(
        "--from-run",
        default=None,
        help="export a completed train.fqe_dev_experiment run's final weights (no retraining)",
    )
    parser.add_argument("--contract", default=None, help="source contract JSON (default: tracked v1)")
    parser.add_argument("--data-root", default=None, help="public data root (default: ml/data/public)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lambda-rank",
        type=float,
        default=None,
        help="default: 0.2 for genuine multi-row runs, 0 for pointwise sources",
    )
    parser.add_argument("--variance-warmup-epochs", type=int, default=None)
    parser.add_argument("--split-registry")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--quantize", action="store_true", help="also attempt INT8 (non-fatal)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu | mps | cuda | auto (default: cuda if available, else cpu)")
    parser.add_argument(
        "--prototype-runtime-eligible",
        action="store_true",
        help="F-006 owner-authorized shadow prototype from PUBLIC roles; requires --prototype-note",
    )
    parser.add_argument("--prototype-note", default=None, help="who authorized the prototype and on what evidence")
    parser.add_argument(
        "--prototype-gate-evidence",
        default=None,
        help=f"{PROTOTYPE_GATE_SCHEMA} JSON stating the F-006 criteria for the --from-run candidate "
        "(required with --prototype-runtime-eligible; its sha256 is recorded)",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.prototype_runtime_eligible and not (args.prototype_note and args.prototype_note.strip()):
        parser.error("--prototype-runtime-eligible requires --prototype-note")
    if args.prototype_runtime_eligible and not args.from_run:
        parser.error("--prototype-runtime-eligible applies only with --from-run")
    if args.prototype_runtime_eligible and not args.prototype_gate_evidence:
        parser.error("--prototype-runtime-eligible requires --prototype-gate-evidence")
    try:
        main(args)
    except RebuildRefusal as exc:
        parser.exit(2, f"refused: {exc}\n")
