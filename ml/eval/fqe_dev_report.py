"""
fqe_dev_report — development comparison of the FQE trials (contract-v1 M4), the
F-010 candidate selection, ONNX parity and the F-006 prototype gate evidence.

Development data only. Every input is a file another step already wrote:
``train.fqe_dev_experiment`` trial directories (fit + predict + calibrate) and
the frozen classical benchmark (``eval.fqe_benchmark`` M2 run: ``config.json`` +
``rows.jsonl``). The ``parity`` subcommand loads ``val`` crops through
``data_pipeline.fqe_sources`` (pinned hashes verified; the sealed test partition
is never requested). Nothing here loads, predicts or scores test rows: any
``test`` partition in a predictions manifest, a predictions file or a classical
row is refused.

Subcommands (cwd ``ml/``)::

    python -B -m eval.fqe_dev_report report --trials RUN_DIR... --classical M2_DIR --out OUT_DIR \\
        [--matrix-contract contract-v1.json] [--matrix-configs ARM_CONFIG_DIR]
    python -B -m eval.fqe_dev_report select --report OUT_DIR/report.json [--out selection.json]
    python -B -m eval.fqe_dev_report parity --trial RUN_DIR --onnx model.onnx [--out parity.json]
    python -B -m eval.fqe_dev_report evidence --selection selection.json --parity parity.json \\
        --report OUT_DIR/report.json --out gate_evidence.json

``report``
    Per trial and per evaluation source (``val`` rows only): HFR MAE, log-HFR
    MAE, signed bias, median relative error and coverage on the full declared
    population (``fqe_benchmark.summarize_rows``); the ``fqe_benchmark`` strata;
    a group-level summary (sim families, HST exposures) instead of crop-level
    confidence intervals; paired comparisons on the common valid population
    against every classical comparator in the M2 rows, with both methods'
    full-population coverage (``fqe_benchmark.paired_comparison``); raw
    predictive uncertainty on val (``calibrate_uncertainty.NLL_CONVENTION``,
    68%/95% interval coverage) and, for the synthetic source only, the
    calibrated values of the trial's ``calibrate`` output (fit on the
    calibration partition, assessed on val). Per arm, the three seeds are kept
    individually with a descriptive min/median/max. Synthetic latent-truth
    results are reported separately from HST measurement-reference agreement.
    Refuses trials whose config/data identity differs from the frozen source
    contract (canonical contract sha256 and pinned file hashes) or whose
    optimizer budget differs from the matrix budget; with ``--matrix-configs``
    each trial must also have been fit from exactly its matrix arm config file.
``select``
    Decision F-010, exactly: an arm is eligible when all of its declared seeds
    completed with finite training and finite predictions on both development
    populations; a seed's score is the mean of its log-HFR MAE on the synthetic
    val population and on the HST val population; the arm score is the mean
    over seeds and the lowest wins; the candidate is the median-score seed of
    that arm. Writes the full ranking.
``parity``
    Max |delta mu| and |delta log_var| between the torch final weights (CPU)
    and onnxruntime on deterministic fixture inputs and on the val crops of
    every evaluation source; passes when every output is finite and both maxima
    are <= 1e-4.
``evidence``
    The ``fqe-prototype-gate-evidence-v1`` JSON read by
    ``pipelines.rebuild_fqe.verify_prototype_gate``, with the F-006 gate
    computed from the trial's predictions and the M2 rows (re-derived and
    cross-checked against the report): finite outputs; ONNX parity <= 1e-4;
    on the synthetic val population the candidate's MAE on the rows both
    methods measured is <= the normalized-input classical HFR's and its
    full-population coverage is >= the classical coverage. The Rust legacy
    (g1) and constant comparisons are attached as informational, non-gating
    context. The selection must be exactly the F-010 outcome of the report
    (recomputed here); a hand-edited candidate or ranking is refused. The
    record is always written, but it carries the accepted schema only when
    every check passed: a failed gate is written as
    ``fqe-prototype-gate-result-failed-v1``, which the installer refuses, and
    the exit status is 1.

Exit codes: 0 ok, 1 gate/parity failed, no eligible arm (select) or unexpected
error, 2 refused, 3 data unavailable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from data_pipeline import fqe_sources as fs  # noqa: E402
from data_pipeline.fqe_sources import SealedPartitionError  # noqa: E402
from eval import fqe_benchmark as fb  # noqa: E402
from pipelines import rebuild_fqe  # noqa: E402
from train import fqe_dev_experiment as dx  # noqa: E402
from train.calibrate_uncertainty import NLL_CONVENTION, gaussian_nll  # noqa: E402
from train.calibrate_uncertainty import coverage as interval_coverage  # noqa: E402

REPORT_SCHEMA = "fqe-dev-report-v1"
SELECTION_SCHEMA = "fqe-dev-selection-v1"
PARITY_SCHEMA = "fqe-dev-onnx-parity-v1"
CANDIDATE_ROW_SCHEMA = fb.ROW_SCHEMA
EVIDENCE_SCHEMA = rebuild_fqe.PROTOTYPE_GATE_SCHEMA
#: Schema of a FAILED gate record. Never the accepted EVIDENCE_SCHEMA, so
#: ``rebuild_fqe.verify_prototype_gate`` refuses it whatever its other fields say.
FAILED_EVIDENCE_SCHEMA = "fqe-prototype-gate-result-failed-v1"
GATE_COMPARATOR = rebuild_fqe.PROTOTYPE_CLASSICAL_COMPARATOR
PARITY_TOLERANCE = rebuild_fqe.PROTOTYPE_PARITY_TOLERANCE
UNAVAILABLE = fb.UNAVAILABLE
SEALED = ("test", "test_sealed")
EVAL_PARTITION = "val"
CALIBRATION_PARTITION = "calibration"
ROLE_SIM = "synthetic_reference"
ROLE_ARCHIVE = "archive_measurement_reference"
CANDIDATE_FAMILY = "fqe_cnn"

EXIT_OK, EXIT_ERROR, EXIT_REFUSED, EXIT_DATA = 0, 1, 2, 3
EXIT_GATE_FAILED = 1
EXIT_NO_CANDIDATE = 1

#: Keys of a trial config that define the matched optimizer budget/recipe.
BUDGET_KEYS = ("batch_size", "updates", "warmup_updates", "lambda_rank", "lr", "weight_decay", "grad_clip_norm",
               "lr_schedule")

#: The frozen M3 matrix (contract-v1 ``training`` + the recipe confirmed in
#: decisions F-009). The budget of every reported trial must equal this.
FROZEN_MATRIX = {
    "arms": ["synthetic_only", "hst_only", "mixed"],
    "seeds": [0, 1, 2],
    "budget": {"batch_size": 32, "updates": 500, "warmup_updates": 50, "lambda_rank": 0.0, "lr": 1e-3,
               "weight_decay": 1e-4, "grad_clip_norm": 1.0, "lr_schedule": "cosine"},
    "source": ("contract-v1 training block (batch_size 32, updates 500, mean_only_warmup_updates 50, "
               "lambda_rank 0, seeds 0/1/2, arms synthetic_only/hst_only/mixed) and the recipe confirmed in "
               "decisions F-009 before any trial (AdamW lr 1e-3, weight_decay 1e-4, grad clip 1.0, cosine "
               "annealing over the update budget)"),
}
#: contract-v1 ``training`` keys -> FROZEN_MATRIX budget keys (checked with --matrix-contract).
MATRIX_CONTRACT_KEYS = {"batch_size": "batch_size", "updates": "updates",
                        "mean_only_warmup_updates": "warmup_updates", "lambda_rank": "lambda_rank"}

EVIDENCE_TYPE = {
    ROLE_SIM: ("synthetic latent truth: the noiseless finite-aperture reference of simulated stars (agreement "
               "with simulated truth, not with a measurement)"),
    ROLE_ARCHIVE: ("measurement-reference agreement: the target is a SEP measurement on calibrated HST electrons "
                   "(the teacher), not independent physical truth"),
}
INTERPRETATION = {
    ROLE_SIM: "synthetic families (3 variants share one target); a family is the independent unit",
    ROLE_ARCHIVE: ("HST validation exposures/dates repeat one field with one instrument: agreement here is not "
                   "evidence about unseen stars, fields, cameras or rigs; an exposure is the independent unit and "
                   "there are very few of them"),
}
GROUP_LEVEL = {ROLE_SIM: "family", ROLE_ARCHIVE: "exposure (date)"}
SWEEP_RANK = ("unavailable: singleton archive/simulation rows have no within-run focus ordering "
              "(ranking_eligible false)")
F010_RULE = (
    "F-010 (declared before any validation result was viewed): eligible arm = all declared seeds finished with "
    "finite training and finite predictions on both development populations; seed score = mean(log-HFR MAE on "
    "the synthetic val population, log-HFR MAE on the HST val population); arm score = mean over seeds, lowest "
    "wins; candidate = the median-score seed of the selected arm (not the best seed). If no arm is eligible, "
    "nothing is exported."
)
TIE_RULE = ("ties: arms by (score, arm name); seeds by (score, seed) before taking the median position "
            "(index (n-1)//2: the middle of 3 seeds; the lower middle for an even count, fixture matrices only)")


class ReportRefusal(ValueError):
    """The request contradicts the frozen contract, matrix or development-only policy."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _membership_sha256(uids) -> str:
    """Same definition as ``fqe_sources``/``fqe_benchmark``: sha256 of the sorted uids joined by newlines."""
    return hashlib.sha256("\n".join(sorted(str(u) for u in uids)).encode("utf-8")).hexdigest()


def _num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive_finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _canonical(obj) -> str:
    return json.dumps(fb._clean(obj), sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, obj) -> str:
    data = (json.dumps(fb._clean(obj), indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp = Path(path).with_name(Path(path).name + ".partial")
    tmp.write_bytes(data)
    tmp.replace(path)
    return _sha256_bytes(data)


def _read_json(path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReportRefusal(f"{path} does not exist") from exc
    except ValueError as exc:
        raise ReportRefusal(f"{path} is not valid JSON: {exc}") from exc


def _refuse_sealed_name(name, where: str) -> None:
    if str(name) in SEALED:
        raise SealedPartitionError(f"{where}: the test partition is sealed; development reports never read it")


def _quantiles(values) -> dict | str:
    v = np.asarray([x for x in values if _num(x)], dtype=np.float64)
    if v.size == 0:
        return UNAVAILABLE
    p25, p50, p75 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "min": float(v.min()), "p25": float(p25), "median": float(p50), "p75": float(p75),
            "max": float(v.max())}


def spread(per_seed: Mapping) -> dict:
    """Seed values kept individually plus a descriptive min/median/max (no interval claimed)."""
    values = [float(v) for v in per_seed.values() if _num(v)]
    out = {"per_seed": {str(k): v for k, v in per_seed.items()}, "n": len(values)}
    if values:
        out.update(min=float(np.min(values)), median=float(np.median(values)), max=float(np.max(values)))
    else:
        out.update(min=UNAVAILABLE, median=UNAVAILABLE, max=UNAVAILABLE)
    return out


def _out_dir(path, *, empty: bool, create: bool = True) -> Path:
    out = dx.refuse_forbidden_output(path)
    if empty and out.exists() and any(out.iterdir()):
        raise ReportRefusal(f"{out} is not empty; a report is never merged into an earlier one")
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def _out_file(path) -> Path:
    target = dx.refuse_forbidden_output(path)
    if target.exists() and target.is_dir():
        raise ReportRefusal(f"{target} is a directory; give a file path")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


# --------------------------------------------------------------------------- #
# Matrix / contract identity
# --------------------------------------------------------------------------- #
def _tracked_sha256() -> str | None:
    try:
        return fs.load_contract().sha256
    except fs.FqeSourceError:
        return None


def resolve_matrix(contract, *, budget: Mapping | None = None, arms: Sequence[str] | None = None,
                   seeds: Sequence[int] | None = None, matrix_contract=None, matrix_configs=None) -> dict:
    """The matrix every trial is checked against.

    With the tracked source contract the matrix is :data:`FROZEN_MATRIX` and
    cannot be overridden. ``budget``/``arms``/``seeds`` overrides exist for
    fixture contracts only. ``matrix_contract`` (the frozen contract-v1 JSON) is
    verified against the source contract's ``experiment_contract_sha256`` and
    its training block must agree with :data:`FROZEN_MATRIX`. ``matrix_configs``
    (the directory of the matrix's ``<arm>-s<seed>.json`` arm configs) binds
    every trial to the exact file it was fit from (:func:`load_matrix_configs`).
    """
    tracked = contract.sha256 == _tracked_sha256()
    if tracked and any(x is not None for x in (budget, arms, seeds)):
        raise ReportRefusal("the tracked contract's matrix is frozen (FROZEN_MATRIX); budget/arms/seeds overrides "
                            "are for fixture contracts only")
    matrix = {
        "arms": list(arms if arms is not None else FROZEN_MATRIX["arms"]),
        "seeds": [int(s) for s in (seeds if seeds is not None else FROZEN_MATRIX["seeds"])],
        "budget": dict(budget if budget is not None else FROZEN_MATRIX["budget"]),
        "source": FROZEN_MATRIX["source"] if budget is None else "fixture override (non-tracked contract)",
        "tracked_contract": tracked,
        "matrix_contract": None,
    }
    if set(matrix["budget"]) != set(BUDGET_KEYS):
        raise ReportRefusal(f"matrix budget must name exactly {list(BUDGET_KEYS)}")
    if matrix_contract is not None:
        raw = Path(matrix_contract).read_bytes()
        sha = _sha256_bytes(raw)
        if sha != contract.experiment_contract_sha256:
            raise ReportRefusal(f"matrix contract {matrix_contract} sha256 {sha} is not the frozen experiment contract "
                                f"{contract.experiment_contract_sha256} the source contract derives from")
        training = json.loads(raw).get("training") or {}
        for key, budget_key in MATRIX_CONTRACT_KEYS.items():
            if not _num(training.get(key)) or float(training[key]) != float(matrix["budget"][budget_key]):
                raise ReportRefusal(f"matrix contract training.{key}={training.get(key)!r} disagrees with the "
                                    f"matrix budget {budget_key}={matrix['budget'][budget_key]!r}")
        if list(training.get("arms") or []) != matrix["arms"] or \
                [int(s) for s in training.get("seeds") or []] != matrix["seeds"]:
            raise ReportRefusal("matrix contract arms/seeds disagree with the matrix")
        matrix["matrix_contract"] = {"name": Path(matrix_contract).name, "sha256": sha}
    matrix["arm_configs"] = load_matrix_configs(matrix_configs, matrix) if matrix_configs is not None else None
    return matrix


def load_matrix_configs(directory, matrix: Mapping) -> dict:
    """``{trial_id: binding}`` from the matrix's arm config files ``<arm>-s<seed>.json``.

    Each file is resolved exactly as ``fqe_dev_experiment fit`` resolves it
    (defaults filled), must name its own arm/seed, and its budget must equal the
    matrix budget; every arm x seed of the matrix must have one. The binding
    keeps the raw-file sha256 that ``fit`` records as ``input_config_sha256``."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ReportRefusal(f"matrix config directory {directory} does not exist")
    out = {}
    for arm in matrix["arms"]:
        for seed in matrix["seeds"]:
            trial_id = f"{arm}-s{seed}"
            path = directory / f"{trial_id}.json"
            if not path.is_file():
                raise ReportRefusal(f"matrix config {path} is missing (every arm x seed of the matrix needs one)")
            try:
                config, sha = dx.load_config_file(path)
            except dx.ExperimentRefusal as exc:
                raise ReportRefusal(f"matrix config {path}: {exc}") from exc
            if config["arm"] != arm or config["seed"] != seed:
                raise ReportRefusal(f"matrix config {path} names {config['arm']}-s{config['seed']}, not {trial_id}")
            check_budget(config, matrix)
            out[trial_id] = {"name": path.name, "sha256": sha, "sampling": config["sampling"],
                             "sources": list(config["sources"]), "eval_sources": list(config["eval_sources"])}
    return out


def check_matrix_binding(record: Mapping, config: Mapping, matrix: Mapping, trial_id: str) -> dict | None:
    """With matrix configs: the trial was fit from exactly its matrix config file, else refused."""
    configs = matrix.get("arm_configs")
    if not configs:
        return None
    expected = configs.get(trial_id)
    if expected is None:
        raise ReportRefusal(f"{trial_id}: no matrix config for this arm/seed")
    if record.get("input_config_sha256") != expected["sha256"]:
        raise ReportRefusal(f"{trial_id}: fit from an arm config with sha256 {record.get('input_config_sha256')}, not "
                            f"the matrix config {expected['name']} ({expected['sha256']})")
    for key in ("sampling", "sources", "eval_sources"):
        if config.get(key) != expected[key]:
            raise ReportRefusal(f"{trial_id}: {key} {config.get(key)!r} != matrix config {expected[key]!r}")
    return {"name": expected["name"], "sha256": expected["sha256"], "verified": True}


def _expected_sampling(arm: str) -> str:
    return "source_balanced_with_replacement" if arm == "mixed" else "cycled_permutation"


def check_budget(config: Mapping, matrix: Mapping) -> dict:
    """The trial's optimizer budget/recipe equals the matrix budget, else refused."""
    budget = {key: config.get(key) for key in BUDGET_KEYS}
    expected = matrix["budget"]
    diffs = {}
    for key in BUDGET_KEYS:
        got, want = budget[key], expected[key]
        same = (float(got) == float(want)) if _num(got) and _num(want) else got == want
        if not same:
            diffs[key] = {"trial": got, "matrix": want}
    if config.get("sampling") != _expected_sampling(str(config.get("arm"))):
        diffs["sampling"] = {"trial": config.get("sampling"), "matrix": _expected_sampling(str(config.get("arm")))}
    if diffs:
        raise ReportRefusal(f"trial {config.get('arm')}-s{config.get('seed')}: budget differs from the matrix "
                            f"config {diffs}; differently budgeted trials are not part of the controlled matrix")
    return budget


def _check_source_identity(source_id: str, identity: Mapping, contract, where: str) -> None:
    try:
        entry = contract.source(source_id)
    except fs.ContractViolation as exc:
        raise ReportRefusal(f"{where}: source {source_id!r} is not declared by the frozen contract") from exc
    if identity.get("contract_sha256") != contract.sha256:
        raise ReportRefusal(f"{where}: {source_id} was read under contract {identity.get('contract_sha256')}, not the "
                            f"frozen contract {contract.sha256}")
    if identity.get("files") != entry.get("files"):
        raise ReportRefusal(f"{where}: {source_id} pinned file sha256s differ from the frozen contract")
    if identity.get("role") is not None and identity.get("role") != entry.get("role"):
        raise ReportRefusal(f"{where}: {source_id} role {identity.get('role')!r} != contract {entry.get('role')!r}")


def check_trial_identity(config: Mapping, dataset_identity: Mapping, contract, *, tracked: bool) -> None:
    """Config and training-data identity equal the frozen contract, else refused."""
    where = f"trial {config.get('arm')}-s{config.get('seed')}"
    if config.get("contract_sha256") != contract.sha256:
        raise ReportRefusal(f"{where}: config contract_sha256 {config.get('contract_sha256')} is not the frozen "
                            f"contract {contract.sha256}")
    if config.get("runtime_eligible") is not False:
        raise ReportRefusal(f"{where}: config does not record runtime_eligible false")
    if tracked and sorted(config.get("sources") or []) != sorted(dx.ARM_SOURCES.get(config.get("arm"), ())):
        raise ReportRefusal(f"{where}: trains on {config.get('sources')}, not the contract arm's sources")
    sources = (dataset_identity or {}).get("sources") or {}
    if sorted(sources) != sorted(config.get("sources") or []):
        raise ReportRefusal(f"{where}: dataset_identity.json sources {sorted(sources)} != config sources")
    for source_id, identity in sorted(sources.items()):
        _check_source_identity(source_id, identity, contract, f"{where} training data")


# --------------------------------------------------------------------------- #
# Classical (M2) run
# --------------------------------------------------------------------------- #
def load_classical(run_dir, contract) -> dict:
    """The frozen classical benchmark: verified config + rows grouped by source and comparator."""
    run_dir = Path(run_dir).resolve()
    config_path, rows_path = run_dir / "config.json", run_dir / "rows.jsonl"
    config = _read_json(config_path)
    if config.get("schema") != fb.SCHEMA:
        raise ReportRefusal(f"{config_path}: schema {config.get('schema')!r} is not {fb.SCHEMA!r}")
    _refuse_sealed_name(config.get("partition"), f"classical run {run_dir.name}")
    if config.get("partition") != EVAL_PARTITION:
        raise ReportRefusal(f"classical run partition {config.get('partition')!r} is not {EVAL_PARTITION!r}")
    if config.get("runtime_eligible") is not False:
        raise ReportRefusal("classical run does not record runtime_eligible false")
    sources = config.get("sources") or {}
    for source_id, identity in sources.items():
        _check_source_identity(source_id, identity, contract, f"classical run {run_dir.name}")
    if not rows_path.is_file():
        raise ReportRefusal(f"{rows_path} is missing")
    by_source: dict = {}
    with rows_path.open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            _refuse_sealed_name(row.get("partition"), f"{rows_path.name}:{n}")
            if row.get("partition") != EVAL_PARTITION:
                raise ReportRefusal(f"{rows_path.name}:{n}: partition {row.get('partition')!r} is not val")
            if row.get("source") not in sources:
                raise ReportRefusal(f"{rows_path.name}:{n}: source {row.get('source')!r} not in the run config")
            by_source.setdefault(row["source"], {}).setdefault(row["comparator_id"], []).append(row)
    comparators = {}
    for source_id, by_cid in by_source.items():
        expected = (sources[source_id].get("eval_population") or {}).get("membership_sha256")
        info = {}
        for cid, rows in sorted(by_cid.items()):
            uids = [r["sample_uid"] for r in rows]
            if len(set(uids)) != len(uids) or _membership_sha256(uids) != expected:
                raise ReportRefusal(f"classical {cid} on {source_id}: rows do not cover the declared val population "
                                    "exactly once")
            summary = fb.summarize_rows(rows)
            info[cid] = {"family": rows[0].get("family"), "method": rows[0].get("method"),
                         "measurement_role": rows[0].get("measurement_role"), "population_kind": rows[0].get("population"),
                         "arm": rows[0].get("arm"), "context": rows[0].get("context"),
                         "mask_policy": rows[0].get("mask_policy"), "population": summary["population"],
                         "measured": summary["measured"], "coverage_full_population": summary["coverage"],
                         "status_counts": summary["status_counts"], "metrics_own_measured_rows": summary["metrics"],
                         "timing_ms_per_row": summary.get("timing_ms_per_row")}
        comparators[source_id] = info
    return {
        "run_dir": str(run_dir),
        "config_sha256": _sha256_file(config_path),
        "rows_sha256": _sha256_file(rows_path),
        "config": config,
        "rows": by_source,
        "comparators": comparators,
        "arms": {name: (ident or {}).get("sources") for name, ident in (config.get("arms") or {}).items()},
    }


def classical_arm_for(training_sources: Sequence[str], classical: Mapping) -> str | None:
    """The M2 arm fitted on the same training sources as a trial (constant/sharpness comparators)."""
    wanted = sorted(training_sources or [])
    for name, sources in sorted((classical.get("arms") or {}).items()):
        if sorted(sources or []) == wanted:
            return name
    return None


# --------------------------------------------------------------------------- #
# Trials
# --------------------------------------------------------------------------- #
def _read_csv(path: Path) -> list:
    with Path(path).open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _curves_finite(run_dir: Path, updates: int) -> tuple[bool, str | None]:
    path = run_dir / "curves.jsonl"
    if not path.is_file():
        return False, "curves.jsonl missing"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [int(entry["update"]) for entry in lines] != list(range(1, updates + 1)):
        return False, f"curves.jsonl holds {len(lines)} updates, not 1..{updates}"
    if not all(entry.get("finite") is True and _num(entry.get("loss")) for entry in lines):
        return False, "a training loss is non-finite"
    return True, None


def _load_predictions(run_dir: Path, record: Mapping, contract) -> dict:
    """Verified per-source val (and calibration) prediction rows of a completed trial."""
    pred_dir = run_dir / "predictions"
    manifest_path = pred_dir / "manifest.json"
    if not manifest_path.is_file():
        return {"status": "missing", "reason": "predictions/manifest.json missing (predict not run)"}
    for child in pred_dir.iterdir():
        stem = child.name.rsplit(".", 1)[0]
        if any(stem.endswith(f"__{name}") for name in SEALED):
            raise SealedPartitionError(f"{child}: a test-partition prediction file exists; development reports refuse it")
    manifest = _read_json(manifest_path)
    for name in manifest.get("partitions") or []:
        _refuse_sealed_name(name, f"{manifest_path}")
    for entry in manifest.get("outputs") or []:
        _refuse_sealed_name(entry.get("partition"), f"{manifest_path}")
    if manifest.get("model_sha256") != record.get("final_model_sha256") or \
            manifest.get("run_config_sha256") != record.get("config_sha256"):
        raise ReportRefusal(f"{manifest_path}: predictions belong to different weights/config (stale)")
    for source_id, identity in (manifest.get("source_identity") or {}).items():
        _check_source_identity(source_id, identity, contract, f"{run_dir.name} evaluation data")
    sources = {}
    for entry in manifest.get("outputs") or []:
        partition, source_id = entry.get("partition"), entry.get("source_id")
        if partition not in (EVAL_PARTITION, CALIBRATION_PARTITION):
            continue
        block = sources.setdefault(source_id, {})
        if entry.get("status") != "ok":
            block[partition] = {"status": entry.get("status"), "reason": entry.get("reason"), "rows": []}
            continue
        path = pred_dir / entry["file"]
        if not path.is_file() or _sha256_file(path) != entry.get("sha256"):
            raise ReportRefusal(f"{path}: missing or sha256 differs from the predictions manifest")
        rows = _read_csv(path)
        for k, row in enumerate(rows, 2):
            _refuse_sealed_name(row.get("partition"), f"{path.name}:{k}")
            if row.get("partition") != partition or row.get("source_id") != source_id:
                raise ReportRefusal(f"{path.name}:{k}: row partition/source {row.get('partition')!r}/"
                                    f"{row.get('source_id')!r} != manifest {partition!r}/{source_id!r}")
        block[partition] = {"status": "ok", "file": entry["file"], "sha256": entry["sha256"], "rows": rows,
                            "ms_per_crop": entry.get("ms_per_crop"), "inference_seconds": entry.get("inference_seconds")}
    return {"status": "ok", "manifest_sha256": _sha256_file(manifest_path), "device": manifest.get("device"),
            "sources": sources}


def _load_calibration(run_dir: Path, record: Mapping) -> dict:
    path = run_dir / "calibration" / "calibration.json"
    if not path.is_file():
        return {"status": "missing", "reason": "calibration/calibration.json missing (calibrate not run)"}
    data = _read_json(path)
    if data.get("model_sha256") != record.get("final_model_sha256") or \
            data.get("run_config_sha256") != record.get("config_sha256"):
        raise ReportRefusal(f"{path}: calibration belongs to different weights/config (stale)")
    if data.get("test_reported") is not False or data.get("evaluated_partition") != EVAL_PARTITION or \
            data.get("fit_partition") != CALIBRATION_PARTITION:
        raise ReportRefusal(f"{path}: not a development calibration (fit on calibration, assessed on val, no test)")
    return {"status": "ok", "sha256": _sha256_file(path), "nll_convention": data.get("nll_convention"),
            "sources": data.get("sources") or {}}


def load_trial(run_dir, *, contract, matrix: Mapping) -> dict:
    """One trial directory: identity/budget verified (refused on mismatch); status kept honestly."""
    run_dir = Path(run_dir).resolve()
    run_path, config_path = run_dir / "run.json", run_dir / "config.json"
    if not run_path.is_file():
        raise ReportRefusal(f"{run_dir} is not an fqe_dev_experiment trial directory (no run.json)")
    record = _read_json(run_path)
    trial = {"run_dir": str(run_dir), "status": record.get("status"), "error": record.get("error"),
             "runtime_eligible": False}
    if config_path.is_file():
        config_bytes = config_path.read_bytes()
        recorded = (run_dir / "config.sha256").read_text().split()[0] if (run_dir / "config.sha256").is_file() else None
        if _sha256_bytes(config_bytes) != recorded:
            raise ReportRefusal(f"{config_path} does not match config.sha256")
        config = json.loads(config_bytes)
        identity = _read_json(run_dir / "dataset_identity.json")
        check_trial_identity(config, identity, contract, tracked=bool(matrix.get("tracked_contract")))
        trial["budget"] = check_budget(config, matrix)
        trial["matrix_config"] = check_matrix_binding(record, config, matrix,
                                                      f"{config.get('arm')}-s{config.get('seed')}")
        trial.update(arm=config.get("arm"), seed=config.get("seed"), config_sha256=recorded,
                     training_sources=list(config.get("sources") or []),
                     eval_sources=list(config.get("eval_sources") or []), identity_verified=True,
                     contract_sha256=config.get("contract_sha256"))
    else:
        trial.update(arm=record.get("arm"), seed=record.get("seed"), config_sha256=None, identity_verified=False,
                     training_sources=[], eval_sources=[])
    if trial["arm"] not in matrix["arms"] or trial["seed"] not in matrix["seeds"]:
        raise ReportRefusal(f"{run_dir}: arm {trial['arm']!r} seed {trial['seed']!r} is not part of the matrix "
                            f"{matrix['arms']} x {matrix['seeds']}")
    trial["trial_id"] = f"{trial['arm']}-s{trial['seed']}"
    trial["comparator_id"] = f"fqe__{trial['arm']}__s{trial['seed']}"
    if trial["status"] != dx.STATUS_COMPLETED or not trial["identity_verified"]:
        trial.update(training_finite=False, predictions={"status": "not_available",
                                                         "reason": f"trial status {trial['status']!r}"},
                     calibration={"status": "not_available"})
        return trial
    try:
        _, record, _ = dx.load_completed_run(run_dir)
    except dx.ExperimentRefusal as exc:
        raise ReportRefusal(f"{run_dir}: {exc}") from exc
    updates = int(trial["budget"]["updates"])
    if int(record.get("updates_completed", -1)) != updates or int(record.get("final_checkpoint_update", -1)) != updates:
        raise ReportRefusal(f"{run_dir}: completed {record.get('updates_completed')} updates, matrix budget {updates}")
    finite, why = _curves_finite(run_dir, updates)
    trial.update(model_sha256=record["final_model_sha256"], device=record.get("device"),
                 updates_completed=updates, training_finite=finite, training_finite_reason=why,
                 sample_exposure=record.get("sample_exposure"), data_roles=record.get("data_roles"),
                 predictions=_load_predictions(run_dir, record, contract),
                 calibration=_load_calibration(run_dir, record))
    return trial


# --------------------------------------------------------------------------- #
# Candidate rows (fqe_benchmark row format) and per-source metrics
# --------------------------------------------------------------------------- #
def candidate_rows(trial: Mapping, source_id: str, csv_rows: Sequence[Mapping], reference_rows: Mapping,
                   *, ms_per_crop=None) -> list:
    """fqe_benchmark-format rows for a trial's val predictions.

    ``reference_rows`` maps sample_uid -> a classical row of the same source
    (its target and strata are the declared population's). A finite positive
    ``exp(mu)`` is ``measured``; anything else is ``unmeasured`` (never 0)."""
    out = []
    for row in csv_rows:
        uid = row["sample_uid"]
        ref = reference_rows.get(uid)
        if ref is None:
            raise ReportRefusal(f"{trial['trial_id']}: {uid} is not in the classical val population")
        target = float(row["target_hfr_px"])
        if not math.isclose(target, float(ref["target"]), rel_tol=1e-9, abs_tol=0.0):
            raise ReportRefusal(f"{trial['trial_id']}: {uid} target {target} != classical {ref['target']}")
        mu, log_var = float(row["mu"]), float(row["log_var"])
        finite = row.get("status") == "ok" and math.isfinite(mu) and math.isfinite(log_var)
        prediction = _positive_finite(math.exp(mu)) if finite and mu < 700 else None
        secondary = {key[len("secondary_"):]: _positive_finite(value)
                     for key, value in row.items() if key.startswith("secondary_")}
        out.append({
            "schema": CANDIDATE_ROW_SCHEMA, "comparator_id": trial["comparator_id"], "family": CANDIDATE_FAMILY,
            "method": "FocusQualityEstimator(pretrained=False) final weights; prediction = exp(mu)",
            "arm": trial["arm"], "seed": trial["seed"], "trial_id": trial["trial_id"], "context": None,
            "mask_policy": None, "population": fb.POPULATION_PER_CROP, "measurement_role": "experimental_candidate",
            "sample_uid": uid, "group_uid": row["group_uid"], "frame_uid": row["frame_uid"], "source": source_id,
            "partition": EVAL_PARTITION, "target": target, "secondary_targets": secondary,
            "prediction": prediction, "status": "measured" if prediction else "unmeasured",
            "reason": None if prediction else "nonfinite_prediction", "mu": mu, "log_var": log_var,
            "log_target": float(row["log_target"]), "processing_mode": "normalized_64x64_cnn_input",
            "input_definition": fb.INPUT_NORMALIZED, "reference_definition": ref.get("reference_definition"),
            "timing_ms": None if ms_per_crop is None else float(ms_per_crop), "timing_scope": "row_amortized",
            "stages": {name: {"status": "not_applicable", "reason": None} for name in fb.STAGES},
            "strata": dict(ref.get("strata") or {}),
        })
    return out


def raw_uncertainty(rows: Sequence[Mapping]) -> dict:
    """Raw (uncalibrated) predictive uncertainty on ln-HFR: NLL (NLL_CONVENTION) and 68/95% interval coverage.

    sigma = exp(0.5 * clamp(log_var, -8, 8)) as in calibrate_uncertainty; rows with a
    non-finite head are excluded and counted."""
    finite = [r for r in rows if math.isfinite(r["mu"]) and math.isfinite(r["log_var"])
              and math.isfinite(r["log_target"])]
    out = {"units": "ln HFR_px", "nll_convention": NLL_CONVENTION, "calibrated": False, "rows": len(rows),
           "finite_rows": len(finite)}
    if not finite:
        return dict(out, nll=UNAVAILABLE, coverage_68=UNAVAILABLE, coverage_95=UNAVAILABLE)
    mu = np.asarray([r["mu"] for r in finite], dtype=np.float64)
    y = np.asarray([r["log_target"] for r in finite], dtype=np.float64)
    sigma = np.exp(0.5 * np.clip(np.asarray([r["log_var"] for r in finite], dtype=np.float64), -8.0, 8.0))
    inside_68 = np.abs(y - mu) <= np.maximum(sigma, 1e-9)
    return dict(out, nll=gaussian_nll(mu, sigma, y), coverage_68=float(np.mean(inside_68)),
                coverage_95=interval_coverage(mu, sigma, y, level=0.95),
                median_sigma=float(np.median(sigma)))


def calibrated_uncertainty(calibration: Mapping, source_id: str, role: str, raw: Mapping) -> dict:
    """The trial's calibrate output for this source (synthetic source only)."""
    if calibration.get("status") != "ok":
        return {"status": calibration.get("status"), "reason": calibration.get("reason")}
    entry = (calibration.get("sources") or {}).get(source_id)
    if entry is None:
        return {"status": "missing", "reason": f"calibration.json has no entry for {source_id}"}
    if role != ROLE_SIM:
        return {"status": "not_reported", "calibration_status": entry.get("status"), "reason": entry.get("reason") or
                "calibration is assessed for the synthetic source only (contract-v1: HST calibration unavailable)"}
    if entry.get("status") != "ok":
        return {"status": entry.get("status"), "reason": entry.get("reason")}
    val = (entry.get("evaluation") or {}).get(EVAL_PARTITION) or {}
    out = {"status": "ok", "fit_on": entry.get("fit_on"), "n_fit": entry.get("n_fit"),
           "sigma_scale": entry.get("sigma_scale"), "assessed_on": EVAL_PARTITION, "n": val.get("n"),
           "nll_convention": entry.get("nll_convention"),
           "nll_before": val.get("nll_before"), "nll_after": val.get("nll_after"),
           "coverage_68_before": val.get("coverage_68_before"), "coverage_68_after": val.get("coverage_68_after"),
           "coverage_95_before": val.get("coverage_95_before"), "coverage_95_after": val.get("coverage_95_after"),
           "note": "scale fitted on the calibration families only, assessed on the disjoint val families; "
                   "development evidence for simulated crops, not for R50/Z114 data"}
    if _num(raw.get("nll")) and _num(val.get("nll_before")):
        out["consistency_raw_minus_calibration_before"] = {
            "nll": float(raw["nll"]) - float(val["nll_before"]),
            "coverage_95": float(raw["coverage_95"]) - float(val.get("coverage_95_before") or 0.0)}
    return out


def group_summary(rows: Sequence[Mapping], *, level: str) -> dict:
    """Per independent group (family / exposure) metrics and their spread; no crop-level interval."""
    groups: dict = {}
    for row in rows:
        groups.setdefault(row["group_uid"], []).append(row)
    per = {}
    for group, members in sorted(groups.items()):
        s = fb.summarize_rows(members)
        per[group] = {"rows": s["population"], "measured": s["measured"], "coverage": s["coverage"],
                      "hfr_mae_px": s["metrics"]["hfr_mae_px"], "log_hfr_mae": s["metrics"]["log_hfr_mae"],
                      "signed_bias_px": s["metrics"]["signed_bias_px"]}
    maes = [v["hfr_mae_px"] for v in per.values()]
    logs = [v["log_hfr_mae"] for v in per.values()]
    finite = [m for m in maes if _num(m)]
    return {"level": level, "n_groups": len(per), "groups_with_measurements": len(finite),
            "group_mean_hfr_mae_px": float(np.mean(finite)) if finite else UNAVAILABLE,
            "group_hfr_mae_px": _quantiles(maes), "group_log_hfr_mae": _quantiles(logs), "per_group": per,
            "note": "descriptive spread over independent groups; no crop-level confidence interval is claimed"}


def source_block(trial: Mapping, source_id: str, role: str, rows: Sequence[Mapping], classical: Mapping,
                 val_entry: Mapping) -> dict:
    """Every M4 quantity for one trial on one source's val population."""
    secondary = sorted({k for r in rows for k in (r.get("secondary_targets") or {})})
    summary = {t: fb.summarize_rows(rows, target=t) for t in ["target", *secondary]}
    classical_rows = classical["rows"][source_id]
    paired = {cid: fb.paired_comparison({trial["comparator_id"]: rows, cid: crow}, [trial["comparator_id"], cid])
              for cid, crow in sorted(classical_rows.items())}
    raw = raw_uncertainty(rows)
    return {
        "role": role,
        "evidence_type": EVIDENCE_TYPE.get(role, role),
        "comparator_id": trial["comparator_id"],
        "predictions_file": val_entry.get("file"),
        "predictions_sha256": val_entry.get("sha256"),
        "membership_sha256": _membership_sha256(r["sample_uid"] for r in rows),
        "predictions_finite": bool(rows) and all(fb.is_measured(r) for r in rows),
        "summary": summary,
        "strata": fb.standard_strata(rows),
        "groups": group_summary(rows, level=GROUP_LEVEL.get(role, "group")),
        "uncertainty": {"raw": raw,
                        "calibrated": calibrated_uncertainty(trial.get("calibration") or {}, source_id, role, raw)},
        "paired": paired,
        "cost": {"ms_per_crop": val_entry.get("ms_per_crop"), "inference_seconds": val_entry.get("inference_seconds"),
                 "device": (trial.get("predictions") or {}).get("device"),
                 "scope": "batched inference of the val crops on this host (fqe_dev_experiment predict)"},
        "sweep_rank_metrics": SWEEP_RANK,
    }


def trial_source_rows(trial: Mapping, source_id: str, classical: Mapping) -> tuple[list, dict]:
    """Candidate rows for one source, after checking the val population equals the classical one."""
    val_entry = trial["predictions"]["sources"][source_id][EVAL_PARTITION]
    if source_id not in classical["rows"]:
        raise ReportRefusal(f"{trial['trial_id']}: evaluation source {source_id} is not in the classical run")
    expected = (classical["config"]["sources"][source_id].get("eval_population") or {}).get("membership_sha256")
    got = _membership_sha256(r["sample_uid"] for r in val_entry["rows"])
    if got != expected:
        raise ReportRefusal(f"{trial['trial_id']}: {source_id} val membership {got} != classical population {expected}")
    reference = {r["sample_uid"]: r for r in classical["rows"][source_id][GATE_COMPARATOR]} \
        if GATE_COMPARATOR in classical["rows"][source_id] else \
        {r["sample_uid"]: r for r in next(iter(classical["rows"][source_id].values()))}
    rows = candidate_rows(trial, source_id, val_entry["rows"], reference, ms_per_crop=val_entry.get("ms_per_crop"))
    return rows, val_entry


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _source_roles(contract, source_ids: Iterable[str]) -> dict:
    return {sid: contract.source(sid).get("role") for sid in sorted(set(source_ids))}


def _arm_summary(trials: Sequence[Mapping], roles: Mapping, classical: Mapping,
                 arm_order: Sequence[str] = ()) -> dict:
    arms: dict = {}
    for trial in trials:
        arms.setdefault(trial["arm"], []).append(trial)
    order = {arm: i for i, arm in enumerate(arm_order)}
    out = {}
    for arm, members in sorted(arms.items(), key=lambda item: (order.get(item[0], len(order)), item[0])):
        members = sorted(members, key=lambda t: t["seed"])
        entry = {"seeds": [t["seed"] for t in members], "trial_ids": [t["trial_id"] for t in members],
                 "statuses": {str(t["seed"]): t["status"] for t in members}, "sources": {}}
        m2_arm = classical_arm_for(members[0].get("training_sources") or [], classical)
        entry["classical_arm"] = m2_arm
        for source_id, role in roles.items():
            blocks = {t["seed"]: (t.get("sources") or {}).get(source_id) for t in members}
            blocks = {s: b for s, b in blocks.items() if b is not None}
            if not blocks:
                continue
            metric = {}
            for key in ("hfr_mae_px", "log_hfr_mae", "signed_bias_px", "median_relative_error"):
                metric[key] = spread({s: b["summary"]["target"]["metrics"][key] for s, b in blocks.items()})
            metric["coverage"] = spread({s: b["summary"]["target"]["coverage"] for s, b in blocks.items()})
            for key in ("nll", "coverage_68", "coverage_95"):
                metric[f"raw_{key}"] = spread({s: b["uncertainty"]["raw"][key] for s, b in blocks.items()})
            if role == ROLE_SIM:
                for key in ("nll_after", "coverage_68_after", "coverage_95_after", "sigma_scale"):
                    metric[f"calibrated_{key}"] = spread({s: b["uncertainty"]["calibrated"].get(key)
                                                          for s, b in blocks.items()})
            metric["ms_per_crop"] = spread({s: b["cost"]["ms_per_crop"] for s, b in blocks.items()})
            groups: dict = {}
            for seed, block in blocks.items():
                for group, values in block["groups"]["per_group"].items():
                    groups.setdefault(group, {})[seed] = values["hfr_mae_px"]
            paired = {}
            for cid in sorted(next(iter(blocks.values()))["paired"]):
                per_seed = {}
                for seed, block in blocks.items():
                    comp = block["paired"][cid]
                    cand_id = block["comparator_id"]
                    a, b = comp["per_comparator"][cand_id], comp["per_comparator"][cid]
                    pair = comp["pairwise"].get(f"{cand_id} vs {cid}", {})
                    per_seed[seed] = {"common_valid": comp["common_valid"],
                                      "candidate_mae_px": a["common_valid_metrics"]["hfr_mae_px"],
                                      "comparator_mae_px": b["common_valid_metrics"]["hfr_mae_px"],
                                      "candidate_log_mae": a["common_valid_metrics"]["log_hfr_mae"],
                                      "comparator_log_mae": b["common_valid_metrics"]["log_hfr_mae"],
                                      "delta_mae_px": pair.get("mean_abs_error_difference_px", UNAVAILABLE),
                                      "fraction_candidate_closer": pair.get("fraction_a_closer", UNAVAILABLE),
                                      "candidate_coverage": a["coverage_full_population"],
                                      "comparator_coverage": b["coverage_full_population"]}
                paired[cid] = {"per_seed": per_seed,
                               "delta_mae_px": spread({s: v["delta_mae_px"] for s, v in per_seed.items()}),
                               "candidate_mae_px": spread({s: v["candidate_mae_px"] for s, v in per_seed.items()}),
                               "comparator_mae_px": spread({s: v["comparator_mae_px"] for s, v in per_seed.items()}),
                               "info": classical["comparators"][source_id].get(cid, {})}
            entry["sources"][source_id] = {
                "role": role, "evidence_type": EVIDENCE_TYPE.get(role, role), "metrics": metric,
                "groups": {"level": GROUP_LEVEL.get(role, "group"),
                           "group_mean_hfr_mae_px": spread({s: b["groups"]["group_mean_hfr_mae_px"]
                                                            for s, b in blocks.items()}),
                           "median_group_hfr_mae_px": spread({s: (b["groups"]["group_hfr_mae_px"] or {}).get("median")
                                                              if isinstance(b["groups"]["group_hfr_mae_px"], dict)
                                                              else UNAVAILABLE for s, b in blocks.items()}),
                           "per_group_hfr_mae_px": {g: spread(v) for g, v in sorted(groups.items())}},
                "paired": paired,
            }
        out[arm] = entry
    return out


def build_report(trial_dirs: Sequence, classical_dir, out_dir, *, contract=None, budget: Mapping | None = None,
                 arms: Sequence[str] | None = None, seeds: Sequence[int] | None = None, matrix_contract=None,
                 matrix_configs=None) -> dict:
    """Write report.json, report.md and candidate_rows.jsonl under ``out_dir`` (new or empty)."""
    contract = contract or fs.load_contract()
    matrix = resolve_matrix(contract, budget=budget, arms=arms, seeds=seeds, matrix_contract=matrix_contract,
                            matrix_configs=matrix_configs)
    if not trial_dirs:
        raise ReportRefusal("no trials given")
    classical = load_classical(classical_dir, contract)
    trials = [load_trial(d, contract=contract, matrix=matrix) for d in trial_dirs]
    seen: dict = {}
    for trial in trials:
        if trial["trial_id"] in seen:
            raise ReportRefusal(f"{trial['trial_id']} given twice ({seen[trial['trial_id']]} and {trial['run_dir']})")
        seen[trial["trial_id"]] = trial["run_dir"]
    out = _out_dir(out_dir, empty=True, create=False)  # refused early; created only once everything computed

    source_ids = set(classical["rows"])
    all_rows = []
    for trial in trials:
        trial["sources"] = {}
        predictions = trial.get("predictions") or {}
        if predictions.get("status") != "ok":
            continue
        for source_id in sorted(predictions["sources"]):
            val_entry = predictions["sources"][source_id].get(EVAL_PARTITION)
            if not val_entry or val_entry.get("status") != "ok":
                continue
            rows, val_entry = trial_source_rows(trial, source_id, classical)
            role = contract.source(source_id).get("role")
            trial["sources"][source_id] = source_block(trial, source_id, role, rows, classical, val_entry)
            cal_entry = predictions["sources"][source_id].get(CALIBRATION_PARTITION) or {}
            trial["sources"][source_id]["calibration_partition_rows"] = len(cal_entry.get("rows") or [])
            all_rows.extend(rows)
            source_ids.add(source_id)
        predictions["sources"] = {sid: {p: {k: v for k, v in e.items() if k != "rows"} | {"rows": len(e.get("rows") or [])}
                                        for p, e in parts.items()} for sid, parts in predictions["sources"].items()}
    roles = _source_roles(contract, source_ids)
    sources = {}
    for source_id, role in roles.items():
        cfg = (classical["config"]["sources"].get(source_id) or {})
        sources[source_id] = {"role": role, "evidence_type": EVIDENCE_TYPE.get(role, role),
                              "interpretation": INTERPRETATION.get(role, ""),
                              "target_definition_id": cfg.get("target_definition_id"),
                              "population": cfg.get("eval_population"),
                              "group_level": GROUP_LEVEL.get(role, "group"), "sweep_rank_metrics": SWEEP_RANK}
    out = _out_dir(out, empty=True)
    rows_path = out / "candidate_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(fb._clean(row), sort_keys=True) + "\n")
    report = {
        "schema": REPORT_SCHEMA,
        "created_utc": _utc(),
        "development_only": True,
        "test_partition_used": False,
        "runtime_eligible": False,
        "partition": EVAL_PARTITION,
        "seal": "val rows only (calibration rows only through the trial's calibrate output); no test partition "
                "was read; any test partition in an input is refused",
        "contract": {"sha256": contract.sha256, "file_sha256": contract.file_sha256,
                     "experiment_contract_sha256": contract.experiment_contract_sha256,
                     "tracked": bool(matrix["tracked_contract"])},
        "matrix": matrix,
        "classical": {"run_dir": classical["run_dir"], "config_sha256": classical["config_sha256"],
                      "rows_sha256": classical["rows_sha256"], "arms": classical["arms"],
                      "comparators": classical["comparators"]},
        "sources": sources,
        "trials": sorted(trials, key=lambda t: (matrix["arms"].index(t["arm"]), t["seed"])),
        "arms": _arm_summary([t for t in trials if t.get("sources")], roles, classical, matrix["arms"]),
        "candidate_rows": {"file": rows_path.name, "sha256": _sha256_file(rows_path), "rows": len(all_rows)},
        "notes": [
            "Synthetic results are agreement with simulated latent truth; HST results are agreement with a SEP "
            "measurement reference (the teacher), not independent truth. They are reported separately and never "
            "pooled.",
            INTERPRETATION[ROLE_ARCHIVE],
            "Paired comparisons use the rows both methods measured (common valid population); each method's "
            "coverage is reported on the full declared val population, so no method wins by excluding hard rows.",
            "Seed results are kept individually; min/median/max over seeds and the group-level spread are "
            "descriptive. No crop-level confidence interval is reported (few independent groups).",
            "Sweep/rank metrics: " + SWEEP_RANK,
            "Development partition only: no final-test performance and no camera/rig qualification is implied.",
        ],
    }
    report_sha = _write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    report["_sha256"] = report_sha
    return report


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _f(value, spec=".3f") -> str:
    return format(value, spec) if _num(value) else ("-" if value is None else str(value))


def _key_comparators(classical_info: Mapping, m2_arm: str | None) -> list:
    keep = []
    for cid, info in sorted(classical_info.items()):
        family = info.get("family")
        if family in ("normalized_input", "sep_flux_radius"):
            keep.append(cid)
        elif family == "native_rust" and "rust_legacy" in cid and info.get("measured", 0) > 0:
            keep.append(cid)
        elif family in ("constant", "sharpness") and info.get("arm") == m2_arm:
            keep.append(cid)
    return keep


def render_markdown(report: Mapping) -> str:
    m = report["matrix"]
    lines = [
        "# FQE development comparison (M4)", "",
        f"Created {report['created_utc']}. Development partition `val` only; the test partition is sealed and was "
        "not read. Not final-test performance; no camera or rig qualification.", "",
        f"- Source contract `{report['contract']['sha256'][:12]}` (tracked: {report['contract']['tracked']}); "
        f"matrix budget {json.dumps(m['budget'], sort_keys=True)}.",
        f"- Classical run `{Path(report['classical']['run_dir']).name}` (rows sha256 "
        f"`{report['classical']['rows_sha256'][:12]}`). Per-row candidate outputs: candidate_rows.jsonl.",
        "- Matrix binding: experiment contract "
        + (f"`{m['matrix_contract']['name']}` (`{m['matrix_contract']['sha256'][:12]}`, training block checked)"
           if m.get("matrix_contract") else "not given (--matrix-contract)")
        + "; arm configs "
        + ("verified (every trial was fit from its matrix config file: input sha256 matches)"
           if m.get("arm_configs") else "not given (--matrix-configs)") + ".",
        "", "## Trials", "",
        "| trial | status | updates | training finite | device | model sha256 |",
        "|---|---|---|---|---|---|",
    ]
    for t in report["trials"]:
        lines.append(f"| {t['trial_id']} | {t['status']} | {t.get('updates_completed', '-')} | "
                     f"{t.get('training_finite')} | {t.get('device', '-')} | {str(t.get('model_sha256', '-'))[:12]} |")
    lines.append("")
    order = sorted(report["sources"], key=lambda s: (report["sources"][s]["role"] != ROLE_SIM, s))
    for source_id in order:
        info = report["sources"][source_id]
        title = ("Synthetic latent truth" if info["role"] == ROLE_SIM else
                 "HST measurement-reference agreement" if info["role"] == ROLE_ARCHIVE else info["role"])
        pop = info.get("population") or {}
        lines += [f"## {title}: {source_id}", "",
                  f"{info['evidence_type']}. Population: {pop.get('rows')} val rows in {pop.get('groups')} "
                  f"{info['group_level']} groups; target `{info.get('target_definition_id')}`. "
                  f"{info['interpretation']}.", "",
                  "| arm | seed | coverage | HFR MAE px | log MAE | bias px | median rel err | raw NLL | raw cov68 | "
                  "raw cov95 | cal cov68 | cal cov95 | cal NLL | ms/crop |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        secondary_lines = []
        for t in report["trials"]:
            block = (t.get("sources") or {}).get(source_id)
            if block is None:
                lines.append(f"| {t['arm']} | {t['seed']} | not available ({t['status']}) |" + " |" * 11)
                continue
            s, unc = block["summary"]["target"], block["uncertainty"]
            cal = unc["calibrated"]
            ok = cal.get("status") == "ok"
            lines.append(
                f"| {t['arm']} | {t['seed']} | {s['measured']}/{s['population']} | {_f(s['metrics']['hfr_mae_px'])} | "
                f"{_f(s['metrics']['log_hfr_mae'])} | {_f(s['metrics']['signed_bias_px'], '+.3f')} | "
                f"{_f(s['metrics']['median_relative_error'])} | {_f(unc['raw']['nll'])} | "
                f"{_f(unc['raw']['coverage_68'])} | {_f(unc['raw']['coverage_95'])} | "
                f"{_f(cal.get('coverage_68_after')) if ok else '-'} | {_f(cal.get('coverage_95_after')) if ok else '-'} | "
                f"{_f(cal.get('nll_after')) if ok else '-'} | {_f(block['cost']['ms_per_crop'])} |")
            for name, summ in sorted(block["summary"].items()):
                if name == "target":
                    continue
                sm = summ["metrics"]
                secondary_lines.append(
                    f"| {t['arm']} | {t['seed']} | {name} | {summ['measured']}/{summ['population']} | "
                    f"{_f(sm['hfr_mae_px'])} | {_f(sm['log_hfr_mae'])} | {_f(sm['signed_bias_px'], '+.3f')} |")
        if info["role"] == ROLE_SIM:
            lines += ["", "Uncertainty: raw sigma = exp(0.5*clamp(log_var, -8, 8)) in ln-HFR units, NLL "
                      f"convention `{NLL_CONVENTION}`; cal = the trial's own calibrate output (one scale fitted on "
                      "the reserved calibration families only, assessed on these disjoint val families). "
                      "Development evidence for simulated crops, not for R50/Z114 data."]
        else:
            lines += ["", "Uncertainty: raw only and exploratory (no calibration for this source; contract-v1: too "
                      "few independent training exposures). The variance head is not called calibrated here."]
        if secondary_lines:
            lines += ["", "Secondary target (a separate quantity from the primary target; not the gate):", "",
                      "| arm | seed | target | measured | HFR MAE px | log MAE | bias px |",
                      "|---|---|---|---|---|---|---|"] + secondary_lines
        lines += ["", "Seed spread (min / median / max):", "",
                  "| arm | HFR MAE px | log MAE | coverage | group-mean MAE |", "|---|---|---|---|---|"]
        for arm, entry in report["arms"].items():
            src = entry["sources"].get(source_id)
            if not src:
                continue
            mm = src["metrics"]
            row = [f"{_f(mm[k]['min'])} / {_f(mm[k]['median'])} / {_f(mm[k]['max'])}"
                   for k in ("hfr_mae_px", "log_hfr_mae", "coverage")]
            g = src["groups"]["group_mean_hfr_mae_px"]
            lines.append(f"| {arm} (seeds {entry['seeds']}) | {' | '.join(row)} | "
                         f"{_f(g['min'])} / {_f(g['median'])} / {_f(g['max'])} |")
        lines += ["", "Paired on the common valid rows (median over seeds; delta = candidate - comparator, px; "
                  "coverage on the full population):", "",
                  "| arm | comparator | role | common n | candidate MAE | comparator MAE | delta min/median/max | "
                  "candidate cov | comparator cov |", "|---|---|---|---|---|---|---|---|---|"]
        classical_info = report["classical"]["comparators"].get(source_id, {})
        for arm, entry in report["arms"].items():
            src = entry["sources"].get(source_id)
            if not src:
                continue
            for cid in _key_comparators(classical_info, entry.get("classical_arm")):
                p = src["paired"].get(cid)
                if p is None:
                    continue
                per = list(p["per_seed"].values())
                commons = sorted({v["common_valid"] for v in per})
                cand_cov = np.median([v["candidate_coverage"] for v in per if _num(v["candidate_coverage"])] or [np.nan])
                comp_cov = np.median([v["comparator_coverage"] for v in per if _num(v["comparator_coverage"])] or [np.nan])
                d = p["delta_mae_px"]
                lines.append(f"| {arm} | {cid} | {p['info'].get('measurement_role', '-')} | "
                             f"{'/'.join(str(c) for c in commons)} | {_f(p['candidate_mae_px']['median'])} | "
                             f"{_f(p['comparator_mae_px']['median'])} | {_f(d['min'], '+.3f')} / "
                             f"{_f(d['median'], '+.3f')} / {_f(d['max'], '+.3f')} | {_f(cand_cov)} | {_f(comp_cov)} |")
        lines += ["", "Processing cost on this host (candidate: batched val inference by fqe_dev_experiment predict, "
                  "median over seeds; classical: median per-row timing where the M2 row scope is 'row'):", "",
                  "| method | ms per crop | scope |", "|---|---|---|"]
        for arm, entry in report["arms"].items():
            src = entry["sources"].get(source_id)
            if src:
                devices = sorted({str(((t.get("predictions") or {}).get("device"))) for t in report["trials"]
                                  if t["arm"] == arm and (t.get("sources") or {}).get(source_id)})
                lines.append(f"| fqe {arm} | {_f(src['metrics']['ms_per_crop']['median'])} | "
                             f"batched, device {'/'.join(devices)} |")
        for cid in _key_comparators(classical_info, None):
            timing = classical_info[cid].get("timing_ms_per_row") or {}
            lines.append(f"| {cid} | {_f(timing.get('median'))} | {'row' if timing else 'not row-scoped'} |")
        lines += ["", f"Group-level ({info['group_level']}) HFR MAE px per seed:", ""]
        if info["role"] == ROLE_ARCHIVE or (pop.get("groups") or 0) <= 10:
            lines += ["Cells list the seeds in order " + ", ".join(
                f"{arm}: {entry['seeds']}" for arm, entry in report["arms"].items()) + ".", ""]
            groups = sorted({g for e in report["arms"].values() for g in
                             ((e["sources"].get(source_id) or {}).get("groups", {}).get("per_group_hfr_mae_px") or {})})
            sizes = {}
            for t in report["trials"]:
                for g, values in (((t.get("sources") or {}).get(source_id) or {}).get("groups", {})
                                  .get("per_group") or {}).items():
                    sizes.setdefault(g, values.get("rows"))
            lines += ["| arm | " + " | ".join(f"{g.split(':', 1)[-1]} (n={sizes.get(g, '-')})" for g in groups)
                      + " |", "|---|" + "---|" * len(groups)]
            for arm, entry in report["arms"].items():
                src = entry["sources"].get(source_id)
                if not src:
                    continue
                per = src["groups"]["per_group_hfr_mae_px"]
                cells = [" / ".join(_f(v) for v in (per.get(g) or {}).get("per_seed", {}).values()) for g in groups]
                lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        else:
            lines += ["| arm | seed | groups | group-mean MAE | min | p25 | median | p75 | max |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for t in report["trials"]:
                block = (t.get("sources") or {}).get(source_id)
                if block is None:
                    continue
                g = block["groups"]
                q = g["group_hfr_mae_px"] if isinstance(g["group_hfr_mae_px"], dict) else {}
                lines.append(f"| {t['arm']} | {t['seed']} | {g['n_groups']} | {_f(g['group_mean_hfr_mae_px'])} | "
                             + " | ".join(_f(q.get(k)) for k in ("min", "p25", "median", "p75", "max")) + " |")
        lines += ["", "Strata (HFR MAE px, median over seeds; n rows):", "",
                  "| stratum | value | n | " + " | ".join(report["arms"]) + " |",
                  "|---|---|---|" + "---|" * len(report["arms"])]
        strata_keys: dict = {}
        for t in report["trials"]:
            block = (t.get("sources") or {}).get(source_id)
            if block is None:
                continue
            for key, values in block["strata"].items():
                for value, summ in values.items():
                    cell = strata_keys.setdefault((key, value), {"n": summ["population"], "arms": {}})
                    cell["arms"].setdefault(t["arm"], []).append(summ["metrics"]["hfr_mae_px"])
        for (key, value), cell in sorted(strata_keys.items()):
            vals = []
            for arm in report["arms"]:
                xs = [x for x in cell["arms"].get(arm, []) if _num(x)]
                vals.append(_f(float(np.median(xs))) if xs else "-")
            lines.append(f"| {key} | {value} | {cell['n']} | " + " | ".join(vals) + " |")
        lines += ["", f"Sweep/rank metrics: {SWEEP_RANK}.", ""]
    lines += ["## Notes", ""] + [f"- {note}" for note in report["notes"]] + [""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# select (F-010)
# --------------------------------------------------------------------------- #
def _role_source(report: Mapping, role: str) -> str:
    matches = [sid for sid, info in report["sources"].items() if info.get("role") == role]
    if len(matches) != 1:
        raise ReportRefusal(f"F-010 needs exactly one {role} development population; the report has {matches}")
    return matches[0]


def select_candidate(report: Mapping, *, report_sha256: str | None = None) -> dict:
    """Apply decision F-010 to a report (pure; no file access)."""
    if report.get("schema") != REPORT_SCHEMA:
        raise ReportRefusal(f"not an {REPORT_SCHEMA} report")
    if report.get("test_partition_used") is not False:
        raise ReportRefusal("the report does not state test_partition_used: false")
    sim, hst = _role_source(report, ROLE_SIM), _role_source(report, ROLE_ARCHIVE)
    seeds = list(report["matrix"]["seeds"])
    by_arm: dict = {}
    for trial in report["trials"]:
        by_arm.setdefault(trial["arm"], {})[trial["seed"]] = trial
    ranking = []
    for arm in report["matrix"]["arms"]:
        entry = {"arm": arm, "seeds": [], "reasons": []}
        for seed in seeds:
            trial = by_arm.get(arm, {}).get(seed)
            seed_entry = {"seed": seed, "trial_id": f"{arm}-s{seed}", "reasons": []}
            if trial is None:
                seed_entry["reasons"].append("trial not reported")
            else:
                if trial.get("status") != dx.STATUS_COMPLETED:
                    seed_entry["reasons"].append(f"status {trial.get('status')!r}")
                if trial.get("training_finite") is not True:
                    seed_entry["reasons"].append(f"training not finite ({trial.get('training_finite_reason')})")
                values = {}
                for label, source_id in (("sim", sim), ("hst", hst)):
                    block = (trial.get("sources") or {}).get(source_id)
                    if block is None:
                        seed_entry["reasons"].append(f"no val predictions on {source_id}")
                        continue
                    if block.get("predictions_finite") is not True:
                        seed_entry["reasons"].append(f"non-finite predictions on {source_id}")
                    value = block["summary"]["target"]["metrics"]["log_hfr_mae"]
                    if not _num(value):
                        seed_entry["reasons"].append(f"log-HFR MAE on {source_id} is {value!r}")
                    values[label] = value
                seed_entry.update(log_hfr_mae_sim=values.get("sim"), log_hfr_mae_hst=values.get("hst"),
                                  model_sha256=trial.get("model_sha256"), config_sha256=trial.get("config_sha256"),
                                  run_dir=trial.get("run_dir"), comparator_id=trial.get("comparator_id"))
                if _num(values.get("sim")) and _num(values.get("hst")):
                    seed_entry["score"] = (float(values["sim"]) + float(values["hst"])) / 2.0
            seed_entry["eligible"] = not seed_entry["reasons"]
            entry["seeds"].append(seed_entry)
        extra = sorted(set(by_arm.get(arm, {})) - set(seeds))
        if extra:
            entry["reasons"].append(f"undeclared seeds {extra}")
        entry["eligible"] = not entry["reasons"] and all(s["eligible"] for s in entry["seeds"])
        if not entry["eligible"]:
            entry["reasons"] += [f"seed {s['seed']}: {'; '.join(s['reasons'])}" for s in entry["seeds"] if s["reasons"]]
        entry["score"] = float(np.mean([s["score"] for s in entry["seeds"]])) if entry["eligible"] else None
        ranking.append(entry)
    eligible = sorted((e for e in ranking if e["eligible"]), key=lambda e: (e["score"], e["arm"]))
    ineligible = [e for e in ranking if not e["eligible"]]
    for rank, entry in enumerate(eligible, 1):
        entry["rank"] = rank
    notes = []
    candidate, selected = None, None
    if eligible:
        best = eligible[0]
        selected = best["arm"]
        if len(eligible) > 1 and eligible[1]["score"] == best["score"]:
            notes.append(f"arm score tie between {best['arm']} and {eligible[1]['arm']}; broken by arm name")
        ordered = sorted(best["seeds"], key=lambda s: (s["score"], s["seed"]))
        if len({s["score"] for s in ordered}) != len(ordered):
            notes.append("seed score tie inside the selected arm; broken by seed number")
        median = ordered[(len(ordered) - 1) // 2]
        best["seed_order"] = [s["seed"] for s in ordered]
        candidate = {key: median.get(key) for key in ("trial_id", "seed", "score", "log_hfr_mae_sim",
                                                      "log_hfr_mae_hst", "model_sha256", "config_sha256", "run_dir",
                                                      "comparator_id")}
        candidate["arm"] = selected
        candidate["arm_score"] = best["score"]
    return {
        "schema": SELECTION_SCHEMA,
        "created_utc": _utc(),
        "rule": F010_RULE,
        "tie_rule": TIE_RULE,
        "report_sha256": report_sha256,
        "contract_sha256": report["contract"]["sha256"],
        "populations": {"synthetic": {"source_id": sim, "partition": EVAL_PARTITION},
                        "hst": {"source_id": hst, "partition": EVAL_PARTITION}},
        "metric": "log_hfr_mae on each source's full val population (summary.target.metrics.log_hfr_mae)",
        "test_partition_used": False,
        "status": "selected" if candidate else "no_eligible_arm",
        "selected_arm": selected,
        "candidate": candidate,
        "ranking": eligible + ineligible,
        "notes": notes or ["no ties"],
    }


def select_file(report_path, out_path=None) -> dict:
    raw = Path(report_path).read_bytes()
    try:
        report = json.loads(raw)
    except ValueError as exc:
        raise ReportRefusal(f"{report_path} is not valid JSON: {exc}") from exc
    selection = select_candidate(report, report_sha256=_sha256_bytes(raw))
    target = _out_file(out_path or Path(report_path).resolve().parent / "selection.json")
    _write_json(target, selection)
    selection["_path"] = str(target)
    return selection


# --------------------------------------------------------------------------- #
# parity
# --------------------------------------------------------------------------- #
def fixture_inputs() -> dict:
    """Deterministic non-data inputs: 8 uniform-random crops (export_quality_onnx's) and the rebuild ramp."""
    rng = np.random.default_rng(0)
    return {"random_uniform_seed0": rng.random((8, 64, 64)).astype(np.float32),
            "ramp": (np.arange(64 * 64, dtype=np.float32) / (64 * 64)).reshape(1, 64, 64)}


def _compare(model, session, crops: np.ndarray) -> dict:
    from train.calibrate_uncertainty import predict_mu_log_var

    crops = np.asarray(crops, dtype=np.float32)
    t_mu, t_lv = (np.ravel(v) for v in predict_mu_log_var(model, crops, "cpu"))
    o_mu, o_lv = [], []
    for start in range(0, len(crops), 256):
        mu, lv = session.run(None, {"image": np.ascontiguousarray(crops[start:start + 256][:, None])})
        o_mu.append(np.ravel(mu))
        o_lv.append(np.ravel(lv))
    o_mu = np.concatenate(o_mu).astype(np.float64) if o_mu else np.zeros(0)
    o_lv = np.concatenate(o_lv).astype(np.float64) if o_lv else np.zeros(0)
    n = int(len(crops))
    if not (t_mu.shape == t_lv.shape == o_mu.shape == o_lv.shape == (n,)):
        raise ReportRefusal(f"parity: output shapes torch {t_mu.shape}/{t_lv.shape} onnx {o_mu.shape}/{o_lv.shape} "
                            f"are not ({n},)")
    finite = bool(np.all(np.isfinite(t_mu)) and np.all(np.isfinite(t_lv)) and np.all(np.isfinite(o_mu))
                  and np.all(np.isfinite(o_lv)))
    d_mu = float(np.max(np.abs(t_mu - o_mu))) if n else None
    d_lv = float(np.max(np.abs(t_lv - o_lv))) if n else None
    return {"rows": n, "max_abs_delta_mu": d_mu, "max_abs_delta_log_var": d_lv, "finite": finite}


def parity_passed(parity: Mapping, *, tolerance: float = PARITY_TOLERANCE) -> tuple[bool, list]:
    """Recompute the parity verdict from the recorded numbers (never from a stored 'passed')."""
    failures = []
    entries = {f"fixture:{k}": v for k, v in (parity.get("fixtures") or {}).items()}
    entries.update({f"dev:{k}": v for k, v in (parity.get("dev_crops") or {}).items()})
    if not parity.get("fixtures"):
        failures.append("no fixture inputs compared")
    if not parity.get("dev_crops"):
        failures.append("no development crops compared")
    for name, e in sorted(entries.items()):
        if not e.get("rows"):
            failures.append(f"{name}: no rows")
            continue
        if e.get("finite") is not True:
            failures.append(f"{name}: non-finite outputs")
        for key in ("max_abs_delta_mu", "max_abs_delta_log_var"):
            if not _num(e.get(key)) or float(e[key]) > tolerance:
                failures.append(f"{name}: {key}={e.get(key)!r} > {tolerance:g}")
    return not failures, failures


def onnx_parity(trial_dir, onnx_path, *, data_root=None, contract=None) -> dict:
    """Torch final weights (CPU) vs onnxruntime on fixtures and every evaluation source's val crops."""
    import onnxruntime as ort
    import torch

    from models.focus_quality_cnn import FocusQualityEstimator

    trial_dir = Path(trial_dir).resolve()
    try:
        config, record, model_path = dx.load_completed_run(trial_dir)
    except dx.ExperimentRefusal as exc:
        raise ReportRefusal(f"{trial_dir}: {exc}") from exc
    contract = contract or fs.load_contract(config.get("contract_path"))
    if contract.sha256 != config.get("contract_sha256"):
        raise ReportRefusal(f"{trial_dir}: trained under contract {config.get('contract_sha256')}, not "
                            f"{contract.sha256}")
    root = data_root if data_root is not None else config.get("data_root")
    model = FocusQualityEstimator(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    fixtures = {name: _compare(model, session, crops) for name, crops in fixture_inputs().items()}
    dev = {}
    for source_id in sorted(config.get("eval_sources") or []):
        arrays = fs.load_source(source_id, partitions=(EVAL_PARTITION,), contract=contract, data_root=root)
        if set(np.asarray(arrays.partition).astype(str).tolist()) - {EVAL_PARTITION}:
            raise SealedPartitionError(f"{source_id}: the loader returned non-val rows for a val request")
        entry = _compare(model, session, arrays.crops)
        entry.update(partition=EVAL_PARTITION, membership_sha256=_membership_sha256(arrays.sample_uid),
                     role=str(arrays.role))
        dev[source_id] = entry
    result = {
        "schema": PARITY_SCHEMA,
        "created_utc": _utc(),
        "trial_id": f"{config.get('arm')}-s{config.get('seed')}",
        "run_dir": str(trial_dir),
        "model_sha256": record["final_model_sha256"],
        "config_sha256": record["config_sha256"],
        "contract_sha256": contract.sha256,
        "onnx_name": Path(onnx_path).name,
        "onnx_sha256": _sha256_file(onnx_path),
        "torch_device": "cpu",
        "versions": {"torch": torch.__version__, "onnxruntime": ort.__version__},
        "tolerance": PARITY_TOLERANCE,
        "criterion": "every output finite and max |delta mu| and max |delta log_var| <= tolerance on the fixture "
                     "inputs and on the val crops of every evaluation source (F-006/F-010: <= 1e-4)",
        "fixtures": fixtures,
        "dev_crops": dev,
        "test_partition_used": False,
    }
    all_entries = list(fixtures.values()) + list(dev.values())
    result["max_abs_delta_mu"] = max((e["max_abs_delta_mu"] for e in all_entries if _num(e["max_abs_delta_mu"])),
                                     default=None)
    result["max_abs_delta_log_var"] = max((e["max_abs_delta_log_var"] for e in all_entries
                                           if _num(e["max_abs_delta_log_var"])), default=None)
    passed, failures = parity_passed(result)
    result.update(passed=passed, failures=failures)
    return result


def parity_file(trial_dir, onnx_path, out_path=None, *, data_root=None, contract=None) -> dict:
    """Write the parity JSON (default: ``<onnx stem>.parity.json`` next to the ONNX file)."""
    if out_path is None:
        out_path = Path(onnx_path).resolve().with_name(Path(onnx_path).stem + ".parity.json")
    target = _out_file(out_path)
    result = onnx_parity(trial_dir, onnx_path, data_root=data_root, contract=contract)
    _write_json(target, result)
    result["_path"] = str(target)
    return result


# --------------------------------------------------------------------------- #
# evidence (F-006 gate)
# --------------------------------------------------------------------------- #
def _informational(classical_rows: Mapping, cand_id: str, cand_rows: Sequence[Mapping],
                   candidate_arm: str | None = None) -> dict:
    """Non-gating paired comparisons against the Rust legacy (g1) and constant comparators (decision F-010).

    ``candidate_arm`` is the M2 arm fitted on the candidate's training sources
    (:func:`classical_arm_for`), so the matching constant is marked."""
    out = {}
    for cid, rows in sorted(classical_rows.items()):
        if "rust_legacy__g1" not in cid and not cid.startswith("constant__"):
            continue
        comp = fb.paired_comparison({cand_id: cand_rows, cid: rows}, [cand_id, cid])
        a, b = comp["per_comparator"][cand_id], comp["per_comparator"][cid]
        pair = comp["pairwise"].get(f"{cand_id} vs {cid}", {})
        arm = rows[0].get("arm") if rows else None
        out[cid] = {"gating": False, "family": rows[0].get("family") if rows else None, "fitted_arm": arm,
                    "same_training_sources_as_candidate": (arm == candidate_arm) if arm else None,
                    "candidate_no_worse_on_common_rows": (
                        bool(float(a["common_valid_metrics"]["hfr_mae_px"])
                             <= float(b["common_valid_metrics"]["hfr_mae_px"]))
                        if comp["common_valid"] > 0 else UNAVAILABLE),
                    "delta_mae_px": pair.get("mean_abs_error_difference_px", UNAVAILABLE),
                    "common_valid": comp["common_valid"],
                    "candidate_mae_px": a["common_valid_metrics"]["hfr_mae_px"],
                    "comparator_mae_px": b["common_valid_metrics"]["hfr_mae_px"],
                    "candidate_coverage": a["coverage_full_population"],
                    "comparator_coverage": b["coverage_full_population"],
                    "paired_comparison": comp}
    return out


def _selection_differences(selection: Mapping, expected: Mapping) -> list:
    """Top-level fields where a selection file differs from the recomputed F-010 outcome (timestamps ignored)."""
    ignore = {"created_utc", "_path"}
    keys = (set(selection) | set(expected)) - ignore
    return sorted(k for k in keys if _canonical(selection.get(k)) != _canonical(expected.get(k)))


def build_evidence(selection_path, parity_path, report_path, *, contract=None) -> dict:
    """The F-006 gate record (``gate.passed`` computed, never assumed).

    The selection must be exactly what :func:`select_candidate` (F-010) gives
    for this report. The record carries the accepted ``EVIDENCE_SCHEMA`` only
    when every check passed; otherwise ``FAILED_EVIDENCE_SCHEMA``."""
    report_raw = Path(report_path).read_bytes()
    report = json.loads(report_raw)
    selection = _read_json(selection_path)
    parity_raw = Path(parity_path).read_bytes()
    parity = json.loads(parity_raw)
    if report.get("schema") != REPORT_SCHEMA or selection.get("schema") != SELECTION_SCHEMA or \
            parity.get("schema") != PARITY_SCHEMA:
        raise ReportRefusal("evidence needs a report, a selection and a parity file of this module's schemas")
    if selection.get("report_sha256") != _sha256_bytes(report_raw):
        raise ReportRefusal("the selection was made from a different report.json (report_sha256 differs)")
    candidate = selection.get("candidate")
    if not candidate:
        raise ReportRefusal("F-010 selected no candidate (no eligible arm): nothing is exported")
    # Re-apply F-010 to this exact report: the evidence binds the rule's outcome, not whatever the file names.
    expected = select_candidate(report, report_sha256=_sha256_bytes(report_raw))
    if expected["status"] != "selected" or not expected["candidate"]:
        raise ReportRefusal("the selection is not the F-010 outcome of this report: F-010 selects no candidate "
                            "(no eligible arm), so nothing is exported")
    differs = _selection_differences(selection, expected)
    if differs:
        raise ReportRefusal(f"the selection is not the F-010 outcome of this report (differs in {differs}; F-010 "
                            f"selects {expected['candidate']['trial_id']}); re-run select on this report")
    contract = contract or fs.load_contract()
    if report["contract"]["sha256"] != contract.sha256:
        raise ReportRefusal(f"the report was built under contract {report['contract']['sha256']}, not {contract.sha256}")
    if parity.get("model_sha256") != candidate["model_sha256"] or parity.get("config_sha256") != candidate["config_sha256"]:
        raise ReportRefusal("the parity file was computed for different weights/config than the selected candidate")
    if parity.get("contract_sha256") != contract.sha256:
        raise ReportRefusal("the parity file's dev crops were loaded under a different contract")
    for source_id, dev_entry in sorted((parity.get("dev_crops") or {}).items()):
        _refuse_sealed_name(dev_entry.get("partition"), f"parity dev crops {source_id}")
        population = (report["sources"].get(source_id) or {}).get("population") or {}
        if dev_entry.get("partition") != EVAL_PARTITION or \
                dev_entry.get("membership_sha256") != population.get("membership_sha256"):
            raise ReportRefusal(f"parity dev crops of {source_id} are not the report's val population "
                                "(membership sha256 differs)")
    entry = next((t for t in report["trials"] if t["trial_id"] == candidate["trial_id"]), None)
    if entry is None or entry.get("model_sha256") != candidate["model_sha256"]:
        raise ReportRefusal("the selected candidate is not a trial of the report")

    # Re-derive the gate population from the files the report was built from.
    matrix = dict(report["matrix"])
    if matrix.get("tracked_contract") and (matrix["budget"] != FROZEN_MATRIX["budget"]
                                           or matrix["arms"] != FROZEN_MATRIX["arms"]
                                           or matrix["seeds"] != FROZEN_MATRIX["seeds"]):
        raise ReportRefusal("the report's matrix is not the frozen matrix")
    classical = load_classical(report["classical"]["run_dir"], contract)
    if classical["rows_sha256"] != report["classical"]["rows_sha256"] or \
            classical["config_sha256"] != report["classical"]["config_sha256"]:
        raise ReportRefusal("the classical run changed since the report was built")
    trial = load_trial(candidate["run_dir"], contract=contract, matrix=matrix)
    if trial.get("model_sha256") != candidate["model_sha256"] or trial.get("predictions", {}).get("status") != "ok":
        raise ReportRefusal("the candidate trial changed since the report was built or has no predictions")
    sim = _role_source(report, ROLE_SIM)
    cand_id = trial["comparator_id"]
    rows_by_source, finite_by_source = {}, {}
    for source_id in sorted(report["sources"]):
        val_entry = (trial["predictions"]["sources"].get(source_id) or {}).get(EVAL_PARTITION) or {}
        if val_entry.get("status") != "ok":
            finite_by_source[source_id] = False
            continue
        rows, val_entry = trial_source_rows(trial, source_id, classical)
        reported = (entry.get("sources") or {}).get(source_id) or {}
        if val_entry.get("sha256") != reported.get("predictions_sha256"):
            raise ReportRefusal(f"{source_id} predictions changed since the report was built")
        rows_by_source[source_id] = rows
        finite_by_source[source_id] = bool(rows) and all(fb.is_measured(r) for r in rows)
    if sim not in rows_by_source:
        raise ReportRefusal(f"the candidate has no val predictions on the synthetic population {sim}")
    if GATE_COMPARATOR not in classical["rows"].get(sim, {}):
        raise ReportRefusal(f"the classical run has no {GATE_COMPARATOR} rows on {sim}")
    comparison = fb.paired_comparison({cand_id: rows_by_source[sim], GATE_COMPARATOR: classical["rows"][sim][GATE_COMPARATOR]},
                                      [cand_id, GATE_COMPARATOR])
    reported_comparison = entry["sources"][sim]["paired"].get(GATE_COMPARATOR)
    if _canonical(reported_comparison) != _canonical(comparison):
        raise ReportRefusal("the recomputed gate comparison differs from report.json (stale report)")

    a, b = comparison["per_comparator"][cand_id], comparison["per_comparator"][GATE_COMPARATOR]
    cand_mae, comp_mae = a["common_valid_metrics"]["hfr_mae_px"], b["common_valid_metrics"]["hfr_mae_px"]
    cand_cov, comp_cov = a["coverage_full_population"], b["coverage_full_population"]
    parity_ok, parity_failures = parity_passed(parity)
    parity_finite = all(e.get("finite") is True for e in list((parity.get("fixtures") or {}).values())
                        + list((parity.get("dev_crops") or {}).values()))
    if sim not in (parity.get("dev_crops") or {}):
        parity_ok = False
        parity_failures.append(f"no dev-crop parity on the gate population {sim}")
    finite = bool(all(finite_by_source.values()) and finite_by_source and parity_finite)
    checks = {
        "finite": {"passed": finite, "val_predictions_finite": finite_by_source, "onnx_outputs_finite": parity_finite},
        "onnx_parity": {"passed": parity_ok, "tolerance": PARITY_TOLERANCE,
                        "max_abs_delta_mu": parity.get("max_abs_delta_mu"),
                        "max_abs_delta_log_var": parity.get("max_abs_delta_log_var"),
                        "onnx_sha256": parity.get("onnx_sha256"), "failures": parity_failures},
        "mae_no_worse": {"passed": bool(comparison["common_valid"] > 0 and _num(cand_mae) and _num(comp_mae)
                                        and float(cand_mae) <= float(comp_mae)),
                         "candidate_mae_px": cand_mae, "classical_mae_px": comp_mae,
                         "common_valid_rows": comparison["common_valid"]},
        "coverage_no_lower": {"passed": bool(_num(cand_cov) and _num(comp_cov) and float(cand_cov) >= float(comp_cov)),
                              "candidate_coverage": cand_cov, "classical_coverage": comp_cov},
    }
    failures = [f"{name}: {json.dumps(fb._clean({k: v for k, v in c.items() if k != 'passed'}), sort_keys=True)}"
                for name, c in checks.items() if not c["passed"]]
    passed = not failures
    population = report["sources"][sim].get("population") or {}
    return {
        "schema": EVIDENCE_SCHEMA if passed else FAILED_EVIDENCE_SCHEMA,
        "created_utc": _utc(),
        "candidate_model_sha256": candidate["model_sha256"],
        "run_config_sha256": candidate["config_sha256"],
        "contract_sha256": trial["contract_sha256"],
        "population": {"source_id": sim, "partition": EVAL_PARTITION, "rows": population.get("rows"),
                       "groups": population.get("groups"), "membership_sha256": population.get("membership_sha256")},
        "test_partition_used": False,
        "candidate_outputs_finite": finite,
        "candidate_comparator": cand_id,
        "classical_comparator": GATE_COMPARATOR,
        "paired_comparison": comparison,
        "gate": {"decision": "F-006 (declared before validation results); comparator fixed in F-006, restated in F-010",
                 "criteria": rebuild_fqe.PROTOTYPE_GATE_CRITERIA, "checks": checks, "passed": passed,
                 "failures": failures, "computed_by": "eval.fqe_dev_report evidence (recomputed from the trial "
                 "predictions and the classical rows; cross-checked against report.json)",
                 "record_schema": (f"{EVIDENCE_SCHEMA} only when every check passed; a failed gate is recorded "
                                   f"as {FAILED_EVIDENCE_SCHEMA}, which pipelines.rebuild_fqe refuses")},
        "informational_comparators": {
            "note": "not gating: shown so the gate's low bar is visible (decision F-010)",
            "comparators": _informational(classical["rows"][sim], cand_id, rows_by_source[sim],
                                          classical_arm_for(trial.get("training_sources") or [], classical)),
        },
        "candidate": {"trial_id": candidate["trial_id"], "arm": candidate["arm"], "seed": candidate["seed"],
                      "run_name": Path(candidate["run_dir"]).name, "comparator_id": cand_id},
        "inputs": {"report_sha256": _sha256_bytes(report_raw), "selection_sha256": _sha256_file(selection_path),
                   "parity_sha256": _sha256_bytes(parity_raw), "classical_rows_sha256": classical["rows_sha256"],
                   "selection_rule": selection.get("rule")},
        "evidence_types": {"gate_population": EVIDENCE_TYPE[ROLE_SIM],
                           "note": "the gate uses the synthetic population only; HST agreement is in report.json"},
    }


def evidence_file(selection_path, parity_path, report_path, out_path, *, contract=None) -> dict:
    target = _out_file(out_path)
    evidence = build_evidence(selection_path, parity_path, report_path, contract=contract)
    _write_json(target, evidence)
    evidence["_path"] = str(target)
    return evidence


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FQE development report, F-010 selection, ONNX parity and F-006 "
                                                 "gate evidence (development data only).")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("report", help="per-trial/per-arm development comparison")
    p.add_argument("--trials", nargs="+", required=True, help="fqe_dev_experiment trial directories")
    p.add_argument("--classical", required=True, help="frozen eval.fqe_benchmark run directory (val)")
    p.add_argument("--out", required=True, help="new/empty output directory")
    p.add_argument("--contract", default=None, help="source contract JSON (default: tracked v1)")
    p.add_argument("--matrix-contract", default=None,
                   help="frozen experiment contract (contract-v1.json); verified by sha256 and training block")
    p.add_argument("--matrix-configs", default=None,
                   help="directory of the matrix arm configs (<arm>-s<seed>.json); every trial must have been fit "
                        "from its file (run.json input_config_sha256) and every budget must equal the matrix's")
    p.add_argument("--budget-json", default=None,
                   help="fixture contracts only (refused with the tracked contract): matrix budget as a JSON object")
    s = sub.add_parser("select", help="apply decision F-010 to a report")
    s.add_argument("--report", required=True)
    s.add_argument("--out", default=None, help="default: selection.json next to the report")
    q = sub.add_parser("parity", help="torch vs onnxruntime parity on fixtures and val crops")
    q.add_argument("--trial", required=True)
    q.add_argument("--onnx", required=True)
    q.add_argument("--out", default=None, help="parity JSON file (default: <onnx stem>.parity.json next to the ONNX)")
    q.add_argument("--data-root", default=None)
    q.add_argument("--contract", default=None, help="default: the contract the trial was trained under")
    e = sub.add_parser("evidence", help=f"F-006 gate record: {EVIDENCE_SCHEMA} when it passes, "
                                        f"{FAILED_EVIDENCE_SCHEMA} (refused by rebuild_fqe) when it fails")
    e.add_argument("--selection", required=True)
    e.add_argument("--parity", required=True)
    e.add_argument("--report", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--contract", default=None, help="source contract JSON (default: tracked v1)")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        contract = fs.load_contract(args.contract) if getattr(args, "contract", None) else None
        if args.command == "report":
            budget = json.loads(args.budget_json) if args.budget_json else None
            report = build_report(args.trials, args.classical, args.out, contract=contract, budget=budget,
                                  matrix_contract=args.matrix_contract, matrix_configs=args.matrix_configs)
            print(json.dumps({"report": str(Path(args.out).resolve() / "report.json"), "sha256": report["_sha256"],
                              "trials": [{"trial": t["trial_id"], "status": t["status"]} for t in report["trials"]]},
                             indent=2))
            return EXIT_OK
        if args.command == "select":
            selection = select_file(args.report, args.out)
            print(json.dumps({"selection": selection["_path"], "status": selection["status"],
                              "candidate": selection["candidate"]}, indent=2, default=str))
            return EXIT_OK if selection["candidate"] else EXIT_NO_CANDIDATE
        if args.command == "parity":
            result = parity_file(args.trial, args.onnx, args.out, data_root=args.data_root, contract=contract)
            print(json.dumps({"parity": result["_path"], "passed": result["passed"],
                              "max_abs_delta_mu": result["max_abs_delta_mu"],
                              "max_abs_delta_log_var": result["max_abs_delta_log_var"],
                              "failures": result["failures"]}, indent=2))
            return EXIT_OK if result["passed"] else EXIT_GATE_FAILED
        evidence = evidence_file(args.selection, args.parity, args.report, args.out, contract=contract)
        gate = evidence["gate"]
        print(json.dumps({"evidence": evidence["_path"], "schema": evidence["schema"],
                          "gate": "PASS" if gate["passed"] else "FAIL",
                          "failures": gate["failures"]}, indent=2))
        return EXIT_OK if gate["passed"] else EXIT_GATE_FAILED
    except (fs.DataUnavailable, fs.HashMismatch) as exc:
        print(f"data unavailable: {exc}", file=sys.stderr)
        return EXIT_DATA
    except (ReportRefusal, SealedPartitionError, fs.ContractViolation, dx.ExperimentRefusal, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
