"""fqe_dev_experiment — development-only FQE trials on contract sources.

The trainer side of the public-data FQE experiment (frozen contract
``fqe-public-experiment-contract-v1``). Four separate entry points:

  fit         train one arm/seed for a fixed optimizer-update budget. Writes an
              immutable ``config.json`` (+ sha256), the dataset identity, a
              per-update ``curves.jsonl``, resumable checkpoints (model,
              optimizer, scheduler, Python/NumPy/torch RNG, update counter)
              and the final weights at the last update. No validation is
              consulted during training.
  predict     per-row mu/log_var on development partitions (val, calibration,
              train_fit). ``test`` is refused.
  calibrate   quantile interval scale fitted only on each source's
              ``calibration`` partition and assessed on its ``val``; sources
              without calibration groups (HST) are recorded ``unavailable``.
  final-test  the sealed test partition, once, through
              ``fqe_sources.load_final_test``: needs a frozen candidate receipt
              and ``--i-acknowledge-final-test``. No other entry point calls it.

Every result file records ``runtime_eligible: false`` and the data roles.
Outputs may not be written under the repository's ``models/`` (the production
registry) or under the public data root.

Usage (cwd ``ml/``):
    python -B -m train.fqe_dev_experiment fit --config arm.json --out RUN_DIR [--resume]
    python -B -m train.fqe_dev_experiment predict --run RUN_DIR --partition val --partition calibration
    python -B -m train.fqe_dev_experiment calibrate --run RUN_DIR
    python -B -m train.fqe_dev_experiment final-test --run RUN_DIR --source stellar_sim_v1 \\
        --receipt receipt.json --i-acknowledge-final-test

Arm config (JSON; omitted keys take the defaults in ``DEFAULTS``)::

    {"schema": "fqe-dev-arm-config-v1", "arm": "synthetic_only", "seed": 0,
     "batch_size": 32, "updates": 500, "warmup_updates": 50, "lambda_rank": 0,
     "device": "mps"}
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parent

CONFIG_SCHEMA = "fqe-dev-arm-config-v1"
RUN_SCHEMA = "fqe-dev-run-v1"
CHECKPOINT_SCHEMA = "fqe-dev-checkpoint-v1"
PREDICTIONS_SCHEMA = "fqe-dev-predictions-v1"
CALIBRATION_SCHEMA = "fqe-dev-calibration-v1"
FINAL_TEST_SCHEMA = "fqe-dev-final-test-v1"

ARM_SOURCES = {
    "synthetic_only": ("stellar_sim_v1",),
    "hst_only": ("hst_acs_corpus_v1",),
    "mixed": ("hst_acs_corpus_v1", "stellar_sim_v1"),
}
#: Roles an arm may train on when a non-default (fixture) contract names its sources.
ARM_ROLES = {
    "synthetic_only": {"synthetic_reference", "fixture_only"},
    "hst_only": {"archive_measurement_reference", "fixture_only"},
    "mixed": {"synthetic_reference", "archive_measurement_reference", "fixture_only"},
}
DEFAULT_EVAL_SOURCES = ("hst_acs_corpus_v1", "stellar_sim_v1")
PREDICT_PARTITIONS = ("val", "calibration", "train_fit")
SAMPLING = ("cycled_permutation", "source_balanced_with_replacement")
LR_SCHEDULES = ("cosine", "constant")

#: Contract-v1 training settings; AdamW values are the trainer's
#: (train_focus_quality.ADAMW_LR / ADAMW_WEIGHT_DECAY / GRAD_CLIP_NORM).
DEFAULTS = {
    "schema": CONFIG_SCHEMA,
    "arm": None,
    "seed": None,
    "batch_size": 32,
    "updates": 500,
    "warmup_updates": 50,
    "lambda_rank": 0.0,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip_norm": 1.0,
    "lr_schedule": "cosine",
    "checkpoint_every": 50,
    "device": "auto",
    "data_root": None,
    "contract_path": None,
    "sources": None,
    "eval_sources": None,
    "sampling": None,
}

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED_NONFINITE = "failed_nonfinite"
STATUS_INTERRUPTED = "interrupted"
STATUS_DATA_UNAVAILABLE = "data_unavailable"
STATUS_DATA_REJECTED = "data_rejected"
STATUS_FAILED = "failed"
RESUMABLE_STATUSES = (STATUS_RUNNING, STATUS_INTERRUPTED, STATUS_DATA_UNAVAILABLE)

EXIT_OK, EXIT_ERROR, EXIT_REFUSED, EXIT_DATA, EXIT_NONFINITE, EXIT_INTERRUPTED = 0, 1, 2, 3, 4, 5

FINAL_TEST_FLAG = "--i-acknowledge-final-test"


class ExperimentRefusal(ValueError):
    """The request contradicts the experiment contract (raised before any output)."""


class TrialInterrupted(RuntimeError):
    """The trial stopped before its update budget (signal, drill or lost data)."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _sources_module():
    from data_pipeline import fqe_sources

    return fqe_sources


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _canonical(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n").encode("utf-8")


def _write_json_atomic(path: Path, payload) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(_canonical(payload))
    tmp.replace(path)


def _write_exclusive(path: Path, data: bytes, *, read_only: bool = False) -> None:
    with Path(path).open("xb") as fh:
        fh.write(data)
    if read_only:
        os.chmod(path, 0o444)


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _float_or_str(value: float):
    value = float(value)
    return value if math.isfinite(value) else repr(value)


def _resolved(path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path)))).resolve()


def _forbidden_output_roots(data_root=None) -> list[tuple[Path, str]]:
    roots = [(REPO_ROOT / "models", "the production model registry (<repo>/models)"),
             (ML_ROOT / "data" / "public", "the public data root (ml/data/public)")]
    env_root = os.environ.get("ACHERNAR_PUBLIC_DATA_ROOT")
    if env_root:
        roots.append((Path(env_root), "the public data root ($ACHERNAR_PUBLIC_DATA_ROOT)"))
    if data_root:
        roots.append((Path(data_root), "the configured data root"))
    return roots


def _casefold_within(target: Path, root: Path) -> bool:
    """``target`` equals or lies under ``root``, comparing components case-folded."""
    t = [part.casefold() for part in target.parts]
    r = [part.casefold() for part in root.parts]
    return len(t) >= len(r) and t[: len(r)] == r


def _existing_ancestor_ids(path: Path) -> list[tuple[int, int]]:
    """``(st_dev, st_ino)`` of ``path`` and of every existing ancestor."""
    ids = []
    for candidate in (path, *path.parents):
        try:
            st = os.stat(candidate)
        except OSError:
            continue
        ids.append((st.st_dev, st.st_ino))
    return ids


def refuse_forbidden_output(path, *, data_root=None) -> Path:
    """Resolve an output directory and refuse the production model registry
    (``<repo>/models``) and the public data root (link and target).

    String prefixes are not enough on macOS: APFS/HFS+ volumes are normally
    case- (and Unicode-normalization-) insensitive and ``Path.resolve`` does
    not canonicalise case, so ``.../achernar.tech/MODELS/x`` names the same
    directory as ``.../Achernar.tech/models/x``. A path is refused when any of
    its existing ancestors IS a forbidden root (same device and inode), or,
    for roots that do not exist now (an unmounted drive), when its components
    match the root's case-insensitively. The case-folded match also refuses
    case variants on a case-sensitive volume, which errs on the safe side."""
    target = _resolved(path)
    ancestor_ids = set(_existing_ancestor_ids(target))
    for root, why in _forbidden_output_roots(data_root):
        spellings = {_resolved(root), Path(os.path.abspath(os.path.expanduser(str(root))))}
        refused = any(target == c or target.is_relative_to(c) or _casefold_within(target, c) for c in spellings)
        if not refused:
            for candidate in spellings:
                try:
                    st = os.stat(candidate)
                except OSError:
                    continue
                if (st.st_dev, st.st_ino) in ancestor_ids:
                    refused = True
                    break
        if refused:
            raise ExperimentRefusal(f"refusing to write {target}: it lies inside {why}")
    return target


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def resolve_config(raw: dict) -> dict:
    """Validate an arm config and fill defaults. Pure: no data access."""
    if not isinstance(raw, dict):
        raise ExperimentRefusal("arm config must be a JSON object")
    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise ExperimentRefusal(f"unknown arm config keys {unknown}; allowed: {sorted(DEFAULTS)}")
    config = copy.deepcopy(DEFAULTS)
    config.update(copy.deepcopy(raw))
    if config["schema"] != CONFIG_SCHEMA:
        raise ExperimentRefusal(f"schema must be {CONFIG_SCHEMA!r}, got {config['schema']!r}")
    if config["arm"] not in ARM_SOURCES:
        raise ExperimentRefusal(f"arm must be one of {sorted(ARM_SOURCES)}, got {config['arm']!r}")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool) or config["seed"] < 0:
        raise ExperimentRefusal("seed must be a non-negative integer")
    for key in ("batch_size", "updates", "checkpoint_every"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ExperimentRefusal(f"{key} must be a positive integer")
    if config["batch_size"] < 2:
        raise ExperimentRefusal("batch_size must be >= 2 (BatchNorm trains on the batch)")
    if not isinstance(config["warmup_updates"], int) or not 0 <= config["warmup_updates"] <= config["updates"]:
        raise ExperimentRefusal("warmup_updates must be an integer in [0, updates]")
    for key in ("lambda_rank", "lr", "weight_decay", "grad_clip_norm"):
        if not isinstance(config[key], (int, float)) or isinstance(config[key], bool):
            raise ExperimentRefusal(f"{key} must be a number")
        config[key] = float(config[key])
    if config["lr"] <= 0 or config["grad_clip_norm"] <= 0 or config["weight_decay"] < 0 or config["lambda_rank"] < 0:
        raise ExperimentRefusal("lr and grad_clip_norm must be > 0; weight_decay and lambda_rank >= 0")
    if config["lr_schedule"] not in LR_SCHEDULES:
        raise ExperimentRefusal(f"lr_schedule must be one of {LR_SCHEDULES}")
    if config["device"] not in ("auto", "cpu", "mps", "cuda"):
        raise ExperimentRefusal("device must be auto, cpu, mps or cuda")

    default_contract = config["contract_path"] is None
    sources = config["sources"]
    if sources is None:
        sources = list(ARM_SOURCES[config["arm"]])
    elif default_contract:
        if sorted(sources) != sorted(ARM_SOURCES[config["arm"]]):
            raise ExperimentRefusal(
                f"with the tracked contract, arm {config['arm']!r} trains on {list(ARM_SOURCES[config['arm']])}; "
                f"got {sources}"
            )
    if not isinstance(sources, list) or not sources or len(set(sources)) != len(sources):
        raise ExperimentRefusal("sources must be a non-empty list of distinct source ids")
    if config["arm"] == "mixed" and len(sources) < 2:
        raise ExperimentRefusal("the mixed arm needs at least two sources")
    if config["arm"] != "mixed" and len(sources) != 1:
        raise ExperimentRefusal(f"arm {config['arm']!r} trains on exactly one source")
    config["sources"] = sorted(sources)
    eval_sources = config["eval_sources"]
    if eval_sources is None:
        eval_sources = list(DEFAULT_EVAL_SOURCES) if default_contract else list(config["sources"])
    if not isinstance(eval_sources, list) or not eval_sources:
        raise ExperimentRefusal("eval_sources must be a non-empty list")
    config["eval_sources"] = sorted(set(eval_sources))

    sampling = config["sampling"]
    if sampling is None:
        sampling = "source_balanced_with_replacement" if config["arm"] == "mixed" else "cycled_permutation"
    if sampling not in SAMPLING:
        raise ExperimentRefusal(f"sampling must be one of {SAMPLING}")
    if config["arm"] == "mixed" and sampling != "source_balanced_with_replacement":
        raise ExperimentRefusal("the mixed arm samples source-balanced with replacement (contract-v1 mixed_arm)")
    config["sampling"] = sampling
    if config["contract_path"] is not None:
        config["contract_path"] = str(config["contract_path"])
    if config["data_root"] is not None:
        config["data_root"] = str(config["data_root"])
    return config


def load_config_file(path) -> tuple[dict, str]:
    raw_bytes = Path(path).read_bytes()
    try:
        raw = json.loads(raw_bytes)
    except ValueError as exc:
        raise ExperimentRefusal(f"arm config {path} is not valid JSON: {exc}") from exc
    return resolve_config(raw), _sha256_bytes(raw_bytes)


def _load_contract(config: dict):
    fs = _sources_module()
    return fs.load_contract(config.get("contract_path"))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _identity_projection(identity: dict) -> dict:
    """The fields of a source identity that must not change between a fit and a
    resume/predict (``data_root`` may legitimately be spelled differently)."""
    keys = ("source_id", "role", "contract_sha256", "files", "npz_sha256", "registry_sha256",
            "target_definition_id", "batching", "rows_total", "partition_counts", "calibration",
            "sealed_test")
    return {key: identity.get(key) for key in keys if key in identity}


def load_training_sources(config: dict, *, contract=None, data_root=None, loader=None):
    """Load each training source's ``train_fit`` rows and the mixture plan."""
    fs = _sources_module()
    contract = contract if contract is not None else _load_contract(config)
    loader = loader or fs.load_source
    root = data_root if data_root is not None else config.get("data_root")
    sources = [loader(sid, partitions=("train_fit",), contract=contract, data_root=root) for sid in config["sources"]]
    roles = {s.source_id: str(s.role) for s in sources}
    allowed = ARM_ROLES[config["arm"]]
    wrong = {sid: role for sid, role in roles.items() if role not in allowed}
    if wrong:
        raise ExperimentRefusal(f"arm {config['arm']!r} cannot train on roles {wrong} (allowed: {sorted(allowed)})")
    batching = {str(s.batching) for s in sources}
    if batching != {"pointwise"}:
        raise ExperimentRefusal(
            f"fqe_dev_experiment fits pointwise archive/simulation sources; got batching {sorted(batching)}. "
            "Genuine local runs train through train_focus_quality.train (GroupedBatchSampler)."
        )
    try:
        plan = fs.make_mixture(sources, partition="train_fit", contract=contract)
    except fs.ContractViolation as exc:
        # Collisions or undeclared target mixing are request errors, not data faults.
        raise ExperimentRefusal(f"mixture refused: {exc}") from exc
    return sources, plan, contract


# --------------------------------------------------------------------------- #
# RNG / checkpoint state
# --------------------------------------------------------------------------- #
def _capture_rng(device: str) -> dict:
    import torch

    version, internal, gauss = random.getstate()
    algorithm, keys, pos, has_gauss, cached = np.random.get_state()
    state = {
        "python": {"version": int(version), "internal": [int(v) for v in internal], "gauss": gauss},
        "numpy": {"algorithm": str(algorithm), "keys": [int(v) for v in keys], "pos": int(pos),
                  "has_gauss": int(has_gauss), "cached_gaussian": float(cached)},
        "torch_cpu": torch.get_rng_state(),
    }
    if device == "mps":
        state["torch_mps"] = torch.mps.get_rng_state()
    if device.startswith("cuda"):
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict, device: str) -> None:
    import torch

    py = state["python"]
    random.setstate((py["version"], tuple(py["internal"]), py["gauss"]))
    npy = state["numpy"]
    np.random.set_state((npy["algorithm"], np.asarray(npy["keys"], dtype=np.uint32), npy["pos"],
                         npy["has_gauss"], npy["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"])
    if device == "mps" and "torch_mps" in state:
        torch.mps.set_rng_state(state["torch_mps"])
    if device.startswith("cuda") and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _to_cpu(obj):
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def _make_scheduler(optimizer, config: dict):
    import torch

    if config["lr_schedule"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["updates"])
    return None


class _RunFiles:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.config = self.dir / "config.json"
        self.config_sha = self.dir / "config.sha256"
        self.identity = self.dir / "dataset_identity.json"
        self.run = self.dir / "run.json"
        self.curves = self.dir / "curves.jsonl"
        self.discarded = self.dir / "curves.discarded.jsonl"
        self.checkpoints = self.dir / "checkpoints"
        self.final = self.dir / "final"
        self.final_model = self.final / "model_state.pt"

    def checkpoint(self, update: int) -> Path:
        return self.checkpoints / f"state_{update:06d}.pt"


def _determinism_record(device: str) -> dict:
    import torch

    notes = [
        "seeds: python random, numpy global, torch (all devices) = config seed",
        "batches: pure function of (seed, update) (UpdateStream / per-update numpy Generator)",
        "torch.use_deterministic_algorithms(True, warn_only=True) during fit",
    ]
    if device == "mps":
        notes.append(
            "MPS kernels are not guaranteed bitwise reproducible (deterministic-algorithm flags do not "
            "cover every MPS op); reruns/resumes on MPS may differ at floating-point rounding level"
        )
    elif device.startswith("cuda"):
        notes.append("CUDA: cuDNN autotuning not disabled explicitly; bitwise reproducibility not claimed")
    else:
        notes.append(f"CPU with {torch.get_num_threads()} intra-op threads; same thread count expected for "
                     "bitwise-identical reruns")
    return {"device": device, "torch": torch.__version__, "cpu_threads": int(torch.get_num_threads()),
            "deterministic_algorithms": "requested (warn_only)", "notes": notes}


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
def _base_run_record(config: dict, config_sha: str, roles: dict, device: str) -> dict:
    import torch

    return {
        "schema": RUN_SCHEMA,
        "status": STATUS_RUNNING,
        "runtime_eligible": False,
        "promotion": "none (development experiment; no control or promotion claim)",
        "data_roles": roles,
        "arm": config["arm"],
        "seed": config["seed"],
        "config_sha256": config_sha,
        "device": device,
        "determinism": _determinism_record(device),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "created_utc": _utc(),
        "updates_completed": 0,
        "checkpoints": [],
        "resumes": [],
        "validation_consulted_during_training": False,
    }


def fit(config_path, out_dir, *, resume: bool = False, data_root=None, loader=None,
        stop_after: int | None = None) -> dict:
    """Train one arm/seed. Returns the final ``run.json`` record.

    ``stop_after`` stops after that many total updates and marks the trial
    ``interrupted`` (drill for resumability). ``loader`` replaces
    ``fqe_sources.load_source`` (tests)."""
    import torch

    from models.focus_quality_cnn import FocusQualityEstimator
    from models.fqe_losses import NonFiniteLossError, require_finite
    from train.train_focus_quality import (
        UpdateStream,
        fqe_objective,
        require_rank_compatible,
        resolve_device,
        seed_everything,
        state_is_finite,
    )

    fs = _sources_module()
    config, input_sha = load_config_file(config_path)
    root_override = data_root
    effective_root = root_override if root_override is not None else config.get("data_root")
    run_dir = refuse_forbidden_output(out_dir, data_root=effective_root)
    files = _RunFiles(run_dir)

    existing = files.config.exists()
    if existing and not resume:
        raise ExperimentRefusal(f"{run_dir} already holds a run; pass --resume to continue it")
    earlier_attempt = None
    if not existing and run_dir.exists() and any(run_dir.iterdir()):
        # The only thing a run directory may hold before its config is the
        # record of an attempt that found the data unavailable.
        names = {child.name for child in run_dir.iterdir()}
        earlier_attempt = _read_json(files.run) if names == {"run.json"} else None
        if not earlier_attempt or earlier_attempt.get("status") != STATUS_DATA_UNAVAILABLE:
            raise ExperimentRefusal(f"{run_dir} exists and is not an fqe_dev_experiment run directory")
    previous = _read_json(files.run) if existing and files.run.exists() else None
    if previous and previous.get("status") not in RESUMABLE_STATUSES:
        raise ExperimentRefusal(
            f"run status is {previous.get('status')!r}; only {RESUMABLE_STATUSES} runs resume "
            "(a failed trial is rerun under a new run directory)"
        )

    device = resolve_device(config["device"])
    if previous and previous.get("device") and previous["device"] != device:
        raise ExperimentRefusal(f"run was started on {previous['device']!r}; refusing to resume on {device!r}")

    # Data first: a missing drive or a changed file is a recorded trial outcome.
    try:
        sources, plan, contract = load_training_sources(config, data_root=effective_root, loader=loader)
    except (fs.DataUnavailable, fs.HashMismatch, fs.ContractViolation) as exc:
        # Missing drive/file: data_unavailable (resumable once the data is
        # back). Hash mismatch or malformed arrays: data_rejected (not resumable).
        run_dir.mkdir(parents=True, exist_ok=True)
        record = previous or earlier_attempt or {
            "schema": RUN_SCHEMA, "runtime_eligible": False, "arm": config["arm"], "seed": config["seed"],
            "created_utc": _utc(), "updates_completed": 0}
        status = STATUS_DATA_UNAVAILABLE if isinstance(exc, fs.DataUnavailable) else STATUS_DATA_REJECTED
        record.update(status=status, error=f"{type(exc).__name__}: {exc}",
                      data_roles=record.get("data_roles", {}), updated_utc=_utc())
        _write_json_atomic(files.run, record)
        raise

    roles = {s.source_id: str(s.role) for s in sources}
    require_rank_compatible(
        config["lambda_rank"],
        batching=plan.batching,
        run_ids=np.arange(sum(plan.n_rows.values())),
        ranking_eligible=np.concatenate([np.asarray(s.ranking_eligible, dtype=bool)[plan.indices[s.source_id]]
                                         for s in sources]),
        roles=sorted(set(roles.values())),
    )

    stored_config = dict(config)
    stored_config.update(runtime_eligible=False, data_roles=roles,
                         contract_sha256=contract.sha256, contract_file_sha256=contract.file_sha256,
                         architecture="FocusQualityEstimator(pretrained=False)",
                         objective=("mean-only MSE for warmup_updates, then heteroscedastic_nll "
                                    "(models.fqe_losses); lambda_rank x ranknet_within_run"),
                         optimizer="torch.optim.AdamW(lr, weight_decay); clip_grad_norm_(grad_clip_norm)")
    config_bytes = _canonical(stored_config)
    config_sha = _sha256_bytes(config_bytes)
    identity_record = {
        "schema": "fqe-dev-dataset-identity-v1",
        "runtime_eligible": False,
        "data_roles": roles,
        "sources": {s.source_id: s.identity for s in sources},
        "mixture": plan.identity,
        "pool_rows": {sid: int(n) for sid, n in plan.n_rows.items()},
    }
    projection = {"sources": {s.source_id: _identity_projection(s.identity) for s in sources},
                  "mixture_sha256": plan.identity.get("mixture_sha256")}

    if existing:
        if files.config.read_bytes() != config_bytes:
            raise ExperimentRefusal("arm config differs from the run's immutable config.json; refusing to resume")
        stored_identity = _read_json(files.identity)
        stored_projection = {"sources": {sid: _identity_projection(v) for sid, v in stored_identity["sources"].items()},
                             "mixture_sha256": (stored_identity.get("mixture") or {}).get("mixture_sha256")}
        if json.loads(_canonical(stored_projection)) != json.loads(_canonical(projection)):
            raise ExperimentRefusal("dataset identity changed since the run started; refusing to resume")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        files.checkpoints.mkdir(exist_ok=True)
        _write_exclusive(files.config, config_bytes, read_only=True)
        _write_exclusive(files.config_sha, (config_sha + "  config.json\n").encode(), read_only=True)
        _write_exclusive(files.identity, _canonical(identity_record), read_only=True)

    record = previous if previous and previous.get("config_sha256") == config_sha else None
    if record is None:
        record = _base_run_record(config, config_sha, roles, device)
        record["input_config_sha256"] = input_sha
        if earlier_attempt or previous:
            record["earlier_attempts"] = [a for a in (earlier_attempt, previous) if a]
    record.update(status=STATUS_RUNNING, updated_utc=_utc(), data_root_override=root_override)
    record.pop("error", None)
    _write_json_atomic(files.run, record)

    # Model / optimizer / scheduler, then either fresh seeds or a restored state.
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        seed_everything(config["seed"])
        model = FocusQualityEstimator(pretrained=False).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        scheduler = _make_scheduler(optimizer, config)
        start = 0
        parent_sha = None
        good = [c for c in record.get("checkpoints", []) if Path(run_dir / c["path"]).exists()]
        if existing and good:
            last = good[-1]
            ckpt_path = run_dir / last["path"]
            if _sha256_file(ckpt_path) != last["sha256"]:
                raise ExperimentRefusal(f"checkpoint {ckpt_path} does not match its recorded sha256")
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if state.get("config_sha256") != config_sha:
                raise ExperimentRefusal("checkpoint belongs to a different config")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None and state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            _restore_rng(state["rng"], device)
            start = int(state["update"])
            parent_sha = last["sha256"]
            record["checkpoints"] = good
            record.setdefault("resumes", []).append({"from_update": start, "checkpoint_sha256": parent_sha,
                                                     "at_utc": _utc()})
        elif existing:
            record["checkpoints"] = []
        _truncate_curves(files, start)

        tensors = {
            s.source_id: (
                torch.as_tensor(np.ascontiguousarray(s.crops[:, None], dtype=np.float32), device=device),
                torch.as_tensor(np.asarray(s.log_target, dtype=np.float32), device=device),
            )
            for s in sources
        }
        streams = {
            sid: UpdateStream(plan.n_rows[sid], config["batch_size"], config["seed"], stream=i)
            for i, sid in enumerate(plan.source_ids)
        }
        drawn = {sid: 0 for sid in plan.source_ids}

        def batch_rows(update: int) -> dict:
            if config["sampling"] == "source_balanced_with_replacement":
                rng = np.random.default_rng([config["seed"], 7919, update])
                return plan.sample_batch(rng, config["batch_size"])
            (sid,) = plan.source_ids
            return {sid: plan.indices[sid][streams[sid].rows(update)]}

        for update in range(start):  # sample exposure of the already-completed updates
            for sid, rows in batch_rows(update).items():
                drawn[sid] += int(len(rows))

        model.train()
        t0 = time.perf_counter()
        update = start
        with files.curves.open("a", encoding="utf-8") as curves:
            while update < config["updates"]:
                if stop_after is not None and update >= stop_after:
                    raise TrialInterrupted(f"stopped after {update} updates (--stop-after drill)")
                rows = batch_rows(update)
                xb = torch.cat([tensors[sid][0][torch.as_tensor(r, device=device)] for sid, r in rows.items()])
                yb = torch.cat([tensors[sid][1][torch.as_tensor(r, device=device)] for sid, r in rows.items()])
                warm = update < config["warmup_updates"]
                lr = optimizer.param_groups[0]["lr"]
                mu, log_var = model(xb)
                loss = fqe_objective(mu, log_var, yb, mean_warmup=warm, lambda_rank=config["lambda_rank"])
                try:
                    require_finite(loss, f"training loss at update {update + 1}")
                    optimizer.zero_grad()
                    loss.backward()
                    try:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=config["grad_clip_norm"], error_if_nonfinite=True
                        )
                    except RuntimeError as exc:
                        raise NonFiniteLossError(f"non-finite gradient norm at update {update + 1}: {exc}") from exc
                    optimizer.step()
                except NonFiniteLossError as exc:
                    curves.write(json.dumps({"update": update + 1, "loss": _float_or_str(loss.item()),
                                             "objective": "mse_warmup" if warm else "heteroscedastic_nll",
                                             "finite": False, "error": str(exc)}) + "\n")
                    curves.flush()
                    raise
                if scheduler is not None:
                    scheduler.step()
                update += 1
                for sid, r in rows.items():
                    drawn[sid] += int(len(r))
                curves.write(json.dumps({
                    "update": update,
                    "loss": float(loss.item()),
                    "objective": "mse_warmup" if warm else "heteroscedastic_nll",
                    "lr": float(lr),
                    "grad_norm": float(grad_norm.item()),
                    "rows": {sid: int(len(r)) for sid, r in rows.items()},
                    "finite": True,
                }) + "\n")
                if update % config["checkpoint_every"] == 0 or update == config["updates"]:
                    curves.flush()
                    if not state_is_finite(model.state_dict()):
                        raise NonFiniteLossError(f"non-finite weights after update {update}")
                    parent_sha = _save_checkpoint(files, update, model, optimizer, scheduler, device,
                                                  config_sha, parent_sha, record)
                    record.update(updates_completed=update, updated_utc=_utc())
                    _write_json_atomic(files.run, record)

        # Final weights at the last update (no validation-based selection).
        final_state = _to_cpu(model.state_dict())
        if not state_is_finite(final_state):
            raise NonFiniteLossError("non-finite final weights")
        files.final.mkdir(exist_ok=True)
        tmp = files.final_model.with_name("model_state.pt.partial")
        torch.save(final_state, tmp)
        tmp.replace(files.final_model)
        final_sha = _sha256_file(files.final_model)
        (files.final / "model_state.sha256").write_text(final_sha + "  model_state.pt\n")
        record.update(
            status=STATUS_COMPLETED,
            updates_completed=config["updates"],
            final_model="final/model_state.pt",
            final_model_sha256=final_sha,
            final_checkpoint_update=config["updates"],
            sample_exposure={sid: {"rows_drawn": drawn[sid], "pool_rows": int(plan.n_rows[sid]),
                                   "passes": drawn[sid] / max(1, int(plan.n_rows[sid]))}
                             for sid in plan.source_ids},
            source_weight=plan.identity.get("source_weight"),
            sampling=config["sampling"],
            seconds_this_session=time.perf_counter() - t0,
            completed_utc=_utc(),
            updated_utc=_utc(),
        )
        _write_json_atomic(files.run, record)
        return record
    except NonFiniteLossError as exc:
        record.update(status=STATUS_FAILED_NONFINITE, error=str(exc), updated_utc=_utc(),
                      last_good_checkpoint=(record.get("checkpoints") or [{}])[-1].get("path"))
        _write_json_atomic(files.run, record)
        raise
    except (TrialInterrupted, KeyboardInterrupt) as exc:
        record.update(status=STATUS_INTERRUPTED, error=f"{type(exc).__name__}: {exc}", updated_utc=_utc(),
                      last_good_checkpoint=(record.get("checkpoints") or [{}])[-1].get("path"))
        _write_json_atomic(files.run, record)
        raise
    except ExperimentRefusal:
        raise
    except Exception as exc:
        record.update(status=STATUS_FAILED, error=f"{type(exc).__name__}: {exc}", updated_utc=_utc())
        _write_json_atomic(files.run, record)
        raise
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)


def _truncate_curves(files: _RunFiles, start: int) -> None:
    """Keep curve lines up to ``start``; move later lines (an interrupted
    segment that will be replayed) to ``curves.discarded.jsonl``."""
    if not files.curves.exists():
        return
    keep, drop = [], []
    for line in files.curves.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        (keep if int(json.loads(line)["update"]) <= start else drop).append(line)
    if drop:
        with files.discarded.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(drop) + "\n")
    files.curves.write_text("".join(f"{line}\n" for line in keep), encoding="utf-8")


def _save_checkpoint(files, update, model, optimizer, scheduler, device, config_sha, parent_sha, record) -> str:
    import torch

    path = files.checkpoint(update)
    state = {
        "schema": CHECKPOINT_SCHEMA,
        "update": int(update),
        "model": _to_cpu(model.state_dict()),
        "optimizer": _to_cpu(optimizer.state_dict()),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": _to_cpu(_capture_rng(device)),
        "config_sha256": config_sha,
        "parent_checkpoint_sha256": parent_sha,
        "runtime_eligible": False,
    }
    tmp = path.with_name(path.name + ".partial")
    torch.save(state, tmp)
    tmp.replace(path)
    sha = _sha256_file(path)
    record.setdefault("checkpoints", []).append({
        "update": int(update), "path": str(path.relative_to(files.dir)), "sha256": sha,
        "parent_sha256": parent_sha,
    })
    return sha


# --------------------------------------------------------------------------- #
# Completed-run access
# --------------------------------------------------------------------------- #
def load_completed_run(run_dir) -> tuple[dict, dict, Path]:
    files = _RunFiles(_resolved(run_dir))
    if not files.config.exists() or not files.run.exists():
        raise ExperimentRefusal(f"{files.dir} is not an fqe_dev_experiment run directory")
    config_bytes = files.config.read_bytes()
    recorded = files.config_sha.read_text().split()[0]
    if _sha256_bytes(config_bytes) != recorded:
        raise ExperimentRefusal("config.json does not match config.sha256")
    config = json.loads(config_bytes)
    record = _read_json(files.run)
    if record.get("status") != STATUS_COMPLETED:
        raise ExperimentRefusal(f"run status is {record.get('status')!r}; predict/calibrate need a completed run")
    if record.get("config_sha256") != recorded:
        raise ExperimentRefusal("run.json names a different config")
    if _sha256_file(files.final_model) != record.get("final_model_sha256"):
        raise ExperimentRefusal("final weights do not match the recorded sha256")
    return config, record, files.final_model


def _load_model(model_path: Path, device: str):
    import torch

    from models.focus_quality_cnn import FocusQualityEstimator

    model = FocusQualityEstimator(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def _check_training_identity(run_dir: Path, source) -> None:
    """A training source re-loaded for prediction must be the same data."""
    stored = _read_json(Path(run_dir) / "dataset_identity.json")["sources"].get(source.source_id)
    if stored is None:
        return
    if json.loads(_canonical(_identity_projection(stored))) != json.loads(_canonical(_identity_projection(source.identity))):
        raise ExperimentRefusal(f"{source.source_id}: data identity differs from the one the model was trained on")


def _refuse_test(partitions) -> None:
    fs = _sources_module()
    for name in partitions:
        if str(name) in ("test", "test_sealed"):
            raise fs.SealedPartitionError(
                "the test partition is sealed; development commands never predict it (use final-test "
                f"with a frozen candidate receipt and {FINAL_TEST_FLAG})"
            )


def _prepare_output_dir(path: Path, overwrite: bool, data_root=None) -> Path:
    path = refuse_forbidden_output(path, data_root=data_root)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise ExperimentRefusal(f"{path} already holds results; pass --overwrite to replace them")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prediction_rows(source, mu: np.ndarray, log_var: np.ndarray) -> tuple[list[str], list[list]]:
    clamped = np.clip(log_var, -8.0, 8.0)
    sigma = np.exp(0.5 * clamped)
    pred = np.exp(mu)
    finite = np.isfinite(mu) & np.isfinite(log_var)
    secondary = sorted(source.secondary)
    strata = sorted(source.strata)
    header = (["sample_uid", "group_uid", "frame_uid", "source_id", "role", "partition", "target_hfr_px",
               "log_target", "pred_hfr_px", "mu", "log_var", "sigma", "status", "star_snr"]
              + [f"secondary_{name}" for name in secondary] + [f"stratum_{name}" for name in strata])
    rows = []
    for i in range(len(source.target_hfr_px)):
        rows.append(
            [str(source.sample_uid[i]), str(source.group_uid[i]), str(source.frame_uid[i]), source.source_id,
             str(source.role), str(source.partition[i]), float(source.target_hfr_px[i]),
             float(source.log_target[i]), float(pred[i]), float(mu[i]), float(log_var[i]), float(sigma[i]),
             "ok" if finite[i] else "nonfinite_prediction", float(source.star_snr[i])]
            + [float(source.secondary[name][i]) for name in secondary]
            + [str(source.strata[name][i]) for name in strata]
        )
    return header, rows


def _write_csv(path: Path, header, rows) -> None:
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #
def predict(run_dir, *, partitions=("val",), sources=None, device: str = "cpu", data_root=None,
            out_dir=None, overwrite: bool = False, loader=None) -> dict:
    """Per-row development predictions. ``test`` is refused before any load."""
    partitions = tuple(dict.fromkeys(partitions))
    _refuse_test(partitions)
    unknown = [p for p in partitions if p not in PREDICT_PARTITIONS]
    if unknown:
        raise ExperimentRefusal(f"unknown partitions {unknown}; allowed {PREDICT_PARTITIONS}")
    from train.calibrate_uncertainty import predict_mu_log_var
    from train.train_focus_quality import resolve_device

    fs = _sources_module()
    config, record, model_path = load_completed_run(run_dir)
    run_dir = _resolved(run_dir)
    root = data_root if data_root is not None else config.get("data_root")
    out = _prepare_output_dir(Path(out_dir) if out_dir else run_dir / "predictions", overwrite, root)
    device = resolve_device(device)
    model = _load_model(model_path, device)
    contract = _load_contract(config)
    loader = loader or fs.load_source
    source_ids = sorted(set(sources or config["eval_sources"]))

    outputs, roles, identities = [], {}, {}
    for sid in source_ids:
        source = loader(sid, partitions=partitions, contract=contract, data_root=root)
        _check_training_identity(run_dir, source)
        roles[sid] = str(source.role)
        identities[sid] = _identity_projection(source.identity)
        calibration = (source.identity or {}).get("calibration") or {}
        for partition in partitions:
            index = np.flatnonzero(np.asarray(source.partition).astype(str) == partition)
            entry = {"source_id": sid, "role": roles[sid], "partition": partition, "rows": int(index.size)}
            if index.size == 0:
                entry["status"] = "unavailable"
                entry["reason"] = (calibration.get("reason") if partition == "calibration"
                                   and calibration.get("available") is False else "no rows in this partition")
                outputs.append(entry)
                continue
            part = source.subset(index)
            t0 = time.perf_counter()
            mu, log_var = predict_mu_log_var(model, part.crops, device)
            elapsed = time.perf_counter() - t0
            header, rows = _prediction_rows(part, mu, log_var)
            path = out / f"{sid}__{partition}.csv"
            _write_csv(path, header, rows)
            entry.update(status="ok", file=path.name, sha256=_sha256_file(path),
                         finite_rows=int(np.sum(np.isfinite(mu) & np.isfinite(log_var))),
                         inference_seconds=elapsed, ms_per_crop=1000.0 * elapsed / index.size)
            outputs.append(entry)
    manifest = {
        "schema": PREDICTIONS_SCHEMA,
        "runtime_eligible": False,
        "data_roles": roles,
        "training_data_roles": record.get("data_roles"),
        "run_config_sha256": record["config_sha256"],
        "model_sha256": record["final_model_sha256"],
        "arm": config["arm"],
        "seed": config["seed"],
        "device": device,
        "partitions": list(partitions),
        "test_partition": "sealed; never predicted by this command",
        "columns": {"mu": "E[ln HFR_px] (raw head)", "log_var": "raw head output",
                    "sigma": "exp(0.5*clamp(log_var,-8,8)) in ln-HFR units, uncalibrated",
                    "pred_hfr_px": "exp(mu)", "target_hfr_px": "contract target field"},
        "source_identity": identities,
        "outputs": outputs,
        "created_utc": _utc(),
    }
    _write_json_atomic(out / "manifest.json", manifest)
    return manifest


# --------------------------------------------------------------------------- #
# calibrate
# --------------------------------------------------------------------------- #
def calibrate(run_dir, *, sources=None, device: str = "cpu", data_root=None, out_dir=None,
              overwrite: bool = False, loader=None) -> dict:
    """Fit the quantile interval scale on each source's calibration partition and
    assess it on that source's val partition. Never touches test."""
    from train.calibrate_uncertainty import NLL_CONVENTION, calibrate_loaded_sources
    from train.train_focus_quality import resolve_device

    fs = _sources_module()
    config, record, model_path = load_completed_run(run_dir)
    run_dir = _resolved(run_dir)
    root = data_root if data_root is not None else config.get("data_root")
    out = _prepare_output_dir(Path(out_dir) if out_dir else run_dir / "calibration", overwrite, root)
    device = resolve_device(device)
    model = _load_model(model_path, device)
    contract = _load_contract(config)
    loader = loader or fs.load_source
    source_ids = sorted(set(sources or config["eval_sources"]))

    calibration_sources, evaluation_sources, unavailable, roles, identities = {}, {}, {}, {}, {}
    for sid in source_ids:
        source = loader(sid, partitions=("calibration", "val"), contract=contract, data_root=root)
        _check_training_identity(run_dir, source)
        roles[sid] = str(source.role)
        identities[sid] = _identity_projection(source.identity)
        declared = (source.identity or {}).get("calibration") or {}
        partition = np.asarray(source.partition).astype(str)
        if declared.get("available") is False or not np.any(partition == "calibration"):
            unavailable[sid] = str(declared.get("reason") or "no calibration rows for this source")
            continue
        calibration_sources[sid] = source.subset(partition == "calibration")
        evaluation_sources[sid] = {"val": source.subset(partition == "val")}
    per_source = calibrate_loaded_sources(model, calibration_sources, evaluation_sources, device,
                                          unavailable=unavailable)
    result = {
        "schema": CALIBRATION_SCHEMA,
        "runtime_eligible": False,
        "development_only": True,
        "test_reported": False,
        "data_roles": roles,
        "training_data_roles": record.get("data_roles"),
        "run_config_sha256": record["config_sha256"],
        "model_sha256": record["final_model_sha256"],
        "device": device,
        "fit_partition": "calibration",
        "evaluated_partition": "val",
        "units": "ln HFR_px residuals; sigma_scale multiplies exp(0.5*clamp(log_var))",
        "nll_convention": NLL_CONVENTION,
        "source_identity": identities,
        "sources": per_source,
        "created_utc": _utc(),
    }
    _write_json_atomic(out / "calibration.json", result)
    return result


# --------------------------------------------------------------------------- #
# final-test (sealed; explicit, once, receipt-bound)
# --------------------------------------------------------------------------- #
def _point_metrics(target_hfr: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    from train.calibrate_uncertainty import NLL_CONVENTION, gaussian_nll

    finite = np.isfinite(mu) & np.isfinite(log_var)
    out = {"rows": int(target_hfr.size), "finite_rows": int(finite.sum()),
           "coverage": float(finite.mean()) if target_hfr.size else 0.0, "nll_convention": NLL_CONVENTION}
    if not finite.any():
        return out
    t, m, lv = target_hfr[finite].astype(np.float64), mu[finite], np.clip(log_var[finite], -8.0, 8.0)
    pred = np.exp(m)
    log_t = np.log(t)
    sigma = np.exp(0.5 * lv)
    z = np.abs(log_t - m) / sigma
    out.update(
        hfr_mae_px=float(np.mean(np.abs(pred - t))),
        log_hfr_mae=float(np.mean(np.abs(m - log_t))),
        signed_bias_px=float(np.mean(pred - t)),
        median_relative_error=float(np.median(np.abs(pred - t) / t)),
        raw_interval_coverage_68=float(np.mean(z <= 1.0)),
        raw_interval_coverage_95=float(np.mean(z <= 1.95996398)),
        # Same definition as the development calibration's nll_before/nll_after
        # (calibrate_uncertainty.gaussian_nll), so dev and final values compare.
        raw_nll=gaussian_nll(m, sigma, log_t),
    )
    return out


def final_test(run_dir, *, source_id: str, receipt_path, acknowledge: bool, device: str = "cpu",
               data_root=None) -> dict:
    """Evaluate the frozen candidate on one source's sealed test partition, once.

    The receipt must name this run's final weights (``candidate_sha256``) and
    frozen config (``protocol_sha256``) and carry ``fqe_sources``'
    acknowledgement text; the loader refuses a second use and any contract
    that does not enable the final test (the tracked v1 contract does not)."""
    fs = _sources_module()
    if not acknowledge:
        raise fs.SealedPartitionError(f"final-test requires {FINAL_TEST_FLAG}")
    from train.calibrate_uncertainty import predict_mu_log_var
    from train.train_focus_quality import resolve_device

    config, record, model_path = load_completed_run(run_dir)
    run_dir = _resolved(run_dir)
    receipt = _read_json(receipt_path)
    if receipt.get("candidate_sha256") != record["final_model_sha256"]:
        raise fs.SealedPartitionError("receipt candidate_sha256 is not this run's final model sha256")
    if receipt.get("protocol_sha256") != record["config_sha256"]:
        raise fs.SealedPartitionError("receipt protocol_sha256 is not this run's frozen config sha256")
    root = data_root if data_root is not None else config.get("data_root")
    out = refuse_forbidden_output(run_dir / "final_test" / source_id, data_root=root)
    contract = _load_contract(config)
    # Everything that can fail without the sealed rows is proven first: the
    # device, the weights, and one forward pass on a dummy crop. The loader
    # consumes the one-shot receipt, so a setup error after it would burn the test.
    device = resolve_device(device)
    model = _load_model(model_path, device)
    probe_mu, probe_lv = predict_mu_log_var(model, np.zeros((1, 64, 64), dtype=np.float32), device)
    if not (np.all(np.isfinite(probe_mu)) and np.all(np.isfinite(probe_lv))):
        raise ExperimentRefusal("the candidate produces non-finite outputs on a dummy crop; final test not started")
    arrays = fs.load_final_test(source_id, receipt, receipt_dir=out, contract=contract, data_root=root)
    base = {
        "schema": FINAL_TEST_SCHEMA,
        "runtime_eligible": False,
        "data_roles": {source_id: str(arrays.role)},
        "source_id": source_id,
        "candidate_sha256": record["final_model_sha256"],
        "protocol_sha256": record["config_sha256"],
        "receipt": receipt,
        "loader_receipt": arrays.identity.get("final_test_receipt"),
        "device": device,
    }
    receipt_file = out / "evaluation_receipt.json"
    try:
        mu, log_var = predict_mu_log_var(model, arrays.crops, device)
        header, rows = _prediction_rows(arrays, mu, log_var)
        csv_path = out / "predictions.csv"
        _write_csv(csv_path, header, rows)
        result = dict(base, status=STATUS_COMPLETED, predictions_file=csv_path.name,
                      predictions_sha256=_sha256_file(csv_path),
                      metrics=_point_metrics(np.asarray(arrays.target_hfr_px), mu, log_var),
                      evaluated_utc=_utc())
        _write_exclusive(receipt_file, _canonical(result), read_only=True)
    except BaseException as exc:
        # The sealed rows were consumed: record why no result exists (never
        # replacing a receipt that was already written).
        if not receipt_file.exists():
            failure = dict(base, status=STATUS_FAILED, error=f"{type(exc).__name__}: {exc}",
                           note="the sealed test was loaded under the receipt but not evaluated; it is consumed",
                           failed_utc=_utc())
            try:
                _write_exclusive(receipt_file, _canonical(failure), read_only=True)
            except OSError:
                pass
        raise
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="train one arm/seed for a fixed update budget")
    p_fit.add_argument("--config", required=True, help="arm config JSON")
    p_fit.add_argument("--out", required=True, help="run directory (outside <repo>/models and the data root)")
    p_fit.add_argument("--resume", action="store_true", help="continue an interrupted run from its last checkpoint")
    p_fit.add_argument("--data-root", default=None)
    p_fit.add_argument("--stop-after", type=int, default=None,
                       help="interruption drill: stop after N total updates and mark the run interrupted")

    p_pred = sub.add_parser("predict", help="per-row predictions on development partitions")
    p_pred.add_argument("--run", required=True)
    p_pred.add_argument("--partition", action="append", default=None,
                        help=f"development partition (repeatable; {', '.join(PREDICT_PARTITIONS)}); test is refused")
    p_pred.add_argument("--source", action="append", default=None, help="source id (default: config eval_sources)")
    p_pred.add_argument("--device", default="cpu")
    p_pred.add_argument("--data-root", default=None)
    p_pred.add_argument("--out", default=None)
    p_pred.add_argument("--overwrite", action="store_true")

    p_cal = sub.add_parser("calibrate", help="fit the interval scale on calibration rows, assess on val")
    p_cal.add_argument("--run", required=True)
    p_cal.add_argument("--source", action="append", default=None)
    p_cal.add_argument("--device", default="cpu")
    p_cal.add_argument("--data-root", default=None)
    p_cal.add_argument("--out", default=None)
    p_cal.add_argument("--overwrite", action="store_true")

    p_final = sub.add_parser("final-test", help="SEALED: evaluate a frozen candidate on one test partition, once")
    p_final.add_argument("--run", required=True)
    p_final.add_argument("--source", required=True)
    p_final.add_argument("--receipt", required=True, help="frozen candidate/protocol receipt JSON")
    p_final.add_argument(FINAL_TEST_FLAG, dest="acknowledge", action="store_true")
    p_final.add_argument("--device", default="cpu")
    p_final.add_argument("--data-root", default=None)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    fs = _sources_module()
    from models.fqe_losses import NonFiniteLossError

    try:
        if args.command == "fit":
            result = fit(args.config, args.out, resume=args.resume, data_root=args.data_root,
                         stop_after=args.stop_after)
        elif args.command == "predict":
            result = predict(args.run, partitions=tuple(args.partition or ("val",)), sources=args.source,
                             device=args.device, data_root=args.data_root, out_dir=args.out,
                             overwrite=args.overwrite)
        elif args.command == "calibrate":
            result = calibrate(args.run, sources=args.source, device=args.device, data_root=args.data_root,
                               out_dir=args.out, overwrite=args.overwrite)
        else:
            result = final_test(args.run, source_id=args.source, receipt_path=args.receipt,
                                acknowledge=args.acknowledge, device=args.device, data_root=args.data_root)
    except (fs.DataUnavailable, fs.HashMismatch) as exc:
        print(f"data unavailable: {exc}", file=sys.stderr)
        return EXIT_DATA
    except NonFiniteLossError as exc:
        print(f"failed_nonfinite: {exc}", file=sys.stderr)
        return EXIT_NONFINITE
    except (TrialInterrupted, KeyboardInterrupt) as exc:
        print(f"interrupted: {exc}", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (ExperimentRefusal, fs.SealedPartitionError, fs.ContractViolation, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    summary = {key: result.get(key) for key in ("status", "runtime_eligible", "data_roles", "final_model_sha256",
                                                "config_sha256", "outputs", "sources", "metrics")
               if key in result}
    print(json.dumps(summary, indent=2, default=_json_default))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
