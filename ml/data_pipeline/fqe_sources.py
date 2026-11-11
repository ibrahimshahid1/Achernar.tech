"""
fqe_sources — source-aware, test-sealed loading for the FQE public-data experiment.

One loader for every FQE training/evaluation source the experiment admits,
driven by a tracked contract (``fqe_source_contracts_v1.json``) that pins the
sha256 of every file, the target definition, the group/partition rules and the
data role of each source. It is the only development path to the public
archive NPZs; reading them directly skips every check below.

What it guarantees
------------------
* **Mount and identity.** The data root resolves (``ml/data/public``,
  ``$ACHERNAR_PUBLIC_DATA_ROOT`` or an explicit ``data_root``) and every pinned
  file exists and matches its sha256 before any array is read
  (:class:`DataUnavailable`, :class:`HashMismatch`). :func:`verify_source`
  repeats this check cheaply, e.g. before each training trial.
* **Frozen partitions only.** Splits come from the adjacent frozen registry via
  ``split_registry.load_registered_splits``; ``splits_for_dataset`` /
  ``create_splits`` are never called, so nothing is ever written next to an
  NPZ. The registry must be bound to the pinned NPZ, cover every row exactly
  once, and agree with the stored ``split`` column (registry key ``val`` =
  column ``validation``).
* **Sealed test.** Development calls return only ``train_fit``,
  ``calibration`` and ``val`` rows. Asking for ``test`` raises
  :class:`SealedPartitionError`. For public and fixture sources, test rows
  are counted and their membership hashed from identity columns; their value
  columns (crops, targets, SNR) are streamed past without being decoded, and
  their ``records.jsonl`` lines are parsed only for the join/ancestor checks
  (no stratum or join field of a test row is kept). Local verified runs are
  the exception: they are delegated whole to ``load_qualified_dataset``,
  which decodes every row; the loader then returns non-test rows only and
  derives nothing (ranking eligibility included) from test rows.
  :func:`load_final_test` is the only way to obtain test rows and needs a
  contract that enables it with a JSON ``true`` (the tracked contract does
  not), a receipt with an acknowledgement, and an unused entry in the
  final-test ledger: an NPZ's sealed test is consumed once per ledger,
  whatever output directory the caller picks.
* **No writes to protected places.** Nothing here writes under a data root
  (``ml/data`` and its links, ``$ACHERNAR_PUBLIC_DATA_ROOT``, an explicit
  ``data_root``) or the production model registry (``<repo>/models``);
  :func:`write_fixture_source` refuses those locations and never overwrites.
* **Namespaced identity.** ``sample_uid``/``group_uid``/``frame_uid`` are
  ``"<source_id>:<key>"`` strings. Integer ``run_id`` values (which collide
  across public sources) are never exposed. :func:`make_mixture` asserts that
  sources share no sample, group or frame identity.
* **Derivative isolation.** No group, frame or ancestor (records.jsonl
  ``ancestor_id``) may span two partitions, including the calibration split
  carved out of train.
* **Roles stay distinct.** ``synthetic_reference``,
  ``archive_measurement_reference``, ``fixture_only`` and
  ``local_verified_run`` keep their role, batching (``pointwise`` vs
  ``grouped_run``) and ranking eligibility. Local runs are delegated unchanged
  to ``build_focus_quality_dataset.load_qualified_dataset``. The loader never
  grants runtime eligibility.

Nothing in this module computes a statistic of a target. Integrity checks on
loaded rows (finite, positive, stored log consistent, aliased fields equal)
raise :class:`ContractViolation` and report counts only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import zipfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.lib import format as npy_format

ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from data_pipeline.split_registry import load_registered_splits  # noqa: E402

CONTRACT_SCHEMA = "fqe-source-contracts-v1"
ARRAYS_SCHEMA = "fqe-source-arrays-v1"
RECEIPT_SCHEMA = "fqe-final-test-receipt-v1"
LEDGER_SCHEMA = "fqe-final-test-ledger-v1"
DEFAULT_CONTRACT_PATH = Path(__file__).with_name("fqe_source_contracts_v1.json")
DEFAULT_DATA_ROOT = ML_ROOT / "data" / "public"
DATA_ROOT_ENV = "ACHERNAR_PUBLIC_DATA_ROOT"
REPO_DATA_DIR = ML_ROOT / "data"
MODEL_REGISTRY_ROOT = REPO_ROOT / "models"
FINAL_TEST_LEDGER_ENV = "ACHERNAR_FQE_FINAL_TEST_LEDGER"
FINAL_TEST_LEDGER_DIRNAME = "fqe_final_test_ledger"

DEV_PARTITIONS = ("train_fit", "calibration", "val")
SEALED_PARTITION = "test"
REGISTRY_PARTITIONS = ("train", "val", "test")
DEFAULT_SPLIT_COLUMN_LABELS = {"train": "train", "val": "validation", "test": "test"}
ROLES = ("synthetic_reference", "archive_measurement_reference", "local_verified_run", "fixture_only")
SEALED_ROLES = ("diagnostic_sealed",)
PUBLIC_ROLES = ("synthetic_reference", "archive_measurement_reference")
BATCHING = ("pointwise", "grouped_run")
CROP_SHAPE = (64, 64)
FINAL_TEST_ACKNOWLEDGEMENT = (
    "I consume the sealed final test partition of {source_id} once, with the frozen "
    "candidate and protocol named in this receipt."
)
RECEIPT_REQUIRED = ("candidate_id", "candidate_sha256", "protocol_id", "protocol_sha256",
                    "authorized_by", "acknowledgement")
_HEX64 = set("0123456789abcdef")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class FqeSourceError(RuntimeError):
    """Base class of every loader refusal."""


class DataUnavailable(FqeSourceError):
    """The data root, a source directory or a pinned file is missing."""


class HashMismatch(FqeSourceError):
    """A file's sha256 differs from its pin, or a registry is bound elsewhere."""


class ContractViolation(FqeSourceError):
    """The data or the request contradicts the source contract."""


class SealedPartitionError(FqeSourceError):
    """A sealed partition or a sealed source was requested in development mode."""


class CalibrationUnavailable(FqeSourceError):
    """The source has no declared, sufficiently grouped calibration partition."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _canonical_sha256(obj) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def _is_hex64(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def _membership_sha256(uids) -> str:
    return hashlib.sha256("\n".join(sorted(str(u) for u in uids)).encode("utf-8")).hexdigest()


def _namespace(source_id: str, values) -> np.ndarray:
    return np.asarray([f"{source_id}:{v}" for v in np.asarray(values).astype(str).tolist()], dtype=np.str_)


def _dotted(record: Mapping, path: str):
    value = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _column(values: list) -> np.ndarray:
    """Typed array from JSON values (bool, int, float, number pairs or str)."""
    present = [v for v in values if v is not None]
    if present and all(isinstance(v, bool) for v in present) and len(present) == len(values):
        return np.asarray(values, dtype=bool)
    if present and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        if len(present) == len(values) and all(isinstance(v, int) for v in present):
            return np.asarray(values, dtype=np.int64)
        return np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    if present and all(isinstance(v, (list, tuple)) for v in present) and len(present) == len(values):
        widths = {len(v) for v in present}
        if len(widths) == 1 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                    for v in present for x in v):
            ints = all(isinstance(x, int) for v in present for x in v)
            return np.asarray(values, dtype=np.int64 if ints else np.float64)
    return np.asarray(["" if v is None else str(v) for v in values], dtype=np.str_)


def _relative_name_ok(name: str) -> bool:
    p = Path(name)
    return bool(name) and not p.is_absolute() and ".." not in p.parts


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _qualified_manifest_name(npz_name: str) -> str:
    """The manifest ``load_qualified_dataset`` opens next to a local dataset (``<stem>.manifest.json``)."""
    return PurePosixPath(npz_name).with_suffix(".manifest.json").as_posix()


def _protected_roots(data_root=None) -> list[tuple[Path, str]]:
    """Directories this module never writes into, in absolute and resolved form.

    Read at call time (tests substitute stand-ins): an explicit data root, the
    repository link ``ml/data/public``, ``$ACHERNAR_PUBLIC_DATA_ROOT``, the
    repository data directory ``ml/data`` with the target of every link in it,
    and the production model registry ``<repo>/models``.
    """
    named: list[tuple[Path, str]] = []
    if data_root is not None:
        named.append((Path(data_root), "the data root"))
    named.append((Path(DEFAULT_DATA_ROOT), "the public data root (ml/data/public)"))
    env_root = os.environ.get(DATA_ROOT_ENV)
    if env_root:
        named.append((Path(env_root), f"the public data root (${DATA_ROOT_ENV})"))
    data_dir = Path(REPO_DATA_DIR)
    named.append((data_dir, "the repository data directory (ml/data)"))
    if data_dir.is_dir():
        for child in sorted(data_dir.iterdir()):
            if child.is_symlink():
                named.append((child, f"a data root linked from ml/data ({child.name})"))
    named.append((Path(MODEL_REGISTRY_ROOT), "the production model registry"))
    roots: list[tuple[Path, str]] = []
    for path, why in named:
        for candidate in (Path(os.path.abspath(path)), path.resolve()):
            if (candidate, why) not in roots:
                roots.append((candidate, why))
    return roots


def _casefold_within(target: Path, root: Path) -> bool:
    """``target`` equals or lies under ``root``, comparing components case-folded."""
    t = [part.casefold() for part in target.parts]
    r = [part.casefold() for part in root.parts]
    return len(t) >= len(r) and t[: len(r)] == r


def _existing_ancestor_ids(path: Path) -> set:
    """``(st_dev, st_ino)`` of ``path`` and of every existing ancestor."""
    ids = set()
    for candidate in (path, *path.parents):
        try:
            st = os.stat(candidate)
        except OSError:
            continue
        ids.add((st.st_dev, st.st_ino))
    return ids


def _refuse_protected(path, what: str, data_root=None) -> Path:
    """``path`` resolved, or :class:`ContractViolation` when it lies inside a protected root.

    String prefixes are not enough on macOS: APFS is normally case-insensitive
    and ``Path.resolve`` does not canonicalise case, so ``.../MODELS/x`` names
    the same directory as ``.../models/x``. A path is refused when one of its
    existing ancestors is a protected root (same device and inode), or, for
    roots that do not exist now (an unmounted drive), when its components
    match the root's case-insensitively (same rule as
    ``train.fqe_dev_experiment.refuse_forbidden_output``)."""
    target = Path(path).resolve()
    ancestor_ids = _existing_ancestor_ids(target)
    for root, why in _protected_roots(data_root):
        spellings = {Path(root).resolve(), Path(os.path.abspath(os.path.expanduser(str(root))))}
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
            raise ContractViolation(f"{what} {target} lies inside {why}; refusing to write there")
    return target


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
def _validate_entry(source_id: str, entry: Mapping) -> None:
    if not isinstance(entry, Mapping):
        raise ContractViolation(f"{source_id}: contract entry must be an object")
    role = entry.get("role")
    if role in SEALED_ROLES:
        return
    if role not in ROLES:
        raise ContractViolation(f"{source_id}: unknown role {role!r}; expected one of {ROLES + SEALED_ROLES}")
    required = ("directory", "npz", "registry", "files", "target_field", "group_field",
                "batching", "target_definition_id")
    missing = [key for key in required if key not in entry]
    if missing:
        raise ContractViolation(f"{source_id}: contract entry lacks {missing}")
    if not _relative_name_ok(str(entry["directory"])):
        raise ContractViolation(f"{source_id}: directory must be relative to the data root")
    files = entry["files"]
    if not isinstance(files, Mapping) or not files:
        raise ContractViolation(f"{source_id}: files must pin at least one sha256")
    for name, digest in files.items():
        if not _relative_name_ok(name):
            raise ContractViolation(f"{source_id}: pinned file {name!r} must be a relative path")
        if not _is_hex64(digest):
            raise ContractViolation(f"{source_id}: pin for {name!r} is not a lowercase sha256")
    for key in ("npz", "registry", "records", "manifest"):
        if entry.get(key) is not None and entry[key] not in files:
            raise ContractViolation(f"{source_id}: {key} file {entry[key]!r} is not pinned")
    if role == "local_verified_run":
        # load_qualified_dataset opens <stem>.manifest.json beside the NPZ: it must be pinned too.
        manifest_name = _qualified_manifest_name(str(entry["npz"]))
        if manifest_name not in files:
            raise ContractViolation(
                f"{source_id}: local_verified_run must pin the qualified-dataset manifest {manifest_name!r} "
                "that load_qualified_dataset opens"
            )
    final = entry.get("final_test")
    if final is not None:
        if not isinstance(final, Mapping):
            raise ContractViolation(f"{source_id}: final_test must be an object")
        if "enabled" in final and not isinstance(final["enabled"], bool):
            raise ContractViolation(
                f"{source_id}: final_test.enabled must be a JSON boolean, not {final['enabled']!r}"
            )
    if entry["batching"] not in BATCHING:
        raise ContractViolation(f"{source_id}: batching must be one of {BATCHING}")
    expected_batching = "grouped_run" if role == "local_verified_run" else "pointwise"
    if entry["batching"] != expected_batching:
        raise ContractViolation(f"{source_id}: role {role} requires batching {expected_batching!r}")
    rule = (entry.get("calibration") or {"rule": "unavailable"}).get("rule")
    if rule not in ("sha256_lowest_fraction", "unavailable"):
        raise ContractViolation(f"{source_id}: unknown calibration rule {rule!r}")
    if rule == "sha256_lowest_fraction":
        fraction = entry["calibration"].get("fraction")
        if not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
            raise ContractViolation(f"{source_id}: calibration fraction must lie in (0, 1)")


def _validate_contract(data: Mapping) -> None:
    if not isinstance(data, Mapping) or data.get("schema") != CONTRACT_SCHEMA:
        raise ContractViolation(f"source contract schema must be {CONTRACT_SCHEMA!r}")
    sources = data.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ContractViolation("source contract declares no sources")
    for source_id, entry in sources.items():
        if not source_id or ":" in source_id:
            raise ContractViolation(f"invalid source id {source_id!r} (non-empty, no ':')")
        _validate_entry(source_id, entry)
    for mixture in data.get("mixtures", []) or []:
        unknown = [s for s in mixture.get("sources", []) if s not in sources]
        if unknown or len(mixture.get("sources", [])) < 2:
            raise ContractViolation(f"mixture {mixture.get('id')!r} names unknown/too few sources {unknown}")
        if not mixture.get("common_reference_policy"):
            raise ContractViolation(f"mixture {mixture.get('id')!r} declares no common_reference_policy")


@dataclass(frozen=True)
class SourcesContract:
    """A validated, immutable source contract.

    ``sha256`` hashes the canonical JSON content (so an in-memory fixture
    contract has an identity too); ``file_sha256`` hashes the file bytes when
    the contract was read from disk. ``source_file`` is that file's resolved
    location (runtime only, never recorded); it places the default final-test
    ledger. Contracts derived with ``with_source``/``with_mixture`` have none.
    """

    data: dict = field(repr=False)
    sha256: str
    file_sha256: str | None = None
    path: str | None = None
    source_file: str | None = field(default=None, repr=False, compare=False)

    @property
    def sources(self) -> dict:
        return self.data["sources"]

    @property
    def mixtures(self) -> list:
        return list(self.data.get("mixtures", []) or [])

    @property
    def split_column_labels(self) -> dict:
        return dict(self.data.get("split_column_labels") or DEFAULT_SPLIT_COLUMN_LABELS)

    @property
    def experiment_contract_sha256(self) -> str | None:
        return (self.data.get("derived_from") or {}).get("experiment_contract_sha256")

    def source(self, source_id: str) -> dict:
        if source_id not in self.sources:
            raise ContractViolation(
                f"source {source_id!r} is not declared by the contract; declared: {sorted(self.sources)}"
            )
        return copy.deepcopy(self.sources[source_id])

    def with_source(self, source_id: str, entry: Mapping) -> "SourcesContract":
        """A new contract with ``source_id`` added or replaced (fixtures, local runs)."""
        data = copy.deepcopy(self.data)
        data["sources"][source_id] = copy.deepcopy(dict(entry))
        return _contract_from_data(data)

    def with_mixture(self, mixture_id: str, source_ids: Sequence[str], common_reference_policy: str,
                     **extra) -> "SourcesContract":
        """A new contract that declares a mixture of differently defined targets."""
        data = copy.deepcopy(self.data)
        mixture = {"id": mixture_id, "sources": sorted(source_ids),
                   "common_reference_policy": common_reference_policy, **extra}
        data["mixtures"] = [m for m in data.get("mixtures", []) or [] if m.get("id") != mixture_id] + [mixture]
        return _contract_from_data(data)

    def mixture_for(self, source_ids: Iterable[str]) -> dict | None:
        wanted = sorted(set(source_ids))
        for mixture in self.mixtures:
            if sorted(mixture.get("sources", [])) == wanted:
                return copy.deepcopy(mixture)
        return None


def _contract_from_data(data: Mapping, *, file_sha256: str | None = None, path: str | None = None,
                        source_file: str | None = None) -> SourcesContract:
    data = copy.deepcopy(dict(data))
    _validate_contract(data)
    return SourcesContract(data=data, sha256=_canonical_sha256(data), file_sha256=file_sha256, path=path,
                           source_file=source_file)


def empty_contract() -> SourcesContract:
    """A contract skeleton without sources, for fixtures (add with ``with_source``)."""
    data = {"schema": CONTRACT_SCHEMA, "split_column_labels": dict(DEFAULT_SPLIT_COLUMN_LABELS),
            "sources": {}, "mixtures": [], "runtime_eligible": False}
    # Bypass the "no sources" check: a skeleton is only ever extended.
    return SourcesContract(data=data, sha256=_canonical_sha256(data))


def load_contract(path: str | os.PathLike | None = None) -> SourcesContract:
    """Read and validate a source contract (default: the tracked v1 contract)."""
    contract_path = Path(path) if path is not None else DEFAULT_CONTRACT_PATH
    if not contract_path.is_file():
        raise DataUnavailable(f"source contract not found: {contract_path}")
    raw = contract_path.read_bytes()
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ContractViolation(f"source contract {contract_path} is not valid JSON: {exc}") from exc
    return _contract_from_data(data, file_sha256=hashlib.sha256(raw).hexdigest(),
                               path=_display_path(contract_path), source_file=str(contract_path.resolve()))


# --------------------------------------------------------------------------- #
# Data root and file verification
# --------------------------------------------------------------------------- #
def resolve_data_root(data_root: str | os.PathLike | None = None) -> Path:
    """Resolve the public data root and prove it is reachable.

    Precedence: explicit ``data_root``, then ``$ACHERNAR_PUBLIC_DATA_ROOT``,
    then the repository link ``ml/data/public``. A link whose target is absent
    (drive not mounted) raises :class:`DataUnavailable`; nothing is created.
    """
    if data_root is not None:
        candidate, origin = Path(data_root), "argument"
    elif os.environ.get(DATA_ROOT_ENV):
        candidate, origin = Path(os.environ[DATA_ROOT_ENV]), f"${DATA_ROOT_ENV}"
    else:
        candidate, origin = DEFAULT_DATA_ROOT, "ml/data/public"
    if candidate.is_symlink() and not candidate.exists():
        raise DataUnavailable(
            f"data root {candidate} ({origin}) is a link to {os.readlink(candidate)!r}, which does "
            "not exist: is the data drive mounted? Nothing was created."
        )
    if not candidate.exists():
        raise DataUnavailable(f"data root {candidate} ({origin}) does not exist; is the data drive mounted?")
    if not candidate.is_dir():
        raise DataUnavailable(f"data root {candidate} ({origin}) is not a directory")
    return candidate.resolve()


def _source_dir(root: Path, source_id: str, entry: Mapping) -> Path:
    directory = root / entry["directory"]
    if not directory.is_dir():
        raise DataUnavailable(f"{source_id}: source directory {directory} is missing under the data root")
    return directory


def _verify_files(source_id: str, entry: Mapping, directory: Path) -> dict[str, str]:
    verified = {}
    for name, expected in sorted(entry["files"].items()):
        path = directory / name
        if not path.is_file():
            raise DataUnavailable(f"{source_id}: pinned file {name} is missing ({path})")
        actual = _sha256_file(path)
        if actual != expected:
            raise HashMismatch(f"{source_id}: {name} sha256 {actual} != pinned {expected}")
        verified[name] = actual
    return verified


def _check_manifest(source_id: str, entry: Mapping, directory: Path) -> dict:
    name = entry.get("manifest")
    if not name:
        return {}
    try:
        manifest = json.loads((directory / name).read_text())
    except ValueError as exc:
        raise ContractViolation(f"{source_id}: {name} is not valid JSON: {exc}") from exc
    dataset_id = entry.get("dataset_id")
    if dataset_id is not None and manifest.get("dataset_id") != dataset_id:
        raise ContractViolation(
            f"{source_id}: manifest dataset_id {manifest.get('dataset_id')!r} != contract {dataset_id!r}"
        )
    for artifact in manifest.get("artifacts", []) or []:
        pinned = entry["files"].get(artifact.get("path"))
        if pinned is not None and artifact.get("sha256") != pinned:
            raise HashMismatch(
                f"{source_id}: manifest records {artifact.get('path')} as {artifact.get('sha256')}, pinned {pinned}"
            )
    for key, file_name in (entry.get("manifest_bindings") or {}).items():
        if manifest.get(key) != entry["files"].get(file_name):
            raise HashMismatch(f"{source_id}: manifest {key} {manifest.get(key)!r} != pin of {file_name}")
    required = manifest.get("training_config_required") or {}
    if required and required.get("lambda_rank", 0) != 0:
        raise ContractViolation(f"{source_id}: manifest requires lambda_rank {required.get('lambda_rank')}")
    return {"training_config_required": required, "status": manifest.get("status")}


# --------------------------------------------------------------------------- #
# Streaming NPZ reader: value columns are decoded only for selected rows
# --------------------------------------------------------------------------- #
def _read_npy_header(fh):
    version = npy_format.read_magic(fh)
    if version == (1, 0):
        return npy_format.read_array_header_1_0(fh)
    if version == (2, 0):
        return npy_format.read_array_header_2_0(fh)
    raise ContractViolation(f"unsupported .npy format version {version}")


class _NpzReader:
    """Read whole identity columns, or selected rows of value columns, from an NPZ."""

    def __init__(self, path: Path, source_id: str):
        self.path, self.source_id = Path(path), source_id
        try:
            self._zip = zipfile.ZipFile(self.path)
        except (zipfile.BadZipFile, OSError) as exc:
            raise ContractViolation(f"{source_id}: {self.path.name} is not a readable NPZ: {exc}") from exc
        self.headers: dict[str, tuple] = {}
        for info in self._zip.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with self._zip.open(info) as fh:
                shape, fortran, dtype = _read_npy_header(fh)
            if dtype.hasobject:
                raise ContractViolation(f"{source_id}: {info.filename} holds Python objects (pickle)")
            self.headers[info.filename[:-4]] = (tuple(shape), bool(fortran), dtype)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._zip.close()

    def keys(self):
        return set(self.headers)

    def shape(self, key: str) -> tuple:
        return self._header(key)[0]

    def dtype(self, key: str):
        return self._header(key)[2]

    def _header(self, key: str):
        if key not in self.headers:
            raise ContractViolation(f"{self.source_id}: {self.path.name} has no field {key!r}")
        return self.headers[key]

    def _row_layout(self, key: str):
        shape, fortran, dtype = self._header(key)
        if not shape:
            raise ContractViolation(f"{self.source_id}: field {key!r} is a scalar, not a per-row column")
        if fortran and len(shape) > 1:
            raise ContractViolation(f"{self.source_id}: field {key!r} is Fortran-ordered")
        return shape, dtype, dtype.itemsize * int(np.prod(shape[1:], dtype=np.int64))

    def read_full(self, key: str) -> np.ndarray:
        shape, dtype, row_bytes = self._row_layout(key)
        with self._zip.open(key + ".npy") as fh:
            _read_npy_header(fh)
            buf = fh.read(row_bytes * shape[0])
        if len(buf) != row_bytes * shape[0]:
            raise ContractViolation(f"{self.source_id}: field {key!r} is truncated")
        return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()

    def read_rows(self, key: str, rows: np.ndarray) -> np.ndarray:
        """Decode only ``rows`` (strictly increasing); other rows are skipped as bytes."""
        shape, dtype, row_bytes = self._row_layout(key)
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size and (rows[0] < 0 or rows[-1] >= shape[0] or np.any(np.diff(rows) <= 0)):
            raise ValueError("rows must be strictly increasing indices inside the column")
        out = np.empty((rows.size,) + shape[1:], dtype=dtype)
        with self._zip.open(key + ".npy") as fh:
            _read_npy_header(fh)
            position = 0
            for i, row in enumerate(rows.tolist()):
                skip = (row - position) * row_bytes
                while skip:
                    chunk = fh.read(min(skip, 1 << 20))
                    if not chunk:
                        raise ContractViolation(f"{self.source_id}: field {key!r} is truncated")
                    skip -= len(chunk)
                buf = fh.read(row_bytes)
                if len(buf) != row_bytes:
                    raise ContractViolation(f"{self.source_id}: field {key!r} is truncated")
                out[i] = np.frombuffer(buf, dtype=dtype).reshape(shape[1:])
                position = row + 1
        return out


# --------------------------------------------------------------------------- #
# Arrays and mixtures
# --------------------------------------------------------------------------- #
@dataclass
class SourceArrays:
    """Rows of one source, restricted to the requested partitions.

    Arrays are row-aligned. ``npz_row`` is the row in the source NPZ (or in the
    qualified dataset for local runs); ``joins`` carries record fields for
    native joins (e.g. ``observation_index``); ``ancestor_uid`` is the lineage
    key used for derivative isolation.
    """

    source_id: str
    role: str
    crops: np.ndarray
    target_hfr_px: np.ndarray
    log_target: np.ndarray
    secondary: dict
    sample_uid: np.ndarray
    group_uid: np.ndarray
    frame_uid: np.ndarray
    partition: np.ndarray
    star_snr: np.ndarray
    strata: dict
    batching: str
    ranking_eligible: np.ndarray
    identity: dict
    npz_row: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    joins: dict = field(default_factory=dict)
    ancestor_uid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.str_))

    def __len__(self) -> int:
        return int(self.target_hfr_px.shape[0])

    def partition_indices(self, partition: str) -> np.ndarray:
        if partition == SEALED_PARTITION and self.identity.get("mode") != "final_test":
            raise SealedPartitionError("development arrays contain no test rows")
        return np.flatnonzero(self.partition == partition)

    def subset(self, index) -> "SourceArrays":
        """Row subset (boolean mask or integer index); identity is kept, marked as a subset."""
        index = np.asarray(index)
        if index.dtype == bool:
            index = np.flatnonzero(index)
        take = lambda a: a[index]  # noqa: E731
        identity = dict(self.identity)
        identity["subset_of_rows"] = int(len(self))
        identity["subset_rows"] = int(index.size)
        return replace(
            self,
            crops=take(self.crops), target_hfr_px=take(self.target_hfr_px), log_target=take(self.log_target),
            secondary={k: take(v) for k, v in self.secondary.items()},
            sample_uid=take(self.sample_uid), group_uid=take(self.group_uid), frame_uid=take(self.frame_uid),
            partition=take(self.partition), star_snr=take(self.star_snr),
            strata={k: take(v) for k, v in self.strata.items()},
            ranking_eligible=take(self.ranking_eligible), identity=identity,
            npz_row=take(self.npz_row), joins={k: take(v) for k, v in self.joins.items()},
            ancestor_uid=take(self.ancestor_uid),
        )


@dataclass
class MixturePlan:
    """Per-source row pools of one partition plus source-balanced sampling weights.

    ``indices[source_id]`` index rows of that source's :class:`SourceArrays`.
    ``weights[source_id]`` are per-row probabilities: every source sums to
    ``1/k`` regardless of size. :meth:`sample_batch` draws a batch with an
    equal share per source, with replacement (contract-v1 mixed-arm sampling).
    """

    partition: str
    source_ids: tuple
    indices: dict
    weights: dict
    n_rows: dict
    batching: str
    target_definitions: dict
    common_reference_policy: str | None
    mixture_id: str | None
    identity: dict

    def sample_batch(self, rng: np.random.Generator, batch_size: int) -> dict:
        """``{source_id: row indices}`` with ``batch_size // k`` rows per source.

        A remainder is assigned to randomly chosen sources (one extra row each).
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        k = len(self.source_ids)
        base, remainder = divmod(int(batch_size), k)
        extra = set(rng.permutation(k)[:remainder].tolist()) if remainder else set()
        batch = {}
        for i, source_id in enumerate(self.source_ids):
            pool = self.indices[source_id]
            count = base + (1 if i in extra else 0)
            batch[source_id] = pool[rng.integers(0, pool.size, size=count)] if count else pool[:0]
        return batch

    def flat(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(source_position, row, weight)`` over the pooled rows, for weighted samplers."""
        positions, rows, weights = [], [], []
        for i, source_id in enumerate(self.source_ids):
            positions.append(np.full(self.indices[source_id].size, i, dtype=np.int64))
            rows.append(self.indices[source_id])
            weights.append(self.weights[source_id])
        return np.concatenate(positions), np.concatenate(rows), np.concatenate(weights)


def make_mixture(sources: Sequence[SourceArrays], partition: str = "train_fit", *,
                 contract: SourcesContract | None = None) -> MixturePlan:
    """Combine development rows of several sources without identity collisions.

    Refuses (:class:`ContractViolation`): duplicate sources, un-namespaced or
    colliding sample/group/frame identities, mixed batching, sources loaded
    under different contracts, fixtures mixed with real data, local runs mixed
    with public archives, empty pools, and different target definitions
    unless the contract declares that mixture with a common reference policy.
    """
    if partition == SEALED_PARTITION:
        raise SealedPartitionError("mixtures never draw from the sealed test partition")
    if partition not in DEV_PARTITIONS:
        raise ValueError(f"unknown partition {partition!r}; expected one of {DEV_PARTITIONS}")
    sources = list(sources)
    if not sources:
        raise ContractViolation("a mixture needs at least one source")
    ids = [s.source_id for s in sources]
    if len(set(ids)) != len(ids):
        raise ContractViolation(f"duplicate source in mixture: {ids}")
    roles = {s.role for s in sources}
    if "fixture_only" in roles and roles != {"fixture_only"}:
        raise ContractViolation(f"fixture_only sources cannot be mixed with real sources ({sorted(roles)})")
    if "local_verified_run" in roles and roles != {"local_verified_run"}:
        raise ContractViolation(f"local runs and public archives are not a declared mixture ({sorted(roles)})")
    if any(s.identity.get("mode") == "final_test" for s in sources):
        raise SealedPartitionError("final-test arrays cannot enter a mixture")
    batching = {s.batching for s in sources}
    if len(batching) != 1:
        raise ContractViolation(f"sources disagree on batching {sorted(batching)}")
    contract_shas = {s.identity.get("contract_sha256") for s in sources}
    if len(contract_shas) != 1:
        raise ContractViolation("sources were loaded under different source contracts")

    seen: dict[str, dict[str, str]] = {"sample_uid": {}, "group_uid": {}, "frame_uid": {}}
    for s in sources:
        prefix = f"{s.source_id}:"
        for name in seen:
            values = np.asarray(getattr(s, name)).astype(str)
            if values.size and not np.all(np.char.startswith(values, prefix)):
                raise ContractViolation(f"{s.source_id}: {name} values are not namespaced with {prefix!r}")
            unique = set(values.tolist())
            if name == "sample_uid" and len(unique) != values.size:
                raise ContractViolation(f"{s.source_id}: duplicate sample_uid within the source")
            for value in unique:
                owner = seen[name].setdefault(value, s.source_id)
                if owner != s.source_id:
                    raise ContractViolation(
                        f"{name} collision between {owner} and {s.source_id}: {value!r}"
                    )

    targets = {s.source_id: s.identity.get("target_definition_id") for s in sources}
    mixture = None
    if len(set(targets.values())) > 1:
        contract = contract or load_contract()
        if contract.sha256 not in contract_shas:
            raise ContractViolation("mixture contract differs from the contract the sources were loaded under")
        mixture = contract.mixture_for(ids)
        if mixture is None:
            raise ContractViolation(
                f"targets differ ({targets}) and the contract declares no mixture with a common reference "
                "policy for these sources"
            )

    order = sorted(range(len(sources)), key=lambda i: ids[i])
    sources = [sources[i] for i in order]
    k = len(sources)
    indices, weights, n_rows, membership = {}, {}, {}, {}
    for s in sources:
        pool = np.flatnonzero(np.asarray(s.partition) == partition).astype(np.int64)
        if pool.size == 0:
            raise ContractViolation(f"{s.source_id}: no {partition} rows to mix")
        indices[s.source_id] = pool
        weights[s.source_id] = np.full(pool.size, 1.0 / (k * pool.size), dtype=np.float64)
        n_rows[s.source_id] = int(pool.size)
        membership[s.source_id] = _membership_sha256(np.asarray(s.sample_uid)[pool])
    identity = {
        "partition": partition,
        "contract_sha256": next(iter(contract_shas)),
        "sources": {s.source_id: {"role": s.role, "npz_sha256": s.identity.get("npz_sha256"),
                                  "registry_sha256": s.identity.get("registry_sha256"),
                                  "membership_sha256": membership[s.source_id],
                                  "rows": n_rows[s.source_id]} for s in sources},
        "source_weight": {s.source_id: 1.0 / k for s in sources},
    }
    identity["mixture_sha256"] = _canonical_sha256(identity)
    return MixturePlan(
        partition=partition, source_ids=tuple(s.source_id for s in sources), indices=indices,
        weights=weights, n_rows=n_rows, batching=next(iter(batching)),
        target_definitions={s.source_id: targets[s.source_id] for s in sources},
        common_reference_policy=(mixture or {}).get("common_reference_policy"),
        mixture_id=(mixture or {}).get("id"), identity=identity,
    )


# --------------------------------------------------------------------------- #
# Calibration allocation
# --------------------------------------------------------------------------- #
def _calibration_from_rule(source_id: str, rule: Mapping | None, train_group_uids) -> set[str]:
    rule = dict(rule or {"rule": "unavailable", "reason": "no calibration rule declared"})
    if rule.get("rule") == "unavailable":
        raise CalibrationUnavailable(f"{source_id}: calibration unavailable: {rule.get('reason', 'declared')}")
    if rule.get("rule") != "sha256_lowest_fraction":
        raise ContractViolation(f"{source_id}: unknown calibration rule {rule.get('rule')!r}")
    uids = sorted({str(u) for u in train_group_uids})
    prefix = f"{source_id}:"
    if any(not u.startswith(prefix) for u in uids):
        raise ContractViolation(f"{source_id}: calibration takes namespaced group_uids ('{prefix}...')")
    n = len(uids)
    k = _round_half_up(n * float(rule["fraction"]))
    if k < 1 or n - k < 1:
        raise CalibrationUnavailable(
            f"{source_id}: {n} train group(s) cannot yield a non-empty calibration and train_fit split "
            f"at fraction {rule['fraction']}"
        )
    ranked = sorted(uids, key=lambda u: (hashlib.sha256(u.encode("utf-8")).hexdigest(), u))
    return set(ranked[:k])


def calibration_groups(source_id: str, train_group_uids, contract: SourcesContract | None = None) -> set[str]:
    """Calibration groups chosen from identities only (never from data or performance).

    stellar_sim_v1: the lowest ``round_half_up(0.15 * n)`` train groups by
    sha256(group_uid) hex (41 of 271). hst_acs_corpus_v1:
    :class:`CalibrationUnavailable` (6 train exposures).
    """
    contract = contract or load_contract()
    entry = contract.source(source_id)
    if entry.get("role") in SEALED_ROLES:
        raise SealedPartitionError(f"{source_id} is sealed ({entry.get('role')})")
    return _calibration_from_rule(source_id, entry.get("calibration"), train_group_uids)


# --------------------------------------------------------------------------- #
# Public-archive / fixture loading
# --------------------------------------------------------------------------- #
def _normalize_partitions(partitions) -> tuple[str, ...]:
    if isinstance(partitions, str):
        partitions = (partitions,)
    partitions = tuple(partitions)
    if SEALED_PARTITION in partitions or "test_sealed" in partitions:
        raise SealedPartitionError(
            "the test partition is sealed: development loads return train_fit/calibration/val only; "
            "final evaluation goes through load_final_test with a receipt"
        )
    unknown = [p for p in partitions if p not in DEV_PARTITIONS]
    if unknown or not partitions:
        raise ValueError(f"unknown partition(s) {unknown or partitions!r}; expected a subset of {DEV_PARTITIONS}")
    if len(set(partitions)) != len(partitions):
        raise ValueError(f"duplicate partitions {partitions!r}")
    return partitions


def _assert_single_partition(labels: np.ndarray, keys: np.ndarray, what: str, source_id: str) -> None:
    mapping: dict[str, str] = {}
    for key, label in zip(keys.tolist(), labels.tolist()):
        previous = mapping.setdefault(key, label)
        if previous != label:
            raise ContractViolation(
                f"{source_id}: {what} {key!r} spans partitions {previous!r} and {label!r} (derivative leak)"
            )


def _check_registry(source_id, entry, reg_path, npz_path, npz_sha, groups, split_col, labels_map):
    try:
        registry = json.loads(Path(reg_path).read_text())
    except ValueError as exc:
        raise ContractViolation(f"{source_id}: registry is not valid JSON: {exc}") from exc
    if registry.get("dataset_sha256") != npz_sha:
        raise HashMismatch(
            f"{source_id}: registry is bound to dataset {registry.get('dataset_sha256')}, not {npz_sha}"
        )
    if registry.get("group_field", entry["group_field"]) != entry["group_field"]:
        raise ContractViolation(
            f"{source_id}: registry groups by {registry.get('group_field')!r}, contract by {entry['group_field']!r}"
        )
    try:
        splits = load_registered_splits(str(reg_path), str(npz_path), groups)
    except (ValueError, KeyError) as exc:
        raise ContractViolation(f"{source_id}: frozen registry rejected: {exc}") from exc
    n = int(groups.size)
    every = np.concatenate([splits[name] for name in REGISTRY_PARTITIONS])
    if every.size != n or (every.size and (every.min() < 0 or every.max() >= n)) or np.unique(every).size != n:
        raise ContractViolation(f"{source_id}: registry indices must cover each of {n} rows exactly once")
    counts = registry.get("counts") or {}
    for name in REGISTRY_PARTITIONS:
        if name in counts and int(counts[name]) != int(splits[name].size):
            raise ContractViolation(f"{source_id}: registry counts[{name}] disagrees with its indices")
    registry_groups = registry.get("groups") or {}
    for name in REGISTRY_PARTITIONS:
        if name in registry_groups and {str(g) for g in registry_groups[name]} != set(groups[splits[name]].tolist()):
            raise ContractViolation(f"{source_id}: registry groups[{name}] disagree with its indices")
    reg_label = np.empty(n, dtype="<U5")
    for name in REGISTRY_PARTITIONS:
        reg_label[splits[name]] = name
    if split_col is not None:
        expected = np.asarray([labels_map[label] for label in reg_label.tolist()], dtype=np.str_)
        bad = np.flatnonzero(split_col.astype(str) != expected)
        if bad.size:
            raise ContractViolation(
                f"{source_id}: registry and split column disagree on {bad.size} row(s) (registry key 'val' "
                f"maps to column {labels_map['val']!r}); first rows {bad[:5].tolist()}"
            )
    return registry, reg_label


def _join_records(source_id, entry, directory, n, cols, sealed_mask):
    """Join records.jsonl to NPZ rows by index; keep only identity for sealed rows."""
    join = entry.get("records_join")
    if not join or not entry.get("records"):
        return None
    index_field = join["index_field"]
    ancestor_field = join.get("ancestor_field")
    strata_paths = (entry.get("strata") or {}).get("records") or {}
    join_paths = entry.get("joins") or {}
    by_row: list = [None] * n
    with (directory / entry["records"]).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ContractViolation(f"{source_id}: records line {line_no} is not JSON") from exc
            index = record.get(index_field)
            if index is None:
                continue
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < n:
                raise ContractViolation(f"{source_id}: records line {line_no} has {index_field}={index!r}")
            if by_row[index] is not None:
                raise ContractViolation(f"{source_id}: two records claim {index_field}={index}")
            for record_key, column in (join.get("agree") or {}).items():
                if str(record.get(record_key)) != str(cols[column][index]):
                    raise ContractViolation(
                        f"{source_id}: records line {line_no} {record_key}={record.get(record_key)!r} does not "
                        f"match NPZ row {index} {column}={cols[column][index]!r} (bad join)"
                    )
            kept = {"line": line_no}
            if ancestor_field:
                ancestor = record.get(ancestor_field)
                if not isinstance(ancestor, str) or not ancestor:
                    raise ContractViolation(f"{source_id}: records line {line_no} lacks {ancestor_field}")
                kept["ancestor"] = ancestor
            if not sealed_mask[index]:
                kept["strata"] = {name: _dotted(record, path) for name, path in strata_paths.items()}
                kept["joins"] = {name: _dotted(record, path) for name, path in join_paths.items()}
            by_row[index] = kept
            del record
    missing = [i for i, kept in enumerate(by_row) if kept is None]
    if missing:
        raise ContractViolation(f"{source_id}: {len(missing)} NPZ row(s) have no record (first {missing[:5]})")
    return by_row


def _plan_public(source_id: str, entry: dict, contract: SourcesContract, root: Path, *,
                 final_test: bool = False) -> dict:
    """Verify a public/fixture source and assign every row a partition (identity only).

    Record strata/join fields of sealed rows are kept only when ``final_test``.
    """
    directory = _source_dir(root, source_id, entry)
    files = _verify_files(source_id, entry, directory)
    manifest_info = _check_manifest(source_id, entry, directory)
    npz_path, reg_path = directory / entry["npz"], directory / entry["registry"]
    labels_map = contract.split_column_labels
    reader = _NpzReader(npz_path, source_id)
    with reader:
        if "split" not in reader.keys():
            raise ContractViolation(f"{source_id}: NPZ has no split column")
        n = int(reader.shape("split")[0])
        for key in sorted(reader.keys()):
            shape = reader.shape(key)
            if not shape or shape[0] != n:
                raise ContractViolation(f"{source_id}: field {key!r} has shape {shape}, expected leading {n}")
        crops_shape = reader.shape("crops")
        if crops_shape[1:] != CROP_SHAPE:
            raise ContractViolation(f"{source_id}: crops have shape {crops_shape}, expected (N, 64, 64)")
        if reader.dtype("crops").kind != "f":
            raise ContractViolation(f"{source_id}: crops dtype {reader.dtype('crops')} is not floating point")
        value_fields = [entry["target_field"], *(entry.get("target_equals") or []),
                        *(entry.get("secondary_fields") or []), "star_snr"]
        if entry.get("log_field"):
            value_fields.append(entry["log_field"])
        for key in value_fields:
            if reader.shape(key) != (n,) or reader.dtype(key).kind != "f":
                raise ContractViolation(f"{source_id}: {key} must be a float column of length {n}")
        sample_field = entry.get("sample_key_field", "run_key")
        frame_field = entry.get("frame_field", "frame_id")
        identity_fields = {"split", entry["group_field"], sample_field, frame_field}
        identity_fields.update((entry.get("records_join") or {}).get("agree", {}).values())
        identity_fields.update((entry.get("strata") or {}).get("npz") or [])
        for optional in ("ranking_eligible", "run_key_semantics", "training_eligible"):
            if optional in reader.keys():
                identity_fields.add(optional)
        tier = entry.get("tier_filter")
        if tier:
            identity_fields.add(tier["field"])
        cols = {}
        for key in sorted(identity_fields):
            cols[key] = reader.read_full(key)
    groups = cols[entry["group_field"]].astype(str)
    split_col = cols["split"].astype(str)
    registry, reg_label = _check_registry(source_id, entry, reg_path, npz_path, files[entry["npz"]],
                                          groups, split_col, labels_map)

    expected = entry.get("expected_counts") or {}
    for name in ("rows",) + REGISTRY_PARTITIONS:
        if name in expected:
            actual = n if name == "rows" else int((reg_label == name).sum())
            if actual != int(expected[name]):
                raise ContractViolation(f"{source_id}: {name} count {actual} != contract {expected[name]}")
    for name, value in (expected.get("groups") or {}).items():
        actual = len(set(groups[reg_label == name].tolist()))
        if actual != int(value):
            raise ContractViolation(f"{source_id}: {name} group count {actual} != contract {value}")

    sample_keys = cols[sample_field].astype(str)
    if np.unique(sample_keys).size != n:
        raise ContractViolation(f"{source_id}: {sample_field} is not unique per row; it cannot be a sample key")
    sample_uid = _namespace(source_id, sample_keys)
    group_uid = _namespace(source_id, groups)
    frame_uid = _namespace(source_id, cols[frame_field])

    if entry["batching"] == "pointwise":
        if "ranking_eligible" in cols and np.any(cols["ranking_eligible"].astype(bool)):
            raise ContractViolation(f"{source_id}: pointwise source has ranking_eligible rows")
        semantics = entry.get("run_key_semantics")
        if semantics and "run_key_semantics" in cols and np.any(cols["run_key_semantics"].astype(str) != semantics):
            raise ContractViolation(f"{source_id}: run_key_semantics is not {semantics!r} on every row")

    sealed_mask = (reg_label == "test") if not final_test else np.zeros(n, dtype=bool)
    records = _join_records(source_id, entry, directory, n, cols, sealed_mask)
    if records is not None and (entry.get("records_join") or {}).get("ancestor_field"):
        ancestor_uid = _namespace(source_id, [r["ancestor"] for r in records])
    else:
        ancestor_uid = group_uid.copy()

    for what, keys in (("group", group_uid), ("frame", frame_uid), ("ancestor", ancestor_uid)):
        _assert_single_partition(reg_label, keys, what, source_id)

    passes = np.ones(n, dtype=bool)
    if tier:
        passes = cols[tier["field"]].astype(str) == str(tier["equals"])
    if "training_eligible" in cols and entry.get("training_eligible_rule"):
        expected_eligible = (reg_label == "train") & passes
        if np.any(cols["training_eligible"].astype(bool) != expected_eligible):
            raise ContractViolation(f"{source_id}: training_eligible disagrees with {entry['training_eligible_rule']}")

    train_groups = sorted(set(group_uid[reg_label == "train"].tolist()))
    calibration = {"rule": (entry.get("calibration") or {}).get("rule", "unavailable")}
    try:
        chosen = _calibration_from_rule(source_id, entry.get("calibration"), train_groups)
        calibration.update(available=True, fraction=entry["calibration"]["fraction"], groups=len(chosen),
                           train_groups=len(train_groups), groups_sha256=_membership_sha256(chosen))
        expected_groups = entry["calibration"].get("expected_groups")
        if expected_groups is not None and len(chosen) != int(expected_groups):
            raise ContractViolation(f"{source_id}: {len(chosen)} calibration groups != contract {expected_groups}")
    except CalibrationUnavailable as exc:
        chosen = set()
        calibration.update(available=False, reason=str(exc), train_groups=len(train_groups))

    label = np.where(reg_label == "val", "val", np.where(reg_label == "test", "test", "train_fit")).astype("<U11")
    if chosen:
        label[(reg_label == "train") & np.isin(group_uid, sorted(chosen))] = "calibration"
    for what, keys in (("group", group_uid), ("frame", frame_uid), ("ancestor", ancestor_uid)):
        _assert_single_partition(label, keys, what, source_id)

    counts = {}
    for name in DEV_PARTITIONS + (SEALED_PARTITION,):
        member = (label == name)
        counts[name] = {
            "rows": int((member & passes).sum()),
            "groups": len(set(group_uid[member & passes].tolist())),
            "rows_excluded_by_tier": int((member & ~passes).sum()),
            "membership_sha256": _membership_sha256(sample_uid[member & passes]),
        }
    registry_counts = {name: {"rows": int((reg_label == name).sum()),
                              "groups": len(set(groups[reg_label == name].tolist()))}
                       for name in REGISTRY_PARTITIONS}
    identity = {
        "schema": ARRAYS_SCHEMA,
        "source_id": source_id,
        "role": entry["role"],
        "contract_sha256": contract.sha256,
        "contract_file_sha256": contract.file_sha256,
        "experiment_contract_sha256": contract.experiment_contract_sha256,
        "data_root": str(root),
        "files": files,
        "npz": entry["npz"],
        "npz_sha256": files[entry["npz"]],
        "registry": entry["registry"],
        "registry_sha256": files[entry["registry"]],
        "registry_strategy": registry.get("strategy"),
        "registry_seed": registry.get("seed"),
        "group_field": entry["group_field"],
        "target_field": entry["target_field"],
        "target_definition_id": entry["target_definition_id"],
        "label_definition": entry.get("label_definition"),
        "secondary_fields": list(entry.get("secondary_fields") or []),
        "units": entry.get("units"),
        "tier_filter": tier,
        "batching": entry["batching"],
        "ranking_eligible": bool(entry.get("ranking_eligible", False)),
        "rows_total": n,
        "registry_counts": registry_counts,
        "partition_counts": counts,
        "sealed_test": {"rows": counts["test"]["rows"], "groups": counts["test"]["groups"],
                        "membership_sha256": counts["test"]["membership_sha256"],
                        "arrays_returned": False},
        "calibration": calibration,
        "manifest": manifest_info,
        "known_label_issue": entry.get("known_label_issue"),
        "real_multi_run_data": False,
        "runtime_eligible": False,
    }
    return {"entry": entry, "directory": directory, "npz_path": npz_path, "n": n, "cols": cols,
            "label": label, "passes": passes, "records": records, "sample_uid": sample_uid,
            "group_uid": group_uid, "frame_uid": frame_uid, "ancestor_uid": ancestor_uid,
            "identity": identity}


def _materialize(plan: dict, wanted: tuple[str, ...], mode: str) -> SourceArrays:
    entry, identity = plan["entry"], dict(plan["identity"])
    source_id = identity["source_id"]
    rows = np.flatnonzero(np.isin(plan["label"], list(wanted)) & plan["passes"]).astype(np.int64)
    with _NpzReader(plan["npz_path"], source_id) as reader:
        crops = reader.read_rows("crops", rows)
        target = reader.read_rows(entry["target_field"], rows).astype(np.float64)
        aliases = {name: reader.read_rows(name, rows) for name in entry.get("target_equals") or []}
        secondary = {name: reader.read_rows(name, rows).astype(np.float32)
                     for name in entry.get("secondary_fields") or []}
        star_snr = reader.read_rows("star_snr", rows).astype(np.float32)
        stored_log = reader.read_rows(entry["log_field"], rows) if entry.get("log_field") else None

    flat = crops.reshape(rows.size, int(np.prod(crops.shape[1:])))
    bad_crops = ~np.isfinite(flat).all(axis=1)
    if bad_crops.any():
        raise ContractViolation(
            f"{source_id}: {int(bad_crops.sum())} crop(s) contain non-finite pixels (first NPZ rows "
            f"{rows[bad_crops][:5].tolist()})"
        )
    out_of_range = ((flat.min(axis=1) < -1e-6) | (flat.max(axis=1) > 1 + 1e-6)) if rows.size \
        else np.zeros(0, dtype=bool)
    if out_of_range.any():
        raise ContractViolation(
            f"{source_id}: {int(out_of_range.sum())} crop(s) fall outside the normalized [0, 1] range"
        )
    bad_target = ~np.isfinite(target) | (target <= 0)
    if bad_target.any():
        raise ContractViolation(
            f"{source_id}: {int(bad_target.sum())} non-finite or non-positive {entry['target_field']} value(s)"
        )
    for name, values in aliases.items():
        if not np.array_equal(values.astype(np.float64), target):
            raise ContractViolation(f"{source_id}: {name} differs from target {entry['target_field']} on loaded rows")
    log_target = np.log(target).astype(np.float32)
    if stored_log is not None and rows.size and not np.allclose(stored_log.astype(np.float64), np.log(target),
                                                                rtol=0, atol=1e-5):
        raise ContractViolation(f"{source_id}: stored {entry['log_field']} is not log({entry['target_field']})")

    strata = {name: plan["cols"][name][rows].astype(str)
              for name in (entry.get("strata") or {}).get("npz") or []}
    joins = {}
    if plan["records"] is not None:
        kept = [plan["records"][r] for r in rows.tolist()]
        for name in ((entry.get("strata") or {}).get("records") or {}):
            strata[name] = _column([k["strata"][name] for k in kept]) if kept else np.zeros(0, dtype=np.str_)
        for name in (entry.get("joins") or {}):
            joins[name] = _column([k["joins"][name] for k in kept]) if kept else np.zeros(0, dtype=np.int64)
    identity.update(
        mode=mode,
        partitions_returned=list(wanted),
        rows_returned=int(rows.size),
        star_snr_nonfinite=int((~np.isfinite(star_snr)).sum()),
    )
    if mode != "final_test":
        identity["rows_returned_by_partition"] = {p: int((plan["label"][rows] == p).sum()) for p in wanted}
    ranking = np.zeros(rows.size, dtype=bool)
    return SourceArrays(
        source_id=source_id, role=entry["role"], crops=crops.astype(np.float32, copy=False),
        target_hfr_px=target.astype(np.float32), log_target=log_target, secondary=secondary,
        sample_uid=plan["sample_uid"][rows], group_uid=plan["group_uid"][rows],
        frame_uid=plan["frame_uid"][rows], partition=plan["label"][rows].astype(np.str_),
        star_snr=star_snr, strata=strata, batching=entry["batching"], ranking_eligible=ranking,
        identity=identity, npz_row=rows, joins=joins, ancestor_uid=plan["ancestor_uid"][rows],
    )


# --------------------------------------------------------------------------- #
# Local verified runs (delegated unchanged to load_qualified_dataset)
# --------------------------------------------------------------------------- #
def _load_local(source_id: str, entry: dict, contract: SourcesContract, data_root, wanted: tuple[str, ...]):
    """Local run rows via ``load_qualified_dataset``, unchanged.

    The delegate decodes the whole qualified dataset (test rows included) and
    opens ``<stem>.manifest.json``, which the contract must pin. This function
    returns only the requested development rows and derives ranking
    eligibility and ``real_multi_run_data`` from those rows alone; test rows
    contribute counts only.
    """
    if data_root is None:
        raise ContractViolation(f"{source_id}: local_verified_run sources need an explicit data_root")
    root = resolve_data_root(data_root)
    directory = _source_dir(root, source_id, entry)
    files = _verify_files(source_id, entry, directory)
    npz_path, reg_path = directory / entry["npz"], directory / entry["registry"]
    from data_pipeline import build_focus_quality_dataset as builder

    dataset, report = builder.load_qualified_dataset(str(npz_path))
    group_field = entry["group_field"]
    with np.load(npz_path, allow_pickle=False) as npz:
        if group_field not in npz.files:
            raise ContractViolation(f"{source_id}: local dataset has no {group_field!r} column")
        all_groups = np.asarray(npz[group_field]).astype(str)
    registry, reg_label_all = _check_registry(source_id, entry, reg_path, npz_path, files[entry["npz"]],
                                              all_groups, None, contract.split_column_labels)
    group_to_label = {}
    for g, lab in zip(all_groups.tolist(), reg_label_all.tolist()):
        group_to_label.setdefault(g, lab)
    groups = np.asarray(dataset[group_field]).astype(str)
    reg_label = np.asarray([group_to_label.get(g, "") for g in groups.tolist()], dtype="<U5")
    if np.any(reg_label == ""):
        raise ContractViolation(f"{source_id}: qualified rows belong to groups absent from the registry")
    crops = np.asarray(dataset["crops"])
    n = int(crops.shape[0])
    if crops.shape[1:] != CROP_SHAPE:
        raise ContractViolation(f"{source_id}: crops have shape {crops.shape}, expected (N, 64, 64)")
    group_uid = _namespace(source_id, groups)
    frame_ids = np.asarray(dataset["frame_id"]).astype(str)
    frame_uid = _namespace(source_id, frame_ids)
    occurrence: dict[str, int] = {}
    sample_keys = []
    for frame in frame_ids.tolist():
        k = occurrence.get(frame, 0)
        occurrence[frame] = k + 1
        sample_keys.append(f"{frame}:{k}")
    sample_uid = _namespace(source_id, sample_keys)
    for what, keys in (("group", group_uid), ("frame", frame_uid)):
        _assert_single_partition(reg_label, keys, what, source_id)
    train_groups = sorted(set(group_uid[reg_label == "train"].tolist()))
    calibration = {"rule": (entry.get("calibration") or {}).get("rule", "unavailable")}
    try:
        chosen = _calibration_from_rule(source_id, entry.get("calibration"), train_groups)
        calibration.update(available=True, groups=len(chosen), train_groups=len(train_groups),
                           groups_sha256=_membership_sha256(chosen))
    except CalibrationUnavailable as exc:
        chosen = set()
        calibration.update(available=False, reason=str(exc), train_groups=len(train_groups))
    label = np.where(reg_label == "val", "val", np.where(reg_label == "test", "test", "train_fit")).astype("<U11")
    if chosen:
        label[(reg_label == "train") & np.isin(group_uid, sorted(chosen))] = "calibration"
    rows = np.flatnonzero(np.isin(label, list(wanted)))
    target = np.asarray(dataset["hfr_px"], dtype=np.float64)[rows]
    picked = crops[rows].astype(np.float32)
    if not np.isfinite(picked).all():
        raise ContractViolation(f"{source_id}: non-finite crop pixels in loaded rows")
    if np.any(~np.isfinite(target) | (target <= 0)):
        raise ContractViolation(f"{source_id}: non-finite or non-positive hfr_px in loaded rows")
    # Ranking eligibility is derived from the returned (non-test) rows only; groups never span
    # partitions, so each returned group is judged on its own complete set of rows.
    returned_groups = groups[rows]
    returned_positions = np.asarray(dataset["position"])[rows] if "position" in dataset else None
    eligible_groups = set()
    for g in set(returned_groups.tolist()):
        member = returned_groups == g
        if member.sum() >= 2 and (returned_positions is None or np.unique(returned_positions[member]).size >= 2):
            eligible_groups.add(g)
    ranking = np.isin(returned_groups, sorted(eligible_groups))
    counts = {name: {"rows": int((label == name).sum()), "groups": len(set(group_uid[label == name].tolist()))}
              for name in DEV_PARTITIONS + (SEALED_PARTITION,)}
    identity = {
        "schema": ARRAYS_SCHEMA, "source_id": source_id, "role": "local_verified_run", "mode": "development",
        "contract_sha256": contract.sha256, "contract_file_sha256": contract.file_sha256,
        "data_root": str(root), "files": files, "npz": entry["npz"], "npz_sha256": files[entry["npz"]],
        "registry": entry["registry"], "registry_sha256": files[entry["registry"]],
        "registry_strategy": registry.get("strategy"), "group_field": group_field,
        "target_field": "hfr_px", "target_definition_id": entry["target_definition_id"],
        "batching": "grouped_run", "qualified_report": report, "rows_total": n,
        "partition_counts": counts,
        "sealed_test": {"rows": counts["test"]["rows"], "groups": counts["test"]["groups"],
                        "arrays_returned": False},
        "calibration": calibration, "partitions_returned": list(wanted), "rows_returned": int(rows.size),
        "real_multi_run_data": bool(eligible_groups), "runtime_eligible": False,
        "delegated_loader": "build_focus_quality_dataset.load_qualified_dataset",
    }
    secondary = {name: np.asarray(dataset[name])[rows] for name in ("position", "classical_frame_hfr_px")
                 if name in dataset}
    strata = {name: np.asarray(dataset[name]).astype(str)[rows]
              for name in ("rig_id", "camera_id", "filter_name", "night_id") if name in dataset}
    return SourceArrays(
        source_id=source_id, role="local_verified_run", crops=picked,
        target_hfr_px=target.astype(np.float32), log_target=np.log(target).astype(np.float32),
        secondary=secondary, sample_uid=sample_uid[rows], group_uid=group_uid[rows], frame_uid=frame_uid[rows],
        partition=label[rows].astype(np.str_), star_snr=np.asarray(dataset["star_snr"], dtype=np.float32)[rows],
        strata=strata, batching="grouped_run", ranking_eligible=ranking, identity=identity,
        npz_row=rows.astype(np.int64), joins={}, ancestor_uid=group_uid[rows],
    )


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def _entry_for(source_id: str, contract: SourcesContract | None) -> tuple[dict, SourcesContract]:
    contract = contract or load_contract()
    entry = contract.source(source_id)
    if entry.get("role") in SEALED_ROLES:
        raise SealedPartitionError(
            f"{source_id} is {entry['role']} and is never loaded in development mode: {entry.get('reason', '')}"
        )
    return entry, contract


def load_source(source_id: str, partitions=DEV_PARTITIONS, contract: SourcesContract | None = None,
                data_root: str | os.PathLike | None = None) -> SourceArrays:
    """Development rows of one source (``train_fit``/``calibration``/``val``).

    Raises :class:`SealedPartitionError` for ``test`` or a sealed source,
    :class:`DataUnavailable`/:class:`HashMismatch` before any array is read,
    and :class:`ContractViolation` for malformed data or broken joins.
    """
    wanted = _normalize_partitions(partitions)
    entry, contract = _entry_for(source_id, contract)
    if entry["role"] == "local_verified_run":
        return _load_local(source_id, entry, contract, data_root, wanted)
    root = resolve_data_root(data_root)
    plan = _plan_public(source_id, entry, contract, root)
    return _materialize(plan, wanted, "development")


def verify_source(source_id: str, contract: SourcesContract | None = None,
                  data_root: str | os.PathLike | None = None) -> dict:
    """Mount + pinned hashes + registry binding + partition counts, without reading values.

    Cheap enough to call before every trial (contract mid-run check). Test
    rows are counted and membership-hashed only.
    """
    entry, contract = _entry_for(source_id, contract)
    if entry["role"] == "local_verified_run":
        root = resolve_data_root(data_root)
        return {"source_id": source_id, "files": _verify_files(source_id, entry, _source_dir(root, source_id, entry))}
    root = resolve_data_root(data_root)
    return _plan_public(source_id, entry, contract, root)["identity"]


def _validate_receipt(source_id: str, receipt: Mapping) -> dict:
    if not isinstance(receipt, Mapping):
        raise SealedPartitionError("a final-test receipt (dict) is required")
    missing = [key for key in RECEIPT_REQUIRED if not receipt.get(key)]
    if missing:
        raise SealedPartitionError(f"final-test receipt lacks {missing}")
    for key in ("candidate_sha256", "protocol_sha256"):
        if not _is_hex64(receipt[key]):
            raise SealedPartitionError(f"receipt {key} must be the frozen sha256 (64 lowercase hex)")
    if receipt["acknowledgement"] != FINAL_TEST_ACKNOWLEDGEMENT.format(source_id=source_id):
        raise SealedPartitionError(
            "receipt acknowledgement must equal FINAL_TEST_ACKNOWLEDGEMENT formatted with the source id"
        )
    return json.loads(_canonical_json(dict(receipt)))


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def final_test_ledger_dir(contract: SourcesContract) -> Path:
    """Where final-test consumption is recorded, independent of any caller output directory.

    ``$ACHERNAR_FQE_FINAL_TEST_LEDGER`` when set (a fixed, run-level
    location); otherwise ``fqe_final_test_ledger/`` beside the file the
    contract was read from (for the tracked contract: ``ml/data_pipeline/``,
    so a consumed test is committed together with the contract change that
    enabled it). A contract built in memory, without the variable, has no
    ledger and the final test is refused.
    """
    env = os.environ.get(FINAL_TEST_LEDGER_ENV)
    if env:
        return Path(env)
    if contract.source_file:
        return Path(contract.source_file).parent / FINAL_TEST_LEDGER_DIRNAME
    raise SealedPartitionError(
        f"no final-test ledger: set ${FINAL_TEST_LEDGER_ENV} or load the contract from a file "
        "(the ledger then sits beside it)"
    )


def _create_exclusive(path: Path, payload: dict, refusal: str) -> None:
    try:
        with path.open("x", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2, sort_keys=True))
    except FileExistsError as exc:
        raise SealedPartitionError(refusal) from exc


def load_final_test(source_id: str, receipt: Mapping, *, receipt_dir: str | os.PathLike,
                    contract: SourcesContract | None = None,
                    data_root: str | os.PathLike | None = None) -> SourceArrays:
    """The sealed test rows of one source, once per ledger, under a recorded receipt.

    Requires: a contract entry whose ``final_test.enabled`` is JSON ``true``
    (the tracked v1 contract disables it for every source), a receipt with the
    frozen candidate/protocol identity and the exact acknowledgement, a
    ``receipt_dir`` and a ledger (:func:`final_test_ledger_dir`) outside every
    data root and the production model registry, and no earlier ledger entry
    for the pinned NPZ.

    Consumption is keyed by the pinned NPZ sha256 in the ledger, so a second
    call is refused whatever ``receipt_dir`` or source id it names. Order:
    (1) refusals that read no data, including an existing ledger entry or
    receipt; (2) an identity-only verification (pins, registry, counts; test
    rows keep no stratum or join field); (3) the ledger entry
    ``final_test__<npz_sha256>.json`` and then the receipt
    ``final_test_receipt__<source_id>.json`` are created exclusively; (4) only
    then are test-row records joined and test values decoded, after
    re-verifying that the files are unchanged. Both records end as ``loaded``
    or ``failed``. An abort after (3) still consumes the test. Never called
    by a development path.
    """
    entry, contract = _entry_for(source_id, contract)
    if entry["role"] == "local_verified_run":
        raise SealedPartitionError(f"{source_id}: final-test loading of local runs is not implemented")
    final = entry.get("final_test") or {}
    if final.get("enabled") is not True:
        raise SealedPartitionError(
            f"{source_id}: final-test entry is disabled by the contract ({final.get('reason', 'not enabled')})"
        )
    clean_receipt = _validate_receipt(source_id, receipt)
    root = resolve_data_root(data_root)
    target_dir = _refuse_protected(receipt_dir, "receipt_dir", data_root=root)
    ledger_dir = _refuse_protected(final_test_ledger_dir(contract), "final-test ledger", data_root=root)
    npz_sha = entry["files"][entry["npz"]]
    ledger_path = ledger_dir / f"final_test__{npz_sha}.json"
    receipt_path = target_dir / f"final_test_receipt__{source_id}.json"
    consumed = (f"{source_id}: the sealed test of {entry['npz']} (sha256 {npz_sha}) was already consumed "
                f"(ledger entry {ledger_path}); the test is consumed once")
    if ledger_path.exists():
        raise SealedPartitionError(consumed)
    if receipt_path.exists():
        raise SealedPartitionError(
            f"{source_id}: a final-test receipt already exists at {receipt_path}; the test is consumed once"
        )

    plan = _plan_public(source_id, entry, contract, root, final_test=False)
    issued = datetime.now(timezone.utc).isoformat()
    common = {
        "source_id": source_id, "status": "issued", "issued_utc": issued, "receipt": clean_receipt,
        "contract_sha256": contract.sha256, "contract_file_sha256": contract.file_sha256,
        "npz": entry["npz"], "npz_sha256": npz_sha, "registry_sha256": plan["identity"]["registry_sha256"],
        "files": plan["identity"]["files"], "sealed_test": plan["identity"]["sealed_test"],
        "loader": "data_pipeline.fqe_sources.load_final_test",
    }
    ledger_payload = dict(common, schema=LEDGER_SCHEMA, receipt_path=str(receipt_path))
    payload = dict(common, schema=RECEIPT_SCHEMA, ledger_path=str(ledger_path))
    ledger_dir.mkdir(parents=True, exist_ok=True)
    _create_exclusive(ledger_path, ledger_payload, consumed)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        _create_exclusive(receipt_path, payload,
                          f"{source_id}: a final-test receipt already exists at {receipt_path}; "
                          "the test is consumed once")
    except Exception as exc:
        ledger_payload.update(status="aborted_before_read", failed_utc=datetime.now(timezone.utc).isoformat(),
                              error=repr(exc))
        _write_json_atomic(ledger_path, ledger_payload)
        raise
    try:
        sealed_plan = _plan_public(source_id, entry, contract, root, final_test=True)
        for key in ("files", "sealed_test", "partition_counts"):
            if sealed_plan["identity"][key] != plan["identity"][key]:
                raise ContractViolation(f"{source_id}: {key} changed between the identity check and the sealed read")
        arrays = _materialize(sealed_plan, (SEALED_PARTITION,), "final_test")
    except Exception as exc:
        failed = datetime.now(timezone.utc).isoformat()
        for path, record in ((receipt_path, payload), (ledger_path, ledger_payload)):
            record.update(status="failed", failed_utc=failed, error=repr(exc))
            _write_json_atomic(path, record)
        raise
    arrays.identity["sealed_test"] = dict(arrays.identity["sealed_test"], arrays_returned=True)
    arrays.identity["final_test_receipt"] = str(receipt_path)
    arrays.identity["final_test_ledger"] = str(ledger_path)
    loaded = {"status": "loaded", "loaded_utc": datetime.now(timezone.utc).isoformat(),
              "rows_returned": len(arrays), "membership_sha256": _membership_sha256(arrays.sample_uid)}
    for path, record in ((receipt_path, payload), (ledger_path, ledger_payload)):
        record.update(loaded)
        _write_json_atomic(path, record)
    return arrays


# --------------------------------------------------------------------------- #
# Fixtures (tests only; schema-identical to the public importers' FQE NPZs)
# --------------------------------------------------------------------------- #
_PUBLIC_FIXTURE_FILES = ("fqe.npz", "fqe.splits.json", "records.jsonl", "manifest.json")
_LOCAL_FIXTURE_FILES = ("focus_quality_ds.npz", "focus_quality_ds.manifest.json", "focus_quality_ds.splits.json")


@dataclass
class FixtureSource:
    source_id: str
    data_root: Path
    directory: Path
    entry: dict
    contract: SourcesContract
    files: dict
    n_rows: int
    group_split: dict


def _fixture_normalize(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [1.0, 99.0])
    scale = hi - lo if hi > lo else 1.0
    return np.clip((image - lo) / scale, 0.0, 1.0).astype(np.float32)


def _fixture_crop(rng, sigma: float, shape) -> np.ndarray:
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    star = 1000.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    return _fixture_normalize(star + rng.normal(0.0, 3.0, size=shape))


def _default_group_splits(n_groups: int) -> list[str]:
    n_test = max(1, _round_half_up(0.15 * n_groups))
    n_val = max(1, _round_half_up(0.15 * n_groups))
    n_train = n_groups - n_val - n_test
    if n_train < 1:
        raise ValueError("a fixture needs at least 3 groups (train/val/test)")
    return ["train"] * n_train + ["val"] * n_val + ["test"] * n_test


def write_fixture_source(tmp_dir, source_id: str, n_groups: int, rows_per_group: int, *,
                         style: str = "sim", role: str | None = None, group_splits: Sequence[str] | None = None,
                         run_id_start: int = 0, seed: int = 0, calibration: str = "sha256_lowest_fraction",
                         calibration_fraction: float = 0.15, crop_shape=CROP_SHAPE, nonfinite_rows=(),
                         split_column_overrides: Mapping[int, str] | None = None,
                         ancestor_of_group: Mapping[int, int] | None = None, nonstrict_rows=(),
                         target_definition_id: str = "fixture:gaussian_hfr",
                         final_test_enabled: bool = True,
                         base_contract: SourcesContract | None = None) -> FixtureSource:
    """Write a small, deterministic source under ``tmp_dir/<source_id>/`` with matching pins.

    ``style="sim"`` mirrors stellar_sim_v1 (families of variants sharing one
    reference, ``measured_hfr_px`` secondary); ``style="hst"`` mirrors
    hst_acs_corpus_v1 (exposure groups, tier column, ``training_eligible``);
    ``style="local"`` writes a qualified local-v2 builder dataset with its
    manifest (role ``local_verified_run``, multi-position runs). Raw string
    keys and integer ``run_id`` values are identical across fixtures with the
    same arguments, so only namespacing keeps sources apart.

    Corruption knobs (applied before pinning, so they reach the array checks):
    ``crop_shape``, ``nonfinite_rows``, ``split_column_overrides`` (row ->
    stored split label), ``ancestor_of_group`` (group -> donor group whose
    ancestor it shares), ``nonstrict_rows`` (hst tier).

    Refuses (:class:`ContractViolation`, before creating anything) a
    ``tmp_dir`` inside a data root or the production model registry (see
    ``_protected_roots``), and (:class:`FileExistsError`) a target directory
    that already holds any file this helper writes: a fixture never
    overwrites.
    """
    if style not in ("sim", "hst", "local"):
        raise ValueError("style must be 'sim', 'hst' or 'local'")
    if n_groups < 3 or rows_per_group < 1:
        raise ValueError("fixtures need >= 3 groups and >= 1 row per group")
    if not source_id or ":" in source_id or not _relative_name_ok(source_id) or len(Path(source_id).parts) != 1:
        raise ValueError(f"fixture source id {source_id!r} must be one plain path component without ':'")
    role = role or ("local_verified_run" if style == "local" else "fixture_only")
    root = Path(tmp_dir)
    directory = root / source_id
    _refuse_protected(root, "fixture directory")
    _refuse_protected(directory, "fixture directory")
    outputs = _LOCAL_FIXTURE_FILES if style == "local" else _PUBLIC_FIXTURE_FILES
    existing = sorted(name for name in outputs if (directory / name).exists())
    if existing:
        raise FileExistsError(f"write_fixture_source never overwrites: {directory} already holds {existing}")
    directory.mkdir(parents=True, exist_ok=True)
    splits = list(group_splits) if group_splits is not None else _default_group_splits(n_groups)
    if len(splits) != n_groups or any(s not in REGISTRY_PARTITIONS for s in splits):
        raise ValueError("group_splits must name train/val/test for every group")
    if style == "local":
        return _write_local_fixture(root, directory, source_id, splits, rows_per_group, seed=seed,
                                    run_id_start=run_id_start, calibration=calibration,
                                    calibration_fraction=calibration_fraction,
                                    target_definition_id=target_definition_id, base_contract=base_contract,
                                    crop_shape=crop_shape)
    rng = np.random.default_rng(seed)
    column_label = dict(DEFAULT_SPLIT_COLUMN_LABELS)
    n = n_groups * rows_per_group
    crops, references, measured, records = [], [], [], []
    for g in range(n_groups):
        family_sigma = 0.8 + 2.5 * rng.random()
        donor = (ancestor_of_group or {}).get(g, g)
        for v in range(rows_per_group):
            row = g * rows_per_group + v
            sigma = family_sigma if style == "sim" else 0.6 + 1.5 * rng.random()
            reference = float(np.float32(sigma * math.sqrt(2.0 * math.log(2.0))))
            crops.append(_fixture_crop(rng, sigma, tuple(crop_shape)))
            references.append(reference)
            measured.append(reference * (1.0 + 0.02 * rng.standard_normal()) if style == "sim" else reference)
            if style == "sim":
                group = f"family{g:05d}"
                key = f"fix-f{g:05d}-v{v:02d}"
                frame = key
                ancestor = f"family{donor:05d}"
                extra = {"parameters": {"profile": "gaussian" if v % 2 == 0 else "moffat",
                                        "motion": "none" if v % 3 else "linear",
                                        "moffat_beta": 2.5 + v, "blended": False, "edge_case": False},
                         "array_index": 2 * row}
            else:
                group = f"FIX-date:{2000 + g:04d}-01-01"
                frame = f"exp{g:04d}"
                key = f"FIX:{frame}:SCI1:source{v:05d}"
                ancestor = f"FIX:exp{donor:04d}"
                extra = {"hfr_aperture_radius_px": 6.0 + v % 4, "masked_aperture_fraction": 0.0,
                         "observation_index": v, "crop_origin_xy": [10 * v, 20 * g],
                         "all_reference_index": 3 * row}
            records.append({"source_id": key, "frame_id": frame, "split_group": group,
                            "split": column_label[splits[g]], "ancestor_id": ancestor,
                            "star_snr": float(20 + 5 * v), "fqe_index": row, **extra})
    stack = np.stack(crops).astype(np.float32)
    for r in nonfinite_rows:
        stack[r, 0, 0] = np.nan
    refs = np.asarray(references, dtype=np.float32)
    split_labels = np.asarray([rec["split"] for rec in records], dtype="U10")
    arrays = {
        "crops": stack,
        "hfr_px": refs,
        "log_hfr": np.log(refs),
        "reference_hfr_px": refs,
        "measured_hfr_px": np.asarray(measured, dtype=np.float32),
        "run_id": np.arange(run_id_start, run_id_start + n, dtype=np.int64),
        "run_key": np.asarray([rec["source_id"] for rec in records], dtype="U128"),
        "run_key_semantics": np.full(n, "singleton_sample_not_focus_run", dtype="U40"),
        "ranking_eligible": np.zeros(n, dtype=bool),
        "split_group": np.asarray([rec["split_group"] for rec in records], dtype="U100"),
        "split": split_labels.copy(),
        "frame_id": np.asarray([rec["frame_id"] for rec in records], dtype="U128"),
        "star_snr": np.asarray([rec["star_snr"] for rec in records], dtype=np.float32),
        "rig_id": np.full(n, "fixture-rig", dtype="U64"),
        "camera_id": np.full(n, "fixture-camera", dtype="U64"),
        "filter_name": np.full(n, "fixture-filter", dtype="U32"),
        "night_id": np.asarray([rec["split_group"].split(":")[-1] for rec in records], dtype="U32"),
    }
    if style == "hst":
        tiers = np.full(n, "strict_unmasked_aperture", dtype="U24")
        tiers[list(nonstrict_rows)] = "mask_aware_experimental"
        arrays["reference_quality_tier"] = tiers
        arrays["training_eligible"] = (split_labels == "train") & (tiers == "strict_unmasked_aperture")
    for row, value in (split_column_overrides or {}).items():
        arrays["split"][row] = value
    for rec, value in zip(records, arrays["split"].tolist()):
        rec["split"] = value
    npz_path = directory / "fqe.npz"
    np.savez_compressed(npz_path, **arrays)
    npz_sha = _sha256_file(npz_path)
    row_split = np.repeat(np.asarray(splits), rows_per_group)
    registry = {
        "dataset_sha256": npz_sha, "n_samples": n, "group_field": "split_group",
        "strategy": "fixture_group_assignment", "seed": seed,
        "counts": {name: int((row_split == name).sum()) for name in REGISTRY_PARTITIONS},
        "groups": {name: sorted(set(arrays["split_group"][row_split == name].tolist())) for name in REGISTRY_PARTITIONS},
        "indices": {name: np.flatnonzero(row_split == name).tolist() for name in REGISTRY_PARTITIONS},
    }
    (directory / "fqe.splits.json").write_text(json.dumps(registry, indent=2))
    # A few unaccepted observations, as in the real records files.
    unaccepted = [{"source_id": f"rejected-{i}", "frame_id": f"rejected-{i}", "split_group": "family-rejected",
                   "split": "train", "ancestor_id": "family-rejected", "fqe_index": None} for i in range(2)]
    with (directory / "records.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records[: n // 2] + unaccepted + records[n // 2:]:
            fh.write(json.dumps(rec) + "\n")
    manifest = {"dataset_id": source_id, "status": "generated_test_fixture_only",
                "artifacts": [{"path": "fqe.npz", "sha256": npz_sha, "bytes": npz_path.stat().st_size}],
                "training_config_required": {"lambda_rank": 0, "split_registry": "fqe.splits.json"}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    names = _PUBLIC_FIXTURE_FILES
    entry = {
        "role": role, "directory": source_id, "dataset_id": source_id, "npz": "fqe.npz",
        "registry": "fqe.splits.json", "records": "records.jsonl", "manifest": "manifest.json",
        "files": {name: _sha256_file(directory / name) for name in names},
        "target_field": "reference_hfr_px" if style == "sim" else "hfr_px",
        "target_definition_id": target_definition_id,
        "target_equals": ["hfr_px"] if style == "sim" else ["reference_hfr_px", "measured_hfr_px"],
        "log_field": "log_hfr",
        "secondary_fields": ["measured_hfr_px"] if style == "sim" else [],
        "label_definition": f"fixture Gaussian HFR (sigma * sqrt(2 ln 2)), style {style}",
        "units": "pixels; fixture",
        "group_field": "split_group", "sample_key_field": "run_key", "frame_field": "frame_id",
        "batching": "pointwise", "ranking_eligible": False,
        "run_key_semantics": "singleton_sample_not_focus_run",
        "tier_filter": ({"field": "reference_quality_tier", "equals": "strict_unmasked_aperture"}
                        if style == "hst" else None),
        "records_join": {"index_field": "fqe_index",
                         "agree": {"frame_id": "frame_id", "split_group": "split_group", "split": "split",
                                   "source_id": "run_key"},
                         "ancestor_field": "ancestor_id"},
        "strata": {"npz": ["rig_id", "camera_id", "filter_name", "night_id"],
                   "records": ({"profile": "parameters.profile", "motion": "parameters.motion",
                                "moffat_beta": "parameters.moffat_beta"} if style == "sim"
                               else {"hfr_aperture_radius_px": "hfr_aperture_radius_px"})},
        "joins": ({"array_index": "array_index"} if style == "sim"
                  else {"observation_index": "observation_index", "crop_origin_xy": "crop_origin_xy"}),
        "calibration": ({"rule": "sha256_lowest_fraction", "fraction": calibration_fraction, "rounding": "half_up"}
                        if calibration == "sha256_lowest_fraction"
                        else {"rule": "unavailable", "reason": "fixture declares calibration unavailable"}),
        "final_test": {"enabled": bool(final_test_enabled), "reason": "fixture"},
    }
    if style == "hst":
        entry["training_eligible_rule"] = "training_eligible == (split == train and tier passes)"
    contract = (base_contract or empty_contract()).with_source(source_id, entry)
    return FixtureSource(source_id=source_id, data_root=root, directory=directory, entry=entry,
                         contract=contract, files={name: directory / name for name in names}, n_rows=n,
                         group_split={g: splits[g] for g in range(n_groups)})


def _write_local_fixture(root, directory, source_id, splits, rows_per_group, *, seed, run_id_start, calibration,
                         calibration_fraction, target_definition_id, base_contract, crop_shape):
    from data_pipeline import build_focus_quality_dataset as builder
    from labels.psf_measure import SEP_PINNED_VERSION, config_sha256

    rng = np.random.default_rng(seed)
    background_config = {"fixture": "background", "mesh": 64}
    detection_config = {"fixture": "detection", "thresh": 5.0}
    columns = {k: [] for k in ("crops", "hfr_px", "run_id", "position", "star_snr", "run_key", "frame_id",
                               "rig_id", "camera_id", "filter_name", "night_id")}
    for g, _ in enumerate(splits):
        run_key = f"run-{g:04d}"
        for p in range(rows_per_group):
            sigma = 0.8 + 0.4 * abs(p - rows_per_group // 2)
            columns["crops"].append(_fixture_crop(rng, sigma, tuple(crop_shape)))
            columns["hfr_px"].append(sigma * math.sqrt(2.0 * math.log(2.0)))
            columns["run_id"].append(run_id_start + g)
            columns["position"].append(1000 + 50 * p)
            columns["star_snr"].append(30.0 + p)
            columns["run_key"].append(run_key)
            columns["frame_id"].append(f"{run_key}-frame{p:02d}")
            columns["rig_id"].append("fixture-rig")
            columns["camera_id"].append("fixture-camera")
            columns["filter_name"].append("L")
            columns["night_id"].append(f"2026-01-{g + 1:02d}")
    n = len(columns["hfr_px"])
    hfr = np.asarray(columns["hfr_px"], dtype=np.float32)
    dataset = {
        "crops": np.stack(columns["crops"]).astype(np.float32), "hfr_px": hfr, "log_hfr": np.log(hfr),
        "run_id": np.asarray(columns["run_id"], dtype=np.int64),
        "position": np.asarray(columns["position"], dtype=np.int64),
        "star_snr": np.asarray(columns["star_snr"], dtype=np.float32),
    }
    for name in ("run_key", "frame_id", "rig_id", "camera_id", "filter_name", "night_id"):
        dataset[name] = np.asarray(columns[name], dtype=np.str_)
    dataset["label_version"] = np.asarray([builder.LABEL_VERSION] * n, dtype=np.str_)
    dataset["background_config_sha256"] = np.asarray([config_sha256(background_config)] * n, dtype=np.str_)
    dataset["detection_config_sha256"] = np.asarray([config_sha256(detection_config)] * n, dtype=np.str_)
    dataset["saturation_source"] = np.asarray(["declared_sensor_saturation"] * n, dtype=np.str_)
    dataset["split_group"] = dataset["run_key"].copy()
    npz_path = directory / "focus_quality_ds.npz"
    np.savez_compressed(npz_path, **dataset)
    manifest = {
        "dataset": npz_path.name, "sha256": _sha256_file(npz_path), "label_version": builder.LABEL_VERSION,
        "sep_version": SEP_PINNED_VERSION, "background_config": background_config,
        "background_config_sha256": config_sha256(background_config), "detection_config": detection_config,
        "detection_config_sha256": config_sha256(detection_config), "saturation_gate": "enforced",
        "status": "generated_test_fixture_only",
    }
    manifest_path = npz_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    row_split = np.repeat(np.asarray(splits), rows_per_group)
    registry = {
        "dataset_sha256": _sha256_file(npz_path), "n_samples": n, "group_field": "split_group",
        "strategy": "fixture_group_assignment", "seed": seed,
        "counts": {name: int((row_split == name).sum()) for name in REGISTRY_PARTITIONS},
        "groups": {name: sorted(set(dataset["split_group"][row_split == name].tolist())) for name in REGISTRY_PARTITIONS},
        "indices": {name: np.flatnonzero(row_split == name).tolist() for name in REGISTRY_PARTITIONS},
    }
    registry_path = npz_path.with_suffix(".splits.json")
    registry_path.write_text(json.dumps(registry, indent=2))
    names = (npz_path.name, manifest_path.name, registry_path.name)
    entry = {
        "role": "local_verified_run", "directory": source_id, "npz": npz_path.name,
        "registry": registry_path.name, "files": {name: _sha256_file(directory / name) for name in names},
        "target_field": "hfr_px", "target_definition_id": target_definition_id, "group_field": "split_group",
        "batching": "grouped_run",
        "calibration": ({"rule": "sha256_lowest_fraction", "fraction": calibration_fraction}
                        if calibration == "sha256_lowest_fraction"
                        else {"rule": "unavailable", "reason": "fixture declares calibration unavailable"}),
    }
    contract = (base_contract or empty_contract()).with_source(source_id, entry)
    return FixtureSource(source_id=source_id, data_root=root, directory=directory, entry=entry, contract=contract,
                         files={name: directory / name for name in names}, n_rows=n,
                         group_split={g: splits[g] for g in range(len(splits))})


# --------------------------------------------------------------------------- #
# CLI: identity/count report (no target statistics)
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Verify FQE sources and report partition counts (no metrics).")
    parser.add_argument("--source", action="append", required=True, help="source id (repeatable)")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--contract", default=None)
    parser.add_argument("--load", action="store_true",
                        help="also load the development partitions and report returned row counts")
    args = parser.parse_args(argv)
    contract = load_contract(args.contract)
    report = {}
    for source_id in args.source:
        identity = verify_source(source_id, contract=contract, data_root=args.data_root)
        item = {key: identity.get(key) for key in ("role", "files", "registry_sha256", "contract_sha256",
                                                   "registry_counts", "partition_counts", "sealed_test",
                                                   "calibration", "rows_total")}
        if args.load:
            arrays = load_source(source_id, contract=contract, data_root=args.data_root)
            item["loaded"] = {"rows": len(arrays), "by_partition": arrays.identity["rows_returned_by_partition"],
                              "groups_by_partition": {p: len(set(arrays.group_uid[arrays.partition == p].tolist()))
                                                      for p in DEV_PARTITIONS},
                              "crops_shape": list(arrays.crops.shape), "batching": arrays.batching}
        report[source_id] = item
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
