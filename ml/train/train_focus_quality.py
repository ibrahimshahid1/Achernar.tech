"""Train the Focus Quality Estimator (FQE) — Section 7.6.1.

Predicts per-star log(HFR_px) with a heteroscedastic variance head, optionally
regularized by a within-run RankNet ordering loss. Grouped (by-run) splits.

Batching follows the data, not a default:

  * ``grouped_run`` — genuine autofocus runs (local data, the in-process
    synthetic generator). :class:`GroupedBatchSampler` keeps one run per batch so
    the RankNet term sees same-run pairs.
  * ``pointwise`` — independent archive/simulation rows whose run ids are
    singletons. :class:`PointwiseBatchSampler` builds real shuffled minibatches;
    grouped batching would produce one-row batches there.

A nonzero ``lambda_rank`` is refused when no ranking-eligible within-run pairs
exist, rather than silently adding a zero pair loss. The test partition is
evaluated only when the caller passes ``evaluate_test=True``.

Usage:
    python -m train.train_focus_quality --synthetic --epochs 12
    python -m train.train_focus_quality --dataset focus_quality_ds_v1.npz --epochs 50 --evaluate-test

Public archive sources (stellar_sim_v1, hst_acs_corpus_v1) are not loaded by
path here; use ``train.fqe_dev_experiment`` (or ``config["source_ids"]``), which
goes through the source contract and never returns the sealed test partition.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: ``run_key_semantics`` value written by the public importers: every row is an
#: independent sample, not a member of a focus run.
ARCHIVE_SINGLETON_SEMANTICS = "singleton_sample_not_focus_run"

#: Fewest rows a run needs before ``evaluate`` reports its within-run Spearman.
RANK_MIN_ROWS = 4

#: Roles whose rows are never runtime evidence (see pipelines/rebuild_fqe.py).
PUBLIC_ROLES = ("synthetic_reference", "archive_measurement_reference")

#: AdamW settings shared by ``train`` and the development experiment.
ADAMW_LR = 1e-3
ADAMW_WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0


def _load_dataset(config: dict) -> dict:
    if config.get("synthetic") or not config.get("dataset"):
        from data_pipeline.synthetic_psf import make_synthetic_dataset

        return make_synthetic_dataset(
            n_runs=config.get("synthetic_runs", 48), seed=config.get("seed", 0)
        )
    data = np.load(config["dataset"], allow_pickle=False)
    return {k: data[k] for k in data.files}


def is_archive_singleton_dataset(data: dict) -> bool:
    """True when any row declares the public importers' singleton semantics."""
    semantics = data.get("run_key_semantics")
    if semantics is None:
        return False
    return bool(np.any(np.asarray(semantics).astype(str) == ARCHIVE_SINGLETON_SEMANTICS))


def refuse_archive_npz(data: dict, path: str) -> None:
    """Refuse a public archive NPZ on the legacy by-path route.

    That route calls ``splits_for_dataset``, which writes a new registry next to
    an NPZ that lacks one, and it applies no source contract (tier filter,
    calibration groups, sealed test). Archive rows go through
    ``data_pipeline.fqe_sources`` instead.
    """
    if is_archive_singleton_dataset(data):
        raise ValueError(
            f"{path} declares run_key_semantics={ARCHIVE_SINGLETON_SEMANTICS!r}: it is a public "
            "archive/simulation NPZ, not a set of focus runs. Load it through the source "
            "contract (train.fqe_dev_experiment, or config['source_ids'] / rebuild_fqe "
            "--source-id), which verifies hashes and never returns the sealed test partition."
        )


def read_run_key_semantics(path):
    """The ``run_key_semantics`` column of an NPZ, or None, reading nothing else.

    ``np.load`` returns a lazy ``NpzFile``: listing members reads only the zip
    directory, so no crop, target or split column (sealed test rows included)
    is materialised."""
    with np.load(path, allow_pickle=False) as npz:
        if "run_key_semantics" not in npz.files:
            return None
        return np.asarray(npz["run_key_semantics"])


def pinned_public_source(path, sha256: str | None = None) -> str | None:
    """The tracked-contract source whose pinned or excluded files include this
    file's sha256 (every role except local_verified_run), or None.

    Fails closed: an unreadable contract raises instead of returning None,
    because without it a byte copy of a pinned public file (or of an excluded
    one such as the test-era ``fqe_all_references.npz``) could not be
    recognised. The 64-hex digests written in ``excluded_files`` notes count,
    as in ``eval.baselines._public_digests``."""
    import re

    from data_pipeline import fqe_sources

    try:
        contract = fqe_sources.load_contract()
    except (fqe_sources.FqeSourceError, OSError) as exc:
        raise ValueError(
            f"refusing {path}: the source contract is unreadable ({exc}), so a copy of a pinned "
            "public file could not be recognised"
        ) from exc
    digest = sha256
    if digest is None:
        import hashlib

        hasher = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    for source_id, entry in contract.sources.items():
        if entry.get("role") == "local_verified_run":
            continue
        if digest in (entry.get("files") or {}).values():
            return source_id
        for note in (entry.get("excluded_files") or {}).values():
            if digest in re.findall(r"\b[0-9a-f]{64}\b", str(note)):
                return source_id
    return None


def screen_dataset_path(path) -> None:
    """Refuse a public archive NPZ on the legacy by-path route BEFORE any array is read.

    Reads only ``run_key_semantics`` (lazily) and the file's sha256, so a
    refused file never has its crops, targets or sealed test rows loaded and
    ``splits_for_dataset`` never writes a registry next to it."""
    semantics = read_run_key_semantics(path)
    if semantics is not None:
        refuse_archive_npz({"run_key_semantics": semantics}, str(path))
    source_id = pinned_public_source(path)
    if source_id is not None:
        raise ValueError(
            f"{path} is the pinned contract file of public source {source_id!r}; load it through the "
            "source contract (train.fqe_dev_experiment, or config['source_ids'] / rebuild_fqe --source-id)."
        )


class GroupedBatchSampler:
    """Yield batches from one autofocus run at a time.

    A row-shuffled DataLoader usually puts only one crop from a given run in a
    batch, making the within-run RankNet term silently zero. Grouped batches make
    every optimization step contain meaningful same-run pairs while shuffling
    run and crop order each epoch.
    """

    def __init__(self, run_ids, batch_size: int, seed: int = 0):
        self.run_ids = np.asarray(run_ids)
        self.batch_size = max(2, int(batch_size))
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        groups = np.unique(self.run_ids)
        rng.shuffle(groups)
        for group in groups:
            members = np.flatnonzero(self.run_ids == group)
            rng.shuffle(members)
            chunks = [
                members[start : start + self.batch_size]
                for start in range(0, len(members), self.batch_size)
            ]
            if len(chunks) > 1 and chunks[-1].size == 1:
                chunks[-2] = np.concatenate((chunks[-2], chunks[-1]))
                chunks.pop()
            for batch in chunks:
                if batch.size:
                    yield batch.tolist()

    def __len__(self):
        total = 0
        for group in np.unique(self.run_ids):
            count = int(np.sum(self.run_ids == group))
            batches = int(np.ceil(count / self.batch_size))
            if batches > 1 and count % self.batch_size == 1:
                batches -= 1
            total += batches
        return total


class PointwiseBatchSampler:
    """Seeded, shuffled minibatches of independent rows.

    For archive/simulation rows that share no focus run. Each epoch is a fresh
    permutation (seed + epoch, like :class:`GroupedBatchSampler`), cut into
    ``batch_size`` chunks; a one-row tail is merged into the previous batch so
    BatchNorm never trains on a single crop.
    """

    def __init__(self, n_rows: int, batch_size: int, seed: int = 0):
        self.n_rows = int(n_rows)
        if self.n_rows < 1:
            raise ValueError("PointwiseBatchSampler needs at least one row")
        self.batch_size = max(2, int(batch_size))
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(self.n_rows)
        chunks = [
            order[start : start + self.batch_size]
            for start in range(0, self.n_rows, self.batch_size)
        ]
        if len(chunks) > 1 and chunks[-1].size == 1:
            chunks[-2] = np.concatenate((chunks[-2], chunks[-1]))
            chunks.pop()
        for batch in chunks:
            yield batch.tolist()

    def __len__(self):
        batches = int(np.ceil(self.n_rows / self.batch_size))
        if batches > 1 and self.n_rows % self.batch_size == 1:
            batches -= 1
        return batches


class UpdateStream:
    """Epoch-free row stream for a fixed optimizer-update budget.

    The rows for update ``u`` are a pure function of ``(seed, stream, u)``:
    consecutive seeded permutations of ``range(n_rows)`` laid end to end and cut
    into ``per_batch`` slices, so every row is drawn once per pass and a batch
    may straddle two passes. Because no sampler state is carried between
    updates, a resumed run draws exactly the batches an uninterrupted run would.
    """

    def __init__(self, n_rows: int, per_batch: int, seed: int, stream: int = 0):
        if int(n_rows) < 1 or int(per_batch) < 1:
            raise ValueError("UpdateStream needs n_rows >= 1 and per_batch >= 1")
        self.n_rows = int(n_rows)
        self.per_batch = int(per_batch)
        self.seed = int(seed)
        self.stream = int(stream)
        self._cache: dict[int, np.ndarray] = {}

    def _permutation(self, epoch: int) -> np.ndarray:
        if epoch not in self._cache:
            if len(self._cache) > 2:
                self._cache.clear()
            rng = np.random.default_rng([self.seed, self.stream, epoch])
            self._cache[epoch] = rng.permutation(self.n_rows)
        return self._cache[epoch]

    def rows(self, update: int) -> np.ndarray:
        position = int(update) * self.per_batch
        out: list[np.ndarray] = []
        needed = self.per_batch
        while needed:
            epoch, offset = divmod(position, self.n_rows)
            take = self._permutation(epoch)[offset : offset + needed]
            out.append(take)
            needed -= take.size
            position += take.size
        return np.concatenate(out)

    def passes(self, updates: int) -> float:
        """Equivalent passes over the pool after ``updates`` updates."""
        return float(int(updates) * self.per_batch / self.n_rows)


def choose_batching(run_ids, declared: str | None = None) -> tuple[str, str]:
    """Return ``(batching, reason)``.

    ``pointwise`` when the data declares it, or when no run has two rows (a
    grouped sampler would then emit one-row batches). Otherwise ``grouped_run``.
    """
    if declared not in (None, "pointwise", "grouped_run"):
        raise ValueError(f"unknown batching {declared!r}; expected 'pointwise' or 'grouped_run'")
    if declared == "pointwise":
        return "pointwise", "declared pointwise by the source contract"
    counts = np.unique(np.asarray(run_ids), return_counts=True)[1]
    if counts.size == 0 or int(counts.max()) < 2:
        return "pointwise", "no run has >= 2 rows"
    if declared == "grouped_run":
        return "grouped_run", "declared grouped_run (genuine runs)"
    return "grouped_run", "multi-row runs present"


def ranking_pair_runs(run_ids, ranking_eligible=None) -> int:
    """Number of runs with at least two ranking-eligible rows."""
    run_ids = np.asarray(run_ids)
    if ranking_eligible is None:
        eligible = np.ones(run_ids.shape[0], dtype=bool)
    else:
        eligible = np.asarray(ranking_eligible, dtype=bool)
    if not eligible.any():
        return 0
    counts = np.unique(run_ids[eligible], return_counts=True)[1]
    return int(np.sum(counts >= 2))


def require_rank_compatible(
    lambda_rank: float,
    *,
    batching: str,
    run_ids,
    ranking_eligible=None,
    roles=(),
) -> None:
    """Refuse ``lambda_rank > 0`` without ranking-eligible within-run pairs."""
    if not lambda_rank or float(lambda_rank) <= 0:
        return
    pair_runs = ranking_pair_runs(run_ids, ranking_eligible)
    if batching != "grouped_run" or pair_runs == 0:
        role_text = ", ".join(sorted(set(map(str, roles)))) or "unknown"
        raise ValueError(
            f"lambda_rank={lambda_rank} > 0 requires ranking-eligible within-run pairs, but "
            f"this training data has none (data roles: {role_text}; batching: {batching}; "
            f"runs with >= 2 ranking-eligible rows: {pair_runs}). Archive and simulation rows "
            "are independent samples with no focus-run ordering; a within-run pair loss "
            "would be identically zero. Set lambda_rank=0."
        )


def rank_pair_ids(run_ids, ranking_eligible=None) -> np.ndarray:
    """Run ids for the RankNet term; ineligible rows get unique negative ids so
    they are never paired. Equal to ``run_ids`` when every row is eligible."""
    run_ids = np.asarray(run_ids, dtype=np.int64)
    if ranking_eligible is None:
        return run_ids.copy()
    eligible = np.asarray(ranking_eligible, dtype=bool)
    out = run_ids.copy()
    ineligible = np.flatnonzero(~eligible)
    out[ineligible] = -(ineligible.astype(np.int64) + 1)
    return out


def resolve_device(requested: str | None = None) -> str:
    """``None``: CUDA if available, else CPU (the trainer's historical default).
    ``"auto"``: CUDA, then MPS, then CPU. An explicit device must be available."""
    import torch

    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if mps_available else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("device 'mps' requested but torch.backends.mps is not available")
        return "mps"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"device {requested!r} requested but CUDA is not available")
        return requested
    raise ValueError(f"unknown device {requested!r}; expected cpu, mps, cuda, auto")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy's global generator (used by RankNet pair
    subsampling) and torch on every device."""
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def fqe_objective(mu, log_var, target, *, mean_warmup: bool, lambda_rank: float = 0.0, rank_ids=None):
    """The trainer's objective, unchanged: mean-only MSE during warm-up, then
    heteroscedastic NLL, plus ``lambda_rank`` x within-run RankNet."""
    import torch

    from models.fqe_losses import heteroscedastic_nll, ranknet_within_run

    if mean_warmup:
        # Teach the mean/ranking signal before the variance head is allowed to
        # explain residuals (Nix-Weigend failure mode).
        loss = torch.nn.functional.mse_loss(mu, target)
    else:
        loss = heteroscedastic_nll(mu, log_var, target)
    if lambda_rank > 0:
        loss = loss + lambda_rank * ranknet_within_run(mu, target, rank_ids)
    return loss


def state_is_finite(state_dict) -> bool:
    """True when every floating tensor in a state dict is finite."""
    import torch

    for value in state_dict.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if not bool(torch.isfinite(value).all().item()):
                return False
    return True


def evaluate(model, crops, log_hfr, hfr_px, run_id, device, batch_size=256):
    """Return per-star HFR MAE (px), 95% interval coverage, and mean within-run
    Spearman rank correlation of predicted vs. true HFR.

    The rank correlation is the string ``"unavailable"`` (with a reason in
    ``within_run_rank_status``) when no run has ``RANK_MIN_ROWS`` rows with
    varying targets — singleton archive rows have no sweep to order."""
    import torch
    from scipy.stats import spearmanr

    model.eval()
    mus, lvs = [], []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            xb = torch.as_tensor(crops[i : i + batch_size][:, None], device=device)
            mu, lv = model(xb)
            mus.append(mu.cpu().numpy())
            lvs.append(lv.cpu().numpy())
    mu = np.concatenate(mus)
    lv = np.clip(np.concatenate(lvs), -8.0, 8.0)

    hfr_pred = np.exp(mu)
    mae = float(np.mean(np.abs(hfr_pred - hfr_px)))
    sigma = np.sqrt(np.exp(lv))
    covered = np.abs(log_hfr - mu) <= 1.96 * sigma
    coverage = float(np.mean(covered))

    rhos = []
    long_runs = 0
    for run in np.unique(run_id):
        m = run_id == run
        if m.sum() >= RANK_MIN_ROWS:
            long_runs += 1
            if np.ptp(hfr_px[m]) > 0:
                rho = spearmanr(mu[m], hfr_px[m]).statistic
                if np.isfinite(rho):
                    rhos.append(rho)
    if rhos:
        rank_rho: float | str = float(np.mean(rhos))
        rank_status = "ok"
    else:
        rank_rho = "unavailable"
        rank_status = (
            f"unavailable: no run has >= {RANK_MIN_ROWS} rows"
            if long_runs == 0
            else f"unavailable: no run with >= {RANK_MIN_ROWS} rows has a finite rank correlation"
        )
    return {
        "hfr_mae_px": mae,
        "coverage_95": coverage,
        "within_run_rank_rho": rank_rho,
        "within_run_rank_status": rank_status,
        "within_run_rank_runs": len(rhos),
        "n_rows": int(len(crops)),
        "finite_prediction_fraction": float(np.mean(np.isfinite(mu) & np.isfinite(lv))) if len(mu) else 0.0,
    }


def _format_rho(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else str(value)


def load_development_source(fqe_sources, source_id: str, partitions, *, contract=None, data_root=None):
    """``fqe_sources.load_source`` that tolerates an unavailable calibration
    partition (HST): the calibration rows are then simply absent."""
    partitions = tuple(partitions)
    try:
        return fqe_sources.load_source(
            source_id, partitions=partitions, contract=contract, data_root=data_root
        )
    except fqe_sources.CalibrationUnavailable:
        remaining = tuple(p for p in partitions if p != "calibration")
        if remaining == partitions or not remaining:
            raise
        return fqe_sources.load_source(
            source_id, partitions=remaining, contract=contract, data_root=data_root
        )


def _prepare_source_data(config: dict) -> dict:
    """Training arrays from contract sources: train_fit for updates, val for
    epoch selection. No test rows are requested or returned."""
    from data_pipeline import fqe_sources

    contract = config.get("contract")
    if contract is None and config.get("contract_path"):
        contract = fqe_sources.load_contract(config["contract_path"])
    sources = [
        load_development_source(
            fqe_sources,
            source_id,
            ("train_fit", "val"),
            contract=contract,
            data_root=config.get("data_root"),
        )
        for source_id in config["source_ids"]
    ]
    if len(sources) > 1:
        # Identity collision check (and declared common reference for differing targets).
        fqe_sources.make_mixture(sources, partition="train_fit", contract=contract)
    partition = np.concatenate([np.asarray(s.partition).astype(str) for s in sources])
    group_uid = np.concatenate([np.asarray(s.group_uid).astype(str) for s in sources])
    batching_declared = {str(s.batching) for s in sources}
    grouped = batching_declared == {"grouped_run"}
    if grouped:
        run_id = np.unique(group_uid, return_inverse=True)[1].astype(np.int64)
    else:
        # Pointwise rows are their own "runs": no cross-row ordering exists, so
        # evaluate() must not compute a within-group Spearman over them.
        run_id = np.arange(partition.size, dtype=np.int64)
    batching, reason = choose_batching(run_id, "grouped_run" if grouped else "pointwise")
    return {
        "crops": np.concatenate([np.asarray(s.crops, dtype=np.float32) for s in sources]),
        "log_hfr": np.concatenate([np.asarray(s.log_target, dtype=np.float32) for s in sources]),
        "hfr_px": np.concatenate([np.asarray(s.target_hfr_px, dtype=np.float32) for s in sources]),
        "run_id": run_id,
        "ranking_eligible": np.concatenate([np.asarray(s.ranking_eligible, dtype=bool) for s in sources]),
        "splits": {
            "train": np.flatnonzero(partition == "train_fit"),
            "val": np.flatnonzero(partition == "val"),
        },
        "batching": batching,
        "batching_reason": reason,
        "roles": sorted({str(s.role) for s in sources}),
        "source_ids": [str(s.source_id) for s in sources],
        "identity": {str(s.source_id): s.identity for s in sources},
    }


def _prepare_training_data(config: dict) -> dict:
    if config.get("source_ids"):
        return _prepare_source_data(config)

    from data_pipeline.split_registry import create_group_splits, splits_for_dataset

    synthetic = bool(config.get("synthetic") or not config.get("dataset"))
    if not synthetic:
        screen_dataset_path(config["dataset"])  # before any array is materialised
    ds = _load_dataset(config)
    if not synthetic:
        refuse_archive_npz(ds, config["dataset"])
    run_id = ds["run_id"].astype(np.int64)
    split_groups = ds.get("split_group", run_id)
    if not synthetic:
        splits = splits_for_dataset(
            config["dataset"],
            ds,
            seed=config.get("seed", 0),
            registry_path=config.get("split_registry"),
        )
    else:
        splits = create_group_splits(split_groups, seed=config.get("seed", 0))
    ranking_eligible = (
        np.asarray(ds["ranking_eligible"], dtype=bool) if "ranking_eligible" in ds else None
    )
    batching, reason = choose_batching(run_id, config.get("batching"))
    role = "fixture_only" if synthetic else config.get("data_role", "unclassified_npz")
    return {
        "crops": ds["crops"].astype(np.float32),
        "log_hfr": ds["log_hfr"].astype(np.float32),
        "hfr_px": ds["hfr_px"].astype(np.float32),
        "run_id": run_id,
        "ranking_eligible": ranking_eligible,
        "splits": splits,
        "batching": batching,
        "batching_reason": reason,
        "roles": [role],
        "source_ids": [],
        "identity": {},
    }


def train(config: dict, *, evaluate_test: bool = False) -> dict:
    """Train with epoch-level best-validation selection.

    The test partition is evaluated only with ``evaluate_test=True`` (genuine
    local runs; ``rebuild_fqe`` decides by data role). Contract sources
    (``config["source_ids"]``) never expose test rows at all.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from models.focus_quality_cnn import FocusQualityEstimator
    from models.fqe_losses import require_finite

    seed = config.get("seed", 0)
    seed_everything(seed)
    device = resolve_device(config.get("device"))

    data = _prepare_training_data(config)
    crops, log_hfr, hfr_px = data["crops"], data["log_hfr"], data["hfr_px"]
    run_id = data["run_id"]
    splits = data["splits"]
    tr, va = splits["train"], splits["val"]
    batching = data["batching"]

    lambda_rank = config.get("lambda_rank", 0.2)
    lambda_rank = 0.2 if lambda_rank is None else float(lambda_rank)
    eligible = data["ranking_eligible"]
    require_rank_compatible(
        lambda_rank,
        batching=batching,
        run_ids=run_id[tr],
        ranking_eligible=None if eligible is None else eligible[tr],
        roles=data["roles"],
    )
    rank_ids = rank_pair_ids(run_id, eligible)

    tr_ds = TensorDataset(
        torch.as_tensor(crops[tr][:, None]),
        torch.as_tensor(log_hfr[tr]),
        torch.as_tensor(rank_ids[tr]),
    )
    batch_size = config.get("batch_size", 64)
    if batching == "grouped_run":
        batch_sampler = GroupedBatchSampler(run_id[tr], batch_size, seed=seed)
    else:
        batch_sampler = PointwiseBatchSampler(tr.size, batch_size, seed=seed)
    loader = DataLoader(tr_ds, batch_sampler=batch_sampler)

    model = FocusQualityEstimator(pretrained=config.get("pretrained", False)).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.get("lr", ADAMW_LR), weight_decay=ADAMW_WEIGHT_DECAY
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.get("epochs", 12))
    epochs = config.get("epochs", 12)
    variance_warmup_epochs = config.get("variance_warmup_epochs")
    if variance_warmup_epochs is None:
        variance_warmup_epochs = max(1, min(3, epochs // 5))

    best = {"val_mae": float("inf"), "epoch": -1, "state": None}
    for epoch in range(epochs):
        model.train()
        for xb, yb, rb in loader:
            xb, yb = xb.to(device), yb.to(device)
            mu, lv = model(xb)
            loss = fqe_objective(
                mu,
                lv,
                yb,
                mean_warmup=epoch < variance_warmup_epochs,
                lambda_rank=lambda_rank,
                rank_ids=rb.numpy(),
            )
            # A NaN/inf loss would reach every weight through the gradient; stop
            # here instead of continuing and saving poisoned weights.
            require_finite(loss, f"training loss (epoch {epoch})")
            opt.zero_grad()
            loss.backward()
            # Guard the warmup->NLL transition: once the variance head switches on,
            # inv_var = exp(-log_var) (up to exp(8) ~ 3000x) amplifies the mean-head
            # gradient and can transiently diverge (the epoch-3 MAE ~7e10 spike).
            # heteroscedastic_nll clamps log_var's *value* but not the gradient, so
            # clip the global grad norm — standard, and cheap insurance on noisy data.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            opt.step()
        sched.step()

        metrics = evaluate(model, crops[va], log_hfr[va], hfr_px[va], run_id[va], device)
        print(
            f"epoch {epoch:2d}  val HFR MAE {metrics['hfr_mae_px']:.3f}px  "
            f"cov95 {metrics['coverage_95']:.2f}  rank_rho {_format_rho(metrics['within_run_rank_rho'])}"
        )
        if metrics["hfr_mae_px"] < best["val_mae"]:
            best = {
                "val_mae": metrics["hfr_mae_px"],
                "epoch": epoch,
                "state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "metrics": metrics,
            }

    if best["state"] is None or not state_is_finite(best["state"]):
        raise RuntimeError(
            "no epoch produced a finite validation MAE with finite weights; refusing to save "
            "a checkpoint (previously the last-epoch weights were saved silently)"
        )
    model.load_state_dict(best["state"])
    out_path = config.get("out", "focus_quality_fqe.pt")
    torch.save(model.state_dict(), out_path)
    print(f"✓ Saved best model (epoch {best['epoch']}, val MAE {best['val_mae']:.3f}px) -> {out_path}")

    # Final test metrics on the held-out group: opt-in only.
    te = splits.get("test")
    if evaluate_test and te is not None and te.size:
        test_metrics = evaluate(model, crops[te], log_hfr[te], hfr_px[te], run_id[te], device)
        test_status = "evaluated"
    else:
        test_metrics = {}
        test_status = (
            "not_evaluated (evaluate_test=False)"
            if not evaluate_test
            else "not_evaluated (no test partition available)"
        )
    return {
        "best_val": best.get("metrics", {}),
        "best_epoch": best["epoch"],
        "test": test_metrics,
        "test_evaluated": test_status == "evaluated",
        "test_status": test_status,
        "model_path": out_path,
        "split_counts": {name: int(index.size) for name, index in splits.items()},
        "batching": batching,
        "batching_reason": data["batching_reason"],
        "lambda_rank": lambda_rank,
        "device": device,
        "seed": seed,
        "data_roles": data["roles"],
        "source_ids": data["source_ids"],
        "source_identity": data["identity"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-runs", type=int, default=48)
    parser.add_argument("--split-registry")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=ADAMW_LR)
    parser.add_argument("--lambda-rank", type=float, default=0.2)
    parser.add_argument("--variance-warmup-epochs", type=int)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu | mps | cuda | auto (default: cuda if available, else cpu)")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="also evaluate the registry's test split (genuine local runs only; off by default)",
    )
    parser.add_argument("--out", default="focus_quality_fqe.pt")
    args = parser.parse_args()
    cli_config = vars(args)
    evaluate_test_flag = cli_config.pop("evaluate_test")
    result = train(cli_config, evaluate_test=evaluate_test_flag)
    print("RESULT:", result)
