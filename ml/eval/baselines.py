"""
baselines — frozen classical baselines the FQE must beat (Section 7.7.x).

Per the review directive, freeze these BEFORE more CNN training so a learned
model has to earn its place:

  * constant       — predict the fit-partition median HFR for every crop. The
                     sanity floor; if the CNN cannot beat this on MAE it has
                     learned nothing.
  * direct_hfr     — a classical curve-of-growth HFR measured straight from the
                     crop (no learning), EXECUTED and TIMED per crop (not assumed
                     perfect: the audit rightly flagged the old hard-coded 0-MAE
                     entry). Method: the custom Python crop helpers
                     ``python_crop_iterative_clip`` (one global sky for the crop)
                     + ``python_signed_curve_of_growth`` — NOT SEP, NOT the SEP
                     labeler that produced the labels, and NOT the daemon's Rust
                     path (which has its own crop gates and estimator). Caveat:
                     it runs on the normalized [0,1] crops the model sees; the
                     p1/p99 clip slightly flattens bright cores, so this is a
                     deployable-baseline number, not the label oracle. The "do
                     we even need a CNN?" question is answered by comparing the
                     CNN against THIS row.
  * tenengrad      — squared-gradient sharpness energy (Pertuz 2013 gradient
                     family), ranked within run.
  * norm_variance  — normalized grey-level variance (Pertuz statistical family).

For focus, the decisive metric is WITHIN-RUN rank correlation (does the operator
order a sweep correctly?), evaluated on grouped splits. HFR MAE is reported where
a baseline predicts an HFR directly.

Evaluation partition (explicit, never a silent test default)
-----------------------------------------------------------
:func:`freeze_baselines` takes a keyword-only ``partition``. Development
callers pass ``partition="val"``. The historical behaviour (fit on train,
score the test split), which ``pipelines/rebuild_fqe.py`` uses for genuine
local runs, is available only through an explicit ``partition="test"`` on a
legacy dataset dict. Source-aware arrays (``data_pipeline.fqe_sources.
SourceArrays``) never contain test rows; asking them for ``test`` raises
``SealedPartitionError``.

Rank metrics
------------
A within-run Spearman needs a genuine run with at least ``MIN_RUN_ROWS``
finite rows and varying HFR. When no run qualifies (e.g. public archive rows,
which are singleton samples, or a constant predictor, which carries no
ordering) the metric is the string ``"unavailable"`` with a reason — never a
NaN posing as a number and never a hard-coded 0.0. Use :func:`is_available`
before arithmetic on it.

Label provenance: the legacy dict path does not filter rows by label version,
so its numbers are exploratory, not a *qualified* evaluation. A qualified
comparison loads labels through
``data_pipeline.build_focus_quality_dataset.load_qualified_dataset`` (local
runs) or ``data_pipeline.fqe_sources.load_source`` (public archive sources),
which excludes ``legacy_unverified`` rows (BGE-20).

Legacy ``--dataset`` path: refuses any public file (under the data root, the
importers' ``run_key_semantics`` archive marker, or a sha256 the source contract
pins or excludes) and scores only an EXISTING frozen split registry
(``<npz>.splits.json`` or ``--split-registry``); it never creates a split.

Run: python -m eval.baselines --source stellar_sim_v1 --partition val
     python -m eval.baselines --dataset focus_quality_ds_v1.npz --partition val
     python -m eval.baselines --synthetic --partition val
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UNAVAILABLE = "unavailable"
MIN_RUN_ROWS = 4
#: Legacy dict path: the partitions a caller may score (fit is always "train").
LEGACY_EVAL_PARTITIONS = ("val", "test")
#: Source-aware path: development partitions only (test is sealed).
SOURCE_EVAL_PARTITIONS = ("val", "calibration")
DIRECT_HFR_METHOD = "python_crop_iterative_clip+python_signed_curve_of_growth"
DIRECT_HFR_DEFINITION = (
    "labels.psf_measure.estimate_background_detailed (iterative 3-sigma MAD clip over the whole crop, "
    "5 passes; level = median, rms = std of the retained pixels; custom, not SEP) + "
    "labels.psf_measure.curve_of_growth_hfr(crop, level, sky_rms=rms) (centroid of the residual above "
    "3*rms, whole-pixel signed curve of growth, pass 1 capped at min(20, min(h,w)/2-1) px, pass 2 aperture "
    "clip(4*HFR1, 6, min(h,w)/2-1) px). Not sep.flux_radius and not the daemon's Rust estimator."
)
#: Per-crop reasons of :func:`direct_hfr_measure_detailed`.
DIRECT_HFR_REASONS = {
    "bg_unresolved": "the crop background level or noise is unavailable/unresolved (no finite pixel or zero MAD)",
    "no_core_above_3sigma": "no pixel lies above level + 3*rms, so no centroid core exists (on normalized "
                            "crops: the 1/99 stretch puts level + 3*rms above the 1.0 clip ceiling)",
    "cog_none": "a core exists but the curve of growth found no positive enclosed flux",
}
_REQUIRED = object()


def is_available(value) -> bool:
    """True when ``value`` is a finite number (not ``"unavailable"``/None/NaN)."""
    if value is None or isinstance(value, (str, bytes)):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def tenengrad(crop: np.ndarray) -> float:
    c = crop.astype(np.float64)
    gx = np.diff(c, axis=1)
    gy = np.diff(c, axis=0)
    return float((gx**2).mean() + (gy**2).mean())


def normalized_variance(crop: np.ndarray) -> float:
    c = crop.astype(np.float64)
    mu = c.mean()
    return float(((c - mu) ** 2).mean() / (mu + 1e-9))


def _within_run_rho(score, hfr_px, run_id, sign=-1.0):
    """Mean within-run Spearman of ``sign*score`` vs true HFR. `sign=-1` for
    sharpness operators (higher score = sharper = lower HFR); `sign=+1` for
    direct HFR predictors. Non-finite scores (failed measurements) are masked
    per run rather than poisoning the whole run. NaN when no run qualifies
    (internal helper; public results go through :func:`within_run_rank`)."""
    value, _ = _within_run_rho_detail(score, hfr_px, run_id, sign)
    return value


def _within_run_rho_detail(score, hfr_px, run_id, sign):
    from scipy.stats import spearmanr

    score = np.asarray(score, dtype=np.float64)
    hfr_px = np.asarray(hfr_px, dtype=np.float64)
    run_id = np.asarray(run_id)
    rhos = []
    for run in np.unique(run_id):
        m = (run_id == run) & np.isfinite(score)
        if m.sum() >= MIN_RUN_ROWS and np.ptp(hfr_px[m]) > 0:
            rho = spearmanr(sign * score[m], hfr_px[m]).statistic
            if np.isfinite(rho):
                rhos.append(rho)
    return (float(np.mean(rhos)) if rhos else float("nan")), len(rhos)


def within_run_rank(score, hfr_px, run_id, sign=-1.0, *, unavailable_reason: str | None = None) -> dict:
    """``{"within_run_rho": float | "unavailable", "within_run_rho_runs": n, "within_run_rho_reason"}``.

    ``unavailable_reason`` forces ``"unavailable"`` (e.g. singleton archive
    rows or a constant predictor) without computing anything. Otherwise the
    metric is the mean Spearman over runs with >= ``MIN_RUN_ROWS`` finite rows
    and varying HFR, or ``"unavailable"`` when no run qualifies.
    """
    if unavailable_reason is not None:
        return {"within_run_rho": UNAVAILABLE, "within_run_rho_runs": 0, "within_run_rho_reason": unavailable_reason}
    if run_id is None or len(run_id) == 0:
        return {"within_run_rho": UNAVAILABLE, "within_run_rho_runs": 0,
                "within_run_rho_reason": "no evaluation rows"}
    value, n_runs = _within_run_rho_detail(score, hfr_px, run_id, sign)
    if n_runs == 0 or not np.isfinite(value):
        return {"within_run_rho": UNAVAILABLE, "within_run_rho_runs": 0,
                "within_run_rho_reason": f"no run has >= {MIN_RUN_ROWS} finite rows with varying HFR"}
    return {"within_run_rho": float(value), "within_run_rho_runs": int(n_runs), "within_run_rho_reason": None}


def direct_hfr_measure_detailed(crop: np.ndarray) -> dict:
    """The deployable classical crop baseline with an explicit per-crop status.

    Runs exactly :func:`direct_hfr_measure`'s algorithm and adds why a crop
    produced no width: ``{"hfr_px": float | None, "status": "measured" |
    "unmeasured", "reason": None | one of DIRECT_HFR_REASONS, "background":
    {"level", "rms", "level_status", "noise_status", "reasons"}}``. A missing
    width is ``None``, never 0.
    """
    from labels.psf_measure import curve_of_growth_hfr, estimate_background_detailed

    data = np.asarray(crop, dtype=np.float64)
    bg = estimate_background_detailed(data)
    background = {"level": bg.level, "rms": bg.rms, "level_status": bg.level_status,
                  "noise_status": bg.noise_status, "reasons": list(bg.reasons), "n_used": bg.n_used}
    sky, rms = bg.level, bg.rms
    if not (sky is not None and rms is not None and np.isfinite(sky) and np.isfinite(rms)):
        return {"hfr_px": None, "status": "unmeasured", "reason": "bg_unresolved", "background": background}
    hfr = curve_of_growth_hfr(data, sky, sky_rms=rms)
    if hfr is not None and np.isfinite(hfr) and hfr > 0:
        return {"hfr_px": float(hfr), "status": "measured", "reason": None, "background": background}
    # Classify the failure with the same core test curve_of_growth_hfr applies.
    thr = 3.0 * rms if rms > 0 else 0.0
    core_total = float(np.clip(data - sky - thr, 0.0, None).sum())
    reason = "no_core_above_3sigma" if core_total <= 0 else "cog_none"
    return {"hfr_px": None, "status": "unmeasured", "reason": reason, "background": background}


def direct_hfr_measure(crop: np.ndarray) -> float:
    """The deployable classical baseline: the custom crop sky estimate
    (``python_crop_iterative_clip``) + aperture-bounded signed curve-of-growth
    HFR (``python_signed_curve_of_growth``). A Python reference of the same
    idea as the daemon's crop path, not the daemon's implementation and not
    SEP. A crop whose sky level is unavailable or whose noise is unresolved
    (``estimate_background`` returns nan) gets nan — an invalid baseline
    measurement, counted against ``measured_fraction`` — never a width."""
    result = direct_hfr_measure_detailed(crop)
    return float(result["hfr_px"]) if result["hfr_px"] is not None else float("nan")


def _mae(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(pred)
    if not finite.any():
        return UNAVAILABLE
    return float(np.mean(np.abs(pred[finite] - target[finite])))


def _score_partition(crops, hfr_fit, hfr_eval, run_eval, *, rank_unavailable: str | None) -> dict:
    train_median = float(np.median(np.asarray(hfr_fit, dtype=np.float64)))
    hfr_eval = np.asarray(hfr_eval, dtype=np.float64)
    ten = np.array([tenengrad(c) for c in crops], dtype=np.float64)
    nvar = np.array([normalized_variance(c) for c in crops], dtype=np.float64)

    # direct_hfr: actually run the classical measurement per crop, and time it,
    # so the row reports what deploying "no CNN, just measure" would cost/score.
    t0 = time.perf_counter()
    detailed = [direct_hfr_measure_detailed(c) for c in crops]
    ms_per_crop = (time.perf_counter() - t0) * 1000.0 / max(len(detailed), 1)
    direct = np.array([d["hfr_px"] if d["hfr_px"] is not None else np.nan for d in detailed], dtype=np.float64)
    reasons: dict = {}
    for d in detailed:
        if d["reason"]:
            reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1

    def rank(score, sign):
        return within_run_rank(score, hfr_eval, run_eval, sign=sign, unavailable_reason=rank_unavailable)

    constant_rank = within_run_rank(None, None, None, unavailable_reason="a constant predictor has no within-run ordering")
    return {
        "train_median_hfr_px": train_median,
        "constant": dict({"hfr_mae_px": float(np.mean(np.abs(hfr_eval - train_median))) if hfr_eval.size else UNAVAILABLE},
                         **constant_rank),
        "direct_hfr": dict({
            "hfr_mae_px": _mae(direct, hfr_eval),
            "measured_fraction": float(np.mean(np.isfinite(direct))) if direct.size else UNAVAILABLE,
            "unmeasured_reasons": reasons,
            "ms_per_crop": float(ms_per_crop),
            "method": DIRECT_HFR_METHOD,
            "definition": DIRECT_HFR_DEFINITION,
            "note": "custom crop helpers, not SEP; measured on normalized [0,1] crops (deployable baseline, not the label oracle)",
        }, **rank(direct, 1.0)),
        "tenengrad": rank(ten, -1.0),
        "norm_variance": rank(nvar, -1.0),
    }


def _is_source_arrays(dataset) -> bool:
    return all(hasattr(dataset, name) for name in ("source_id", "partition", "target_hfr_px", "crops", "identity"))


def _membership_sha256(uids) -> str:
    return hashlib.sha256("\n".join(sorted(str(u) for u in uids)).encode("utf-8")).hexdigest()


def _freeze_source_arrays(arrays, partition: str, fit_partition: str) -> dict:
    from data_pipeline.fqe_sources import SealedPartitionError

    if partition in ("test", "test_sealed"):
        raise SealedPartitionError(
            f"{arrays.source_id}: source-aware baselines never score the sealed test partition; "
            "development callers pass partition='val'"
        )
    if partition not in SOURCE_EVAL_PARTITIONS:
        raise ValueError(f"partition must be one of {SOURCE_EVAL_PARTITIONS} for source arrays, not {partition!r}")
    if fit_partition != "train_fit":
        raise ValueError("source-aware baselines fit on train_fit only (calibration and val groups are excluded)")
    fit = arrays.partition_indices(fit_partition)
    ev = arrays.partition_indices(partition)
    if fit.size == 0:
        raise ValueError(f"{arrays.source_id}: no {fit_partition} rows were loaded to fit the constant baseline")
    if ev.size == 0:
        raise ValueError(f"{arrays.source_id}: no {partition} rows were loaded to evaluate")
    target = np.asarray(arrays.target_hfr_px, dtype=np.float64)
    if arrays.batching == "grouped_run":
        eligible = np.asarray(arrays.ranking_eligible, dtype=bool)[ev]
        run_eval = np.where(eligible, np.asarray(arrays.group_uid)[ev], np.asarray(arrays.sample_uid)[ev])
        rank_unavailable = None if eligible.any() else "no ranking-eligible genuine run in the evaluation rows"
    else:
        run_eval = np.asarray(arrays.sample_uid)[ev]
        rank_unavailable = ("pointwise source: singleton archive rows have no within-run focus ordering "
                            "(ranking_eligible false)")
    results = _score_partition(arrays.crops[ev], target[fit], target[ev], run_eval, rank_unavailable=rank_unavailable)
    results.update({
        "partition": partition,
        "fit_partition": fit_partition,
        "n_fit": int(fit.size),
        "n_eval": int(ev.size),
        "source": {
            "source_id": arrays.source_id,
            "role": arrays.role,
            "contract_sha256": arrays.identity.get("contract_sha256"),
            "npz_sha256": arrays.identity.get("npz_sha256"),
            "registry_sha256": arrays.identity.get("registry_sha256"),
            "target_field": arrays.identity.get("target_field"),
            "target_definition_id": arrays.identity.get("target_definition_id"),
            "eval_membership_sha256": _membership_sha256(np.asarray(arrays.sample_uid)[ev]),
            "fit_membership_sha256": _membership_sha256(np.asarray(arrays.sample_uid)[fit]),
        },
        "runtime_eligible": False,
    })
    return results


def freeze_baselines(dataset, split_seed: int = 0, splits: dict | None = None, *, partition=_REQUIRED,
                     fit_partition: str | None = None) -> dict:
    """Frozen classical baselines on an explicitly named evaluation partition.

    ``dataset`` is either a ``data_pipeline.fqe_sources.SourceArrays``
    (source-aware path: fit on ``train_fit``, score ``partition`` in
    ``("val", "calibration")``; ``test`` is sealed) or a legacy dict of arrays
    (``crops``, ``hfr_px``, ``run_id``, optional ``split_group``) with
    ``splits`` (``train``/``val``/``test`` row indices; created from
    ``split_group`` with ``split_seed`` when omitted): fit on ``train``, score
    ``partition`` in ``("val", "test")``.

    ``partition`` is keyword-only and has no default: development callers pass
    ``"val"``; ``"test"`` is an explicit final evaluation of genuine local
    runs (``pipelines/rebuild_fqe.py``).
    """
    if partition is _REQUIRED:
        raise TypeError(
            "freeze_baselines() requires the keyword argument partition=...: development callers pass "
            "partition='val'; scoring the test split needs an explicit partition='test' (legacy local-run "
            "datasets only). There is no default."
        )
    if _is_source_arrays(dataset):
        if splits is not None:
            raise ValueError("source arrays carry their frozen partitions; do not pass splits")
        return _freeze_source_arrays(dataset, partition, fit_partition or "train_fit")

    from data_pipeline.split_registry import create_group_splits

    if partition not in LEGACY_EVAL_PARTITIONS:
        raise ValueError(f"partition must be one of {LEGACY_EVAL_PARTITIONS} for a legacy dataset, not {partition!r}")
    if fit_partition not in (None, "train"):
        raise ValueError("legacy datasets fit the constant on the train split only")
    crops = dataset["crops"]
    hfr_px = np.asarray(dataset["hfr_px"])
    run_id = np.asarray(dataset["run_id"])
    splits = splits or create_group_splits(dataset.get("split_group", run_id), seed=split_seed)
    tr, ev = np.asarray(splits["train"]), np.asarray(splits[partition])
    if tr.size == 0 or ev.size == 0:
        raise ValueError(f"empty train ({tr.size}) or {partition} ({ev.size}) split")
    results = _score_partition(crops[ev], hfr_px[tr], hfr_px[ev], run_id[ev], rank_unavailable=None)
    results.update({"partition": partition, "fit_partition": "train", "n_fit": int(tr.size), "n_eval": int(ev.size)})
    return results


def _format_value(value, fmt: str) -> str:
    return format(value, fmt) if is_available(value) else str(value)


def _print_results(label: str, results: dict) -> None:
    print(f"Frozen classical baselines ({label}; fit {results['fit_partition']} -> score {results['partition']}):")
    print(f"  fit-partition median HFR = {results['train_median_hfr_px']:.3f} px")
    for name in ("constant", "direct_hfr", "tenengrad", "norm_variance"):
        r = results[name]
        mae = r.get("hfr_mae_px")
        mae_s = f"MAE {_format_value(mae, '.3f')}px  " if mae is not None else " " * 12
        timing = f"  ({r['ms_per_crop']:.2f} ms/crop)" if "ms_per_crop" in r else ""
        print(f"  {name:<14} {mae_s}within-run rho {_format_value(r['within_run_rho'], '+.3f')}{timing}")


_HEX64_RE = re.compile(r"\b[0-9a-f]{64}\b")


def _sha256_path(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_digests(contract_path=None) -> dict[str, str]:
    """``sha256 -> description`` of every file a source contract pins or excludes.

    Reads the tracked contract and, when given, ``contract_path``. Every role
    except ``local_verified_run`` counts (public sources and fixtures), and the
    64-hex digests written in ``excluded_files`` notes (e.g. the test-era
    ``fqe_all_references.npz``) count too. An unreadable contract fails closed:
    without it a byte copy of a public file could not be recognised.
    """
    from data_pipeline import fqe_sources

    digests: dict[str, str] = {}
    for path in [None] + ([contract_path] if contract_path else []):
        try:
            contract = fqe_sources.load_contract(path)
        except (fqe_sources.FqeSourceError, OSError) as exc:
            raise SystemExit(
                f"refusing the legacy --dataset path: the source contract is unreadable ({exc}), so a copy of a "
                "pinned public file could not be recognised"
            ) from exc
        for source_id, entry in contract.sources.items():
            if entry.get("role") == "local_verified_run":
                continue
            for name, digest in (entry.get("files") or {}).items():
                digests.setdefault(str(digest), f"{source_id}/{name}")
            for name, note in (entry.get("excluded_files") or {}).items():
                for digest in _HEX64_RE.findall(str(note)):
                    digests.setdefault(digest, f"{source_id}/{name} (excluded file)")
    return digests


def _refuse_public_dataset(dataset_path: str, registry_path=None, *, contract_path=None) -> None:
    """Refuse a public file on the legacy ``--dataset`` path, before any array is read.

    Four independent screens (any one refuses): (1) the dataset or registry
    resolves under the public data root (catches symlinks); (2) the NPZ's
    ``run_key_semantics`` column, read lazily and alone, carries the public
    importers' archive-singleton marker (catches copies and re-saves that keep
    the column); (3) the NPZ's or registry's sha256 equals a file the source
    contract pins or excludes (catches byte copies and hard links anywhere,
    including ``fqe_all_references.npz``). The fourth screen is in
    :func:`_legacy_registry`: the legacy path never creates a split, so a
    copy without a registry cannot be scored on a fresh random split that
    mixes in sealed test groups.
    """
    from data_pipeline import fqe_sources
    from train.train_focus_quality import (
        ARCHIVE_SINGLETON_SEMANTICS,
        is_archive_singleton_dataset,
        read_run_key_semantics,
    )

    candidates = []
    try:
        candidates.append(fqe_sources.resolve_data_root(None))
    except fqe_sources.FqeSourceError:
        pass
    candidates.append(Path(fqe_sources.DEFAULT_DATA_ROOT).resolve())
    paths = [("--dataset", dataset_path)] + ([("split registry", registry_path)] if registry_path else [])
    for what, path in paths:
        target = Path(path).resolve()
        for root in candidates:
            if target == root or target.is_relative_to(root):
                raise SystemExit(
                    f"refusing {what} {path}: it lies under the public data root {root}. Public sources are "
                    "loaded only through data_pipeline.fqe_sources (use --source <id>)."
                )
    if not Path(dataset_path).is_file():
        raise SystemExit(f"--dataset {dataset_path} is not a file")
    semantics = read_run_key_semantics(dataset_path)
    if semantics is not None and is_archive_singleton_dataset({"run_key_semantics": semantics}):
        raise SystemExit(
            f"refusing --dataset {dataset_path}: it declares run_key_semantics={ARCHIVE_SINGLETON_SEMANTICS!r}, "
            "i.e. it is (a copy of) a public archive/simulation NPZ. Public sources are loaded only through "
            "data_pipeline.fqe_sources (use --source <id>), which never returns the sealed test partition."
        )
    digests = _public_digests(contract_path)
    for what, path in paths:
        if Path(path).is_file():
            match = digests.get(_sha256_path(path))
            if match is not None:
                raise SystemExit(
                    f"refusing {what} {path}: its sha256 equals the source contract's {match}. Public sources are "
                    "loaded only through data_pipeline.fqe_sources (use --source <id>)."
                )


def _legacy_registry(dataset_path: str, split_registry: str | None) -> Path:
    """The existing frozen split registry of a legacy dataset; never created here.

    ``split_registry.splits_for_dataset`` would write a fresh random split next
    to an NPZ that lacks one. An evaluation must score the split the model was
    trained against, so a missing registry is an error, not a prompt to make one.
    """
    path = Path(split_registry) if split_registry else Path(dataset_path).with_suffix(".splits.json")
    if not path.is_file():
        raise SystemExit(
            f"no split registry at {path}: the legacy --dataset path scores an existing frozen split and never "
            "creates one. Create it deliberately with data_pipeline.split_registry (the split the model trained "
            "on), or pass --split-registry."
        )
    return path


def main(dataset_path: str | None = None, synthetic: bool = False, split_registry: str | None = None, *,
         partition=_REQUIRED, sources=None, data_root=None, contract_path=None, out: str | None = None):
    if partition is _REQUIRED or partition is None:
        raise SystemExit("--partition is required (development: val); there is no default evaluation partition")
    outputs = {}
    if sources:
        from data_pipeline import fqe_sources

        contract = fqe_sources.load_contract(contract_path)
        for source_id in sources:
            arrays = fqe_sources.load_source(source_id, partitions=("train_fit", partition), contract=contract,
                                             data_root=data_root)
            outputs[source_id] = freeze_baselines(arrays, partition=partition)
            _print_results(source_id, outputs[source_id])
    elif synthetic or not dataset_path:
        from data_pipeline.synthetic_psf import make_synthetic_dataset

        data = make_synthetic_dataset(n_runs=48, seed=0)
        outputs["synthetic"] = freeze_baselines(data, partition=partition)
        _print_results("synthetic", outputs["synthetic"])
    else:
        candidate = split_registry or str(Path(dataset_path).with_suffix(".splits.json"))
        _refuse_public_dataset(dataset_path, candidate, contract_path=contract_path)
        registry = _legacy_registry(dataset_path, split_registry)
        from data_pipeline.split_registry import load_registered_splits

        with np.load(dataset_path, allow_pickle=False) as npz:
            data = {k: npz[k] for k in npz.files}
        group_field = "split_group" if "split_group" in data else "run_id"
        splits = load_registered_splits(str(registry), dataset_path, data[group_field])
        outputs[dataset_path] = freeze_baselines(data, splits=splits, partition=partition)
        _print_results(dataset_path, outputs[dataset_path])
    print(
        "\nPromotion gate: the FQE must beat `constant` on HFR MAE AND approach "
        "`tenengrad`/`direct_hfr` on within-run rho (where a genuine run makes it available) before it may be trusted."
    )
    if out:
        Path(out).write_text(json.dumps(outputs, indent=2, sort_keys=True, default=str))
    return outputs if len(outputs) != 1 else next(iter(outputs.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--split-registry")
    parser.add_argument("--source", action="append", help="fqe_sources source id (repeatable)")
    parser.add_argument("--data-root")
    parser.add_argument("--contract")
    parser.add_argument("--partition", required=True, choices=("val", "calibration", "test"),
                        help="evaluation partition (development: val). test applies to legacy local datasets "
                             "only and is refused for public sources")
    parser.add_argument("--out", help="optional JSON output path")
    args = parser.parse_args()
    main(args.dataset, args.synthetic, args.split_registry, partition=args.partition, sources=args.source,
         data_root=args.data_root, contract_path=args.contract, out=args.out)
