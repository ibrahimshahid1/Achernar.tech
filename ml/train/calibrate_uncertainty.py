"""
calibrate_uncertainty — post-hoc quantile interval scaling for FQE uncertainty
(Section 7.7.3; the regression analogue of temperature scaling, Guo 2017).

The audit measured nominal-95% coverage of 0.54: the variance head is far too
confident. Quantile scaling fixes the *interval widths* without touching the mean:

    z_i = (y_i - mu_i) / sigma_i            (standardized residuals, val split)
    s   = quantile(|z|, 0.95) / 1.96        (one scalar)
    sigma_calibrated = s * sigma

By construction, mu +/- 1.96 * s * sigma then covers exactly 95% of the val
split; the honest number to report is coverage on the *test* split, which the
scale was never fitted on. The scalar ships in calibration.json next to the
ONNX model; the Rust runner (`fqe_runner.rs::load_sigma_scale`) multiplies it
into every predicted sigma and reports `is_calibrated() == true` only when the
field is present — closing the audit's "runner calls its output calibrated"
gap.

Development-only mode (public-data experiment): the scale is fitted only on a
reserved ``calibration`` partition and assessed on the disjoint ``val``
partition; the test partition is never predicted or reported. Where the source
contract has no calibration groups (HST: 6 train exposures) the record says
``status: unavailable`` instead of fitting on too few groups.

Usage (standalone; rebuild_fqe.py drives this automatically):
    python -m train.calibrate_uncertainty --model focus_quality_fqe.pt \
        --synthetic --output calibration.json
    python -m train.calibrate_uncertainty --model run/final/model_state.pt \
        --source-id stellar_sim_v1 --output calibration.json     # development-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


#: Partitions a scale may be fitted on. ``val`` is the legacy choice (it is also
#: the trainer's model-selection split); ``calibration`` is a reserved subset of
#: the original training groups.
FIT_PARTITIONS = ("val", "calibration")

#: Partitions development calibration never predicts, fits or reports.
SEALED_PARTITIONS = ("test", "test_sealed")


def _refuse_sealed(partition: str) -> None:
    if str(partition) in SEALED_PARTITIONS:
        raise ValueError(
            f"partition {partition!r} is sealed: development calibration never fits on or "
            "reports the test partition"
        )


def predict_mu_log_var(model, crops: np.ndarray, device: str, batch_size: int = 256):
    """Run the FQE over crops, returning the raw ``(mu, log_var)`` heads (float64)."""
    import torch

    model.eval()
    mus, lvs = [], []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            xb = torch.as_tensor(
                np.ascontiguousarray(crops[i : i + batch_size][:, None], dtype=np.float32),
                device=device,
            )
            mu, lv = model(xb)
            mus.append(mu.cpu().numpy())
            lvs.append(lv.cpu().numpy())
    if not mus:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.concatenate(mus).astype(np.float64), np.concatenate(lvs).astype(np.float64)


def predict_arrays(model, crops: np.ndarray, device: str, batch_size: int = 256):
    """Run the FQE over crops, returning (mu, sigma) as numpy arrays.
    sigma = exp(0.5 * clamp(log_var)) — the RAW aleatoric sigma, pre-scaling."""
    mu, lv = predict_mu_log_var(model, crops, device, batch_size)
    return mu, np.exp(0.5 * np.clip(lv, -8.0, 8.0))


def fit_sigma_scale(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, target: float = 0.95) -> float:
    """One scalar s such that mu +/- 1.96*s*sigma covers `target` of (mu, sigma, y)."""
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
    z = np.abs((np.asarray(y, dtype=np.float64) - np.asarray(mu, dtype=np.float64)) / sigma)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 1.0
    # Use the exact same normal quantile as `coverage`. A previously rounded
    # divisor put the reconstructed threshold one ulp below a tied residual and
    # could drop an entire tie group (71% observed coverage instead of 95%).
    scale = max(np.quantile(z, target) / _NORMAL_Z[0.95], 1e-6)
    return float(np.nextafter(scale, np.inf))


_NORMAL_Z = {
    0.50: 0.67448975,
    0.80: 1.28155157,
    0.90: 1.64485363,
    0.95: 1.95996398,
}


def coverage(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    scale: float = 1.0,
    level: float = 0.95,
) -> float:
    """Fraction of targets inside the central Gaussian interval at `level`."""
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
    z = _NORMAL_Z[level]
    inside = np.abs(np.asarray(y) - np.asarray(mu)) <= z * scale * sigma
    return float(np.mean(inside))


#: The NLL definition every record in this module (and fqe_dev_experiment)
#: uses: the 0.5*ln(2*pi) constant is omitted, as in models.fqe_losses.
NLL_CONVENTION = "mean(0.5*((y-mu)/sigma)^2 + ln sigma) in ln-HFR units; constant 0.5*ln(2*pi) omitted"


def gaussian_nll(mu, sigma, y, scale: float = 1.0) -> float:
    """Mean Gaussian NLL without the 0.5*ln(2*pi) constant (``NLL_CONVENTION``)."""
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64) * scale, 1e-9)
    residual = np.asarray(y, dtype=np.float64) - np.asarray(mu, dtype=np.float64)
    return float(np.mean(0.5 * (residual / sigma) ** 2 + np.log(sigma)))


def aggregate_frame_distributions(mu, sigma, y, frame_ids):
    """Pool per-star distributions and labels into one row per frame.

    This mirrors ``fqe_runner::aggregate_fqe``: median predicted log-HFR,
    between-star MAD, and mean aleatoric variance divided by star count. The
    returned sigma is deliberately *pre frame-scale* so a second held-out scale
    can calibrate the actual quantity published on the Shared Perception Bus.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    frame_ids = np.asarray(frame_ids)
    frame_mu, frame_sigma, frame_y = [], [], []
    for frame_id in np.unique(frame_ids):
        mask = frame_ids == frame_id
        if not mask.any():
            continue
        frame_med = float(np.median(mu[mask]))
        between = float(np.median(np.abs(mu[mask] - frame_med)) * 1.4826)
        aleatoric = float(np.mean(sigma[mask] ** 2) / int(mask.sum()))
        frame_mu.append(frame_med)
        frame_sigma.append(np.sqrt(between**2 + aleatoric))
        frame_y.append(float(np.median(y[mask])))
    return tuple(np.asarray(values, dtype=np.float64) for values in (frame_mu, frame_sigma, frame_y))


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibrate_from_arrays(
    mu_val,
    sigma_val,
    y_val,
    mu_test=None,
    sigma_test=None,
    y_test=None,
    dataset: str = "unknown",
    frame_id_val=None,
    frame_id_test=None,
) -> dict:
    """Fit on val, report honestly on test (never on the fitting split alone)."""
    scale = fit_sigma_scale(mu_val, sigma_val, y_val)
    levels = tuple(_NORMAL_Z)
    record = {
        "calibration_method": "quantile_interval_scaling",
        "sigma_scale": scale,
        "fit_on": "val",
        "n_val": int(len(y_val)),
        "coverage_val_before": {
            str(level): coverage(mu_val, sigma_val, y_val, level=level) for level in levels
        },
        "coverage_val_after": {
            str(level): coverage(mu_val, sigma_val, y_val, scale, level) for level in levels
        },
        "coverage_95_val_before": coverage(mu_val, sigma_val, y_val),
        "coverage_95_val_after": coverage(mu_val, sigma_val, y_val, scale),
        "nll_val_before": gaussian_nll(mu_val, sigma_val, y_val),
        "nll_val_after": gaussian_nll(mu_val, sigma_val, y_val, scale),
        "dataset": dataset,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Per-star sigma = sigma_scale * exp(0.5*log_var). After robust star "
            "pooling, frame sigma is multiplied by frame_sigma_scale. Both scales "
            "are fit only on validation groups; test coverage is the honest "
            "generalization number. Consumed by fqe_runner.rs."
        ),
    }
    if mu_test is not None and y_test is not None and len(y_test):
        record.update(
            evaluated_on="test",
            n_test=int(len(y_test)),
            coverage_95_test_before=coverage(mu_test, sigma_test, y_test),
            coverage_95_test_after=coverage(mu_test, sigma_test, y_test, scale),
            coverage_test_before={
                str(level): coverage(mu_test, sigma_test, y_test, level=level)
                for level in levels
            },
            coverage_test_after={
                str(level): coverage(mu_test, sigma_test, y_test, scale, level)
                for level in levels
            },
            nll_test_before=gaussian_nll(mu_test, sigma_test, y_test),
            nll_test_after=gaussian_nll(mu_test, sigma_test, y_test, scale),
        )
    if frame_id_val is not None:
        frame_mu_val, frame_sigma_val, frame_y_val = aggregate_frame_distributions(
            mu_val, np.asarray(sigma_val) * scale, y_val, frame_id_val
        )
        frame_scale = fit_sigma_scale(frame_mu_val, frame_sigma_val, frame_y_val)
        record.update(
            frame_sigma_scale=frame_scale,
            n_val_frames=int(frame_y_val.size),
            frame_coverage_val_before={
                str(level): coverage(
                    frame_mu_val, frame_sigma_val, frame_y_val, level=level
                )
                for level in levels
            },
            frame_coverage_val_after={
                str(level): coverage(
                    frame_mu_val, frame_sigma_val, frame_y_val, frame_scale, level
                )
                for level in levels
            },
            frame_coverage_95_val_after=coverage(
                frame_mu_val, frame_sigma_val, frame_y_val, frame_scale
            ),
        )
        if frame_id_test is not None and mu_test is not None and len(y_test):
            frame_mu_test, frame_sigma_test, frame_y_test = aggregate_frame_distributions(
                mu_test, np.asarray(sigma_test) * scale, y_test, frame_id_test
            )
            record.update(
                n_test_frames=int(frame_y_test.size),
                frame_coverage_test_before={
                    str(level): coverage(
                        frame_mu_test, frame_sigma_test, frame_y_test, level=level
                    )
                    for level in levels
                },
                frame_coverage_test_after={
                    str(level): coverage(
                        frame_mu_test,
                        frame_sigma_test,
                        frame_y_test,
                        frame_scale,
                        level,
                    )
                    for level in levels
                },
                frame_coverage_95_test_after=coverage(
                    frame_mu_test, frame_sigma_test, frame_y_test, frame_scale
                ),
                frame_nll_test_after=gaussian_nll(
                    frame_mu_test, frame_sigma_test, frame_y_test, frame_scale
                ),
            )
    return record


def calibration_unavailable_record(
    reason: str,
    *,
    fit_partition: str = "calibration",
    dataset: str = "unknown",
    **extra,
) -> dict:
    """An explicit "no calibration" result. ``sigma_scale`` is null so nothing
    downstream (``fqe_runner.rs`` included) can call the sigma calibrated."""
    record = {
        "status": "unavailable",
        "reason": str(reason),
        "calibration_method": "quantile_interval_scaling",
        "sigma_scale": None,
        "frame_sigma_scale": None,
        "fit_on": fit_partition,
        "development_only": True,
        "test_reported": False,
        "dataset": dataset,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }
    record.update(extra)
    return record


def _interval_block(mu, sigma, y, scale: float) -> dict:
    levels = tuple(_NORMAL_Z)
    return {
        "n": int(len(y)),
        "coverage_before": {str(level): coverage(mu, sigma, y, level=level) for level in levels},
        "coverage_after": {str(level): coverage(mu, sigma, y, scale, level) for level in levels},
        "coverage_95_before": coverage(mu, sigma, y),
        "coverage_95_after": coverage(mu, sigma, y, scale),
        "coverage_68_before": _coverage_68(mu, sigma, y, 1.0),
        "coverage_68_after": _coverage_68(mu, sigma, y, scale),
        "nll_before": gaussian_nll(mu, sigma, y),
        "nll_after": gaussian_nll(mu, sigma, y, scale),
    }


def _coverage_68(mu, sigma, y, scale: float) -> float:
    """Central ~68.27% interval (|z| <= 1) coverage; reported beside the
    tabulated levels because the experiment contract names 68/95%."""
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
    inside = np.abs(np.asarray(y) - np.asarray(mu)) <= 1.0 * scale * sigma
    return float(np.mean(inside)) if inside.size else float("nan")


def calibrate_development_arrays(
    mu_fit,
    sigma_fit,
    y_fit,
    *,
    fit_partition: str = "calibration",
    evaluations: dict | None = None,
    frame_id_fit=None,
    dataset: str = "unknown",
    min_fit_rows: int = 1,
) -> dict:
    """Development-only calibration: fit the quantile scale on ``fit_partition``
    and assess it on each named disjoint partition in ``evaluations``
    (``{name: {"mu", "sigma", "y", "frame_id"?}}``). A sealed partition name is
    refused; no test field is ever written."""
    _refuse_sealed(fit_partition)
    if fit_partition not in FIT_PARTITIONS:
        raise ValueError(f"fit_partition must be one of {FIT_PARTITIONS}, got {fit_partition!r}")
    evaluations = dict(evaluations or {})
    for name in evaluations:
        _refuse_sealed(name)
        if name == fit_partition:
            raise ValueError(f"evaluation partition {name!r} is the fitting partition")

    mu_fit = np.asarray(mu_fit, dtype=np.float64)
    sigma_fit = np.asarray(sigma_fit, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    finite = np.isfinite(mu_fit) & np.isfinite(sigma_fit) & np.isfinite(y_fit)
    if int(finite.sum()) < max(1, int(min_fit_rows)):
        return calibration_unavailable_record(
            f"{int(finite.sum())} finite rows in {fit_partition!r} (< {max(1, int(min_fit_rows))})",
            fit_partition=fit_partition,
            dataset=dataset,
            n_fit=int(y_fit.size),
        )
    scale = fit_sigma_scale(mu_fit[finite], sigma_fit[finite], y_fit[finite])
    fit_block = _interval_block(mu_fit, sigma_fit, y_fit, scale)
    record = {
        "status": "ok",
        "calibration_method": "quantile_interval_scaling",
        "sigma_scale": scale,
        "fit_on": fit_partition,
        "n_fit": int(y_fit.size),
        "n_fit_finite": int(finite.sum()),
        "fit": fit_block,
        "nll_convention": NLL_CONVENTION,
        "development_only": True,
        "test_reported": False,
        "dataset": dataset,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            f"Per-star sigma = sigma_scale * exp(0.5*clamp(log_var)); sigma_scale fitted only on "
            f"the {fit_partition!r} partition and assessed on the disjoint partitions listed in "
            "evaluated_on. The test partition is sealed and never predicted here."
        ),
    }
    frame_scale = None
    if frame_id_fit is not None:
        frame_ids = np.asarray(frame_id_fit)
        frame_mu, frame_sigma, frame_y = aggregate_frame_distributions(
            mu_fit[finite], sigma_fit[finite] * scale, y_fit[finite], frame_ids[finite]
        )
        frame_scale = fit_sigma_scale(frame_mu, frame_sigma, frame_y)
        record.update(
            frame_sigma_scale=frame_scale,
            n_fit_frames=int(frame_y.size),
            frame_fit=_interval_block(frame_mu, frame_sigma, frame_y, frame_scale),
        )
    evaluated = {}
    for name, values in evaluations.items():
        mu = np.asarray(values["mu"], dtype=np.float64)
        sigma = np.asarray(values["sigma"], dtype=np.float64)
        y = np.asarray(values["y"], dtype=np.float64)
        block = _interval_block(mu, sigma, y, scale)
        if frame_scale is not None and values.get("frame_id") is not None and len(y):
            frame_mu, frame_sigma, frame_y = aggregate_frame_distributions(
                mu, sigma * scale, y, values["frame_id"]
            )
            block["frames"] = _interval_block(frame_mu, frame_sigma, frame_y, frame_scale)
        evaluated[name] = block
    record["evaluated_on"] = list(evaluated)
    record["evaluation"] = evaluated
    return record


def calibrate_loaded_sources(
    model,
    calibration_sources: dict,
    evaluation_sources: dict,
    device: str,
    *,
    unavailable: dict | None = None,
) -> dict:
    """Per-source development calibration from already-loaded source arrays.

    ``calibration_sources[sid]`` holds that source's calibration rows and
    ``evaluation_sources[sid][partition]`` its disjoint evaluation rows (objects
    with ``crops``, ``log_target``, ``frame_uid``). ``unavailable[sid]`` is a
    reason string for sources whose contract has no calibration partition.
    """
    per_source = {}
    for source_id, reason in (unavailable or {}).items():
        per_source[source_id] = calibration_unavailable_record(reason, dataset=source_id)
    for source_id, cal in calibration_sources.items():
        if len(cal.log_target) == 0:
            per_source[source_id] = calibration_unavailable_record(
                "the calibration partition is empty", dataset=source_id
            )
            continue
        mu_c, sig_c = predict_arrays(model, np.asarray(cal.crops, dtype=np.float32), device)
        evaluations = {}
        for partition, rows in (evaluation_sources.get(source_id) or {}).items():
            _refuse_sealed(partition)
            mu_e, sig_e = predict_arrays(model, np.asarray(rows.crops, dtype=np.float32), device)
            evaluations[partition] = {
                "mu": mu_e,
                "sigma": sig_e,
                "y": np.asarray(rows.log_target, dtype=np.float64),
                "frame_id": np.asarray(rows.frame_uid),
            }
        per_source[source_id] = calibrate_development_arrays(
            mu_c,
            sig_c,
            np.asarray(cal.log_target, dtype=np.float64),
            fit_partition="calibration",
            evaluations=evaluations,
            frame_id_fit=np.asarray(cal.frame_uid),
            dataset=source_id,
        )
    return per_source


def _subset(source, mask):
    """A shallow per-partition view of a source-arrays object."""
    from types import SimpleNamespace

    mask = np.asarray(mask, dtype=bool)
    return SimpleNamespace(
        crops=np.asarray(source.crops)[mask],
        log_target=np.asarray(source.log_target)[mask],
        target_hfr_px=np.asarray(source.target_hfr_px)[mask],
        frame_uid=np.asarray(source.frame_uid)[mask],
        group_uid=np.asarray(source.group_uid)[mask],
        sample_uid=np.asarray(source.sample_uid)[mask],
    )


def calibrate_sources(
    model_path: str,
    source_ids,
    output_path: str | None = None,
    *,
    contract=None,
    data_root: str | None = None,
    device: str | None = None,
    evaluate_partitions=("val",),
) -> dict:
    """Development-only calibration over contract sources (``fqe_sources``):
    fit on each source's ``calibration`` partition, assess on ``val``. A source
    whose contract has no calibration groups is recorded ``unavailable``."""
    import torch

    from data_pipeline import fqe_sources
    from models.focus_quality_cnn import FocusQualityEstimator
    from train.train_focus_quality import resolve_device

    for name in evaluate_partitions:
        _refuse_sealed(name)
    device = resolve_device(device)
    model = FocusQualityEstimator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    calibration_sources, evaluation_sources, unavailable, roles, identity = {}, {}, {}, {}, {}
    for source_id in source_ids:
        partitions = ("calibration",) + tuple(evaluate_partitions)
        try:
            source = fqe_sources.load_source(
                source_id, partitions=partitions, contract=contract, data_root=data_root
            )
        except fqe_sources.CalibrationUnavailable as err:
            unavailable[source_id] = str(err) or "calibration unavailable for this source"
            continue
        roles[source_id] = str(source.role)
        identity[source_id] = source.identity
        declared = (source.identity or {}).get("calibration") or {}
        if declared.get("available") is False:
            unavailable[source_id] = str(declared.get("reason") or "calibration unavailable for this source")
            continue
        partition = np.asarray(source.partition).astype(str)
        calibration_sources[source_id] = _subset(source, partition == "calibration")
        evaluation_sources[source_id] = {
            name: _subset(source, partition == name) for name in evaluate_partitions
        }
    per_source = calibrate_loaded_sources(
        model, calibration_sources, evaluation_sources, device, unavailable=unavailable
    )
    record = {
        "schema": "fqe-dev-calibration-v1",
        "development_only": True,
        "test_reported": False,
        "runtime_eligible": False,
        "data_roles": roles,
        "source_identity": identity,
        "device": device,
        "model_state_sha256": _sha256(model_path),
        "sources": per_source,
    }
    if output_path:
        Path(output_path).write_text(json.dumps(record, indent=2, default=str))
        print(f"wrote {output_path}")
    return record


def calibrate(
    model_path: str,
    dataset_path: str | None,
    output_path: str,
    synthetic: bool,
    seed: int = 0,
    split_registry: str | None = None,
    synthetic_runs: int = 48,
    *,
    fit_partition: str = "val",
    development_only: bool = False,
    evaluate_partitions=None,
    device: str | None = None,
) -> dict:
    """Fit the interval scale on ``fit_partition``.

    Legacy mode (``fit_partition="val"``, ``development_only=False``) is the
    unchanged genuine-run behaviour: fit on val, report on test. Any other
    combination is development-only: the test partition is never predicted,
    the scale is assessed only on ``evaluate_partitions`` (default ``("val",)``
    when fitting elsewhere), and a missing fitting partition writes an explicit
    ``status: unavailable`` record.
    """
    import torch

    from data_pipeline.split_registry import create_group_splits, splits_for_dataset
    from models.focus_quality_cnn import FocusQualityEstimator
    from train.train_focus_quality import refuse_archive_npz, resolve_device, screen_dataset_path

    _refuse_sealed(fit_partition)
    if fit_partition not in FIT_PARTITIONS:
        raise ValueError(f"fit_partition must be one of {FIT_PARTITIONS}, got {fit_partition!r}")
    development_only = bool(development_only or fit_partition != "val")
    if evaluate_partitions is None:
        evaluate_partitions = () if fit_partition == "val" else ("val",)
    evaluate_partitions = tuple(evaluate_partitions)
    for name in evaluate_partitions:
        _refuse_sealed(name)

    if synthetic or not dataset_path:
        from data_pipeline.synthetic_psf import make_synthetic_dataset

        ds = make_synthetic_dataset(n_runs=synthetic_runs, seed=seed)
        dataset_name = f"synthetic(n_runs={synthetic_runs}, seed={seed})"
    else:
        screen_dataset_path(dataset_path)  # before any array is materialised
        npz = np.load(dataset_path, allow_pickle=False)
        ds = {k: npz[k] for k in npz.files}
        dataset_name = dataset_path
        refuse_archive_npz(ds, dataset_path)

    crops = ds["crops"].astype(np.float32)
    log_hfr = ds["log_hfr"].astype(np.float64)
    run_id = ds["run_id"].astype(np.int64)
    frame_id = ds.get("frame_id")
    if frame_id is None:
        # Older synthetic fixtures had no explicit frame key. A focus position is
        # one frame in that generator; real datasets always carry persisted UUIDs.
        frame_id = np.asarray(
            [f"run-{run}-position-{position}" for run, position in zip(run_id, ds["position"])]
        )
    if synthetic or not dataset_path:
        splits = create_group_splits(ds.get("split_group", run_id), seed=seed)
    else:
        splits = splits_for_dataset(
            dataset_path, ds, seed=seed, registry_path=split_registry
        )

    device = resolve_device(device)
    model = FocusQualityEstimator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    if development_only:
        missing = [name for name in (fit_partition, *evaluate_partitions) if name not in splits]
        if fit_partition in missing:
            record = calibration_unavailable_record(
                f"dataset has no {fit_partition!r} partition (available: {sorted(splits)})",
                fit_partition=fit_partition,
                dataset=dataset_name,
            )
        else:
            if missing:
                raise ValueError(f"evaluation partitions {missing} are not in the split registry")
            fit_idx = splits[fit_partition]
            mu_fit, sig_fit = predict_arrays(model, crops[fit_idx], device)
            evaluations = {}
            for name in evaluate_partitions:
                idx = splits[name]
                mu_e, sig_e = predict_arrays(model, crops[idx], device)
                evaluations[name] = {
                    "mu": mu_e,
                    "sigma": sig_e,
                    "y": log_hfr[idx],
                    "frame_id": frame_id[idx],
                }
            record = calibrate_development_arrays(
                mu_fit,
                sig_fit,
                log_hfr[fit_idx],
                fit_partition=fit_partition,
                evaluations=evaluations,
                frame_id_fit=frame_id[fit_idx],
                dataset=dataset_name,
            )
        record["device"] = device
        record["model_state_sha256"] = _sha256(model_path)
        record["dataset_sha256"] = _sha256(dataset_path) if dataset_path else None
        with open(output_path, "w") as fh:
            json.dump(record, fh, indent=2)
        print(
            f"development calibration ({record['status']}): fit_on={fit_partition} "
            f"sigma_scale={record['sigma_scale']} evaluated_on={record.get('evaluated_on', [])}"
        )
        print(f"wrote {output_path}")
        return record

    va, te = splits["val"], splits["test"]
    mu_va, sig_va = predict_arrays(model, crops[va], device)
    if te.size:
        mu_te, sig_te = predict_arrays(model, crops[te], device)
        y_te = log_hfr[te]
    else:
        mu_te, sig_te, y_te = None, None, None

    record = calibrate_from_arrays(
        mu_va,
        sig_va,
        log_hfr[va],
        mu_te,
        sig_te,
        y_te,
        dataset=dataset_name,
        frame_id_val=frame_id[va],
        frame_id_test=frame_id[te] if te.size else None,
    )
    record["model_state_sha256"] = _sha256(model_path)
    record["dataset_sha256"] = _sha256(dataset_path) if dataset_path else None
    with open(output_path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(
        f"sigma_scale={record['sigma_scale']:.4f}  "
        f"val cov95 {record['coverage_95_val_before']:.3f} -> {record['coverage_95_val_after']:.3f}"
        + (
            f"  test cov95 {record['coverage_95_test_before']:.3f} -> {record['coverage_95_test_after']:.3f}"
            if "coverage_95_test_after" in record
            else ""
        )
        + (
            f"  frame test cov95 {record['frame_coverage_95_test_after']:.3f}"
            if "frame_coverage_95_test_after" in record
            else ""
        )
    )
    print(f"wrote {output_path}")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="trained FQE .pt state dict")
    parser.add_argument("--dataset", help=".npz dataset; omit with --synthetic")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic-runs", type=int, default=48)
    parser.add_argument("--split-registry")
    parser.add_argument("--output", required=True, help="calibration.json path")
    parser.add_argument("--fit-partition", choices=FIT_PARTITIONS, default="val")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="never predict or report the test partition",
    )
    parser.add_argument(
        "--evaluate-partition",
        action="append",
        default=None,
        help="development partition to assess the fitted scale on (repeatable; never test)",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        default=None,
        help="contract source (data_pipeline.fqe_sources); implies development-only "
        "calibration-partition fitting",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.source_id:
        calibrate_sources(
            args.model,
            args.source_id,
            args.output,
            data_root=args.data_root,
            device=args.device,
            evaluate_partitions=tuple(args.evaluate_partition or ("val",)),
        )
    else:
        calibrate(
            args.model,
            args.dataset,
            args.output,
            args.synthetic,
            args.seed,
            args.split_registry,
            args.synthetic_runs,
            fit_partition=args.fit_partition,
            development_only=args.development_only,
            evaluate_partitions=args.evaluate_partition,
            device=args.device,
        )
