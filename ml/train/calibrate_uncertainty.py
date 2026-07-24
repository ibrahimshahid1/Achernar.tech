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

Usage (standalone; rebuild_fqe.py drives this automatically):
    python -m train.calibrate_uncertainty --model focus_quality_fqe.pt \
        --synthetic --output calibration.json
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


def predict_arrays(model, crops: np.ndarray, device: str, batch_size: int = 256):
    """Run the FQE over crops, returning (mu, sigma) as numpy arrays.
    sigma = exp(0.5 * clamp(log_var)) — the RAW aleatoric sigma, pre-scaling."""
    import torch

    model.eval()
    mus, lvs = [], []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            xb = torch.as_tensor(crops[i : i + batch_size][:, None], device=device)
            mu, lv = model(xb)
            mus.append(mu.cpu().numpy())
            lvs.append(lv.cpu().numpy())
    mu = np.concatenate(mus).astype(np.float64)
    lv = np.clip(np.concatenate(lvs).astype(np.float64), -8.0, 8.0)
    return mu, np.exp(0.5 * lv)


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


def gaussian_nll(mu, sigma, y, scale: float = 1.0) -> float:
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


def calibrate(
    model_path: str,
    dataset_path: str | None,
    output_path: str,
    synthetic: bool,
    seed: int = 0,
    split_registry: str | None = None,
    synthetic_runs: int = 48,
) -> dict:
    import torch

    from data_pipeline.split_registry import create_group_splits, splits_for_dataset
    from models.focus_quality_cnn import FocusQualityEstimator

    if synthetic or not dataset_path:
        from data_pipeline.synthetic_psf import make_synthetic_dataset

        ds = make_synthetic_dataset(n_runs=synthetic_runs, seed=seed)
        dataset_name = f"synthetic(n_runs={synthetic_runs}, seed={seed})"
    else:
        npz = np.load(dataset_path, allow_pickle=False)
        ds = {k: npz[k] for k in npz.files}
        dataset_name = dataset_path

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
    va, te = splits["val"], splits["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FocusQualityEstimator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

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
    args = parser.parse_args()
    calibrate(
        args.model,
        args.dataset,
        args.output,
        args.synthetic,
        args.seed,
        args.split_registry,
        args.synthetic_runs,
    )
