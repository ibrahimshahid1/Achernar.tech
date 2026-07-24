"""
eval_quality — FQE evaluation (Section 7.7.x).

Reports the metrics that actually matter for autofocus, on a GROUPED held-out
split (never per-sample random): per-star HFR error, within-run rank correlation
(does the model order a sweep correctly?), best-focus-position error (does the
aggregated curve minimum land at the right place?), and interval coverage (is the
predicted uncertainty calibrated?). Also compares against a classical
gradient-energy baseline so a learned model has to *earn* its place.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def gradient_energy_baseline(crop: np.ndarray) -> float:
    """Classical Tenengrad-ish sharpness (higher = sharper). A learned FQE must
    beat this on within-run ranking to justify itself (Pertuz 2013)."""
    c = crop.astype(np.float64)
    gx = np.diff(c, axis=1)
    gy = np.diff(c, axis=0)
    return float((gx**2).mean() + (gy**2).mean())


def best_position_error(mu, hfr_px, run_id, position):
    """Mean |argmin position from aggregated predictions − argmin from labels|,
    over runs (in focuser steps)."""
    from models.fqe_losses import aggregate_stars

    errs = []
    for run in np.unique(run_id):
        m = run_id == run
        pos = position[m]
        upos = np.unique(pos)
        if upos.size < 3:
            continue
        pred_hfr, true_hfr = [], []
        for p in upos:
            pm = m & (position == p)
            agg, _ = aggregate_stars(mu[pm], np.zeros(pm.sum()))
            pred_hfr.append(agg)
            true_hfr.append(np.median(hfr_px[pm]))
        errs.append(abs(upos[int(np.argmin(pred_hfr))] - upos[int(np.argmin(true_hfr))]))
    return float(np.mean(errs)) if errs else float("nan")


def evaluate(
    model_path: str,
    test_data: str,
    split: str = "test",
    split_registry: str | None = None,
    calibration_path: str | None = None,
) -> dict:
    import torch
    from scipy.stats import spearmanr

    from data_pipeline.split_registry import splits_for_dataset
    from models.focus_quality_cnn import FocusQualityEstimator
    from train.calibrate_uncertainty import aggregate_frame_distributions, coverage

    loaded = np.load(test_data, allow_pickle=False)
    data = {key: loaded[key] for key in loaded.files}
    crops, log_hfr, hfr_px = data["crops"], data["log_hfr"], data["hfr_px"]
    run_id, position = data["run_id"], data["position"]
    frame_id = data.get("frame_id")
    if frame_id is None:
        frame_id = np.asarray(
            [f"run-{run}-position-{pos}" for run, pos in zip(run_id, position)]
        )
    idx = splits_for_dataset(
        test_data, data, registry_path=split_registry
    )[split]
    crops, log_hfr, hfr_px, run_id, position, frame_id = (
        crops[idx],
        log_hfr[idx],
        hfr_px[idx],
        run_id[idx],
        position[idx],
        frame_id[idx],
    )

    model = FocusQualityEstimator()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    mus, lvs = [], []
    with torch.no_grad():
        for i in range(0, len(crops), 256):
            mu, lv = model(torch.as_tensor(crops[i : i + 256][:, None]))
            mus.append(mu.numpy())
            lvs.append(lv.numpy())
    mu = np.concatenate(mus)
    lv = np.clip(np.concatenate(lvs), -8.0, 8.0)

    hfr_pred = np.exp(mu)
    sigma_scale = 1.0
    frame_sigma_scale = 1.0
    per_star_calibrated = False
    frame_calibrated = False
    if calibration_path:
        calibration = json.loads(Path(calibration_path).read_text())
        expected_state_hash = calibration.get("model_state_sha256")
        actual_state_hash = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        if expected_state_hash != actual_state_hash:
            raise ValueError("calibration file does not belong to this model state")
        sigma_scale = float(calibration["sigma_scale"])
        per_star_calibrated = True
        if calibration.get("frame_sigma_scale") is not None:
            frame_sigma_scale = float(calibration["frame_sigma_scale"])
            frame_calibrated = True
    sigma = np.sqrt(np.exp(lv)) * sigma_scale
    frame_mu, frame_sigma, frame_y = aggregate_frame_distributions(
        mu, sigma, log_hfr, frame_id
    )
    rhos_model, rhos_base = [], []
    for run in np.unique(run_id):
        m = run_id == run
        if m.sum() >= 4 and np.ptp(hfr_px[m]) > 0:
            rhos_model.append(spearmanr(mu[m], hfr_px[m]).statistic)
            base = np.array([gradient_energy_baseline(c) for c in crops[m]])
            # higher gradient energy = sharper = lower HFR, so negate for rank sign
            rhos_base.append(spearmanr(-base, hfr_px[m]).statistic)

    metrics = {
        "n": int(len(crops)),
        "hfr_mae_px": float(np.mean(np.abs(hfr_pred - hfr_px))),
        "hfr_rmse_px": float(np.sqrt(np.mean((hfr_pred - hfr_px) ** 2))),
        "coverage_50": coverage(mu, sigma, log_hfr, level=0.50),
        "coverage_80": coverage(mu, sigma, log_hfr, level=0.80),
        "coverage_90": coverage(mu, sigma, log_hfr, level=0.90),
        "coverage_95": coverage(mu, sigma, log_hfr, level=0.95),
        "uncertainty_calibrated": per_star_calibrated and frame_calibrated,
        "per_star_uncertainty_calibrated": per_star_calibrated,
        "frame_uncertainty_calibrated": frame_calibrated,
        "n_frames": int(frame_y.size),
        "frame_coverage_50": coverage(
            frame_mu, frame_sigma, frame_y, frame_sigma_scale, level=0.50
        ),
        "frame_coverage_80": coverage(
            frame_mu, frame_sigma, frame_y, frame_sigma_scale, level=0.80
        ),
        "frame_coverage_90": coverage(
            frame_mu, frame_sigma, frame_y, frame_sigma_scale, level=0.90
        ),
        "frame_coverage_95": coverage(
            frame_mu, frame_sigma, frame_y, frame_sigma_scale, level=0.95
        ),
        "within_run_rank_rho_model": float(np.nanmean(rhos_model)) if rhos_model else float("nan"),
        "within_run_rank_rho_baseline": float(np.nanmean(rhos_base)) if rhos_base else float("nan"),
        "best_position_err_steps": best_position_error(mu, hfr_px, run_id, position),
    }
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--split-registry")
    parser.add_argument("--calibration")
    args = parser.parse_args()
    evaluate(
        args.model,
        args.test_data,
        args.split,
        args.split_registry,
        args.calibration,
    )
