"""
baselines — frozen classical baselines the FQE must beat (Section 7.7.x).

Per the review directive, freeze these BEFORE more CNN training so a learned
model has to earn its place:

  * constant       — predict the training-set median HFR for every crop. The
                     sanity floor; if the CNN cannot beat this on MAE it has
                     learned nothing.
  * direct_hfr     — the classical curve-of-growth HFR measured straight from the
                     crop (no learning), EXECUTED and TIMED per crop (not assumed
                     perfect: the audit rightly flagged the old hard-coded 0-MAE
                     entry). Caveat: it runs on the normalized [0,1] crops the
                     model sees; the p1/p99 clip slightly flattens bright cores,
                     so this is the deployable-baseline number, not the label
                     oracle. The "do we even need a CNN?" question is answered
                     by comparing the CNN against THIS row.
  * tenengrad      — squared-gradient sharpness energy (Pertuz 2013 gradient
                     family), ranked within run.
  * norm_variance  — normalized grey-level variance (Pertuz statistical family).

For focus, the decisive metric is WITHIN-RUN rank correlation (does the operator
order a sweep correctly?), evaluated on grouped splits. HFR MAE is reported where
a baseline predicts an HFR directly.

Run: python -m eval.baselines --dataset focus_quality_ds_v1.npz
     python -m eval.baselines --synthetic
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    per run rather than poisoning the whole run."""
    from scipy.stats import spearmanr

    rhos = []
    for run in np.unique(run_id):
        m = (run_id == run) & np.isfinite(score)
        if m.sum() >= 4 and np.ptp(hfr_px[m]) > 0:
            rho = spearmanr(sign * score[m], hfr_px[m]).statistic
            if np.isfinite(rho):
                rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")


def direct_hfr_measure(crop: np.ndarray) -> float:
    """The deployable classical baseline: local sky estimate + aperture-bounded
    curve-of-growth HFR on the crop, exactly what the daemon computes."""
    from labels.psf_measure import curve_of_growth_hfr, estimate_background

    sky, rms = estimate_background(np.asarray(crop, dtype=np.float64))
    hfr = curve_of_growth_hfr(np.asarray(crop, dtype=np.float64), sky, sky_rms=rms)
    return float(hfr) if hfr is not None and np.isfinite(hfr) else float("nan")


def freeze_baselines(dataset: dict, split_seed: int = 0, splits: dict | None = None) -> dict:
    from data_pipeline.split_registry import create_group_splits

    crops = dataset["crops"]
    hfr_px = dataset["hfr_px"]
    run_id = dataset["run_id"]

    splits = splits or create_group_splits(dataset.get("split_group", run_id), seed=split_seed)
    tr, te = splits["train"], splits["test"]
    train_median = float(np.median(hfr_px[tr]))

    ten = np.array([tenengrad(c) for c in crops[te]])
    nvar = np.array([normalized_variance(c) for c in crops[te]])

    # direct_hfr: actually run the classical measurement per crop, and time it,
    # so the row reports what deploying "no CNN, just measure" would cost/score.
    import time

    t0 = time.perf_counter()
    direct = np.array([direct_hfr_measure(c) for c in crops[te]])
    ms_per_crop = (time.perf_counter() - t0) * 1000.0 / max(len(direct), 1)
    finite = np.isfinite(direct)

    results = {
        "train_median_hfr_px": train_median,
        "constant": {
            "hfr_mae_px": float(np.mean(np.abs(hfr_px[te] - train_median))),
            "within_run_rho": 0.0,  # a constant cannot rank
        },
        "direct_hfr": {
            "hfr_mae_px": float(np.mean(np.abs(direct[finite] - hfr_px[te][finite])))
            if finite.any()
            else float("nan"),
            "within_run_rho": _within_run_rho(direct, hfr_px[te], run_id[te], sign=1.0),
            "measured_fraction": float(np.mean(finite)),
            "ms_per_crop": float(ms_per_crop),
            "note": "measured on normalized [0,1] crops (deployable baseline, not the label oracle)",
        },
        "tenengrad": {
            "within_run_rho": _within_run_rho(ten, hfr_px[te], run_id[te], sign=-1.0),
        },
        "norm_variance": {
            "within_run_rho": _within_run_rho(nvar, hfr_px[te], run_id[te], sign=-1.0),
        },
    }
    return results


def main(dataset_path: str | None, synthetic: bool, split_registry: str | None = None):
    if synthetic or not dataset_path:
        from data_pipeline.synthetic_psf import make_synthetic_dataset

        data = make_synthetic_dataset(n_runs=48, seed=0)
        splits = None
    else:
        from data_pipeline.split_registry import splits_for_dataset

        npz = np.load(dataset_path, allow_pickle=False)
        data = {k: npz[k] for k in npz.files}
        splits = splits_for_dataset(dataset_path, data, registry_path=split_registry)

    results = freeze_baselines(data, splits=splits)
    print("Frozen classical baselines (grouped test split):")
    print(f"  train-median HFR = {results['train_median_hfr_px']:.3f} px")
    for name in ("constant", "direct_hfr", "tenengrad", "norm_variance"):
        r = results[name]
        mae = r.get("hfr_mae_px")
        mae_s = f"MAE {mae:.3f}px  " if mae is not None else " " * 12
        timing = f"  ({r['ms_per_crop']:.2f} ms/crop)" if "ms_per_crop" in r else ""
        print(f"  {name:<14} {mae_s}within-run rho {r['within_run_rho']:+.3f}{timing}")
    print(
        "\nPromotion gate: the FQE must beat `constant` on HFR MAE AND approach "
        "`tenengrad`/`direct_hfr` on within-run rho before it may be trusted."
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--split-registry")
    args = parser.parse_args()
    main(args.dataset, args.synthetic, args.split_registry)
