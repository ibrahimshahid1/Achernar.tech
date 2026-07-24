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
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_dataset(args) -> tuple[dict, str]:
    if args.synthetic or not args.dataset:
        from data_pipeline.synthetic_psf import make_synthetic_dataset

        return (
            make_synthetic_dataset(n_runs=args.synthetic_runs, seed=args.seed),
            f"synthetic(n_runs={args.synthetic_runs}, seed={args.seed})",
        )
    npz = np.load(args.dataset, allow_pickle=False)
    return {k: npz[k] for k in npz.files}, args.dataset


def main(args) -> dict:
    ml_dir = Path(__file__).resolve().parents[1]
    repo_root = ml_dir.parent
    models_root = Path(args.models_root) if args.models_root else repo_root / "models"
    out_dir = models_root / "focus_quality" / args.version
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Train -----------------------------------------------------------
    from train.train_focus_quality import train

    pt_path = str(out_dir / "focus_quality_fqe.pt")
    train_result = train(
        dict(
            synthetic=args.synthetic,
            dataset=args.dataset,
            synthetic_runs=args.synthetic_runs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_rank=args.lambda_rank,
            variance_warmup_epochs=args.variance_warmup_epochs,
            pretrained=args.pretrained,
            seed=args.seed,
            split_registry=args.split_registry,
            out=pt_path,
        )
    )
    test_metrics = train_result.get("test", {}) or {}

    # ---- 2. Calibrate (fit val, report test) --------------------------------
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

    # ---- 3. Frozen baselines (the gate the FQE must beat) -------------------
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
    baselines = freeze_baselines(dataset, split_seed=args.seed, splits=baseline_splits)

    # ---- 4. Export (FP32 parity-gated) ---------------------------------------
    from export.export_quality_onnx import export as export_onnx

    onnx_path = str(out_dir / "model.onnx")
    export_result = export_onnx(pt_path, onnx_path, crop_size=64, quantize=args.quantize)

    # ---- 5. Publish honest metadata ------------------------------------------
    from export.write_model_metadata import write_metadata

    mae = test_metrics.get("hfr_mae_px")
    rho = test_metrics.get("within_run_rank_rho")
    real_data = bool(args.dataset and not args.synthetic)
    tenengrad_rho = baselines["tenengrad"]["within_run_rho"]
    direct_rho = baselines["direct_hfr"]["within_run_rho"]
    finite_baseline_rhos = [
        value for value in (tenengrad_rho, direct_rho) if np.isfinite(value)
    ]
    rank_floor = max(finite_baseline_rhos) - 0.05 if finite_baseline_rhos else None
    calibrated_coverage = calib.get("coverage_95_test_after")
    frame_calibrated_coverage = calib.get("frame_coverage_95_test_after")
    gate = {
        "policy": (
            "SHADOW ONLY. Classical curve-of-growth HFR remains the control "
            "authority. Promote by hand only after beating the frozen "
            "baselines on real multi-rig data (spec Section 7.10)."
        ),
        "beats_constant_mae": bool(
            mae is not None and mae < baselines["constant"]["hfr_mae_px"]
        ),
        "real_multi_run_data": real_data,
        "meets_classical_rank_floor": bool(
            rank_floor is not None
            and rho is not None
            and np.isfinite(rho)
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
        "constant_baseline_mae_px": baselines["constant"]["hfr_mae_px"],
        "test_within_run_rho": rho,
        "tenengrad_within_run_rho": baselines["tenengrad"]["within_run_rho"],
    }
    gate["ready_for_promotion_review"] = bool(
        gate["real_multi_run_data"]
        and gate["beats_constant_mae"]
        and gate["meets_classical_rank_floor"]
        and gate["coverage_95_in_tolerance"]
        and gate["frame_coverage_95_in_tolerance"]
    )
    metadata = write_metadata(
        onnx_path,
        "focus_quality",
        args.version,
        "focus_quality",
        framework="onnx",
        input_format={"image": ["batch", 1, 64, 64]},
        output_format={"mu_log_hfr": ["batch"], "log_var": ["batch"]},
        training_info={
            "status": "trained",
            "promotion_status": "shadow",
            # Synthetic-only artifacts remain available by explicit path for
            # integration tests, but the daemon's semver resolver will not load
            # them automatically as live shadow evidence.
            "runtime_eligible": real_data,
            "promotion_gate": gate,
            "architecture": "MobileNetV3-Small (grayscale stem) + heteroscedastic head",
            "output_semantics": (
                "mu_log_hfr = E[ln HFR_px]; sigma_log_hfr = exp(0.5*log_var) * "
                "sigma_scale per star; robust pooled frame sigma is then multiplied "
                "by frame_sigma_scale (calibration.json)"
            ),
            "loss": f"heteroscedastic_nll + {args.lambda_rank} * ranknet_within_run",
            "optimizer": "AdamW, cosine annealing",
            "dataset": dataset_name,
            "seed": args.seed,
            "epochs": args.epochs,
            "best_val": train_result.get("best_val", {}),
            "test": test_metrics,
            "baselines": baselines,
            "calibration": {
                "sigma_scale": calib["sigma_scale"],
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
        },
    )

    # Bind calibration to the exact ONNX graph consumed at the edge. The Rust
    # runner refuses to call a sigma calibrated if this hash and metadata disagree.
    calib["model_sha256"] = metadata["sha256"]
    calib["model_filename"] = Path(onnx_path).name
    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=2))

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
    sigma_cal = float(np.exp(0.5 * lv) * calib["sigma_scale"])
    # Same fields/format as fqe_runner.rs test's PARITY_RUST line (the runner
    # applies sigma_scale from calibration.json, so calibrated sigma matches).
    print(
        f"PARITY_PY mu_log_hfr={mu:.6f} hfr_px={np.exp(mu):.6f} sigma_log_hfr={sigma_cal:.6f}"
    )

    verdict = "PASSES" if gate["beats_constant_mae"] else "DOES NOT PASS"
    print(
        f"\n=== FQE {args.version}: published to {out_dir} ===\n"
        f"    test MAE {mae if mae is not None else float('nan')} px vs constant "
        f"{baselines['constant']['hfr_mae_px']:.3f} px -> {verdict} the sanity floor\n"
        f"    rank rho {rho} vs tenengrad {baselines['tenengrad']['within_run_rho']:+.3f}\n"
        f"    coverage95 (test) {calib.get('coverage_95_test_before')} -> "
        f"{calib.get('coverage_95_test_after')} after sigma_scale={calib['sigma_scale']:.3f}\n"
        f"    frame coverage95 (test) {calib.get('frame_coverage_95_test_after')} "
        f"after frame_sigma_scale={calib.get('frame_sigma_scale')}\n"
        f"    promotion_status = SHADOW (daemon publishes fqe_* fields alongside "
        f"classical; classical stays authority)"
    )
    return {"out_dir": str(out_dir), "gate": gate, "calibration": calib}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="1.1.0")
    parser.add_argument("--models-root", default=None, help="default: <repo>/models")
    parser.add_argument("--dataset", help=".npz dataset; omit with --synthetic")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-runs", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-rank", type=float, default=0.2)
    parser.add_argument("--variance-warmup-epochs", type=int, default=None)
    parser.add_argument("--split-registry")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--quantize", action="store_true", help="also attempt INT8 (non-fatal)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
