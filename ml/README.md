# Achernar ML

Model development for Achernar's intelligence layer:

- **AstroM3** — the tri-modal embedding (spec §7.6.9). This *is* AstroM3
  (Rizhko & Bloom 2024); see [`ml/astrom3/`](astrom3/README.md). Two deployments:
  cloud/planning (light curve + spectrum + metadata) and an edge/daemon
  **retrofit** (image crop + capture metadata + on-mount telemetry).
- **Per-frame edge models** — focus quality, cloud/obstruction, trailing,
  background/SNR, focus offset/drift, exposure bandit, filter recommender.

## Layout

```
ml/
  astrom3/       AstroM3 integration: download.py, loader.py (cloud), edge.py (daemon retrofit)
  models/        per-frame model architectures (PyTorch); tri_modal_embedding.py = AstroM3 entrypoint
  datasets/      dataset builders / splits
  preprocess/    crops, normalization, feature builders
  train/         training entry points
  eval/          evaluation + policy replay
  export/        ONNX export + metadata writer
  pipelines/     runnable end-to-end pipelines
  checkpoints/   downloaded weights (git-ignored)
```

Trained **edge** models are published to the layout the daemon consumes
(spec §7.6.10), at the repo-root `models/` directory:

```
models/<model_name>/<version>/
    model.onnx         # two-head graph loaded by vision/fqe_runner.rs
    metadata.json      # graph contract, provenance, hash, and promotion gates
    calibration.json   # hash-bound per-star + post-pooling uncertainty scales
    norm_params.json   # exact per-crop preprocessing contract
```

`metadata.json` schema (matches the Rust `ModelMetadata` struct):

```json
{
  "model_name": "focus_quality", "version": "1.0.0", "task": "focus_quality",
  "framework": "onnx",
  "input_format":  { "image": ["batch",1,64,64] },
  "output_format": { "mu_log_hfr": ["batch"], "log_var": ["batch"] },
  "sha256": "…", "created_at": "…Z", "training_info": { … }
}
```

## Quickstart

```bash
# 1. Install ML deps (PyTorch etc.)
pip install -r ml/requirements.txt

# 2. AstroM3 (the tri-modal embedding) — download the pretrained weights.
./scripts/download_astrom3.sh            # -> ml/checkpoints/astrom3/ (git-ignored)

# 3. Per-frame edge models: generate shape-correct REFERENCE ONNX (no training)
#    to wire daemon integration immediately.
python ml/pipelines/build_reference_models.py
#    -> models/focus_quality/1.0.0/model.onnx   (verified to load in onnxruntime)

# 4. Build a real labeled dataset from daemon autofocus provenance.
python -m ml.datasets.build_focus_quality_dataset \
  --db /path/to/achernar.db --output focus_quality_ds_v1.npz

# 5. Train, calibrate, baseline, export, hash-bind, and publish one shadow model.
cd ml
../venv/bin/python -m pipelines.rebuild_fqe \
  --dataset ../focus_quality_ds_v1.npz --epochs 50 --pretrained --version 1.1.0
```

The checked-in `1.0.0` graph has random reference weights and
`runtime_eligible=false`; it is useful for interface tests but the daemon will
never auto-select it. `rebuild_fqe` also leaves synthetic-only artifacts
explicit-path-only and stamps every trained artifact `shadow`. Promotion remains
a separate decision after grouped real-data gates pass.

## AstroM3 — the tri-modal embedding (§7.6.9)

AstroM3 is the CLIP-style joint embedding (light curve + spectrum + metadata).
Full details, download, and the daemon **retrofit** (`AstroM3Edge`: image +
metadata + telemetry, reusing AstroM3's encoders) are documented in
[`ml/astrom3/README.md`](astrom3/README.md).

```python
from ml.models.tri_modal_embedding import load_cloud, build_edge
cloud = load_cloud("base")   # AstroM3 (planning / apps.ai_engine)
edge  = build_edge()         # AstroM3Edge (daemon)
```

## Daemon integration

The vision loop batches centered crops from classically gated stars through
`vision/fqe_runner.rs`, robustly pools them, and publishes shadow-only fields on
`SharedObservation`: `fqe_hfr_px`, `fqe_sigma_log_hfr`, `fqe_star_count`,
`fqe_model_version`, and `fqe_calibrated`. Classical HFR remains the autofocus
authority. Missing measurements stay `null`; there is deliberately no scalar
`focus_score`, `sharpness_score`, or fabricated seeing value.
