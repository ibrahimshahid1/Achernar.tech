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
    model.onnx        # the graph (loaded by achernar-daemon/src/models/onnx_runner.rs)
    metadata.json     # schema below (read by achernar-daemon/src/models/model_registry.rs)
```

`metadata.json` schema (matches the Rust `ModelMetadata` struct):

```json
{
  "model_name": "focus_quality", "version": "1.0.0", "task": "focus_quality",
  "framework": "onnx",
  "input_format":  { "image": ["batch",1,64,64] },
  "output_format": { "score": ["batch",1] },
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

# 4. Train the real edge models, then export over the reference graphs.
python ml/train/train_focus_quality.py
```

Export a trained **edge** model into the layout from your training script:

```python
import torch
from ml.models.focus_quality_cnn import FocusQualityEstimator
from ml.export.onnx_export import export_to_edge

model = FocusQualityEstimator().eval()    # load trained weights first
export_to_edge(
    model,
    sample_inputs=(torch.randn(1, 1, 64, 64),),
    models_root="models", model_name="focus_quality", version="1.1.0",
    task="focus_quality",
    input_format={"image": ["batch", 1, 64, 64]},
    output_format={"score": ["batch", 1]},
    input_names=["image"], output_names=["score"],
)
```

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

## Daemon integration — the `/api/v1/ai/analyze` contract

The daemon loads edge models through `model_registry.rs` (scans
`models/<name>/<version>/`) and runs them via `onnx_runner.rs` (ort). The agreed
per-frame analysis endpoint the frontend/relay will call:

```
POST /api/v1/ai/analyze
  request : { "frame_id": "<id>" }                # or an inline frame reference
  response: {
    "frame_id": "<id>",
    "focus_score": 0.0-1.0,                        # focus_quality model
    "background_adu": <float>, "snr": <float>,     # background/SNR model
    "cloud_prob": 0.0-1.0,                          # cloud/obstruction model
    "trailing": { "elongation": <float>, "angle_deg": <float> },
    "embedding": [512 floats]                       # AstroM3Edge joint embedding (optional)
  }
```

The frontend's previous dead `daemonApi.analyze()` helper was removed; implement
this endpoint in the daemon (wire `vision/*_runner.rs` + `onnx_runner.rs`) and the
frontend can call it through both transports (direct in desktop, relayed in browser).
Until then, `build_reference_models.py` gives you valid graphs to develop against.
