# Achernar ML

Training dataset locations and external-drive requirements are documented in
[`docs/training-data-storage.md`](../docs/training-data-storage.md).

Model development for Achernar's intelligence layer:

- **AstroM3** — the tri-modal embedding (spec §7.6.9). This *is* AstroM3
  (Rizhko & Bloom 2024); see [`ml/astrom3/`](astrom3/README.md). The native
  integration uses light curves, spectra and metadata. The proposed edge/daemon
  **retrofit** (image crop + capture metadata + on-mount telemetry) is a research
  prototype requiring alignment training and validation before deployment.
- **Per-frame edge models** — focus quality, cloud/obstruction, trailing,
  background/SNR, focus offset/drift, exposure bandit, filter recommender.

The [v6 architecture source](../achernar_v6.tex) and
[compute-aware runtime ADR](../docs/adr/compute-aware-model-runtime.md) describe
the planned deployment contract. The implemented daemon neural path is the
CPU-default FQE runner; COC/TED daemon neural deployments remain planned.
COC now has an [offline research segmentation pipeline](coc/README.md): tiny
and small PyTorch models, human partial-mask training, optional separate teacher
distillation, grouped validation, safe checkpoints, ONNX parity/export and
region-aware diagnostic inference. It predicts sky/cloud/contamination, with
no inferred five-class transparency labels or automatic promotion. This does
not add manifest-v2 reader support or automatic compute-aware selection.

## Layout

```
ml/
  astrom3/       AstroM3 integration: download.py, loader.py (cloud), edge.py (daemon retrofit)
  models/        per-frame model architectures (PyTorch); tri_modal_embedding.py = AstroM3 entrypoint
  data_pipeline/ dataset builders / splits
  preprocess/    crops, normalization, feature builders
  train/         training entry points
  eval/          evaluation + policy replay
  export/        ONNX export + metadata writer
  pipelines/     runnable end-to-end pipelines
  checkpoints/   downloaded weights (git-ignored)
```

The **current** FQE exporter and daemon resolver use the following layout at the
repo-root `models/` directory (v4 §7.6.10):

```
models/<model_name>/<version>/
    model.onnx         # FQE two-head graph loaded by vision/fqe_runner.rs
    metadata.json      # graph contract, provenance, hash, and promotion gates
    calibration.json   # hash-bound per-star + post-pooling uncertainty scales
    norm_params.json   # exact per-crop preprocessing contract
```

The current `metadata.json` fields (abridged; the ellipses below are descriptive)
match the Rust `ModelMetadata` struct:

```json
{
  "model_name": "focus_quality", "version": "1.0.0", "task": "focus_quality",
  "framework": "onnx",
  "input_format":  { "image": ["batch",1,64,64] },
  "output_format": { "mu_log_hfr": ["batch"], "log_var": ["batch"] },
  "sha256": "…", "created_at": "…Z", "training_info": { … }
}
```

This is the existing schema, not the proposed manifest v2. New documentation
must not cause an existing bundle to be relabelled as a different precision,
scientific variant or promotion state.

## Planned manifest v2 and compute-aware deployment

The v5 design separates scientific identity from the local execution choice:

| Identity | Meaning | Example |
|----------|---------|---------|
| Family | Stable scientific output and units | FQE: predicted HFR in pixels and calibrated log-HFR uncertainty |
| Variant | Architecture, capacity and input strategy evaluated together | A compact 64×64 model or a separately trained larger-crop model |
| Artifact | Immutable weights, graph, actual precision and contract hashes | An exported FP32 graph or a separately validated INT8 export |
| Runtime candidate | Artifact, execution provider/options and input/batch profile | One artifact on CPU with a specified thread count and star-count profile |
| Local qualification | Correctness and performance evidence for that candidate | Golden-vector parity, memory reserve and end-to-end latency results |

The proposed catalog will retain every available variant/artifact rather than
only the highest semantic version. A candidate layout is
`models/<family>/<variant>/<artifact-version>/`; adoption requires exporter and
daemon changes plus explicit migration of legacy bundles. The current daemon
continues to consume the existing layout above.

Manifest v2 must bind the following immutable information to artifact hashes:

- Family, variant and artifact identifiers; schema version; provenance and
  training/evaluation dataset revisions.
- Actual graph precision, input/output names, shapes and units. A provider's
  FP16 or INT8 option is an execution request, not proof of the resulting
  precision or graph placement; record and validate effective execution.
- Preprocessing and postprocessing contracts, including crop size, star gates,
  normalization, allowed batching, pooling and missing-value behavior.
- Golden inputs and expected outputs with justified numerical tolerances for
  Python-to-daemon and provider parity. Include preprocessing and aggregation,
  not just the ONNX tensor boundary.
- Hash-bound calibration and a family-specific scientific quality report.
  Compare variants on the same frozen real-data splits and grouped evaluation
  slices, preserving telescope/camera/session separation. FQE error, uncertainty
  coverage and failure rates are distinct from COC classification metrics.
- Supported runtime requirements, including operators, schema/runtime versions
  and shape bounds.

Promotion and revocation are separate attestations bound to the immutable bundle
digest. Updating lifecycle status does not edit the quality report. Installation
or a successful benchmark cannot confer scientific promotion or control authority.

Maintain independent state for artifact lifecycle (`reference`, `shadow`,
`promoted`, `retired`, `revoked`), installation/integrity, local qualification,
and current selection. These states serve different purposes: a verified file
can still be unqualified, and a locally fast shadow model still cannot control
equipment. Reference/random artifacts remain integration fixtures. Initially,
qualified FQE output remains shadow-only while classical HFR governs autofocus;
future authority must also pass the existing focus autonomy and safety gates.

Benchmark records and provider engine caches belong in the daemon's mutable
data directory, outside the immutable model bundle. Cache identity must include
artifact and contract hashes, daemon/runtime/provider versions, provider
options, device/driver, power mode, thread settings and input/batch/load profile.
Measure preprocessing, transfers, inference and postprocessing together; report
setup separately and record tail latency, deadline misses and peak memory under
representative sustained observing load. Selection must preserve memory and
timing reserves for capture, guiding and streaming.

The first implementation should start with classical processing and a qualified
CPU candidate when available, qualify additional candidates sequentially in an
allowed operating window, then pin the choice for an observing session. A GPU
inventory is advisory: only providers
compiled and successfully probed in that release are candidates. The planned
inspection APIs are `GET /api/v1/ai/models` and `GET /api/v1/ai/runtime`; they are
not present in the current daemon.

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
python -m ml.data_pipeline.build_focus_quality_dataset \
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
Full details, download, and the proposed daemon **retrofit** (`AstroM3Edge`: image +
metadata + telemetry, reusing AstroM3's components) are documented in
[`ml/astrom3/README.md`](astrom3/README.md).

```python
from ml.models.tri_modal_embedding import load_cloud, build_edge
cloud = load_cloud("base")   # AstroM3 (planning / apps.ai_engine)
edge  = build_edge()         # research prototype, not a qualified daemon model
```

## Daemon integration

The vision loop already batches 64×64 centered grayscale crops from classically
gated stars through `vision/fqe_runner.rs`. The session uses the default CPU
provider; there is no current GPU/provider selection. It robustly pools the
predictions and publishes shadow-only fields on
`SharedObservation`: `fqe_hfr_px`, `fqe_sigma_log_hfr`, `fqe_star_count`,
`fqe_model_version`, and `fqe_calibrated`. Classical HFR remains the autofocus
authority. Missing measurements stay `null`; there is deliberately no scalar
`focus_score`, `sharpness_score`, or fabricated seeing value.

When running from `achernar-daemon/`, set `ACHERNAR_MODELS_DIR=../models` to
discover eligible bundles in the repository's model directory. The current
automatic FQE resolver filters runtime eligibility and hashes before selecting
the newest semantic version. `[focus.models].quality_model_path` is an explicit
development override; output remains shadow. The proposed runtime must apply
artifact integrity, lifecycle and local qualification gates to every selection,
including manual pins and explicit paths.

## Public data intake for v6

The [public dataset runbook](data_pipeline/PUBLIC_DATA.md) documents the
pinned source releases, import commands, model tensor contracts, frozen splits
and remaining local-data requirements. The live intake validation report is
`ml/data/public/validation_report.json`; bulk transfer progress is
`ml/data/public/lenghu_sky8_bulk/progress.json`. Importing these datasets does
not train or promote a model. The [v6 architecture](../achernar_v6.tex) and
[v6 collection plan](../achernar_data_collection_v6.tex) define this workflow.
