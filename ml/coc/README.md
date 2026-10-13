# COC segmentation research runbook

This implementation makes the optional segmentation auxiliary task in the v6
COC plan executable. Its versioned contract is `coc-segmentation-v1`: rendered
RGB images produce three pixel logits, ordered **sky=0, cloud=1,
contamination=2**. Human target value **255 means ignore**. Unannotated or
conflicting regions never become clear-sky examples.

The five legacy diagnostic names—clear, thin, moderate, heavy and overcast—are
not training targets here. Segmentation coverage does not identify optical
transparency, rain risk, or the correct decision to pause observing. The legacy
`CloudClassifier` constructor remains explicitly unsupported; use
`CloudSegmenter` for this separate task. These artifacts carry
`runtime_eligible=false` and confer no daemon or equipment-control authority.

## Data and model contract

The reviewed adapter at `ml/data/public/lenghu_sky8_human` contains the complete
1,111-image partially annotated Lenghu benchmark. Its frozen partitions are:

| Partition | Source records | Eligible images | Temporal episode groups | Annotated pixels at native 512×512 |
| --- | ---: | ---: | ---: | ---: |
| Training | 768 | 767 | 167 | 22.22% |
| Validation | 150 | 150 | 28 | 19.89% |
| Reserved final test | 193 | 193 | 47 | 24.15% |

One training image, `2018-05-15-17-19-12`, has only excluded open annotations and
therefore has no supervised pixels. The loader omits it from training. The
reported annotation fractions use all native source records; resized run
metrics record the annotation support actually used at that resolution.

Groups conservatively join adjacent filename dates and exact RGB duplicates.
The filename timezone and astronomical nighttime status remain unverified.
Both source camera eras appear in every partition, so an era breakdown does
not establish unseen-camera performance. These are source-domain episode
groups, not certified independent nights or local Canon R50 validation.

The training inspector checks image/mask hashes, class ontology, and separation
of sample IDs, ancestors, duplicate-image hashes and episode groups. RGB inputs
are bilinearly resized to a square and divided by 255; categorical targets use
nearest-neighbor resize. Only training receives paired rotations and flips.
The trainer derives inverse-square-root class weights from training pixels at
the selected resolution. Validation uses unweighted partial cross entropy.

| Variant | Parameters | Architecture |
| --- | ---: | --- |
| `tiny` | 27,499 | Depthwise separable U-Net, base width 8, GroupNorm |
| `small` | 98,259 | Same architecture, base width 16 |

Neither downloads pretrained weights. Training sizes must be at least 32 and a
multiple of 16; the examples use 128×128. Hold architecture and resolution
fixed when comparing human-only and teacher-expanded training. Changing both
capacity and data at once cannot isolate the value of the public expansion.

Almería is a separate daylight diagnostic source. This trainer deliberately
requires the reviewed Lenghu human adapter rather than silently pooling those
domains. Dataset intake and its source/license records are described in
`ml/data_pipeline/PUBLIC_DATA.md`.

## Setup and checks

Run commands from the repository root. The lock file records the Python 3.12
macOS arm64 dependency closure used for this implementation; another platform
needs its own installation and numerical-parity checks.

```sh
cd /Users/ibrahimshahid/Documents/Github/Achernar.tech
export COC_ENV=/private/tmp/achernar-coc-python
export UV_CACHE_DIR=/private/tmp/achernar-coc-uv-cache
uv venv --python 3.12 "$COC_ENV"
uv pip install --python "$COC_ENV/bin/python" -r ml/requirements-coc.lock.txt
export COC_PYTHON="$COC_ENV/bin/python"

"$COC_PYTHON" -m pytest -q \
  ml/tests/test_cloud_segmentation_model.py \
  ml/tests/test_cloud_segmentation_metrics.py \
  ml/tests/test_cloud_segmentation_pipeline.py \
  ml/tests/test_coc_experiment_comparison.py \
  ml/tests/test_public_cloud_import.py
```

The isolated environment is temporary and may need recreating after system
cleanup. `ml/requirements-coc.txt` records the direct dependencies;
`ml/requirements-coc.lock.txt` pins their complete installed closure. The tests
cover model/loss contracts, ignored pixels, episode metrics, paired transforms,
provenance rejection, and an actual synthetic-fixture training/export round
trip. Synthetic fixtures establish execution correctness, not astronomical
quality.

The human adapter must already exist. To create it independently of the large
teacher release:

```sh
"$COC_PYTHON" -m ml.data_pipeline.import_public_cloud \
  --root ml/data/public --dataset lenghu
```

This small import is sufficient for the initial human-only experiment. It does
not require the approximately 79 GB bulk release.

## Human-only experiment

Use a fresh output directory for every run; the trainer rejects nonempty
directories. The following is an explicit CPU baseline configuration:

```sh
"$COC_PYTHON" -m ml.train.train_cloud_segmentation \
  --human-root ml/data/public/lenghu_sky8_human \
  --output ml/checkpoints/coc/human-tiny-experiment01 \
  --variant tiny --size 128 --epochs 12 --batch-size 16 \
  --learning-rate 0.001 --seed 17 --device cpu --threads 4
```

`--device auto` selects an available CUDA or MPS backend before CPU. Backend
availability alone does not establish performance parity; the explicit CPU
configuration provides a reproducible local starting point. `--no-cache`
reduces resident dataset memory. `--no-augment` permits a separately named
augmentation ablation.

Training writes `run.json`, epoch `history.json`, `last.pt`, the
validation-selected `best.pt`, and a final validation-only `quality.json`.
Metadata records configuration, versions, dataset fingerprints, source hashes,
class weights and research status. Keep checkpoint and sidecars together and
do not edit them after training. Model selection uses minimum unweighted
validation partial cross entropy. The trainer does not compute final-test
performance.

The built-in reference predicts the majority class among **training annotated
pixels**. It is a segmentation reference, not a comparison against the daemon's
existing frame-level cloud heuristic: partial region labels and that heuristic
do not currently share a verified whole-frame target. Such a comparison needs
adjudicated local frame/decision labels on identical held-out groups.

The initial `human-tiny-v1` CPU experiment completed 12 epochs. Its validation
mean IoU is 0.507 versus 0.182 for the majority reference, with contamination
recall only 0.332. It remains a research baseline. See the
[execution report](../../docs/coc-implementation-2026-10-12/README.md) for the
bound artifacts, parity evidence, previews and limitations. The small variant
is implemented and tested but has not been trained in this initial experiment.

## Explicit teacher ablation

Start this comparison after freezing the available bulk intake/index revision.
Do not rebuild its SQLite pairing index while a downloader/finalizer or
training job is using it. A teacher root must contain `pairs.sqlite3`, the
producer geometry masks, the adapter manifest, and checksum-verified source
TARs. A complete intake is preferable when comparing reproducible subsets;
if using a partial intake, name and preserve that revision explicitly.

Keep the human partitions, architecture, resolution, seed and optimizer setup
the same as the human-only comparison. These teacher settings are a bounded
experimental configuration, not validated optimal hyperparameters:

```sh
"$COC_PYTHON" -m ml.train.train_cloud_segmentation \
  --human-root ml/data/public/lenghu_sky8_human \
  --output ml/checkpoints/coc/human-teacher-tiny-experiment01 \
  --variant tiny --size 128 --epochs 12 --batch-size 16 \
  --learning-rate 0.001 --seed 17 --device cpu --threads 4 \
  --teacher-root ml/data/public/lenghu_sky8_bulk \
  --teacher-limit 4096 --teacher-weight 0.25 \
  --teacher-temperature 2 --teacher-confidence 0.8
```

`--teacher-limit` bounds the selected candidate pool. Selection hashes the seed
and sample ID across the available release rather than taking a chronological
prefix; the run preserves `teacher_selected_ids.json` and its digest. One
teacher batch accompanies each human batch, restarting the teacher iterator
if exhausted. A selected pool member is not guaranteed to be visited every
epoch, or at all in a short run. Keep the pool and dataset revision fixed for a
comparison; expanding a partial intake can change the selected pool.

Teacher candidates exclude human validation/test filename dates and adjacent
dates. Each loaded teacher image is also rejected if its decoded RGB hash
duplicates a human holdout. That lazy check does not constitute a completed
exact-duplicate audit of every unvisited pool member. The teacher loss uses
only producer-content pixels meeting its softmax selection threshold, aligns
student logits to the 64×64 teacher grid, and keeps teacher tensors detached.
Teacher confidence is uncalibrated; logits remain model-derived supervision
rather than additional human truth. No teacher-derived targets enter the human
validation metrics. Increase the pool only to answer a named validation
residual, using the same human-only comparison.

For an already started candidate, the finite finalizer waits for `quality.json`,
checks the matched experiment and bound artifacts, exports with the full
validation parity check, and writes a descriptive comparison. It exits after
completion, an error or the bounded wait; it does not schedule another run or
promote a model. Use fresh bundle/report paths:

```sh
"$COC_PYTHON" -m ml.pipelines.finalize_coc_experiment \
  --baseline ml/checkpoints/coc/human-tiny-v1 \
  --candidate ml/checkpoints/coc/human-teacher-tiny-v1 \
  --bundle ml/artifacts/coc/human-teacher-tiny-v1 \
  --report docs/coc-implementation-2026-10-12/teacher-comparison.json \
  --parity-image ml/data/public/lenghu_sky8_human/images/2018-05-29-02-49-49.png \
  --training-log ml/checkpoints/coc/human-teacher-tiny-v1.log
```

This instance completed and exited. Read
`ml/checkpoints/coc/human-teacher-tiny-v1-job.json` for its phase and error
details; do not launch a duplicate against its existing outputs. Use fresh
candidate, bundle and report paths to reproduce the workflow. A successful result binds the comparison
to the exported manifest. A validation gain is evidence for another research
experiment, not an operational promotion criterion.

The completed teacher-assisted run selected epoch 11. Mean IoU rose from
0.5066 to 0.6334 on the same partial human validation regions, while
contamination recall remained low at 0.4035. Its export passes all validation
logit tolerances, with one class-index change on an ignored near-tie pixel.
See the [implementation and comparison report](../../docs/coc-implementation-2026-10-12/README.md)
for per-class results, fixed previews, hashes and remaining limits.

## Evaluate without consuming the final test

The default evaluation path checks checkpoint provenance and the frozen split
revision, then evaluates validation on CPU:

```sh
"$COC_PYTHON" -m ml.eval.eval_cloud_segmentation \
  --checkpoint ml/checkpoints/coc/human-tiny-experiment01/best.pt \
  --split validation \
  --output ml/checkpoints/coc/human-tiny-experiment01/validation-recheck.json
```

Reports include labeled-pixel counts, annotated coverage, confusion matrices,
class IoU/precision/recall, balanced accuracy, episode macro metrics, and episode
and camera-era strata. Unsupported denominators yield JSON `null`. Accuracy on
these annotated regions is not whole-sky coverage accuracy, calibrated cloud
presence, thin-cloud recall or a false-pause rate. Group summaries are not
confidence intervals or proof of independent nights.

Reserve the 193-image test for a frozen final comparison. The evaluator rejects
`--split test` unless explicitly given `--allow-final-test`. Only after
architecture, transforms, thresholds and comparisons are fixed would a final
evaluation use:

```sh
"$COC_PYTHON" -m ml.eval.eval_cloud_segmentation \
  --checkpoint ml/checkpoints/coc/human-tiny-experiment01/best.pt \
  --split test --allow-final-test \
  --output ml/checkpoints/coc/human-tiny-experiment01/frozen-final-test.json
```

Do not include this command in routine development runs. If final-test failures
guide training or threshold changes, record that the test has been consumed
and obtain fresh independent evidence before another final claim.

## Export and image inference

Export requires completed training sidecars and the bound best checkpoint. Use
a training or validation image for numerical parity, with a fresh export folder:

```sh
"$COC_PYTHON" -m ml.export.export_cloud_segmentation \
  --checkpoint ml/checkpoints/coc/human-tiny-experiment01/best.pt \
  --output ml/artifacts/coc/human-tiny-experiment01 \
  --parity-image ml/data/public/lenghu_sky8_human/images/2018-04-16-02-01-38.png
```

The ONNX graph has FP32 weights, convolutions and I/O, with FP64 centered-moment
normalization arithmetic. This mixed internal precision resolves measured
export discrepancies and is recorded explicitly. It has dynamic batch size and
fixed trained spatial size. Export checks real-image, black, white, random
batch-of-two, and every frozen validation image against PyTorch at
`atol=rtol=1e-4`. A failed comparison emits no valid bundle manifest.
The bundle includes the graph, hash-bound manifest, preprocessing/postprocessing
contracts, golden tensors, parity evidence, training metadata and validation
quality report. This research bundle is separate from daemon manifest v2 and
is not installed into the daemon.

Supply a rendered 8-bit image with EXIF orientation already applied. Preserve
the RAW original and the rendering recipe separately; linear RAW/FITS/ADU data
are not interchangeable with this model's RGB input. For a diagnostic region
summary, provide an **independently defined** camera/sky-content mask matching
the original image dimensions. Use 0 outside and 1 or 255 inside. A human cloud
annotation, the model's predicted sky, and an automatic all-image mask are not
substitutes for a reviewed valid input region.

Replace these two example paths with the rendered image and its independent ROI:

```sh
"$COC_PYTHON" -m ml.coc.predict \
  --bundle ml/artifacts/coc/human-tiny-experiment01 \
  --image /absolute/path/to/rendered-sky.png \
  --roi /absolute/path/to/camera-content-mask.png \
  --output ml/artifacts/coc/prediction-experiment01
```

Use `--checkpoint .../best.pt` instead of `--bundle ...` for PyTorch CPU
inference. Both paths write class-index `classes.png`, uncalibrated
`scores.npz`, a color `overlay.png`, and provenance/summary `prediction.json` at
the model's spatial resolution. With no ROI, maps are still produced but the
region summary abstains. Empty ROIs, uniform RGB regions, no classified sky and
contamination above the diagnostic threshold also abstain. The default contamination threshold of
0.3 is an unqualified research setting.

Cloud area within the ROI, cloud fraction among pixels classified as sky/cloud,
and contamination fraction are separate model estimates. Softmax scores do not
become calibrated whole-frame cloud probabilities. Before operational use,
the remaining work includes adjudicated local clear/thin and contamination
examples, a matched frame-level heuristic comparison, independent local
validation, calibration and runtime/input parity. Temporal hysteresis and
weather safety remain outside this network. Training and exporting a bundle
do not promote it or change any existing observing policy.
