# Public datasets for Achernar v6

This intake implements the source and label boundaries in `achernar_v6.tex` and
`achernar_data_collection_v6.tex`. Importing a file establishes neither model
quality nor authority to control equipment. The machine-readable live status is
`ml/data/public/validation_report.json`; a source manifest describes only that
source. Download progress lives in `lenghu_sky8_bulk/progress.json`.

## Admitted inputs and model contracts

| Folder | Contents and consumer | Split / label boundary |
|---|---|---|
| `lenghu_sky8_human` | All 1,111 embedded RGB images and partial human masks; `CloudSegmentationDataset` returns float32 `[3,512,512]`, int64 `[512,512]` targets | 768 train / 150 validation / 193 test; sky=0, cloud=1, contamination=2, ignore=255. Unannotated/conflicting pixels are ignored. These are not five transparency classes. |
| `lenghu_sky8_bulk` | Full 200 TAR image/logit release, 79,422,294,016 bytes, 429,620 pairs when complete; `LenghuTeacherDataset` lazily reads images and float32 raw logits `[3,64,64]` | Teacher predictions only. Training candidates exclude every human holdout filename date and its adjacent dates. Returns a boolean producer content mask `[64,64]`; no pseudo-label evaluation set is manufactured. |
| `almeria_v2` | All 818 daylight frames and original height masks; separate cloud auxiliary dataset | 251 train / 519 validation / 48 official test. Validation expanded to whole camera/date groups. Cloud height is not optical transparency. |
| `astrom3_full42` | Exactly 21,440 rows from one pinned `full_42` configuration; safe ragged NPZ shards and `NativeAstroM3Dataset` | Official 17,063 / 2,152 / 2,225 partitions. Outputs photometry `[200,9]`, mask `[200]`, spectra `[3,2575]`, metadata `[34]`, integer label. No object IDs or image pairing are invented. |
| `stellar_sim_v1` | 2,400 observations from 800 analytic source families; 1,025 accepted FQE crops, 1,398 TED-eligible examples, 32 BGE scenes | All family derivatives stay together. FQE `[N,64,64]` + `log_hfr`, TED `[N,32,32]` + measured finite-pixel shape references, BGE known background and preclip variance. This generator is not GalSim or a calibrated rig. |
| `stellar_fits_pilot` | Original HST ACS/WFC subarray FLT, SCI/ERR/DQ, 49 quality-accepted HFR references | One exposure, diagnostic test only. Calibrated electrons are not raw Canon ADU. |
| `hst_acs_corpus_v1` | Selected 16 original HST ACS/WFC FLT exposures, 2,694,182,400 source bytes; 1,059 strict FQE references and 4,825 references including experimental masked wings | Actual FITS dates: before 2011 train; 2011–2013 validation; 2014 onward test. Strict counts: 777 / 159 / 123. Same field/instrument: a temporal feasibility test, not unseen-rig qualification. |

Use the final manifests for actual completed counts; a queued archive is not a
downloaded archive. The full Lenghu transfer is staged at the user's request;
its availability does not make every pseudo-label useful. Compare the human-only
baseline with a bounded distillation subset before consuming the full corpus.

FQE NPZs match `ml/train/train_focus_quality.py`: `crops`, `hfr_px`, `log_hfr`,
`run_id`, plus provenance and frozen `fqe.splits.json`. Compatibility `run_id`
values are **singleton sample IDs**, explicitly not focus runs. Set
`--lambda-rank 0`; rank ordering and signed offsets are unsupported. The current
trainer's grouped sampler then uses singleton batches, so a normal shuffled
pointwise sampler should be implemented before an efficient large training run.
No training, export, promotion, uncertainty calibration or hardware actuation
is performed by these import commands.

TED uses the v6 native 32-pixel research contract, with ellipticity `1-b/a`
and axial image angle in radians. `ml/models/trailing_cnn.py` still declares
a 64-pixel probability-head skeleton and raises `NotImplementedError`.
Reconcile that model with the v6 task before neural training; resizing these
stamps or calling an image-axis label RA/Dec would change the scientific target.

The cloud adapter supports paired resize (bilinear images, nearest masks).
Its research segmentation contract differs from the legacy whole-frame COC
head. An appropriate segmentation trainer/export/runtime path still needs to
be built. TED and BGE start with their classical baselines; source availability
alone is not a reason to train their optional neural variants.

## Reproduce or resume

Run these commands from the repository root with Python 3.12 or later. Install
the isolated import requirements rather than changing the application's venv:

```sh
python3.12 -m venv .venv-public-data
.venv-public-data/bin/python -m pip install -r ml/requirements-public-data.lock.txt
.venv-public-data/bin/python -m ml.data_pipeline.prepare_public_data --stage small
.venv-public-data/bin/python -m ml.data_pipeline.prepare_public_data --stage bulk --workers 4
.venv-public-data/bin/python -m ml.data_pipeline.validate_public_data --hashes --require-complete
```

`--root /Volumes/AchernarData/public` is supported by each import entry point.
Keep all datasets under that one root so the bulk holdout exclusions can find
the human records. The store is ignored by Git. Do not run two bulk downloaders
against the same root. Downloaded complete files are checksum-verified and
reused; interrupted individual files are downloaded again. The downloader
retains a 20 GB free-space reserve and uses a bounded 85 GB archive budget.
TARs stay packed and are indexed by byte offset, avoiding a second 79 GB copy.
Source hashes are verified at download time; `--hashes` additionally rechecks
prepared AstroM3 shards, and FQE registry loading verifies its dataset hash.
Full cloud payloads are decoded during validation; bulk tensors are validated
when loaded, with 32 spread-out pairs sampled in the aggregate report.

For an already-running bulk transfer, the finite completion job can wait for
it, retry incomplete downloads, finalize HST metadata, build the complete
pair index and validate the prepared data:

```sh
python -m ml.data_pipeline.finalize_public_data --wait-download
```

`ml/data/public/intake_job.json` records its PID, phase and errors. The current
run also writes `intake_job.log`. It is not a scheduled automation and does not
train anything. Run only one completion job against a store. A failed job
retains complete downloads; rerun after addressing the reported error.

Individual stages are available:

```sh
python -m ml.data_pipeline.import_public_cloud --dataset all
python -m ml.data_pipeline.import_public_astrom3
python -m ml.data_pipeline.import_public_imaging --max-real-sources 2000
python -m ml.data_pipeline.import_hst_corpus --max-sources 2000
python -m ml.data_pipeline.import_lenghu_bulk --workers 4
python -m ml.data_pipeline.lenghu_bulk_dataset
```

Source pins: Lenghu bulk `30f97ef28201a4eb4ec639c20f19c3b166c56bcb`;
AstroM3Processed `26805c17447e720e49403051e80177e980b51f1e`. HST product IDs
and expected byte counts are frozen in `hst_acs_selection_v1.json`; SHA-256
receipts pin the acquired revisions of archive products. HST products can be
reprocessed upstream; a future replacement must become a new recorded intake.
The local synthetic generator records its source hash, seed and library version.

## Known quality limitations

Lenghu contains misspelled labels, four open line annotations and one two-point
polygon. Explicit spelling aliases and ignored invalid shapes are recorded in
the repair ledger. Human splits merge adjacent calendar dates and exact RGB
duplicates into 242 episodes; their timezone and independence as observing
nights remain unverified. Bulk pairs use exact source timestamps and conservative
date exclusions; full cross-date bulk pixel deduplication has not been performed.
The bulk corpus also includes daylight; no unverified timestamp-to-solar-altitude
conversion or nighttime-only claim is made. Producer calibration coefficients,
special camera variants, content masks and interruption CSV are retained.

Almería's source YAML has inconsistent structure/names. The importer preserves
it without guessing joins. AstroM3's official partitions are preserved, with no
exact full-modality duplicates found across them, but processed rows lack IDs
needed to independently establish object disjointness or train AstroM3Edge.

The default HST corpus `fqe.npz` contains only unmasked-aperture references.
`fqe_all_references.npz` retains the additional experimental masked-wing
references with explicit quality tiers; they are not enabled by the default
dataset. The source-level pilot is diagnostic only.

HST source selection is deliberately narrow: NGC104 aliases, ACS/WFC, 10–120 s,
one selected exposure per year from the first 100 query results. Only SCI1 is
processed; the complete original retains both chips. SEP masks nonzero DQ,
nonfinite data and invalid ERR. Masked cores, saturation, crowded apertures,
blends, severe elongation and low SNR are rejected. References with at most 2%
masked wings are marked `mask_aware_experimental`, remain unqualified and have
masked input pixels replaced by the estimated background. The raw science,
mask, error estimate and rejection reasons remain available. FLT optical
distortion, residual detector effects and omitted CTE/pixel-area corrections
limit transfer; HFR is a measurement reference, not physical ground truth.
See [ACS DQ definitions](https://www.stsci.edu/hst/instrumentation/acs/data-analysis/dq-flag-definitions)
and [MAST data use](https://stdatu.stsci.edu/publishing/data-use).

## Sources deferred under v6

| Proposal | Disposition |
|---|---|
| MILAN monthly raw FITS / DeepSpaceYolo | Research-only option under NC-ND terms, not necessary for the first permitted training corpus. Native CFA metadata and measured local transfer would need a separate pilot. Object boxes are not focus/tracking labels. |
| HSC visits / PSF models | Noncommercial archive conditions and professional-survey domain; optional research extension, not a product-training prerequisite. |
| GalSim | Optional stronger simulator; this intake labels its own analytic generator honestly. Adopt it for named optical deficiencies rather than mislabeling existing simulations. |
| SWINSEG / SWINySEG / WOANC | Request-based access and/or noncommercial terms; Singapore releases overlap. No form submitted or author contacted. |
| DeepSky | Release completeness not established; no guessed download. |
| LCO, Rubin OpSim, PHD2 | Parser/simulator/telemetry research inputs; do not supply identified local focus trajectories, logging propensities, causal policy rewards or mount residuals. Not bulk-downloaded as substitute control-training data. |
| BTSbot / Trailblazer | Optional artifact/streak challenge sets; satellite streaks and transient detection examples are not labeled tracking-induced stellar elongation. |
| Multimodal Universe | Optional AstroM3Edge research. A 100 TB archive is unnecessary for native AstroM3, and the required object-level image/modality matches do not yet exist here. |

FOP still needs verified motor positions and focus outcomes; FDF needs per-rig
thermal history; exposure/filter/guiding/dither policies need attributed actions
and outcomes; pointing models need actual mount residuals. Local sensor
calibration remains necessary. Existing R50 originals and provisional labels
were left untouched. The full source assessments are in
`docs/public-dataset-audit-2026-10-12/`.

## External backup disk

`/Volumes/SEGATE_1TB` has an APFS Backup role and a root ACL denying folder/file
creation. Its approximately 634 GB free space does not make the backup volume
writable for dataset storage. A separate ordinary APFS volume can share the
existing container's space, as described in
[Apple's guidance](https://support.apple.com/guide/mac-help/types-of-disks-you-can-use-with-time-machine-mh15139/mac).
No backup ACL, snapshot, partition or volume was changed during the probe.
Adding `AchernarData` with a 400 GB quota was presented for approval. Downloads
continue internally until a usable data volume is available.
