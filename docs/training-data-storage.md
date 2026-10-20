# Achernar training data storage

Updated: October 17, 2026.

## Migration status

**Complete — October 17, 2026, 8:11 p.m. EDT.** Every one of the 7,049 files
matched its source SHA-256 hash. Directory entry sets, file types, sizes and
permissions also matched. Dataset loading passed through both the explicit
external root and the compatibility links before the internal copies were
removed.

Internal available space rose from **3.47 GB to 102.94 GB** during removal
(an observed increase of 99.47 GB, including filesystem allocation effects and
any concurrent system activity). The migrated file contents total 98.37 GB.
The completion receipt is
`/Volumes/AchernarData/achernar_training_data/migration-2026-10-17/verification.json`.

Loader smoke checks covered samples from all cloud and native AstroM3 splits,
five spread-out Lenghu teacher pairs, SQLite integrity, all teacher holdout-date
exclusions, both hash-bound FQE split registries, and presence of the RAW store
and capture utility. These are storage/adapter checks; no model was trained.
The existing human training eligibility exclusion is preserved: 768 train rows,
767 eligible loader examples, 150 validation and 193 test examples.

## External location

The user approved a separate ordinary APFS volume, **AchernarData**, on the
physical Seagate USB drive that also contains **SEGATE_1TB**. The existing
SEGATE_1TB volume is a protected Time Machine backup volume; it denies creating
ordinary folders. Its ACL and backup contents must not be changed for ML data.

AchernarData has a **400 GB quota**, no reserved allocation, and shares the
container's free space with Time Machine. Creating it did not erase or
repartition the drive. The training data root is:

```text
/Volumes/AchernarData/achernar_training_data/
  public/                     Public sources, prepared datasets, manifests and indexes
  raws/                       Existing local cloud originals and capture utility
  migration-2026-10-17/        SHA-256 inventory, transfer log and verification receipt
```

Volume UUID: `C7A2214B-913A-4A19-91B3-33D6F3FB60BE`.
The protected Time Machine volume retains UUID
`38AB9DB6-503F-4D57-93B4-F7E1939C1E92`.

The migration covers **7,049 files**, **98,372,905,967 logical bytes**
(98.37 GB / 91.62 GiB):

| Folder | Files | Logical bytes |
| --- | ---: | ---: |
| `public` | 6,505 | 86,107,549,487 |
| `raws` | 544 | 12,265,356,480 |

## Paths for future agents

Compatibility symlinks preserve the existing repository paths:

```text
ml/data/public -> /Volumes/AchernarData/achernar_training_data/public
ml/data/raws   -> /Volumes/AchernarData/achernar_training_data/raws
```

Keep these machine-specific symlinks and all dataset payloads out of Git. The
repository's source code, environments, model artifacts and other data are not
part of this move. Existing manifests, frozen splits, original images, packed
TAR archives and intake receipts retain their contents. Historical absolute
paths in intake receipts describe their original execution location.

Before training, importing or resuming downloads, confirm that the actual
external volume is mounted:

```sh
mount | rg 'on /Volumes/AchernarData '
test -r /Volumes/AchernarData/achernar_training_data/public/validation_report.json
```

Do not create a replacement directory under `/Volumes/AchernarData` when the
drive is absent, or replace a dangling compatibility symlink with an internal
directory. Mount the correct drive first. The import tools also accept an
explicit public-data root:

```sh
python -m ml.data_pipeline.validate_public_data \
  --root /Volumes/AchernarData/achernar_training_data/public \
  --hashes --require-complete
```

Run from the repository root with an existing environment containing the ML
dependencies. The full validator writes refreshed derived reports/indexes; the
migration verification receipt describes the initial byte-preserving copy.
Keep all public dataset subfolders together so teacher holdout exclusions can
find the human annotations. Keep the Lenghu TARs packed rather than expanding
another full copy.

## Training implications

External storage is compatible with the current path-based dataset adapters.
Moving unchanged inputs preserves their hashes and frozen partitions. USB/HDD
read speed and random seeks can limit batch loading, especially the Lenghu
image/logit TAR reader. Benchmark an actual training run before changing
workers or caching; a bounded active subset can be staged on a faster SSD if
needed. PyTorch recommends tuning loader workers to the workload and data
location in its [performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

Keep the drive connected during data access and eject it after jobs stop.
The adjacent Time Machine volume is on the same physical disk and does not
constitute an independent backup of the relocated local RAW originals.
This storage migration does not establish model quality or hardware qualification.

Apple documents [adding APFS volumes and shared space](https://support.apple.com/guide/disk-utility/add-delete-or-erase-apfs-volumes-dskua9e6a110/mac)
and [using a separate volume alongside Time Machine](https://support.apple.com/guide/mac-help/types-of-disks-you-can-use-with-time-machine-mh15139/mac).
