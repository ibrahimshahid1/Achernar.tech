"""Pinned native AstroM3 intake, retaining official splits and ragged photometry.

Run ``python -m ml.data_pipeline.import_public_astrom3 --root ml/data/public``.
This imports catalog modalities, not the proposed image/guider Edge retrofit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .public_common import download_file, sha256_file, write_json, write_jsonl

REPOSITORY = "AstroMLCore/AstroM3Processed"
REVISION = "26805c17447e720e49403051e80177e980b51f1e"
DATASET_ID = "astrom3_full42"
FILES = [
    ("train", "train-00000-of-00002.parquet", 232042521,
     "638e011ad37e6426e0269cbaa5d5501db6741dab1377036062c3b025d6296c98"),
    ("train", "train-00001-of-00002.parquet", 228830000,
     "ac9b0a977ade397a9d627b87dc2eb574b8e7c740db035e3058bcf06221ca7e97"),
    ("validation", "validation-00000-of-00001.parquet", 58651209,
     "6f81b420220c9000af123ba1a344d6f1a4d799d25723c9c86743677c6ce72b06"),
    ("test", "test-00000-of-00001.parquet", 60521504,
     "8da6daf8af06742fd9004751b9652399e48548e60b5b72ee19274c4713d6e497"),
]
EXPECTED_COUNTS = {"train": 17063, "validation": 2152, "test": 2225}


def process_photometry(photometry, seq_len=200, how="center", rng=None):
    """Upstream trimming/padding, without changing already processed features."""
    array = np.asarray(photometry, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 9 or not np.isfinite(array).all():
        raise ValueError("Expected finite native photometry [observations,9]")
    if seq_len <= 0 or how not in {"center", "random"}:
        raise ValueError("Invalid sequence policy")
    length = len(array)
    if length > seq_len:
        start = ((length - seq_len) // 2 if how == "center" else
                 int((rng or np.random.default_rng()).integers(0, length - seq_len + 1)))
        return array[start:start + seq_len].copy(), np.ones(seq_len, dtype=np.float32)
    output = np.zeros((seq_len, 9), dtype=np.float32)
    output[:length] = array
    mask = np.zeros(seq_len, dtype=np.float32)
    mask[:length] = 1
    return output, mask


def validate_row(row):
    photo = np.asarray(row["photometry"], dtype=np.float32)
    spectra = np.asarray(row["spectra"], dtype=np.float32)
    metadata = np.asarray(row["metadata"], dtype=np.float32)
    label = int(row["label"])
    if photo.ndim != 2 or photo.shape[1] != 9 or len(photo) == 0:
        raise ValueError("Invalid native photometry dimensions")
    if spectra.shape != (3, 2575) or metadata.shape != (34,):
        raise ValueError("Native checkpoint contract mismatch")
    if not all(np.isfinite(value).all() for value in (photo, spectra, metadata)):
        raise ValueError("Nonfinite native modality value")
    if not 0 <= label < 10:
        raise ValueError("Unknown variable-star class")
    return photo, spectra, metadata, label


def pack_rows(rows):
    validated = [validate_row(row) for row in rows]
    offsets = np.cumsum([0] + [len(row[0]) for row in validated], dtype=np.int64)
    return {
        "photometry_values": np.concatenate([row[0] for row in validated]),
        "photometry_offsets": offsets,
        "spectra": np.stack([row[1] for row in validated]),
        "metadata": np.stack([row[2] for row in validated]),
        "label": np.asarray([row[3] for row in validated], dtype=np.int64),
    }


def prepare(root: str | Path, shard_size=256, download_only=False) -> dict:
    import pyarrow.parquet as pq

    root = Path(root) / DATASET_ID
    raw = root / "source"
    raw.mkdir(parents=True, exist_ok=True)
    sources = []
    for split, filename, size, digest in FILES:
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/full_42/{filename}"
        receipt = download_file(url, raw / filename, expected_sha256=digest,
                                expected_size=size, max_bytes=250_000_000)
        receipt["path"] = str(Path(receipt["path"]).relative_to(root))
        sources.append(receipt)
    card_url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/README.md"
    card = download_file(card_url, raw / "README.md", max_bytes=100_000)
    card["path"] = str(Path(card["path"]).relative_to(root))
    sources.append(card)
    write_json(root / "source_receipts.json", sources)
    if download_only:
        return {"dataset_id": DATASET_ID, "downloaded_bytes": sum(s["bytes"] for s in sources)}
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")

    output = root / "shards"
    output.mkdir(exist_ok=True)
    records, shards, label_names = [], [], None
    counts, labels = Counter(), Counter()
    fingerprints = {}
    cross_split_duplicates = []
    for split, filename, _, _ in FILES:
        parquet = pq.ParquetFile(raw / filename)
        metadata = parquet.schema_arrow.metadata or {}
        if b"huggingface" in metadata:
            features = json.loads(metadata[b"huggingface"])["info"]["features"]
            names = features["label"].get("names")
            if label_names is not None and names != label_names:
                raise ValueError("Class ordering changed between source files")
            label_names = names
        file_row = 0
        for batch_index, batch in enumerate(parquet.iter_batches(batch_size=shard_size)):
            rows = batch.to_pylist()
            arrays = pack_rows(rows)
            shard_path = output / f"{Path(filename).stem}-{batch_index:04d}.npz"
            temporary = shard_path.with_suffix(".npz.partial")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(shard_path)
            relative = str(shard_path.relative_to(root))
            shards.append({"path": relative, "split": split, "rows": len(rows),
                           "sha256": sha256_file(shard_path), "bytes": shard_path.stat().st_size})
            for index, row in enumerate(rows):
                photo, spec, meta, label = validate_row(row)
                # Exact full-modality duplicates are detectable even without object IDs.
                digest = hashlib.sha256(photo.tobytes() + spec.tobytes() + meta.tobytes()).hexdigest()
                sample_id = f"{filename}:{file_row + index}"
                if digest in fingerprints and fingerprints[digest][0] != split:
                    cross_split_duplicates.append([fingerprints[digest][1], sample_id])
                fingerprints.setdefault(digest, (split, sample_id))
                records.append({"sample_id": sample_id, "source_id": f"{REPOSITORY}@{REVISION}",
                    "source_file": f"source/{filename}", "source_row": file_row + index,
                    "split": split, "split_group": f"source-row:{sample_id}",
                    "split_group_semantics": "official_split_row_identity_unverified",
                    "object_id": None, "identity_verified": False,
                    "image_pairing_eligible": False, "training_eligible": split == "train",
                    "task": "native_astrom3", "shard": relative, "shard_row": index,
                    "modality_sha256": digest, "photometry_length": len(photo), "label": label})
                counts[split] += 1
                labels[f"{split}:{label}"] += 1
            file_row += len(rows)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected split population: {dict(counts)}")
    if cross_split_duplicates:
        write_json(root / "cross_split_duplicates.json", cross_split_duplicates)
        raise ValueError("Exact modality duplicates cross official splits; review before training")

    write_jsonl(root / "records.jsonl", records)
    manifest = {
        "schema_version": 1, "dataset_id": DATASET_ID, "source_repository": REPOSITORY,
        "source_revision": REVISION, "configuration": "full_42", "task": "native_astrom3",
        "status": "prepared_native_research", "record_count": len(records),
        "split_counts": dict(counts), "label_counts": dict(labels), "label_names": label_names,
        "records": "records.jsonl", "sources": sources, "shards": shards,
        "contract": {"photometry": [200, 9], "photometry_mask": [200],
                     "spectra": [3, 2575], "metadata": [34], "label": "integer 0..9",
                     "dtype": "float32 modalities; int64 labels and offsets",
                     "storage": "ragged photometry preserved in safe NPZ shards",
                     "normalization": "upstream processed values; no second scaling",
                     "training_window": "seeded random crop, zero pad with mask",
                     "evaluation_window": "center crop, zero pad with mask"},
        "license": {"processed_card": "MIT", "original_loader": "CC BY 4.0",
                    "checkpoint_weights": "CC BY 4.0",
                    "disposition": "native research with retained attribution; provenance discrepancy unresolved"},
        "attribution": ["Rizhko & Bloom, AstroM3, arXiv:2411.08842",
                        f"https://huggingface.co/datasets/{REPOSITORY}",
                        "https://huggingface.co/datasets/AstroMLCore/AstroM3Dataset"],
        "checks": {"finite_modalities": True, "shapes": True, "official_splits_preserved": True,
                   "exact_modality_cross_split_duplicates": 0},
        "limitations": ["Processed rows lack object IDs; row index is not astronomical identity.",
                        "Official partitions retained; object-level disjointness not independently verified.",
                        "No image/telemetry pairing or AstroM3Edge supervision created.",
                        "No checkpoint inference or training is established by import."]}
    write_json(root / "manifest.json", manifest)
    return manifest


class NativeAstroM3Dataset:
    """NumPy-returning map dataset compatible with torch DataLoader collation.

    Training windows vary deterministically with ``set_epoch``. Only the source
    partition requested is opened. No model or torch dependency is required here.
    """
    def __init__(self, path: str | Path, split="train", seed=42):
        self.root = Path(path)
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        self.split, self.seed, self.epoch = split, seed, 0
        with (self.root / "records.jsonl").open() as stream:
            self.records = [row for line in stream if (row := json.loads(line))["split"] == split]
        self._cache_path, self._cache = None, None

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.root / record["shard"]
        if self._cache_path != path:
            with np.load(path, allow_pickle=False) as source:
                self._cache = {key: source[key] for key in source.files}
            self._cache_path = path
        i = record["shard_row"]
        lo, hi = self._cache["photometry_offsets"][i:i + 2]
        raw_photo = self._cache["photometry_values"][lo:hi]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        photo, mask = process_photometry(raw_photo, how="random" if self.split == "train" else "center", rng=rng)
        return {"photometry": photo, "photometry_mask": mask,
                "spectra": self._cache["spectra"][i].copy(),
                "metadata": self._cache["metadata"][i].copy(),
                "label": self._cache["label"][i]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="ml/data/public")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    result = prepare(args.root, args.shard_size, args.download_only)
    print(json.dumps({key: result[key] for key in ("dataset_id", "record_count", "split_counts", "downloaded_bytes") if key in result}))


if __name__ == "__main__":
    main()
