"""Frozen human partial-mask training input and explicit teacher-only input."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from ml.coc import CLASS_NAMES, IGNORE_INDEX
from ml.data_pipeline.public_common import sha256_file
from ml.data_pipeline.import_public_cloud import CloudSegmentationDataset
from ml.data_pipeline.lenghu_bulk_dataset import LenghuTeacherDataset, excluded_dates, filename_timestamp


def asset(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Dataset path escapes its root")
    return path


def inspect_human_dataset(root, verify_assets=True):
    """Reject task/split drift before optimization; hashes bind every run."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {str(i): name for i, name in enumerate(CLASS_NAMES)} | {"255": "ignore"}
    if manifest.get("dataset_id") != "lenghu_sky8_human" or manifest.get("task") != "cloud_segmentation":
        raise ValueError("This trainer requires the reviewed Lenghu human adapter")
    if manifest["contract"]["classes"] != expected:
        raise ValueError("Human class ontology changed")
    rows = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
    if len(rows) != manifest["record_count"]:
        raise ValueError("Human population changed")
    seen_ids, groups, ancestors, hashes = set(), {}, {}, {}
    eligible = Counter()
    for row in rows:
        split = row["split"]
        if split not in {"train", "validation", "test"} or row["sample_id"] in seen_ids:
            raise ValueError("Duplicate sample ID or invalid frozen split")
        seen_ids.add(row["sample_id"])
        if row.get("label_type") != "human_partial_regions":
            raise ValueError("Human supervision cannot contain pseudo-labels")
        if row.get("five_class_label") is not None or row.get("transparency_label") is not None:
            raise ValueError("Unsupported transparency labels")
        for lookup, key in ((groups, row["split_group"]), (ancestors, row["ancestor"]), (hashes, row["rgb_sha256"])):
            if lookup.setdefault(key, split) != split:
                raise ValueError("Ancestor, image duplicate or episode crosses splits")
        if split != "train" or row["training_eligible"]:
            eligible[split] += 1
        if verify_assets:
            for field in ("image", "mask"):
                if sha256_file(asset(root, row[field])) != row[field + "_sha256"]:
                    raise ValueError(f"Human {field} content changed: {row['sample_id']}")
    return {
        "dataset_id": manifest["dataset_id"], "manifest_sha256": sha256_file(root / "manifest.json"),
        "records_sha256": sha256_file(root / "records.jsonl"), "record_count": len(rows),
        "eligible_split_counts": dict(eligible), "group_counts": dict(Counter(groups.values())),
        "source_assets": manifest["sources"], "license": manifest["license"],
        "limitations": manifest["limitations"], "assets_verified": verify_assets,
    }


class HumanCloudDataset(Dataset):
    def __init__(self, root, split, size=128, augment=False, cache=False):
        if size < 32 or size % 16:
            raise ValueError("Training size must be >=32 and a multiple of 16")
        if augment and split != "train":
            raise ValueError("Held-out data must not be augmented")
        self.base = CloudSegmentationDataset(root, split=split, size=size)
        self.records = self.base.records
        self.augment = augment
        self.cache = [self.base[i] for i in range(len(self.base))] if cache else None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.cache[index] if self.cache is not None else self.base[index]
        image = torch.from_numpy(np.ascontiguousarray(item["image"]))
        mask = torch.from_numpy(item["mask"].copy())
        if self.augment:
            # Apply identical D4 transforms to image and partial labels, never
            # interpolate categorical masks or fill ignored pixels with sky.
            k = int(torch.randint(4, ()).item())
            image, mask = torch.rot90(image, k, (-2, -1)), torch.rot90(mask, k, (-2, -1))
            if bool(torch.randint(2, ()).item()):
                image, mask = image.flip(-1), mask.flip(-1)
        row = self.records[index]
        return {"image": image.contiguous(), "mask": mask.contiguous(), "sample_id": item["sample_id"],
                "split_group": row["split_group"], "camera_era": row.get("camera_era", "unknown")}

    def training_pixel_counts(self):
        # Count the exact resized targets used for optimization, not holdouts.
        counts = np.zeros(3, dtype=np.int64)
        for i in range(len(self.base)):
            item = self.cache[i] if self.cache is not None else self.base[i]
            mask = item["mask"]
            if not np.isin(mask, [0, 1, 2, IGNORE_INDEX]).all():
                raise ValueError("Unexpected human mask labels")
            counts += np.bincount(mask[mask != IGNORE_INDEX], minlength=3)
        return counts


class TeacherCloudDataset(Dataset):
    """A bounded, deterministically sampled pseudo-label ablation, never eval."""
    def __init__(self, root, human_root, size=128, limit=4096, seed=0):
        if limit < 1:
            raise ValueError("Teacher limit must be positive")
        self.base = LenghuTeacherDataset(root)
        self.size = size
        forbidden = excluded_dates(human_root)
        if any(str(filename_timestamp(row[0]).date()) in forbidden for row in self.base.records):
            raise ValueError("Stale teacher index includes a human holdout date")
        human_rows = [json.loads(line) for line in (Path(human_root) / "records.jsonl").read_text().splitlines()]
        heldout_hashes = {r["rgb_sha256"] for r in human_rows if r["split"] != "train"}
        # Deterministic hash sampling spans the release instead of taking its
        # chronological prefix. Selection is recorded for reproducible ablations.
        self.indices = sorted(range(len(self.base)), key=lambda i: hashlib.sha256(
            f"{seed}:{self.base.records[i][0]}".encode()).digest())[:limit]
        if not self.indices:
            raise ValueError("No imported teacher training candidates")
        self.ids = [self.base.records[i][0] for i in self.indices]
        self.heldout_hashes = heldout_hashes
        self.identity = {"supervision": "producer_teacher_logits", "selected_count": len(self.indices),
                         "selected_ids_sha256": hashlib.sha256("\n".join(self.ids).encode()).hexdigest(),
                         "adapter_manifest_sha256": sha256_file(Path(root) / "adapter_manifest.json"),
                         "human_holdout_dates_excluded": True,
                         "exact_rgb_duplicate_policy": "reject every loaded duplicate before loss; unvisited pool members not audited",
                         "calibrated_teacher": False, "source_timezone_verified": False}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = self.base[self.indices[index]]
        rgb = np.rint(item["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        if hashlib.sha256(rgb.tobytes()).hexdigest() in self.heldout_hashes:
            # Fail closed rather than silently accepting duplicate holdouts from
            # another filename/date. This check applies to every selected image.
            raise ValueError("Teacher image duplicates a human holdout")
        image = Image.fromarray(rgb).resize((self.size, self.size), Image.Resampling.BILINEAR)
        return {"image": torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255),
                "teacher_logits": torch.from_numpy(item["teacher_logits"].copy()),
                "content_mask": torch.from_numpy(item["content_mask"].copy()), "sample_id": item["sample_id"]}
