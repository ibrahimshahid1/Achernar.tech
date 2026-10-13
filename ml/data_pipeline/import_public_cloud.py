"""Human cloud segmentation intake; never invent five transparency labels."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from .public_common import download_file, grouped_split, sha256_file, write_json, write_jsonl

CLASS_MAP = {"sky": 0, "cloud": 1, "contamination": 2}
LABEL_ALIASES = {"contination": "contamination", "contimination": "contamination", "clode": "cloud"}
IGNORE = 255
LENGHU_SHA = "5414955ad66aacc21c545c820b6aafeb2c2cd4e494347737e9c6a764aaf31839"
LENGHU_URL = "https://huggingface.co/datasets/ruiyicheng/LenghuSky-8/resolve/main/data/segmentation.zip"
ALMERIA = [
    ("kontas_2017.zip", 15626097, "cf4967a91954ada20847c79ce7d1e4f80d61fa31ff639ac7740cdd083d1eea53"),
    ("test_set.zip", 1198235, "a022ee092fdccecc5173cf87faa7d481698954f3a12f50cf8de282e512730476"),
]


def rasterize_annotation(annotation):
    """Conservative partial regions; conflicting labels and open lines ignored."""
    width, height = int(annotation["imageWidth"]), int(annotation["imageHeight"])
    if (width, height) != (512, 512):
        raise ValueError("Unexpected human benchmark dimensions")
    classes = np.zeros((3, height, width), dtype=bool)
    ledger = []
    for index, shape in enumerate(annotation["shapes"]):
        original = shape["label"]
        canonical = LABEL_ALIASES.get(original, original)
        if canonical not in CLASS_MAP:
            raise ValueError(f"Unknown source class: {original}")
        if original != canonical:
            ledger.append({"shape_index": index, "original": original, "canonical": canonical,
                           "action": "explicit_spelling_alias", "geometry_unchanged": True})
        if shape.get("shape_type", "polygon") != "polygon":
            ledger.append({"shape_index": index, "original": original,
                           "action": "open_shape_ignored", "shape_type": shape.get("shape_type")})
            continue
        points = np.asarray(shape["points"], dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            ledger.append({"shape_index": index, "original": original,
                           "action": "invalid_polygon_ignored", "point_count": len(points)})
            continue
        region = Image.new("L", (width, height), 0)
        ImageDraw.Draw(region).polygon([tuple(point) for point in points], fill=1)
        classes[CLASS_MAP[canonical]] |= np.asarray(region, dtype=bool)
    coverage = classes.sum(axis=0)
    mask = np.full((height, width), IGNORE, dtype=np.uint8)
    for label in range(3):
        mask[classes[label] & (coverage == 1)] = label
    return mask, ledger, int((coverage > 1).sum())


def temporal_groups(stamps, image_hashes):
    """Merge adjacent calendar dates and exact-image duplicates conservatively.

    Source timezone is unverified. These are temporal episodes, not certified
    independent nights or solar-altitude labels. Duplicates override dates.
    """
    dates = sorted(set(stamp.date() for stamp in stamps))
    episode, mapping, previous = None, {}, None
    for date in dates:
        if previous is None or (date - previous).days > 1:
            episode = str(date)
        mapping[date] = f"lenghu-episode:{episode}"
        previous = date
    groups = [mapping[stamp.date()] for stamp in stamps]
    parents = {group: group for group in set(groups)}
    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group
    duplicate_groups = {}
    for group, digest in zip(groups, image_hashes, strict=True):
        if digest in duplicate_groups:
            a, b = find(group), find(duplicate_groups[digest])
            parents[max(a, b)] = min(a, b)
        else:
            duplicate_groups[digest] = group
    return [find(group) for group in groups]


def prepare_lenghu(root):
    destination = Path(root) / "lenghu_sky8_human"
    source = destination / "source"
    receipt = download_file(LENGHU_URL, source / "segmentation.zip", expected_sha256=LENGHU_SHA,
                            expected_size=22104744, max_bytes=25_000_000)
    receipt["path"] = "source/segmentation.zip"
    for folder in ("images", "masks", "annotations"):
        (destination / folder).mkdir(parents=True, exist_ok=True)
    records, stamps, hashes, correction_ledger = [], [], [], []
    with zipfile.ZipFile(source / "segmentation.zip") as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".json"))
        if len(names) != 1111:
            raise ValueError("Unexpected source human benchmark count")
        for name in names:
            annotation_bytes = archive.read(name)
            annotation = json.loads(annotation_bytes)
            stem = Path(name).stem
            stamp = datetime.strptime(stem, "%Y-%m-%d-%H-%M-%S")
            embedded = base64.b64decode(annotation["imageData"], validate=True)
            with Image.open(io.BytesIO(embedded)) as image:
                rgb = image.convert("RGB")
                if rgb.size != (512, 512):
                    raise ValueError("Unexpected embedded image dimensions")
                rgb.save(destination / "images" / f"{stem}.png")
                digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            mask, ledger, conflicts = rasterize_annotation(annotation)
            Image.fromarray(mask).save(destination / "masks" / f"{stem}.png")
            (destination / "annotations" / f"{stem}.json").write_bytes(annotation_bytes)
            for event in ledger:
                correction_ledger.append({"sample_id": stem, **event})
            stamps.append(stamp)
            hashes.append(digest)
            counts = {str(value): int((mask == value).sum()) for value in (0, 1, 2, IGNORE)}
            records.append({"sample_id": stem, "source_id": "LenghuSky-8/human",
                "ancestor": f"segmentation.zip:{name}", "source_member": name,
                "image": f"images/{stem}.png", "mask": f"masks/{stem}.png",
                "annotation": f"annotations/{stem}.json", "task": "cloud_segmentation",
                "image_sha256": sha256_file(destination / "images" / f"{stem}.png"),
                "mask_sha256": sha256_file(destination / "masks" / f"{stem}.png"),
                "rgb_sha256": digest, "source_annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
                "source_embedded_image_sha256": hashlib.sha256(embedded).hexdigest(),
                "source_timestamp": stamp.isoformat(), "source_timezone": None,
                "provisional_observing_date": str((stamp - timedelta(hours=12)).date()),
                "nighttime_verified": False, "illumination": "unverified",
                "camera_era": "before_geometry_change" if stem < "2023-09-27-18-09-48" else "after_geometry_change",
                "pixel_counts": counts, "conflict_pixels_ignored": conflicts,
                "annotation_events": ledger, "label_type": "human_partial_regions",
                "five_class_label": None, "transparency_label": None,
                "training_eligible": counts["255"] < 512 * 512})
    groups = temporal_groups(stamps, hashes)
    partitions = grouped_split(groups)
    for record, group in zip(records, groups, strict=True):
        record["split_group"] = group
        record["split"] = partitions[group]
    write_jsonl(destination / "records.jsonl", records)
    write_json(destination / "annotation_ledger.json", correction_ledger)
    manifest = {"schema_version": 1, "dataset_id": "lenghu_sky8_human", "task": "cloud_segmentation",
        "status": "prepared_partial_human", "record_count": len(records),
        "records": "records.jsonl", "sources": [receipt],
        "split_counts": dict(Counter(record["split"] for record in records)),
        "independent_group_count": len(set(groups)),
        "contract": {"image": [512, 512, 3], "image_dtype": "uint8 RGB",
                     "mask": [512, 512], "mask_dtype": "uint8",
                     "classes": {"0": "sky", "1": "cloud", "2": "contamination", "255": "ignore"},
                     "preprocessing": "RGB /255 for starter loader; source image transform retained",
                     "task_excludes": ["five transparency classes", "absolute optical attenuation"]},
        "license": {"dataset_card": "Apache-2.0", "teacher_weights_used": False},
        "annotation_event_counts": dict(Counter(event["action"] for event in correction_ledger)),
        "attribution": ["https://huggingface.co/datasets/ruiyicheng/LenghuSky-8",
                        "https://arxiv.org/abs/2603.16429"],
        "limitations": ["Human benchmark includes partial masks; unannotated/conflicting regions ignored.",
                        "Filename timezone and astronomical nighttime unverified; adjacent dates grouped conservatively.",
                        "Spelling aliases recorded explicitly; four open line shapes and one two-point polygon excluded.",
                        "Source geometry change retained; no claim of unseen-camera or local R50 validity."]}
    write_json(destination / "manifest.json", manifest)
    return manifest


def prepare_almeria(root):
    destination = Path(root) / "almeria_v2"
    source = destination / "source"
    receipts = []
    for name, size, digest in ALMERIA:
        receipt = download_file(f"https://zenodo.org/records/16647156/files/{name}?download=1",
                                source / name, expected_sha256=digest, expected_size=size, max_bytes=20_000_000)
        receipt["path"] = f"source/{name}"
        receipts.append(receipt)
    for folder in ("images", "masks", "height_masks"):
        (destination / folder).mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(source / "kontas_2017.zip") as archive:
        entries = archive.read("kontas_2017/validation.csv").decode().splitlines()[1:]
        validation = {entry.strip().rstrip(",") for entry in entries if entry.strip()}
    for archive_name in ("kontas_2017.zip", "test_set.zip"):
        official_test = archive_name == "test_set.zip"
        with zipfile.ZipFile(source / archive_name) as archive:
            for name in sorted(n for n in archive.namelist() if n.lower().endswith(".jpg")):
                stem = Path(name).stem
                camera = Path(name).parent.name if official_test else "Cloud_Cam_Kontas"
                timestamp = (datetime.strptime(stem[:14], "%Y%m%d%H%M%S") if official_test else
                             datetime.strptime(stem.split("_")[-1], "%y%m%d%H%M%S"))
                sample_id = f"{camera}_{stem}"
                label_name = name.replace("/images/", "/seg_masks/").rsplit(".", 1)[0] + ".png"
                image_bytes, mask_bytes = archive.read(name), archive.read(label_name)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    rgb = image.convert("RGB")
                    if rgb.size != (512, 512):
                        raise ValueError("Almeria image dimensions changed")
                    rgb.save(destination / "images" / f"{sample_id}.png")
                    rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
                with Image.open(io.BytesIO(mask_bytes)) as image:
                    height_mask = np.asarray(image)
                if height_mask.shape != (512, 512) or not set(np.unique(height_mask)) <= {0, 1, 2, 3, 4}:
                    raise ValueError("Unknown Almeria mask contract")
                mask = np.full(height_mask.shape, IGNORE, dtype=np.uint8)
                mask[height_mask == 1] = 0
                mask[np.isin(height_mask, [2, 3, 4])] = 1
                Image.fromarray(mask).save(destination / "masks" / f"{sample_id}.png")
                (destination / "height_masks" / f"{sample_id}.png").write_bytes(mask_bytes)
                records.append({"sample_id": sample_id, "source_id": "zenodo:16647156",
                    "ancestor": f"{archive_name}:{name}", "source_member": name,
                    "image": f"images/{sample_id}.png", "mask": f"masks/{sample_id}.png",
                    "height_mask": f"height_masks/{sample_id}.png", "rgb_sha256": rgb_sha,
                    "image_sha256": sha256_file(destination / "images" / f"{sample_id}.png"),
                    "mask_sha256": sha256_file(destination / "masks" / f"{sample_id}.png"),
                    "camera": camera, "source_timestamp": timestamp.isoformat(),
                    "source_timezone": None, "illumination": "daytime_source_domain",
                    "split_group": f"almeria:{camera}:{timestamp.date()}",
                    "official_split": "test" if official_test else "validation" if stem in validation else "train",
                    "task": "cloud_segmentation_daylight_diagnostic", "five_class_label": None,
                    "transparency_label": None, "training_eligible": False})
    # Entire camera/date follows validation if any member was officially validation.
    validation_groups = {row["split_group"] for row in records if row["official_split"] == "validation"}
    for record in records:
        record["split"] = ("test" if record["official_split"] == "test" else
                           "validation" if record["split_group"] in validation_groups else "train")
        record["training_eligible"] = record["split"] == "train"
    # Reject cross-partition exact duplicates rather than quietly leaking a test.
    seen = {}
    for row in records:
        old = seen.setdefault(row["rgb_sha256"], row["split"])
        if old != row["split"]:
            raise ValueError("Almeria exact image duplicate crosses official-derived partitions")
    if len(records) != 818 or sum(row["official_split"] == "test" for row in records) != 48:
        raise ValueError("Unexpected Almeria source population")
    write_jsonl(destination / "records.jsonl", records)
    manifest = {"schema_version": 1, "dataset_id": "almeria_v2", "record_count": len(records),
        "task": "cloud_segmentation_daylight_diagnostic", "status": "prepared_auxiliary_diagnostic",
        "records": "records.jsonl", "sources": receipts,
        "split_counts": dict(Counter(row["split"] for row in records)),
        "official_split_counts": dict(Counter(row["official_split"] for row in records)),
        "contract": {"image": [512, 512, 3], "mask": [512, 512],
                     "classes": {"0": "sky", "1": "cloud", "255": "ignore"},
                     "source_height_classes": {"0": "camera_geometry", "1": "sky", "2": "low", "3": "mid", "4": "high"}},
        "license": {"dataset": "CC BY 4.0"},
        "attribution": ["https://doi.org/10.5281/zenodo.16647156",
                        "https://amt.copernicus.org/articles/15/797/2022/"],
        "limitations": ["Daylight diagnostic; not nighttime transparency labels.",
                        "Official validation expanded to whole camera/date groups; official test preserved.",
                        "Source metadata YAML is malformed/inconsistent; camera paths retained without speculative joins.",
                        "Test filenames include 2021 and 2023; source description year not substituted."]}
    write_json(destination / "manifest.json", manifest)
    return manifest


class CloudSegmentationDataset:
    """Returns normalized RGB CHW and uint8 targets, with optional paired resize."""
    def __init__(self, path, split="train", size=None):
        self.root, self.size = Path(path), size
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        with (self.root / "records.jsonl").open() as stream:
            self.records = [row for line in stream if (row := json.loads(line))["split"] == split
                            and (split != "train" or row["training_eligible"])]
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(self.root / row["image"]) as source:
            image = source.convert("RGB")
            if self.size:
                image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        with Image.open(self.root / row["mask"]) as source:
            mask = source
            if self.size:
                mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
            targets = np.asarray(mask, dtype=np.int64).copy()
        return {"image": pixels, "mask": targets, "sample_id": row["sample_id"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="ml/data/public")
    parser.add_argument("--dataset", choices=["all", "lenghu", "almeria"], default="all")
    args = parser.parse_args()
    for key, function in (("lenghu", prepare_lenghu), ("almeria", prepare_almeria)):
        if args.dataset in {"all", key}:
            result = function(args.root)
            print(json.dumps({key: result[key] for key in ("dataset_id", "record_count", "split_counts")}))


if __name__ == "__main__":
    main()
