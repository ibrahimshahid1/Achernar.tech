"""Hash and join the existing R50 collection without creating cloud labels.

Only originals, embedded camera JPEGs, and the prior human-review inventory are
read. No model, checkpoint, prediction, or automatic weather join is consulted.
The output distinguishes capture ancestors, exploratory clock-gap groups, and
approved labels. Existing images are development material, not prospective test.

Requires Pillow and ExifTool. Run from any directory with --repo-root.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import PIL
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path, key: str) -> dict[str, dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row[key]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate {key} in {path}")
    return indexed


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inventory_derivatives(prior: Path, originals: list[Path], native_hashes: dict[str, str]) -> list[dict]:
    """Keep display derivatives separate; verify full JPEGs when available."""
    full_paths = sorted((prior / "previews").glob("IMG_*_full.jpg"))
    full_originals = [next(path for path in originals if path.stem == full.stem.removesuffix("_full")) for full in full_paths]
    full_embedded = {}
    if full_originals:
        result = subprocess.run(
            ["exiftool", "-j", "-b", "-JpgFromRaw", *map(str, full_originals)],
            check=True, capture_output=True,
        )
        for item in json.loads(result.stdout):
            value = item.get("JpgFromRaw", "")
            if not value.startswith("base64:"):
                raise ValueError(f"Missing full embedded JPEG for {item['SourceFile']}")
            full_embedded[Path(item["SourceFile"]).stem] = base64.b64decode(value[7:], validate=True)
    assets = []
    for path in sorted((prior / "previews").glob("*.jpg")):
        match = re.fullmatch(r"(IMG_\d+)(?:_(full|center_crop|diagnostic))?\.jpg", path.name)
        stem, kind = match.groups() if match else ("", "composite_display")
        kind = kind or "base_preview"
        data = path.read_bytes()
        verified = kind == "base_preview"
        if kind == "full":
            if data != full_embedded[stem]:
                raise ValueError(f"Full JPEG differs from original JpgFromRaw: {path.name}")
            verified = True
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
        recipe = {
            "base_preview": "byte-identical camera embedded PreviewImage; verified against CR3",
            "full": "byte-identical camera embedded JpgFromRaw; verified against CR3",
            "center_crop": "prior native-resolution center-crop display; exact crop geometry/code not retained in the audit",
            "diagnostic": "prior notes declare gamma-0.5 diagnostic display; exact render pipeline not reproduced",
            "composite_display": "contact sheet with text/tiles; no single exposure ancestor",
        }[kind]
        assets.append({
            "asset_path": str(path.resolve()), "asset_sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "width": width, "height": height, "asset_kind": kind,
            "claimed_parent_filename": f"{stem}.CR3" if stem else "",
            "claimed_parent_native_sha256": native_hashes.get(f"{stem}.CR3", ""),
            "exact_embedded_lineage_verified": verified,
            "transform_provenance": recipe,
            "paired_inference_base_input": kind == "base_preview",
            "additional_independent_capture": False,
            "role": "base_RGB_input" if kind == "base_preview" else "adjudication_reference_only_not_extra_observation",
        })
    return assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    prior = repo / "docs/cloud-data-audit-2026-10-11"
    raw_dir = repo / "ml/data/raws/clouds"
    output = (args.output or repo / "docs/coc-error-audit-2026-10-13/r50-inventory").resolve()
    if output == raw_dir or raw_dir in output.parents or output == prior or prior in output.parents:
        raise ValueError("Output must not overwrite originals or the prior audit")

    originals = sorted(raw_dir.glob("*.CR3"))
    review = csv_rows(prior / "review_queue.csv", "filename")
    labels = csv_rows(prior / "qualitative/cloud_labels.csv", "filename")
    preview_metrics = csv_rows(prior / "previews/preview_metrics.csv", "file")
    metadata = {row["FileName"]: row for row in json.loads((prior / "metadata.json").read_text())}
    expected = set(review)
    if not expected or any(set(rows) != expected for rows in (labels, preview_metrics, metadata)):
        raise ValueError("Prior review/label/preview/metadata inventories do not have identical keys")
    if {path.name for path in originals} != expected:
        raise ValueError("Current original population differs from the prior audited population; review additions/removals explicitly")

    before = {path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in originals}
    with ThreadPoolExecutor(max_workers=4) as pool:
        native_hashes = dict(zip((path.name for path in originals), pool.map(sha256, originals)))
    mismatches = [name for name in expected if native_hashes[name] != review[name]["sha256"]]
    if mismatches:
        raise ValueError(f"Originals changed since prior audit: {mismatches}")
    print(f"Verified {len(originals)} original hashes", flush=True)

    extraction = subprocess.run(
        ["exiftool", "-j", "-b", "-PreviewImage", *map(str, originals)],
        check=True, capture_output=True,
    )
    embedded = {}
    for item in json.loads(extraction.stdout):
        name = Path(item["SourceFile"]).name
        encoded = item.get("PreviewImage", "")
        if name in embedded or not encoded.startswith("base64:"):
            raise ValueError(f"Missing or duplicate embedded preview for {name}")
        embedded[name] = base64.b64decode(encoded[len("base64:"):], validate=True)
    if set(embedded) != expected:
        raise ValueError("Embedded preview extraction did not cover every original")

    rows = []
    input_hash_groups = defaultdict(list)
    decoded_hash_groups = defaultdict(list)
    for path in originals:
        name = path.name
        previous = review[name]
        label = labels[name]
        meta = metadata[name]
        preview = prior / "previews" / f"{path.stem}.jpg"
        data = preview.read_bytes()
        preview_hash = hashlib.sha256(data).hexdigest()
        if data != embedded[name] or preview_hash != preview_metrics[name]["preview_sha256"]:
            raise ValueError(f"Existing preview is not identical to its original embedded JPEG: {name}")
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.mode != "RGB" or image.size != (1620, 1080):
                raise ValueError(f"Unexpected preview mode/size for {name}: {image.mode}/{image.size}")
            rgb_hash = hashlib.sha256(image.tobytes()).hexdigest()
            width, height = image.size
        input_hash_groups[preview_hash].append(name)
        decoded_hash_groups[rgb_hash].append(name)
        status = label["annotation_status"]
        invalid = status.startswith("invalid")
        role = "development_rejection_or_contamination_challenge" if invalid else (
            "development_blinded_region_adjudication" if label["cloud_class_code"] else
            "development_input_and_blinded_review"
        )
        rows.append({
            "filename": name,
            "original_path": str(path.resolve()),
            "original_bytes": path.stat().st_size,
            "original_sha256": native_hashes[name],
            "ancestor_id": "sha256:" + native_hashes[name],
            "capture_family_proxy": previous["run_gap_30min"],
            "proximity_burst_proxy": previous["burst_gap_5min"],
            "family_basis": "prior camera-clock gap grouping; actual independent session/night unverified",
            "conservative_split_group": "r50_existing_reviewed_collection_all_v1",
            "camera_timestamp_unreliable": previous["camera_timestamp"],
            "camera_clock_reliable": False,
            "corrected_capture_time": "",
            "actual_observing_night": "",
            "camera_model": meta.get("Model", ""),
            "lens_model": meta.get("LensModel", ""),
            "exposure_seconds": previous["exposure_seconds"],
            "iso": previous["iso"],
            "f_number": previous["f_number"],
            "focal_length_mm": previous["focal_length_mm"],
            "focus_mode": previous["focus_mode"],
            "white_balance": previous["white_balance"],
            "sensor_scalar_black_adu_prior": previous["raw_scalar_black_adu"],
            "sensor_white_adu_prior": previous["raw_white_adu"],
            "sensor_clipped_fraction_prior": previous["raw_sample_fraction_at_white"],
            "preview_path": str(preview.resolve()),
            "preview_sha256": preview_hash,
            "preview_rgb_sha256": rgb_hash,
            "preview_width": width,
            "preview_height": height,
            "embedded_preview_identity_verified": True,
            "render_contract": "camera_embedded_PreviewImage_JPEG_bytes_v1",
            "decoded_contract": f"Pillow-{PIL.__version__};RGB;no-resize;no-exif-transpose;no-stretch",
            "prior_initial_disposition": previous["initial_disposition"],
            "prior_screening_reason": previous["screening_reason"],
            "prior_annotation_status": status,
            "prior_cloud_class_code": label["cloud_class_code"],
            "prior_cloud_class_name": label["cloud_class_name"],
            "prior_alternate_classes": label["alternate_class_codes"],
            "prior_label_source": label["label_source"],
            "prior_label_uncertainty": label["label_uncertainty"],
            "prior_visual_reason": label["visual_reason"],
            "prior_review_method": label["review_method"],
            "prior_label_training_approved": label["training_approved"],
            "adjudication_status": "not_adjudicated_for_three_class_segmentation",
            "approved_segmentation_pixels": 0,
            "independent_roi_approved": False,
            "cloud_metrics_eligible": False,
            "proposed_role": role,
            "prospective_test_eligible": False,
            "automatic_training_approval": False,
        })
    assets = inventory_derivatives(prior, originals, native_hashes)
    after = {path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in originals}
    if before != after:
        raise ValueError("An original's size/mtime changed during inventory")

    duplicate_jpeg = [names for names in input_hash_groups.values() if len(names) > 1]
    duplicate_rgb = [names for names in decoded_hash_groups.values() if len(names) > 1]
    families = []
    for family in sorted({row["capture_family_proxy"] for row in rows}):
        members = [row for row in rows if row["capture_family_proxy"] == family]
        families.append({
            "capture_family_proxy": family, "files": len(members),
            "filename_first": members[0]["filename"], "filename_last": members[-1]["filename"],
            "burst_proxies": sorted({row["proximity_burst_proxy"] for row in members}),
            "prior_annotation_counts": dict(Counter(row["prior_annotation_status"] for row in members)),
            "prior_class_counts": dict(Counter(row["prior_cloud_class_name"] for row in members)),
            "exposure_seconds_counts": dict(Counter(row["exposure_seconds"] for row in members)),
            "focal_length_counts": dict(Counter(row["focal_length_mm"] for row in members)),
            "role": "development_only; independent night/session unverified",
        })

    sources = [
        prior / "review_queue.csv", prior / "inventory.csv", prior / "metadata.json",
        prior / "qualitative/cloud_labels.csv", prior / "qualitative/CLOUD_CLASSES.md",
        prior / "previews/preview_metrics.csv", prior / "previews/adjacent_preview_differences.csv",
        prior / "visual_notes.md", Path(__file__).resolve(),
    ]
    with (prior / "previews/adjacent_preview_differences.csv").open(newline="") as handle:
        differences = list(csv.DictReader(handle))
    exact_only_previews = [p for p in (prior / "previews").glob("IMG_*.jpg") if re.fullmatch(r"IMG_\d+\.jpg", p.name)]
    derived_extra = sorted(str(p.relative_to(repo)) for p in (prior / "previews").glob("IMG_*.jpg") if p not in exact_only_previews)
    summary = {
        "schema_version": "r50_existing_inventory_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "original_files": len(rows), "original_bytes": sum(row["original_bytes"] for row in rows),
        "unique_original_sha256": len(set(native_hashes.values())),
        "all_original_hashes_match_prior": True, "original_size_mtime_unchanged_during_read": True,
        "embedded_previews_verified": len(rows), "unique_preview_byte_hashes": len(input_hash_groups),
        "unique_decoded_rgb_hashes": len(decoded_hash_groups),
        "duplicate_preview_byte_groups": duplicate_jpeg, "duplicate_decoded_rgb_groups": duplicate_rgb,
        "prior_annotation_status_counts": dict(Counter(row["prior_annotation_status"] for row in rows)),
        "prior_cloud_class_counts": dict(Counter(row["prior_cloud_class_name"] for row in rows)),
        "automatic_label_approvals": 0, "approved_segmentation_pixels": 0, "approved_roi_images": 0,
        "verified_actual_capture_times": 0, "verified_observing_nights": 0,
        "prospective_test_images": 0,
        "prior_adjacent_display_pairs": len(differences),
        "prior_adjacent_display_mae_below_1_over_255": sum(float(row["display_resized_rgb_mae"]) < 1 / 255 for row in differences),
        "adjacent_metric_caveat": "Prior resized-JPEG appearance metric, not independent-scene count or RAW duplicate test",
        "Pillow_version": PIL.__version__, "ExifTool_version": subprocess.check_output(["exiftool", "-ver"], text=True).strip(),
        "derived_display_assets": len(assets),
        "derived_asset_kind_counts": dict(Counter(asset["asset_kind"] for asset in assets)),
        "excluded_derived_individual_jpegs": derived_extra,
        "sources": [{"path": str(p.relative_to(repo)), "sha256": sha256(p)} for p in sources],
        "model_outputs_consulted": False,
        "limitations": [
            "The old 0-4 labels are provisional image review; they do not label sky/cloud/contamination pixels",
            "Camera clock is known wrong; gap groups are proxies, never verified nights",
            "All existing material is reviewed development input, not an untouched prospective test",
            "Camera-rendered JPEG is a separate RGB input domain, not linear RAW or optical attenuation truth",
            "No per-frame sky ROI or segmentation annotation is approved",
            "No new RAW decode, deterministic RAW render, pixel mask, weather join, capture, training or model inference was performed",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "inventory.csv", rows)
    write_csv(output / "derived-assets.csv", assets)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "capture-families.json").write_text(json.dumps(families, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ["original_files", "unique_original_sha256", "embedded_previews_verified", "unique_preview_byte_hashes", "prior_annotation_status_counts", "prior_cloud_class_counts", "prospective_test_images"]}, indent=2))


if __name__ == "__main__":
    main()
