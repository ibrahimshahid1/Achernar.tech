"""Paired research inference on hash-verified R50 camera-rendered previews.

This computes model agreement and conditional ROI diagnostics, never local
accuracy from provisional ordinal labels. It never opens the public final test.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from ml.coc.inference import preprocess_image, summarize_logits
from ml.coc.predict import verify_bundle
from ml.data_pipeline.public_common import sha256_file


def checked_input(row, size):
    path = Path(row["preview_path"])
    if sha256_file(path) != row["preview_sha256"]:
        raise ValueError(f"Changed preview: {row['filename']}")
    image, geometry = preprocess_image(path, size)
    if geometry != [int(row["preview_width"]), int(row["preview_height"])]:
        raise ValueError("Preview geometry does not match inventory")
    return image, geometry


def reviewed_region(row, review, size):
    if review is None:
        return None
    if (review["filename"] != row["filename"]
            or review["preview_sha256"] != row["preview_sha256"]
            or review["original_sha256"] != row["original_sha256"]):
        raise ValueError("Review must identify this exact source and preview")
    rect = review.get("rect")
    if rect is None:
        return None
    if (len(rect) != 4 or not all(np.isfinite(rect))
            or not 0 <= rect[0] < rect[2] <= 1
            or not 0 <= rect[1] < rect[3] <= 1):
        raise ValueError("Invalid independent review rectangle")
    # Match the deployed rectangle's original-pixel-center selection followed
    # by the same nearest-neighbor target resize as the offline ROI loader.
    width, height = int(row["preview_width"]), int(row["preview_height"])
    xs = (np.arange(width) + .5) / width
    ys = (np.arange(height) + .5) / height
    region = ((ys[:, None] >= rect[1]) & (ys[:, None] < rect[3])
              & (xs[None, :] >= rect[0]) & (xs[None, :] < rect[2]))
    return np.asarray(Image.fromarray(region.astype(np.uint8)).resize(
        (size, size), Image.Resampling.NEAREST)).astype(bool)


def output_record(logits, region, image):
    if (logits.shape != (3, image.shape[-2], image.shape[-1])
            or logits.dtype != np.float32 or not np.isfinite(logits).all()):
        raise ValueError("Unexpected model output")
    summary, classes, scores = summarize_logits(logits, region)
    if region is not None and region.any() and np.all(np.ptp(image[:, region], axis=1) <= 1e-6):
        summary.update(status="abstained", reason="uninformative_rgb_input",
                       cloud_area_fraction_of_roi=None, cloud_fraction_of_classified_sky=None,
                       contamination_area_fraction_of_roi=None, mean_soft_cloud_share_of_sky=None)
    return {
        "summary": summary,
        "full_frame_predicted_class_counts": np.bincount(classes.ravel(), minlength=3).tolist(),
        "full_frame_mean_max_softmax": float(scores.max(axis=0).mean()),
        "full_frame_normalized_entropy": float((-(scores * np.log(scores.clip(1e-12))).sum(axis=0)
                                               / np.log(3)).mean()),
        "full_frame_distribution_scope": "unmasked diagnostic class counts; not sky coverage or accuracy",
    }, classes


def render_review(output, stem, image, maps, region, records):
    size = image.shape[-1]
    rgb = np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8)
    colors = np.array([[50, 120, 220], [240, 240, 240], [230, 110, 40]], dtype=np.uint8)
    width, height = 384, 450
    panel = Image.new("RGB", (width * 3, height), (25, 28, 34))
    draw = ImageDraw.Draw(panel)
    for column, label in enumerate(["Exact model input", "Human only", "Teacher assisted"]):
        pixels = rgb if column == 0 else np.rint(.55 * rgb + .45 * colors[maps[column - 1]]).astype(np.uint8)
        tile = Image.fromarray(pixels).resize((width, width), Image.Resampling.NEAREST)
        panel.paste(tile, (column * width, 28))
        draw.text((column * width + 8, 8), f"{stem} | {label}", fill="white")
        if region is not None:
            ys, xs = np.where(region)
            if len(xs):
                draw.rectangle((column * width + xs.min() * width / size, 28 + ys.min() * width / size,
                                column * width + (xs.max() + 1) * width / size,
                                28 + (ys.max() + 1) * width / size), outline=(0, 255, 90), width=2)
        if column:
            s = records[column - 1]["summary"]
            value = s.get("cloud_fraction_of_classified_sky")
            caption = f"Cloud/classified sky: {value:.1%}" if value is not None else f"Abstained: {s['reason']}"
            draw.text((column * width + 8, 416), caption, fill="white")
            c, x = s.get("cloud_area_fraction_of_roi"), s.get("contamination_area_fraction_of_roi")
            if c is not None and x is not None:
                draw.text((column * width + 8, 433), f"Cloud/ROI {c:.1%} | contamination/ROI {x:.1%}", fill="white")
    panel.save(output / "reviews" / f"{stem}.png")


def compare(inventory, review_path, bundle_paths, output, batch_size=16, threads=2):
    import onnxruntime as ort
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh paired inference directory")
    inventory_bytes, review_bytes = Path(inventory).read_bytes(), Path(review_path).read_bytes()
    input_hashes = {"inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
                    "review_sha256": hashlib.sha256(review_bytes).hexdigest()}
    source_root = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__).resolve(), source_root / "ml/coc/inference.py",
                    source_root / "ml/coc/predict.py", source_root / "ml/data_pipeline/public_common.py"]
    sources = {str(p.relative_to(source_root)): p.read_bytes() for p in source_paths}
    source_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    (snapshots / "inventory.csv").write_bytes(inventory_bytes)
    (snapshots / "roi_review.json").write_bytes(review_bytes)
    for name, data in sources.items():
        path = snapshots / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    rows = list(csv.DictReader(io.StringIO(inventory_bytes.decode("utf-8"))))
    if not rows or len({r["filename"] for r in rows}) != len(rows):
        raise ValueError("Inventory must contain unique original filenames")
    reviews_document = json.loads(review_bytes)
    if reviews_document.get("selected_before_prediction") is not True:
        raise ValueError("The region review must predate model inspection")
    reviews = {r["filename"]: r for r in reviews_document["records"]}
    if len(reviews) != len(reviews_document["records"]):
        raise ValueError("Review filenames must be unique")
    if not set(reviews).issubset({r["filename"] for r in rows}):
        raise ValueError("Review includes unknown originals")
    sessions, models = [], []
    for path in bundle_paths:
        manifest = verify_bundle(path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(Path(path) / "model.onnx"), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        sessions.append(session)
        models.append({"bundle": str(path), "model_sha256": sha256_file(Path(path) / "model.onnx"),
                       "manifest_sha256": sha256_file(Path(path) / "manifest.json"),
                       "image_size": manifest["image_size"]})
    if models[0]["image_size"] != models[1]["image_size"]:
        raise ValueError("This paired audit requires identical input resolution")
    size = models[0]["image_size"]
    for part in ["human_only", "teacher", "reviews", "inputs", "regions"]:
        (output / part).mkdir()
    records = []
    start = time.perf_counter()
    with (output / "predictions.jsonl").open("w") as stream:
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            tensors = [checked_input(row, size)[0][0] for row in batch]
            x = np.stack(tensors)
            outputs = [s.run(["logits"], {"image": x})[0] for s in sessions]
            for position, row in enumerate(batch):
                image = tensors[position]
                stem = Path(row["filename"]).stem
                region = reviewed_region(row, reviews.get(row["filename"]), size)
                diagnostics, maps = [], []
                for index, model_name in enumerate(["human_only", "teacher"]):
                    record, classes = output_record(outputs[index][position], region, image)
                    path = output / model_name / f"{stem}.png"
                    Image.fromarray(classes).save(path)
                    record["class_map_sha256"] = sha256_file(path)
                    diagnostics.append(record)
                    maps.append(classes)
                record = {"filename": row["filename"], "ancestor_id": row["ancestor_id"],
                          "original_sha256": row["original_sha256"], "preview_sha256": row["preview_sha256"],
                          "model_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                          "capture_family_proxy": row["capture_family_proxy"],
                          "conservative_split_group": row["conservative_split_group"],
                          "prior_annotation_status": row["prior_annotation_status"],
                          "prior_cloud_class_code_not_used_as_truth": row["prior_cloud_class_code"],
                          "exposure_seconds": row["exposure_seconds"], "iso": row["iso"],
                          "focal_length_mm": row["focal_length_mm"],
                          "pixel_agreement_full_frame": float((maps[0] == maps[1]).mean()),
                          "pixel_agreement_roi": float((maps[0][region] == maps[1][region]).mean()) if region is not None and region.any() else None,
                          "review": reviews.get(row["filename"]),
                          "human_only": diagnostics[0], "teacher": diagnostics[1],
                          "training_approved": False, "accuracy_eligible": False}
                if row["filename"] in reviews:
                    Image.fromarray(np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8)).save(output / "inputs" / f"{stem}.png")
                    if region is not None:
                        Image.fromarray(region.astype(np.uint8) * 255).save(output / "regions" / f"{stem}.png")
                    render_review(output, stem, image, maps, region, diagnostics)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                records.append(record)
            print(f"Paired {min(offset + batch_size, len(rows))}/{len(rows)} previews", flush=True)
    aggregates = {}
    for group, members in [("all", records)] + [(g, [r for r in records if r["capture_family_proxy"] == g])
                                               for g in sorted({r["capture_family_proxy"] for r in records})]:
        aggregates[group] = {"images": len(members),
                             "mean_pixel_agreement_full_frame": float(np.mean([r["pixel_agreement_full_frame"] for r in members])),
                             "human_only_full_frame_counts": np.sum([r["human_only"]["full_frame_predicted_class_counts"] for r in members], axis=0).tolist(),
                             "teacher_full_frame_counts": np.sum([r["teacher"]["full_frame_predicted_class_counts"] for r in members], axis=0).tolist()}
    if sha256_file(inventory) != input_hashes["inventory_sha256"] or sha256_file(review_path) != input_hashes["review_sha256"]:
        raise ValueError("Inventory or review changed during inference")
    if any(sha256_file(source_root / name) != digest for name, digest in source_hashes.items()):
        raise ValueError("Inference sources changed during the audit; rerun with frozen helpers")
    summary = {"status": "completed_paired_development_diagnostic", "images": len(records),
               "reviewed_inputs": len(reviews), "prediction_independent_roi_count": sum(r["rect"] is not None for r in reviews.values()),
               "unique_originals": len({r["original_sha256"] for r in records}),
               "unique_previews": len({r["preview_sha256"] for r in records}),
               "unique_model_rgb": len({r["model_rgb_sha256"] for r in records}),
               "models": models, **input_hashes, "sources_sha256": source_hashes,
               "predictions_sha256": sha256_file(output / "predictions.jsonl"),
               "script_sha256": sha256_file(__file__), "elapsed_seconds": time.perf_counter() - start,
               "versions": {"onnxruntime": ort.__version__, "numpy": np.__version__, "pillow": Image.__version__},
               "aggregates": aggregates, "accuracy_metrics": None, "final_test_used": False,
               "local_target_domain_qualified": False, "runtime_eligible": False,
               "limitations": ["No adjudicated pixel-level R50 ground truth; model agreement is not accuracy.",
                               "Camera-rendered embedded JPEG domain is not proven equivalent to live MJPEG or public S3 enhancement.",
                               "Wrong camera clock; all existing images conservatively share one development split family.",
                               "ROI percentages are conditional pixel areas, not hemispheric angular sky cover or atmospheric transparency."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--human-bundle", required=True)
    parser.add_argument("--teacher-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64 or not 1 <= args.threads <= 8:
        parser.error("Use batch size 1–64 and threads 1–8")
    result = compare(args.inventory, args.review, [args.human_bundle, args.teacher_bundle],
                     args.output, args.batch_size, args.threads)
    print(json.dumps({k: result[k] for k in ["status", "images", "unique_previews", "elapsed_seconds"]}))


if __name__ == "__main__":
    main()
