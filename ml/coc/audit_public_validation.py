"""Audit two frozen COC checkpoints on public validation only; never train/test.

From the repository root::

    python -m ml.coc.audit_public_validation --self-test
    python -m ml.coc.audit_public_validation --output docs/coc-error-audit-2026-10-13/public

Confusion, NLL, Brier score and confidence bins are independently implemented
with NumPy; the existing metric/evaluation helpers are not called. Model/source
inputs are read only. The output must be a fresh directory. Input-first panels
and blank review forms precede separate model-identity-blinded overlay panels;
generating them is not human adjudication or an independent label collection.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import torch

from ml.coc import CLASS_NAMES, CONTRACT_VERSION
from ml.coc.checkpoint import load_checkpoint


COLORS = np.array([[35, 180, 235], [255, 167, 32], [205, 70, 180]], dtype=np.uint8)
MODEL_NAMES = ("human", "teacher")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def metrics(cm):
    """Rows=true, columns=predicted; absent IoU union/support returns None."""
    cm = np.asarray(cm, dtype=np.int64)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    result = {"confusion_matrix": cm.tolist(), "labeled_pixel_count": int(cm.sum()), "per_class": {}}
    for i, name in enumerate(CLASS_NAMES):
        result["per_class"][name] = {
            "support": int(support[i]), "predicted_support": int(predicted[i]),
            "precision": ratio(cm[i, i], predicted[i]), "recall": ratio(cm[i, i], support[i]),
            "iou": ratio(cm[i, i], support[i] + predicted[i] - cm[i, i]),
        }
    ious = [v["iou"] for v in result["per_class"].values() if v["iou"] is not None]
    recalls = [v["recall"] for v in result["per_class"].values() if v["recall"] is not None]
    result.update(mean_iou=float(np.mean(ious)) if ious else None,
                  balanced_accuracy=float(np.mean(recalls)) if recalls else None,
                  pixel_accuracy=ratio(np.trace(cm), cm.sum()))
    return result


def pixel_diagnostics(logits, target):
    """Descriptive confidence statistics on human-labeled pixels only."""
    target = np.asarray(target)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (3, *target.shape) or not np.isfinite(logits).all():
        raise ValueError("Expected finite three-class logits matching the target")
    if not np.isin(target, [0, 1, 2, 255]).all():
        raise ValueError("Unexpected target class")
    probabilities = np.exp(logits - logits.max(axis=0, keepdims=True))
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    predicted = logits.argmax(axis=0)
    valid = target != 255
    truth, pred = target[valid].astype(np.int64), predicted[valid]
    probability = probabilities[:, valid].T
    confidence = probability.max(axis=1)
    correct = pred == truth
    cm = np.bincount(3 * truth + pred, minlength=9).reshape(3, 3)
    high = confidence >= 0.8  # Diagnostic bin, not a calibrated operating gate.
    high_cm = np.bincount(3 * truth[high] + pred[high], minlength=9).reshape(3, 3)
    bins = []
    for i in range(10):
        selected = (confidence >= i / 10) & ((confidence < (i + 1) / 10) if i < 9 else confidence <= 1)
        bins.append({"lower": i / 10, "upper": (i + 1) / 10, "count": int(selected.sum()),
                     "correct": int(correct[selected].sum()),
                     "confidence_sum": float(confidence[selected].sum())})
    # Log-sum-exp NLL avoids clipping a very small probability.
    selected_logits = logits[:, valid].T
    maximum = selected_logits.max(axis=1)
    nll = maximum + np.log(np.exp(selected_logits - maximum[:, None]).sum(axis=1))
    nll -= selected_logits[np.arange(len(truth)), truth]
    one_hot = np.eye(3)[truth]
    errors = ~correct
    values = metrics(cm)
    values.update(total_pixel_count=int(target.size), ignored_pixel_count=int((~valid).sum()),
                  annotated_pixel_fraction=ratio(valid.sum(), target.size),
                  nll_sum=float(nll.sum()), brier_sum=float(((probability - one_hot) ** 2).sum()),
                  wrong_pixel_count=int(errors.sum()),
                  high_confidence_confusion_matrix=high_cm.tolist(),
                  high_confidence_wrong_pixel_count=int((errors & high).sum()),
                  confidence_sum_correct=float(confidence[correct].sum()),
                  confidence_sum_wrong=float(confidence[errors].sum()), confidence_bins=bins)
    return values, predicted.astype(np.uint8), confidence, valid


def aggregate(rows):
    result = metrics(np.sum([r["confusion_matrix"] for r in rows], axis=0))
    for key in ("total_pixel_count", "ignored_pixel_count", "nll_sum", "brier_sum", "wrong_pixel_count",
                "high_confidence_wrong_pixel_count", "confidence_sum_correct", "confidence_sum_wrong"):
        result[key] = sum(r[key] for r in rows)
    count, wrong = result["labeled_pixel_count"], result["wrong_pixel_count"]
    result.update(image_count=len(rows), annotated_pixel_fraction=ratio(count, result["total_pixel_count"]),
                  unweighted_partial_cross_entropy=ratio(result.pop("nll_sum"), count),
                  multiclass_brier_score=ratio(result.pop("brier_sum"), count),
                  mean_confidence_correct=ratio(result.pop("confidence_sum_correct"), count - wrong),
                  mean_confidence_wrong=ratio(result.pop("confidence_sum_wrong"), wrong),
                  fraction_of_errors_with_confidence_ge_0_8=ratio(result["high_confidence_wrong_pixel_count"], wrong))
    result["high_confidence_confusion_matrix"] = np.sum(
        [r["high_confidence_confusion_matrix"] for r in rows], axis=0).tolist()
    bins = []
    for i in range(10):
        count = sum(r["confidence_bins"][i]["count"] for r in rows)
        correct = sum(r["confidence_bins"][i]["correct"] for r in rows)
        conf = sum(r["confidence_bins"][i]["confidence_sum"] for r in rows)
        bins.append({"lower": i / 10, "upper": (i + 1) / 10, "count": count,
                     "accuracy": ratio(correct, count), "mean_confidence": ratio(conf, count)})
    result["confidence_bins"] = bins
    result["descriptive_top_label_ece_10_equal_width_bins"] = ratio(sum(
        b["count"] * abs(b["accuracy"] - b["mean_confidence"]) for b in bins if b["count"]),
        result["labeled_pixel_count"])
    return result


def verify_reference(calculated, reference):
    if calculated["confusion_matrix"] != reference["confusion_matrix"]:
        raise ValueError("Recomputed confusion differs from the frozen quality report")
    for key in ("mean_iou", "balanced_accuracy", "pixel_accuracy", "labeled_pixel_count"):
        if not np.isclose(calculated[key], reference[key], atol=1e-12, rtol=0):
            raise ValueError(f"Independent metric differs: {key}")


def safe_asset(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Dataset asset escapes root")
    return path


def load_validation(root, size, rows):
    images, masks = [], []
    for row in rows:
        if row["split"] != "validation" or row["label_type"] != "human_partial_regions":
            raise ValueError("This audit opens validation human assets only")
        for key in ("image", "mask"):
            if sha256(safe_asset(root, row[key])) != row[key + "_sha256"]:
                raise ValueError(f"Changed validation {key}: {row['sample_id']}")
        with Image.open(safe_asset(root, row["image"])) as source:
            rgb = source.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            images.append(np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        with Image.open(safe_asset(root, row["mask"])) as source:
            masks.append(np.asarray(source.resize((size, size), Image.Resampling.NEAREST), dtype=np.uint8))
    return np.stack(images), np.stack(masks)


def representative_selection(rows, results, limit=12):
    """Predeclared error types, support >=100, no more than two per episode."""
    chosen, used, groups = [], set(), defaultdict(int)
    definitions = [("teacher_contamination_to_sky", 2, 0), ("teacher_contamination_to_cloud", 2, 1),
                   ("teacher_cloud_to_sky", 1, 0), ("teacher_sky_to_cloud", 0, 1)]
    rankings = []
    for name, true, pred in definitions:
        ranked = []
        for i, row in enumerate(rows):
            cm = np.asarray(results["teacher"][i]["confusion_matrix"])
            if cm[true].sum() >= 100 and cm[true, pred] > 0:
                ranked.append((float(cm[true, pred] / cm[true].sum()), row["sample_id"], i))
        rankings.append((name, sorted(ranked, key=lambda x: (-x[0], x[1]))))
    # Rate extremes can come from very small masks. Also cover high absolute
    # burdens, while retaining a cap on repeated images from one episode.
    for name, true, pred in (("teacher_contamination_to_sky_pixel_burden", 2, 0),
                             ("teacher_cloud_to_sky_pixel_burden", 1, 0)):
        ranked = [(results["teacher"][i]["confusion_matrix"][true][pred], row["sample_id"], i)
                  for i, row in enumerate(rows)]
        rankings.append((name, sorted(ranked, key=lambda x: (-x[0], x[1]))))
    for name, direction in (("largest_teacher_regression", -1), ("largest_teacher_improvement", 1)):
        ranked = [(direction * (results["teacher"][i]["mean_iou"] - results["human"][i]["mean_iou"]),
                   row["sample_id"], i) for i, row in enumerate(rows)]
        rankings.append((name, sorted(ranked, key=lambda x: (-x[0], x[1]))))
    ranked = [(results["teacher"][i]["high_confidence_wrong_pixel_count"] /
               results["teacher"][i]["labeled_pixel_count"], row["sample_id"], i) for i, row in enumerate(rows)]
    rankings.append(("teacher_high_confidence_error_fraction", sorted(ranked, key=lambda x: (-x[0], x[1]))))
    for name, ranking in rankings:
        for score, _, i in ranking:
            group = rows[i]["split_group"]
            if i not in used and groups[group] < 2 and score > 0:
                chosen.append({"index": i, "selection_reason": name, "selection_score": score})
                used.add(i)
                groups[group] += 1
                break
    # Deterministic, input-independent coverage sample complements mined errors.
    ranking = sorted(range(len(rows)), key=lambda i: hashlib.sha256(
        ("public-audit-complement-v1:" + rows[i]["sample_id"]).encode()).digest())
    for i in ranking:
        if len(chosen) >= limit:
            break
        if i not in used and groups[rows[i]["split_group"]] < 2:
            chosen.append({"index": i, "selection_reason": "hash_sample_complement", "selection_score": None})
            used.add(i)
            groups[rows[i]["split_group"]] += 1
    return sorted(chosen, key=lambda item: hashlib.sha256(
        ("blind-order-v1:" + rows[item["index"]]["sample_id"]).encode()).digest())


def font(size=14):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_panels(output, root, rows, selected, images, masks, predictions):
    """Separate stage1 blind inputs, stage2 overlays and an explicit unblind key."""
    width, cell, header = 224, 224, 46
    key, form = [], []
    for page, offset in enumerate(range(0, len(selected), 4), 1):
        subset = selected[offset:offset + 4]
        blind = Image.new("RGB", (4 * width, cell + header), "#151a21")
        reveal = Image.new("RGB", (6 * width, len(subset) * (cell + header)), "#151a21")
        for position, item in enumerate(subset):
            i = item["index"]
            review_id = f"V{offset + position + 1:02}"
            with Image.open(safe_asset(root, rows[i]["image"])) as source:
                native = source.convert("RGB")
                native.save(output / f"stage1-native-input-{review_id}.png")
            rgb = np.rint(images[i].transpose(1, 2, 0) * 255).astype(np.uint8)
            target = masks[i]
            labeled = target != 255
            base = rgb.astype(np.float32)
            partial = base.copy()
            partial[labeled] = .55 * base[labeled] + .45 * COLORS[target[labeled]]
            panels = [rgb, partial.astype(np.uint8)]
            for name in ("teacher", "human"):  # Hidden until separate unblinding key.
                panels.append((.55 * base + .45 * COLORS[predictions[name][i]]).astype(np.uint8))
            for name in ("teacher", "human"):
                error = labeled & (predictions[name][i] != target)
                overlay = (base * .38).astype(np.uint8)
                overlay[error] = (base[error] * .3 + np.array([255, 35, 45]) * .7).astype(np.uint8)
                overlay[labeled & ~error] = (base[labeled & ~error] * .6).astype(np.uint8)
                panels.append(overlay)
            blind.paste(native.resize((cell, cell), Image.Resampling.BILINEAR), (position * width, header))
            ImageDraw.Draw(blind).text((position * width + 8, 12), f"{review_id}: input only", fill="white", font=font())
            labels = ["input", "partial human labels", "anonymous model A", "anonymous model B",
                      "A labeled errors (red)", "B labeled errors (red)"]
            for column, (panel, label) in enumerate(zip(panels, labels)):
                x, y = column * width, position * (cell + header)
                reveal.paste(Image.fromarray(panel).resize((cell, cell), Image.Resampling.NEAREST), (x, y + header))
                draw = ImageDraw.Draw(reveal)
                draw.text((x + 8, y + 8), f"{review_id}: {label}", fill="white", font=font(13))
                if column == 1:
                    draw.text((x + 8, y + 25), "cyan sky / orange cloud / purple contam", fill="#c4cbd4", font=font(10))
            key.append({"review_id": review_id, "sample_id": rows[i]["sample_id"],
                        "split_group": rows[i]["split_group"], "camera_era": rows[i]["camera_era"],
                        **{k: v for k, v in item.items() if k != "index"}})
            form.append({"review_id": review_id, "input_first_observation": "", "visible_content_ambiguity": "",
                         "proposed_independent_roi_notes": "", "then_partial_label_review": "",
                         "then_model_overlay_review": "", "human_adjudication_status": "not_performed"})
        blind.save(output / f"stage1-input-only-{page:02}.png")
        reveal.save(output / f"stage2-anonymous-overlays-{page:02}.png")
    dump(output / "stage3-unblinding-key.json", {"model_A": "teacher", "model_B": "human", "selection": key,
         "blinding_scope": "Model identity and mining reason hidden in stages 1/2; audit author knows algorithm. Not independent human annotation."})
    write_csv(output / "review-template.csv", form)
    return key


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    # Ignores must not contaminate confidence or confusion. Empty classes must
    # remain undefined, including perfect one-class images with absent classes.
    logits = np.array([[[5, 0, 0, 99]], [[0, 5, 7, -99]], [[0, 0, 0, 0]]], dtype=float)
    target = np.array([[0, 1, 2, 255]], dtype=np.uint8)
    value, pred, _, valid = pixel_diagnostics(logits, target)
    assert value["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
    assert value["wrong_pixel_count"] == value["high_confidence_wrong_pixel_count"] == 1
    assert value["labeled_pixel_count"] == int(valid.sum()) == 3
    assert pred.shape == target.shape
    combined = aggregate([value, value])
    assert combined["labeled_pixel_count"] == 6 and combined["image_count"] == 2
    assert np.isclose(combined["pixel_accuracy"], 2 / 3)
    assert np.isclose(combined["balanced_accuracy"], 2 / 3)
    assert np.isclose(combined["mean_iou"], 1 / 2)
    absent = metrics([[2, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert absent["per_class"]["cloud"]["iou"] is None and absent["mean_iou"] == 1
    assert sum(b["count"] for b in combined["confidence_bins"]) == 6
    ignored, *_ = pixel_diagnostics(logits, np.full_like(target, 255))
    assert ignored["labeled_pixel_count"] == 0 and ignored["pixel_accuracy"] is None
    json.dumps(aggregate([ignored]), allow_nan=False)
    print("Public-audit arithmetic self-test PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-root", type=Path, default=Path("ml/data/public/lenghu_sky8_human"))
    parser.add_argument("--human-checkpoint", type=Path, default=Path("ml/checkpoints/coc/human-tiny-v1/best.pt"))
    parser.add_argument("--teacher-checkpoint", type=Path, default=Path("ml/checkpoints/coc/human-teacher-tiny-v1/best.pt"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        if args.output is None:
            return
    if args.output is None or args.output.exists():
        parser.error("--output must name a fresh directory")
    if not 1 <= args.threads <= 2 or args.batch_size < 1:
        parser.error("Use one or two CPU threads and a positive batch size")
    root, output = args.human_root.resolve(), args.output.resolve()
    if output.is_relative_to(root) or any(output.is_relative_to(p.parent.resolve()) for p in
                                         (args.human_checkpoint, args.teacher_checkpoint)):
        parser.error("Output must not be inside a dataset or checkpoint directory")
    records_hash, manifest_hash = sha256(root / "records.jsonl"), sha256(root / "manifest.json")
    # The full split ledger is metadata only. Only validation image/mask assets
    # are opened, decoded, hashed or passed through a model.
    all_rows = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
    rows = [row for row in all_rows if row["split"] == "validation"]
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Missing/duplicate validation records")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    runs, qualities, models, provenance = {}, {}, {}, {}
    size = None
    for name, path in zip(MODEL_NAMES, (args.human_checkpoint, args.teacher_checkpoint)):
        model, state = load_checkpoint(path)
        run_path, quality_path = path.parent / "run.json", path.parent / "quality.json"
        run, quality = json.loads(run_path.read_text()), json.loads(quality_path.read_text())
        if sha256(run_path) != state["run_sha256"] or quality["run_sha256"] != state["run_sha256"]:
            raise ValueError("Checkpoint/run/quality binding differs")
        if quality["checkpoint_sha256"] != sha256(path) or run["final_test_used"] or quality["final_test_used"]:
            raise ValueError("Changed checkpoint or previously final-tested candidate")
        if any(run["dataset"][k] != v for k, v in (("records_sha256", records_hash), ("manifest_sha256", manifest_hash))):
            raise ValueError("Dataset/split differs from frozen training run")
        if size is not None and state["size"] != size:
            raise ValueError("Paired audit requires matching preprocessing resolution")
        size = state["size"]
        runs[name], qualities[name], models[name] = run, quality, model
        provenance[name] = {"checkpoint_path": str(path.resolve()), "checkpoint_sha256": sha256(path),
                            "run_sha256": sha256(run_path), "quality_sha256": sha256(quality_path),
                            "best_epoch": quality["best_epoch"], "seed": run["config"]["seed"],
                            "training_epochs": run["config"]["epochs"], "parameter_count": run["parameter_count"]}
    images, masks = load_validation(root, size, rows)
    results, predictions, transitions = {}, {}, np.zeros(4, dtype=np.int64)
    for name in MODEL_NAMES:
        logits = []
        with torch.inference_mode():
            for start in range(0, len(rows), args.batch_size):
                value = models[name](torch.from_numpy(images[start:start + args.batch_size]))
                logits.extend(value.numpy())
        values = [pixel_diagnostics(value, mask) for value, mask in zip(logits, masks)]
        results[name] = [v[0] for v in values]
        predictions[name] = np.stack([v[1] for v in values])
        print(f"Recomputed {name}: {len(rows)} validation images", flush=True)
    overall, grouped = {}, {"by_episode": {}, "by_camera_era": {}}
    for name in MODEL_NAMES:
        overall[name] = aggregate(results[name])
        reference = qualities[name]["validation"]
        verify_reference(overall[name], reference["overall"])
        if not np.isclose(overall[name]["unweighted_partial_cross_entropy"], reference["unweighted_partial_cross_entropy"], atol=1e-6, rtol=0):
            raise ValueError("Independent NLL differs from frozen validation cross entropy")
        for field, key in (("by_episode", "split_group"), ("by_camera_era", "camera_era")):
            for group in sorted({r[key] for r in rows}):
                selected = [results[name][i] for i, row in enumerate(rows) if row[key] == group]
                report = aggregate(selected)
                verify_reference(report, reference[field][group])
                grouped[field].setdefault(group, {})[name] = report
    image_rows, detailed_rows = [], []
    for i, row in enumerate(rows):
        truth = masks[i]
        valid = truth != 255
        human_correct = predictions["human"][i][valid] == truth[valid]
        teacher_correct = predictions["teacher"][i][valid] == truth[valid]
        counts = [int((human_correct & teacher_correct).sum()), int((~human_correct & teacher_correct).sum()),
                  int((human_correct & ~teacher_correct).sum()), int((~human_correct & ~teacher_correct).sum())]
        transitions += counts
        record = {"sample_id": row["sample_id"], "split_group": row["split_group"], "camera_era": row["camera_era"],
                  "image_sha256": row["image_sha256"], "mask_sha256": row["mask_sha256"],
                  "both_correct_pixels": counts[0], "repaired_by_teacher_pixels": counts[1],
                  "new_teacher_error_pixels": counts[2], "both_wrong_pixels": counts[3]}
        flat = dict(record)
        for name in MODEL_NAMES:
            r = results[name][i]
            record[name] = r
            for key in ("labeled_pixel_count", "annotated_pixel_fraction", "mean_iou", "pixel_accuracy",
                        "balanced_accuracy", "wrong_pixel_count", "high_confidence_wrong_pixel_count"):
                flat[f"{name}_{key}"] = r[key]
            for c in CLASS_NAMES:
                for metric in ("support", "precision", "recall", "iou"):
                    flat[f"{name}_{c}_{metric}"] = r["per_class"][c][metric]
            for true in range(3):
                for pred in range(3):
                    flat[f"{name}_{CLASS_NAMES[true]}_to_{CLASS_NAMES[pred]}"] = r["confusion_matrix"][true][pred]
        flat["teacher_minus_human_mean_iou"] = record["teacher"]["mean_iou"] - record["human"]["mean_iou"]
        image_rows.append(flat)
        detailed_rows.append(record)
    episode_rows = []
    for group, values in grouped["by_episode"].items():
        entry = {"split_group": group, "image_count": values["human"]["image_count"],
                 "labeled_pixel_count": values["human"]["labeled_pixel_count"]}
        for name in MODEL_NAMES:
            entry[name + "_mean_iou"] = values[name]["mean_iou"]
            for c in CLASS_NAMES:
                entry[f"{name}_{c}_recall"] = values[name]["per_class"][c]["recall"]
        entry["teacher_minus_human_mean_iou"] = entry["teacher_mean_iou"] - entry["human_mean_iou"]
        episode_rows.append(entry)
    selection = representative_selection(rows, results)
    output.mkdir(parents=True)
    write_csv(output / "per-image.csv", image_rows)
    write_csv(output / "per-episode.csv", sorted(episode_rows, key=lambda r: r["teacher_minus_human_mean_iou"]))
    with (output / "per-image.jsonl").open("w") as stream:
        for row in detailed_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    dump(output / "grouped-metrics.json", grouped)
    review = make_panels(output, root, rows, selection, images, masks, predictions)
    summary = {"schema_version": "coc_public_validation_error_audit_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
               "contract_version": CONTRACT_VERSION, "class_order": list(CLASS_NAMES), "ignore_index": 255,
               "confusion_orientation": "rows=true,columns=predicted", "split": "validation", "final_test_assets_opened": 0,
               "final_test_used": False, "training_performed": False, "runtime_eligible": False,
               "human_adjudication_performed": False, "independent_metric_verification": "PASS against both frozen quality reports, overall and every episode/camera-era confusion",
               "validation_images": len(rows), "validation_episode_groups": len(grouped["by_episode"]),
               "input_size": size, "preprocessing": "RGB /255; PIL bilinear RGB resize; nearest mask resize; no augmentation",
               "dataset_records_sha256": records_hash, "dataset_manifest_sha256": manifest_hash, "models": provenance,
               "overall": overall, "paired_pixel_transitions": dict(zip(
                   ("both_correct", "repaired_by_teacher", "new_teacher_error", "both_wrong"), transitions.tolist())),
               "episode_improved": sum(e["teacher_minus_human_mean_iou"] > 0 for e in episode_rows),
               "episode_worsened": sum(e["teacher_minus_human_mean_iou"] < 0 for e in episode_rows),
               "image_improved": sum(e["teacher_minus_human_mean_iou"] > 0 for e in image_rows),
               "image_worsened": sum(e["teacher_minus_human_mean_iou"] < 0 for e in image_rows),
               "selected_review_images": len(review), "threads": args.threads, "batch_size": args.batch_size,
               "versions": {"Python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "Pillow": PIL.__version__},
               "script_sha256": sha256(__file__), "source_sha256": {p: sha256(p) for p in
                   ("ml/coc/checkpoint.py", "ml/models/cloud_classifier.py", "ml/coc/__init__.py")},
               "limitations": ["Only pre-existing human partial regions are scored; ignored pixels are never counted as correct or clear sky",
                   "Validation already selected these checkpoints; all confidence and error statistics are descriptive development diagnostics",
                   "Pixels/images within episodes are correlated; 488825 labeled pixels are not independent trials",
                   "The 0.8 confidence diagnostic is uncalibrated, not a certified rejection threshold",
                   "No optical transparency, five-condition class, all-sky solid-angle coverage, weather risk or R50 transfer is measured",
                   "Error-mined review examples are not prevalence estimates and do not create approved human labels",
                   "No final-test image/mask assets are opened; the split ledger and its population counts are metadata only"]}
    dump(output / "summary.json", summary)
    print(json.dumps({"output": str(output), "verified": "PASS", "per_model_mean_iou": {n: overall[n]["mean_iou"] for n in MODEL_NAMES},
                      "transitions": summary["paired_pixel_transitions"]}), flush=True)


if __name__ == "__main__":
    main()
