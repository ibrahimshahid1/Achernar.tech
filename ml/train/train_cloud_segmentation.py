"""Train COC segmentation on frozen human regions; optional explicit teacher ablation.

python -m ml.train.train_cloud_segmentation --output ml/checkpoints/coc/human-tiny
The untouched final test is never opened by this entry point.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from importlib.metadata import version
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.coc import CLASS_NAMES, CONTRACT_VERSION
from ml.coc.data import HumanCloudDataset, TeacherCloudDataset, inspect_human_dataset
from ml.coc.provenance import capture_sources
from ml.data_pipeline.public_common import sha256_file, write_json
from ml.eval.cloud_segmentation_metrics import CloudSegmentationMetrics
from ml.models.cloud_classifier import CloudSegmenter, variant_config
from ml.models.cloud_losses import masked_cross_entropy, masked_teacher_kl


def choose_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")
    return requested


def evaluate(model, loader, device):
    metrics = CloudSegmentationMetrics()
    loss_sum, valid_count = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            image, target = batch["image"].to(device), batch["mask"].to(device)
            logits = model(image)
            count = int((target != 255).sum().item())
            loss_sum += float(masked_cross_entropy(logits, target).item()) * count
            valid_count += count
            metrics.update(logits.argmax(1).cpu().numpy(), target.cpu().numpy(),
                           split_groups=batch["split_group"], camera_eras=batch["camera_era"])
    result = metrics.compute()
    result["unweighted_partial_cross_entropy"] = loss_sum / valid_count if valid_count else None
    return result


def constant_baseline(loader, training_counts):
    """Train-majority reference at the same resized annotated pixels."""
    metrics = CloudSegmentationMetrics()
    label = int(np.argmax(training_counts))
    for batch in loader:
        target = batch["mask"].numpy()
        metrics.update(np.full_like(target, label), target,
                       split_groups=batch["split_group"], camera_eras=batch["camera_era"])
    return {"reference": "train_pixel_majority", "predicted_class": CLASS_NAMES[label], "metrics": metrics.compute()}


def checkpoint_write(path, value):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(config):
    output = Path(config["output"])
    # Prevent accidental mixing/overwriting of evidence from different runs.
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh output directory; existing runs are immutable")
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("Invalid optimizer configuration")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    device = choose_device(config["device"])
    identity = inspect_human_dataset(config["human_root"])
    train_data = HumanCloudDataset(config["human_root"], "train", config["size"],
                                   augment=config["augment"], cache=config["cache"])
    validation = HumanCloudDataset(config["human_root"], "validation", config["size"], cache=config["cache"])
    if not len(train_data) or not len(validation):
        raise ValueError("Nonempty train and validation partitions required")
    counts = train_data.training_pixel_counts()
    if np.any(counts == 0):
        raise ValueError("Each class needs human training support")
    # Inverse-square-root weights are train-only, normalized to mean 1.
    weights = 1 / np.sqrt(counts.astype(np.float64))
    weights /= weights.mean()
    loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(validation, batch_size=config["batch_size"], num_workers=0)
    teacher, teacher_loader = None, None
    if config["teacher_root"]:
        if config["teacher_weight"] <= 0:
            raise ValueError("Teacher ablation needs positive explicit teacher weight")
        teacher = TeacherCloudDataset(config["teacher_root"], config["human_root"], config["size"],
                                       config["teacher_limit"], seed)
        teacher_loader = DataLoader(teacher, batch_size=config["batch_size"], shuffle=True,
                                    generator=torch.Generator().manual_seed(seed + 1), num_workers=0)
        write_json(output / "teacher_selected_ids.json", teacher.ids)
    elif config["teacher_weight"] != 0:
        raise ValueError("Teacher weight requires an explicit teacher root")
    model = CloudSegmenter(variant=config["variant"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    baseline = constant_baseline(val_loader, counts)
    repo_root = Path(__file__).resolve().parents[2]
    source_files = [repo_root / path for path in ["ml/train/train_cloud_segmentation.py", "ml/coc/data.py",
        "ml/coc/__init__.py", "ml/coc/provenance.py", "ml/models/cloud_classifier.py", "ml/models/cloud_losses.py",
        "ml/eval/cloud_segmentation_metrics.py", "ml/data_pipeline/import_public_cloud.py",
        "ml/data_pipeline/public_common.py", "ml/data_pipeline/lenghu_bulk_dataset.py", "ml/requirements-coc.lock.txt"]]
    run = {"contract_version": CONTRACT_VERSION, "task": "cloud_segmentation", "class_order": list(CLASS_NAMES),
           "created_at": datetime.now(timezone.utc).isoformat(), "config": config, "device": device,
           "versions": {"python": os.sys.version.split()[0], "torch": str(torch.__version__),
                        "numpy": np.__version__, "pillow": version("pillow")},
           "architecture": variant_config(config["variant"]),
           "parameter_count": sum(p.numel() for p in model.parameters()), "dataset": identity,
           "training_images": len(train_data), "validation_images": len(validation), "training_pixel_counts": counts.tolist(),
           "class_weights": weights.tolist(), "teacher": teacher.identity if teacher else None,
           "source_sha256": {str(p.relative_to(repo_root)): sha256_file(p) for p in source_files},
           "baseline": baseline, "selection": "minimum unweighted validation partial cross entropy",
           "runtime_eligible": False, "promotion_status": "research", "final_test_used": False,
           "limitations": ["Partial-region metrics; unannotated pixels ignored.",
                           "No calibrated cloud presence, transparency, rain or control authority.",
                           "Temporal episodes share source cameras; local R50 transfer unvalidated."]}
    write_json(output / "run.json", run)
    capture_sources(output, source_files, repo_root)
    best_loss, best_epoch, history = float("inf"), 0, []
    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        human_loss, kd_loss, steps, teacher_pixels = 0.0, 0.0, 0, 0
        teacher_iterator = iter(teacher_loader) if teacher_loader is not None else None
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"].to(device))
            loss_human = masked_cross_entropy(logits, batch["mask"].to(device), class_weights=class_weights)
            loss = loss_human
            if teacher_iterator is not None:
                try:
                    tb = next(teacher_iterator)
                except StopIteration:
                    teacher_iterator = iter(teacher_loader)
                    tb = next(teacher_iterator)
                teacher_logits = model(tb["image"].to(device))
                loss_teacher, selected = masked_teacher_kl(teacher_logits, tb["teacher_logits"].to(device),
                    tb["content_mask"].to(device), temperature=config["teacher_temperature"],
                    confidence_threshold=config["teacher_confidence"])
                loss = loss + config["teacher_weight"] * loss_teacher
                kd_loss += float(loss_teacher.detach().item())
                teacher_pixels += selected
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; stop rather than saving invalid weights")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            human_loss += float(loss_human.detach().item())
            steps += 1
        report = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_weighted_ce_batch_mean": human_loss / steps,
                  "teacher_kl_batch_mean": kd_loss / steps if teacher is not None else None,
                  "teacher_selected_pixels": teacher_pixels, "validation": report,
                  "elapsed_seconds": time.perf_counter() - start}
        history.append(record)
        state = {"contract_version": CONTRACT_VERSION, "variant": config["variant"], "size": config["size"],
                 "class_order": list(CLASS_NAMES), "epoch": epoch, "state_dict": model.cpu().state_dict(),
                 "run_sha256": sha256_file(output / "run.json")}
        checkpoint_write(output / "last.pt", state)
        model.to(device)
        score = report["unweighted_partial_cross_entropy"]
        if score is not None and score < best_loss:
            best_loss, best_epoch = score, epoch
            checkpoint_write(output / "best.pt", state)
        write_json(output / "history.json", history)
        print(json.dumps({"epoch": epoch, "train_ce": record["train_weighted_ce_batch_mean"],
                          "validation_ce": score, "elapsed_seconds": round(record["elapsed_seconds"], 1)}), flush=True)
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["state_dict"])
    model.to(device)
    quality = {"contract_version": CONTRACT_VERSION, "status": "research_validation_only", "best_epoch": best_epoch,
               "checkpoint_sha256": sha256_file(output / "best.pt"), "run_sha256": sha256_file(output / "run.json"),
               "dataset": identity, "validation": evaluate(model, val_loader, device), "baseline": baseline,
               "final_test_used": False, "runtime_eligible": False, "probability_calibrated": False,
               "local_target_domain_qualified": False, "training_seconds": time.perf_counter() - start}
    write_json(output / "quality.json", quality)
    return quality


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--human-root", default="ml/data/public/lenghu_sky8_human")
    p.add_argument("--output", required=True)
    p.add_argument("--variant", choices=["tiny", "small"], default="tiny")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--teacher-root")
    p.add_argument("--teacher-limit", type=int, default=4096)
    p.add_argument("--teacher-weight", type=float, default=0.0)
    p.add_argument("--teacher-temperature", type=float, default=2.0)
    p.add_argument("--teacher-confidence", type=float, default=0.8)
    args = vars(p.parse_args())
    quality = train(args)
    print(json.dumps({"output": args["output"], "best_epoch": quality["best_epoch"], "status": quality["status"]}), flush=True)


if __name__ == "__main__":
    main()
