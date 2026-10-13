"""Finite COC job: wait for an existing training run, export, compare validation.

No scheduler, model promotion or equipment control. This exits after completion,
a detected training error, or a bounded timeout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import zipfile

from ml.coc import CLASS_NAMES, CONTRACT_VERSION
from ml.coc.checkpoint import load_checkpoint
from ml.data_pipeline.public_common import sha256_file, write_json
from ml.export.export_cloud_segmentation import export


COMPUTATION_SOURCES = (
    "ml/train/train_cloud_segmentation.py", "ml/coc/data.py", "ml/coc/__init__.py",
    "ml/models/cloud_classifier.py", "ml/models/cloud_losses.py",
    "ml/eval/cloud_segmentation_metrics.py", "ml/data_pipeline/import_public_cloud.py",
    "ml/data_pipeline/public_common.py",
)
# These exact revisions were independently reviewed. Unknown changes fail closed;
# a source hash mismatch never becomes a blanket exemption for a whole module.
AUDITED_SOURCE_DIFFERENCES = {
    ("ml/train/train_cloud_segmentation.py",
     "d02d11fea75e98755313df35af9b52cff3ced7404f37e68199993b090c192ad4",
     "2d275ba408c3b50c1afea5b3f105f68aa2376ce637f288413e333a450a856237"):
        "Provenance source capture and Pillow version recording only; optimization unchanged.",
    ("ml/coc/data.py",
     "0e3dc18667def530423c5ee7972232544f1a632a1311d99972a050d77d6098f1",
     "fa52c58236eb2facc7efdd20c83e059905d3b5bf9cc8180d6829bd5c90d01a36"):
        "Teacher identity wording clarifies lazy duplicate checks; loaders and transforms unchanged.",
    ("ml/coc/__init__.py",
     "0c72fe41a3004cb31f90ba29b761e446b532ab201344840a3a909328fc86dbb4",
     "2fc43858bf4af1196af666f37f92b4e40912e35a7b7000117693e09edeed20e6"):
        "Added export precision metadata; training class order and contract unchanged.",
}


def source_evidence(root, run):
    provenance_path = root / "supplemental_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    snapshot = root / "training_source_snapshot.zip"
    if (provenance["run_sha256"] != sha256_file(root / "run.json")
            or provenance["source_snapshot_sha256"] != sha256_file(snapshot)):
        raise ValueError("Training source snapshot is not bound to its run")
    with zipfile.ZipFile(snapshot) as archive:
        fingerprints = {name: hashlib.sha256(archive.read(name)).hexdigest()
                        for name in COMPUTATION_SOURCES}
    for name, digest in fingerprints.items():
        if (provenance["transform_sources_sha256"].get(name) != digest
                or (name in run["source_sha256"] and run["source_sha256"][name] != digest)):
            raise ValueError("Training source fingerprint changed: " + name)
    return {"snapshot_sha256": sha256_file(snapshot), "provenance_sha256": sha256_file(provenance_path),
            "computation_sha256": fingerprints,
            "versions": {"python": run["versions"]["python"], **{
                key: provenance["versions"][key] for key in ["torch", "numpy", "pillow"]}},
            "capture_timing": "before_optimization" if provenance.get("captured_before_optimization") else "during_training"}


def compare(baseline_dir, candidate_dir):
    roots = [Path(baseline_dir), Path(candidate_dir)]
    runs = [json.loads((r / "run.json").read_text()) for r in roots]
    qualities = [json.loads((r / "quality.json").read_text()) for r in roots]
    evidence = []
    for root, run, quality in zip(roots, runs, qualities):
        if (quality["contract_version"] != CONTRACT_VERSION or quality["final_test_used"] is not False
                or quality["runtime_eligible"] is not False
                or quality["checkpoint_sha256"] != sha256_file(root / "best.pt")
                or quality["run_sha256"] != sha256_file(root / "run.json")):
            raise ValueError("Comparison requires bound validation-only research artifacts")
        if (run["contract_version"] != CONTRACT_VERSION or run["class_order"] != list(CLASS_NAMES)
                or run["task"] != "cloud_segmentation" or run["runtime_eligible"] is not False
                or run["final_test_used"] is not False or quality["dataset"] != run["dataset"]
                or quality["validation"]["class_order"] != list(CLASS_NAMES)
                or quality["validation"]["ignore_index"] != 255):
            raise ValueError("Run, quality and validation ontology or dataset differ")
        _, checkpoint = load_checkpoint(root / "best.pt")
        if (checkpoint["run_sha256"] != sha256_file(root / "run.json")
                or checkpoint["variant"] != run["config"]["variant"]
                or checkpoint["size"] != run["config"]["size"]):
            raise ValueError("Checkpoint metadata differs from bound training run")
        evidence.append(source_evidence(root, run))
    if runs[0]["teacher"] is not None or runs[1]["teacher"] is None:
        raise ValueError("Compare a human-only baseline with an explicit teacher candidate")
    for field in ["variant", "size", "epochs", "batch_size", "learning_rate", "seed", "augment", "device", "threads"]:
        if runs[0]["config"][field] != runs[1]["config"][field]:
            raise ValueError("Unmatched experiment configuration: " + field)
    if (runs[0]["architecture"] != runs[1]["architecture"]
            or runs[0]["dataset"]["records_sha256"] != runs[1]["dataset"]["records_sha256"]
            or runs[0]["dataset"]["manifest_sha256"] != runs[1]["dataset"]["manifest_sha256"]
            or runs[0]["class_weights"] != runs[1]["class_weights"]):
        raise ValueError("Architecture, human data or class weights differ")
    for field in ["device", "selection", "training_images", "validation_images", "training_pixel_counts"]:
        if runs[0][field] != runs[1][field]:
            raise ValueError("Unmatched execution or training contract: " + field)
    if evidence[0]["versions"] != evidence[1]["versions"]:
        raise ValueError("Unmatched computation dependency versions")
    source_audit = []
    for name in COMPUTATION_SOURCES:
        before, after = [item["computation_sha256"][name] for item in evidence]
        if before != after:
            reason = AUDITED_SOURCE_DIFFERENCES.get((name, before, after))
            if reason is None:
                raise ValueError("Unreviewed computation source difference: " + name)
            source_audit.append({"path": name, "baseline_sha256": before, "candidate_sha256": after, "review": reason})
    metrics = [q["validation"]["overall"] for q in qualities]
    for key in ["image_count", "total_pixel_count", "labeled_pixel_count", "ignored_pixel_count"]:
        if metrics[0][key] != metrics[1][key]:
            raise ValueError("Unmatched validation support: " + key)
    if any(metrics[0]["per_class"][name]["support"] != metrics[1]["per_class"][name]["support"]
           for name in CLASS_NAMES):
        raise ValueError("Unmatched validation class support")
    if qualities[0]["validation"]["by_episode"].keys() != qualities[1]["validation"]["by_episode"].keys():
        raise ValueError("Unmatched validation episode groups")
    delta = {key: metrics[1][key] - metrics[0][key]
             for key in ["mean_iou", "balanced_accuracy", "pixel_accuracy"]}
    return {"status": "completed_validation_comparison", "contract_version": CONTRACT_VERSION,
            "baseline": {"directory": str(roots[0]), "checkpoint_sha256": qualities[0]["checkpoint_sha256"],
                         "run_sha256": sha256_file(roots[0] / "run.json"), "quality_sha256": sha256_file(roots[0] / "quality.json"),
                         "source_evidence": evidence[0], "metrics": metrics[0]},
            "candidate": {"directory": str(roots[1]), "checkpoint_sha256": qualities[1]["checkpoint_sha256"],
                          "run_sha256": sha256_file(roots[1] / "run.json"), "quality_sha256": sha256_file(roots[1] / "quality.json"),
                          "source_evidence": evidence[1], "metrics": metrics[1], "teacher": runs[1]["teacher"]},
            "matched_config": {key: runs[0]["config"][key] for key in
                ["variant", "size", "epochs", "batch_size", "learning_rate", "seed", "augment", "device", "threads"]},
            "audited_source_differences": source_audit,
            "candidate_minus_baseline": delta, "final_test_used": False, "runtime_eligible": False,
            "limitations": ["Descriptive source-domain validation; no confidence intervals or local-camera qualification.",
                            "Human partial labels and uncalibrated teacher supervision remain distinct."]}


def positive_hours(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("max-hours must be finite and positive")
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--parity-image", required=True)
    p.add_argument("--training-log", type=Path)
    p.add_argument("--max-hours", type=positive_hours, default=1)
    a = p.parse_args()
    status_path = a.candidate.parent / (a.candidate.name + "-job.json")
    state = {"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(), "candidate": str(a.candidate),
             "runtime_eligible": False}
    def status(phase, **extra):
        state.update(status=phase, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(status_path, state)
        print(json.dumps(state), flush=True)
    started = time.monotonic()
    try:
        status("waiting_for_training")
        while not (a.candidate / "quality.json").exists():
            if a.training_log and a.training_log.exists() and "Traceback (most recent call last)" in a.training_log.read_text():
                raise RuntimeError("Training failed; inspect " + str(a.training_log))
            if time.monotonic() - started > a.max_hours * 3600:
                raise TimeoutError("Bounded training wait expired; preserved all run files")
            time.sleep(10)
        if a.report.exists():
            raise ValueError("Comparison report already exists; refusing to overwrite evidence")
        status("exporting_and_comparing")
        result = compare(a.baseline, a.candidate)
        export(a.candidate / "best.pt", a.bundle, a.parity_image)
        result["bundle_manifest_sha256"] = sha256_file(a.bundle / "manifest.json")
        write_json(a.report, result)
        status("complete", report=str(a.report), bundle=str(a.bundle), candidate_minus_baseline=result["candidate_minus_baseline"])
    except Exception as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
