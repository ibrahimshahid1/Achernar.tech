"""Reject misleading COC comparisons before export or evidence publication.

These synthetic sidecars exercise provenance and matching, not model accuracy.
"""
from __future__ import annotations

import json
import hashlib
import zipfile

import pytest

torch = pytest.importorskip("torch")

from ml.coc import CLASS_NAMES, CONTRACT_VERSION
from ml.data_pipeline.public_common import sha256_file, write_json
from ml.models.cloud_classifier import CloudSegmenter
from ml.pipelines.finalize_coc_experiment import COMPUTATION_SOURCES, compare, positive_hours


@pytest.fixture
def experiments(tmp_path):
    roots = [tmp_path / "human", tmp_path / "teacher"]
    for index, root in enumerate(roots):
        root.mkdir()
        fingerprints = {}
        with zipfile.ZipFile(root / "training_source_snapshot.zip", "w") as archive:
            for name in COMPUTATION_SOURCES:
                payload = b"synthetic saved source fixture: " + name.encode()
                archive.writestr(name, payload)
                fingerprints[name] = hashlib.sha256(payload).hexdigest()
        run = {
            "contract_version": CONTRACT_VERSION, "class_order": list(CLASS_NAMES), "task": "cloud_segmentation",
            "runtime_eligible": False, "final_test_used": False, "device": "cpu", "selection": "validation CE",
            "training_images": 3, "validation_images": 3, "training_pixel_counts": [10, 10, 10],
            "versions": {"python": "fixture"}, "source_sha256": fingerprints,
            "config": {"variant": "tiny", "size": 128, "epochs": 12, "batch_size": 16,
                       "learning_rate": 0.001, "seed": 17, "augment": True, "device": "cpu", "threads": 4},
            "architecture": {"name": "fixture"}, "class_weights": [1, 1, 1],
            "dataset": {"records_sha256": "fixed-records", "manifest_sha256": "fixed-manifest"},
            "teacher": {"selected_count": 4096} if index else None,
        }
        write_json(root / "run.json", run)
        torch.save({"contract_version": CONTRACT_VERSION, "class_order": list(CLASS_NAMES), "size": 128,
                    "variant": "tiny", "state_dict": CloudSegmenter().state_dict(),
                    "run_sha256": sha256_file(root / "run.json")}, root / "best.pt")
        write_json(root / "supplemental_provenance.json", {
            "run_sha256": sha256_file(root / "run.json"), "captured_before_optimization": True,
            "source_snapshot_sha256": sha256_file(root / "training_source_snapshot.zip"),
            "transform_sources_sha256": fingerprints,
            "versions": {key: "fixture" for key in ["torch", "numpy", "pillow"]},
        })
        write_json(root / "quality.json", {
            "contract_version": CONTRACT_VERSION, "final_test_used": False, "runtime_eligible": False,
            "checkpoint_sha256": sha256_file(root / "best.pt"), "run_sha256": sha256_file(root / "run.json"),
            "dataset": run["dataset"],
            "validation": {"class_order": list(CLASS_NAMES), "ignore_index": 255,
                           "by_episode": {"fixture": {}},
                           "overall": {"image_count": 3, "total_pixel_count": 12, "labeled_pixel_count": 9,
                                       "ignored_pixel_count": 3, "per_class": {name: {"support": 3} for name in CLASS_NAMES},
                                       "mean_iou": 0.4 + index * 0.1,
                                       "balanced_accuracy": 0.6 - index * 0.1,
                                       "pixel_accuracy": 0.75}},
        })
    return roots


def rewrite_run(root, mutate):
    run = json.loads((root / "run.json").read_text())
    mutate(run)
    write_json(root / "run.json", run)
    quality = json.loads((root / "quality.json").read_text())
    quality["run_sha256"] = sha256_file(root / "run.json")
    write_json(root / "quality.json", quality)
    state = torch.load(root / "best.pt", weights_only=True)
    state["run_sha256"] = quality["run_sha256"]
    torch.save(state, root / "best.pt")
    quality["checkpoint_sha256"] = sha256_file(root / "best.pt")
    write_json(root / "quality.json", quality)
    provenance_path = root / "supplemental_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["run_sha256"] = quality["run_sha256"]
    write_json(provenance_path, provenance)


def test_matched_validation_comparison_preserves_gains_and_regressions(experiments):
    result = compare(*experiments)
    assert result["candidate_minus_baseline"]["mean_iou"] == pytest.approx(0.1)
    assert result["candidate_minus_baseline"]["balanced_accuracy"] == pytest.approx(-0.1)
    assert result["candidate_minus_baseline"]["pixel_accuracy"] == 0
    assert result["runtime_eligible"] is False and result["final_test_used"] is False
    assert result["baseline"]["quality_sha256"] == sha256_file(experiments[0] / "quality.json")
    assert result["audited_source_differences"] == []


@pytest.mark.parametrize("artifact", ["best.pt", "run.json"])
def test_comparison_rejects_tampered_bound_artifacts(experiments, artifact):
    path = experiments[1] / artifact
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="bound validation-only"):
        compare(*experiments)


@pytest.mark.parametrize("field,value", [("final_test_used", True), ("runtime_eligible", True),
                                        ("contract_version", "other-task")])
def test_comparison_rejects_consumed_test_or_other_contract(experiments, field, value):
    path = experiments[1] / "quality.json"
    quality = json.loads(path.read_text())
    quality[field] = value
    write_json(path, quality)
    with pytest.raises(ValueError, match="bound validation-only"):
        compare(*experiments)


@pytest.mark.parametrize("change,match", [
    (lambda run: run["config"].update(seed=19), "configuration"),
    (lambda run: run["dataset"].update(records_sha256="new-split"), "dataset differ"),
    (lambda run: run["dataset"].update(manifest_sha256="new-ontology"), "dataset differ"),
    (lambda run: run.update(class_weights=[2, 1, 1]), "class weights"),
    (lambda run: run.update(teacher=None), "human-only baseline"),
])
def test_comparison_rejects_changed_split_training_or_supervision(experiments, change, match):
    rewrite_run(experiments[1], change)
    with pytest.raises(ValueError, match=match):
        compare(*experiments)


def test_comparison_rejects_different_actual_device_even_when_request_matches(experiments):
    rewrite_run(experiments[1], lambda run: run.update(device="cuda"))
    with pytest.raises(ValueError, match="execution.*device"):
        compare(*experiments)


def test_comparison_rejects_dependency_change(experiments):
    path = experiments[1] / "supplemental_provenance.json"
    provenance = json.loads(path.read_text())
    provenance["versions"]["pillow"] = "different"
    write_json(path, provenance)
    with pytest.raises(ValueError, match="dependency versions"):
        compare(*experiments)


def test_comparison_rejects_rebound_but_unreviewed_computation_change(experiments):
    root = experiments[1]
    name = "ml/models/cloud_losses.py"
    with zipfile.ZipFile(root / "training_source_snapshot.zip") as archive:
        sources = {key: archive.read(key) for key in archive.namelist()}
    sources[name] += b"changed loss"
    with zipfile.ZipFile(root / "training_source_snapshot.zip", "w") as archive:
        for key, value in sources.items():
            archive.writestr(key, value)
    digest = hashlib.sha256(sources[name]).hexdigest()
    rewrite_run(root, lambda run: run["source_sha256"].update({name: digest}))
    path = root / "supplemental_provenance.json"
    provenance = json.loads(path.read_text())
    provenance["transform_sources_sha256"][name] = digest
    provenance["source_snapshot_sha256"] = sha256_file(root / "training_source_snapshot.zip")
    write_json(path, provenance)
    with pytest.raises(ValueError, match="Unreviewed computation source"):
        compare(*experiments)


def test_comparison_rejects_rebound_checkpoint_with_wrong_embedded_run(experiments):
    root = experiments[0]
    path = root / "best.pt"
    state = torch.load(path, weights_only=True)
    state["run_sha256"] = "other-run"
    torch.save(state, path)
    quality = json.loads((root / "quality.json").read_text())
    quality["checkpoint_sha256"] = sha256_file(path)
    write_json(root / "quality.json", quality)
    with pytest.raises(ValueError, match="Checkpoint metadata"):
        compare(*experiments)


def test_comparison_rejects_different_annotation_support(experiments):
    path = experiments[1] / "quality.json"
    quality = json.loads(path.read_text())
    quality["validation"]["overall"]["per_class"]["cloud"]["support"] += 1
    write_json(path, quality)
    with pytest.raises(ValueError, match="validation class support"):
        compare(*experiments)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1"])
def test_training_wait_cannot_be_unbounded_by_nonfinite_timeout(value):
    from argparse import ArgumentTypeError
    with pytest.raises(ArgumentTypeError, match="finite and positive"):
        positive_hours(value)
