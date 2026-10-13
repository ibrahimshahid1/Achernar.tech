"""COC data, research inference, checkpoint and actual export round-trip tests.

All images here are synthetic contract fixtures. The tiny training test checks
execution/provenance, not astronomical model quality or field qualification.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")

from ml.coc import CLASS_NAMES, CONTRACT_VERSION, ONNX_PRECISION
from ml.coc.checkpoint import load_checkpoint
from ml.coc.data import HumanCloudDataset, asset, inspect_human_dataset
from ml.coc.inference import load_region, preprocess_image, summarize_logits
from ml.coc.predict import predict, verify_bundle
from ml.data_pipeline.public_common import sha256_file
from ml.models.cloud_classifier import CloudSegmenter


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


@pytest.fixture
def human_fixture(tmp_path):
    root = tmp_path / "human"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    mask = np.repeat(np.array([0, 1, 2, 255], dtype=np.uint8), 8)[None].repeat(32, axis=0)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0]], dtype=np.uint8)
    rows = []
    for index, split in enumerate(["train", "train", "validation", "validation", "test", "test"]):
        sample_id = f"fixture-{index}"
        rgb = colors[np.where(mask == 255, 3, mask)].copy()
        # Unique original images in each partition; ignored pixels remain ignored.
        rgb[0, -1] = [index + 1, 13, 47]
        image_name, mask_name = f"images/{sample_id}.png", f"masks/{sample_id}.png"
        Image.fromarray(rgb).save(root / image_name)
        Image.fromarray(mask).save(root / mask_name)
        rows.append({
            "sample_id": sample_id, "image": image_name, "mask": mask_name,
            "image_sha256": sha256_file(root / image_name), "mask_sha256": sha256_file(root / mask_name),
            "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "split": split, "split_group": f"fixture-episode-{index}", "ancestor": f"fixture-source-{index}",
            "training_eligible": True, "label_type": "human_partial_regions", "camera_era": "fixture",
            "five_class_label": None, "transparency_label": None,
        })
    (root / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    _write_json(root / "manifest.json", {
        "dataset_id": "lenghu_sky8_human", "task": "cloud_segmentation", "record_count": len(rows),
        "contract": {"classes": {"0": "sky", "1": "cloud", "2": "contamination", "255": "ignore"}},
        "sources": ["synthetic test fixture; no external source"], "license": "test fixture only",
        "limitations": ["Synthetic contract fixture; not astronomical evidence"],
    })
    return root, rows


def _rewrite_rows(root, rows):
    (root / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def _logits_for(classes):
    classes = np.asarray(classes, dtype=np.int64)
    return np.stack([np.where(classes == index, 8.0, -8.0) for index in range(3)]).astype(np.float32)


def test_region_is_required_even_when_all_predictions_look_clear():
    summary, classes, scores = summarize_logits(_logits_for(np.zeros((2, 2), dtype=int)))
    assert summary["status"] == "abstained" and summary["reason"] == "valid_region_required"
    assert summary["cloud_area_fraction_of_roi"] is None
    assert summary["cloud_fraction_of_classified_sky"] is None
    assert summary["five_class_condition"] is None and summary["transparency"] is None
    assert summary["runtime_eligible"] is False and summary["probability_calibrated"] is False
    assert summary["class_order"] == ["sky", "cloud", "contamination"]
    assert classes.shape == (2, 2) and scores.shape == (3, 2, 2)
    np.testing.assert_allclose(scores.sum(axis=0), 1)


def test_empty_region_abstains_and_does_not_report_zero_cloud_as_clear():
    summary, _, _ = summarize_logits(_logits_for([[1, 1], [1, 1]]), np.zeros((2, 2), dtype=bool))
    assert summary["reason"] == "empty_valid_region" and summary["valid_region_pixels"] == 0
    assert summary["cloud_area_fraction_of_roi"] is None


def test_empty_spatial_logits_are_rejected_before_reduction():
    with pytest.raises(ValueError, match="finite logits"):
        summarize_logits(np.empty((3, 0, 2), dtype=np.float32))


def test_summary_separates_roi_coverage_classified_sky_and_contamination():
    summary, _, _ = summarize_logits(_logits_for([[0, 1], [2, 1]]), np.ones((2, 2), dtype=bool))
    assert summary["status"] == "research_diagnostic"
    assert summary["cloud_area_fraction_of_roi"] == 0.5
    assert summary["cloud_fraction_of_classified_sky"] == pytest.approx(2 / 3)
    assert summary["contamination_area_fraction_of_roi"] == 0.25
    assert summary["five_class_condition"] is None and summary["transparency"] is None


@pytest.mark.parametrize("classes,reason", [
    ([[2, 2], [1, 0]], "contamination_exceeds_diagnostic_threshold"),
    ([[2, 2], [2, 2]], "no_classified_sky"),
])
def test_contaminated_region_abstains_instead_of_becoming_cloud_or_clear(classes, reason):
    summary, _, _ = summarize_logits(_logits_for(classes), np.ones((2, 2), dtype=bool))
    assert summary["status"] == "abstained" and summary["reason"] == reason
    assert summary["cloud_fraction_of_classified_sky"] is None
    assert summary["mean_soft_cloud_share_of_sky"] is None


@pytest.mark.parametrize("region", [np.ones((2, 2), dtype=np.uint8), np.ones((3, 2), dtype=bool)])
def test_summary_rejects_nonboolean_or_mismatched_region(region):
    with pytest.raises(ValueError, match="boolean"):
        summarize_logits(_logits_for([[0, 1], [1, 0]]), region)


def test_roi_loading_preserves_binary_geometry_and_rejects_other_values(tmp_path):
    path = tmp_path / "roi.png"
    Image.fromarray(np.array([[0, 255], [1, 0]], dtype=np.uint8)).save(path)
    region = load_region(path, size=4, original_size=[2, 2])
    np.testing.assert_array_equal(region, np.array([[False, False, True, True], [False, False, True, True],
                                                  [True, True, False, False], [True, True, False, False]]))
    with pytest.raises(ValueError, match="geometry"):
        load_region(path, size=4, original_size=[3, 2])
    Image.fromarray(np.array([[0, 128], [255, 0]], dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="binary"):
        load_region(path, size=4, original_size=[2, 2])


def test_preprocess_requires_rendered_pixels_with_applied_orientation(tmp_path):
    path = tmp_path / "sensor.tif"
    Image.fromarray(np.ones((32, 32), dtype=np.uint16)).save(path)
    with pytest.raises(ValueError, match="rendered"):
        preprocess_image(path, 32)
    path = tmp_path / "oriented.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (32, 32)).save(path, exif=exif)
    with pytest.raises(ValueError, match="orientation"):
        preprocess_image(path, 32)


def test_human_augmentation_keeps_images_partial_labels_and_ignore_regions_paired(human_fixture):
    root, _ = human_fixture
    dataset = HumanCloudDataset(root, "train", size=32, augment=True, cache=True)
    for seed in range(8):
        torch.manual_seed(seed)
        example = dataset[0]
        image, target = example["image"], example["mask"]
        assert image.is_contiguous() and target.is_contiguous()
        assert set(torch.unique(target).tolist()) == {0, 1, 2, 255}
        # Fixture RGB channels encode the annotated class. Any mismatched D4
        # transform would break agreement even though both shapes remain valid.
        keep = target != 255
        assert torch.equal(image.argmax(0)[keep], target[keep])
        assert int(keep.sum()) == 32 * 24
    np.testing.assert_array_equal(dataset.training_pixel_counts(), [512, 512, 512])


def test_validation_cannot_be_augmented_and_train_excludes_ineligible_rows(human_fixture):
    root, rows = human_fixture
    with pytest.raises(ValueError, match="Held-out"):
        HumanCloudDataset(root, "validation", size=32, augment=True)
    rows[0]["training_eligible"] = False
    _rewrite_rows(root, rows)
    assert len(HumanCloudDataset(root, "train", size=32)) == 1
    assert len(HumanCloudDataset(root, "validation", size=32)) == 2


@pytest.mark.parametrize("field", ["split_group", "ancestor", "rgb_sha256"])
def test_inspection_rejects_a_shared_episode_ancestor_or_duplicate_across_splits(human_fixture, field):
    root, rows = human_fixture
    rows[2][field] = rows[0][field]
    _rewrite_rows(root, rows)
    with pytest.raises(ValueError, match="crosses splits"):
        inspect_human_dataset(root)


def test_inspection_rejects_teacher_labels_or_tampered_assets(human_fixture):
    root, rows = human_fixture
    rows[0]["label_type"] = "producer_teacher_logits"
    _rewrite_rows(root, rows)
    with pytest.raises(ValueError, match="pseudo-labels"):
        inspect_human_dataset(root)
    rows[0]["label_type"] = "human_partial_regions"
    _rewrite_rows(root, rows)
    Image.new("RGB", (32, 32), color=(20, 20, 20)).save(root / rows[0]["image"])
    with pytest.raises(ValueError, match="content changed"):
        inspect_human_dataset(root)


def test_dataset_paths_cannot_escape_the_root(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        asset(tmp_path, "../unrelated.png")


@pytest.mark.parametrize("override", [
    {"contract_version": "legacy-classifier-v1"},
    {"class_order": ["clear", "thin", "moderate", "heavy", "overcast"]},
    {"class_order": ["cloud", "sky", "contamination"]},
])
def test_checkpoint_rejects_wrong_task_or_class_order_before_loading_weights(tmp_path, override):
    path = tmp_path / "wrong.pt"
    state = {"contract_version": CONTRACT_VERSION, "class_order": list(CLASS_NAMES), **override}
    torch.save(state, path)
    with pytest.raises(ValueError, match="task or class order"):
        load_checkpoint(path)


def test_checkpoint_rejects_nonfinite_weights(tmp_path):
    state = CloudSegmenter().state_dict()
    state["head.weight"].view(-1)[0] = float("nan")
    path = tmp_path / "nonfinite.pt"
    torch.save({"contract_version": CONTRACT_VERSION, "class_order": list(CLASS_NAMES),
                "size": 32, "variant": "tiny", "state_dict": state}, path)
    with pytest.raises(ValueError, match="Nonfinite"):
        load_checkpoint(path)


def test_bundle_verification_rejects_tampering_and_paths_outside_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    graph = bundle / "model.onnx"
    graph.write_bytes(b"hash verification fixture; not an ONNX graph")
    manifest = {"contract_version": CONTRACT_VERSION, "class_order": list(CLASS_NAMES),
                "runtime_eligible": False, "promotion_status": "research", "image_size": 32,
                "task": "cloud_segmentation", "family": "cloud_segmentation", "framework": "onnx",
                "precision": ONNX_PRECISION, "input": {"image": ["batch", 3, 32, 32]},
                "output": {"logits": ["batch", 3, 32, 32]},
                "files_sha256": {"model.onnx": sha256_file(graph)}}
    _write_json(bundle / "manifest.json", manifest)
    verify_bundle(bundle)
    graph.write_bytes(graph.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        verify_bundle(bundle)
    manifest["files_sha256"]["model.onnx"] = sha256_file(graph)
    outside = tmp_path / "unrelated.json"
    outside.write_text("{}")
    manifest["files_sha256"]["../unrelated.json"] = sha256_file(outside)
    _write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ValueError, match="integrity"):
        verify_bundle(bundle)


@pytest.mark.parametrize("field,value", [("image_size", True), ("task", "legacy_cloud_classifier"),
                                         ("precision", "INT8")])
def test_bundle_rejects_relabelled_contract_before_graph_execution(tmp_path, field, value):
    bundle = tmp_path / "wrong-contract"
    bundle.mkdir()
    manifest = {"contract_version": CONTRACT_VERSION, "class_order": list(CLASS_NAMES),
                "runtime_eligible": False, "promotion_status": "research", "image_size": 32,
                "task": "cloud_segmentation", "family": "cloud_segmentation", "framework": "onnx",
                "precision": ONNX_PRECISION, "input": {"image": ["batch", 3, 32, 32]},
                "output": {"logits": ["batch", 3, 32, 32]}, field: value}
    _write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ValueError, match="image size|tensor/task"):
        verify_bundle(bundle)


def test_final_test_cli_guard_runs_before_opening_checkpoint_or_dataset(tmp_path):
    result = subprocess.run([sys.executable, "-m", "ml.eval.eval_cloud_segmentation",
                             "--checkpoint", str(tmp_path / "does-not-exist.pt"),
                             "--split", "test", "--output", str(tmp_path / "report.json")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "final test is reserved" in result.stderr
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("color", [(0, 0, 0), (255, 255, 255), (30, 100, 160)])
def test_uniform_rgb_region_abstains_even_if_model_returns_classes(tmp_path, color):
    checkpoint = tmp_path / "uniform.pt"
    torch.save({"contract_version": CONTRACT_VERSION, "class_order": list(CLASS_NAMES), "size": 32,
                "variant": "tiny", "state_dict": CloudSegmenter().state_dict()}, checkpoint)
    image, roi = tmp_path / "image.png", tmp_path / "roi.png"
    Image.new("RGB", (32, 32), color=color).save(image)
    Image.new("L", (32, 32), color=255).save(roi)
    result = predict(image, tmp_path / "prediction", checkpoint=checkpoint, roi=roi)
    assert result["reason"] == "uninformative_rgb_input" and result["status"] == "abstained"
    assert result["cloud_area_fraction_of_roi"] is None
    assert result["cloud_fraction_of_classified_sky"] is None


def test_train_validate_export_and_predict_round_trip_uses_no_final_test(human_fixture, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from ml.export.export_cloud_segmentation import export
    from ml.train.train_cloud_segmentation import train

    root, rows = human_fixture
    output = tmp_path / "train"
    config = {"human_root": str(root), "output": str(output), "variant": "tiny", "size": 32,
              "epochs": 1, "batch_size": 2, "learning_rate": 0.001, "device": "cpu", "threads": 1,
              "seed": 17, "augment": True, "cache": True, "teacher_root": None, "teacher_limit": 16,
              "teacher_weight": 0.0, "teacher_temperature": 2.0, "teacher_confidence": 0.8}
    torch.manual_seed(config["seed"])
    initial_head = CloudSegmenter().head.weight.detach().clone()
    original_threads = torch.get_num_threads()
    original_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        quality = train(config)
        model, state = load_checkpoint(output / "best.pt")
        assert not torch.equal(initial_head, model.head.weight.detach())  # Actual optimizer update.
        assert state["epoch"] == quality["best_epoch"] == 1
        assert quality["final_test_used"] is False and quality["runtime_eligible"] is False
        assert quality["validation"]["overall"]["image_count"] == 2
        assert quality["validation"]["overall"]["labeled_pixel_count"] == 2 * 32 * 24
        run = json.loads((output / "run.json").read_text())
        assert run["training_images"] == run["validation_images"] == 2
        assert run["teacher"] is None
        assert run["training_pixel_counts"] == [512, 512, 512]
        assert sha256_file(output / "run.json") == state["run_sha256"]

        image_path = root / rows[2]["image"]  # Validation fixture, not final test.
        bundle = tmp_path / "export"
        result = export(output / "best.pt", bundle, image_path)
        assert len(result["parity"]) == 4 and all(case["passed"] for case in result["parity"])
        manifest = verify_bundle(bundle)
        assert manifest["family"] == manifest["task"] == "cloud_segmentation"
        assert manifest["runtime_eligible"] is False
        assert manifest["output"] == {"logits": ["batch", 3, 32, 32]}
        assert not (bundle / "metadata.json").exists()  # No daemon promotion manifest.

        no_region = predict(image_path, tmp_path / "no-roi", bundle=bundle)
        assert no_region["reason"] == "valid_region_required"
        roi = tmp_path / "roi.png"
        Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(roi)
        onnx_output, torch_output = tmp_path / "prediction-onnx", tmp_path / "prediction-torch"
        onnx_summary = predict(image_path, onnx_output, bundle=bundle, roi=roi)
        torch_summary = predict(image_path, torch_output, checkpoint=output / "best.pt", roi=roi)
        assert onnx_summary["valid_region_pixels"] == torch_summary["valid_region_pixels"] == 1024
        assert onnx_summary["artifact"]["backend"] == "onnxruntime_cpu"
        assert torch_summary["artifact"]["backend"] == "pytorch_cpu"
        assert onnx_summary["five_class_condition"] is None and onnx_summary["transparency"] is None
        assert onnx_summary["runtime_eligible"] is False and onnx_summary["probability_calibrated"] is False
        with np.load(onnx_output / "scores.npz", allow_pickle=False) as actual:
            with np.load(torch_output / "scores.npz", allow_pickle=False) as expected:
                np.testing.assert_allclose(actual["scores"], expected["scores"], atol=1e-4, rtol=1e-4)
        classes = np.asarray(Image.open(onnx_output / "classes.png"))
        assert np.isin(classes, [0, 1, 2]).all()
        assert json.loads((onnx_output / "prediction.json").read_text())["roi_sha256"] == sha256_file(roi)
        with pytest.raises(ValueError, match="fresh prediction"):
            predict(image_path, onnx_output, bundle=bundle, roi=roi)
        with pytest.raises(ValueError, match="immutable"):
            train(config)
    finally:
        torch.set_num_threads(original_threads)
        torch.use_deterministic_algorithms(original_determinism)
