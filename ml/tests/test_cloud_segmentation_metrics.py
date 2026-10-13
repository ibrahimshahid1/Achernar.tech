import json

import numpy as np
import pytest

from ml.eval.cloud_segmentation_metrics import CloudSegmentationMetrics, confusion_metrics


def test_confusion_orientation_and_class_metrics_are_explicit():
    # Sky has 2/3 recall but 2/4 precision; swapping matrix axes changes them.
    report = confusion_metrics(np.array([[2, 1, 0], [2, 1, 0], [0, 1, 1]]))
    assert report["labeled_pixel_count"] == 8
    assert report["pixel_accuracy"] == 0.5
    sky = report["per_class"]["sky"]
    assert sky["precision"] == 0.5 and sky["recall"] == pytest.approx(2 / 3)
    assert sky["iou"] == pytest.approx(2 / 5)
    assert report["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 3 + 1 / 2) / 3)


def test_ignored_regions_never_become_clear_sky_or_counted_errors():
    target = np.array([[0, 1, 255], [2, 255, 255]])
    prediction = np.array([[0, 2, -100], [2, 17, 255]])
    report = CloudSegmentationMetrics().update(prediction, target).compute()["overall"]
    assert report["confusion_matrix"] == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
    assert report["labeled_pixel_count"] == report["ignored_pixel_count"] == 3
    assert report["annotated_pixel_fraction"] == 0.5
    assert report["per_class"]["cloud"]["recall"] == 0
    assert report["per_class"]["cloud"]["precision"] is None


def test_absent_classes_and_empty_annotation_metrics_are_json_safe():
    meter = CloudSegmentationMetrics()
    meter.update(np.zeros((2, 2), dtype=int), np.full((2, 2), 255), split_groups="unknown-label-episode")
    empty = meter.compute()
    assert empty["overall"]["image_count"] == 1
    assert empty["overall"]["images_with_labels"] == 0
    assert empty["overall"]["balanced_accuracy"] is None
    assert empty["episode_macro"]["mean_iou"] is None
    json.dumps(empty, allow_nan=False)
    meter.update(np.zeros((1, 2), dtype=int), np.zeros((1, 2), dtype=int))
    overall = meter.compute()["overall"]
    assert overall["balanced_accuracy"] == 1
    assert overall["balanced_accuracy_supported_classes"] == ["sky"]
    assert overall["per_class"]["cloud"]["iou"] is None
    assert overall["per_class"]["contamination"]["recall"] is None
    json.dumps(meter.compute(), allow_nan=False)


def test_streaming_strata_pool_repeated_episodes_but_macro_weights_episodes_equally():
    meter = CloudSegmentationMetrics()
    meter.update(np.zeros((2, 2, 2), dtype=int), np.zeros((2, 2, 2), dtype=int),
                 split_groups=["long", "long"], camera_eras=["old", "old"])
    meter.update(np.ones((1, 1), dtype=int), np.zeros((1, 1), dtype=int),
                 split_groups="short", camera_eras="new")
    report = meter.compute()
    assert report["overall"]["labeled_pixel_count"] == 9
    assert report["overall"]["balanced_accuracy"] == pytest.approx(8 / 9)
    assert report["by_episode"]["long"]["image_count"] == 2
    assert report["by_episode"]["long"]["labeled_pixel_count"] == 8
    assert report["episode_macro"]["balanced_accuracy"] == 0.5
    assert report["by_camera_era"]["old"]["pixel_accuracy"] == 1
    assert report["by_camera_era"]["new"]["pixel_accuracy"] == 0
    assert report["episode_macro"]["per_class_iou"]["sky"] == 0.5


@pytest.mark.parametrize("prediction,target,metadata", [
    (np.zeros((1, 2), int), np.zeros((2, 1), int), {}),
    (np.zeros((2, 2), float), np.zeros((2, 2), int), {}),
    (np.zeros((2, 2), int), np.full((2, 2), 3), {}),
    (np.full((2, 2), 3), np.zeros((2, 2), int), {}),
    (np.zeros((2, 2, 2), int), np.zeros((2, 2, 2), int), {"split_groups": "shared"}),
    (np.zeros((2, 2, 2), int), np.zeros((2, 2, 2), int), {"camera_eras": ["old"]}),
])
def test_invalid_contracts_fail_before_accumulating(prediction, target, metadata):
    meter = CloudSegmentationMetrics()
    with pytest.raises(ValueError):
        meter.update(prediction, target, **metadata)
    assert meter.compute()["overall"]["image_count"] == 0
