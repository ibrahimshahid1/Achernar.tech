import hashlib

import numpy as np
from PIL import Image
import pytest

from ml.coc.compare_r50 import checked_input, output_record, reviewed_region


def test_review_is_bound_to_exact_original_and_preview():
    row = {"filename": "sample.CR3", "original_sha256": "a" * 64,
           "preview_sha256": "b" * 64, "preview_width": "4", "preview_height": "2"}
    review = {**{k: row[k] for k in ["filename", "original_sha256", "preview_sha256"]},
              "rect": [.25, 0, .75, .5]}
    region = reviewed_region(row, review, 4)
    assert region.tolist() == [[False, True, True, False]] * 2 + [[False] * 4] * 2
    with pytest.raises(ValueError, match="exact source"):
        reviewed_region(row, {**review, "original_sha256": "c" * 64}, 4)
    with pytest.raises(ValueError, match="Invalid independent"):
        reviewed_region(row, {**review, "rect": [0, 0, float("nan"), 1]}, 4)


def test_changed_rendered_input_cannot_reuse_inventory_identity(tmp_path):
    path = tmp_path / "input.png"
    Image.new("RGB", (4, 2), (50, 80, 100)).save(path)
    row = {"filename": "sample.CR3", "preview_path": str(path),
           "preview_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "preview_width": "4", "preview_height": "2"}
    assert checked_input(row, 32)[0].shape == (1, 3, 32, 32)
    Image.new("RGB", (4, 2), (60, 80, 100)).save(path)
    with pytest.raises(ValueError, match="Changed preview"):
        checked_input(row, 32)


def test_unmasked_counts_are_never_sky_coverage_or_accuracy():
    logits = np.zeros((3, 2, 2), np.float32)
    logits[1] = 3
    record, classes = output_record(logits, None, np.zeros((3, 2, 2), np.float32))
    assert record["full_frame_predicted_class_counts"] == [0, 4, 0]
    assert record["summary"]["status"] == "abstained"
    assert record["summary"]["cloud_fraction_of_classified_sky"] is None
    assert "not sky coverage or accuracy" in record["full_frame_distribution_scope"]


def test_uniform_roi_rejects_apparent_perfect_cloud_coverage():
    logits = np.zeros((3, 2, 2), np.float32)
    logits[1] = 3
    record, _ = output_record(logits, np.ones((2, 2), bool), np.zeros((3, 2, 2), np.float32))
    assert record["summary"]["reason"] == "uninformative_rgb_input"
    assert record["summary"]["cloud_area_fraction_of_roi"] is None
