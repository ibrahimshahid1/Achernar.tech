from datetime import datetime

import numpy as np
import pytest

from ml.data_pipeline.import_public_cloud import rasterize_annotation, temporal_groups, CloudSegmentationDataset
from ml.data_pipeline.public_common import grouped_split, write_jsonl


def polygon(label, points):
    return {"label": label, "shape_type": "polygon", "points": points}


def annotation(shapes):
    return {"imageWidth": 512, "imageHeight": 512, "shapes": shapes}


def test_unannotated_is_ignored_and_conflicting_regions_are_not_order_dependent():
    a = polygon("sky", [[2, 2], [20, 2], [20, 20], [2, 20]])
    b = polygon("cloud", [[10, 10], [30, 10], [30, 30], [10, 30]])
    mask, _, conflicts = rasterize_annotation(annotation([a, b]))
    reverse, _, _ = rasterize_annotation(annotation([b, a]))
    np.testing.assert_array_equal(mask, reverse)
    assert mask[0, 0] == 255 and mask[5, 5] == 0 and mask[25, 25] == 1
    assert mask[15, 15] == 255 and conflicts > 0


def test_alias_and_open_or_degenerate_shapes_leave_inspectable_ledger():
    alias = polygon("clode", [[2, 2], [8, 2], [8, 8]])
    open_shape = {"label": "cloud", "shape_type": "linestrip", "points": [[30, 30], [50, 50]]}
    degenerate = polygon("contamination", [[60, 60], [70, 70]])
    mask, ledger, _ = rasterize_annotation(annotation([alias, open_shape, degenerate]))
    assert {row["action"] for row in ledger} == {"explicit_spelling_alias", "open_shape_ignored", "invalid_polygon_ignored"}
    assert mask[40, 40] == 255 and mask[65, 65] == 255
    assert (mask == 1).any()


def test_unrecognized_label_rejected():
    with pytest.raises(ValueError):
        rasterize_annotation(annotation([polygon("thin", [[2, 2], [8, 2], [8, 8]])]))


def test_adjacent_dates_and_duplicates_remain_in_same_split():
    stamps = [datetime(2020, 1, 1, 23), datetime(2020, 1, 2, 1), datetime(2020, 1, 8), datetime(2020, 2, 1)]
    groups = temporal_groups(stamps, ["a", "b", "a", "c"])
    assert groups[0] == groups[1] == groups[2] and groups[3] != groups[0]
    before = grouped_split(groups)
    after = grouped_split(groups + ["new-group"])
    assert all(before[group] == after[group] for group in groups)


def test_loader_nearest_resize_preserves_ignore_and_excludes_ineligible(tmp_path):
    from PIL import Image
    Image.new("RGB", (512, 512), (10, 20, 30)).save(tmp_path / "image.png")
    mask = np.full((512, 512), 255, np.uint8)
    mask[:256] = 1
    Image.fromarray(mask).save(tmp_path / "mask.png")
    write_jsonl(tmp_path / "records.jsonl", [{"sample_id": "valid", "split": "train", "image": "image.png", "mask": "mask.png", "training_eligible": True},
        {"sample_id": "invalid", "split": "train", "image": "image.png", "mask": "mask.png", "training_eligible": False}])
    dataset = CloudSegmentationDataset(tmp_path, size=128)
    assert len(dataset) == 1
    row = dataset[0]
    assert row["image"].shape == (3, 128, 128)
    assert set(np.unique(row["mask"])) == {1, 255}
