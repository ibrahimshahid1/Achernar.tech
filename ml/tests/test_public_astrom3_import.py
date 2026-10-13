import numpy as np
import pytest

from ml.data_pipeline.import_public_astrom3 import process_photometry, pack_rows, validate_row


def row(length=5):
    return {"photometry": np.arange(length * 9, dtype=np.float32).reshape(length, 9),
            "spectra": np.ones((3, 2575), dtype=np.float32),
            "metadata": np.arange(34, dtype=np.float32), "label": 2}


def test_padding_has_explicit_mask_and_preserves_processed_values():
    data = row()["photometry"]
    output, mask = process_photometry(data)
    assert output.shape == (200, 9)
    np.testing.assert_array_equal(output[:5], data)
    assert mask.sum() == 5
    assert not output[5:].any()


def test_center_window_matches_upstream():
    data = row(286)["photometry"]
    output, mask = process_photometry(data)
    np.testing.assert_array_equal(output, data[43:243])
    assert mask.sum() == 200


def test_seeded_training_window_reproducible_and_within_bounds():
    data = row(400)["photometry"]
    a, _ = process_photometry(data, how="random", rng=np.random.default_rng(42))
    b, _ = process_photometry(data, how="random", rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    assert data[0, 0] <= a[0, 0] <= data[200, 0]


def test_ragged_storage_round_trip():
    rows = [row(3), row(8)]
    arrays = pack_rows(rows)
    assert arrays["photometry_offsets"].tolist() == [0, 3, 11]
    np.testing.assert_array_equal(arrays["photometry_values"][3:11], rows[1]["photometry"])
    assert arrays["spectra"].shape == (2, 3, 2575)


@pytest.mark.parametrize("key,value", [("spectra", np.zeros((1, 1024))),
                                     ("metadata", np.zeros(16)), ("label", 10)])
def test_reject_edge_or_illustrative_wrong_contract(key, value):
    data = row()
    data[key] = value
    with pytest.raises(ValueError):
        validate_row(data)


def test_reject_nonfinite():
    data = row()
    data["metadata"][0] = np.nan
    with pytest.raises(ValueError):
        validate_row(data)
