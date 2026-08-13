"""Focused contract tests for FQE splitting, batching, and calibration."""

import os
import sys
import tempfile
import unittest

import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.split_registry import (  # noqa: E402
    create_group_splits,
    create_splits,
    create_temporal_group_splits,
    load_registered_splits,
    splits_for_dataset,
)
from train.calibrate_uncertainty import calibrate_from_arrays  # noqa: E402
from train.train_focus_quality import GroupedBatchSampler  # noqa: E402
from preprocess.normalize import robust_normalize  # noqa: E402


class TestNormalizationParity(unittest.TestCase):
    def test_nonfinite_pixels_map_to_zero(self):
        image = np.array([[0.0, 1.0], [np.nan, np.inf]], dtype=np.float32)
        normalized, _ = robust_normalize(image)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertEqual(float(normalized[1, 0]), 0.0)
        self.assertEqual(float(normalized[1, 1]), 0.0)


class TestGroupedSplits(unittest.TestCase):
    def test_all_splits_are_nonempty_and_groups_are_disjoint(self):
        groups = np.repeat(np.array([f"run-{i}" for i in range(8)]), 3)
        splits = create_group_splits(groups, seed=7)
        self.assertTrue(all(index.size > 0 for index in splits.values()))
        split_groups = {name: set(groups[index]) for name, index in splits.items()}
        self.assertTrue(split_groups["train"].isdisjoint(split_groups["val"]))
        self.assertTrue(split_groups["train"].isdisjoint(split_groups["test"]))
        self.assertTrue(split_groups["val"].isdisjoint(split_groups["test"]))

    def test_too_few_groups_fails_instead_of_emitting_empty_holdouts(self):
        with self.assertRaises(ValueError):
            create_group_splits(np.array([0, 0, 1, 1]))

    def test_registry_is_reused_and_rejects_a_changed_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = os.path.join(directory, "fqe.npz")
            data = {
                "run_id": np.repeat(np.arange(4), 2),
                "split_group": np.repeat(np.array(["a", "b", "c", "d"]), 2),
            }
            np.savez_compressed(dataset_path, **data)
            splits = splits_for_dataset(dataset_path, data, seed=3)
            registry = os.path.splitext(dataset_path)[0] + ".splits.json"
            loaded = load_registered_splits(registry, dataset_path, data["split_group"])
            for name in splits:
                np.testing.assert_array_equal(splits[name], loaded[name])

            np.savez_compressed(dataset_path, **data, changed=np.array([1]))
            with self.assertRaises(ValueError):
                load_registered_splits(registry, dataset_path, data["split_group"])


class TestGroupedBatching(unittest.TestCase):
    def test_every_rank_batch_contains_one_run_with_pairs(self):
        run_ids = np.repeat(np.arange(5), 9)
        sampler = GroupedBatchSampler(run_ids, batch_size=4, seed=0)
        batches = list(iter(sampler))
        self.assertGreater(len(batches), 5)
        for batch in batches:
            self.assertEqual(len(np.unique(run_ids[batch])), 1)
            self.assertGreaterEqual(len(batch), 2)


class TestRegressionCalibration(unittest.TestCase):
    def test_quantile_boundary_includes_tied_residuals(self):
        # Reproduces the synthetic-pipeline failure where the 95th percentile was
        # shared by many residuals. A rounded 1.96 divisor reconstructed a value
        # one ulp below the tie and reported 71% rather than >=95% coverage.
        y_val = np.concatenate([np.ones(30), np.full(12, 2.0)])
        zeros = np.zeros_like(y_val)
        ones = np.ones_like(y_val)
        record = calibrate_from_arrays(zeros, ones, y_val)

        self.assertGreaterEqual(record["coverage_95_val_after"], 0.95)

    def test_std_scaling_reports_multiple_levels_and_heldout_coverage(self):
        rng = np.random.default_rng(4)
        y_val = rng.normal(0.0, 2.0, 4000)
        y_test = rng.normal(0.0, 2.0, 4000)
        mu_val = np.zeros_like(y_val)
        mu_test = np.zeros_like(y_test)
        sigma_val = np.ones_like(y_val)
        sigma_test = np.ones_like(y_test)
        record = calibrate_from_arrays(
            mu_val,
            sigma_val,
            y_val,
            mu_test,
            sigma_test,
            y_test,
            dataset="fixture",
        )

        self.assertGreater(record["sigma_scale"], 1.7)
        self.assertIn("0.5", record["coverage_test_after"])
        self.assertIn("0.95", record["coverage_test_after"])
        self.assertAlmostEqual(record["coverage_95_test_after"], 0.95, delta=0.02)

    def test_frame_scale_calibrates_the_post_pooling_distribution(self):
        rng = np.random.default_rng(8)
        n_frames = 200
        stars_per_frame = 5
        frame_ids = np.repeat(np.arange(n_frames), stars_per_frame)
        true_frame = rng.normal(1.0, 0.25, n_frames)
        y = np.repeat(true_frame, stars_per_frame)
        # Model residuals are much wider than its per-star sigma claims.
        mu = y + rng.normal(0.0, 0.30, y.size)
        sigma = np.full(y.size, 0.08)
        split = n_frames // 2 * stars_per_frame
        record = calibrate_from_arrays(
            mu[:split],
            sigma[:split],
            y[:split],
            mu[split:],
            sigma[split:],
            y[split:],
            dataset="frame-fixture",
            frame_id_val=frame_ids[:split],
            frame_id_test=frame_ids[split:],
        )

        self.assertIn("frame_sigma_scale", record)
        self.assertGreater(record["frame_sigma_scale"], 0.0)
        self.assertAlmostEqual(record["frame_coverage_95_test_after"], 0.95, delta=0.06)



class TestTemporalSplits(unittest.TestCase):
    """Spec Section 7.7.2's oldest-train / newest-test split.

    The random-group split asks whether the model generalizes to an unseen run.
    This one asks whether it survives a month of the focuser being re-greased,
    the tube being swapped and the seeing season turning — drift that a shuffled
    split hides completely by scattering each night across all three sets.
    """

    @staticmethod
    def _dataset(n_groups=10, rows_per_group=3):
        groups = np.repeat(
            np.array([f"run-{i:02d}" for i in range(n_groups)]), rows_per_group
        )
        # One run per night, ascending. Deliberately *not* in group-name order
        # for the shuffled case below.
        stamps = np.repeat(
            np.array([f"2026-11-{i + 1:02d}T22:00:00+00:00" for i in range(n_groups)]),
            rows_per_group,
        )
        return groups, stamps

    def test_test_set_is_strictly_newer_than_train(self):
        groups, stamps = self._dataset()
        splits = create_temporal_group_splits(groups, stamps)

        def span(name):
            values = stamps[splits[name]].tolist()
            return min(values), max(values)

        self.assertLess(span("train")[1], span("val")[0])
        self.assertLess(span("val")[1], span("test")[0])

    def test_groups_never_straddle_a_boundary(self):
        # A run is placed by its earliest row, so every row of it lands on the
        # same side even when the run spans a boundary timestamp.
        groups, stamps = self._dataset()
        splits = create_temporal_group_splits(groups, stamps)
        sets = {name: set(groups[index]) for name, index in splits.items()}
        self.assertTrue(sets["train"].isdisjoint(sets["val"]))
        self.assertTrue(sets["train"].isdisjoint(sets["test"]))
        self.assertTrue(sets["val"].isdisjoint(sets["test"]))
        self.assertEqual(
            sum(len(s) for s in sets.values()), len(set(groups.tolist()))
        )

    def test_ordering_follows_time_not_group_name(self):
        # The whole split would be silently wrong if it fell back to sorting by
        # id, which for `run-00 .. run-09` happens to look plausible.
        groups = np.repeat(np.array(["run-a", "run-b", "run-c", "run-d"]), 2)
        stamps = np.repeat(
            np.array(
                [
                    "2026-12-04T22:00:00+00:00",  # run-a is the NEWEST
                    "2026-12-01T22:00:00+00:00",
                    "2026-12-02T22:00:00+00:00",
                    "2026-12-03T22:00:00+00:00",
                ]
            ),
            2,
        )
        splits = create_temporal_group_splits(groups, stamps)
        self.assertEqual(set(groups[splits["test"]]), {"run-a"})

    def test_all_splits_are_nonempty(self):
        groups, stamps = self._dataset(n_groups=3, rows_per_group=1)
        splits = create_temporal_group_splits(groups, stamps)
        self.assertTrue(all(index.size > 0 for index in splits.values()))

    def test_too_few_groups_fails(self):
        with self.assertRaises(ValueError):
            create_temporal_group_splits(np.array(["a", "a"]), np.array(["t1", "t2"]))

    def test_misaligned_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            create_temporal_group_splits(np.array(["a", "b", "c"]), np.array(["t1"]))

    def test_registry_records_the_strategy_and_boundaries(self):
        import json

        groups, stamps = self._dataset()
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = os.path.join(directory, "fqe.npz")
            np.savez_compressed(
                dataset_path, split_group=groups, captured_at=stamps
            )
            create_splits(dataset_path, strategy="temporal")

            registry = json.loads(
                Path(os.path.splitext(dataset_path)[0] + ".splits.json").read_text()
            )
            self.assertEqual(registry["strategy"], "temporal")
            self.assertEqual(registry["order_field"], "captured_at")
            # Spec 7.7.2 ratios, not the 80/10/10 the random-group split uses.
            self.assertAlmostEqual(registry["ratios"]["train"], 0.70)
            self.assertAlmostEqual(registry["ratios"]["val"], 0.15)
            # A deterministic ordering has no seed to record; claiming one would
            # imply the split could be re-rolled.
            self.assertIsNone(registry["seed"])
            self.assertLess(
                registry["boundaries"]["train"]["last"],
                registry["boundaries"]["test"]["first"],
            )

    def test_temporal_strategy_needs_a_timestamp_column(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = os.path.join(directory, "fqe.npz")
            np.savez_compressed(
                dataset_path, split_group=np.array(["a", "b", "c"])
            )
            with self.assertRaises(ValueError) as ctx:
                create_splits(dataset_path, strategy="temporal")
            self.assertIn("captured_at", str(ctx.exception))

    def test_random_group_remains_the_default(self):
        # Every existing registry was built this way; changing the default would
        # silently re-split datasets that models were already trained against.
        groups, stamps = self._dataset()
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = os.path.join(directory, "fqe.npz")
            np.savez_compressed(
                dataset_path, split_group=groups, captured_at=stamps
            )
            create_splits(dataset_path, seed=3)

            import json

            registry = json.loads(
                Path(os.path.splitext(dataset_path)[0] + ".splits.json").read_text()
            )
            self.assertEqual(registry["strategy"], "random_group")
            self.assertEqual(registry["seed"], 3)
            self.assertAlmostEqual(registry["ratios"]["train"], 0.8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
