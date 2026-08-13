"""The FOP input contract must not drift — spec Section 7.6.2.

`contracts/fop_features.py` validates its own structure at import, so most of
what these tests guard is the boundary *between* representations: the committed
JSON that the Rust runtime will read, and the feature-vector layout that a
trained artifact bakes in permanently.

The specific failure being defended against is the FQE's: three files describing
one input, no test comparing them, and the disagreement surfacing only as a model
that scored well offline and behaved oddly on the daemon.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.fop_features import (  # noqa: E402
    CONTRACT_JSON,
    FEATURE_NAMES,
    FEATURES,
    GROUP_SIZES,
    N_FEATURES,
    Availability,
    as_dict,
    by_group,
    trainable_indices,
    unavailable,
)


class TestContractJsonIsCurrent(unittest.TestCase):
    """The committed JSON is generated, so it can silently go stale."""

    def test_committed_json_matches_the_module(self):
        self.assertTrue(
            CONTRACT_JSON.exists(),
            f"{CONTRACT_JSON.name} is missing; regenerate with "
            "`python -m contracts.fop_features`",
        )
        committed = json.loads(CONTRACT_JSON.read_text())
        self.assertEqual(
            committed,
            as_dict(),
            f"{CONTRACT_JSON.name} is stale. The Rust runtime reads this file; "
            "regenerate it with `python -m contracts.fop_features`.",
        )


class TestVectorLayout(unittest.TestCase):
    """Index N is a permanent promise: a trained ONNX graph encodes it."""

    def test_width_matches_the_spec(self):
        self.assertEqual(len(FEATURES), 38)
        self.assertEqual(N_FEATURES, 38)

    def test_group_sizes_match_the_spec_table(self):
        # 3 quality, 4 classical, 5 position, 4 temperature, 7 filter,
        # 6 history, 3 direction, 3 rig, 3 sky.
        self.assertEqual(sum(GROUP_SIZES.values()), 38)
        for group, size in GROUP_SIZES.items():
            self.assertEqual(len(by_group(group)), size, group)

    def test_indices_are_contiguous_and_ordered(self):
        self.assertEqual([f.index for f in FEATURES], list(range(38)))

    def test_names_are_unique(self):
        self.assertEqual(len(set(FEATURE_NAMES)), 38)

    def test_filter_block_is_the_seven_spec_filters_in_order(self):
        # Order is load-bearing: a one-hot block read in a different order maps
        # Ha frames onto the B weight without any error surfacing.
        self.assertEqual(
            [f.name for f in by_group("filter")],
            ["filter_l", "filter_r", "filter_g", "filter_b",
             "filter_ha", "filter_oiii", "filter_sii"],
        )


class TestAvailabilityBookkeeping(unittest.TestCase):
    """The point of the contract is an honest account of what we can produce."""

    def test_every_unavailable_feature_explains_itself(self):
        for feature in unavailable():
            self.assertTrue(
                feature.note.strip(),
                f"{feature.name} is {feature.availability.value} with no reason given",
            )

    def test_trainable_set_excludes_everything_not_available(self):
        trainable = set(trainable_indices())
        for feature in unavailable():
            self.assertNotIn(
                feature.index,
                trainable,
                f"{feature.name} is {feature.availability.value} but marked trainable",
            )

    def test_the_whole_filter_block_is_degenerate_together(self):
        # D4: partial availability here would be a bug in the declarations —
        # either a wheel exists and all seven vary, or none do.
        verdicts = {f.availability for f in by_group("filter")}
        self.assertEqual(verdicts, {Availability.DEGENERATE})

    def test_direction_features_are_collectable(self):
        # These were DEGENERATE while the agent could only sweep monotonically:
        # with no direction change anywhere in the dataset the direction head
        # had no varying target and could not be trained at all. The agent now
        # has SweepPattern::Reversal, so the capability exists.
        #
        # What this cannot check is whether any *night* used it. The contract
        # notes say to verify the collected mix before training, and that check
        # belongs in the dataset builder against real rows, not here.
        direction = {f.name: f.availability for f in by_group("direction")}
        self.assertEqual(direction["dir_changed_recently"], Availability.AVAILABLE)
        self.assertEqual(direction["steps_since_flip_norm"], Availability.AVAILABLE)

    def test_no_feature_silently_lacks_provenance(self):
        for feature in FEATURES:
            if feature.availability is Availability.REJECTED:
                continue
            self.assertTrue(
                feature.source.strip() and feature.source != "(none)",
                f"{feature.name} has no source column recorded",
            )


class TestLabelDefinition(unittest.TestCase):
    def test_label_is_retrospective_and_grouped_by_run(self):
        from contracts.fop_features import LABEL

        self.assertTrue(LABEL["retrospective"])
        self.assertEqual(LABEL["group"], "focus_runs.id")
        self.assertIn("best_position", LABEL["definition"])

    def test_offset_buckets_match_the_balanced_sampler(self):
        from contracts.fop_features import OFFSET_BUCKETS

        self.assertEqual(OFFSET_BUCKETS[:5], (0, 5, 15, 40, 120))
        self.assertEqual(OFFSET_BUCKETS[-1], float("inf"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
