"""
split_registry — Section 7.7.5.
Deterministic, GROUPED train/val/test splits + a JSON registry for reproducibility.

Splits are grouped by run (all stars/frames of one autofocus run stay on the same
side of the split). Random per-sample splitting would leak: stars from the same
frame are near-duplicates, so a random split inflates metrics and hides
generalization failure (Kaufman & Rosset 2012; Oakden-Rayner 2020).
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np


def create_group_splits(
    run_ids, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 0
) -> dict[str, np.ndarray]:
    """Return index arrays for train/val/test, split by unique group (run) id."""
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and leave a positive test ratio")
    run_ids = np.asarray(run_ids)
    groups = np.unique(run_ids)
    if groups.size < 3:
        raise ValueError(
            f"grouped train/val/test splitting needs at least 3 groups, got {groups.size}"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n = len(groups)
    test_ratio = 1.0 - train_ratio - val_ratio
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * test_ratio)))
    # Preserve at least one training group for tiny-but-valid fixtures.
    while n_val + n_test >= n:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            raise ValueError("could not allocate non-empty grouped splits")
    n_train = n - n_val - n_test
    train_g = set(groups[:n_train].tolist())
    val_g = set(groups[n_train : n_train + n_val].tolist())
    test_g = set(groups[n_train + n_val : n_train + n_val + n_test].tolist())

    def idx(group_set):
        if not group_set:
            return np.array([], dtype=np.int64)
        return np.where(np.isin(run_ids, list(group_set)))[0]

    splits = {"train": idx(train_g), "val": idx(val_g), "test": idx(test_g)}
    # Guarantee no group appears in two splits.
    assert train_g.isdisjoint(val_g) and train_g.isdisjoint(test_g) and val_g.isdisjoint(test_g)
    return splits


def create_temporal_group_splits(
    group_ids, order_keys, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> dict[str, np.ndarray]:
    """Grouped split ordered by time: oldest train, newest test — Section 7.7.2.

    `create_group_splits` shuffles groups with a seed, which asks "does this
    model generalize to a run it has not seen?". This asks a different and
    harder question: "does it still work next month?". Focus data drifts for
    reasons that have nothing to do with the model — the focuser is re-greased,
    the mirror is collimated, a tube gets swapped, the seeing season turns — and
    a randomly-shuffled split hides all of it by scattering every night's runs
    across train, val and test.

    Neither split is the right one. The specification requires this one; random
    grouping is the fair test of rig generalization. Report both, and treat a
    large gap between them as the finding it is.

    `order_keys` is a per-row timestamp; a group is placed by its *earliest*
    row, so a run cannot straddle a boundary. ISO-8601 strings sort
    chronologically as text, which is what makes them safe to compare directly
    — but only if they share a timezone. The daemon writes RFC 3339 UTC
    throughout; a mixed-offset column would sort wrongly and silently.
    """
    group_ids = np.asarray(group_ids)
    order_keys = np.asarray(order_keys)
    if group_ids.shape != order_keys.shape:
        raise ValueError(
            f"group_ids {group_ids.shape} and order_keys {order_keys.shape} must align"
        )
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and leave a positive test ratio")

    groups = np.unique(group_ids)
    if groups.size < 3:
        raise ValueError(
            f"grouped train/val/test splitting needs at least 3 groups, got {groups.size}"
        )

    # Earliest row per group. Ties broken by group id so the order is total and
    # a rebuild cannot silently reshuffle two runs that started in the same second.
    # Python's `min` over a list rather than `ndarray.min()`: numpy has no
    # minimum ufunc for unicode dtypes, and ISO-8601 timestamps are exactly the
    # string column this is meant to take.
    first_seen = {
        group: min(order_keys[group_ids == group].tolist())
        for group in groups.tolist()
    }
    ordered = sorted(groups.tolist(), key=lambda g: (first_seen[g], g))

    n = len(ordered)
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * (1.0 - train_ratio - val_ratio))))
    while n_val + n_test >= n:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            raise ValueError("could not allocate non-empty temporal splits")
    n_train = n - n_val - n_test

    assignment = {
        "train": set(ordered[:n_train]),
        "val": set(ordered[n_train : n_train + n_val]),
        "test": set(ordered[n_train + n_val :]),
    }

    def idx(group_set):
        if not group_set:
            return np.array([], dtype=np.int64)
        return np.where(np.isin(group_ids, list(group_set)))[0]

    splits = {name: idx(members) for name, members in assignment.items()}
    assert assignment["train"].isdisjoint(assignment["val"])
    assert assignment["train"].isdisjoint(assignment["test"])
    assert assignment["val"].isdisjoint(assignment["test"])
    return splits


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value):
    return value.item() if isinstance(value, np.generic) else value


#: Ratios per strategy. Temporal uses the 70/15/15 the specification names in
#: Section 7.7.2; random-group keeps the 80/10/10 this pipeline has always used,
#: because it can afford a smaller test set when groups are interchangeable.
DEFAULT_RATIOS = {
    "random_group": (0.8, 0.1),
    "temporal": (0.70, 0.15),
}


def create_splits(
    dataset_path: str,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    seed: int = 0,
    registry_path: str | None = None,
    group_field: str | None = None,
    strategy: str = "random_group",
    order_field: str | None = None,
):
    """Create grouped splits from an .npz dataset and write a registry JSON.

    `strategy` selects what the held-out set is a test *of*:

      * ``random_group`` — groups shuffled by seed. Tests generalization to an
        unseen run. The default, and what every existing registry used.
      * ``temporal`` — groups ordered by time, oldest train / newest test, per
        specification Section 7.7.2. Tests whether the model survives drift.

    Report both. A model that scores well under one and badly under the other
    has told you something specific, and averaging them away would waste it.
    """
    if strategy not in DEFAULT_RATIOS:
        raise ValueError(
            f"unknown split strategy {strategy!r}; expected one of {sorted(DEFAULT_RATIOS)}"
        )
    default_train, default_val = DEFAULT_RATIOS[strategy]
    train_ratio = default_train if train_ratio is None else train_ratio
    val_ratio = default_val if val_ratio is None else val_ratio

    data = np.load(dataset_path, allow_pickle=False)
    group_field = group_field or ("split_group" if "split_group" in data else "run_id")
    if group_field not in data:
        raise ValueError(f"dataset must contain grouping field {group_field!r}")
    groups = np.asarray(data[group_field])

    registry = {
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "n_samples": int(groups.size),
        "group_field": group_field,
        "strategy": strategy,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": 1 - train_ratio - val_ratio},
    }

    if strategy == "temporal":
        order_field = order_field or "captured_at"
        if order_field not in data:
            raise ValueError(
                f"temporal splitting needs a timestamp column; dataset has no {order_field!r}. "
                f"Available: {sorted(data.files)}"
            )
        order_keys = np.asarray(data[order_field])
        splits = create_temporal_group_splits(groups, order_keys, train_ratio, val_ratio)
        registry["order_field"] = order_field
        # The boundaries are the interpretable part of a temporal split: "test
        # is every run after this timestamp" is a sentence you can put in a
        # results table, unlike a list of forty run UUIDs.
        registry["boundaries"] = {
            name: {
                "first": _json_value(min(order_keys[index].tolist())),
                "last": _json_value(max(order_keys[index].tolist())),
            }
            for name, index in splits.items()
            if index.size
        }
        # A seed plays no part in a deterministic ordering. Recording one would
        # imply the split could be re-rolled, and it cannot.
        registry["seed"] = None
    else:
        splits = create_group_splits(groups, train_ratio, val_ratio, seed)
        registry["seed"] = seed

    registry["counts"] = {k: int(v.size) for k, v in splits.items()}
    registry["groups"] = {
        name: [_json_value(value) for value in np.unique(groups[index])]
        for name, index in splits.items()
    }
    registry["indices"] = {k: v.tolist() for k, v in splits.items()}

    out = Path(registry_path) if registry_path else Path(dataset_path).with_suffix(".splits.json")
    out.write_text(json.dumps(registry, indent=2))
    print(f"✓ Wrote {out}  ({strategy}, {registry['counts']})")
    return splits


def load_split(registry_path: str, split_name: str) -> np.ndarray:
    """Load a previously registered split index array."""
    registry = json.loads(Path(registry_path).read_text())
    return np.asarray(registry["indices"][split_name], dtype=np.int64)


def load_registered_splits(
    registry_path: str, dataset_path: str, group_values
) -> dict[str, np.ndarray]:
    """Load a frozen registry and reject stale/reordered datasets loudly."""
    registry = json.loads(Path(registry_path).read_text())
    groups = np.asarray(group_values)
    if registry.get("dataset_sha256") != _sha256(dataset_path):
        raise ValueError("split registry dataset hash does not match the dataset file")
    if registry.get("n_samples") != int(groups.size):
        raise ValueError("split registry sample count does not match the dataset")
    splits = {
        name: np.asarray(registry["indices"][name], dtype=np.int64)
        for name in ("train", "val", "test")
    }
    if any(index.size == 0 for index in splits.values()):
        raise ValueError("split registry contains an empty train/val/test split")
    split_groups = {name: set(groups[index].tolist()) for name, index in splits.items()}
    if not (
        split_groups["train"].isdisjoint(split_groups["val"])
        and split_groups["train"].isdisjoint(split_groups["test"])
        and split_groups["val"].isdisjoint(split_groups["test"])
    ):
        raise ValueError("split registry leaks a group across train/val/test")
    return splits


def splits_for_dataset(
    dataset_path: str,
    data,
    *,
    seed: int = 0,
    registry_path: str | None = None,
) -> dict[str, np.ndarray]:
    """Use the adjacent frozen registry when present; create it once otherwise."""
    group_field = "split_group" if "split_group" in data else "run_id"
    path = Path(registry_path) if registry_path else Path(dataset_path).with_suffix(".splits.json")
    if path.exists():
        return load_registered_splits(str(path), dataset_path, data[group_field])
    return create_splits(
        dataset_path,
        seed=seed,
        registry_path=str(path),
        group_field=group_field,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--registry")
    parser.add_argument("--group-field")
    parser.add_argument(
        "--strategy",
        choices=sorted(DEFAULT_RATIOS),
        default="random_group",
        help=(
            "random_group: shuffled groups, tests generalization to an unseen "
            "run. temporal: oldest-to-newest, tests survival of drift "
            "(spec 7.7.2). Build both and report both."
        ),
    )
    parser.add_argument(
        "--order-field",
        default=None,
        help="Timestamp column for --strategy temporal (default: captured_at).",
    )
    args = parser.parse_args()
    create_splits(
        args.dataset,
        seed=args.seed,
        registry_path=args.registry,
        group_field=args.group_field,
        strategy=args.strategy,
        order_field=args.order_field,
    )
