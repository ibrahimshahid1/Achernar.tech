"""Builds the 38-feature vector for the Focus Offset Predictor — Section 7.6.2.

The layout is **not** defined here. It lives in `contracts.fop_features`, which
is the single authority for names, order, normalization and provenance; this
module only assembles numbers into that layout. The names previously hard-coded
here were a fourth, wrong description of the vector (15 entries, features like
`altitude` and `seeing_estimate` that nothing in the daemon produces) and have
been removed rather than corrected — there is exactly one list now.

The builder itself is deliberately not implemented yet. Seven of the thirty-eight
features are `MISSING` and fourteen are `DEGENERATE` (run
`python -m contracts.fop_features` for the current count), so anything written
today would encode assumptions about columns that do not yet hold data. Writing
it against real rows, once the focuser exists and sweeps are landing, is cheaper
and far less likely to bake in a wrong join than writing it now and revising it.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.fop_features import (  # noqa: E402
    FEATURE_INDEX,
    FEATURE_NAMES,
    N_FEATURES,
    unavailable,
)

__all__ = ["FEATURE_NAMES", "FEATURE_INDEX", "N_FEATURES", "build_features",
           "build_feature_matrix", "empty_vector"]


def empty_vector() -> np.ndarray:
    """A correctly-shaped, all-zero vector in contract order.

    Zero is the right filler for the blocks that are structurally absent (an
    unset one-hot, a rig with no measured backlash). It is *not* a safe filler
    for a missing measurement, which is why the builder masks rather than
    imputes — see the note on `sensor_temp_norm` in the contract.
    """
    return np.zeros(N_FEATURES, dtype=np.float32)


def _not_yet(what: str):
    blocked = ", ".join(f.name for f in unavailable())
    return NotImplementedError(
        f"{what} is not implemented yet. The contract in contracts/fop_features.py "
        f"currently reports these features as unusable:\n  {blocked}\n"
        "Implement this against real focus_samples rows once the focuser lands, "
        "so the joins can be checked rather than guessed."
    )


def build_features(sample: dict) -> np.ndarray:
    """Build one 38-element feature vector from a focus sample row."""
    raise _not_yet("build_features")


def build_feature_matrix(samples: list[dict]) -> np.ndarray:
    """Build an (N, 38) feature matrix from ordered focus sample rows."""
    raise _not_yet("build_feature_matrix")
