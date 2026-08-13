"""The 38-feature input contract for the Focus Offset Predictor — spec Section 7.6.2.

This module is the single authority on what the FOP consumes. Before it existed
there were three disagreeing descriptions in the tree:

  * spec Section 7.6.2 — 38 features in 9 groups, the intended design;
  * ``models/focus_offset/0.3.0/metadata.json`` — a 10-feature list that matched
    nothing and named features (``altitude``, ``quality_score``) the daemon does
    not produce;
  * ``achernar-daemon/src/focus/focus_features.rs`` — a 15-field struct plus a
    ``raw: Vec<f64>`` escape hatch and a ``todo!()`` tensor builder.

That is the same failure mode the FQE hit, and it is worse here. The FQE's input
is *pixels*: if the contract drifts, the crops are still on disk and the dataset
can be rebuilt. The FOP's input is **derived state that exists only at the moment
of capture**. A feature that was not logged during a sweep cannot be recovered
from the sweep afterwards, at any cost, ever. So the contract has to be settled
before the first run rather than after the first training failure.

Accordingly this module records more than names and indices. For every feature it
records **provenance** — the exact column or JSON key that supplies it — and an
:class:`Availability` verdict describing whether the system can actually produce
it today. Nine of the thirty-eight cannot be produced right now, and the reasons
are tracked here rather than discovered during training.

Usage::

    from contracts.fop_features import FEATURES, FEATURE_NAMES, unavailable

    len(FEATURES)                      # 38, enforced at import
    FEATURE_NAMES[:3]                  # ('predicted_hfr', 'sharpness_score', ...)
    [f.name for f in unavailable()]    # what to drop before training

The committed ``fop_features_v1.json`` beside this file is generated from these
declarations by ``write_contract_json``; ``ml/tests/test_fop_contract.py`` fails
if the two drift apart. That JSON is the artifact the Rust runtime will read when
the daemon starts building the vector online, so the two sides cannot diverge
silently the way the FQE's did.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

CONTRACT_VERSION = "1.0.0"
N_FEATURES = 38

#: Where the generated machine-readable form is committed.
CONTRACT_JSON = Path(__file__).with_name("fop_features_v1.json")


class Availability(str, Enum):
    """Whether the running system can actually supply a feature.

    The distinction that matters for training is not "is this field in the
    schema" but "will this column contain something that varies". A constant
    column is worse than a missing one: it trains to a weight of roughly zero,
    exports into the ONNX graph as a live input, and then silently misleads the
    first rig that *can* vary it.
    """

    #: Logged today, varies today. Safe to train on.
    AVAILABLE = "available"

    #: The column exists and is written, but holds one value on every row for a
    #: known reason (no filter wheel, monotone sweeps). Drop the block before
    #: training and record the drop in ``norm_params.json``.
    DEGENERATE = "degenerate"

    #: Nothing writes it. Needs a code or schema change before the first sweep,
    #: because it cannot be backfilled from data collected without it.
    MISSING = "missing"

    #: Deliberately removed upstream and not coming back. The slot is held at a
    #: constant so the vector keeps its documented width and indices, but it
    #: carries no information and must be dropped alongside the degenerate ones.
    REJECTED = "rejected"


@dataclass(frozen=True)
class FeatureSpec:
    """One slot in the FOP input vector."""

    index: int
    name: str
    group: str
    #: Normalization, verbatim from the spec Section 7.6.2 table.
    norm: str
    #: What the number means, in its natural units, before normalization.
    meaning: str
    #: The column, JSON key, or derivation that supplies it. Written as it would
    #: be read from SQLite so the dataset builder can be checked against it.
    source: str
    availability: Availability
    #: Why it is not AVAILABLE, and what would make it so. Empty when available.
    note: str = ""


def _f(*args, **kwargs) -> FeatureSpec:
    return FeatureSpec(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# The vector. Order is the contract: index N here is column N of the matrix and
# input N of the ONNX graph, forever. Append-only — never reorder, never reuse a
# retired slot, because a trained artifact on disk has these positions baked in.
# ─────────────────────────────────────────────────────────────────────────────

FEATURES: tuple[FeatureSpec, ...] = (
    # ── Quality (3) — the FQE's view of this frame ───────────────────────────
    _f(
        0, "predicted_hfr", "quality", "z-score",
        "FQE-predicted frame HFR (px)",
        "frames.quality_json -> fqe_hfr_px",
        Availability.AVAILABLE,
        "Written by autofocus_quality_json since the write path was widened; "
        "SharedObservation carried it all along and persist_observed_frame used "
        "to drop it, so every frame captured before that fix has a null here. "
        "Null also when no FQE model is loaded — mask on it, do not impute.",
    ),
    _f(
        1, "sharpness_score", "quality", "z-score",
        "held at 0.0 — see note",
        "(none)",
        Availability.REJECTED,
        "vision/shared_observation.rs removed focus_quality / sharpness_score / "
        "frame_quality_score on the grounds that they were 'three views of one "
        "HFR value with no calibrated meaning'. That reasoning is right and the "
        "spec's slot is redundant with predicted_hfr. The slot is held at a "
        "constant to preserve indices 2..37 rather than renumbering a published "
        "contract.",
    ),
    _f(
        2, "quality_uncertainty", "quality", "z-score",
        "predictive sigma of log-HFR",
        "frames.quality_json -> fqe_sigma_log_hfr (preferred), "
        "else quality_uncertainty",
        Availability.AVAILABLE,
        "Persisted alongside predicted_hfr. The classical path's own "
        "quality_uncertainty is documented as always None, so fqe_sigma_log_hfr "
        "is the real source and it exists only while FQE runs in shadow. "
        "fqe_calibrated says whether the sigma is a fitted predictive interval "
        "or raw between-star scatter; the two are not interchangeable.",
    ),

    # ── Classical (4) — the measurement the model must beat ──────────────────
    _f(
        3, "classical_hfr", "classical", "log",
        "SEP half-flux radius (px)",
        "focus_samples.hfr",
        Availability.AVAILABLE,
    ),
    _f(
        4, "classical_fwhm", "classical", "log",
        "fitted PSF FWHM (px)",
        "focus_samples.fwhm",
        Availability.MISSING,
        "The column exists and is always NULL: focus_agent.rs states fwhm stays "
        "None 'unless a Gaussian/Moffat fit was actually performed', and none "
        "is. focus_samples.hfd is populated but is 2*HFR, deliberately named so "
        "it is never mistaken for a fitted FWHM — substituting it would feed the "
        "model classical_hfr twice under two names.",
    ),
    _f(
        5, "fit_quality", "classical", "z-score",
        "r^2 of the focus curve fit over the samples so far",
        "derived: ClassicalFocusSolver over samples with captured_at <= this row",
        Availability.MISSING,
        "PARITY HAZARD. The daemon fits the curve once, at the end of the run, "
        "and does not persist r^2. Training must therefore recompute it as a "
        "*partial* fit over only the samples preceding each row — using the "
        "whole-run fit would leak the answer, since the fit optimum is the "
        "label. Whatever partial-fit rule training adopts, focus_inference must "
        "use the identical one online.",
    ),
    _f(
        6, "star_count", "classical", "log",
        "stars detected in the frame",
        "focus_samples.star_count",
        Availability.AVAILABLE,
    ),

    # ── Position (5) — where the focuser is within this run ──────────────────
    _f(
        7, "pos_norm", "position", "p / p_max",
        "absolute focuser position, normalized to travel",
        "focus_samples.position / focuser_profiles.max_position",
        Availability.AVAILABLE,
    ),
    _f(
        8, "dist_from_start_norm", "position", "/ max step",
        "signed steps from where the run began",
        "focus_samples.position - focus_runs.start_position",
        Availability.AVAILABLE,
    ),
    _f(
        9, "sample_idx_norm", "position", "as defined",
        "ordinal of this sample within the run, normalized",
        "rank of focus_samples.captured_at within run_id",
        Availability.AVAILABLE,
    ),
    _f(
        10, "dir_encoded", "position", "as defined",
        "-1 inward / 0 first sample / +1 outward",
        "sign(position - previous position within run)",
        Availability.AVAILABLE,
        "Varies only across runs that actually reverse. The agent grew a "
        "SweepPattern::Reversal ordering, so the capability exists; whether the "
        "column varies is now a collection-practice question. The plan calls "
        "for roughly one run in three — check the mix before training rather "
        "than assuming it.",
    ),
    _f(
        11, "run_initial_hfr", "position", "log",
        "HFR of the run's first sample (px)",
        "first focus_samples.hfr by captured_at within run_id",
        Availability.AVAILABLE,
    ),

    # ── Temperature (4) — the thermal state the drift model shares ───────────
    _f(
        12, "ambient_temp_norm", "temperature", "z-score",
        "outside air temperature (degC)",
        "environment_log.ambient_temp_c, nearest observed_at within the night; "
        "join focus_runs.night_id = environment_log.night_id",
        Availability.AVAILABLE,
        "Joinable since migration 007 put night_id on focus_runs — before that "
        "the observer's readings went into a table nothing could reach from a "
        "run. Present only for nights the observer actually logged, which no "
        "code can guarantee. Ambient is not substitutable by sensor "
        "temperature: a DSLR die self-heats through a session, so the two "
        "diverge in exactly the direction that matters most.",
    ),
    _f(
        13, "sensor_temp_norm", "temperature", "z-score",
        "camera-reported temperature (degC)",
        "focus_samples.temperature, provenance in quality_json.temperature_state",
        Availability.AVAILABLE,
        "Populated with an explicit temperature_state of measured/estimated/"
        "missing beside it. Rows whose state is not 'measured' should be masked, "
        "not imputed.",
    ),
    _f(
        14, "temp_delta_from_run_start", "temperature", "z-score",
        "degC change since the run's first sample",
        "focus_samples.temperature - first temperature within run_id",
        Availability.AVAILABLE,
    ),
    _f(
        15, "temp_coeff_norm", "temperature", "z-score",
        "the rig's fitted alpha (steps/degC)",
        "focuser_profiles.temp_coefficient",
        Availability.MISSING,
        "NULL until the FDF fits it, which needs a thermal night, which needs "
        "the focuser. Expected to fill in naturally; not a code defect.",
    ),

    # ── Filter (7) — one-hot, all seven dead without a wheel ─────────────────
    *(
        _f(
            16 + i, f"filter_{name.lower()}", "filter", "one-hot",
            f"1.0 when the run used {name}",
            "focus_runs.filter_name",
            Availability.DEGENERATE,
            "D4: no filter wheel exists and hardware/filter_wheel.rs is two "
            "todo!() calls, so the whole block is constant. The data-collection "
            "document is explicit that this block must be dropped before "
            "training and the drop recorded in norm_params.json — otherwise "
            "seven dead inputs enter the ONNX graph and the first rig with a "
            "wheel silently misuses them.",
        )
        for i, name in enumerate(("L", "R", "G", "B", "Ha", "OIII", "SII"))
    ),

    # ── History (6) — what the last three moves did ──────────────────────────
    *(
        _f(
            23 + i, f"prev_delta_{i + 1}", "history", "/ max step",
            f"signed steps moved {i + 1} sample(s) ago",
            "difference of consecutive focus_samples.position within run_id",
            Availability.AVAILABLE,
            "A monotone sweep makes all three equal to the constant step size; "
            "a reversal run varies their sign. Available since the agent gained "
            "SweepPattern::Reversal, subject to the same collection-mix caveat "
            "as dir_encoded.",
        )
        for i in range(3)
    ),
    *(
        _f(
            26 + i, f"prev_hfr_delta_{i + 1}", "history", "/ max step",
            f"change in HFR {i + 1} sample(s) ago (px)",
            "difference of consecutive focus_samples.hfr within run_id",
            Availability.AVAILABLE,
        )
        for i in range(3)
    ),

    # ── Direction (3) — the backlash-aware part of the state ─────────────────
    _f(
        29, "dir_changed_recently", "direction", "bool",
        "1.0 if the focuser reversed within the last 3 samples",
        "derived from the sign sequence of position differences",
        Availability.AVAILABLE,
        "This is the feature the direction head learns from, so a dataset with "
        "no reversals cannot train the head at all — not merely train it "
        "poorly. SweepPattern::Reversal produces exactly one flip per run, "
        "asserted by reversal_sweep_changes_direction_exactly_once. Verify the "
        "collected mix before training: the capability existing is not the same "
        "as the nights having used it.",
    ),
    _f(
        30, "steps_since_flip_norm", "direction", "/ max step",
        "steps travelled since the last direction change",
        "derived from the sign sequence of position differences",
        Availability.AVAILABLE,
        "Same source as dir_changed_recently; with no flip in a run it counts "
        "the whole run every time.",
    ),
    _f(
        31, "consec_improvements_norm", "direction", "as defined",
        "consecutive samples whose HFR improved, normalized",
        "derived from the sign sequence of focus_samples.hfr differences",
        Availability.AVAILABLE,
    ),

    # ── Rig params (3) — constants of the instrument ─────────────────────────
    _f(
        32, "backlash_norm", "rig", "/ max step",
        "measured backlash (steps)",
        "focuser_profiles.backlash_steps",
        Availability.AVAILABLE,
        "Defaults to 0 until BP-5 measures it; 0 is a truthful default here "
        "because an unmeasured focuser is compensated as if it had none.",
    ),
    _f(
        33, "cfz_norm", "rig", "/ max step",
        "critical focus zone (steps)",
        "derived: 4.88 * lambda * rigs.focal_ratio^2 "
        "* focuser_profiles.steps_per_micron",
        Availability.AVAILABLE,
        "steps_per_micron landed in migration 005 and is measurable from a "
        "V-curve wing (focus_drift::steps_per_micron_from_vcurve), so this is "
        "derivable without new plumbing once a rig row exists.",
    ),
    _f(
        34, "filter_offset_norm", "rig", "/ max step",
        "per-filter focus offset (steps)",
        "focuser_filter_offsets.offset_steps",
        Availability.DEGENERATE,
        "D4 again: no wheel, so no offsets, so constant 0.",
    ),

    # ── Sky (3) — conditions the offset has to survive ───────────────────────
    _f(
        35, "background_norm", "sky", "z-score",
        "sky background level (ADU)",
        "frames.quality_json -> background_adu",
        Availability.AVAILABLE,
        "Already written per autofocus frame and joinable through "
        "focus_samples.frame_id.",
    ),
    _f(
        36, "saturation_frac", "sky", "0-1",
        "fraction of detected stars that clipped",
        "frames.quality_json -> saturation_fraction",
        Availability.AVAILABLE,
        "Only meaningful when quality_json.saturation_gate_applied is true. On a "
        "body with no declared saturation_adu this reads 0.0 and means 'never "
        "tested', not 'clean' — the same D1 trap the FQE dataset builder now "
        "refuses. Mask on the gate flag, do not read the value alone.",
    ),
    _f(
        37, "humidity_norm", "sky", "0-1",
        "relative humidity (%)",
        "environment_log.humidity_pct, nearest observed_at within the night; "
        "join focus_runs.night_id = environment_log.night_id",
        Availability.AVAILABLE,
        "Same join as ambient_temp_norm, opened by migration 007, and the same "
        "caveat: recorded only when the observer wrote it on the card.",
    ),
)

#: Group sizes exactly as the spec Section 7.6.2 table gives them. Checked at
#: import, so a mis-typed feature cannot quietly land in the wrong group.
GROUP_SIZES: dict[str, int] = {
    "quality": 3,
    "classical": 4,
    "position": 5,
    "temperature": 4,
    "filter": 7,
    "history": 6,
    "direction": 3,
    "rig": 3,
    "sky": 3,
}

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
FEATURE_INDEX: dict[str, int] = {f.name: f.index for f in FEATURES}


# ─────────────────────────────────────────────────────────────────────────────
# The label. Recorded here because it is as easy to get wrong as the inputs, and
# because its retrospective nature is the reason aborted runs are worthless.
# ─────────────────────────────────────────────────────────────────────────────

LABEL = {
    "name": "delta_steps",
    "definition": "focus_runs.best_position - focus_samples.position",
    "units": "focuser steps, signed (positive = move outward to reach focus)",
    "retrospective": True,
    "note": (
        "best_position exists only once the run converges, so a run that failed "
        "or was aborted produces zero labelled rows no matter how many samples "
        "it captured. It is still worth persisting those samples — they carry "
        "valid FQE crops — but the FOP builder must exclude runs whose state is "
        "not Converged, and must never fall back to argmin(hfr) as a stand-in: "
        "the fitted optimum is interior to the sample grid and the argmin is not."
    ),
    "group": "focus_runs.id",
    "secondary_group": "focus_runs.night_id",
}

#: Balanced-sampler buckets over |delta_steps|, spec Section 7.6.2 Listing 12.
#: Recorded here because collection has to fill them: sweeps that always start
#: near focus leave the top two empty, and no amount of resampling invents rows.
OFFSET_BUCKETS = (0, 5, 15, 40, 120, float("inf"))


# ─────────────────────────────────────────────────────────────────────────────
# Queries over the contract
# ─────────────────────────────────────────────────────────────────────────────


def by_group(group: str) -> tuple[FeatureSpec, ...]:
    """Every feature in `group`, in vector order."""
    if group not in GROUP_SIZES:
        raise KeyError(f"unknown feature group {group!r}; expected one of {sorted(GROUP_SIZES)}")
    return tuple(f for f in FEATURES if f.group == group)


def unavailable() -> tuple[FeatureSpec, ...]:
    """Features that carry no usable signal today.

    This is the list a trainer must drop, and the list that shrinks as the
    blockers close. It is deliberately one function rather than a hand-kept
    constant so it cannot fall out of date with the declarations above.
    """
    return tuple(f for f in FEATURES if f.availability is not Availability.AVAILABLE)


def trainable_indices() -> tuple[int, ...]:
    """Indices of the features worth handing to a model right now."""
    return tuple(f.index for f in FEATURES if f.availability is Availability.AVAILABLE)


def blockers() -> dict[str, tuple[str, ...]]:
    """Availability verdict -> the feature names sitting behind it."""
    out: dict[str, list[str]] = {}
    for f in FEATURES:
        if f.availability is not Availability.AVAILABLE:
            out.setdefault(f.availability.value, []).append(f.name)
    return {k: tuple(v) for k, v in out.items()}


def as_dict() -> dict:
    """The whole contract, in the shape written to `fop_features_v1.json`."""
    return {
        "contract_version": CONTRACT_VERSION,
        "spec_section": "7.6.2",
        "n_features": N_FEATURES,
        "group_sizes": GROUP_SIZES,
        "features": [asdict(f) | {"availability": f.availability.value} for f in FEATURES],
        "label": LABEL,
        "offset_buckets": [b if b != float("inf") else None for b in OFFSET_BUCKETS],
    }


def write_contract_json(path: Path | str | None = None) -> Path:
    """Regenerate the committed JSON form. Run after editing the declarations."""
    target = Path(path) if path is not None else CONTRACT_JSON
    target.write_text(json.dumps(as_dict(), indent=2) + "\n")
    return target


def _validate() -> None:
    """Structural invariants, checked at import so a bad edit fails loudly.

    Cheap, and it catches the whole class of mistake that produced the three
    disagreeing descriptions this module replaces.
    """
    if len(FEATURES) != N_FEATURES:
        raise AssertionError(f"contract declares {len(FEATURES)} features, expected {N_FEATURES}")

    for expected, feature in enumerate(FEATURES):
        if feature.index != expected:
            raise AssertionError(
                f"feature {feature.name!r} has index {feature.index}, expected {expected}; "
                "indices must be contiguous and in declaration order"
            )

    if len(set(FEATURE_NAMES)) != len(FEATURE_NAMES):
        duplicates = sorted({n for n in FEATURE_NAMES if FEATURE_NAMES.count(n) > 1})
        raise AssertionError(f"duplicate feature names: {duplicates}")

    counted: dict[str, int] = {}
    for feature in FEATURES:
        counted[feature.group] = counted.get(feature.group, 0) + 1
    if counted != GROUP_SIZES:
        raise AssertionError(
            f"group sizes {counted} do not match spec Section 7.6.2 table {GROUP_SIZES}"
        )

    if sum(GROUP_SIZES.values()) != N_FEATURES:
        raise AssertionError("GROUP_SIZES does not sum to N_FEATURES")

    for feature in FEATURES:
        if feature.availability is not Availability.AVAILABLE and not feature.note:
            raise AssertionError(
                f"{feature.name!r} is {feature.availability.value} but gives no reason; "
                "an unavailable feature must say what would make it available"
            )


_validate()


if __name__ == "__main__":
    written = write_contract_json()
    print(f"✓ wrote {written}")
    print(f"  {len(trainable_indices())}/{N_FEATURES} features are usable today")
    for verdict, names in sorted(blockers().items()):
        print(f"  {verdict:11s} ({len(names):2d}): {', '.join(names)}")
