"""
fqe_benchmark — classical development benchmark for the FQE public-data experiment (contract-v1, M2).

Every comparator scores the declared development population of each source
(contract ``development_populations``: stellar_sim_v1 val rows, target
``reference_hfr_px``; hst_acs_corpus_v1 strict val rows, target ``hfr_px``)
and emits one row per population member, including the rows it could not
measure. Rows are loaded only through ``data_pipeline.fqe_sources``; the sealed
test partition is never requested (``--partition test`` is refused) and the
public NPZs are never ``np.load``-ed here.

Comparators (contract ``classical_comparators``; settings are declared and
frozen, nothing is tuned on development results):

``constant__<arm>``
    The median target of the arm's ``train_fit`` rows (calibration groups
    excluded), predicted for every val row. Arms: ``synthetic_only``
    (stellar_sim_v1), ``hst_only`` (hst_acs_corpus_v1) and ``mixed`` (the
    weighted median under the contract's source-balanced sampling weights,
    from ``fqe_sources.make_mixture``).
``normalized_input__direct_hfr``
    ``eval.baselines.direct_hfr_measure_detailed`` (custom crop
    iterative-clip sky + signed curve of growth) on exactly the CNN's
    normalized 64x64 inputs. Unmeasured rows carry the reason
    (``bg_unresolved``, ``no_core_above_3sigma``, ``cog_none``).
``sharpness__<score>__<arm>``
    ``tenengrad`` and ``normalized_variance`` (``eval.baselines``; the only
    sharpness operators in the repository) on the normalized crops, mapped
    to HFR by the declared monotone ``loglog_ols_v1`` fit
    ``ln HFR = a + b ln score`` on the arm's ``train_fit`` rows only. Sweep /
    rank metrics are ``unavailable`` (singleton archive rows).
``sep_flux_radius__...``
    Declared measurement-reference comparators, not deployable estimators.
    stellar_sim_v1: ``sep.flux_radius`` (frac 0.5, subpix 5, r = 24 px) on
    the native noisy observation minus the true background and the 500 e
    bias, at the true centre (an oracle: it uses simulated truth; recomputed
    from the native crop and checked against the stored ``measured_hfr_px``).
    hst_acs_corpus_v1: the teacher's own chain reproduced on the parent
    validation frames (stored SEP background map, mask, positions and
    ``rmax``): agreement is identity, not independent truth.
``native_rust__<method>__<context>__<mask policy>``
    The daemon's classical pipeline (background -> detection -> star gates
    -> curve-of-growth HFR) through ``eval.bge.bridge`` on NATIVE linear
    pixels, method ids of docs/bge-v2-spec.md section 4 only
    (``rust_legacy__g1`` = legacy global clip + peak_count_v1,
    ``rust_legacy__g2``, ``rust_sep__n1g1``, ``rust_sep__n2g2``). Contexts:
    ``sim_observation64`` (the isolated synthetic observation is the frame),
    ``hst_crop64`` (an isolated 64x64 native crop of an HST frame: not a
    daemon context), ``hst_parent_frame`` (the full SCI1 validation frame:
    the daemon's end-to-end selection population with its 48-examined /
    16-used caps and gates). SEP methods on 64x64 crops are ``unsupported``
    (frame smaller than the 2 x 64 px mesh grid), never ``failed``.

Row statuses: ``measured`` (finite positive width), ``unmeasured`` (the
method ran and produced no width for this row: reason given), ``unsupported``
(the method cannot run in this context), ``failed`` (infrastructure or join
failure). Only ``measured`` rows count toward coverage; an unmeasured row is
never a zero width.

Run (from ``ml/``)::

    python -B -m eval.fqe_benchmark --partition val \
        --sources stellar_sim_v1,hst_acs_corpus_v1 --out <dir> [--methods ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from data_pipeline import fqe_sources  # noqa: E402
from data_pipeline.fqe_sources import SealedPartitionError  # noqa: E402
from eval import baselines  # noqa: E402

SCHEMA = "fqe-classical-benchmark-v1"
ROW_SCHEMA = "fqe-classical-row-v1"
UNAVAILABLE = baselines.UNAVAILABLE
STATUSES = ("measured", "unmeasured", "unsupported", "failed")
DEV_EVAL_PARTITIONS = ("val",)
FIT_PARTITION = "train_fit"
STAGES = ("background", "extraction", "acceptance", "width")
FAMILIES = ("constant", "normalized_input", "sharpness", "sep_flux_radius", "native_rust")
POPULATION_PER_CROP = "per_crop_width"
POPULATION_END_TO_END = "end_to_end_selection"

#: Arm names of the contract's training conditions (any other source: "<source>_only").
ARM_OF_SOURCE = {"stellar_sim_v1": "synthetic_only", "hst_acs_corpus_v1": "hst_only"}
MIXED_ARM = "mixed"

#: Rust method ids run by default (docs/bge-v2-spec.md section 4). The plain
#: backend strings ``legacy_global_clip``/``sep`` run the daemon's current
#: defaults and are never accepted here.
DEFAULT_RUST_METHODS = ("rust_legacy__g1", "rust_legacy__g2", "rust_sep__n1g1", "rust_sep__n2g2")
SHARPNESS_SCORES = ("tenengrad", "normalized_variance")
MAPPING_ID = "loglog_ols_v1"

#: Declared, frozen settings (recorded in config.json). Bins are half-open
#: [lo, hi); ``None`` is an open upper edge.
FROZEN = {
    "width_bins_px": [0.0, 0.7, 1.0, 1.5, 2.5, 4.0, None],
    "star_snr_bins": [0.0, 20.0, 50.0, 100.0, 300.0, None],
    "categorical_strata": ["profile", "motion", "blended", "edge_case", "night_id", "aperture_at_floor",
                           "neighbour_within_24px"],
    "group_stratum_max_groups": 20,
    "match": {"sim_radius_px": 1.5, "hst_radius_px": 1.0, "ambiguity_radius_px": 3.0,
              "rule": "mutual nearest neighbour between references and used stars; ambiguous when another "
                      "used star lies within ambiguity_radius of the reference or another reference within "
                      "ambiguity_radius of the star"},
    "rust_caps": {"max_stars_examined": 48, "max_stars_used": 16,
                  "source": "achernar-daemon/src/vision/quality.rs MAX_STARS_EXAMINED / MAX_STARS_USED "
                            "(recorded, not set)"},
    "sim_saturation_adu": 65535.0,
    "sim_saturation_note": "the bridge's legacy peak gate is 0.98 * saturation_adu = 64224.3 e, the importer's "
                           "saturation_numeric_level; passing 64224 would apply 0.98 twice",
    "hst_saturation_adu": None,
    "hst_saturation_note": "electrons: no container ceiling applies, so the peak gate is not performed; native "
                           "saturation comes from the DQ mask",
    "hst_native_saturation_dq_bits": 256 | 2048,
    "hst_neighbour_radius_px": 24.0,
    "sim_oracle_radius_px": 24.0,
    "sep_flux_radius": {"frac": 0.5, "subpix": 5},
    "mapping": {"id": MAPPING_ID, "definition": "ln(HFR_px) = a + b * ln(score), weighted least squares on the "
                                                "arm's train_fit rows with finite positive score; monotone by "
                                                "construction; mixed arm uses the source-balanced weights"},
    "constant": "weighted median of the arm's train_fit targets (equal weights within a source; mixed arm: "
                "source-balanced weights 1/(k * n_source)); ties at exactly half the weight average the two "
                "neighbouring values (np.median for equal weights)",
    "native_timing": "bridge timing_ms.background + timing_ms.quality of a release test build on this host "
                     "(trailing excluded); one process per manifest",
}

INPUT_NORMALIZED = ("fqe.npz crops as loaded by fqe_sources (the CNN input): 64x64 float32, per-crop NumPy "
                    "1/99 percentile stretch clipped to [0, 1]")
INPUT_NONE = "no image input (a constant fitted on train_fit targets)"
INPUT_SIM_NATIVE = ("stellar_sim_v1 observations.npz raw_crops[array_index]: native 64x64 electrons + 500 bias, "
                    "gain 1, rounded and clipped at 65535; masks from pixel_flags (bit 1 invalid, bit 2 "
                    "saturated)")
INPUT_HST_CROP = ("hst_acs_corpus_v1 exposures/<frame>/observations.npz raw_crops64[observation_index] = "
                  "frame.npz science_electrons[y0:y0+64, x0:x0+64] (calibrated FLT electrons)")
INPUT_HST_FRAME = ("hst_acs_corpus_v1 exposures/<frame>/frame.npz science_electrons, full 2048x4096 SCI1 "
                   "validation frame (calibrated FLT electrons)")
MASK_POLICIES = {
    "pixel_flags": "invalid = pixel_flags bit 1, native_saturation = pixel_flags bit 2 (sim importer flags)",
    "teacher_dq": "invalid = ~valid_mask (DQ != 0 | non-finite | ERR <= 0, the teacher's mask; non-finite "
                  "pixels set to 0 and masked), native_saturation = DQ & (256 | 2048)",
    "imputed_saturation_only": "nondaemon_mask_policy: pixels = the teacher's background-imputed crop (masked "
                               "pixels replaced by the stored SEP background map, i.e. the CNN's "
                               "pre-normalization image); only native_saturation = DQ & (256 | 2048) is masked",
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _sha256_obj(obj) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def _clean(obj):
    """Replace non-finite floats with None so outputs are strict JSON."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def _strip(uid: str) -> str:
    return str(uid).split(":", 1)[1] if ":" in str(uid) else str(uid)


def _width_or_none(value) -> float | None:
    """A finite positive width, else None (never 0 for a missing width)."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _membership_sha256(uids) -> str:
    return hashlib.sha256("\n".join(sorted(str(u) for u in uids)).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Comparator specs and rows
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ComparatorSpec:
    comparator_id: str
    family: str
    method: str
    processing_mode: str
    input_definition: str
    measurement_role: str
    population: str = POPULATION_PER_CROP
    timing_scope: str = "row"
    arm: str | None = None
    context: str | None = None
    mask_policy: str | None = None
    definition: str = ""
    settings: dict = field(default_factory=dict)

    def describe(self) -> dict:
        out = asdict(self)
        out["settings_sha256"] = _sha256_obj(self.settings)
        return out


def _stage(status: str, reason=None, **detail) -> dict:
    out = {"status": status, "reason": reason}
    out.update({k: v for k, v in detail.items() if v is not None})
    return out


def _na_stages() -> dict:
    return {name: _stage("not_applicable") for name in STAGES}


STRATA_KEYS = ("profile", "motion", "blended", "edge_case", "moffat_beta", "night_id",
               "hfr_aperture_radius_px", "masked_aperture_fraction")


def row_strata(arrays, i: int, extra: Mapping | None = None) -> dict:
    out = {"star_snr": float(arrays.star_snr[i]) if np.isfinite(arrays.star_snr[i]) else None}
    for key in STRATA_KEYS:
        if key in arrays.strata:
            value = arrays.strata[key][i]
            out[key] = value.item() if hasattr(value, "item") else value
    if "hfr_aperture_radius_px" in out and out["hfr_aperture_radius_px"] is not None:
        try:
            out["aperture_at_floor"] = bool(float(out["hfr_aperture_radius_px"]) == 6.0)
        except (TypeError, ValueError):
            pass
    for key, value in (extra or {}).items():
        out[key] = value
    return out


def make_row(spec: ComparatorSpec, arrays, i: int, *, prediction, status: str, reason=None,
             timing_ms=None, stages: Mapping | None = None, extra: Mapping | None = None,
             strata_extra: Mapping | None = None) -> dict:
    """One per-row output. Enforces the status contract: a ``measured`` row
    has a finite positive prediction; every other row has ``prediction=None``
    and a reason."""
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}; expected one of {STATUSES}")
    if status == "measured":
        value = _width_or_none(prediction)
        if value is None:
            raise ValueError(f"{spec.comparator_id}: a measured row needs a finite positive width, got {prediction!r}")
        prediction = value
    else:
        if prediction is not None:
            raise ValueError(f"{spec.comparator_id}: a {status} row carries no prediction")
        if not reason:
            raise ValueError(f"{spec.comparator_id}: a {status} row needs a reason")
    secondary = {name: (float(values[i]) if np.isfinite(values[i]) else None)
                 for name, values in (arrays.secondary or {}).items()}
    row = {
        "schema": ROW_SCHEMA,
        "comparator_id": spec.comparator_id,
        "family": spec.family,
        "method": spec.method,
        "arm": spec.arm,
        "context": spec.context,
        "mask_policy": spec.mask_policy,
        "population": spec.population,
        "measurement_role": spec.measurement_role,
        "sample_uid": str(arrays.sample_uid[i]),
        "group_uid": str(arrays.group_uid[i]),
        "frame_uid": str(arrays.frame_uid[i]),
        "source": arrays.source_id,
        "partition": str(arrays.partition[i]),
        "target": float(arrays.target_hfr_px[i]),
        "secondary_targets": secondary,
        "prediction": prediction,
        "status": status,
        "reason": reason,
        "processing_mode": spec.processing_mode,
        "input_definition": spec.input_definition,
        "reference_definition": arrays.identity.get("target_definition_id"),
        "timing_ms": None if timing_ms is None else float(timing_ms),
        "timing_scope": spec.timing_scope,
        "stages": dict(stages) if stages is not None else _na_stages(),
        "strata": row_strata(arrays, i, strata_extra),
    }
    if extra:
        row.update(extra)
    return row


def prediction_rows(spec: ComparatorSpec, arrays, indices, predictions, *, timing_ms=None,
                    unmeasured_reason: str = "non_finite_prediction") -> list:
    """Rows for any per-row predictor (e.g. a CNN arm in M4): finite positive
    predictions are measured, anything else is unmeasured with a reason."""
    rows = []
    for k, i in enumerate(np.asarray(indices).tolist()):
        value = _width_or_none(predictions[k])
        t = None if timing_ms is None else (timing_ms[k] if np.ndim(timing_ms) else timing_ms)
        rows.append(make_row(spec, arrays, i, prediction=value, status="measured" if value else "unmeasured",
                             reason=None if value else unmeasured_reason, timing_ms=t))
    return rows


# --------------------------------------------------------------------------- #
# Metrics (pure functions; hand-checked in tests)
# --------------------------------------------------------------------------- #
def is_measured(row: Mapping) -> bool:
    return row.get("status") == "measured" and _width_or_none(row.get("prediction")) is not None


def _target_of(row: Mapping, target: str):
    if target == "target":
        return _width_or_none(row.get("target"))
    return _width_or_none((row.get("secondary_targets") or {}).get(target))


def error_metrics(prediction, target) -> dict:
    """HFR MAE (px), log-HFR MAE, signed bias (mean prediction - target, px)
    and median relative error (median |prediction - target| / target) on
    paired finite positive values."""
    p = np.asarray(prediction, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=np.float64).ravel()
    if p.shape != t.shape:
        raise ValueError("prediction and target lengths differ")
    if p.size and (not np.all(np.isfinite(p) & (p > 0)) or not np.all(np.isfinite(t) & (t > 0))):
        raise ValueError("error_metrics takes finite positive widths only (filter unmeasured rows first)")
    if p.size == 0:
        return {"n": 0, "hfr_mae_px": UNAVAILABLE, "log_hfr_mae": UNAVAILABLE, "signed_bias_px": UNAVAILABLE,
                "median_relative_error": UNAVAILABLE}
    err = p - t
    return {
        "n": int(p.size),
        "hfr_mae_px": float(np.mean(np.abs(err))),
        "log_hfr_mae": float(np.mean(np.abs(np.log(p) - np.log(t)))),
        "signed_bias_px": float(np.mean(err)),
        "median_relative_error": float(np.median(np.abs(err) / t)),
    }


def coverage(rows: Sequence[Mapping]) -> float | str:
    """Measured fraction of the FULL declared population (every row counts)."""
    if not rows:
        return UNAVAILABLE
    return float(sum(1 for r in rows if is_measured(r)) / len(rows))


def summarize_rows(rows: Sequence[Mapping], *, target: str = "target") -> dict:
    """Coverage on the full population plus error metrics on its measured rows."""
    rows = [r for r in rows if _target_of(r, target) is not None]
    measured = [r for r in rows if is_measured(r)]
    statuses = Counter(r["status"] for r in rows)
    reasons = Counter(str(r.get("reason")) for r in rows if not is_measured(r))
    out = {
        "target": target,
        "population": len(rows),
        "measured": len(measured),
        "coverage": coverage(rows),
        "status_counts": {s: int(statuses.get(s, 0)) for s in STATUSES},
        "unmeasured_reasons": dict(reasons.most_common(12)),
        "metrics": error_metrics([r["prediction"] for r in measured], [_target_of(r, target) for r in measured]),
    }
    timings = [r["timing_ms"] for r in rows if r.get("timing_ms") is not None and r.get("timing_scope") == "row"]
    if timings:
        out["timing_ms_per_row"] = {"median": float(np.median(timings)), "mean": float(np.mean(timings)),
                                    "n": len(timings)}
    return out


def paired_comparison(rows_by_comparator: Mapping[str, Sequence[Mapping]], comparator_ids: Sequence[str], *,
                      target: str = "target", restrict_to: Iterable[str] | None = None) -> dict:
    """Errors on the common valid population plus each method's full-population coverage.

    The population is the set of sample_uids every listed comparator has a row
    for (optionally intersected with ``restrict_to``, e.g. an end-to-end
    matched set). ``common`` is the subset measured by all of them. Paired
    deltas (``mean |e_a| - |e_b|``) use only ``common``, so no method wins by
    excluding its hard cases; coverage is reported on the full population.
    """
    ids = list(comparator_ids)
    if not ids:
        raise ValueError("no comparators to compare")
    by_uid = {}
    for cid in ids:
        rows = rows_by_comparator[cid]
        uids = [r["sample_uid"] for r in rows]
        if len(set(uids)) != len(uids):
            raise ValueError(f"{cid}: duplicate sample_uid rows")
        by_uid[cid] = {r["sample_uid"]: r for r in rows if _target_of(r, target) is not None}
    population = set.intersection(*(set(m) for m in by_uid.values()))
    if restrict_to is not None:
        population &= set(restrict_to)
    common = sorted(u for u in population if all(is_measured(by_uid[c][u]) for c in ids))
    out = {"target": target, "comparators": ids, "population": len(population), "common_valid": len(common),
           "common_membership_sha256": _membership_sha256(common), "per_comparator": {}, "pairwise": {}}
    abs_err = {}
    for cid in ids:
        rows = [by_uid[cid][u] for u in sorted(population)]
        pred = [by_uid[cid][u]["prediction"] for u in common]
        tgt = [_target_of(by_uid[cid][u], target) for u in common]
        abs_err[cid] = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(tgt, dtype=np.float64))
        out["per_comparator"][cid] = {"coverage_full_population": coverage(rows),
                                      "measured_full_population": sum(1 for r in rows if is_measured(r)),
                                      "common_valid_metrics": error_metrics(pred, tgt)}
    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1:]:
            if not common:
                out["pairwise"][f"{a} vs {b}"] = {"n": 0, "mean_abs_error_difference_px": UNAVAILABLE,
                                                  "fraction_a_closer": UNAVAILABLE}
                continue
            diff = abs_err[a] - abs_err[b]
            out["pairwise"][f"{a} vs {b}"] = {"n": len(common), "mean_abs_error_difference_px": float(np.mean(diff)),
                                              "fraction_a_closer": float(np.mean(diff < 0)),
                                              "fraction_tied": float(np.mean(diff == 0))}
    return out


def _bin_label(value, edges) -> str:
    if value is None or not np.isfinite(value):
        return "missing"
    for lo, hi in zip(edges[:-1], edges[1:]):
        if value >= lo and (hi is None or value < hi):
            return f"[{lo:g}, {'inf' if hi is None else format(hi, 'g')})"
    return "out_of_range"


def stratify(rows: Sequence[Mapping], key: Callable[[Mapping], str], *, target: str = "target") -> dict:
    """``{stratum: summarize_rows(rows in stratum)}`` (coverage within the stratum)."""
    groups: dict = {}
    for row in rows:
        groups.setdefault(str(key(row)), []).append(row)
    return {name: summarize_rows(members, target=target) for name, members in sorted(groups.items())}


def standard_strata(rows: Sequence[Mapping], *, target: str = "target") -> dict:
    """Declared strata: target width bins, star_snr bins, categorical strata
    present in the rows, and the group (date/exposure) when a source has few groups."""
    out = {
        "width_bin": stratify(rows, lambda r: _bin_label(_target_of(r, target), FROZEN["width_bins_px"]),
                              target=target),
        "star_snr_bin": stratify(rows, lambda r: _bin_label((r.get("strata") or {}).get("star_snr"),
                                                            FROZEN["star_snr_bins"]), target=target),
    }
    for name in FROZEN["categorical_strata"]:
        if any(name in (r.get("strata") or {}) for r in rows):
            out[name] = stratify(rows, lambda r, n=name: (r.get("strata") or {}).get(n, "missing"), target=target)
    groups = {r["group_uid"] for r in rows}
    if 1 < len(groups) <= FROZEN["group_stratum_max_groups"]:
        out["group"] = stratify(rows, lambda r: r["group_uid"], target=target)
    return out


# --------------------------------------------------------------------------- #
# Matching (pure functions)
# --------------------------------------------------------------------------- #
def _distances(a_xy, b_xy) -> np.ndarray:
    a = np.asarray(a_xy, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(b_xy, dtype=np.float64).reshape(-1, 2)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    return np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])


def match_mutual_nearest(ref_xy, star_xy, *, radius: float, ambiguity_radius: float) -> list:
    """Match each reference to a used star by mutual nearest neighbour within ``radius``.

    Per reference: ``{"star_index": j | None, "distance_px": d | None,
    "nearest_star_distance_px", "ambiguous": bool, "n_stars_within_ambiguity",
    "n_refs_within_ambiguity_of_star", "claimed_by_other_reference": bool}``.
    A match is ambiguous when another star lies within ``ambiguity_radius`` of
    the reference or another reference within ``ambiguity_radius`` of the star.
    """
    d = _distances(ref_xy, star_xy)
    n_ref, n_star = d.shape
    out = []
    for i in range(n_ref):
        if n_star == 0:
            out.append({"star_index": None, "distance_px": None, "nearest_star_distance_px": None,
                        "ambiguous": False, "n_stars_within_ambiguity": 0, "n_refs_within_ambiguity_of_star": 0,
                        "claimed_by_other_reference": False})
            continue
        j = int(np.argmin(d[i]))
        nearest = float(d[i, j])
        near_stars = int(np.sum(d[i] <= ambiguity_radius))
        mutual = int(np.argmin(d[:, j])) == i
        tie = int(np.sum(d[i] == nearest)) > 1 or int(np.sum(d[:, j] == d[:, j].min())) > 1
        near_refs = int(np.sum(d[:, j] <= ambiguity_radius))
        matched = nearest <= radius and mutual
        out.append({
            "star_index": j if matched else None,
            "distance_px": nearest if matched else None,
            "nearest_star_distance_px": nearest,
            "ambiguous": bool(matched and (near_stars > 1 or near_refs > 1 or tie)),
            "n_stars_within_ambiguity": near_stars,
            "n_refs_within_ambiguity_of_star": near_refs,
            "claimed_by_other_reference": bool(nearest <= radius and not mutual),
        })
    return out


def examined_count(candidates_xy, used_xy, *, max_examined: int, max_used: int) -> int:
    """How many flux-sorted candidates the daemon's gate loop examined.

    The loop takes the first ``max_examined`` candidates and stops once
    ``max_used`` stars were used; used stars carry their candidate's exact
    centroid, so the stop rank is the rank of the last used star.
    """
    n = len(candidates_xy)
    if len(used_xy) >= max_used and n:
        d = _distances(used_xy, candidates_xy)
        ranks = [int(np.argmin(row)) for row in d]
        return min(max_examined, max(ranks) + 1)
    return min(max_examined, n)


def classify_reference(ref_xy, candidates_xy, used_xy, *, radius: float, max_examined: int, max_used: int,
                       examined: int | None = None) -> dict:
    """End-to-end stage of one unmatched reference: ``not_detected``,
    ``beyond_examined_cap`` (flux rank >= max_examined), ``after_used_cap``
    (the loop stopped after max_used stars), ``used_star_claimed_elsewhere``
    or ``gate_rejected``. Ranks are 0-based in the daemon's flux order."""
    cand = np.asarray(candidates_xy, dtype=np.float64).reshape(-1, 2)
    if examined is None:
        examined = examined_count(cand, used_xy, max_examined=max_examined, max_used=max_used)
    if cand.shape[0] == 0:
        return {"stage": "not_detected", "candidate_rank": None, "candidate_distance_px": None, "examined": 0}
    d = _distances([ref_xy], cand)[0]
    k = int(np.argmin(d))
    if d[k] > radius:
        return {"stage": "not_detected", "candidate_rank": None, "candidate_distance_px": float(d[k]),
                "examined": int(examined)}
    used = np.asarray(used_xy, dtype=np.float64).reshape(-1, 2)
    is_used = used.shape[0] > 0 and float(_distances([cand[k]], used)[0].min()) < 1e-3
    if is_used:
        stage = "used_star_claimed_elsewhere"
    elif k >= max_examined:
        stage = "beyond_examined_cap"
    elif k >= examined:
        stage = "after_used_cap"
    else:
        stage = "gate_rejected"
    return {"stage": stage, "candidate_rank": k, "candidate_distance_px": float(d[k]), "examined": int(examined)}


# --------------------------------------------------------------------------- #
# Fits on train_fit rows only
# --------------------------------------------------------------------------- #
class MappingUnavailable(RuntimeError):
    """Too few usable training scores to fit a mapping."""


def weighted_median(values, weights=None) -> float:
    """Median under per-row weights; equals ``np.median`` for equal weights."""
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if v.size == 0 or v.shape != w.shape or not np.all(np.isfinite(v)) or np.any(w <= 0):
        raise ValueError("weighted_median needs equal-length finite values and positive weights")
    order = np.argsort(v, kind="mergesort")
    v, w = v[order], w[order]
    cw = np.cumsum(w) / w.sum()
    k = int(np.searchsorted(cw, 0.5 - 1e-12))
    if abs(cw[k] - 0.5) <= 1e-12 and k + 1 < v.size:
        return float(0.5 * (v[k] + v[k + 1]))
    return float(v[k])


def fit_loglog(scores, targets, weights=None) -> dict:
    """``ln target = a + b ln score`` by weighted least squares on finite positive pairs."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    t = np.asarray(targets, dtype=np.float64).ravel()
    w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    keep = np.isfinite(s) & (s > 0) & np.isfinite(t) & (t > 0) & (w > 0)
    if keep.sum() < 2 or np.ptp(np.log(s[keep])) == 0:
        raise MappingUnavailable(f"{int(keep.sum())} usable training score(s) without spread")
    x, y, ww = np.log(s[keep]), np.log(t[keep]), w[keep] / w[keep].sum()
    xm, ym = float(np.sum(ww * x)), float(np.sum(ww * y))
    b = float(np.sum(ww * (x - xm) * (y - ym)) / np.sum(ww * (x - xm) ** 2))
    a = ym - b * xm
    return {"mapping": MAPPING_ID, "a": a, "b": b, "n_fit": int(keep.sum()), "n_dropped": int((~keep).sum())}


def apply_loglog(mapping: Mapping, score) -> float | None:
    s = _width_or_none(score)
    if s is None:
        return None
    return _width_or_none(math.exp(mapping["a"] + mapping["b"] * math.log(s)))


@dataclass
class Arm:
    name: str
    sources: tuple
    #: per source: (row indices into that source's SourceArrays, weights)
    pools: dict
    identity: dict


def build_arms(arrays_by_source: Mapping, contract) -> dict:
    """Single-source arms plus the mixed arm (when >= 2 sources load), each over train_fit rows only."""
    arms = {}
    for source_id, arrays in arrays_by_source.items():
        rows = arrays.partition_indices(FIT_PARTITION)
        name = ARM_OF_SOURCE.get(source_id, f"{source_id}_only")
        arms[name] = Arm(name=name, sources=(source_id,), pools={source_id: (rows, np.full(rows.size, 1.0 / rows.size))},
                         identity={"sources": [source_id], "fit_partition": FIT_PARTITION, "rows": int(rows.size),
                                   "membership_sha256": _membership_sha256(arrays.sample_uid[rows])})
    if len(arrays_by_source) >= 2:
        plan = fqe_sources.make_mixture(list(arrays_by_source.values()), FIT_PARTITION, contract=contract)
        arms[MIXED_ARM] = Arm(name=MIXED_ARM, sources=plan.source_ids,
                              pools={s: (plan.indices[s], plan.weights[s]) for s in plan.source_ids},
                              identity={"sources": list(plan.source_ids), "fit_partition": FIT_PARTITION,
                                        "mixture_id": plan.mixture_id, "mixture_sha256": plan.identity["mixture_sha256"],
                                        "source_weight": plan.identity["source_weight"],
                                        "common_reference_policy": plan.common_reference_policy})
    return arms


def _arm_values(arm: Arm, arrays_by_source: Mapping, per_source_values: Mapping):
    values, weights = [], []
    for source_id in arm.sources:
        rows, w = arm.pools[source_id]
        values.append(np.asarray(per_source_values[source_id], dtype=np.float64)[rows])
        weights.append(w)
    return np.concatenate(values), np.concatenate(weights)


# --------------------------------------------------------------------------- #
# Python comparators: constant, normalized-input HFR, sharpness
# --------------------------------------------------------------------------- #
def constant_rows(arms: Mapping, arrays_by_source: Mapping, eval_index: Mapping) -> tuple[list, dict, dict]:
    rows, fits, specs = [], {}, {}
    targets = {s: a.target_hfr_px for s, a in arrays_by_source.items()}
    for name, arm in arms.items():
        values, weights = _arm_values(arm, arrays_by_source, targets)
        median = weighted_median(values, weights)
        fit = {"arm": name, "median_hfr_px": median, "n_fit": int(values.size), "arm_identity": arm.identity}
        fit["fit_sha256"] = _sha256_obj(fit)
        fits[name] = fit
        spec = ComparatorSpec(
            comparator_id=f"constant__{name}", family="constant", method="train_fit_target_median",
            processing_mode="training_fit_constant", input_definition=INPUT_NONE, measurement_role="fitted_baseline",
            arm=name, definition=FROZEN["constant"], settings={"fit_sha256": fit["fit_sha256"]})
        specs[spec.comparator_id] = spec
        for source_id, arrays in arrays_by_source.items():
            for i in eval_index[source_id].tolist():
                rows.append(make_row(spec, arrays, i, prediction=median, status="measured", timing_ms=0.0,
                                     extra={"fit_sha256": fit["fit_sha256"]}))
    return rows, fits, specs


NORMALIZED_SPEC = ComparatorSpec(
    comparator_id="normalized_input__direct_hfr", family="normalized_input", method=baselines.DIRECT_HFR_METHOD,
    processing_mode="normalized_64x64_cnn_input", input_definition=INPUT_NORMALIZED,
    measurement_role="deployable_classical", definition=baselines.DIRECT_HFR_DEFINITION,
    settings={"sigma": 3.0, "iters": 5, "core_threshold_sigma": 3.0})


def normalized_input_rows(arrays_by_source: Mapping, eval_index: Mapping) -> list:
    rows = []
    for source_id, arrays in arrays_by_source.items():
        for i in eval_index[source_id].tolist():
            t0 = time.perf_counter()
            result = baselines.direct_hfr_measure_detailed(arrays.crops[i])
            ms = (time.perf_counter() - t0) * 1000.0
            bg = result["background"]
            bg_ok = result["reason"] != "bg_unresolved"
            core_ok = result["reason"] not in ("bg_unresolved", "no_core_above_3sigma")
            stages = {
                "background": _stage("completed" if bg_ok else "unavailable",
                                     None if bg_ok else ",".join(bg["reasons"]) or "bg_unresolved",
                                     level=bg["level"], rms=bg["rms"], method="python_crop_iterative_clip"),
                "extraction": _stage("completed" if core_ok else ("not_run" if not bg_ok else "no_source"),
                                     None if core_ok else result["reason"], method="core_above_3sigma_centroid"),
                "acceptance": _stage("not_applicable", None, note="no star gates in this baseline"),
                "width": _stage("measured" if result["hfr_px"] else "unmeasured", result["reason"],
                                method="python_signed_curve_of_growth"),
            }
            measured = result["hfr_px"] is not None
            rows.append(make_row(NORMALIZED_SPEC, arrays, i, prediction=result["hfr_px"],
                                 status="measured" if measured else "unmeasured", reason=result["reason"],
                                 timing_ms=ms, stages=stages))
    return rows


SCORE_FUNCTIONS = {"tenengrad": baselines.tenengrad, "normalized_variance": baselines.normalized_variance}


def sharpness_rows(arms: Mapping, arrays_by_source: Mapping, eval_index: Mapping) -> tuple[list, dict, dict]:
    """Raw sharpness scores on normalized crops, mapped per arm by a train_fit-only log-log fit."""
    rows, fits, specs = [], {}, {}
    scores, timing = {}, {}
    for source_id, arrays in arrays_by_source.items():
        for name, fn in SCORE_FUNCTIONS.items():
            vals = np.empty(len(arrays), dtype=np.float64)
            ms = np.empty(len(arrays), dtype=np.float64)
            for i in range(len(arrays)):
                t0 = time.perf_counter()
                vals[i] = fn(arrays.crops[i])
                ms[i] = (time.perf_counter() - t0) * 1000.0
            scores[(source_id, name)] = vals
            timing[(source_id, name)] = ms
    targets = {s: a.target_hfr_px for s, a in arrays_by_source.items()}
    for score_name in SCORE_FUNCTIONS:
        for arm_name, arm in arms.items():
            s, w = _arm_values(arm, arrays_by_source, {src: scores[(src, score_name)] for src in arm.sources})
            t, _ = _arm_values(arm, arrays_by_source, targets)
            key = f"{score_name}__{arm_name}"
            try:
                mapping = fit_loglog(s, t, w)
                mapping.update(score=score_name, arm=arm_name, arm_identity=arm.identity)
                mapping["mapping_sha256"] = _sha256_obj(mapping)
                failure = None
            except MappingUnavailable as exc:
                mapping, failure = None, str(exc)
            fits[key] = mapping or {"score": score_name, "arm": arm_name, "unavailable": failure}
            spec = ComparatorSpec(
                comparator_id=f"sharpness__{key}", family="sharpness", method=score_name,
                processing_mode="normalized_64x64_cnn_input+training_fit_monotone_mapping",
                input_definition=INPUT_NORMALIZED, measurement_role="fitted_baseline", arm=arm_name,
                definition=f"eval.baselines.{SCORE_FUNCTIONS[score_name].__name__} on the normalized crop, mapped "
                           f"by {MAPPING_ID} fitted on arm {arm_name} train_fit rows only",
                settings={"mapping_sha256": None if mapping is None else mapping["mapping_sha256"]})
            specs[spec.comparator_id] = spec
            for source_id, arrays in arrays_by_source.items():
                for i in eval_index[source_id].tolist():
                    raw = float(scores[(source_id, score_name)][i])
                    if mapping is None:
                        value, reason = None, f"mapping_unavailable: {failure}"
                    else:
                        value = apply_loglog(mapping, raw)
                        reason = None if value else "score_nonpositive_or_nonfinite"
                    rows.append(make_row(
                        spec, arrays, i, prediction=value, status="measured" if value else "unmeasured",
                        reason=reason, timing_ms=float(timing[(source_id, score_name)][i]),
                        extra={"raw_score": raw, "mapping_id": MAPPING_ID,
                               "mapping_sha256": None if mapping is None else mapping["mapping_sha256"]}))
    return rows, fits, specs


# --------------------------------------------------------------------------- #
# Native access (contract-checked joins; row-selective reads; no np.load)
# --------------------------------------------------------------------------- #
class NativeJoinError(RuntimeError):
    """A native observation does not reproduce the loaded row (join/contract problem)."""


def _source_directory(arrays, contract, data_root) -> tuple[Path, dict]:
    entry = contract.source(arrays.source_id)
    root = fqe_sources.resolve_data_root(data_root)
    directory = root / entry["directory"]
    if not directory.is_dir():
        raise fqe_sources.DataUnavailable(f"{arrays.source_id}: {directory} is missing")
    return directory, entry


def _check_pin(arrays, entry, directory: Path, name: str) -> str:
    pinned = entry["files"].get(name)
    if pinned is None:
        raise fqe_sources.ContractViolation(f"{arrays.source_id}: {name} is not pinned by the contract")
    actual = _sha256_file(directory / name)
    if actual != pinned:
        raise fqe_sources.HashMismatch(f"{arrays.source_id}: {name} sha256 {actual} != pinned {pinned}")
    return actual


def _records_for_rows(arrays, entry, directory: Path, rows: np.ndarray) -> dict:
    """``{npz_row: record}`` for the requested rows, identity-checked against the loaded arrays."""
    index_field = (entry.get("records_join") or {}).get("index_field", "fqe_index")
    wanted = {int(arrays.npz_row[i]): int(i) for i in rows.tolist()}
    found = {}
    with (directory / entry["records"]).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            index = record.get(index_field)
            if index is None or index not in wanted:
                continue
            i = wanted[index]
            checks = {"frame_id": _strip(arrays.frame_uid[i]), "split_group": _strip(arrays.group_uid[i]),
                      "source_id": _strip(arrays.sample_uid[i])}
            for key, expected in checks.items():
                if str(record.get(key)) != expected:
                    raise NativeJoinError(f"{arrays.source_id}: record {index_field}={index} {key}="
                                          f"{record.get(key)!r} != loaded row {expected!r}")
            found[index] = record
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise NativeJoinError(f"{arrays.source_id}: no record for npz rows {missing[:5]}")
    return found


def _dotted(record: Mapping, path: str):
    value = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


@dataclass
class NativeObservation:
    index: int                      # row in the SourceArrays
    raw: np.ndarray                 # native 64x64 float32
    masks: dict                     # name -> uint8 0/1 planes (per declared policy)
    ref_xy: tuple                   # reference position inside the 64x64 crop
    record: dict
    join_ok: bool
    join_detail: str | None
    extra: dict = field(default_factory=dict)


def load_sim_native(arrays, rows, *, contract, data_root=None) -> dict:
    """stellar_sim_v1-style native observations of ``rows`` (row-selective reads)."""
    from preprocess.normalize import robust_normalize

    directory, entry = _source_directory(arrays, contract, data_root)
    provenance = {name: _check_pin(arrays, entry, directory, name) for name in ("records.jsonl", "observations.npz")
                  if name in entry["files"]}
    if "observations.npz" not in provenance:
        raise fqe_sources.ContractViolation(f"{arrays.source_id}: observations.npz is not pinned")
    rows = np.asarray(rows, dtype=np.int64)
    records = _records_for_rows(arrays, entry, directory, rows)
    array_index = {}
    for i in rows.tolist():
        rec = records[int(arrays.npz_row[i])]
        ai = rec.get("array_index")
        if "array_index" in arrays.joins and int(arrays.joins["array_index"][i]) != ai:
            raise NativeJoinError(f"{arrays.source_id}: array_index of row {i} disagrees with the loader join")
        array_index[i] = int(ai)
    order = sorted(set(array_index.values()))
    with fqe_sources._NpzReader(directory / "observations.npz", arrays.source_id) as reader:
        raw_rows = reader.read_rows("raw_crops", np.asarray(order, dtype=np.int64))
        flag_rows = reader.read_rows("pixel_flags", np.asarray(order, dtype=np.int64))
        groups = reader.read_full("split_group").astype(str) if "split_group" in reader.keys() else None
    position = {a: k for k, a in enumerate(order)}
    out = {}
    for i in rows.tolist():
        rec = records[int(arrays.npz_row[i])]
        k = position[array_index[i]]
        raw = raw_rows[k].astype(np.float32)
        flags = flag_rows[k].astype(np.uint8)
        detail = None
        if groups is not None and groups[array_index[i]] != _strip(arrays.group_uid[i]):
            detail = "observations.npz split_group disagrees with the loaded group"
        elif not np.array_equal(robust_normalize(raw)[0], arrays.crops[i]):
            detail = "robust_normalize(raw_crops[array_index]) != loaded crop"
        center = _dotted(rec, "parameters.center")
        out[i] = NativeObservation(
            index=i, raw=raw,
            masks={"invalid": ((flags & 1) != 0).astype(np.uint8), "native_saturation": ((flags & 2) != 0).astype(np.uint8)},
            ref_xy=(float(center[0]), float(center[1])) if center else (float("nan"), float("nan")),
            record={"array_index": array_index[i], "noise": rec.get("noise"), "bias": rec.get("bias_numeric_level"),
                    "saturation_numeric_level": rec.get("saturation_numeric_level"), "center": center},
            join_ok=detail is None, join_detail=detail)
    return {"observations": out, "provenance": {"files_sha256": provenance, "rows": int(rows.size),
                                                 "array_index_membership_sha256": _membership_sha256(order)}}


def _exposure_artifacts(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    return {a.get("path"): a.get("sha256") for a in manifest.get("artifacts", []) or []}


@dataclass
class HstFrame:
    frame_id: str
    group_uid: str
    science: np.ndarray
    valid: np.ndarray
    dq: np.ndarray
    background: np.ndarray
    rows: list
    refs_xy: np.ndarray
    neighbours_xy: np.ndarray | None
    provenance: dict


def iter_hst_native(arrays, rows, *, contract, data_root=None, allowed_partitions=("val",)):
    """Yield one :class:`HstFrame` (plus per-row native crops) per exposure of ``rows``.

    Frames are opened only when every loaded row of that exposure lies in
    ``allowed_partitions`` (val by default): a test-era exposure is never a
    development input (the loader returns no test rows, and this check keeps
    it that way). Each exposure file is verified against the exposure's own
    manifest artifact sha256; the native crop must equal the frame cut-out and
    reproduce the loaded normalized crop after the teacher's background
    imputation.
    """
    from preprocess.normalize import robust_normalize

    directory, entry = _source_directory(arrays, contract, data_root)
    records_sha = _check_pin(arrays, entry, directory, entry["records"])
    rows = np.asarray(rows, dtype=np.int64)
    records = _records_for_rows(arrays, entry, directory, rows)
    by_frame: dict = {}
    for i in rows.tolist():
        by_frame.setdefault(_strip(arrays.frame_uid[i]), []).append(i)
    for frame_id in sorted(by_frame):
        member_rows = np.flatnonzero(np.asarray(arrays.frame_uid).astype(str) == f"{arrays.source_id}:{frame_id}")
        partitions = set(np.asarray(arrays.partition)[member_rows].tolist())
        if not partitions <= set(allowed_partitions):
            raise SealedPartitionError(f"{arrays.source_id}: exposure {frame_id} holds rows of {sorted(partitions)}; "
                                       f"only {list(allowed_partitions)} exposures are opened")
        exposure = directory / "exposures" / frame_id
        if not exposure.is_dir():
            raise fqe_sources.DataUnavailable(f"{arrays.source_id}: exposure directory {exposure} is missing")
        artifacts = _exposure_artifacts(exposure)
        provenance = {"records.jsonl(corpus)": records_sha}
        for name in ("observations.npz", "frame.npz"):
            actual = _sha256_file(exposure / name)
            if artifacts.get(name) != actual:
                raise fqe_sources.HashMismatch(f"{arrays.source_id}: {frame_id}/{name} sha256 {actual} != exposure "
                                               f"manifest {artifacts.get(name)}")
            provenance[f"{frame_id}/{name}"] = actual
        with fqe_sources._NpzReader(exposure / "frame.npz", arrays.source_id) as reader:
            science = reader.read_full("science_electrons").astype(np.float32)
            valid = reader.read_full("valid_mask").astype(bool)
            dq = reader.read_full("dq")
            background = reader.read_full("sep_background_estimate_electrons").astype(np.float32)
        frame_rows = by_frame[frame_id]
        obs_index = {i: int(records[int(arrays.npz_row[i])]["observation_index"]) for i in frame_rows}
        order = sorted(set(obs_index.values()))
        with fqe_sources._NpzReader(exposure / "observations.npz", arrays.source_id) as reader:
            raw_rows = reader.read_rows("raw_crops64", np.asarray(order, dtype=np.int64))
        position = {o: k for k, o in enumerate(order)}
        neighbours = None
        if (exposure / "records.jsonl").is_file():
            provenance[f"{frame_id}/records.jsonl"] = _sha256_file(exposure / "records.jsonl")
            xy = []
            with (exposure / "records.jsonl").open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        if rec.get("x_native_pixel") is not None:
                            xy.append((float(rec["x_native_pixel"]), float(rec["y_native_pixel"])))
            neighbours = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        crops, refs = {}, []
        for i in frame_rows:
            rec = records[int(arrays.npz_row[i])]
            x0, y0 = (int(v) for v in rec["crop_origin_xy"])
            if "observation_index" in arrays.joins and int(arrays.joins["observation_index"][i]) != obs_index[i]:
                raise NativeJoinError(f"{arrays.source_id}: observation_index of row {i} disagrees with the loader")
            raw = raw_rows[position[obs_index[i]]].astype(np.float32)
            cut = science[y0:y0 + 64, x0:x0 + 64]
            valid_crop = valid[y0:y0 + 64, x0:x0 + 64]
            imputed = raw.copy()
            imputed[~valid_crop] = background[y0:y0 + 64, x0:x0 + 64][~valid_crop]
            detail = None
            if cut.shape != (64, 64) or not np.array_equal(raw, cut, equal_nan=True):
                detail = "raw_crops64[observation_index] != frame cut-out at crop_origin_xy"
            elif not np.array_equal(robust_normalize(imputed)[0], arrays.crops[i]):
                detail = "robust_normalize(background-imputed native crop) != loaded crop"
            ref = (float(rec["x_native_pixel"]), float(rec["y_native_pixel"]))
            refs.append(ref)
            neighbour_flag = None
            if neighbours is not None and neighbours.size:
                dist = np.hypot(neighbours[:, 0] - ref[0], neighbours[:, 1] - ref[1])
                neighbour_flag = bool(np.any((dist > 0.01) & (dist <= FROZEN["hst_neighbour_radius_px"])))
            crops[i] = NativeObservation(
                index=i, raw=raw, masks={}, ref_xy=(ref[0] - x0, ref[1] - y0),
                record={"crop_origin_xy": [x0, y0], "observation_index": obs_index[i],
                        "x_native_pixel": ref[0], "y_native_pixel": ref[1],
                        "hfr_aperture_radius_px": rec.get("hfr_aperture_radius_px")},
                join_ok=detail is None, join_detail=detail,
                extra={"valid": valid_crop.copy(), "dq": dq[y0:y0 + 64, x0:x0 + 64].copy(), "imputed": imputed,
                       "neighbour_within_24px": neighbour_flag})
        yield HstFrame(frame_id=frame_id, group_uid=str(arrays.group_uid[frame_rows[0]]), science=science,
                       valid=valid, dq=dq, background=background, rows=frame_rows,
                       refs_xy=np.asarray(refs, dtype=np.float64).reshape(-1, 2), neighbours_xy=neighbours,
                       provenance=provenance), crops


def hst_crop_masks(obs: NativeObservation, policy: str) -> tuple[np.ndarray, dict]:
    """Pixels and masks of an isolated HST crop under a declared mask policy."""
    sat = ((obs.extra["dq"].astype(np.int64) & FROZEN["hst_native_saturation_dq_bits"]) != 0).astype(np.uint8)
    if policy == "teacher_dq":
        invalid = (~obs.extra["valid"]).astype(np.uint8)
        pixels = obs.raw.copy()
        bad = ~np.isfinite(pixels)
        pixels[bad] = 0.0
        invalid[bad] = 1
        return pixels, {"invalid": invalid, "native_saturation": sat}
    if policy == "imputed_saturation_only":
        pixels = obs.extra["imputed"].copy()
        bad = ~np.isfinite(pixels)
        pixels[bad] = 0.0
        masks = {"native_saturation": sat}
        if bad.any():
            masks["invalid"] = bad.astype(np.uint8)
        return pixels, masks
    raise ValueError(f"unknown mask policy {policy!r}")


def frame_masks(frame: HstFrame) -> tuple[np.ndarray, dict]:
    pixels = frame.science.copy()
    bad = ~np.isfinite(pixels)
    pixels[bad] = 0.0
    invalid = (~frame.valid | bad).astype(np.uint8)
    sat = ((frame.dq.astype(np.int64) & FROZEN["hst_native_saturation_dq_bits"]) != 0).astype(np.uint8)
    return pixels, {"invalid": invalid, "native_saturation": sat}


# --------------------------------------------------------------------------- #
# SEP flux_radius reference comparators
# --------------------------------------------------------------------------- #
def sim_oracle_background(noise: Mapping, size: int = 64) -> np.ndarray:
    """The frozen generator's sky (stellar_sim_v1 generator_source._background)."""
    sky = float(noise["sky_electrons"])
    gx, gy = float(noise["gradient_x_per_frame"]), float(noise["gradient_y_per_frame"])
    y, x = np.indices((size, size), dtype=np.float64)
    return np.maximum(sky + gx * (x / size - 0.5) + gy * (y / size - 0.5), 0.01)


def sep_aperture_hfr(signal: np.ndarray, center, radius: float) -> tuple[float | None, str | None]:
    """``sep.flux_radius`` (frac 0.5, subpix 5) at a fixed centre and radius, signed flux
    (the frozen generator's ``aperture_hfr``); ``(None, reason)`` when invalid."""
    import sep

    signal = np.asarray(signal, dtype=np.float64)
    y, x = np.indices(signal.shape, dtype=np.float64)
    selected = (np.hypot(x - center[0], y - center[1]) <= radius) & np.isfinite(signal)
    if selected.sum() < 4:
        return None, "fewer_than_4_aperture_pixels"
    total = signal[selected].sum()
    if not np.isfinite(total) or total <= 0:
        return None, "nonpositive_aperture_flux"
    if not np.isfinite(signal).all():
        return None, "nonfinite_pixels"
    result, flags = sep.flux_radius(np.ascontiguousarray(signal), np.asarray([center[0]], dtype=float),
                                    np.asarray([center[1]], dtype=float), np.asarray([radius], dtype=float),
                                    FROZEN["sep_flux_radius"]["frac"], subpix=FROZEN["sep_flux_radius"]["subpix"])
    if flags[0] or not np.isfinite(result[0]) or result[0] <= 0:
        return None, f"sep_flags_{int(flags[0])}"
    return float(result[0]), None


SIM_SEP_SPEC = ComparatorSpec(
    comparator_id="sep_flux_radius__sim_oracle_bg_true_centre_r24", family="sep_flux_radius",
    method="sep.flux_radius", processing_mode="native_linear_64x64_observation", input_definition=INPUT_SIM_NATIVE,
    measurement_role="oracle_measurement_reference",
    definition="sep.flux_radius frac 0.5 subpix 5 on raw - true background (generator sky + gradients from "
               "records.noise) - 500 bias, at the true centre (records.parameters.center), fixed r = 24 px, no mask: "
               "the stored measured_hfr_px estimator. Uses simulated truth (background, centre): not deployable; it "
               "isolates the effect of photon/read noise on the target's own estimator.",
    settings={"radius_px": FROZEN["sim_oracle_radius_px"], **FROZEN["sep_flux_radius"]})

HST_SEP_SPEC = ComparatorSpec(
    comparator_id="sep_flux_radius__hst_teacher_reproduction", family="sep_flux_radius", method="sep.flux_radius",
    processing_mode="native_linear_parent_frame", input_definition=INPUT_HST_FRAME,
    measurement_role="teacher_identity", timing_scope="row_amortized",
    definition="the teacher's chain on the parent validation frame: masked pixels set to the median of the valid "
               "science pixels, minus the stored SEP background map, sep.flux_radius frac 0.5 subpix 5 with the "
               "teacher mask at the stored isophotal position (x/y_native_pixel) and rmax = hfr_aperture_radius_px "
               "(= clip(6a, 6, 16)). This IS the label estimator: agreement is identity, not independent truth.",
    settings=dict(FROZEN["sep_flux_radius"]))


def sim_sep_rows(arrays, native: Mapping, eval_rows) -> list:
    rows = []
    for i in eval_rows.tolist():
        obs = native.get(i)
        if obs is None or not obs.join_ok:
            rows.append(make_row(SIM_SEP_SPEC, arrays, i, prediction=None, status="failed",
                                 reason=f"native_join_mismatch: {obs.join_detail if obs else 'no native row'}"))
            continue
        t0 = time.perf_counter()
        noise, bias, center = obs.record.get("noise"), obs.record.get("bias"), obs.record.get("center")
        if not noise or bias is None or not center:
            rows.append(make_row(SIM_SEP_SPEC, arrays, i, prediction=None, status="failed",
                                 reason="record lacks noise/bias/centre for the oracle background"))
            continue
        signal = obs.raw.astype(np.float64) - sim_oracle_background(noise, obs.raw.shape[0]) - float(bias)
        value, why = sep_aperture_hfr(signal, center, FROZEN["sim_oracle_radius_px"])
        ms = (time.perf_counter() - t0) * 1000.0
        stored = arrays.secondary.get("measured_hfr_px")
        extra = {"stored_measured_hfr_px": None if stored is None else float(stored[i])}
        if value is not None and stored is not None:
            extra["recomputation_rel_diff"] = abs(value - float(stored[i])) / float(stored[i])
        stages = {"background": _stage("completed", None, method="oracle_true_background"),
                  "extraction": _stage("not_applicable", None, note="true analytic centre"),
                  "acceptance": _stage("not_applicable"),
                  "width": _stage("measured" if value else "unmeasured", why, method="sep.flux_radius r24")}
        rows.append(make_row(SIM_SEP_SPEC, arrays, i, prediction=value, status="measured" if value else "unmeasured",
                             reason=why, timing_ms=ms, stages=stages, extra=extra))
    return rows


def hst_teacher_rows(arrays, frame: HstFrame, crops: Mapping) -> list:
    import sep

    rows = []
    t0 = time.perf_counter()
    mask = ~frame.valid
    cleaned = frame.science.copy()
    cleaned[mask] = float(np.median(frame.science[~mask]))
    residual = np.ascontiguousarray(cleaned - frame.background)
    xs = np.asarray([crops[i].record["x_native_pixel"] for i in frame.rows], dtype=float)
    ys = np.asarray([crops[i].record["y_native_pixel"] for i in frame.rows], dtype=float)
    rmax = np.asarray([crops[i].record["hfr_aperture_radius_px"] for i in frame.rows], dtype=float)
    hfr, flags = sep.flux_radius(residual, xs, ys, rmax, FROZEN["sep_flux_radius"]["frac"], mask=mask,
                                 subpix=FROZEN["sep_flux_radius"]["subpix"])
    per_row_ms = (time.perf_counter() - t0) * 1000.0 / max(len(frame.rows), 1)
    fatal = ~int(sep.APER_HASMASKED)
    for k, i in enumerate(frame.rows):
        if not crops[i].join_ok:
            rows.append(make_row(HST_SEP_SPEC, arrays, i, prediction=None, status="failed",
                                 reason=f"native_join_mismatch: {crops[i].join_detail}"))
            continue
        value = _width_or_none(hfr[k]) if not (int(flags[k]) & fatal) else None
        reason = None if value else f"sep_flags_{int(flags[k])}"
        stages = {"background": _stage("completed", None, method="stored teacher SEP background map"),
                  "extraction": _stage("not_applicable", None, note="stored teacher position"),
                  "acceptance": _stage("not_applicable"),
                  "width": _stage("measured" if value else "unmeasured", reason, method="sep.flux_radius")}
        rows.append(make_row(HST_SEP_SPEC, arrays, i, prediction=value, status="measured" if value else "unmeasured",
                             reason=reason, timing_ms=per_row_ms, stages=stages,
                             extra={"abs_diff_to_teacher_px": None if value is None else abs(value - float(arrays.target_hfr_px[i])),
                                    "frame_id": frame.frame_id}))
    return rows


# --------------------------------------------------------------------------- #
# Native Rust comparator through the bridge
# --------------------------------------------------------------------------- #
@dataclass
class BridgeCase:
    """Duck-typed scenario for ``eval.bge.bridge.write_inputs``."""

    id: str
    image: np.ndarray
    masks: dict
    metadata: dict

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


def canonical_rust_methods(methods: Iterable[str]) -> list:
    """Canonical section-4 ids; the plain backend strings are refused."""
    from eval.bge import bridge

    out = []
    for method in methods:
        if method in bridge.RUST_BACKENDS:
            raise ValueError(f"{method!r} runs the daemon's current defaults; use a section-4 method id "
                             f"({sorted(bridge.RUST_VARIANTS)})")
        canonical, _ = bridge.rust_variant(method)
        if canonical not in out:
            out.append(canonical)
    return out


def rust_spec(method: str, context: str, mask_policy: str | None) -> ComparatorSpec:
    from eval.bge import bridge

    canonical, variant = bridge.rust_variant(method)
    frame = context == "hst_parent_frame"
    inputs = {"sim_observation64": INPUT_SIM_NATIVE, "hst_crop64": INPUT_HST_CROP, "hst_parent_frame": INPUT_HST_FRAME}
    modes = {"sim_observation64": "native_linear_64x64_observation (the isolated observation is the frame)",
             "hst_crop64": "native_linear_64x64_isolated_crop (non-daemon context: an isolated cut-out of a frame)",
             "hst_parent_frame": "native_linear_parent_frame (daemon end-to-end context)"}
    suffix = f"__{mask_policy}" if mask_policy else ""
    return ComparatorSpec(
        comparator_id=f"native_rust__{canonical}__{context}{suffix}", family="native_rust", method=canonical,
        processing_mode=modes[context], input_definition=inputs[context], measurement_role="deployable_classical",
        population=POPULATION_END_TO_END if frame else POPULATION_PER_CROP,
        timing_scope="frame" if frame else "row", context=context, mask_policy=mask_policy,
        definition=f"achernar-daemon bge_eval_bridge: analyze_background -> estimate_quality_with_policy "
                   f"(flux-sorted candidates, <= 48 examined, <= 16 used; gates: peak saturation, edge, deblended, "
                   f"masked/saturated support, SNR >= 5, >= 5 core px, ellipticity <= 0.35, blend test) -> "
                   f"compute_hfr_with_sky; variant {canonical} = backend {variant['backend']}, settings "
                   f"{json.dumps(variant['settings'], sort_keys=True)}, star_gates "
                   f"{json.dumps(variant['star_gates'], sort_keys=True)}. Prediction = HFR of the used star matched "
                   f"to the reference (not the frame median).",
        settings={"method": canonical, "backend": variant["backend"], "settings": variant["settings"],
                  "star_gates": variant["star_gates"], "mask_policy": mask_policy,
                  "mask_policy_definition": MASK_POLICIES.get(mask_policy), "match": FROZEN["match"],
                  "saturation_adu": FROZEN["sim_saturation_adu"] if context == "sim_observation64"
                  else FROZEN["hst_saturation_adu"]})


def default_bridge_runner(repo_root: Path, cargo_target_dir: Path, jobs: int = 4) -> Callable:
    from eval.bge import bridge

    def run(manifest_path: Path, out_dir: Path, log_path: Path) -> dict:
        info = bridge.run_bridge(manifest_path, out_dir, repo_root, cargo_target_dir, log_path, jobs=jobs)
        # cargo prints the test binary relative to the daemon crate; hash it from there.
        binary = info.get("binary")
        if binary and not info.get("binary_sha256"):
            path = Path(binary)
            for candidate in (path, Path(repo_root) / "achernar-daemon" / path):
                if candidate.is_file():
                    info["binary_sha256"] = _sha256_file(candidate)
                    info["binary_resolved"] = str(candidate.resolve())
                    break
        return info

    return run


def run_bridge_cases(cases: Sequence[BridgeCase], methods: Sequence[str], work_dir: Path, *, runner: Callable,
                     label: str) -> dict:
    """Write inputs + one manifest for ``methods`` and run the bridge once."""
    from eval.bge import bridge

    case_dir = Path(work_dir) / label
    case_dir.mkdir(parents=True, exist_ok=True)
    entries = bridge.write_inputs(cases, case_dir)
    manifest = bridge.write_methods_manifest(entries, case_dir, methods, write_maps=False,
                                             extras={"benchmark": SCHEMA, "context": label})
    out_dir = case_dir / "out"
    t0 = time.perf_counter()
    info = runner(manifest, out_dir, case_dir / "bridge.log")
    info = dict(info or {})
    info.setdefault("wall_s", time.perf_counter() - t0)
    info.update(manifest=str(manifest), out_dir=str(out_dir), cases=len(cases), methods=list(methods),
                manifest_sha256=_sha256_file(manifest))
    return info


def _quality(payload: Mapping) -> dict:
    return payload.get("quality") or {}


def _xy(items) -> np.ndarray:
    return np.asarray([(float(c["x"]), float(c["y"])) for c in items or []], dtype=np.float64).reshape(-1, 2)


def case_selection(payload: Mapping) -> dict:
    """End-to-end selection summary of one bridge case (caps, gates, tallies)."""
    q = _quality(payload)
    bg = payload.get("background") or {}
    candidates = payload.get("candidates") or []
    used = q.get("used_stars") or []
    caps = FROZEN["rust_caps"]
    return {
        "background_method": bg.get("method"), "background_level_status": bg.get("level_status"),
        "background_reasons": bg.get("reasons"), "background_config_sha256": bg.get("config_sha256"),
        "catalog_error": payload.get("catalog_error"),
        "quality_status": q.get("status"), "rejection_reasons": q.get("rejection_reasons"),
        "star_count_detected": q.get("star_count_detected"), "candidates": len(candidates),
        "examined": examined_count(_xy(candidates), _xy(used), max_examined=caps["max_stars_examined"],
                                   max_used=caps["max_stars_used"]),
        "used": len(used), "star_rejections": q.get("star_rejections"),
        "gate_diagnostics": q.get("gate_diagnostics"), "star_gates_config_sha256": q.get("star_gates_config_sha256"),
        "frame_hfr_median_px": q.get("hfr_px"), "catalog_objects": None if payload.get("catalog") is None
        else len(payload["catalog"]), "timing_ms": payload.get("timing_ms"),
    }


def _bridge_failure(read: Mapping, method: str) -> tuple[str, str]:
    return "failed", f"bridge_{read.get('status')}: {read.get('reason')}"


def reference_outcomes(payload: Mapping, refs_xy, *, radius: float) -> list:
    """Per-reference outcome of one completed bridge case (crop or frame).

    Returns ``(status, prediction, reason, stages, match)`` dicts. Background
    failure because the frame is below the SEP mesh grid is ``unsupported``.
    """
    caps, amb = FROZEN["rust_caps"], FROZEN["match"]["ambiguity_radius_px"]
    bg = payload.get("background") or {}
    q = _quality(payload)
    reasons = [str(r) for r in (bg.get("reasons") or [])]
    level_ok = bg.get("level_status") == "measured"
    refs = np.asarray(refs_xy, dtype=np.float64).reshape(-1, 2)
    bg_stage = _stage("completed" if level_ok else "unavailable", None if level_ok else ",".join(reasons) or None,
                      method=bg.get("method"), backend_error=bg.get("backend_error"))
    if "fallback_used" in reasons:
        reason = (f"fallback_used: the selected backend failed ({bg.get('backend_error')}); the labelled legacy "
                  "fallback is not this method's measurement")
        stages = {"background": dict(bg_stage, status="fallback"), "extraction": _stage("not_scored"),
                  "acceptance": _stage("not_scored"), "width": _stage("not_scored")}
        return [{"status": "unmeasured", "prediction": None, "reason": reason, "stages": stages, "match": None}
                for _ in range(refs.shape[0])]
    if not level_ok:
        small = "frame_smaller_than_mesh_grid" in reasons
        status = "unsupported" if small else "unmeasured"
        reason = ("frame_smaller_than_mesh_grid: the SEP backend needs >= 2 meshes of 64x64 px per axis; a 64x64 "
                  "crop has no parent frame in this context" if small
                  else f"background_unavailable: {','.join(reasons) or bg.get('backend_error')}")
        stages = {"background": bg_stage, "extraction": _stage("not_run"), "acceptance": _stage("not_run"),
                  "width": _stage("not_run")}
        return [{"status": status, "prediction": None, "reason": reason, "stages": stages, "match": None}
                for _ in range(refs.shape[0])]
    candidates = payload.get("candidates") or []
    used = q.get("used_stars") or []
    detection_ran = payload.get("detection") is not None and q.get("status") not in ("BackgroundUnavailable",
                                                                                       "DetectionUnavailable")
    if not detection_ran:
        reason = f"detection_unavailable: quality status {q.get('status')}; catalog_error {payload.get('catalog_error')}"
        stages = {"background": bg_stage, "extraction": _stage("unavailable", reason), "acceptance": _stage("not_run"),
                  "width": _stage("not_run")}
        return [{"status": "unmeasured", "prediction": None, "reason": reason, "stages": stages, "match": None}
                for _ in range(refs.shape[0])]
    cand_xy, used_xy = _xy(candidates), _xy(used)
    examined = examined_count(cand_xy, used_xy, max_examined=caps["max_stars_examined"], max_used=caps["max_stars_used"])
    matches = match_mutual_nearest(refs, used_xy, radius=radius, ambiguity_radius=amb)
    tallies = {k: v for k, v in (q.get("star_rejections") or {}).items() if v}
    rejected_examined = examined - len(used)
    # SEP backends also emit the raw extraction catalog: record whether the
    # reference was extracted at all (mutual nearest neighbour), separately
    # from the gate candidates (families are regrouped under g2).
    catalog = payload.get("catalog")
    catalog_hits = (match_mutual_nearest(refs, _xy(catalog), radius=radius, ambiguity_radius=amb)
                    if catalog is not None else [None] * refs.shape[0])
    out = []
    for r, match in enumerate(matches):
        hit = catalog_hits[r]
        catalog_detail = {} if hit is None else {
            "in_sep_catalog": hit["star_index"] is not None, "sep_catalog_distance_px": hit["distance_px"],
            "sep_catalog_ambiguous": hit["ambiguous"]}
        extraction = _stage("completed", None, candidates=len(candidates), **catalog_detail)
        if match["star_index"] is not None:
            star = used[match["star_index"]]
            value = _width_or_none(star.get("hfr_px"))
            stages = {"background": bg_stage, "extraction": extraction,
                      "acceptance": _stage("accepted", None),
                      "width": _stage("measured" if value else "unmeasured", None if value else "nonfinite_used_hfr")}
            out.append({"status": "measured" if value else "unmeasured", "prediction": value,
                        "reason": None if value else "nonfinite_used_hfr", "stages": stages,
                        "match": dict(match, stage="accepted")})
            continue
        cls = classify_reference(refs[r], cand_xy, used_xy, radius=radius, max_examined=caps["max_stars_examined"],
                                 max_used=caps["max_stars_used"], examined=examined)
        stage = cls["stage"]
        if stage == "gate_rejected":
            if rejected_examined == 1:
                reason = "gate_rejected: " + (",".join(sorted(tallies)) or "no tally")
            else:
                reason = (f"gate_rejected: not attributable ({rejected_examined} examined candidates rejected; "
                          f"frame tallies {json.dumps(tallies, sort_keys=True)})")
        elif stage == "used_star_claimed_elsewhere":
            reason = "match_ambiguous: the nearest used star is the mutual match of another reference"
        else:
            reason = stage
        extraction = _stage("completed" if stage != "not_detected" else "not_detected",
                            None if stage != "not_detected" else "no candidate within the match radius",
                            candidates=len(candidates), candidate_rank=cls["candidate_rank"],
                            candidate_distance_px=cls["candidate_distance_px"], **catalog_detail)
        acceptance = _stage("not_run" if stage == "not_detected" else
                            ("not_examined" if stage in ("beyond_examined_cap", "after_used_cap") else "rejected"),
                            None if stage == "not_detected" else reason)
        stages = {"background": bg_stage, "extraction": extraction, "acceptance": acceptance,
                  "width": _stage("not_run")}
        out.append({"status": "unmeasured", "prediction": None, "reason": reason, "stages": stages,
                    "match": dict(match, **cls)})
    return out


def rust_rows_for_context(spec: ComparatorSpec, arrays, run_info: Mapping, case_rows: Sequence[tuple],
                          *, radius: float, width: int, height: int, join_failed: Mapping | None = None) -> tuple[list, list]:
    """Rows of one Rust method in one context.

    ``case_rows`` = ``[(case_id, [row indices], refs_xy)]``. A missing or
    failed bridge output fails its rows; it never becomes a width.
    """
    from eval.bge import bridge

    rows, selections = [], []
    join_failed = join_failed or {}
    for case_id, indices, refs in case_rows:
        failed_join = [i for i in indices if i in join_failed]
        if run_info.get("status") != "completed":
            read = {"status": run_info.get("status", "failed"), "reason": run_info.get("reason")}
        else:
            read = bridge.read_output(Path(run_info["out_dir"]), case_id, spec.method, width, height)
        if read.get("status") != "completed":
            status, reason = _bridge_failure(read, spec.method)
            for i in indices:
                rows.append(make_row(spec, arrays, i, prediction=None, status=status, reason=reason))
            selections.append({"case_id": case_id, "bridge_status": read.get("status"), "reason": read.get("reason")})
            continue
        payload = read["json"]
        selection = case_selection(payload)
        timing = payload.get("timing_ms") or {}
        case_ms = None
        if timing.get("background") is not None and timing.get("quality") is not None:
            case_ms = float(timing["background"]) + float(timing["quality"])
        outcomes = reference_outcomes(payload, refs, radius=radius)
        n_matched = 0
        for i, outcome in zip(indices, outcomes):
            if i in failed_join:
                rows.append(make_row(spec, arrays, i, prediction=None, status="failed",
                                     reason=f"native_join_mismatch: {join_failed[i]}"))
                continue
            n_matched += outcome["status"] == "measured"
            rows.append(make_row(spec, arrays, i, prediction=outcome["prediction"], status=outcome["status"],
                                 reason=outcome["reason"], timing_ms=case_ms, stages=outcome["stages"],
                                 extra={"match": outcome["match"], "bridge_case": case_id}))
        selection.update(case_id=case_id, references=len(indices), matched=n_matched,
                         stage_counts=dict(Counter((o["match"] or {}).get("stage", o["status"]) for o in outcomes)))
        selections.append(selection)
    return rows, selections


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _resolve_families(methods: Sequence[str] | None) -> tuple[set, list]:
    if not methods:
        return set(FAMILIES), list(DEFAULT_RUST_METHODS)
    families, rust = set(), []
    for token in methods:
        if token in FAMILIES:
            families.add(token)
        else:
            rust.extend(canonical_rust_methods([token]))
            families.add("native_rust")
    if "native_rust" in families and not rust:
        rust = list(DEFAULT_RUST_METHODS)
    return families, canonical_rust_methods(rust)


def _guard_out_dir(out_dir: Path, data_root) -> None:
    out = Path(out_dir).resolve()
    forbidden = [(Path(REPO_ROOT / "models").resolve(), "the production model registry")]
    try:
        forbidden.append((fqe_sources.resolve_data_root(data_root), "the public data root"))
    except fqe_sources.FqeSourceError:
        pass
    for root, why in forbidden:
        if out == root or out.is_relative_to(root):
            raise ValueError(f"--out {out} lies inside {why}")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"--out {out} is not empty; results are never merged into an earlier run")


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def code_provenance() -> dict:
    files = ["ml/eval/fqe_benchmark.py", "ml/eval/baselines.py", "ml/data_pipeline/fqe_sources.py",
             "ml/data_pipeline/fqe_source_contracts_v1.json", "ml/labels/psf_measure.py", "ml/preprocess/normalize.py",
             "ml/eval/bge/bridge.py", "achernar-daemon/src/vision/bge_eval.rs", "achernar-daemon/src/vision/quality.rs",
             "achernar-daemon/src/vision/background_dispatch.rs", "achernar-daemon/src/vision/sep_backend.rs"]
    out = {"commit": _git("rev-parse", "HEAD"), "files": {}}
    for rel in files:
        path = REPO_ROOT / rel
        if path.is_file():
            out["files"][rel] = {"sha256": _sha256_file(path),
                                 "dirty": bool(_git("status", "--porcelain", "--", rel))}
    return out


def environment() -> dict:
    import scipy
    import sep

    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "sep": sep.__version__, "platform": platform.platform(), "machine": platform.machine()}


def run_benchmark(sources: Sequence[str], *, out_dir, partition: str = "val", methods: Sequence[str] | None = None,
                  contract=None, data_root=None, run_rust: bool = True, parent_frames: bool = True,
                  bridge_runner: Callable | None = None, cargo_target_dir=None) -> dict:
    """Run the declared comparators on the development population and write the results.

    Writes ``config.json`` (frozen definitions, fits, provenance), ``rows.jsonl``
    (one row per comparator x population member), ``summary.json`` and
    ``populations.json`` (end-to-end matched sample_uids per method) under
    ``out_dir``, which must be new or empty and outside the data root and
    ``models/``.
    """
    if partition in ("test", "test_sealed"):
        raise SealedPartitionError("the classical benchmark never scores the sealed test partition; use --partition val")
    if partition not in DEV_EVAL_PARTITIONS:
        raise ValueError(f"partition must be one of {DEV_EVAL_PARTITIONS} (development population), not {partition!r}")
    if not sources:
        raise ValueError("no sources requested")
    contract = contract or fqe_sources.load_contract()
    out_dir = Path(out_dir)
    _guard_out_dir(out_dir, data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    families, rust_methods = _resolve_families(methods)
    if not run_rust:
        families.discard("native_rust")
    started = datetime.now(timezone.utc).isoformat()

    arrays_by_source, eval_index, identities = {}, {}, {}
    for source_id in sources:
        arrays = fqe_sources.load_source(source_id, partitions=(FIT_PARTITION, partition), contract=contract,
                                         data_root=data_root)
        arrays_by_source[source_id] = arrays
        eval_index[source_id] = arrays.partition_indices(partition)
        ident = {k: arrays.identity.get(k) for k in ("role", "contract_sha256", "contract_file_sha256",
                                                     "experiment_contract_sha256", "files", "npz_sha256",
                                                     "registry_sha256", "target_field", "target_definition_id",
                                                     "label_definition", "secondary_fields", "units", "tier_filter",
                                                     "partition_counts", "sealed_test", "rows_returned_by_partition")}
        ident["eval_population"] = {"partition": partition, "rows": int(eval_index[source_id].size),
                                    "groups": len(set(arrays.group_uid[eval_index[source_id]].tolist())),
                                    "membership_sha256": _membership_sha256(arrays.sample_uid[eval_index[source_id]])}
        identities[source_id] = ident

    arms = build_arms(arrays_by_source, contract)
    rows, fits, specs, selections, runs, native_provenance, notes = [], {}, {}, {}, {}, {}, []
    native_strata: dict = {}

    def add(new_rows):
        for row in new_rows:
            rows.append(row)

    if "constant" in families:
        new, fits["constant"], new_specs = constant_rows(arms, arrays_by_source, eval_index)
        specs.update(new_specs)
        add(new)
    if "normalized_input" in families:
        add(normalized_input_rows(arrays_by_source, eval_index))
    if "sharpness" in families:
        new, fits["sharpness"], new_specs = sharpness_rows(arms, arrays_by_source, eval_index)
        specs.update(new_specs)
        add(new)

    needs_native = families & {"sep_flux_radius", "native_rust"}
    work_dir = out_dir / "bridge"
    runner = bridge_runner
    if "native_rust" in families and runner is None:
        target = Path(cargo_target_dir) if cargo_target_dir else REPO_ROOT / "achernar-daemon" / "target"
        runner = default_bridge_runner(REPO_ROOT, target)

    for source_id, arrays in arrays_by_source.items():
        if not needs_native:
            break
        ev = eval_index[source_id]
        if "array_index" in arrays.joins:
            native = load_sim_native(arrays, ev, contract=contract, data_root=data_root)
            native_provenance[source_id] = native["provenance"]
            obs = native["observations"]
            join_failed = {i: o.join_detail for i, o in obs.items() if not o.join_ok}
            if join_failed:
                notes.append(f"{source_id}: {len(join_failed)} native join mismatch(es)")
            if "sep_flux_radius" in families:
                add(sim_sep_rows(arrays, obs, ev))
            if "native_rust" in families:
                cases, case_rows = [], []
                for k, i in enumerate(ev.tolist()):
                    o = obs[i]
                    case_id = f"sim_{k:05d}"
                    cases.append(BridgeCase(id=case_id, image=o.raw, masks=dict(o.masks),
                                            metadata={"saturation_adu": FROZEN["sim_saturation_adu"],
                                                      "generator_version": SCHEMA, "seed": None}))
                    case_rows.append((case_id, [i], [o.ref_xy]))
                info = run_bridge_cases(cases, rust_methods, work_dir, runner=runner,
                                        label=f"{source_id}__sim_observation64".replace("__", "_"))
                runs[f"{source_id}/sim_observation64"] = info
                for method in rust_methods:
                    spec = rust_spec(method, "sim_observation64", "pixel_flags")
                    specs[spec.comparator_id] = spec
                    new, sel = rust_rows_for_context(spec, arrays, info, case_rows, radius=FROZEN["match"]["sim_radius_px"],
                                                     width=64, height=64, join_failed=join_failed)
                    add(new)
                    selections[f"{source_id}/{spec.comparator_id}"] = sel
        elif "observation_index" in arrays.joins:
            frame_cases, frame_case_rows, crop_cases = [], [], {p: ([], []) for p in ("teacher_dq", "imputed_saturation_only")}
            native_provenance[source_id] = {"frames": {}}
            join_failed = {}
            for frame, crops in iter_hst_native(arrays, ev, contract=contract, data_root=data_root):
                native_provenance[source_id]["frames"][frame.frame_id] = dict(frame.provenance, rows=len(frame.rows))
                for i, o in crops.items():
                    if not o.join_ok:
                        join_failed[i] = o.join_detail
                    native_strata[(source_id, str(arrays.sample_uid[i]))] = {
                        "neighbour_within_24px": o.extra.get("neighbour_within_24px")}
                if "sep_flux_radius" in families:
                    add(hst_teacher_rows(arrays, frame, crops))
                if "native_rust" in families:
                    for policy, (cases, case_rows) in crop_cases.items():
                        for i in frame.rows:
                            o = crops[i]
                            pixels, masks = hst_crop_masks(o, policy)
                            case_id = f"hst_{policy[:7]}_{len(cases):05d}"
                            cases.append(BridgeCase(id=case_id, image=pixels, masks=masks,
                                                    metadata={"saturation_adu": FROZEN["hst_saturation_adu"],
                                                              "generator_version": SCHEMA, "seed": None}))
                            case_rows.append((case_id, [i], [o.ref_xy]))
                    if parent_frames:
                        pixels, masks = frame_masks(frame)
                        case_id = f"frame_{frame.frame_id.lower()}"
                        case = BridgeCase(id=case_id, image=pixels, masks=masks,
                                          metadata={"saturation_adu": FROZEN["hst_saturation_adu"],
                                                    "generator_version": SCHEMA, "seed": None})
                        from eval.bge import bridge as _bridge

                        frame_dir = work_dir / f"{source_id}_hst_parent_frame".replace("__", "_")
                        frame_dir.mkdir(parents=True, exist_ok=True)
                        frame_cases.extend(_bridge.write_inputs([case], frame_dir))
                        frame_case_rows.append((case_id, list(frame.rows), frame.refs_xy))
                del frame
            if "native_rust" in families:
                for policy, (cases, case_rows) in crop_cases.items():
                    if not cases:
                        continue
                    info = run_bridge_cases(cases, rust_methods, work_dir, runner=runner,
                                            label=f"{source_id}_hst_crop64_{policy}")
                    runs[f"{source_id}/hst_crop64/{policy}"] = info
                    for method in rust_methods:
                        spec = rust_spec(method, "hst_crop64", policy)
                        specs[spec.comparator_id] = spec
                        new, sel = rust_rows_for_context(spec, arrays, info, case_rows,
                                                         radius=FROZEN["match"]["hst_radius_px"], width=64, height=64,
                                                         join_failed=join_failed)
                        add(new)
                        selections[f"{source_id}/{spec.comparator_id}"] = sel
                if parent_frames and frame_cases:
                    from eval.bge import bridge as _bridge

                    frame_dir = work_dir / f"{source_id}_hst_parent_frame".replace("__", "_")
                    manifest = _bridge.write_methods_manifest(frame_cases, frame_dir, rust_methods, write_maps=False,
                                                              extras={"benchmark": SCHEMA, "context": "hst_parent_frame"})
                    t0 = time.perf_counter()
                    info = dict(runner(manifest, frame_dir / "out", frame_dir / "bridge.log") or {})
                    info.setdefault("wall_s", time.perf_counter() - t0)
                    info.update(manifest=str(manifest), out_dir=str(frame_dir / "out"), cases=len(frame_cases),
                                methods=list(rust_methods), manifest_sha256=_sha256_file(manifest))
                    runs[f"{source_id}/hst_parent_frame"] = info
                    for method in rust_methods:
                        spec = rust_spec(method, "hst_parent_frame", "teacher_dq")
                        specs[spec.comparator_id] = spec
                        sel_all, new_all = [], []
                        for case_id, indices, refs in frame_case_rows:
                            entry = next(c for c in frame_cases if c["id"] == case_id)
                            new, sel = rust_rows_for_context(spec, arrays, info, [(case_id, indices, refs)],
                                                             radius=FROZEN["match"]["hst_radius_px"],
                                                             width=entry["width"], height=entry["height"],
                                                             join_failed=join_failed)
                            new_all.extend(new)
                            sel_all.extend(sel)
                        add(new_all)
                        selections[f"{source_id}/{spec.comparator_id}"] = sel_all
        else:
            notes.append(f"{source_id}: no native join declared (array_index / observation_index); native comparators "
                         "not run")

    present = {row["comparator_id"] for row in rows}
    for spec in (NORMALIZED_SPEC, SIM_SEP_SPEC, HST_SEP_SPEC):
        if spec.comparator_id in present:
            specs[spec.comparator_id] = spec
    # Native-derived strata (e.g. HST neighbour within 24 px) apply to every comparator's row.
    for row in rows:
        row["strata"].update(native_strata.get((row["source"], row["sample_uid"]), {}))

    check_complete(rows, {s: arrays_by_source[s].sample_uid[eval_index[s]] for s in arrays_by_source})
    summary = summarize_benchmark(rows, arrays_by_source)
    populations = end_to_end_populations(rows)
    config = {
        "schema": SCHEMA, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "partition": partition, "fit_partition": FIT_PARTITION, "sources": identities, "frozen": FROZEN,
        "families": sorted(families), "rust_methods": rust_methods if "native_rust" in families else [],
        "mask_policies": MASK_POLICIES, "arms": {k: v.identity for k, v in arms.items()}, "fits": fits,
        "comparators": {cid: spec.describe() for cid, spec in sorted(specs.items())},
        "bridge_runs": runs, "native_provenance": native_provenance, "notes": notes,
        "per_reference_parent_context_width": "unsupported in bridge v1: no measure-at-positions mode (the daemon's "
                                              "measure_stars is private to quality.rs); parent-frame rows report the "
                                              "end-to-end selection population only",
        "code": code_provenance(), "environment": environment(),
        "seal": "development population only; load_source never returned test rows; no test-era exposure opened",
        "runtime_eligible": False,
    }
    _write_json(out_dir / "config.json", config)
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_clean(row), sort_keys=True) + "\n")
    _write_json(out_dir / "summary.json", {"schema": SCHEMA, "summary": summary, "selections": selections})
    _write_json(out_dir / "populations.json", populations)
    (out_dir / "summary.md").write_text(render_markdown(summary, config))
    return {"config": config, "summary": summary, "rows": rows, "selections": selections, "populations": populations}


def _write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(_clean(obj), indent=2, sort_keys=True))


def check_complete(rows: Sequence[Mapping], populations: Mapping) -> None:
    """Every comparator has exactly one row per population member of each source it scores."""
    by = {}
    for row in rows:
        by.setdefault((row["comparator_id"], row["source"]), []).append(row["sample_uid"])
    for (cid, source), uids in by.items():
        expected = set(np.asarray(populations[source]).astype(str).tolist())
        if len(uids) != len(set(uids)) or set(uids) != expected:
            raise RuntimeError(f"{cid} on {source}: rows do not cover the declared population exactly once "
                               f"({len(uids)} rows, {len(set(uids) & expected)}/{len(expected)} members)")


def summarize_benchmark(rows: Sequence[Mapping], arrays_by_source: Mapping) -> dict:
    out = {}
    for source_id, arrays in arrays_by_source.items():
        src_rows = [r for r in rows if r["source"] == source_id]
        by_cid: dict = {}
        for row in src_rows:
            by_cid.setdefault(row["comparator_id"], []).append(row)
        targets = ["target"] + [name for name in (arrays.secondary or {})]
        per = {}
        for cid, crow in sorted(by_cid.items()):
            entry = {"population_kind": crow[0]["population"], "family": crow[0]["family"],
                     "measurement_role": crow[0]["measurement_role"],
                     "summary": {t: summarize_rows(crow, target=t) for t in targets},
                     "strata": standard_strata(crow)}
            if crow[0]["family"] in ("sharpness", "constant", "normalized_input"):
                entry["sweep_rank_metrics"] = (UNAVAILABLE + ": singleton archive rows have no within-run focus "
                                               "ordering (ranking_eligible false)")
            if crow[0]["timing_scope"] == "frame":
                frames = {}
                for r in crow:
                    if r.get("timing_ms") is not None:
                        frames[r.get("bridge_case")] = r["timing_ms"]
                entry["timing_ms_per_frame"] = frames
            per[cid] = entry
        anchors = [c for c in ("normalized_input__direct_hfr", f"constant__{ARM_OF_SOURCE.get(source_id, source_id + '_only')}")
                   if c in by_cid]
        paired = {}
        for anchor in anchors:
            for cid in sorted(by_cid):
                if cid != anchor:
                    paired[f"{cid} | {anchor}"] = paired_comparison(by_cid, [cid, anchor])
        e2e = [c for c in by_cid if by_cid[c][0]["population"] == POPULATION_END_TO_END]
        matched_views = {}
        for cid in e2e:
            matched = {r["sample_uid"] for r in by_cid[cid] if is_measured(r)}
            stages = Counter(((r.get("match") or {}).get("stage") or r["status"]) for r in by_cid[cid])
            matched_views[cid] = {
                "full_population_coverage": coverage(by_cid[cid]), "population": len(by_cid[cid]),
                "matched": len(matched), "matched_membership_sha256": _membership_sha256(matched),
                "matched_ambiguous": sum(1 for r in by_cid[cid] if is_measured(r) and (r.get("match") or {}).get("ambiguous")),
                "reference_stage_counts": dict(stages),
                "matched_population_summary": summarize_rows([r for r in by_cid[cid] if r["sample_uid"] in matched]),
                "per_crop_comparators_on_matched_population": {
                    other: summarize_rows([r for r in by_cid[other] if r["sample_uid"] in matched])
                    for other in sorted(by_cid) if other not in e2e},
            }
        out[source_id] = {"comparators": per, "paired_vs_anchor": paired, "end_to_end": matched_views}
    return out


def end_to_end_populations(rows: Sequence[Mapping]) -> dict:
    """Matched sample_uids of each end-to-end comparator (for evaluating a CNN on the same rows)."""
    out = {}
    for row in rows:
        if row["population"] != POPULATION_END_TO_END:
            continue
        entry = out.setdefault(f"{row['source']}/{row['comparator_id']}", {"population": 0, "matched": []})
        entry["population"] += 1
        if is_measured(row):
            entry["matched"].append(row["sample_uid"])
    for entry in out.values():
        entry["matched"] = sorted(entry["matched"])
        entry["matched_membership_sha256"] = _membership_sha256(entry["matched"])
    return out


def _fmt(value, spec=".3f") -> str:
    return format(value, spec) if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def render_markdown(summary: Mapping, config: Mapping) -> str:
    lines = [f"# Classical FQE development benchmark ({config['partition']})", "",
             f"Started {config['started_utc']}; commit {config['code']['commit']}. Development population only; "
             "test sealed. Per-row outputs: rows.jsonl; definitions and fits: config.json.", ""]
    for source_id, block in summary.items():
        pop = config["sources"][source_id]["eval_population"]
        lines += [f"## {source_id} ({pop['rows']} rows, {pop['groups']} groups; reference "
                  f"{config['sources'][source_id]['target_definition_id']})", "",
                  "| comparator | role | population | coverage | measured | HFR MAE px | log MAE | bias px | "
                  "median rel err | top unmeasured reason |", "|---|---|---|---|---|---|---|---|---|---|"]
        for cid, entry in block["comparators"].items():
            s = entry["summary"]["target"]
            m = s["metrics"]
            reason = next(iter(s["unmeasured_reasons"]), "")
            lines.append(f"| {cid} | {entry['measurement_role']} | {entry['population_kind']} | {_fmt(s['coverage'])} | "
                         f"{s['measured']}/{s['population']} | {_fmt(m['hfr_mae_px'])} | {_fmt(m['log_hfr_mae'])} | "
                         f"{_fmt(m['signed_bias_px'])} | {_fmt(m['median_relative_error'])} | {reason[:70]} |")
        lines.append("")
    lines += ["Rank/sweep metrics: unavailable (singleton archive rows). Per-reference parent-context width: "
              + config["per_reference_parent_context_width"] + ".", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Classical FQE development benchmark (contract-v1 M2).")
    parser.add_argument("--partition", required=True, help="development evaluation partition: val (test is refused)")
    parser.add_argument("--sources", required=True, help="comma-separated fqe_sources source ids")
    parser.add_argument("--out", required=True, help="new/empty output directory (outside the data root and models/)")
    parser.add_argument("--methods", nargs="*", default=None,
                        help=f"families {FAMILIES} and/or Rust method ids (default: all families, "
                             f"{', '.join(DEFAULT_RUST_METHODS)})")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--contract", default=None)
    parser.add_argument("--no-rust", action="store_true", help="skip the native Rust bridge comparators")
    parser.add_argument("--no-parent-frames", action="store_true", help="skip the HST parent-frame bridge context")
    parser.add_argument("--cargo-target-dir", default=None)
    args = parser.parse_args(argv)
    if args.partition in ("test", "test_sealed"):
        print("refused: --partition test. The classical benchmark is development-only; the sealed test partition "
              "is consumed only through fqe_sources.load_final_test under a separate owner instruction.",
              file=sys.stderr)
        return 2
    if args.partition not in DEV_EVAL_PARTITIONS:
        print(f"refused: --partition {args.partition!r}; expected one of {DEV_EVAL_PARTITIONS}", file=sys.stderr)
        return 2
    contract = fqe_sources.load_contract(args.contract)
    result = run_benchmark([s for s in args.sources.split(",") if s], out_dir=args.out, partition=args.partition,
                           methods=args.methods, contract=contract, data_root=args.data_root,
                           run_rust=not args.no_rust, parent_frames=not args.no_parent_frames,
                           cargo_target_dir=args.cargo_target_dir)
    print(render_markdown(result["summary"], result["config"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
