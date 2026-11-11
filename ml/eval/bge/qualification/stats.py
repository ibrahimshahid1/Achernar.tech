"""Paired comparisons, bootstrap intervals and statistical-power notes.

With a few nights only margin-based claims are possible. A bootstrap CI is
computed only with ``MIN_BOOTSTRAP_UNITS`` or more independent units (nights,
or sweeps/groups as stated, which are clustered within nights); otherwise the
comparison says why there is none. No significance test is predeclared, so
none is reported.
"""

from __future__ import annotations

import math

import numpy as np

MIN_BOOTSTRAP_UNITS = 5
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261111


def bootstrap_ci(values, stat=np.median, n=N_BOOTSTRAP, seed=BOOTSTRAP_SEED, level=0.95):
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=np.float64)
    if v.size < MIN_BOOTSTRAP_UNITS:
        return None, f"n={v.size} units < {MIN_BOOTSTRAP_UNITS}: too few for a bootstrap CI (margin claim only)"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n, v.size))
    boots = np.apply_along_axis(stat, 1, v[idx])
    lo, hi = np.percentile(boots, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return [float(lo), float(hi)], None


def paired(subject: dict, comparator: dict, unit: str) -> dict:
    """Paired differences subject - comparator over the units both have."""
    keys = sorted(k for k in subject if k in comparator and subject[k] is not None and comparator[k] is not None
                  and math.isfinite(subject[k]) and math.isfinite(comparator[k]))
    diffs = [subject[k] - comparator[k] for k in keys]
    out = {"unit": unit, "n_units": len(keys), "units": keys,
           "median_difference": float(np.median(diffs)) if diffs else None,
           "mean_difference": float(np.mean(diffs)) if diffs else None,
           "subject_median": float(np.median([subject[k] for k in keys])) if keys else None,
           "comparator_median": float(np.median([comparator[k] for k in keys])) if keys else None}
    ci, why = bootstrap_ci(diffs)
    out["bootstrap_ci95_median_difference"] = ci
    if why:
        out["ci_note"] = why
    elif unit != "night":
        out["ci_note"] = f"bootstrap over {unit}s, which are clustered within nights: optimistic"
    return out


def power_note(criterion_id: str, n_nights_evaluable: int, n_nights_total: int, extra: str | None = None,
               split: str = "hold-out") -> str:
    k = n_nights_evaluable
    if k == 0:
        base = f"no evaluable {split} night (of {n_nights_total})"
    elif k == 1:
        base = "1 evaluable night: margin claim only; no between-night variability is observed"
    elif k < MIN_BOOTSTRAP_UNITS:
        base = (f"{k} evaluable nights: per-night margin claims; too few nights for a bootstrap CI or any "
                f"significance statement")
    else:
        base = f"{k} evaluable nights: per-night margin claims, bootstrap CI over nights descriptive only"
    if n_nights_evaluable < n_nights_total:
        base += f" ({n_nights_total - n_nights_evaluable} night(s) not evaluable)"
    return base + (f"; {extra}" if extra else "")


def binomial_se(p, n) -> float | None:
    if p is None or not n:
        return None
    return math.sqrt(max(p * (1 - p), 0.0) / n)
