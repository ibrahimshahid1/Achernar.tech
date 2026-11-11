"""Load ``bge-real-thresholds-v2`` and evaluate criteria R1-R17 and R17b per night.

The numbers live in the JSON (frozen in git before a hold-out run); the
formula for each criterion is implemented here (frozen with the code). Each
criterion returns, per night, ``pass`` / ``fail`` / ``not_evaluable`` with the
values compared and the reason, and an overall verdict with a power note.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

from . import THRESHOLDS_SCHEMA_V2, sha256_file
from .stats import binomial_se, paired, power_note

CRITERIA = tuple(f"R{i}" for i in range(1, 18)) + ("R17b",)
ROLES = ("sep_v2", "legacy_background", "legacy_gate")


class ThresholdsError(ValueError):
    pass


def load_thresholds(path, now=None) -> dict:
    th = json.loads(Path(path).read_text())
    if th.get("schema") != THRESHOLDS_SCHEMA_V2:
        raise ThresholdsError(f"{path}: schema must be {THRESHOLDS_SCHEMA_V2!r}")
    try:
        declared = dt.datetime.fromisoformat(th["predeclared_utc"])
    except (KeyError, ValueError) as exc:
        raise ThresholdsError(f"{path}: no valid predeclared_utc ({exc})") from exc
    if declared.tzinfo is None:
        raise ThresholdsError("predeclared_utc must carry a timezone")
    if declared >= (now or dt.datetime.now(dt.timezone.utc)):
        raise ThresholdsError("thresholds are not predeclared (predeclared_utc is not in the past)")
    ids = [c.get("id") for c in th.get("criteria", [])]
    if tuple(ids) != CRITERIA:
        raise ThresholdsError(f"criteria must be exactly {CRITERIA} in order, got {ids}")
    for role in ROLES:
        if not th.get("variants", {}).get(role):
            raise ThresholdsError(f"variants.{role} missing")
    th["_sha256"] = sha256_file(path)
    th["_path"] = str(path)
    return th


def _get(d, dotted):
    for key in dotted.split("."):
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def _ne(reason, **values):
    return {"status": "not_evaluable", "reason": reason, "values": values}


def _verdict(ok, **values):
    return {"status": "pass" if ok else "fail", "values": values}


def _num(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def _r_simple(c, subj, comp, ctx, min_key, count_path):
    n = _get(subj, count_path)
    need = c["min_data"][min_key]
    if not n or n < need:
        return _ne(f"{n or 0} {min_key} < {need}")
    v = _get(subj, c["metric"])
    if not _num(v):
        return _ne(f"{c['metric']} unavailable")
    if c["op"] == "<=":
        return _verdict(v <= c["value"], value=v, threshold=c["value"])
    if c["op"] == "abs<=":
        return _verdict(abs(v) <= c["value"], value=v, threshold=c["value"])
    raise ThresholdsError(f"{c['id']}: unsupported op {c['op']}")


def _within_s0(c, subj, ctx, min_key, count_path):
    n = _get(subj, count_path)
    need = c["min_data"][min_key]
    if not n or n < need:
        return _ne(f"{n or 0} {min_key} < {need}")
    v = _get(subj, c["metric"])
    if not _num(v):
        return _ne(f"{c['metric']} unavailable")
    s0 = ctx["s0"]
    return _verdict(abs(v - s0["value"]) <= c["tolerance"], value=v, s0=s0["value"],
                    s0_calibrated=s0["calibrated"], tolerance=c["tolerance"],
                    **({} if s0["calibrated"] else {"flag": "s0_uncalibrated (s0 = 1.0)"}))


def r1(c, s, k, ctx):
    return _r_simple(c, s, k, ctx, "frames", "pull.n_frames")


def r2(c, s, k, ctx):
    return _within_s0(c, s, ctx, "frames", "pull.n_frames")


def r3(c, s, k, ctx):
    return _r_simple(c, s, k, ctx, "frames", "pull.n_frames")


def r4(c, s, k, ctx):
    return _within_s0(c, s, ctx, "pairs", "diff_pull.n_pairs")


def r5(c, s, k, ctx):
    need = c["min_data"]["frames"]
    for who, m in (("subject", s), ("comparator", k)):
        if (_get(m, "negated.n_frames") or 0) < need:
            return _ne(f"{who}: {_get(m, 'negated.n_frames') or 0} frames < {need}")
    v, ref = _get(s, c["metric"]), _get(k, c["metric"])
    if not (_num(v) and _num(ref)):
        return _ne("rate unavailable")
    rhs = max(c["floor_per_mpx"], ref)
    return _verdict(v <= rhs, value=v, comparator=ref, threshold=rhs)


def r6(c, s, k, ctx):
    sb, kb = _get(s, "injection.bins"), _get(k, "injection.bins")
    if not sb or not kb:
        return _ne("no injection-recovery data")
    checked, ok, why = [], True, []
    for a, b in zip(sb, kb):
        if a["lo"] < c["min_bin_lower_edge"]:
            continue
        if a["n_injected"] < c["min_injections_per_bin"]:
            why.append(f"bin {a['bin']}: {a['n_injected']} injections < {c['min_injections_per_bin']}")
            continue
        if a["completeness"] is None or b["completeness"] is None:
            why.append(f"bin {a['bin']}: completeness unavailable")
            continue
        passed = a["completeness"] >= b["completeness"] - c["margin"]
        ok &= passed
        checked.append({"bin": a["bin"], "n": a["n_injected"], "subject": a["completeness"],
                        "comparator": b["completeness"], "pass": passed,
                        "binomial_se_subject": binomial_se(a["completeness"], a["n_injected"])})
    n_bins = sum(1 for a in sb if a["lo"] >= c["min_bin_lower_edge"])
    if len(checked) < n_bins:
        return _ne("; ".join(why), bins=checked)
    return _verdict(ok, bins=checked, margin=c["margin"])


def r7(c, s, k, ctx):
    path = "injection.snr_ge_20"
    n_s, n_k = _get(s, path + ".n_hfr") or 0, _get(k, path + ".n_hfr") or 0
    need = c["min_data"]["matched"]
    if n_s < need or n_k < need:
        return _ne(f"matched accepted injections subject {n_s} / comparator {n_k} < {need}")
    v, ref = _get(s, c["metric"]), _get(k, c["metric"])
    if not (_num(v) and _num(ref)):
        return _ne("HFR bias unavailable")
    ok = abs(v) <= c["value"] and abs(v) <= abs(ref) + c["comparator_margin"]
    return _verdict(ok, value=v, comparator=ref, abs_limit=c["value"], comparator_margin=c["comparator_margin"])


def r8(c, s, k, ctx):
    return _r_simple(c, s, k, ctx, "measured", "injection.snr_ge_20.n_flux")


def r9(c, s, k, ctx):
    return _r_simple(c, s, k, ctx, "matched", "injection.snr_ge_20.n_centroid")


def r10(c, s, k, ctx):
    return _r_simple(c, s, k, ctx, "stars", "injection.delta_b.n")


def r11(c, s, k, ctx):
    trials_s = _get(s, "dark_injection.trials") or {}
    trials_k = _get(k, "dark_injection.trials") or {}
    need = c["min_data"]["trials"]
    ok_s = {t: v for t, v in trials_s.items() if v.get("status") == "completed"}
    if len(ok_s) < need:
        return _ne(f"{len(ok_s)} completed dark trials < {need}")
    bias = float(np.median([abs(v["bias_sigma"]) for v in ok_s.values()]))
    common = [t for t in ok_s if trials_k.get(t, {}).get("status") == "completed"]
    if len(common) < need:
        return _ne(f"{len(common)} dark trials completed by both < {need}", median_abs_bias_sigma=bias)
    rmse_s = float(np.median([ok_s[t]["background"]["rmse_adu"] for t in common]))
    rmse_k = float(np.median([trials_k[t]["background"]["rmse_adu"] for t in common]))
    return _verdict(bias <= c["value"] and rmse_s <= rmse_k, median_abs_bias_sigma=bias, threshold=c["value"],
                    median_rmse_adu=rmse_s, comparator_median_rmse_adu=rmse_k, n_trials=len(common))


def r12(c, s, k, ctx):
    need = c["min_data"]["frames"]
    for who, m in (("subject", s), ("comparator", k)):
        if (m.get("n_lights") or 0) < need:
            return _ne(f"{who}: {m.get('n_lights') or 0} light frames < {need}")
    v, ref = s.get("status_failure_rate"), k.get("status_failure_rate")
    if not (_num(v) and _num(ref)):
        return _ne("status-failure rate unavailable")
    ok = v <= c["value"] and v <= ref + c["comparator_margin"]
    return _verdict(ok, value=v, comparator=ref, limit=c["value"], comparator_margin=c["comparator_margin"])


def r13(c, s, k, ctx):
    gs, gk = _get(s, "repeatability.groups") or {}, _get(k, "repeatability.groups") or {}
    common = [g for g in gs if _num(gs[g].get("hfr_cv")) and _num((gk.get(g) or {}).get("hfr_cv"))]
    need = c["min_data"]["groups"]
    if len(common) < need:
        return _ne(f"{len(common)} repeat groups with an HFR CV for both < {need}")
    v = float(np.median([gs[g]["hfr_cv"] for g in common]))
    ref = float(np.median([gk[g]["hfr_cv"] for g in common]))
    return _verdict(v <= c["ratio"] * ref, value=v, comparator=ref, ratio=c["ratio"], n_groups=len(common))


def _common_sweeps(s, k):
    ss, sk = _get(s, "focus.sweeps") or {}, _get(k, "focus.sweeps") or {}
    return ss, sk, [r for r in ss if ss[r].get("status") == "completed" and (sk.get(r) or {}).get("status") == "completed"]


def r14(c, s, k, ctx):
    need = c["min_data"]["sweeps"]
    n_s, n_k = _get(s, "focus.n_fits") or 0, _get(k, "focus.n_fits") or 0
    if n_s < need or n_k < need:
        return _ne(f"successful sweep fits subject {n_s} / comparator {n_k} < {need}")
    v, ref, cfz = _get(s, "focus.p0_std"), _get(k, "focus.p0_std"), _get(s, "focus.cfz_steps")
    if not (_num(v) and _num(ref)):
        return _ne("p0 scatter unavailable")
    if not _num(cfz):
        return _ne("CFZ unknown (focal_ratio or focuser_stepsize_um missing)", value=v, comparator=ref)
    ok = v <= c["ratio"] * ref and v <= c["max_cfz_fraction"] * cfz
    return _verdict(ok, value=v, comparator=ref, cfz_steps=cfz, ratio=c["ratio"], max_cfz_fraction=c["max_cfz_fraction"])


def r15(c, s, k, ctx):
    ss, sk, common = _common_sweeps(s, k)
    need = c["min_data"]["sweeps"]
    if len(common) < need:
        return _ne(f"{len(common)} sweeps fitted by both < {need}")
    cfz = _get(s, "focus.cfz_steps")
    if not _num(cfz):
        return _ne("CFZ unknown (focal_ratio or focuser_stepsize_um missing)")
    v = float(np.median([abs(ss[r]["p0"] - sk[r]["p0"]) for r in common]))
    return _verdict(v <= c["max_cfz_fraction"] * cfz, value=v, cfz_steps=cfz, threshold=c["max_cfz_fraction"] * cfz,
                    n_sweeps=len(common))


def r16(c, s, k, ctx):
    ss, sk, common = _common_sweeps(s, k)
    need = c["min_data"]["sweeps"]
    if len(common) < need:
        return _ne(f"{len(common)} sweeps fitted by both < {need}")
    v = float(np.median([ss[r]["rms_residual_px"] for r in common]))
    ref = float(np.median([sk[r]["rms_residual_px"] for r in common]))
    return _verdict(v <= c["ratio"] * ref, value=v, comparator=ref, ratio=c["ratio"], n_sweeps=len(common))


def r17(c, s, k, ctx):
    need = c["min_data"]["sweep_frames"]
    n = _get(s, "focus.coverage.n_frames") or 0
    if n < need:
        return _ne(f"{n} sweep frames < {need}")
    v, ref = _get(s, c["metric"]), _get(k, c["metric"])
    if not (_num(v) and _num(ref)):
        return _ne("coverage unavailable for subject or comparator")
    return _verdict(v >= c["value"] and v >= ref, value=v, comparator=ref, threshold=c["value"])


def r17b(c, s, k, ctx):
    need = c["min_data"]["sweep_frames"]
    n = _get(s, "focus.coverage.n_frames") or 0
    if n < need:
        return _ne(f"{n} sweep frames < {need}")
    v = _get(s, c["metric"])
    if not _num(v):
        return _ne("coverage unavailable")
    return _verdict(v >= c["value"], value=v, threshold=c["value"])


IMPL = {"R1": r1, "R2": r2, "R3": r3, "R4": r4, "R5": r5, "R6": r6, "R7": r7, "R8": r8, "R9": r9, "R10": r10,
        "R11": r11, "R12": r12, "R13": r13, "R14": r14, "R15": r15, "R16": r16, "R17": r17, "R17b": r17b}
EXTRA_POWER = {
    "R6": "per-bin binomial SE = sqrt(p(1-p)/n) is given per night; 400 injections give SE <= 0.025",
    "R14": "std over n sweeps has relative SE ~ 1/sqrt(2(n-1)) (35 % at n = 5)",
    "R13": "CV over few frames per group is itself noisy",
}


def evaluate(thresholds: dict, night_metrics: dict, s0: dict, roles: dict | None = None, split: str = "hold-out") -> dict:
    """``night_metrics``: ``{night: {method_id: metrics}}``. ``s0``: ``{method_id: {"value", "calibrated"}}``."""
    roles = roles or thresholds["variants"]
    out = {}
    for c in thresholds["criteria"]:
        subj_id = roles[c["subject"]]
        comp_id = roles.get(c.get("comparator")) if c.get("comparator") else None
        per_night = {}
        for night, methods in sorted(night_metrics.items()):
            s = methods.get(subj_id)
            k = methods.get(comp_id) if comp_id else {}
            if s is None or s.get("availability", "available") != "available":
                per_night[night] = _ne(f"subject {subj_id} unavailable: {(s or {}).get('availability')}")
                continue
            if comp_id and (k is None or k.get("availability", "available") != "available"):
                per_night[night] = _ne(f"comparator {comp_id} unavailable: {(k or {}).get('availability')}")
                continue
            ctx = {"s0": s0.get(subj_id) or {"value": 1.0, "calibrated": False}}
            per_night[night] = IMPL[c["id"]](c, s, k or {}, ctx)
        statuses = [v["status"] for v in per_night.values()]
        n_eval = sum(st != "not_evaluable" for st in statuses)
        overall = ("not_evaluable" if n_eval == 0 else "fail" if "fail" in statuses else "pass")
        out[c["id"]] = {"title": c["title"], "status": overall, "subject": subj_id, "comparator": comp_id,
                        "per_night": per_night, "n_nights": len(per_night), "n_nights_evaluable": n_eval,
                        "statistical_power": power_note(c["id"], n_eval, len(per_night), EXTRA_POWER.get(c["id"]),
                                                        split),
                        "provenance": c.get("provenance")}
    return out


PAIRED_METRICS = {
    "pull.median_abs_clipped_mean": "night", "pull.median_robust_std": "night", "pull.median_frac_abs_gt5": "night",
    "diff_pull.median_robust_std": "night", "negated.rate_per_mpx": "night",
    "negated.fraction_near_bright": "night", "status_failure_rate": "night",
    "injection.snr_ge_20.median_rel_hfr": "night", "injection.snr_ge_20.median_rel_flux": "night",
    "injection.snr_ge_20.median_centroid_px": "night", "injection.delta_b.median_sigma": "night",
    "dark_injection.median_abs_bias_sigma": "night", "dark_injection.median_rmse_adu": "night",
    "repeatability.median_hfr_cv": "night", "focus.p0_std": "night", "focus.coverage.fraction": "night",
}


def observations(thresholds: dict, night_metrics: dict, roles: dict | None = None) -> dict:
    """Reported-only quantities (e.g. R7-obs): per role and night, no verdict."""
    roles = roles or thresholds["variants"]
    out = {}
    for o in thresholds.get("observations", []):
        rows = {}
        for role in o["roles"]:
            mid = roles[role]
            rows[role] = {"method": mid, "per_night": {n: _get(ms.get(mid) or {}, o["metric"])
                                                       for n, ms in sorted(night_metrics.items())}}
        out[o["id"]] = {"title": o["title"], "statistic": o["statistic"], "values": rows,
                        "note": "observation only: no threshold, no pass/fail"}
    return out


GATE_METRICS = ("focus.coverage.fraction",)


def paired_comparisons(night_metrics: dict, methods, comparator: str, gate_comparator: str | None = None) -> dict:
    """Every metric, every method vs the legacy comparator (``gate_comparator``
    for gate metrics), plus per-sweep and per-repeat-group units."""
    gate_comparator = gate_comparator or comparator

    def per_night(m, metric):
        return {n: v for n, v in ((n, _get(ms.get(m) or {}, metric)) for n, ms in night_metrics.items()) if _num(v)}

    def collect(m, path, key):
        out = {}
        for ms in night_metrics.values():
            for unit, rec in (_get(ms.get(m) or {}, path) or {}).items():
                if path == "focus.sweeps" and rec.get("status") != "completed":
                    continue
                if _num(rec.get(key)):
                    out[unit] = rec[key]
        return out

    out = {}
    for m in methods:
        rows = {}
        for metric, unit in PAIRED_METRICS.items():
            comp = gate_comparator if metric in GATE_METRICS else comparator
            if m != comp:
                rows[metric] = dict(paired(per_night(m, metric), per_night(comp, metric), unit), comparator=comp)
        if m != comparator:
            for name, path, key, unit in (("sweep.p0", "focus.sweeps", "p0", "sweep"),
                                          ("sweep.rms_residual_px", "focus.sweeps", "rms_residual_px", "sweep"),
                                          ("group.hfr_cv", "repeatability.groups", "hfr_cv", "repeat group")):
                rows[name] = dict(paired(collect(m, path, key), collect(comparator, path, key), unit),
                                  comparator=comparator)
        if rows:
            out[m] = rows
    return out
