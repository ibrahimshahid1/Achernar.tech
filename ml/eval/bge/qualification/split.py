"""Night split rule (``split-rule-v1.json``).

Nights are ordered chronologically by the earliest ``timestamp_utc`` of any
frame assigned to them; positions 2, 4, 6, ... (1-based) are hold-out and the
rest development. Calibration frames carry their night's split and a night is
never split. The rule is deterministic: the same set of (night_id,
timestamp) pairs always gives the same assignment, whatever the frame order.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from . import DEFAULT_SPLIT_RULE, SPLIT_RULE_SCHEMA, SPLITS


class SplitRuleError(ValueError):
    pass


def load_rule(path=DEFAULT_SPLIT_RULE) -> dict:
    rule = json.loads(Path(path).read_text())
    if rule.get("schema") != SPLIT_RULE_SCHEMA:
        raise SplitRuleError(f"{path}: expected schema {SPLIT_RULE_SCHEMA!r}")
    every, first = rule.get("holdout_every"), rule.get("holdout_first_position")
    if not (isinstance(every, int) and every >= 2 and isinstance(first, int) and 1 <= first <= every):
        raise SplitRuleError(f"{path}: holdout_every >= 2 and 1 <= holdout_first_position <= holdout_every required")
    return rule


def parse_utc(text):
    """An aware datetime from an ISO-8601 string (``Z`` allowed), or None."""
    if text is None or text == "":
        return None
    try:
        value = dt.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if value.tzinfo is None:
        return None
    return value.astimezone(dt.timezone.utc)


def night_order(frames) -> list:
    """``[(night_id, earliest_utc_iso_or_None)]`` in rule order."""
    first = {}
    for f in frames:
        night = f.get("night_id")
        if not night:
            raise SplitRuleError(f"frame {f.get('id')!r} has no night_id")
        t = parse_utc(f.get("timestamp_utc"))
        if night not in first:
            first[night] = t
        elif t is not None and (first[night] is None or t < first[night]):
            first[night] = t
    timed = sorted((t, n) for n, t in first.items() if t is not None)
    untimed = sorted(n for n, t in first.items() if t is None)
    return [(n, t.isoformat()) for t, n in timed] + [(n, None) for n in untimed]


def assign_splits(frames, rule=None) -> dict:
    """``{night_id: "development" | "holdout"}`` by the rule."""
    rule = rule or load_rule()
    every, first = rule["holdout_every"], rule["holdout_first_position"]
    out = {}
    for index, (night, _t) in enumerate(night_order(frames)):
        position = index + 1
        out[night] = "holdout" if (position - first) % every == 0 and position >= first else "development"
    return out


def apply_rule(frames, rule=None) -> list:
    """Return the night table and set each frame's ``split`` in place."""
    rule = rule or load_rule()
    order = night_order(frames)
    splits = assign_splits(frames, rule)
    for f in frames:
        f["split"] = splits[f["night_id"]]
    counts = {}
    for f in frames:
        counts[f["night_id"]] = counts.get(f["night_id"], 0) + 1
    return [{"night_id": n, "position": i + 1, "first_timestamp_utc": t, "split": splits[n], "n_frames": counts[n]}
            for i, (n, t) in enumerate(order)]


def check_manifest(frames, rule=None) -> list:
    """Mismatches between the frames' recorded split and the rule (empty if consistent)."""
    rule = rule or load_rule()
    problems = []
    by_night = {}
    for f in frames:
        if f.get("split") not in SPLITS:
            problems.append(f"frame {f.get('id')!r}: split {f.get('split')!r} is not one of {SPLITS}")
        by_night.setdefault(f["night_id"], set()).add(f.get("split"))
    for night, splits in by_night.items():
        if len(splits) > 1:
            problems.append(f"night {night}: frames carry more than one split {sorted(map(str, splits))}")
    expected = assign_splits(frames, rule)
    for night, splits in sorted(by_night.items()):
        if len(splits) == 1 and next(iter(splits)) != expected[night]:
            problems.append(f"night {night}: manifest says {next(iter(splits))!r}, rule says {expected[night]!r}")
    return problems
