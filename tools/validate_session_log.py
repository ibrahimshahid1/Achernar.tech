"""Check a field session log before import (achernar_data_collection.pdf, Appendix C).

Run it the same night you transcribe the paper card, while a missing entry is
still recoverable from memory or the phone photos:

    python3 tools/validate_session_log.py ~/coc/2026-10-08/session.json

Exit status 1 means the import would produce wrong or unjoinable rows. Warnings
are things the importer tolerates but a model will be weaker for. Nothing here
fills a value in: unknown stays null (Rule 2).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_session import parse_utc_offset  # noqa: E402

RIGS = {"R1", "R2", "R3", "R4", "R5"}
RUN_TYPES = {"focus_sweep", "trailing_ladder", "cloud_block", "drift", "bandit", "calibration"}
DISTURBANCE_KINDS = {"bump", "refocus", "cloud", "headlights", "dew", "battery", "wind_gust", "other"}
# A placeholder left over from the template. Rejected rather than imported.
PLACEHOLDER = "YYYY"


def _aware(value, where: str, errors: list[str]) -> datetime | None:
    if not isinstance(value, str) or PLACEHOLDER in value:
        errors.append(f"{where}: missing or still a template placeholder")
        return None
    try:
        t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{where}: not an ISO 8601 timestamp: {value!r}")
        return None
    if t.tzinfo is None:
        errors.append(f"{where}: needs a UTC offset (e.g. -04:00) so it can be matched to frames")
        return None
    return t


def validate(log: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    session = log.get("session") or {}
    runs = log.get("runs") or []
    environment = log.get("environment") or []
    calibration_only = bool(runs) and all(r.get("run_type") == "calibration" for r in runs)

    night = session.get("night_id")
    try:
        date.fromisoformat(str(night))
    except ValueError:
        errors.append("session.night_id must be YYYY-MM-DD, the local date of sunset")
    if session.get("rig_id") not in RIGS:
        errors.append(f"session.rig_id must be one of {sorted(RIGS)}")
    if not session.get("camera"):
        errors.append("session.camera is required")

    offset = session.get("camera_utc_offset")
    if offset is None:
        warnings.append(
            "session.camera_utc_offset is unset: frames without an EXIF offset will "
            "be timed in this computer's zone (and flagged host_local_assumed)"
        )
    elif parse_utc_offset(offset) is None:
        errors.append(f"session.camera_utc_offset must look like -04:00, got {offset!r}")

    if not calibration_only:
        for key in ("site_lat", "site_lon"):
            if session.get(key) is None:
                warnings.append(f"session.{key} is null: moon/airmass covariates cannot be computed")
    if session.get("saturation_adu") is None:
        warnings.append(
            "session.saturation_adu is null: fine for cloud/background data, but the "
            "FQE builder will drop these frames until BP-2 measures it"
        )

    for i, e in enumerate(environment):
        _aware(e.get("t"), f"environment[{i}].t", errors)
        sky = e.get("sky_class")
        if sky is not None and sky not in range(5):
            errors.append(f"environment[{i}].sky_class must be 0-4 or null")
        wind = e.get("wind_beaufort")
        if wind is not None and not (isinstance(wind, int) and 0 <= wind <= 12):
            errors.append(f"environment[{i}].wind_beaufort must be an integer 0-12 or null")
        humidity = e.get("humidity_pct")
        if humidity is not None and not 0 <= humidity <= 100:
            errors.append(f"environment[{i}].humidity_pct must be 0-100 or null")

    run_ids: set[str] = set()
    for i, r in enumerate(runs):
        run_id = r.get("run_id")
        if not run_id or PLACEHOLDER in str(run_id):
            errors.append(f"runs[{i}].run_id is required")
        elif run_id in run_ids:
            errors.append(f"runs[{i}].run_id {run_id!r} is used twice")
        else:
            run_ids.add(run_id)
        if r.get("run_type") not in RUN_TYPES:
            errors.append(f"runs[{i}].run_type must be one of {sorted(RUN_TYPES)}")
        if r.get("status") not in ("completed", "aborted"):
            errors.append(f"runs[{i}].status must be 'completed' or 'aborted'")
        if r.get("status") == "aborted" and not r.get("abort_reason"):
            errors.append(f"runs[{i}] is aborted but has no abort_reason")
        if r.get("exposure_s") is None and session.get("exposure_s") is None:
            warnings.append(
                f"runs[{i}] has no exposure_s: RAW-only (CR3) frames carry no "
                "readable EXIF, so the log is their only exposure record"
            )

    if any(r.get("run_type") == "cloud_block" for r in runs):
        labels = [e for e in environment if e.get("sky_class") is not None]
        if not labels:
            errors.append("cloud_block run but no sky_class labels in environment[]")
        elif len(labels) < 3:
            warnings.append(f"only {len(labels)} sky_class label(s); RB-5 wants one every 10 minutes")

    for i, d in enumerate(log.get("disturbances") or []):
        _aware(d.get("t"), f"disturbances[{i}].t", errors)
        if d.get("kind") not in DISTURBANCE_KINDS:
            errors.append(f"disturbances[{i}].kind must be one of {sorted(DISTURBANCE_KINDS)}")
        affected = d.get("affected_run_id")
        if affected is not None and affected not in run_ids:
            errors.append(f"disturbances[{i}].affected_run_id {affected!r} matches no run")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    try:
        log = json.loads(args.log.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR cannot read {args.log}: {exc}")
        return 1
    if not isinstance(log, dict):
        print("ERROR session log must be a JSON object")
        return 1
    errors, warnings = validate(log)
    for msg in errors:
        print(f"ERROR {msg}")
    for msg in warnings:
        print(f"warn  {msg}")
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
