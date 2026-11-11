"""Manifest builders: directory + sidecar log, import_session DB, daemon DB + capture archive.

Every builder reads metadata only (SQLite opened read-only, JSON descriptors,
file names, sha256 of files). None decodes a RAW: the analysis input must be
the representation the daemon analyses, which the daemon archives per capture
as ``analysis/linear.fits`` (float32, ``achernar-linear-bilinear-rgb-mean-v1``)
with ``analysis/{bad_pixel,invalid,edge,native_saturation}.fits`` and the
descriptor ``analysis/analysis.json`` (``storage/analysis_assets.rs``,
``fits.rs::raw_analysis_assets``, ``api/camera.rs``, ``api/focus.rs``). A frame
without such a bundle is listed under ``excluded`` with the reason.

Capture archive layout (``storage/capture_store.rs``)::

    <frames_dir>/captures/<rig>/<night>/<session|no-session>/<frame-id>/
        manifest.v1.json
        native/<camera file>          (native_raw | camera_jpeg)
        derived/image.fits|sensor.fits (sensor_fits: CFA mosaic, NOT the analysis input)
        analysis/linear.fits, analysis/<mask>.fits, analysis/analysis.json, analysis/luma.png

``frame_assets.relative_path`` is relative to the capture root
(``<frames_dir>/captures``). ``tools/import_session.py`` writes into the same
``frames``/``frame_assets``/``focus_runs``/``focus_samples`` tables with
``driver_name = 'offline_import'`` and paths ``<night>/<type>/<id>_<role>.*``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from pathlib import Path

from . import (DAEMON_ANALYSIS_REPRESENTATION, DEFAULT_SPLIT_RULE, MANIFEST_SCHEMA_V2, MASK_NAMES,
               QUALIFICATION_VERSION, REPO_ROOT, sha256_file)
from . import split as SPLIT
from .manifest import validate_manifest
from .sidecar import SidecarLog, pointing_from

RAW_SUFFIXES = {".cr2", ".cr3", ".nef", ".arw", ".rw2", ".orf", ".raf", ".dng", ".pef", ".srw"}
FITS_SUFFIXES = {".fits", ".fit", ".fts"}
CAPTURE_TYPE_TO_FRAME_TYPE = {"light": "light", "dark": "dark", "bias": "bias", "flat": "flat", "focus": "light",
                              "verification": "light"}
EXCLUDED_CAPTURE_TYPES = {"preview"}
EXCLUDED_DOMAINS = {"simulation", "preview"}
#: Fields a DB records authoritatively; the sidecar may only fill them when null.
DB_AUTHORITATIVE = ("timestamp_utc", "exposure_s", "iso", "gain", "focus_position", "focus_run_id", "camera",
                    "rig_id", "sensor_temp_c")
SIDECAR_COPY = ("rig_id", "camera", "optical_train", "iso", "gain", "exposure_s", "sensor_temp_c", "ambient_temp_c",
                "focus_position", "focuser_stepsize_um", "focal_ratio", "focus_run_id", "sweep_direction",
                "repeat_group", "notes", "saturation_adu", "gain_e_per_adu", "timestamp_utc")
LUMA_REPRESENTATION = "import_session_luma_unverified"


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _import_session():
    from .manifest import import_session_module

    return import_session_module()


def utc_iso(text, assume_local: bool = False):
    """``(iso_utc | None, note | None)`` from a stored timestamp string."""
    if text in (None, ""):
        return None, None
    try:
        value = dt.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None, "unparseable_timestamp"
    note = None
    if value.tzinfo is None:
        if not assume_local:
            return None, "naive_timestamp"
        value = value.astimezone()
        note = "naive_timestamp_assumed_host_local"
    return value.astimezone(dt.timezone.utc).isoformat(), note


def night_for(timestamp_utc, site_utc_offset=None):
    """Night id by the local-noon rule (``storage/night.rs``).

    With ``site_utc_offset`` (e.g. ``-05:00``) the rule is applied in that
    zone; otherwise ``tools/import_session.night_id_for`` (host local zone)."""
    if not timestamp_utc:
        return None
    if site_utc_offset:
        tz = _import_session().parse_utc_offset(site_utc_offset)
        if tz is None:
            raise ValueError(f"site_utc_offset {site_utc_offset!r} is not like -05:00")
        local = dt.datetime.fromisoformat(timestamp_utc).astimezone(tz)
        if local.hour < 12:
            local -= dt.timedelta(days=1)
        return local.strftime("%Y-%m-%d")
    return _import_session().night_id_for(timestamp_utc, None)


def _apply_tags(entry: dict, tags: dict, authoritative=()) -> list:
    """Copy sidecar values; return conflicts with DB-authoritative fields."""
    conflicts = []
    for key in SIDECAR_COPY:
        if key not in tags:
            continue
        if key in authoritative and entry.get(key) is not None:
            if entry[key] != tags[key]:
                conflicts.append({"field": key, "db": entry[key], "sidecar": tags[key], "kept": "db"})
            continue
        entry[key] = tags[key]
    pointing = pointing_from(tags)
    if pointing:
        entry["pointing"] = pointing
    return conflicts


def _mask_entry(path: Path, sha=None) -> dict:
    fmt = "fits" if path.suffix.lower() in FITS_SUFFIXES else "u8"
    return {"path": str(path), "format": fmt, "sha256": sha or sha256_file(path)}


def parse_bundle(bundle_dir: Path, verify: bool = True) -> tuple:
    """``(entry_fields, bundle_manifest | None)`` for a daemon capture bundle,
    or raise ValueError with the reason it cannot be used."""
    bundle_dir = Path(bundle_dir)
    desc_path = bundle_dir / "analysis" / "analysis.json"
    if not desc_path.exists():
        raise ValueError("no analysis/analysis.json (daemon analysis pending or not produced)")
    descriptor = json.loads(desc_path.read_text())
    analysis = descriptor.get("analysis") or {}
    intensity = descriptor.get("intensity") or {}
    if not intensity.get("filename"):
        raise ValueError("descriptor has no intensity filename")
    ipath = bundle_dir / "analysis" / intensity["filename"]
    if not ipath.exists():
        raise ValueError(f"intensity {ipath.name} missing on disk")
    isha = intensity.get("sha256")
    if verify:
        actual = sha256_file(ipath)
        if isha and actual != isha:
            raise ValueError("intensity sha256 does not match the descriptor")
        isha = actual
    masks = {}
    for record in descriptor.get("masks") or []:
        kind = record.get("kind")
        if kind not in MASK_NAMES:
            continue
        mpath = bundle_dir / "analysis" / record["filename"]
        if not mpath.exists():
            raise ValueError(f"mask {record['filename']} missing on disk")
        msha = record.get("sha256")
        if verify:
            actual = sha256_file(mpath)
            if msha and actual != msha:
                raise ValueError(f"mask {kind} sha256 does not match the descriptor")
            msha = actual
        masks[kind] = {"path": str(mpath), "format": "fits", "sha256": msha}
    fields = {
        "path": str(ipath), "sha256": isha, "format": "daemon_bundle", "masks": masks,
        "analysis_input": {
            "representation": analysis.get("processing_version") or "undeclared",
            "processing_version": analysis.get("processing_version"),
            "config_sha256": analysis.get("config_sha256"),
            "units": analysis.get("units"),
            "descriptor_path": str(desc_path),
            "descriptor_sha256": sha256_file(desc_path),
            "calibration_state": descriptor.get("calibration_state"),
            "validity": descriptor.get("validity"),
        },
    }
    sat = ((analysis.get("sensor") or {}).get("saturation") or {}).get("adu")
    if sat is not None:
        fields["saturation_adu"] = float(sat)
    bundle_manifest = None
    if (bundle_dir / "manifest.v1.json").exists():
        bundle_manifest = json.loads((bundle_dir / "manifest.v1.json").read_text())
    return fields, bundle_manifest


def _frame_id(*parts) -> str:
    import hashlib

    return "f" + hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def finalize(frames: list, excluded: list, builder: dict, rule_path=DEFAULT_SPLIT_RULE) -> dict:
    """Assign nights' splits by the rule and assemble a validated v2 manifest."""
    rule = SPLIT.load_rule(rule_path)
    nights = SPLIT.apply_rule(frames, rule) if frames else []
    reps = sorted({(f.get("analysis_input") or {}).get("representation") for f in frames} - {None})
    manifest = {
        "schema": MANIFEST_SCHEMA_V2,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "qualification_version": QUALIFICATION_VERSION,
        "builder": builder,
        "analysis_input": {
            "representations": reps,
            "daemon_representation": DAEMON_ANALYSIS_REPRESENTATION,
            "loader": "ml/eval/bge/qualification/manifest.py::load_frame (Astropy for daemon bundles/FITS; "
                      "tools/import_session.load_luma_with_container for import_luma)",
            "note": "the analysed pixels are the daemon's archived analysis plane; RAW is never decoded here",
        },
        "split_rule": {"path": str(Path(rule_path).resolve().relative_to(REPO_ROOT))
                       if Path(rule_path).resolve().is_relative_to(REPO_ROOT) else str(rule_path),
                       "sha256": sha256_file(rule_path)},
        "nights": nights,
        "frames": sorted(frames, key=lambda f: (f["night_id"], f.get("timestamp_utc") or "", f["id"])),
        "excluded": excluded,
    }
    validate_manifest(manifest)
    return manifest


def write_manifest(manifest: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return path


def _finish_entry(entry: dict, tags: dict, excluded: list, source: str) -> dict | None:
    """Night, frame_type and exclusion checks common to every builder."""
    if tags.get("exclude"):
        excluded.append({"source": source, "reason": f"excluded by sidecar: {tags.get('exclude_reason', '')}".strip()})
        return None
    if not entry.get("frame_type"):
        excluded.append({"source": source, "reason": "frame_type missing (sidecar frame_type or capture_type)"})
        return None
    if not entry.get("night_id"):
        entry["night_id"] = tags.get("night_id") or night_for(entry.get("timestamp_utc"), tags.get("site_utc_offset"))
    if not entry.get("night_id"):
        excluded.append({"source": source, "reason": "no night_id and no timestamp_utc to derive it"})
        return None
    entry.setdefault("flags", [])
    entry["split"] = "development"  # replaced by the split rule in finalize()
    return entry


# --------------------------------------------------------------------------
# (a) Directory of frames + sidecar log
# --------------------------------------------------------------------------


def _fits_timestamp(path: Path):
    from astropy.io import fits

    try:
        header = fits.getheader(path)
    except Exception:  # noqa: BLE001
        return None
    value = header.get("DATE-OBS")
    if not value:
        return None
    text = str(value)
    if not (text.endswith("Z") or "+" in text[10:] or "-" in text[10:]):
        text += "+00:00"  # FITS DATE-OBS is UTC by convention
    return utc_iso(text)[0]


def build_from_directory(frames_dir, sidecar: SidecarLog | None = None, verify: bool = True,
                         rule_path=DEFAULT_SPLIT_RULE) -> dict:
    frames_dir = Path(frames_dir).resolve()
    sidecar = sidecar or SidecarLog.empty()
    frames, excluded, ignored = [], [], 0
    bundles = sorted({p.parent.parent for p in frames_dir.rglob("analysis/analysis.json")})
    inside = lambda p: any(b == p or b in p.parents for b in bundles)  # noqa: E731
    candidates = [("bundle", b) for b in bundles]
    for p in sorted(frames_dir.rglob("*")):
        if not p.is_file() or inside(p):
            continue
        suffix = p.suffix.lower()
        if suffix in FITS_SUFFIXES or suffix == ".f32":
            candidates.append(("file", p))
        elif suffix in RAW_SUFFIXES:
            candidates.append(("raw", p))
        else:
            ignored += 1
    mask_paths = sidecar.mask_paths(frames_dir)
    for kind, p in candidates:
        rel = p.relative_to(frames_dir).as_posix()
        if rel in mask_paths:
            continue
        entry = {"provenance": {"origin": "directory", "relative_path": rel}}
        bundle_manifest = None
        try:
            if kind == "bundle":
                fields, bundle_manifest = parse_bundle(p, verify)
            elif kind == "raw":
                pre = sidecar.tags_for(file=rel)
                if not pre.get("analysis_bundle"):
                    excluded.append({"source": rel, "reason": "raw_without_daemon_analysis_bundle: RAW is not decoded "
                                     "here; give the daemon bundle in the sidecar's analysis_bundle column"})
                    continue
                fields, bundle_manifest = parse_bundle(frames_dir / pre["analysis_bundle"], verify)
                entry["source"] = {"path": str(p), "sha256": sha256_file(p) if verify else None, "role": "native_raw"}
            else:
                fields = None
        except ValueError as exc:
            excluded.append({"source": rel, "reason": f"daemon bundle unusable: {exc}"})
            continue
        if bundle_manifest:
            entry.update({k: v for k, v in {
                "camera": bundle_manifest.get("camera_name"), "rig_id": bundle_manifest.get("rig_id"),
                "iso": bundle_manifest.get("iso"), "exposure_s": bundle_manifest.get("applied_exposure_s"),
                "timestamp_utc": utc_iso(bundle_manifest.get("exposure_started_at"))[0],
                "frame_type": CAPTURE_TYPE_TO_FRAME_TYPE.get(bundle_manifest.get("capture_type")),
            }.items() if v is not None})
            entry["provenance"].update({"frame_id": bundle_manifest.get("frame_id"),
                                        "capture_id": bundle_manifest.get("capture_id"),
                                        "capture_type": bundle_manifest.get("capture_type")})
            if bundle_manifest.get("capture_type") in EXCLUDED_CAPTURE_TYPES:
                excluded.append({"source": rel, "reason": f"capture_type {bundle_manifest.get('capture_type')}"})
                continue
        tags = sidecar.tags_for(file=rel, frame_id=entry["provenance"].get("frame_id"),
                                capture_id=entry["provenance"].get("capture_id"),
                                timestamp=entry.get("timestamp_utc"))
        if fields is None:  # plain FITS / f32: the representation must be declared
            rep = tags.get("analysis_representation")
            if not rep:
                excluded.append({"source": rel, "reason": "analysis_representation undeclared for a plain FITS/f32 "
                                 "(a CFA mosaic or a display image is not the daemon analysis input)"})
                continue
            fields = {"path": str(p), "sha256": sha256_file(p), "format": "f32" if p.suffix.lower() == ".f32" else "fits",
                      "analysis_input": {"representation": rep}, "masks": {}}
            if fields["format"] == "f32":
                if not (tags.get("width") and tags.get("height")):
                    excluded.append({"source": rel, "reason": "f32 frame needs width and height in the sidecar"})
                    continue
                fields["width"], fields["height"] = tags["width"], tags["height"]
            elif "hdu" in tags:
                fields["hdu"] = tags["hdu"]
            for name in MASK_NAMES:
                if tags.get(f"mask_{name}"):
                    fields["masks"][name] = _mask_entry((frames_dir / tags[f"mask_{name}"]).resolve())
            if not tags.get("timestamp_utc") and fields["format"] == "fits":
                ts = _fits_timestamp(p)
                if ts:
                    entry["timestamp_utc"] = ts
                    entry["flags"] = ["timestamp_from_fits_date_obs"]
        entry.update(fields)
        if tags.get("frame_type"):
            if entry.get("frame_type") not in (None, "light", tags["frame_type"]):
                entry.setdefault("provenance", {})["sidecar_conflicts"] = [
                    {"field": "frame_type", "bundle": entry["frame_type"], "sidecar": tags["frame_type"], "kept": "bundle"}]
            else:
                entry["frame_type"] = tags["frame_type"]
        conflicts = _apply_tags(entry, tags, authoritative=("timestamp_utc",) if bundle_manifest else ())
        if conflicts:
            entry["provenance"].setdefault("sidecar_conflicts", []).extend(conflicts)
        entry["id"] = _frame_id("dir", entry["provenance"].get("frame_id") or rel)
        entry = _finish_entry(entry, tags, excluded, rel)
        if entry is not None:
            frames.append(entry)
    builder = {"kind": "directory", "frames_dir": str(frames_dir), "sidecar": sidecar.source,
               "ignored_files": ignored, "unused_sidecar_rows": sidecar.unused_rows()}
    return finalize(frames, excluded, builder, rule_path)


# --------------------------------------------------------------------------
# (b, c) SQLite DB (import_session or daemon) + capture archive
# --------------------------------------------------------------------------


def _ro_connect(db_path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _tables(con) -> set:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(con, table) -> set:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def _json(text) -> dict:
    try:
        value = json.loads(text or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _sweep_directions(samples_by_run: dict) -> dict:
    out = {}
    for run, rows in samples_by_run.items():
        ordered = [r["position"] for r in sorted(rows, key=lambda r: str(r["captured_at"]))]
        diffs = [b - a for a, b in zip(ordered, ordered[1:]) if b != a]
        out[run] = ("increasing" if diffs and all(d > 0 for d in diffs) else
                    "decreasing" if diffs and all(d < 0 for d in diffs) else "mixed" if diffs else None)
    return out


def _origin(frame_row: dict, acquisition: dict) -> str:
    if frame_row.get("driver_name") == "offline_import" or acquisition.get("import_tool"):
        return "import_session"
    return "daemon_capture"


def _stepsize_by_rig(con) -> dict:
    if "focuser_profiles" not in _tables(con) or "steps_per_micron" not in _columns(con, "focuser_profiles"):
        return {}
    out = {}
    rows = con.execute("SELECT rig_id, focuser_name, steps_per_micron, scale_measured_at FROM focuser_profiles "
                       "WHERE steps_per_micron IS NOT NULL AND steps_per_micron > 0 "
                       "ORDER BY scale_measured_at").fetchall()
    for r in rows:  # latest measurement wins
        out[r["rig_id"]] = {"um_per_step": 1.0 / float(r["steps_per_micron"]),
                            "source": f"focuser_profiles.steps_per_micron ({r['focuser_name']}, {r['scale_measured_at']})"}
    return out


def _focal_ratio_by_rig(con) -> dict:
    if "rigs" not in _tables(con) or "focal_ratio" not in _columns(con, "rigs"):
        return {}
    return {r["id"]: float(r["focal_ratio"]) for r in con.execute("SELECT id, focal_ratio FROM rigs")
            if r["focal_ratio"] is not None}


def _ambient(con, timestamp_utc, window_s=1800.0):
    if not timestamp_utc or "environment_log" not in _tables(con):
        return None
    t = dt.datetime.fromisoformat(timestamp_utc)
    best = None
    for r in con.execute("SELECT observed_at, ambient_temp_c FROM environment_log WHERE ambient_temp_c IS NOT NULL"):
        when, _ = utc_iso(r["observed_at"])
        if when is None:
            continue
        gap = abs((dt.datetime.fromisoformat(when) - t).total_seconds())
        if gap <= window_s and (best is None or gap < best[0]):
            best = (gap, float(r["ambient_temp_c"]))
    return None if best is None else best[1]


PROVENANCE_COLUMNS = ("position_readback_before", "position_readback_after", "position_source", "focuser_move_id",
                      "settled_at", "exposure_started_at", "capture_id", "clock_quality", "moved_during_exposure",
                      "filter_verified", "temperature_source", "temperature_age_ms", "tempcomp_state", "backlash_model")


def build_from_db(db_path, capture_root, sidecar: SidecarLog | None = None, origins=("daemon_capture",),
                  verify: bool = False, rule_path=DEFAULT_SPLIT_RULE) -> dict:
    """Frames from an Achernar SQLite DB (read-only) and its capture archive.

    ``origins``: ``daemon_capture`` (the daemon's own captures and autofocus
    sweeps) and/or ``import_session`` (rows written by tools/import_session.py).
    """
    capture_root = Path(capture_root).resolve()
    sidecar = sidecar or SidecarLog.empty()
    con = _ro_connect(db_path)
    try:
        tables = _tables(con)
        if "frames" not in tables:
            raise ValueError(f"{db_path}: no frames table")
        frames_rows = [dict(r) for r in con.execute("SELECT * FROM frames ORDER BY captured_at")]
        assets = {}
        if "frame_assets" in tables:
            for r in con.execute("SELECT * FROM frame_assets ORDER BY role, variant"):
                assets.setdefault(r["frame_id"], []).append(dict(r))
        samples, runs = {}, {}
        if "focus_samples" in tables:
            cols = _columns(con, "focus_samples")
            prov = [c for c in PROVENANCE_COLUMNS if c in cols]
            select = ", ".join(["run_id", "position", "frame_id", "captured_at"] + prov)
            for r in con.execute(f"SELECT {select} FROM focus_samples WHERE frame_id IS NOT NULL"):
                samples[r["frame_id"]] = dict(r)
        if "focus_runs" in tables:
            runs = {r["id"]: dict(r) for r in con.execute("SELECT * FROM focus_runs")}
        by_run = {}
        for s in samples.values():
            by_run.setdefault(s["run_id"], []).append(s)
        directions = _sweep_directions(by_run)
        stepsize = _stepsize_by_rig(con)
        focal = _focal_ratio_by_rig(con)
        frames, excluded = [], []
        missing_fields = {"optical_train": 0, "repeat_group": 0, "focuser_stepsize_um": 0, "focal_ratio": 0}
        for row in frames_rows:
            acquisition, quality = _json(row.get("acquisition_json")), _json(row.get("quality_json"))
            origin = _origin(row, acquisition)
            source = f"db frame {row['id']}"
            if origin not in origins:
                continue
            domain = acquisition.get("acquisition_domain") or quality.get("acquisition_domain")
            if domain in EXCLUDED_DOMAINS:
                excluded.append({"source": source, "reason": f"acquisition_domain {domain}"})
                continue
            status = row.get("capture_status") or "committed"
            if status != "committed":
                excluded.append({"source": source, "reason": f"capture_status {status}"})
                continue
            ctype = row.get("capture_type") or "light"
            if ctype in EXCLUDED_CAPTURE_TYPES:
                excluded.append({"source": source, "reason": f"capture_type {ctype}"})
                continue
            ts, ts_note = utc_iso(row.get("exposure_started_at") or row.get("captured_at"),
                                  assume_local=origin == "import_session")
            entry = {"frame_type": CAPTURE_TYPE_TO_FRAME_TYPE.get(ctype), "timestamp_utc": ts,
                     "night_id": row.get("night_id"), "rig_id": row.get("rig_id"), "camera": row.get("camera_name"),
                     "iso": row.get("iso"), "gain": row.get("gain"), "exposure_s": row.get("exposure_s"),
                     "flags": [ts_note] if ts_note else [],
                     "provenance": {"origin": origin, "frame_id": row["id"], "capture_id": row.get("capture_id"),
                                    "capture_type": ctype, "session_id": row.get("session_id"),
                                    "driver_name": row.get("driver_name"),
                                    "software_version": row.get("software_version"),
                                    "temperature_source": row.get("temperature_source"),
                                    "temperature_state": row.get("temperature_state")}}
            if row.get("temperature") is not None and row.get("temperature_state") == "measured":
                entry["sensor_temp_c"] = float(row["temperature"])
            if ctype == "verification" or acquisition.get("quality", {}).get("autofocus_role") == "verification":
                entry["flags"].append("autofocus_verification_frame")
            sample = samples.get(row["id"])
            if sample is not None:
                entry["focus_run_id"] = sample["run_id"]
                entry["focus_position"] = float(sample["position"])
                entry["sweep_direction"] = directions.get(sample["run_id"])
                prov = {k: sample[k] for k in PROVENANCE_COLUMNS if k in sample and sample[k] is not None}
                entry["provenance"]["focus_sample"] = prov
                if prov.get("moved_during_exposure"):
                    entry["flags"].append("moved_during_exposure")
                before, after = prov.get("position_readback_before"), prov.get("position_readback_after")
                if before is not None and after is not None and before != after:
                    entry["flags"].append("position_readback_changed")
                if after is not None and after != sample["position"]:
                    entry["flags"].append("position_readback_mismatch")
                run = runs.get(sample["run_id"]) or {}
                entry["provenance"]["focus_run"] = {k: run.get(k) for k in ("state", "start_position", "best_position",
                                                                            "started_at", "night_id", "temperature")}
                if not entry.get("night_id"):
                    entry["night_id"] = run.get("night_id")
            elif row.get("focuser_position") is not None:
                entry["focus_position"] = float(row["focuser_position"])
            unit = acquisition.get("focuser_position_unit")
            if unit and str(unit).endswith("um"):
                try:
                    entry["focuser_stepsize_um"] = float(str(unit)[:-2])
                    entry["provenance"]["focuser_stepsize_source"] = f"acquisition_json.focuser_position_unit={unit}"
                except ValueError:
                    pass
            if entry.get("focuser_stepsize_um") is None and row.get("rig_id") in stepsize:
                entry["focuser_stepsize_um"] = stepsize[row["rig_id"]]["um_per_step"]
                entry["provenance"]["focuser_stepsize_source"] = stepsize[row["rig_id"]]["source"]
            if row.get("rig_id") in focal:
                entry["focal_ratio"] = focal[row["rig_id"]]
                entry["provenance"]["focal_ratio_source"] = "rigs.focal_ratio"
            ambient = _ambient(con, ts)
            if ambient is not None:
                entry["ambient_temp_c"] = ambient
            entry = {k: v for k, v in entry.items() if v is not None}
            # Analysis input.
            fa = assets.get(row["id"], [])
            roles = {a["role"]: a for a in fa}
            names = [a.get("original_filename") for a in fa if a.get("original_filename")]
            try:
                if "analysis_descriptor" in roles:
                    desc_rel = Path(roles["analysis_descriptor"]["relative_path"])
                    bundle_dir = capture_root / desc_rel.parent.parent
                    fields, _bm = parse_bundle(bundle_dir, verify)
                    if "analysis_fits" in roles and not verify:
                        fields["sha256"] = roles["analysis_fits"]["checksum_sha256"]
                elif "analysis_luma" in roles:
                    lpath = capture_root / roles["analysis_luma"]["relative_path"]
                    if not lpath.exists():
                        raise ValueError(f"analysis_luma {lpath.name} missing on disk")
                    fields = {"path": str(lpath), "format": "import_luma",
                              "sha256": sha256_file(lpath) if verify else roles["analysis_luma"]["checksum_sha256"],
                              "analysis_input": {"representation": LUMA_REPRESENTATION,
                                                 "note": "analysis_luma asset read with tools/import_session."
                                                         "load_luma_with_container; NOT the daemon's linear analysis "
                                                         "plane unless shown otherwise"}}
                    entry["flags"].append("not_daemon_analysis_representation")
                else:
                    have = sorted(roles) or ["no assets"]
                    raise ValueError(("raw_without_daemon_analysis_bundle" if "native_raw" in roles else
                                      "no analysis asset") + f" (assets: {', '.join(have)})")
            except ValueError as exc:
                excluded.append({"source": source, "reason": str(exc)})
                continue
            raw = roles.get("native_raw")
            if raw:
                entry["source"] = {"path": str(capture_root / raw["relative_path"]), "sha256": raw["checksum_sha256"],
                                   "role": "native_raw", "original_filename": raw.get("original_filename")}
            entry.update(fields)
            tags = sidecar.tags_for(file=names[0] if names else None, frame_id=row["id"],
                                    capture_id=row.get("capture_id"), timestamp=ts)
            for alt in names[1:]:
                tags.update({k: v for k, v in sidecar.tags_for(file=alt).items() if k not in sidecar.defaults})
            if tags.get("frame_type") and tags["frame_type"] != entry.get("frame_type"):
                if ctype in ("light",) or not entry.get("frame_type"):
                    entry["frame_type"] = tags["frame_type"]
                    entry["provenance"]["frame_type_source"] = "sidecar (DB capture_type was light/absent)"
                else:
                    entry["provenance"].setdefault("sidecar_conflicts", []).append(
                        {"field": "frame_type", "db": entry.get("frame_type"), "sidecar": tags["frame_type"], "kept": "db"})
            conflicts = _apply_tags(entry, tags, authoritative=DB_AUTHORITATIVE)
            if conflicts:
                entry["provenance"].setdefault("sidecar_conflicts", []).extend(conflicts)
            for key in missing_fields:
                if entry.get(key) is None and (key != "repeat_group" or entry.get("frame_type") == "light"):
                    missing_fields[key] += 1
            entry["id"] = _frame_id("db", row["id"])
            entry = _finish_entry(entry, tags, excluded, source)
            if entry is not None:
                frames.append(entry)
    finally:
        con.close()
    builder = {"kind": "sqlite_db", "origins": list(origins), "db": str(Path(db_path).resolve()),
               "capture_root": str(capture_root), "sidecar": sidecar.source, "read_only": True,
               "fields_missing_from_db_and_sidecar": missing_fields,
               "db_gaps": ["optical_train (not a DB column: sidecar)", "repeat_group (not a DB column: sidecar)",
                           "pointing per frame (not a DB column: sidecar or plate_solve_events)",
                           "focuser_stepsize_um only via focuser_profiles.steps_per_micron when measured",
                           "focal_ratio only via rigs.focal_ratio when configured"],
               "unused_sidecar_rows": sidecar.unused_rows()}
    return finalize(frames, excluded, builder, rule_path)


def build_from_import_session_db(db_path, capture_root, sidecar=None, **kw) -> dict:
    return build_from_db(db_path, capture_root, sidecar, origins=("import_session",), **kw)


def build_from_daemon_db(db_path, capture_root, sidecar=None, include_imports: bool = False, **kw) -> dict:
    origins = ("daemon_capture", "import_session") if include_imports else ("daemon_capture",)
    return build_from_db(db_path, capture_root, sidecar, origins=origins, **kw)


def default_capture_root(db_path) -> Path:
    """``<frames_dir>/captures`` per main.rs; frames_dir defaults beside the DB."""
    return Path(os.environ.get("ACHERNAR_CAPTURE_ROOT") or (Path(db_path).resolve().parent / "frames" / "captures"))
