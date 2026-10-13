#!/usr/bin/env python3
"""
import_session — ingest frames the daemon did not capture.

Closes gaps G2, G4 and G5 from the data-collection plan. Every open-loop
campaign produces files with no database row: the cloud/obstruction campaign on
an intervalometer, any body that will not tether, calibration frames shot by
hand. Without this tool those files are scientifically orphaned — bytes on a
card with no provenance, no checksum, and no way to join them to the conditions
under which they were taken.

WHAT IT MIRRORS

`storage/capture_store.rs` commits a capture in a specific order, and the order
is the point: stage to `.partial` while hashing in one pass, write a manifest,
fsync, open a transaction, insert `frames` + `frame_assets`, atomically rename,
commit. This tool follows the same sequence so that imported rows are
indistinguishable in durability from captured ones.

WHAT IT DELIBERATELY DOES NOT DO

It writes no `capture_attempts` row. That table records what the daemon
*commanded*; an offline import commanded nothing. Inventing attempt rows would
corrupt the denominator in every yield calculation — "attempted 400, committed
390" is a meaningful sentence only when the attempts were real.

HONESTY ABOUT METADATA

Exposure, ISO and timestamp are read from EXIF where a file exposes it, then
from the session log, then from file mtime — in that order, with the source
recorded in `acquisition_json.metadata_source`. A timestamp derived from mtime
is not the same observation as one from the shutter, and a model trained on a
mixture that cannot tell them apart is learning noise.

Usage
-----
    python3 tools/import_session.py \
        --db achernar-daemon/achernar.db \
        --files ~/coc/2026-08-07/ \
        --log  ~/coc/2026-08-07/session.json \
        --capture-root ~/achernar-frames

    # see what would happen, touch nothing
    python3 tools/import_session.py ... --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ─── File-role classification ────────────────────────────────────────────────
# Roles match the `frame_assets.role` vocabulary in migration 004.

RAW_SUFFIXES = {
    ".cr2", ".cr3", ".nef", ".arw", ".rw2", ".orf", ".raf", ".dng", ".pef", ".srw",
}
JPEG_SUFFIXES = {".jpg", ".jpeg"}
FITS_SUFFIXES = {".fits", ".fit", ".fts"}
PNG_SUFFIXES = {".png", ".tif", ".tiff"}

VALID_CAPTURE_TYPES = {"light", "dark", "flat", "bias"}

MANIFEST_VERSION = "manifest.v1.json"

# Mirrors `saturation_threshold` in achernar-daemon/src/vision/background.rs and
# SATURATION_FRACTION in ml/data_pipeline/build_focus_quality_dataset.py. Sensor
# response goes non-linear below the hard clip, so the gate sits at 98%.
SATURATION_FRACTION = 0.98

# A container ceiling at or below this is an 8-bit stream, where the container
# *is* the clip point — the 8-bit arm of `resolve_saturation_adu` in
# vision/quality.rs. Above it we are looking at a 16-bit container, whose
# ceiling says nothing about where the sensor actually saturates.
EIGHT_BIT_MAX_ADU = 255.0

# How far from a position's logged timestamp a frame may sit and still be
# considered part of that position. Only used when the field card gives neither
# filenames nor frame ids. A focus sweep dwells seconds per position, so this is
# deliberately tight — a wrong match here silently corrupts the position axis,
# which is the one thing a sweep exists to record.
DEFAULT_MATCH_WINDOW_S = 30.0

# `focus_samples.position` is an INTEGER axis with no unit of its own. A
# motorised focuser reports steps; a dial indicator reports displacement. The
# unit is therefore recorded per frame in `acquisition_json.focuser_position_unit`
# rather than assumed, and `--position-unit-um` converts a `reading_um` field
# card entry into whatever integer axis the run is using. The default of 1.0
# stores micrometres directly, which is self-describing and monotonic — the two
# properties the focus models actually require of this axis.
DEFAULT_POSITION_UNIT_UM = 1.0

# Run states written by this tool. Deliberately NOT the daemon's `Converged`:
# a hand-cranked sweep has no fitted vertex, and claiming one would let a
# sample argmin be mistaken for a solver result.
RUN_STATE_IMPORTED = "Imported"
RUN_STATE_ABORTED = "Aborted"


def _ensure_ml_on_path() -> None:
    """Make `ml/labels/psf_measure.py` importable however this tool is invoked.

    The field HFR must come from the same labeler the dataset builder uses, and
    that module lives under `ml/`, which is a sibling of `tools/` rather than a
    package on the default path. Without this the measurement quietly fails and
    a whole night's sweep imports with no samples — the failure is reported, but
    reporting it at 1 a.m. is too late.
    """
    ml_dir = Path(__file__).resolve().parent.parent / "ml"
    if ml_dir.is_dir() and str(ml_dir) not in sys.path:
        sys.path.insert(0, str(ml_dir))


_ensure_ml_on_path()


def classify(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in RAW_SUFFIXES:
        return "native_raw"
    if suffix in JPEG_SUFFIXES:
        return "camera_jpeg"
    if suffix in FITS_SUFFIXES:
        return "sensor_fits"
    if suffix in PNG_SUFFIXES:
        return "analysis_luma"
    return None


def media_type(role: str, path: Path) -> str:
    return {
        "native_raw": "image/x-raw",
        "camera_jpeg": "image/jpeg",
        "sensor_fits": "image/fits",
        "analysis_luma": "image/png" if path.suffix.lower() == ".png" else "image/tiff",
    }.get(role, "application/octet-stream")


# ─── Hashing and staging ─────────────────────────────────────────────────────


def copy_and_hash(src: Path, dest: Path) -> tuple[str, int]:
    """Copy to `dest.partial`, hashing in the same pass. Returns (sha256, bytes).

    One pass, not two: reading the file twice to hash it and then copy it is
    both slower and a correctness hazard, because the file could change between
    the two reads.
    """
    digest = hashlib.sha256()
    total = 0
    partial = dest.with_name(dest.name + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fin, partial.open("wb") as fout:
        while chunk := fin.read(1024 * 1024):
            digest.update(chunk)
            fout.write(chunk)
            total += len(chunk)
        fout.flush()
        os.fsync(fout.fileno())
    return digest.hexdigest(), total


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ─── EXIF ────────────────────────────────────────────────────────────────────


def read_exif(path: Path) -> dict:
    """Best-effort EXIF. Returns {} when the format is unreadable.

    Deliberately dependency-free: PIL reads JPEG and TIFF, which covers the
    JPEG half of a RAW+JPEG pair. Canon CR3 is ISO-BMFF and no pure-Python
    reader handles it reliably, so RAW-only imports fall through to the session
    log — which is why the log has exposure and ISO fields.
    """
    try:
        from PIL import Image
    except ImportError:
        return {}
    try:
        with Image.open(path) as img:
            raw = img.getexif()
            if not raw:
                return {}
            from PIL.ExifTags import TAGS

            flat = {TAGS.get(k, str(k)): v for k, v in raw.items()}
            for ifd_id in (0x8769,):  # ExifOffset
                try:
                    for k, v in raw.get_ifd(ifd_id).items():
                        flat[TAGS.get(k, str(k))] = v
                except Exception:
                    pass
            return flat
    except Exception:
        return {}


def parse_utc_offset(value) -> timezone | None:
    """'-04:00' / '+0530' / 'Z' -> tzinfo, or None if absent or malformed."""
    if value is None:
        return None
    text = str(value).strip().replace("\x00", "")
    if text in ("Z", "z", "+00:00", "-00:00"):
        return timezone.utc
    try:
        return datetime.strptime(text.replace(":", ""), "%z").tzinfo
    except ValueError:
        return None


def exif_datetime(
    exif: dict, camera_utc_offset: str | None = None
) -> tuple[str | None, str | None]:
    """Shutter time as an aware ISO timestamp, plus where its offset came from.

    EXIF DateTimeOriginal is the camera's *wall clock* with no zone. Reading it
    as UTC put an evening frame four hours away from the -04:00 sky labels on
    the field card, so the 15-minute label match missed or hit the wrong frame.
    Precedence, recorded as the second return value:

    1. ``exif_offset`` — OffsetTimeOriginal/OffsetTime, which Canon EOS R
       bodies write when the in-camera time zone is set.
    2. ``declared`` — ``camera_utc_offset`` from the session log or CLI.
    3. ``host_local_assumed`` — the importing machine's zone. A guess, and
       labelled as one so it can be filtered out later.
    """
    exif_offset = None
    for key in ("OffsetTimeOriginal", "OffsetTime", "36881", "36880"):
        exif_offset = parse_utc_offset(exif.get(key))
        if exif_offset is not None:
            break
    declared = parse_utc_offset(camera_utc_offset)

    for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
        value = exif.get(key)
        if not value:
            continue
        try:
            dt = datetime.strptime(str(value).strip(), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            continue
        if exif_offset is not None:
            return dt.replace(tzinfo=exif_offset).isoformat(), "exif_offset"
        if declared is not None:
            return dt.replace(tzinfo=declared).isoformat(), "declared"
        return dt.astimezone().isoformat(), "host_local_assumed"
    return None, None


def exif_number(exif: dict, key: str) -> float | None:
    value = exif.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        try:  # PIL IFDRational
            return float(value.numerator) / float(value.denominator)
        except Exception:
            return None


# ─── Session log ─────────────────────────────────────────────────────────────


def load_log(path: Path | None) -> dict:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: session log must be a JSON object")
    return data


def night_id_for(timestamp: str, fallback: str) -> str:
    """Local date of sunset. Anything before local noon belongs to the previous night.

    Must stay identical to `storage::night::night_id` in the daemon. The two
    write rows that later get grouped together, and two definitions of "night"
    would cut one session in half with nothing to show it had happened.

    The noon test is applied in **local** time. An aware timestamp is converted
    first, because the UTC hour is not the observer's hour: 2026-11-15T14:00Z is
    09:00 Eastern, still the night of the 14th, but its UTC hour of 14 reads as
    an afternoon and would have opened a new night in the middle of a session.
    A naive timestamp — what a field card written by hand actually contains — is
    taken as already local.
    """
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return fallback
    if dt.tzinfo is not None:
        dt = dt.astimezone()
    if dt.hour < 12:
        return (dt - timedelta(days=1)).strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m-%d")


# ─── Frame measurement (for the field V-curve check) ─────────────────────────


def load_luma(path: Path):
    """Read an image file as a 2-D float array, or return None.

    Only formats a hand-run sweep actually produces are handled: the 16-bit PNG
    the daemon writes for focus frames, TIFF, FITS, and the JPEG half of a
    RAW+JPEG pair. RAW is deliberately NOT decoded here — doing it correctly
    means LibRaw semantics (no black subtraction, no rescaling) and this tool
    has no business owning a second, divergent RAW path. A RAW-only sweep still
    imports; it just carries no field HFR.
    """
    import numpy as np

    # A frame staged by this import is still called `<name>.png.partial`, so the
    # last suffix is the staging marker rather than the format. Strip it before
    # dispatching, or every in-transaction measurement silently reads as an
    # unknown format and the sweep loses every sample it just imported.
    name = path.name
    if name.endswith(".partial"):
        name = name[: -len(".partial")]
    suffix = Path(name).suffix.lower()

    if suffix in FITS_SUFFIXES:
        try:
            from astropy.io import fits
        except ImportError:
            return None
        with fits.open(path) as hdul:
            for hdu in hdul:
                if getattr(hdu, "data", None) is not None and hdu.data.ndim == 2:
                    return np.asarray(hdu.data, dtype=np.float64)
        return None

    if suffix in PNG_SUFFIXES or suffix in JPEG_SUFFIXES:
        try:
            from PIL import Image
        except ImportError:
            return None
        with Image.open(path) as image:
            array = np.asarray(image)
        if array.ndim == 3:
            # Rec. 601 luma. Only reached for the JPEG half of a pair; the
            # daemon's own focus frames are already single-channel.
            array = (
                0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
            )
        if array.ndim != 2:
            return None
        return array.astype(np.float64)

    return None


def container_max_adu(image) -> float:
    """Full scale of the container the samples arrived in — 255 or 65535.

    The container ceiling, never the sensor's saturation point. Conflating the
    two is defect D1, and the whole reason `LumaFrame` carries both.
    """
    return EIGHT_BIT_MAX_ADU if float(image.max()) <= EIGHT_BIT_MAX_ADU else 65535.0


def resolve_saturation_adu(image, declared: float | None) -> float | None:
    """The sensor's linear saturation point in this frame's units, or None.

    Mirrors `resolve_saturation_adu` in vision/quality.rs exactly:

    * a declared value always wins (it came from BP-2, per body and ISO);
    * an 8-bit container *is* its own clip point, because an 8-bit sample
      physically cannot exceed 255 and one that reads 255 is clipped;
    * a 16-bit container says nothing — a 14-bit body sends 16-bit samples that
      stop at 16383, and nothing in the pixels reveals that. Return None, so
      the gate reports itself as not performed rather than silently passing.
    """
    if declared is not None:
        try:
            value = float(declared)
        except (TypeError, ValueError):
            return None
        return value if value > 0.0 else None

    max_adu = container_max_adu(image)
    return max_adu if max_adu <= EIGHT_BIT_MAX_ADU else None


def readable_path(path: Path) -> Path | None:
    """The path a frame's bytes can actually be read from right now.

    Focus runs are resolved *inside* the import transaction, which is before
    staged files are renamed off `.partial` — the same ordering that makes a
    crash mid-import leave nothing half-recorded. So a frame committed moments
    ago is readable only under its staging name. Frames imported on an earlier
    run are readable under their final name.
    """
    if path.exists():
        return path
    partial = path.with_name(path.name + ".partial")
    return partial if partial.exists() else None


def measure_frame(path: Path, declared_saturation: float | None) -> dict | None:
    """Frame-level HFR for a sweep frame, or None when it cannot be measured.

    Runs the *same* labeler the dataset builder runs (`measure_frame_sep`), so a
    V-curve inspected at 2 a.m. and the labels produced weeks later come from
    one implementation. This value is a field sanity check, not the training
    label: `build_focus_quality_dataset.py` re-measures every crop from the
    preserved pixels.

    Returns None when the frame is unreadable, has no measurable stars, or SEP
    is unavailable. None is propagated as "no sample", never as a zero — zero is
    the mathematically best value for a width metric and would put a blank frame
    at the bottom of a V-curve.
    """
    image = load_luma(path)
    if image is None:
        return None

    sat = resolve_saturation_adu(image, declared_saturation)
    gate_applied = sat is not None
    threshold = sat * SATURATION_FRACTION if sat is not None else float("inf")

    try:
        from labels.psf_measure import measure_frame_sep

        _sources, label = measure_frame_sep(
            image, saturation_level=threshold, fit_fwhm=False
        )
    except Exception:
        return None

    if label.hfr_px_median is None:
        return None

    return {
        "hfr_px": float(label.hfr_px_median),
        "hfd_px": float(label.hfr_px_median) * 2.0,
        "star_count": int(label.star_count_detected),
        "star_count_used": int(label.star_count_used),
        "background_adu": float(label.sky_median_adu),
        "background_sigma": float(label.sky_rms_adu),
        "max_adu": container_max_adu(image),
        "saturation_adu": sat,
        "saturation_gate_applied": gate_applied,
        "saturation_fraction": float(label.saturated_star_fraction),
    }


def quality_json_for(measurement: dict | None, declared_saturation: float | None) -> str:
    """The `frames.quality_json` an imported frame carries.

    This is not decoration. `build_focus_quality_dataset.py` reads
    `saturation_adu` / `saturation_gate_applied` / `max_adu` out of this column
    to decide whether a frame may enter the training set at all, and an empty
    `{}` — what this tool used to write — resolves to "unknown", which the
    builder correctly treats as a reason to drop the frame. An import that
    writes no quality_json produces a zero-row dataset and no error message.

    When nothing can be measured, `saturation_gate_applied` is written as
    `false` explicitly. That is the honest record and it makes the builder's
    refusal legible instead of mysterious.
    """
    if measurement is None:
        payload = {
            "source": "offline_import",
            "observation_valid": False,
            "saturation_gate_applied": False,
        }
        if declared_saturation is not None:
            payload["saturation_adu"] = float(declared_saturation)
            payload["saturation_gate_applied"] = True
        return json.dumps(payload)

    payload = {k: v for k, v in measurement.items()}
    payload["source"] = "offline_import"
    payload["observation_valid"] = True
    payload["status"] = "Valid"
    return json.dumps(payload)


# ─── Import ──────────────────────────────────────────────────────────────────


def group_by_capture(files: list[Path]) -> dict[str, list[Path]]:
    """A RAW and its in-camera JPEG are one exposure, not two.

    Grouped by directory plus filename stem, which is how every camera in the
    plan names a RAW+JPEG pair (IMG_0001.CR3 / IMG_0001.JPG). The directory is
    part of the key because card counters repeat: night1/IMG_0001 and
    night2/IMG_0001 are different exposures and must not merge.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(files):
        groups[str(path.with_suffix(""))].append(path)
    return dict(groups)


def existing_checksums(con: sqlite3.Connection) -> set[str]:
    try:
        return {
            row[0]
            for row in con.execute("SELECT checksum_sha256 FROM frame_assets")
            if row[0]
        }
    except sqlite3.OperationalError:
        return set()


def ensure_schema(con: sqlite3.Connection, migrations_dir: Path) -> None:
    """Apply any migration whose tables are absent. Idempotent by construction —
    every statement in migrations/ is CREATE TABLE IF NOT EXISTS or an ALTER
    guarded by the caller."""
    have = {
        row[0]
        for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if "environment_log" in have and "disturbance_log" in have:
        return
    sql = (migrations_dir / "006_environment_log.sql").read_text()
    con.executescript(sql)


def import_session(
    db_path: Path,
    files_dir: Path,
    capture_root: Path,
    log_path: Path | None = None,
    capture_type: str = "light",
    rig_id: str | None = None,
    dry_run: bool = False,
    migrations_dir: Path | None = None,
    position_unit_um: float = DEFAULT_POSITION_UNIT_UM,
    match_window_s: float = DEFAULT_MATCH_WINDOW_S,
    camera_utc_offset: str | None = None,
) -> dict:
    if capture_type not in VALID_CAPTURE_TYPES:
        raise ValueError(
            f"capture_type must be one of {sorted(VALID_CAPTURE_TYPES)}, got {capture_type!r}"
        )

    log = load_log(log_path)
    session = log.get("session", {})
    rig_id = rig_id or session.get("rig_id")
    session_id = session.get("session_id") or str(uuid.uuid4())
    camera_name = session.get("camera")
    # What the observer wrote on the card, if anything. Kept separate from the
    # fallback below because "the card says 14 Sept" and "nobody said, so assume
    # today" are different claims and only the first may override a timestamp.
    declared_night = session.get("night_id")
    default_night = declared_night or datetime.now().strftime("%Y-%m-%d")
    # Appendix C.1 carries `saturation_adu` per body and ISO, measured by BP-2.
    # It is the difference between a frame the dataset builder will accept and
    # one it drops as ungated, so it is read here rather than guessed later.
    declared_saturation = session.get("saturation_adu")
    # The camera clock's UTC offset when EXIF does not carry one (see
    # exif_datetime). The CLI flag wins over the card.
    camera_utc_offset = camera_utc_offset or session.get("camera_utc_offset")
    if camera_utc_offset is not None and parse_utc_offset(camera_utc_offset) is None:
        raise ValueError(f"camera_utc_offset must look like -04:00, got {camera_utc_offset!r}")

    candidates = [
        p for p in sorted(files_dir.rglob("*")) if p.is_file() and classify(p)
    ]
    if not candidates:
        raise RuntimeError(f"No importable image files found under {files_dir}")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    if migrations_dir:
        ensure_schema(con, migrations_dir)
    seen = existing_checksums(con)

    stats = {
        "groups": 0,
        "frames_inserted": 0,
        "assets_inserted": 0,
        "skipped_duplicate": 0,
        "metadata_from_exif": 0,
        "metadata_from_log": 0,
        "metadata_from_mtime": 0,
        "environment_rows": 0,
        "disturbance_rows": 0,
        "labels_written": 0,
        "focus_runs_inserted": 0,
        "focus_samples_inserted": 0,
        "positions_unresolved": 0,
        "samples_unmeasurable": 0,
        "timezone_assumed": 0,
    }
    staged: list[Path] = []
    # Original filename -> frame id, so the field card's `files` entries can be
    # linked to the rows this import just created.
    filename_to_frame: dict[str, str] = {}

    try:
        for stem, group in group_by_capture(candidates).items():
            stats["groups"] += 1

            # Metadata: EXIF from whichever file in the group yields it.
            exif: dict = {}
            for path in group:
                exif = read_exif(path)
                if exif:
                    break

            captured_at, timezone_source = exif_datetime(exif, camera_utc_offset)
            if timezone_source == "host_local_assumed":
                stats["timezone_assumed"] += 1
            metadata_source = "exif"
            if captured_at is None:
                captured_at = session.get("captured_at")
                metadata_source = "session_log"
            if captured_at is None:
                mtime = min(p.stat().st_mtime for p in group)
                captured_at = datetime.fromtimestamp(mtime, timezone.utc).isoformat()
                metadata_source = "file_mtime"
            stats[f"metadata_from_{ {'exif':'exif','session_log':'log','file_mtime':'mtime'}[metadata_source] }"] += 1

            exposure_s = exif_number(exif, "ExposureTime")
            iso = exif_number(exif, "ISOSpeedRatings") or exif_number(exif, "PhotographicSensitivity")
            if exposure_s is None:
                exposure_s = session.get("exposure_s")
            if iso is None:
                iso = session.get("iso_or_gain")

            body = exif.get("Model") or camera_name
            # A night derived from file mtime is a filesystem artifact, not an
            # observation: copying a card in the morning would file every frame
            # under the wrong night, and night is a split key. When the shutter
            # time is unknown and the card declares a night, the card wins.
            night = (
                declared_night
                if (metadata_source == "file_mtime" and declared_night)
                else night_id_for(captured_at, default_night)
            )
            frame_id = str(uuid.uuid4())

            assets = []
            duplicate = False
            for path in group:
                role = classify(path)
                rel = Path(night) / capture_type / f"{frame_id}_{role}{path.suffix.lower()}"
                dest = capture_root / rel
                if dry_run:
                    checksum, size = sha256_of(path), path.stat().st_size
                else:
                    checksum, size = copy_and_hash(path, dest)
                    staged.append(dest.with_name(dest.name + ".partial"))
                if checksum in seen:
                    duplicate = True
                assets.append(
                    {
                        "id": str(uuid.uuid4()),
                        "role": role,
                        "relative_path": str(rel),
                        "media_type": media_type(role, path),
                        "byte_length": size,
                        "checksum_sha256": checksum,
                        "original_filename": path.name,
                        "dest": dest,
                    }
                )

            if duplicate:
                stats["skipped_duplicate"] += 1
                for asset in assets:
                    partial = asset["dest"].with_name(asset["dest"].name + ".partial")
                    if partial.exists():
                        partial.unlink()
                        staged.remove(partial)
                continue

            primary = next(
                (a for a in assets if a["role"] == "native_raw"),
                next((a for a in assets if a["role"] == "sensor_fits"), assets[0]),
            )

            acquisition = {
                "import_tool": "tools/import_session.py",
                "metadata_source": metadata_source,
                "source_directory": str(files_dir),
                "session_log": str(log_path) if log_path else None,
                "exif_model": str(body) if body else None,
                "group_stem": Path(stem).name,
                "source_relative_path": str(Path(stem).relative_to(files_dir))
                if Path(stem).is_relative_to(files_dir) else stem,
                # Only meaningful when metadata_source == "exif"; None otherwise.
                "timezone_source": timezone_source,
                # `focus_samples.position` and `frames.focuser_position` are
                # bare integers. A motorised run counts steps and a dial
                # indicator counts displacement; nothing in the number says
                # which, so the unit travels with the frame.
                "focuser_position_unit": f"{position_unit_um:g}um",
            }

            manifest = {
                "version": 1,
                "frame_id": frame_id,
                "captured_at": captured_at,
                "capture_type": capture_type,
                "assets": [
                    {k: v for k, v in a.items() if k != "dest"} for a in assets
                ],
            }

            if not dry_run:
                manifest_path = capture_root / night / capture_type / f"{frame_id}.{MANIFEST_VERSION}"
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(manifest, indent=2))
                fsync_dir(manifest_path.parent)

                con.execute(
                    """
                    INSERT INTO frames (
                        id, session_id, local_path, checksum_sha256,
                        exposure_s, gain, captured_at, quality_json,
                        rig_id, camera_name, capture_type, capture_status,
                        iso, temperature_state, driver_name, acquisition_json,
                        night_id
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        frame_id,
                        session_id,
                        str(capture_root / primary["relative_path"]),
                        primary["checksum_sha256"],
                        exposure_s,
                        None,
                        captured_at,
                        # Placeholder. A frame that turns out to belong to a
                        # sweep gets a measured quality_json from
                        # `import_focus_runs`; one that does not keeps this,
                        # which the dataset builder correctly reads as "gate
                        # never ran" and drops from any FQE set.
                        quality_json_for(None, declared_saturation),
                        rig_id,
                        str(body) if body else None,
                        capture_type,
                        "committed",
                        iso,
                        # No thermometer inside the camera we can trust here.
                        # The ambient series lives in environment_log instead.
                        "missing",
                        "offline_import",
                        json.dumps(acquisition),
                        night,
                    ),
                )
                for asset in assets:
                    con.execute(
                        """
                        INSERT INTO frame_assets (
                            id, frame_id, role, variant, original_filename,
                            relative_path, media_type, byte_length,
                            checksum_sha256, provenance_json, upload_policy,
                            created_at
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            asset["id"],
                            frame_id,
                            asset["role"],
                            0,
                            asset["original_filename"],
                            asset["relative_path"],
                            asset["media_type"],
                            asset["byte_length"],
                            asset["checksum_sha256"],
                            json.dumps({"imported_from": str(files_dir)}),
                            "metadata_only",
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    stats["assets_inserted"] += 1

            for asset in assets:
                seen.add(asset["checksum_sha256"])
                filename_to_frame[asset["original_filename"]] = frame_id
                filename_to_frame.setdefault(Path(asset["original_filename"]).stem, frame_id)
            stats["frames_inserted"] += 1

        # ─── Environment + disturbance series ───
        if not dry_run:
            for entry in log.get("environment", []):
                con.execute(
                    """
                    INSERT INTO environment_log (
                        session_id, night_id, rig_id, observed_at,
                        ambient_temp_c, humidity_pct, dew_point_c, wind_beaufort,
                        sky_class, transparency_nelm, seeing_impression,
                        source, notes
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        session_id,
                        night_id_for(entry.get("t", ""), default_night),
                        rig_id,
                        entry.get("t"),
                        entry.get("ambient_temp_c"),
                        entry.get("humidity_pct"),
                        entry.get("dew_point_c"),
                        entry.get("wind_beaufort"),
                        entry.get("sky_class"),
                        entry.get("transparency_nelm"),
                        entry.get("seeing_impression"),
                        entry.get("source", "manual_log"),
                        entry.get("notes"),
                    ),
                )
                stats["environment_rows"] += 1

            for entry in log.get("disturbances", []):
                con.execute(
                    """
                    INSERT INTO disturbance_log (
                        session_id, night_id, occurred_at, kind,
                        affected_run_id, description
                    ) VALUES (?,?,?,?,?,?)
                    """,
                    (
                        session_id,
                        night_id_for(entry.get("t", ""), default_night),
                        entry.get("t"),
                        entry.get("kind", "other"),
                        entry.get("affected_run_id"),
                        entry.get("description"),
                    ),
                )
                stats["disturbance_rows"] += 1

            stats["labels_written"] = attach_sky_labels(con, session_id)

            # Focus runs last: every route in `resolve_position_frames` reads
            # `frames` rows, so they must already be inserted, and the whole
            # thing still rides the one transaction opened above.
            stats.update(
                import_focus_runs(
                    con,
                    log,
                    filename_to_frame,
                    session_id=session_id,
                    rig_id=rig_id,
                    default_night=default_night,
                    declared_night=declared_night,
                    capture_root=capture_root,
                    declared_saturation=declared_saturation,
                    position_unit_um=position_unit_um,
                    match_window_s=match_window_s,
                    dry_run=False,
                )
            )
            con.commit()

            # Only after the transaction commits do the staged files become
            # real. This is the ordering `capture_store.rs` uses and the reason
            # a crash mid-import leaves no half-recorded frame.
            for partial in staged:
                partial.rename(partial.with_name(partial.name[: -len(".partial")]))
            staged.clear()

    except Exception:
        con.rollback()
        for partial in staged:
            if partial.exists():
                partial.unlink()
        raise
    finally:
        con.close()

    return stats


def resolve_position_frames(
    con: sqlite3.Connection,
    entry: dict,
    filename_to_frame: dict[str, str],
    session_id: str,
    match_window_s: float,
) -> tuple[list[str], str | None]:
    """Frame ids belonging to one sweep position, and how they were found.

    Three routes, tried in order of how much they can be trusted:

    1. ``frame_ids`` — unambiguous. For a scripted capture that recorded them.
    2. ``files`` — the schema's own field (Appendix C.5). Matched against the
       frames imported in this run, then against `frame_assets.original_filename`
       for frames imported earlier, so a re-run of the card after the files are
       already in does not need the files present again.
    3. ``t`` — a timestamp window, for frames the *daemon* captured during the
       sweep while the operator turned the knob by hand. Last resort and
       deliberately narrow: a sweep dwells seconds per position, so a loose
       window silently attaches a frame to the wrong position, and a corrupted
       position axis is indistinguishable from bad seeing afterwards.

    Returns ``([], None)`` when nothing matches, so the caller can report the
    position as unresolved instead of inventing a link.
    """
    explicit = entry.get("frame_ids") or []
    if explicit:
        found = [
            row[0]
            for fid in explicit
            for row in con.execute("SELECT id FROM frames WHERE id = ?", (str(fid),))
        ]
        if found:
            return found, "frame_ids"

    names = entry.get("files") or []
    if names:
        resolved: list[str] = []
        for name in names:
            key = Path(str(name)).name
            frame_id = filename_to_frame.get(key) or filename_to_frame.get(Path(key).stem)
            if frame_id is None:
                row = con.execute(
                    "SELECT frame_id FROM frame_assets WHERE original_filename = ? LIMIT 1",
                    (key,),
                ).fetchone()
                frame_id = row[0] if row else None
            if frame_id and frame_id not in resolved:
                resolved.append(frame_id)
        if resolved:
            return resolved, "files"

    stamp = entry.get("t")
    if stamp:
        try:
            target = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        except ValueError:
            return [], None
        if target.tzinfo is None:
            target = target.astimezone()
        rows = con.execute(
            "SELECT id, captured_at FROM frames WHERE session_id = ?", (session_id,)
        ).fetchall()
        hits = []
        for row in rows:
            try:
                got = datetime.fromisoformat(str(row[1]).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue
            if got.tzinfo is None:
                got = got.astimezone()
            if abs((got - target).total_seconds()) <= match_window_s:
                hits.append(row[0])
        if hits:
            return hits, "timestamp"

    return [], None


def position_value(entry: dict, position_unit_um: float) -> int | None:
    """The integer position axis for one sweep entry.

    ``position`` is taken verbatim when present — it is already in whatever unit
    the run uses. ``reading_um`` is a dial-indicator displacement in micrometres
    and is divided by ``position_unit_um`` to land on the same axis. The unit is
    recorded per frame rather than assumed, because a motorised run's steps and
    a hand-cranked run's micrometres are both integers and nothing in the number
    itself says which it is.
    """
    if entry.get("position") is not None:
        try:
            return int(round(float(entry["position"])))
        except (TypeError, ValueError):
            return None
    if entry.get("reading_um") is not None:
        try:
            return int(round(float(entry["reading_um"]) / float(position_unit_um)))
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return None


def import_focus_runs(
    con: sqlite3.Connection,
    log: dict,
    filename_to_frame: dict[str, str],
    session_id: str,
    rig_id: str | None,
    default_night: str,
    capture_root: Path,
    declared_saturation: float | None,
    declared_night: str | None = None,
    position_unit_um: float = DEFAULT_POSITION_UNIT_UM,
    match_window_s: float = DEFAULT_MATCH_WINDOW_S,
    dry_run: bool = False,
) -> dict:
    """Turn the field card's ``runs[]`` array into `focus_runs` + `focus_samples`.

    WHY THIS EXISTS

    Spec Section 9.1: "Every autofocus run generates labeled training data
    automatically — the highest-value dataset in the system." That holds for a
    motorised focuser, where `/focus/run` drives the sweep and
    `api/focus.rs::persist_focus_run` writes the rows. On a manual rack-and-
    pinion it does not: the daemon cannot command the focuser, so no run is
    recorded, and `build_focus_quality_dataset.py` — which reads
    ``FROM focus_samples JOIN frames ON frames.id = focus_samples.frame_id`` —
    cannot see the frames at all.

    Runbook RB-2 is the primary focus-data procedure until an electronic
    focuser exists. Without this function it produces files that no downstream
    stage can read. With it, a hand-cranked sweep and a daemon-driven one land
    in the same two tables and split on the same `run_id` group key.

    WHAT IT DOES NOT CLAIM

    ``state`` is `Imported`, never `Converged`. ``best_position`` is the
    **sample argmin** over measured positions, not a fitted V-curve vertex;
    on a 10 µm dial against a ~42 µm critical focus zone that is a coarse
    estimate, and a Focus Offset Predictor built on these runs should refit
    rather than trust it. Runs the card marks aborted keep every sample and are
    stored as `Aborted`, because dropping them would put survivorship bias into
    the one dataset that is expensive to collect.
    """
    stats = {
        "focus_runs_inserted": 0,
        "focus_samples_inserted": 0,
        "positions_unresolved": 0,
        "samples_unmeasurable": 0,
    }

    runs = [
        run
        for run in log.get("runs", [])
        if str(run.get("run_type", "focus_sweep")) == "focus_sweep"
    ]
    if not runs:
        return stats

    for run in runs:
        run_id = str(run.get("run_id") or uuid.uuid4())
        positions = run.get("positions") or []
        if not positions:
            continue

        samples: list[tuple] = []
        for entry in positions:
            position = position_value(entry, position_unit_um)
            if position is None:
                stats["positions_unresolved"] += 1
                continue

            frame_ids, _route = resolve_position_frames(
                con, entry, filename_to_frame, session_id, match_window_s
            )
            if not frame_ids:
                stats["positions_unresolved"] += 1
                continue

            for frame_id in frame_ids:
                row = con.execute(
                    "SELECT local_path, captured_at FROM frames WHERE id = ?",
                    (frame_id,),
                ).fetchone()
                if row is None:
                    stats["positions_unresolved"] += 1
                    continue

                path = Path(str(row[0]))
                if not path.is_absolute():
                    path = capture_root / path
                source = readable_path(path)
                measurement = measure_frame(source, declared_saturation) if source else None
                if measurement is None:
                    # `focus_samples.hfr` is NOT NULL and there is no honest
                    # number to put there. Skipping loses this frame from the
                    # run; fabricating one would corrupt the V-curve for every
                    # frame in it.
                    stats["samples_unmeasurable"] += 1
                    continue

                samples.append(
                    (
                        run_id,
                        position,
                        measurement["hfr_px"],
                        measurement["hfd_px"],
                        measurement["star_count"],
                        entry.get("t") or row[1],
                        frame_id,
                        measurement,
                    )
                )

        if not samples:
            continue

        best = min(samples, key=lambda s: s[2])
        status = str(run.get("status", "completed")).lower()
        state = RUN_STATE_ABORTED if status == "aborted" else RUN_STATE_IMPORTED
        started = run.get("started_at") or samples[0][5]
        completed = run.get("completed_at") or samples[-1][5]

        if dry_run:
            stats["focus_runs_inserted"] += 1
            stats["focus_samples_inserted"] += len(samples)
            continue

        con.execute(
            """
            INSERT OR REPLACE INTO focus_runs (
                id, rig_id, filter_name, start_position, best_position,
                best_hfr, state, temperature, started_at, completed_at,
                night_id, session_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                rig_id or "unknown-rig",
                run.get("filter"),
                position_value(run, position_unit_um) or samples[0][1],
                best[1] if state == RUN_STATE_IMPORTED else None,
                best[2] if state == RUN_STATE_IMPORTED else None,
                state,
                run.get("ambient_temp_c"),
                started,
                completed,
                # A sweep happens inside one observing night by definition,
                # so a declared night on the card is authoritative for the run.
                declared_night or night_id_for(str(started), default_night),
                session_id,
            ),
        )
        stats["focus_runs_inserted"] += 1

        for run_key, position, hfr, hfd, stars, captured, frame_id, measurement in samples:
            con.execute(
                """
                INSERT INTO focus_samples (
                    run_id, position, hfr, hfd, star_count, captured_at, frame_id
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (run_key, position, hfr, hfd, stars, captured, frame_id),
            )
            # The position axis and the measurement both belong on the frame
            # too: the FOP reads `frames.focuser_position`, and the dataset
            # builder gates on `frames.quality_json`.
            con.execute(
                """
                UPDATE frames
                   SET focuser_position = ?,
                       quality_json = ?
                 WHERE id = ?
                """,
                (position, quality_json_for(measurement, declared_saturation), frame_id),
            )
            stats["focus_samples_inserted"] += 1

    return stats


def attach_sky_labels(con: sqlite3.Connection, session_id: str, max_gap_minutes: int = 15) -> int:
    """Stamp each frame with the observer's sky class from the nearest log entry.

    The label goes into `frame_measurements` rather than into `frames`, because
    that table is append-only and versioned: a human label and a future COC
    model's prediction are both opinions about the same frame, and keeping them
    side by side is what makes the model checkable against the person.
    """
    entries = con.execute(
        "SELECT observed_at, sky_class FROM environment_log "
        "WHERE session_id = ? AND sky_class IS NOT NULL ORDER BY observed_at",
        (session_id,),
    ).fetchall()
    if not entries:
        return 0

    def parse(value: str) -> datetime | None:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    stamps = [(parse(r["observed_at"]), r["sky_class"]) for r in entries]
    stamps = [(t, c) for t, c in stamps if t is not None]
    if not stamps:
        return 0

    written = 0
    frames = con.execute(
        "SELECT id, captured_at FROM frames WHERE session_id = ?", (session_id,)
    ).fetchall()
    for frame in frames:
        taken = parse(frame["captured_at"])
        if taken is None:
            continue
        nearest, gap = None, None
        for stamp, sky_class in stamps:
            delta = abs((taken - stamp).total_seconds())
            if gap is None or delta < gap:
                nearest, gap = sky_class, delta
        if nearest is None or gap is None or gap > max_gap_minutes * 60:
            continue
        con.execute(
            """
            INSERT INTO frame_measurements (
                frame_id, measurement_type, value, payload_json,
                pipeline_version, valid
            ) VALUES (?,?,?,?,?,1)
            """,
            (
                frame["id"],
                "cloud",
                float(nearest),
                json.dumps(
                    {
                        "sky_class": nearest,
                        # 5-class collapses to the spec's binary head; see D7.
                        "sky_class_binary": 0 if nearest <= 1 else 1,
                        "label_source": "observer",
                        "gap_seconds": round(gap, 1),
                    }
                ),
                "manual_log_v1",
            ),
        )
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import frames the daemon did not capture (gaps G2/G4/G5)."
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--files", required=True, type=Path, help="Directory of camera files")
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--log", type=Path, help="Session JSON log (Appendix C.5 schema)")
    parser.add_argument("--capture-type", default="light", choices=sorted(VALID_CAPTURE_TYPES))
    parser.add_argument("--rig-id")
    parser.add_argument("--migrations", type=Path, help="Path to achernar-daemon/migrations")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--position-unit-um",
        type=float,
        default=DEFAULT_POSITION_UNIT_UM,
        help=(
            "Micrometres per unit of the integer position axis, applied to a "
            "field card's `reading_um`. Default 1.0 stores micrometres. Use 10 "
            "to store 0.01 mm dial-indicator divisions. Recorded per frame in "
            "acquisition_json.focuser_position_unit."
        ),
    )
    parser.add_argument(
        "--match-window-s",
        type=float,
        default=DEFAULT_MATCH_WINDOW_S,
        help=(
            "Seconds a frame may sit from a sweep position's logged timestamp "
            "and still be attached to it. Only used when the card gives no "
            "`files` or `frame_ids`. Keep it tight: a wrong match corrupts the "
            "position axis invisibly."
        ),
    )
    parser.add_argument(
        "--camera-utc-offset",
        help=(
            "UTC offset of the camera clock, e.g. -04:00 for EDT. Used only when "
            "EXIF carries no OffsetTimeOriginal. Overrides session.camera_utc_offset."
        ),
    )
    args = parser.parse_args()

    stats = import_session(
        db_path=args.db,
        files_dir=args.files,
        capture_root=args.capture_root,
        log_path=args.log,
        capture_type=args.capture_type,
        rig_id=args.rig_id,
        dry_run=args.dry_run,
        migrations_dir=args.migrations,
        position_unit_um=args.position_unit_um,
        match_window_s=args.match_window_s,
        camera_utc_offset=args.camera_utc_offset,
    )

    prefix = "would import" if args.dry_run else "imported"
    print(f"✓ {prefix} {stats['frames_inserted']} frames / {stats['assets_inserted']} assets")
    for key in (
        "skipped_duplicate", "metadata_from_exif", "metadata_from_log",
        "metadata_from_mtime", "environment_rows", "disturbance_rows",
        "labels_written",
    ):
        if stats[key]:
            print(f"  {key.replace('_', ' ')}: {stats[key]}")

    if stats["timezone_assumed"]:
        print(
            f"  ! {stats['timezone_assumed']} frame(s) had no EXIF UTC offset; assumed this "
            "machine's time zone. Pass --camera-utc-offset or set the camera's time zone."
        )

    if stats["focus_runs_inserted"]:
        print(
            f"  focus runs: {stats['focus_runs_inserted']} "
            f"({stats['focus_samples_inserted']} samples)"
        )
    if stats["positions_unresolved"]:
        print(
            f"  ! {stats['positions_unresolved']} sweep position(s) matched no frame — "
            "check the `files` entries on the field card against the filenames on the card reader"
        )
    if stats["samples_unmeasurable"]:
        print(
            f"  ! {stats['samples_unmeasurable']} frame(s) in a sweep yielded no HFR "
            "(too few stars, unreadable format, or RAW without a derived FITS) and "
            "were left out of the run rather than given a fabricated width"
        )
    if stats["metadata_from_mtime"]:
        print(
            "  ! some timestamps came from file mtime, not the shutter — "
            "recorded in acquisition_json.metadata_source"
        )
    if stats["frames_inserted"] and not stats["focus_runs_inserted"] and stats["focus_samples_inserted"] == 0:
        print(
            "  note: no focus runs in this import. Frames carry "
            "saturation_gate_applied=false and the FQE dataset builder will skip "
            "them — expected for a COC/calibration campaign, a problem for a sweep."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
