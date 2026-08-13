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


def exif_datetime(exif: dict) -> str | None:
    for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
        value = exif.get(key)
        if not value:
            continue
        try:
            dt = datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    return None


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


# ─── Import ──────────────────────────────────────────────────────────────────


def group_by_capture(files: list[Path]) -> dict[str, list[Path]]:
    """A RAW and its in-camera JPEG are one exposure, not two.

    Grouped by filename stem, which is how every camera in the plan names a
    RAW+JPEG pair (IMG_0001.CR3 / IMG_0001.JPG).
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(files):
        groups[path.stem].append(path)
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
    default_night = session.get("night_id") or datetime.now().strftime("%Y-%m-%d")

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
    }
    staged: list[Path] = []

    try:
        for stem, group in group_by_capture(candidates).items():
            stats["groups"] += 1

            # Metadata: EXIF from whichever file in the group yields it.
            exif: dict = {}
            for path in group:
                exif = read_exif(path)
                if exif:
                    break

            captured_at = exif_datetime(exif)
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
            night = night_id_for(captured_at, default_night)
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
                "group_stem": stem,
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
                        iso, temperature_state, driver_name, acquisition_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        frame_id,
                        session_id,
                        str(capture_root / primary["relative_path"]),
                        primary["checksum_sha256"],
                        exposure_s,
                        None,
                        captured_at,
                        "{}",
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
    if stats["metadata_from_mtime"]:
        print(
            "  ! some timestamps came from file mtime, not the shutter — "
            "recorded in acquisition_json.metadata_source"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
