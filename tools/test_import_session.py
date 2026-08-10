#!/usr/bin/env python3
"""
Tests for tools/import_session.py, run against the *real* migration files.

Applying migrations/001..006 to a fresh database is itself part of the test:
if the schema and the importer ever disagree about a column name, this fails
rather than silently writing a row the daemon cannot read.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from import_session import (  # noqa: E402
    classify,
    group_by_capture,
    import_session,
    night_id_for,
)

REPO = Path(__file__).resolve().parent.parent
MIGRATIONS = REPO / "achernar-daemon" / "migrations"


def apply_migrations(db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    for sql_file in sorted(MIGRATIONS.glob("*.sql")):
        con.executescript(sql_file.read_text())
    con.commit()
    con.close()


def write_jpeg(path: Path, exif_datetime: str | None = None) -> None:
    from PIL import Image

    img = Image.new("RGB", (16, 16), (10, 10, 30))
    if exif_datetime:
        exif = img.getexif()
        exif[306] = exif_datetime          # DateTime
        exif[36867] = exif_datetime        # DateTimeOriginal
        img.save(path, exif=exif)
    else:
        img.save(path)


class TestHelpers(unittest.TestCase):
    def test_classify_roles(self):
        self.assertEqual(classify(Path("IMG_0001.CR3")), "native_raw")
        self.assertEqual(classify(Path("IMG_0001.rw2")), "native_raw")
        self.assertEqual(classify(Path("IMG_0001.JPG")), "camera_jpeg")
        self.assertEqual(classify(Path("frame.fits")), "sensor_fits")
        self.assertIsNone(classify(Path("notes.txt")))

    def test_raw_and_jpeg_pair_is_one_exposure(self):
        groups = group_by_capture(
            [Path("IMG_1.CR3"), Path("IMG_1.JPG"), Path("IMG_2.CR3")]
        )
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups["IMG_1"]), 2)

    def test_night_id_rolls_at_noon_not_midnight(self):
        # 01:30 belongs to the night that began the previous evening.
        self.assertEqual(night_id_for("2026-08-08T01:30:00+00:00", "x"), "2026-08-07")
        self.assertEqual(night_id_for("2026-08-07T22:30:00+00:00", "x"), "2026-08-07")


class TestImport(unittest.TestCase):
    def _setup(self, d: Path):
        db = d / "achernar.db"
        apply_migrations(db)
        files = d / "cards"
        files.mkdir()
        root = d / "archive"
        return db, files, root

    def test_imports_frames_assets_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            db, files, root = self._setup(d)

            for i in range(3):
                write_jpeg(files / f"IMG_000{i}.JPG", f"2026:08:07 22:{10 + i * 10:02d}:00")

            log = d / "session.json"
            log.write_text(json.dumps({
                "session": {"session_id": "sess-1", "night_id": "2026-08-07",
                            "rig_id": "R4", "camera": "Canon EOS 760D"},
                "environment": [
                    {"t": "2026-08-07T22:10:00+00:00", "ambient_temp_c": 21.4,
                     "humidity_pct": 68, "wind_beaufort": 1, "sky_class": 0},
                    {"t": "2026-08-07T22:30:00+00:00", "ambient_temp_c": 20.9,
                     "sky_class": 2},
                ],
                "disturbances": [
                    {"t": "2026-08-07T22:25:00+00:00", "kind": "headlights",
                     "description": "car in driveway"},
                ],
            }))

            stats = import_session(
                db_path=db, files_dir=files, capture_root=root, log_path=log,
                capture_type="light", migrations_dir=MIGRATIONS,
            )

            self.assertEqual(stats["frames_inserted"], 3)
            self.assertEqual(stats["assets_inserted"], 3)
            self.assertEqual(stats["environment_rows"], 2)
            self.assertEqual(stats["disturbance_rows"], 1)
            self.assertEqual(stats["metadata_from_exif"], 3)

            con = sqlite3.connect(db)
            con.row_factory = sqlite3.Row

            frames = con.execute("SELECT * FROM frames").fetchall()
            self.assertEqual(len(frames), 3)
            for row in frames:
                self.assertEqual(row["driver_name"], "offline_import")
                self.assertEqual(row["capture_type"], "light")
                self.assertEqual(row["rig_id"], "R4")
                # Honest about what was never measured.
                self.assertEqual(row["temperature_state"], "missing")
                acq = json.loads(row["acquisition_json"])
                self.assertEqual(acq["metadata_source"], "exif")
                self.assertTrue(Path(row["local_path"]).exists())

            # Assets landed, checksums match the bytes on disk, no .partial left.
            for asset in con.execute("SELECT * FROM frame_assets").fetchall():
                path = root / asset["relative_path"]
                self.assertTrue(path.exists(), path)
                import hashlib
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    asset["checksum_sha256"],
                )
            self.assertEqual(list(root.rglob("*.partial")), [])

            # The observer's sky labels reached frame_measurements.
            labels = con.execute(
                "SELECT * FROM frame_measurements WHERE measurement_type='cloud'"
            ).fetchall()
            self.assertEqual(len(labels), 3)
            payloads = [json.loads(r["payload_json"]) for r in labels]
            self.assertEqual({p["label_source"] for p in payloads}, {"observer"})
            # sky_class 0 and 1 collapse to binary 0; 2+ to binary 1.
            for p in payloads:
                self.assertEqual(p["sky_class_binary"], 0 if p["sky_class"] <= 1 else 1)
            self.assertTrue(all(r["pipeline_version"] == "manual_log_v1" for r in labels))
            con.close()

    def test_reimport_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            db, files, root = self._setup(d)
            write_jpeg(files / "IMG_0001.JPG", "2026:08:07 22:10:00")

            first = import_session(db_path=db, files_dir=files, capture_root=root,
                                   migrations_dir=MIGRATIONS)
            second = import_session(db_path=db, files_dir=files, capture_root=root,
                                    migrations_dir=MIGRATIONS)

            self.assertEqual(first["frames_inserted"], 1)
            self.assertEqual(second["frames_inserted"], 0)
            self.assertEqual(second["skipped_duplicate"], 1)

            con = sqlite3.connect(db)
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM frames").fetchone()[0], 1
            )
            con.close()

    def test_dry_run_touches_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            db, files, root = self._setup(d)
            write_jpeg(files / "IMG_0001.JPG", "2026:08:07 22:10:00")

            stats = import_session(db_path=db, files_dir=files, capture_root=root,
                                   dry_run=True, migrations_dir=MIGRATIONS)
            self.assertEqual(stats["frames_inserted"], 1)

            con = sqlite3.connect(db)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM frames").fetchone()[0], 0)
            con.close()
            self.assertFalse(root.exists() and any(root.rglob("*")))

    def test_no_capture_attempts_rows_are_invented(self):
        """An offline import commanded nothing, so it must not claim attempts."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            db, files, root = self._setup(d)
            write_jpeg(files / "IMG_0001.JPG", "2026:08:07 22:10:00")
            import_session(db_path=db, files_dir=files, capture_root=root,
                           migrations_dir=MIGRATIONS)
            con = sqlite3.connect(db)
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM capture_attempts").fetchone()[0], 0
            )
            con.close()

    def test_missing_exif_falls_back_and_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            db, files, root = self._setup(d)
            write_jpeg(files / "IMG_0001.JPG", None)  # no EXIF datetime

            stats = import_session(db_path=db, files_dir=files, capture_root=root,
                                   migrations_dir=MIGRATIONS)
            self.assertEqual(stats["metadata_from_mtime"], 1)

            con = sqlite3.connect(db)
            con.row_factory = sqlite3.Row
            acq = json.loads(
                con.execute("SELECT acquisition_json FROM frames").fetchone()[0]
            )
            self.assertEqual(acq["metadata_source"], "file_mtime")
            con.close()

    def test_rejects_bad_capture_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            db, files, root = self._setup(d)
            write_jpeg(files / "IMG_0001.JPG", "2026:08:07 22:10:00")
            with self.assertRaises(ValueError):
                import_session(db_path=db, files_dir=files, capture_root=root,
                               capture_type="lights", migrations_dir=MIGRATIONS)


class TestMigrations(unittest.TestCase):
    def test_all_migrations_apply_to_a_fresh_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "fresh.db"
            apply_migrations(db)
            con = sqlite3.connect(db)
            tables = {
                r[0] for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            for expected in (
                "frames", "frame_assets", "frame_measurements", "capture_attempts",
                "focus_runs", "focus_samples", "environment_log", "disturbance_log",
            ):
                self.assertIn(expected, tables)
            con.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
