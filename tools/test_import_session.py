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
# `measure_frame` calls the same labeler the dataset builder uses.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))

from import_session import (  # noqa: E402
    classify,
    group_by_capture,
    import_session,
    night_id_for,
    position_value,
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


# ═══════════════════════════════════════════════════════════════════════════
# Manual focus sweeps (RB-2)
#
# The gap these cover: `build_focus_quality_dataset.py` reads
# `FROM focus_samples JOIN frames ON frames.id = focus_samples.frame_id`, and
# `focus_samples` rows were previously written only by a daemon-driven
# `/focus/run`. A hand-cranked sweep on a manual focuser produced frames no
# downstream stage could see.
# ═══════════════════════════════════════════════════════════════════════════


def write_star_field(
    path: Path,
    hfr_sigma: float,
    n_stars: int = 14,
    size: int = 160,
    bit_depth: int = 8,
    seed: int = 7,
    noise_seed: int = 0,
) -> None:
    """A synthetic star field whose PSF width is set by `hfr_sigma`.

    Star positions come from `seed` and are held fixed across a sweep, so the
    only thing varying between positions is the width — which is what a focus
    sweep is, and what makes the resulting V-curve interpretable.

    Noise comes from `noise_seed`, which must differ per frame. A sweep is
    symmetric about focus, so two positions either side share a sigma; with a
    shared noise draw they would be byte-identical and the importer's
    content-hash duplicate detection would (correctly) drop one of them. Real
    frames are never byte-identical, so the fixture must not be either.
    """
    import numpy as np
    from PIL import Image

    star_rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(1000 + noise_seed)
    ceiling = 255.0 if bit_depth == 8 else 65535.0
    sky = ceiling * 0.08
    image = noise_rng.normal(sky, ceiling * 0.006, size=(size, size))

    yy, xx = np.mgrid[0:size, 0:size]
    margin = 24
    for _ in range(n_stars):
        cx = star_rng.uniform(margin, size - margin)
        cy = star_rng.uniform(margin, size - margin)
        amplitude = ceiling * star_rng.uniform(0.25, 0.45)
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        image += amplitude * np.exp(-r2 / (2.0 * hfr_sigma**2))

    image = np.clip(image, 0, ceiling)
    if bit_depth == 8:
        Image.fromarray(image.astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray(image.astype(np.uint16)).save(path)


def sweep_log(
    positions_um: list[int],
    files_per_position: list[list[str]],
    *,
    session_id: str = "sess-sweep",
    run_id: str = "run-sweep-1",
    status: str = "completed",
    saturation_adu: float | None = None,
) -> dict:
    session = {
        "session_id": session_id,
        "night_id": "2026-09-14",
        "rig_id": "R2",
        "camera": "SVBony SV105",
    }
    if saturation_adu is not None:
        session["saturation_adu"] = saturation_adu
    return {
        "session": session,
        "runs": [
            {
                "run_id": run_id,
                "run_type": "focus_sweep",
                "target_name": "M39",
                "direction": "outward",
                "status": status,
                "positions": [
                    {"index": i, "reading_um": pos, "files": files}
                    for i, (pos, files) in enumerate(zip(positions_um, files_per_position))
                ],
            }
        ],
    }


class TestManualFocusSweep(unittest.TestCase):
    def _setup(self, d: Path):
        db = d / "achernar.db"
        apply_migrations(db)
        files = d / "cards"
        files.mkdir()
        root = d / "archive"
        return db, files, root

    def _write_v_curve(self, files: Path, bit_depth: int = 8):
        """Seven positions, sharpest in the middle — a V-curve with an interior
        minimum, which is RB-2's acceptance criterion."""
        sigmas = [3.4, 2.8, 2.2, 1.8, 2.2, 2.8, 3.4]
        positions = [1180, 1194, 1208, 1222, 1236, 1250, 1264]
        names = []
        for i, sigma in enumerate(sigmas):
            name = f"SWEEP_{i:03d}.png"
            write_star_field(files / name, sigma, bit_depth=bit_depth, noise_seed=i)
            names.append([name])
        return positions, names

    def test_sweep_becomes_focus_runs_and_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            db, files, root = self._setup(d)
            positions, names = self._write_v_curve(files)

            log = d / "session.json"
            log.write_text(json.dumps(sweep_log(positions, names)))

            stats = import_session(
                db_path=db, files_dir=files, capture_root=root,
                log_path=log, migrations_dir=MIGRATIONS,
            )

            self.assertEqual(stats["frames_inserted"], 7)
            self.assertEqual(stats["focus_runs_inserted"], 1)
            self.assertEqual(stats["focus_samples_inserted"], 7)
            self.assertEqual(stats["positions_unresolved"], 0)

            con = sqlite3.connect(db)
            con.row_factory = sqlite3.Row

            run = con.execute("SELECT * FROM focus_runs").fetchone()
            self.assertEqual(run["state"], "Imported")
            self.assertEqual(run["rig_id"], "R2")
            self.assertEqual(run["night_id"], "2026-09-14")

            rows = con.execute(
                "SELECT position, hfr, star_count, frame_id FROM focus_samples ORDER BY position"
            ).fetchall()
            self.assertEqual([r["position"] for r in rows], positions)
            self.assertTrue(all(r["frame_id"] for r in rows))
            self.assertTrue(all(r["hfr"] > 0 for r in rows))

            # The measured curve must actually turn over, and the recorded
            # best_position must be the argmin of what was measured.
            hfrs = [r["hfr"] for r in rows]
            self.assertLess(min(hfrs), hfrs[0])
            self.assertLess(min(hfrs), hfrs[-1])
            self.assertEqual(run["best_position"], rows[hfrs.index(min(hfrs))]["position"])
            con.close()

    def test_frames_carry_position_and_a_gated_quality_json(self):
        """The two columns the downstream stages actually read."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            db, files, root = self._setup(d)
            positions, names = self._write_v_curve(files)
            log = d / "session.json"
            log.write_text(json.dumps(sweep_log(positions, names)))

            import_session(db_path=db, files_dir=files, capture_root=root,
                           log_path=log, migrations_dir=MIGRATIONS)

            con = sqlite3.connect(db)
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT focuser_position, quality_json, night_id, acquisition_json "
                "FROM frames ORDER BY focuser_position"
            ).fetchall()
            self.assertEqual([r["focuser_position"] for r in rows], positions)

            for row in rows:
                q = json.loads(row["quality_json"])
                # 8-bit container: the container IS the clip point, so the gate
                # fires by construction — the SV105 case.
                self.assertTrue(q["saturation_gate_applied"])
                self.assertEqual(q["max_adu"], 255.0)
                self.assertEqual(q["saturation_adu"], 255.0)
                self.assertGreater(q["hfr_px"], 0)
                self.assertGreater(q["star_count"], 0)
                self.assertEqual(row["night_id"], "2026-09-14")
                acq = json.loads(row["acquisition_json"])
                self.assertEqual(acq["focuser_position_unit"], "1um")
            con.close()

    def test_sixteen_bit_without_a_declared_saturation_is_not_gated(self):
        """The D1 case. A 16-bit container says nothing about where the sensor
        clips, so the gate must report itself as not performed — which makes the
        dataset builder drop the frame instead of silently trusting it."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            db, files, root = self._setup(d)
            positions, names = self._write_v_curve(files, bit_depth=16)
            log = d / "session.json"
            log.write_text(json.dumps(sweep_log(positions, names)))

            import_session(db_path=db, files_dir=files, capture_root=root,
                           log_path=log, migrations_dir=MIGRATIONS)

            con = sqlite3.connect(db)
            q = json.loads(
                con.execute("SELECT quality_json FROM frames LIMIT 1").fetchone()[0]
            )
            self.assertFalse(q["saturation_gate_applied"])
            self.assertIsNone(q["saturation_adu"])
            self.assertEqual(q["max_adu"], 65535.0)
            con.close()

    def test_declared_saturation_gates_a_sixteen_bit_sweep(self):
        """…and BP-2's measured value closes it."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            db, files, root = self._setup(d)
            positions, names = self._write_v_curve(files, bit_depth=16)
            log = d / "session.json"
            log.write_text(json.dumps(
                sweep_log(positions, names, saturation_adu=16383.0)
            ))

            import_session(db_path=db, files_dir=files, capture_root=root,
                           log_path=log, migrations_dir=MIGRATIONS)

            con = sqlite3.connect(db)
            q = json.loads(
                con.execute("SELECT quality_json FROM frames LIMIT 1").fetchone()[0]
            )
            self.assertTrue(q["saturation_gate_applied"])
            self.assertEqual(q["saturation_adu"], 16383.0)
            con.close()

    def test_position_unit_scales_dial_readings(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            db, files, root = self._setup(d)
            positions, names = self._write_v_curve(files)
            log = d / "session.json"
            log.write_text(json.dumps(sweep_log(positions, names)))

            import_session(db_path=db, files_dir=files, capture_root=root,
                           log_path=log, migrations_dir=MIGRATIONS,
                           position_unit_um=10.0)

            con = sqlite3.connect(db)
            got = [r[0] for r in con.execute(
                "SELECT position FROM focus_samples ORDER BY position")]
            self.assertEqual(got, sorted(round(p / 10) for p in positions))
            acq = json.loads(
                con.execute("SELECT acquisition_json FROM frames LIMIT 1").fetchone()[0]
            )
            self.assertEqual(acq["focuser_position_unit"], "10um")
            con.close()

    def test_aborted_run_keeps_its_samples_and_claims_no_best(self):
        """Dropping a failed run would put survivorship bias into the one
        dataset that is expensive to collect."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            db, files, root = self._setup(d)
            positions, names = self._write_v_curve(files)
            log = d / "session.json"
            log.write_text(json.dumps(
                sweep_log(positions, names, status="aborted")
            ))

            stats = import_session(db_path=db, files_dir=files, capture_root=root,
                                   log_path=log, migrations_dir=MIGRATIONS)
            self.assertEqual(stats["focus_samples_inserted"], 7)

            con = sqlite3.connect(db)
            con.row_factory = sqlite3.Row
            run = con.execute("SELECT * FROM focus_runs").fetchone()
            self.assertEqual(run["state"], "Aborted")
            self.assertIsNone(run["best_position"])
            self.assertIsNone(run["best_hfr"])
            con.close()

    def test_unmatched_filename_is_reported_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            db, files, root = self._setup(d)
            positions, names = self._write_v_curve(files)
            names[3] = ["NOT_ON_THE_CARD.png"]
            log = d / "session.json"
            log.write_text(json.dumps(sweep_log(positions, names)))

            stats = import_session(db_path=db, files_dir=files, capture_root=root,
                                   log_path=log, migrations_dir=MIGRATIONS)
            self.assertEqual(stats["positions_unresolved"], 1)
            self.assertEqual(stats["focus_samples_inserted"], 6)

    def test_starless_frame_is_skipped_rather_than_given_a_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            db, files, root = self._setup(d)
            positions, names = self._write_v_curve(files)
            # Overwrite one position with an empty sky: no sources, so no HFR.
            write_star_field(files / names[2][0], 2.2, n_stars=0, noise_seed=99)
            log = d / "session.json"
            log.write_text(json.dumps(sweep_log(positions, names)))

            stats = import_session(db_path=db, files_dir=files, capture_root=root,
                                   log_path=log, migrations_dir=MIGRATIONS)
            self.assertEqual(stats["samples_unmeasurable"], 1)
            self.assertEqual(stats["focus_samples_inserted"], 6)

    def test_a_card_with_no_runs_still_imports_frames(self):
        """COC and calibration campaigns have no sweeps and must be unaffected."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            db, files, root = self._setup(d)
            write_jpeg(files / "IMG_0001.JPG", "2026:08:07 22:10:00")
            log = d / "session.json"
            log.write_text(json.dumps({
                "session": {"session_id": "sess-coc", "night_id": "2026-08-07",
                            "rig_id": "R4"},
                "environment": [
                    {"t": "2026-08-07T22:10:00-04:00", "sky_class": 1,
                     "ambient_temp_c": 21.0}
                ],
            }))
            stats = import_session(db_path=db, files_dir=files, capture_root=root,
                                   log_path=log, migrations_dir=MIGRATIONS)
            self.assertEqual(stats["frames_inserted"], 1)
            self.assertEqual(stats["focus_runs_inserted"], 0)
            self.assertEqual(stats["environment_rows"], 1)


class TestPositionValue(unittest.TestCase):
    def test_explicit_position_wins_over_reading(self):
        self.assertEqual(position_value({"position": 4200, "reading_um": 99}, 10.0), 4200)

    def test_reading_is_scaled(self):
        self.assertEqual(position_value({"reading_um": 1240}, 10.0), 124)
        self.assertEqual(position_value({"reading_um": 1240}, 1.0), 1240)

    def test_missing_position_is_none_not_zero(self):
        self.assertIsNone(position_value({"t": "2026-09-14T22:00:00"}, 1.0))
