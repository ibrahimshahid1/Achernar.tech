"""
Validate the FQE dataset builder against a synthetic SQLite + PNG fixture.

This exercises the exact join/detect/label/normalize path the real builder uses,
without needing real autofocus data — it constructs the same `frames` +
`focus_samples` schema the daemon writes, drops synthetic star PNGs on disk, and
asserts the builder recovers labeled crops grouped by run.
"""
import os
import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.build_focus_quality_dataset import (  # noqa: E402
    build_dataset,
    saturation_level_from_quality,
)
from data_pipeline.synthetic_psf import render_star  # noqa: E402


def _write_frame_png(path, hfr, rng):
    # A frame with a single centered star; the builder detects + crops it.
    img = render_star(hfr, size=96, amp=12000.0, sky=200.0, rng=rng).astype(np.uint16)
    Image.fromarray(img).save(path)


def _make_schema(db_path):
    """The subset of the daemon schema the builder joins across."""
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE frames (id TEXT PRIMARY KEY, local_path TEXT NOT NULL, "
        "checksum_sha256 TEXT, quality_json TEXT NOT NULL DEFAULT '{}', "
        "filter_name TEXT, temperature REAL, exposure_s REAL, gain REAL, captured_at TEXT)"
    )
    con.execute(
        "CREATE TABLE focus_runs (id TEXT PRIMARY KEY, rig_id TEXT, filter_name TEXT, "
        "started_at TEXT)"
    )
    con.execute(
        "CREATE TABLE focus_samples (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, "
        "position INTEGER, hfr REAL, frame_id TEXT)"
    )
    return con


class TestDatasetBuilder(unittest.TestCase):
    def test_builds_labeled_crops_from_fixture(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "achernar.db")
            frames_dir = os.path.join(d, "frames")
            os.makedirs(frames_dir)

            con = sqlite3.connect(db_path)
            con.execute(
                "CREATE TABLE frames (id TEXT PRIMARY KEY, local_path TEXT NOT NULL, "
                "checksum_sha256 TEXT, quality_json TEXT NOT NULL DEFAULT '{}', "
                "filter_name TEXT, temperature REAL, exposure_s REAL, gain REAL, captured_at TEXT)"
            )
            con.execute(
                "CREATE TABLE focus_runs (id TEXT PRIMARY KEY, rig_id TEXT, filter_name TEXT, "
                "started_at TEXT)"
            )
            con.execute(
                "CREATE TABLE focus_samples (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, "
                "position INTEGER, hfr REAL, frame_id TEXT)"
            )

            # Two runs, each a small HFR V-curve.
            n_kept_expected = 0
            for run in range(2):
                con.execute(
                    "INSERT INTO focus_runs (id, rig_id, filter_name, started_at) VALUES (?, ?, ?, ?)",
                    (f"run-{run}", f"rig-{run}", "L", f"2026-07-{20 + run:02d}T01:00:00Z"),
                )
                for i, hfr in enumerate([2.0, 3.5, 5.0]):
                    fid = f"frame-{run}-{i}"
                    png = os.path.join(frames_dir, f"{fid}.png")
                    _write_frame_png(png, hfr, rng)
                    checksum = hashlib.sha256(Path(png).read_bytes()).hexdigest()
                    con.execute(
                        "INSERT INTO frames (id, local_path, checksum_sha256, quality_json, "
                        "filter_name, exposure_s, gain, captured_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            fid,
                            png,
                            checksum,
                            # What the daemon actually writes: the container
                            # ceiling AND the declared sensor saturation point.
                            # The synthetic stars peak near 12000 ADU, so a
                            # 16383 clip point (14-bit body) gates nothing here.
                            f'{{"max_adu": 65535, "saturation_adu": 16383, '
                            f'"saturation_gate_applied": true, '
                            f'"camera_id": "camera-{run}"}}',
                            "L",
                            1.0,
                            100.0,
                            f"2026-07-{20 + run:02d}T01:0{i}:00Z",
                        ),
                    )
                    con.execute(
                        "INSERT INTO focus_samples (run_id, position, hfr, frame_id) VALUES (?, ?, ?, ?)",
                        (f"run-{run}", 10000 + i * 250, hfr, fid),
                    )
                    n_kept_expected += 1
            con.commit()
            con.close()

            out = os.path.join(d, "ds.npz")
            ds = build_dataset(db_path, out)

            self.assertTrue(os.path.exists(out))
            self.assertGreater(ds["crops"].shape[0], 0)
            self.assertEqual(ds["crops"].shape[1:], (64, 64))
            # Two distinct runs -> two group ids.
            self.assertEqual(len(np.unique(ds["run_id"])), 2)
            self.assertEqual(set(ds["run_key"]), {"run-0", "run-1"})
            self.assertEqual(set(ds["rig_id"]), {"rig-0", "rig-1"})
            self.assertTrue(np.all(ds["label_source"] == "sep_flux_radius"))
            self.assertTrue(np.all(np.char.str_len(ds["frame_checksum"]) == 64))
            # Labels should be finite, positive, and in a sane pixel range.
            self.assertTrue(np.all(ds["hfr_px"] > 0))
            self.assertTrue(np.all(np.isfinite(ds["log_hfr"])))
            # Crops are normalized to [0, 1].
            self.assertGreaterEqual(ds["crops"].min(), 0.0)
            self.assertLessEqual(ds["crops"].max(), 1.0)


class TestSaturationGate(unittest.TestCase):
    """The D1 regression: the builder must read `saturation_adu`, not `max_adu`.

    A 14-bit body in a 16-bit container clips near 16383. The old rule derived
    its threshold from the container (65535 * 0.98 = 64230), so no star was ever
    rejected and clipped cores — which measure artificially *small* HFR —
    labelled the brightest samples in the set.
    """

    def test_declared_saturation_is_used(self):
        q = '{"saturation_adu": 16383, "max_adu": 65535}'
        self.assertAlmostEqual(saturation_level_from_quality(q), 16383 * 0.98, places=3)

    def test_container_ceiling_alone_is_not_a_saturation_point(self):
        # 16-bit container, nothing declared -> unknowable, not 64230.
        self.assertIsNone(saturation_level_from_quality('{"max_adu": 65535}'))

    def test_eight_bit_container_is_its_own_clip_point(self):
        # Mirrors the 8-bit arm of resolve_saturation_adu in vision/quality.rs:
        # for a UVC/Mono8 stream the container really is where it clips.
        self.assertAlmostEqual(
            saturation_level_from_quality('{"max_adu": 255}'), 255 * 0.98, places=3
        )

    def test_explicit_gate_not_applied_overrides_any_value(self):
        q = '{"saturation_adu": 16383, "saturation_gate_applied": false}'
        self.assertIsNone(saturation_level_from_quality(q))

    def test_missing_and_malformed_are_unknown(self):
        for q in ("", "{}", "not json", '{"saturation_adu": null}',
                  '{"saturation_adu": 0}', '{"saturation_adu": "abc"}'):
            self.assertIsNone(saturation_level_from_quality(q), q)

    def test_builder_drops_frames_with_unknown_saturation(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "achernar.db")
            frames_dir = os.path.join(d, "frames")
            os.makedirs(frames_dir)
            con = _make_schema(db_path)
            # One run whose frames carry only a container ceiling.
            con.execute(
                "INSERT INTO focus_runs (id, rig_id, filter_name, started_at) "
                "VALUES ('run-x', 'rig-x', 'L', '2026-07-20T01:00:00Z')"
            )
            for i, hfr in enumerate([2.0, 3.5, 5.0]):
                fid = f"frame-x-{i}"
                png = os.path.join(frames_dir, f"{fid}.png")
                _write_frame_png(png, hfr, rng)
                con.execute(
                    "INSERT INTO frames (id, local_path, checksum_sha256, quality_json, "
                    "filter_name, exposure_s, gain, captured_at) VALUES (?,?,?,?,?,?,?,?)",
                    (fid, png, hashlib.sha256(Path(png).read_bytes()).hexdigest(),
                     '{"max_adu": 65535}', "L", 1.0, 100.0,
                     f"2026-07-20T01:0{i}:00Z"),
                )
                con.execute(
                    "INSERT INTO focus_samples (run_id, position, hfr, frame_id) "
                    "VALUES (?,?,?,?)", ("run-x", 10000 + i * 250, hfr, fid),
                )
            con.commit()
            con.close()

            out = os.path.join(d, "ds.npz")
            # Strict mode: every frame is ungatable, so nothing survives.
            with self.assertRaises(RuntimeError) as ctx:
                build_dataset(db_path, out)
            self.assertIn("saturation", str(ctx.exception).lower())

            # Permissive mode still builds, but stamps the manifest.
            ds = build_dataset(db_path, out, allow_missing_saturation=True)
            self.assertGreater(ds["crops"].shape[0], 0)
            manifest = json.loads(
                Path(out).with_suffix(".manifest.json").read_text()
            )
            self.assertEqual(manifest["saturation_gate"], "permissive")


if __name__ == "__main__":
    unittest.main(verbosity=2)
