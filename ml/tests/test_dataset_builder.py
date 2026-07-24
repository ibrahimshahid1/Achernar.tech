"""
Validate the FQE dataset builder against a synthetic SQLite + PNG fixture.

This exercises the exact join/detect/label/normalize path the real builder uses,
without needing real autofocus data — it constructs the same `frames` +
`focus_samples` schema the daemon writes, drops synthetic star PNGs on disk, and
asserts the builder recovers labeled crops grouped by run.
"""
import os
import hashlib
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.build_focus_quality_dataset import build_dataset  # noqa: E402
from data_pipeline.synthetic_psf import render_star  # noqa: E402


def _write_frame_png(path, hfr, rng):
    # A frame with a single centered star; the builder detects + crops it.
    img = render_star(hfr, size=96, amp=12000.0, sky=200.0, rng=rng).astype(np.uint16)
    Image.fromarray(img, mode="I;16").save(path)


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
                            f'{{"max_adu": 65535, "camera_id": "camera-{run}"}}',
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
