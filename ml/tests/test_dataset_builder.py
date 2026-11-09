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
    DEFAULT_OUTPUT,
    LABEL_VERSION,
    LEGACY_UNVERIFIED,
    PERMISSIVE_SATURATION,
    QUALIFIED,
    LegacyUnverifiedLabels,
    build_dataset,
    label_provenance_status,
    load_qualified_dataset,
    qualified_row_mask,
    saturation_decision_from_quality,
    saturation_level_from_quality,
)
from labels.psf_measure import SepBackgroundParams, SepDetectionParams, require_sep  # noqa: E402


class SepRequired(unittest.TestCase):
    """Builder tests run the SEP labeler: without sep==1.4.1 they fail with
    "missing prerequisite: sep==1.4.1" instead of skipping."""

    @classmethod
    def setUpClass(cls):
        require_sep()
from data_pipeline.synthetic_psf import render_star  # noqa: E402


def _write_frame_png(path, hfr, rng, size=192):
    # A frame with a single centered star; the builder detects + crops it.
    # 192 px holds a 3x3 grid of the labeler's 64-px SEP meshes. The former
    # 96-px fixture holds fewer than the required 2 meshes per axis, so its
    # "mesh" background was one global estimate in disguise and is now
    # reported unavailable (test_psf_measure_spatial: frame_smaller_than_mesh_grid).
    img = render_star(hfr, size=size, amp=12000.0, sky=200.0, rng=rng).astype(np.uint16)
    Image.fromarray(img).save(path)


FROZEN_V1 = Path(__file__).resolve().parent.parent / "data_pipeline" / "focus_quality_synth_v1.npz"


def _fixture_db(d, *, n_runs=2, size=192, rng=None):
    """A daemon-shaped DB with declared-saturation frames; returns the DB path."""
    rng = rng or np.random.default_rng(0)
    db_path = os.path.join(d, "achernar.db")
    con = _make_schema(db_path)
    for run in range(n_runs):
        con.execute(
            "INSERT INTO focus_runs (id, rig_id, filter_name, started_at) VALUES (?, ?, ?, ?)",
            (f"run-{run}", f"rig-{run}", "L", f"2026-07-{20 + run:02d}T01:00:00Z"),
        )
        for i, hfr in enumerate([2.0, 3.5, 5.0]):
            fid = f"frame-{run}-{i}"
            png = os.path.join(d, f"{fid}.png")
            _write_frame_png(png, hfr, rng, size=size)
            con.execute(
                "INSERT INTO frames (id, local_path, checksum_sha256, quality_json) VALUES (?, ?, ?, ?)",
                (fid, png, hashlib.sha256(Path(png).read_bytes()).hexdigest(),
                 '{"max_adu": 65535, "saturation_adu": 16383, "saturation_gate_applied": true}'),
            )
            con.execute(
                "INSERT INTO focus_samples (run_id, position, hfr, frame_id) VALUES (?, ?, ?, ?)",
                (f"run-{run}", 10000 + i * 250, hfr, fid),
            )
    con.commit()
    con.close()
    return db_path


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


class TestDatasetBuilder(SepRequired):
    def test_acquisition_ineligibility_cannot_be_overridden_by_permissive_saturation(self):
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as d:
            db_path = str(Path(d) / "achernar.db")
            con = _make_schema(db_path)
            con.execute("INSERT INTO focus_runs (id, rig_id) VALUES ('run', 'rig')")
            for i, markers in enumerate((
                {"training_label_eligible": False},
                {"acquisition_domain": "simulation"},
                {"acquisition_domain": "preview"},
                {},
            )):
                path = Path(d) / f"frame-{i}.png"
                _write_frame_png(path, 3.0, rng)
                quality = {"saturation_adu": 16383, "saturation_gate_applied": True, **markers}
                con.execute(
                    "INSERT INTO frames (id, local_path, checksum_sha256, quality_json) VALUES (?, ?, ?, ?)",
                    (f"frame-{i}", str(path), hashlib.sha256(path.read_bytes()).hexdigest(), json.dumps(quality)),
                )
                con.execute(
                    "INSERT INTO focus_samples (run_id, position, hfr, frame_id) VALUES ('run', ?, 3.0, ?)",
                    (1000 + i * 100, f"frame-{i}"),
                )
            con.commit()
            con.close()

            for permissive in (False, True):
                output = str(Path(d) / f"dataset-{permissive}.npz")
                dataset = build_dataset(db_path, output, allow_missing_saturation=permissive)
                self.assertEqual(set(dataset["frame_id"]), {"frame-3"})
                manifest = json.loads(Path(output).with_suffix(".manifest.json").read_text())
                self.assertEqual(manifest["frames_skipped_ineligible"], 3)

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


class TestLabelProvenance(SepRequired):
    """BGE-20: new labels carry version + SEP config provenance; frozen
    datasets are never overwritten; legacy rows never enter a qualified
    evaluation."""

    def test_manifest_and_rows_carry_sep_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = _fixture_db(d)
            out = os.path.join(d, "v2", "ds.npz")
            ds = build_dataset(db_path, out)
            manifest = json.loads(Path(out).with_suffix(".manifest.json").read_text())
            self.assertEqual(manifest["label_version"], LABEL_VERSION)
            self.assertEqual(manifest["sep_version"], "1.4.1")
            self.assertEqual(manifest["background_config"], SepBackgroundParams().config())
            self.assertEqual(manifest["background_config_sha256"], SepBackgroundParams().config_sha256())
            self.assertEqual(manifest["detection_config"], SepDetectionParams().config())
            self.assertEqual(manifest["detection_config_sha256"], SepDetectionParams().config_sha256())
            self.assertEqual(manifest["noise_source"], "empirical_mesh_rms")
            self.assertEqual(set(manifest["width_definition"]), {"hfr", "hfd", "fwhm"})
            self.assertEqual(manifest["frames_by_saturation_source"], {"declared_sensor_saturation": 6})
            self.assertEqual(manifest["frames_by_saturation_gate"], {"applied": 6})
            self.assertIsInstance(manifest["rejection_counts"], dict)
            self.assertEqual(manifest["frames_background_unavailable"], {})
            n = len(ds["hfr_px"])
            self.assertTrue(np.all(ds["label_version"] == LABEL_VERSION))
            self.assertTrue(np.all(ds["background_config_sha256"] == manifest["background_config_sha256"]))
            self.assertTrue(np.all(ds["detection_config_sha256"] == manifest["detection_config_sha256"]))
            self.assertTrue(np.all(ds["saturation_source"] == "declared_sensor_saturation"))
            self.assertEqual(ds["local_rms_adu"].shape, (n,))
            self.assertTrue(np.all(ds["local_rms_adu"] > 0))
            self.assertEqual(label_provenance_status(manifest), QUALIFIED)
            loaded, report = load_qualified_dataset(out)
            self.assertEqual(report["rows_qualified"], n)
            self.assertEqual(len(loaded["hfr_px"]), n)

    def test_existing_dataset_and_manifest_are_never_overwritten(self):
        self.assertNotEqual(Path(DEFAULT_OUTPUT).name, "focus_quality_ds_v1.npz")
        with tempfile.TemporaryDirectory() as d:
            db_path = _fixture_db(d, n_runs=1)
            # A frozen legacy artifact (a copy of the repository's frozen v1 set)
            # sitting where the caller points the builder.
            frozen = Path(d) / "frozen" / "ds.npz"
            frozen.parent.mkdir()
            frozen.write_bytes(FROZEN_V1.read_bytes())
            legacy_manifest = frozen.with_suffix(".manifest.json")
            legacy_manifest.write_text('{"label_method": "SEP mesh background", "saturation_gate": "enforced"}')
            before = (hashlib.sha256(frozen.read_bytes()).hexdigest(), legacy_manifest.read_text())
            for overwrite in (False, True):
                with self.assertRaises(FileExistsError):
                    build_dataset(db_path, str(frozen), overwrite=overwrite)
            self.assertEqual((hashlib.sha256(frozen.read_bytes()).hexdigest(), legacy_manifest.read_text()), before)

            # A manifest alone (dataset moved away) is protected too.
            lonely = Path(d) / "lonely.npz"
            lonely.with_suffix(".manifest.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                build_dataset(db_path, str(lonely))
            self.assertFalse(lonely.exists())

            # Rebuilding a v2 dataset in place needs an explicit overwrite.
            fresh = Path(d) / "fresh.npz"
            build_dataset(db_path, str(fresh))
            with self.assertRaises(FileExistsError):
                build_dataset(db_path, str(fresh))
            build_dataset(db_path, str(fresh), overwrite=True)

    def test_legacy_rows_are_excluded_from_qualified_evaluation(self):
        # The repository's frozen synthetic set has no manifest: all legacy.
        with self.assertRaises(LegacyUnverifiedLabels):
            load_qualified_dataset(str(FROZEN_V1))
        self.assertEqual(label_provenance_status(None), LEGACY_UNVERIFIED)
        self.assertEqual(label_provenance_status({"label_method": "SEP mesh background"}), LEGACY_UNVERIFIED)

        with tempfile.TemporaryDirectory() as d:
            db_path = _fixture_db(d, n_runs=1)
            out = Path(d) / "ds.npz"
            ds = build_dataset(db_path, str(out))
            manifest = json.loads(out.with_suffix(".manifest.json").read_text())
            n = len(ds["hfr_px"])

            # Mixed rows: append rows lacking the provenance columns' values.
            mixed = {k: np.concatenate([v, v]) for k, v in ds.items()}
            mixed["label_version"][n:] = ""
            mixed["background_config_sha256"][n:] = ""
            mask = qualified_row_mask(mixed, manifest)
            self.assertEqual(int(mask.sum()), n)
            self.assertFalse(mask[n:].any())
            mixed_path = Path(d) / "mixed.npz"
            np.savez_compressed(mixed_path, **mixed)
            mixed_path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
            loaded, report = load_qualified_dataset(str(mixed_path))
            self.assertEqual((report["rows_total"], report["rows_qualified"], report["rows_legacy_unverified"]),
                             (2 * n, n, n))
            self.assertEqual(len(loaded["hfr_px"]), n)
            self.assertTrue(np.all(loaded["label_version"] == LABEL_VERSION))

            # A manifest whose config no longer matches its recorded hash is not trusted.
            tampered = dict(manifest, detection_config=dict(manifest["detection_config"], thresh_sigma=3.0))
            self.assertEqual(label_provenance_status(tampered), LEGACY_UNVERIFIED)
            self.assertFalse(qualified_row_mask(ds, tampered).any())
            # A permissive (ungated) build is never qualified either.
            self.assertEqual(label_provenance_status(dict(manifest, saturation_gate="permissive")),
                             PERMISSIVE_SATURATION)

    def test_frames_below_the_mesh_grid_are_counted_not_labelled(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = _fixture_db(d, n_runs=1, size=96)
            with self.assertRaises(RuntimeError) as ctx:
                build_dataset(db_path, os.path.join(d, "ds.npz"))
            self.assertIn("frame_smaller_than_mesh_grid", str(ctx.exception))


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
        # Decision D-011: for a UVC/Mono8 stream 255 is the real ceiling of the
        # recorded data. Its provenance is the container, not the sensor.
        self.assertAlmostEqual(
            saturation_level_from_quality('{"max_adu": 255}'), 255 * 0.98, places=3
        )
        threshold, source = saturation_decision_from_quality('{"max_adu": 255}')
        self.assertEqual(source, "container_ceiling_8bit")
        threshold, source = saturation_decision_from_quality('{"saturation_adu": 16383, "max_adu": 255}')
        self.assertEqual((threshold, source), (16383 * 0.98, "declared_sensor_saturation"))
        self.assertEqual(saturation_decision_from_quality('{"max_adu": 65535}'), (None, None))

    def test_explicit_gate_not_applied_overrides_any_value(self):
        q = '{"saturation_adu": 16383, "saturation_gate_applied": false}'
        self.assertIsNone(saturation_level_from_quality(q))

    def test_missing_and_malformed_are_unknown(self):
        for q in ("", "{}", "not json", '{"saturation_adu": null}',
                  '{"saturation_adu": 0}', '{"saturation_adu": "abc"}'):
            self.assertIsNone(saturation_level_from_quality(q), q)

    def test_builder_drops_frames_with_unknown_saturation(self):
        require_sep()
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
