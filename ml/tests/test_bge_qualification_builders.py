"""Manifest v2, sidecar log, split rule, manifest builders and the Rust bridge plumbing
of ml/eval/bge/qualification. Synthetic data generated in tmp dirs only.

Run: ``python -m pytest -p no:cacheprovider -q ml/tests/test_bge_qualification_builders.py``
"""

import datetime as dt
import json
import sqlite3

import numpy as np
import pytest

from ml.eval.bge.qualification import DAEMON_ANALYSIS_REPRESENTATION as REP
from ml.eval.bge.qualification import builders as B
from ml.eval.bge.qualification import manifest as MF
from ml.eval.bge.qualification import methods as M
from ml.eval.bge.qualification import split as SPLIT
from ml.eval.bge.qualification import testing as T
from ml.eval.bge.qualification.sidecar import SidecarError, SidecarLog

UTC = dt.timezone.utc


# --------------------------------------------------------------------------
# Split rule
# --------------------------------------------------------------------------


def _frames_for_nights():
    # Nights given out of order; darks shot in the afternoon still belong to their night.
    spec = [("2026-11-20", "2026-11-21T01:00:00+00:00"), ("2026-11-14", "2026-11-15T02:00:00+00:00"),
            ("2026-11-17", "2026-11-18T03:00:00+00:00"), ("2026-11-15", "2026-11-16T04:00:00+00:00"),
            ("2026-11-22", "2026-11-23T05:00:00+00:00")]
    frames = []
    for night, ts in spec:
        frames.append({"id": f"{night}-light", "night_id": night, "timestamp_utc": ts, "frame_type": "light"})
        frames.append({"id": f"{night}-dark", "night_id": night, "frame_type": "dark",
                       "timestamp_utc": ts.replace("T0", "T2").replace("+00:00", "+00:00")})
    return frames


def test_split_rule_alternates_nights_chronologically_and_keeps_nights_whole():
    frames = _frames_for_nights()
    splits = SPLIT.assign_splits(frames)
    assert splits == {"2026-11-14": "development", "2026-11-15": "holdout", "2026-11-17": "development",
                      "2026-11-20": "holdout", "2026-11-22": "development"}
    # Deterministic whatever the frame order.
    assert SPLIT.assign_splits(list(reversed(frames))) == splits
    table = SPLIT.apply_rule(frames)
    assert [n["split"] for n in table] == ["development", "holdout", "development", "holdout", "development"]
    for f in frames:  # calibration frames carry their night's split
        assert f["split"] == splits[f["night_id"]]
    assert SPLIT.check_manifest(frames) == []
    frames[0]["split"] = "development"  # 2026-11-20 is hold-out by the rule: one frame flipped
    problems = SPLIT.check_manifest(frames)
    assert any("more than one split" in p for p in problems)


def test_split_rule_orders_untimestamped_nights_last():
    frames = [{"id": "a", "night_id": "2026-11-30"}, {"id": "b", "night_id": "2026-11-01",
                                                     "timestamp_utc": "2026-11-02T01:00:00Z"},
              {"id": "c", "night_id": "2026-11-05"}]
    assert [n for n, _ in SPLIT.night_order(frames)] == ["2026-11-01", "2026-11-05", "2026-11-30"]
    with pytest.raises(SPLIT.SplitRuleError):
        SPLIT.night_order([{"id": "x"}])


# --------------------------------------------------------------------------
# Sidecar log
# --------------------------------------------------------------------------


def test_sidecar_csv_jsonl_windows_precedence_and_strictness(tmp_path):
    csv_path = tmp_path / "log.csv"
    csv_path.write_text(
        "# field log\n"
        "file,t_start,t_end,frame_type,optical_train,repeat_group,iso,sweep_direction\n"
        ",2026-11-15T02:00:00Z,2026-11-15T03:00:00Z,,barlow2x,,,\n"
        "a.fits,,,light,,rg1,1600,out\n")
    log = SidecarLog.read(csv_path)
    tags = log.tags_for(file="sub/a.fits", timestamp="2026-11-15T02:30:00+00:00")
    assert tags["optical_train"] == "barlow2x" and tags["repeat_group"] == "rg1" and tags["iso"] == 1600.0
    assert tags["sweep_direction"] == "increasing"
    assert log.tags_for(file="b.fits", timestamp="2026-11-15T04:00:00+00:00") == {}
    jl = tmp_path / "log.jsonl"
    jl.write_text('{"defaults": {"rig_id": "z114", "camera": "Canon EOS R50"}}\n'
                  '{"frame_id": "F1", "frame_type": "dark"}\n{"file": "unused.fits", "notes": "typo?"}\n')
    log2 = SidecarLog.read(jl)
    assert log2.tags_for(frame_id="F1") == {"rig_id": "z114", "camera": "Canon EOS R50", "frame_type": "dark"}
    assert log2.unused_rows() == [{"file": "unused.fits", "notes": "typo?"}]
    bad = tmp_path / "bad.csv"
    bad.write_text("file,repeat_grp\na.fits,1\n")
    with pytest.raises(SidecarError, match="unknown column"):
        SidecarLog.read(bad)
    bad.write_text("file,timestamp_utc\na.fits,2026-11-15T02:00:00\n")
    with pytest.raises(SidecarError, match="UTC offset"):
        SidecarLog.read(bad)
    bad.write_text("file,frame_id\na.fits,F1\n")
    with pytest.raises(SidecarError, match="exactly one selector"):
        SidecarLog.read(bad)


# --------------------------------------------------------------------------
# (a) Directory + sidecar log
# --------------------------------------------------------------------------


def test_build_from_directory_bundles_fits_raw_and_verified_loading(tmp_path):
    frames_dir = tmp_path / "frames"
    img, *_ = T.sky_frame(1)
    bad = np.zeros(img.shape, np.uint8)
    bad[5, 5] = 1
    capture = {"frame_id": "frame-1", "capture_id": "cap-1", "capture_type": "focus", "rig_id": "z114",
               "camera_name": "Canon EOS R50", "iso": 1600.0, "applied_exposure_s": 2.0,
               "exposure_started_at": "2026-11-15T02:10:00+00:00"}
    T.write_bundle(frames_dir / "bundleA", img, {"bad_pixel": bad}, capture)
    T.write_bundle(frames_dir / "bundleB", img + 1, None, dict(capture, frame_id="frame-2", capture_id="cap-2",
                                                               capture_type="light",
                                                               exposure_started_at="2026-11-16T02:10:00+00:00"))
    T.write_fits(frames_dir / "plain.fits", img, header={"DATE-OBS": "2026-11-17T03:00:00"})
    T.write_fits(frames_dir / "plain_bad.fits", img)
    T.write_fits(frames_dir / "undeclared.fits", img)
    (frames_dir / "IMG_0001.CR3").write_bytes(b"not a real raw")
    sidecar = tmp_path / "log.csv"
    sidecar.write_text(
        "file,frame_type,analysis_representation,repeat_group,optical_train,focus_position,focuser_stepsize_um,"
        "focal_ratio,mask_bad_pixel\n"
        f"bundleA,,,,native,1200,2.0,3.95,\n"
        f"plain.fits,dark,{REP},,,,,,plain_bad.fits\n"
        f"bundleB,,,rg1,barlow2x,,,,\n")
    m = B.build_from_directory(frames_dir, SidecarLog.read(sidecar))
    by_rel = {f["provenance"]["relative_path"]: f for f in m["frames"]}
    assert set(by_rel) == {"bundleA", "bundleB", "plain.fits"}
    a = by_rel["bundleA"]
    assert a["format"] == "daemon_bundle" and a["frame_type"] == "light"  # capture_type focus -> light
    assert a["analysis_input"]["representation"] == REP and a["focus_position"] == 1200.0
    assert a["timestamp_utc"].startswith("2026-11-15T02:10") and a["rig_id"] == "z114"
    assert set(a["masks"]) == {"bad_pixel", "invalid", "edge", "native_saturation"}
    assert a["saturation_adu"] == 16383.0
    assert by_rel["bundleB"]["repeat_group"] == "rg1" and by_rel["bundleB"]["optical_train"] == "barlow2x"
    p = by_rel["plain.fits"]
    assert p["frame_type"] == "dark" and "timestamp_from_fits_date_obs" in p["flags"]
    assert p["masks"]["bad_pixel"]["format"] == "fits"
    reasons = {e["source"]: e["reason"] for e in m["excluded"]}
    assert reasons["IMG_0001.CR3"].startswith("raw_without_daemon_analysis_bundle")
    assert "analysis_representation undeclared" in reasons["undeclared.fits"]
    assert "plain_bad.fits" not in reasons  # a mask file, not a frame
    assert [n["split"] for n in m["nights"]] == ["development", "holdout", "development"]
    # Loading verifies sha256 and returns 0/1 masks.
    image, masks = MF.load_frame(a, tmp_path)
    assert image.dtype == np.float32 and np.allclose(image, img) and masks["bad_pixel"][5, 5] == 1
    path = tmp_path / "m.json"
    B.write_manifest(m, path)
    assert MF.load_manifest(path)["schema"] == "bge-real-frames-v2"
    T.write_fits(frames_dir / "bundleA" / "analysis" / "linear.fits", img + 5)
    with pytest.raises(MF.ChecksumMismatch):
        MF.load_frame(a, tmp_path)


def test_manifest_validation_and_v1_upgrade(tmp_path):
    frame = {"id": "x", "path": "x.fits", "sha256": "0" * 64, "format": "fits", "frame_type": "light",
             "night_id": "n1", "split": "development", "analysis_input": {"representation": REP}}
    MF.validate_manifest({"schema": "bge-real-frames-v2", "frames": [frame]})
    with pytest.raises(MF.ManifestError, match="never split"):
        MF.validate_manifest({"schema": "bge-real-frames-v2",
                              "frames": [frame, dict(frame, id="y", split="holdout")]})
    with pytest.raises(MF.ManifestError, match="unknown fields"):
        MF.validate_manifest({"schema": "bge-real-frames-v2", "frames": [dict(frame, colour="red")]})
    v1 = {"schema": "bge-real-frames-v1", "frames": [{"id": "a", "path": "a.f32", "format": "f32", "width": 4,
                                                      "height": 4, "split": "development", "group": "g"}]}
    (tmp_path / "v1.json").write_text(json.dumps(v1))
    up = MF.load_manifest(tmp_path / "v1.json")
    assert up["upgraded_from"] == "bge-real-frames-v1" and up["frames"][0]["repeat_group"] == "g"
    assert up["frames"][0]["night_id"] == "unknown-night"


# --------------------------------------------------------------------------
# (b) import_session DB (rows written by tools/import_session.py itself)
# --------------------------------------------------------------------------


def _png16(path, image):
    from PIL import Image

    Image.fromarray(np.clip(image, 0, 65535).astype(np.uint16)).save(path)


def test_build_from_import_session_db(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    imp = MF.import_session_module()
    db = tmp_path / "achernar.db"
    T.migrate(db).close()
    lights = tmp_path / "lights"
    lights.mkdir()
    positions = [900, 950, 1000, 1050, 1100]
    for i, p in enumerate(positions):
        img, *_ = T.sky_frame(10 + i, psf_sigma=T.focus_hfr_sigma(p), n_stars=8, flux=(2e4, 1e5))
        _png16(lights / f"sweep_{i}.png", img)
    img, *_ = T.sky_frame(30)
    _png16(lights / "IMG_0100.png", img)
    (lights / "IMG_0200.CR3").write_bytes(b"raw only")
    log = {"session": {"rig_id": "z114", "camera": "Canon EOS R50", "captured_at": "2026-11-14T21:30:00-05:00",
                       "exposure_s": 2.0, "iso_or_gain": 1600},
           "runs": [{"run_id": "run-1", "positions": [{"position": p, "files": [f"sweep_{i}.png"]}
                                                      for i, p in enumerate(positions)]}],
           "environment": [{"t": "2026-11-14T21:35:00-05:00", "ambient_temp_c": 4.5}]}
    (tmp_path / "session.json").write_text(json.dumps(log))
    capture_root = tmp_path / "captures"
    stats = imp.import_session(db, lights, capture_root, tmp_path / "session.json", rig_id="z114")
    assert stats["frames_inserted"] == 7 and stats["focus_samples_inserted"] >= 4
    darks = tmp_path / "darks"
    darks.mkdir()
    _png16(darks / "dark_1.png", T.dark_frame(1) + 1000)
    imp.import_session(db, darks, capture_root, tmp_path / "session.json", capture_type="dark", rig_id="z114")
    side = tmp_path / "side.csv"
    side.write_text("file,repeat_group,optical_train,focal_ratio\nIMG_0100.png,rg1,native,3.95\n")
    m = B.build_from_import_session_db(db, capture_root, SidecarLog.read(side))
    assert m["builder"]["read_only"] and m["builder"]["origins"] == ["import_session"]
    types = sorted(f["frame_type"] for f in m["frames"])
    assert types.count("dark") == 1 and types.count("light") == 6
    sweep = [f for f in m["frames"] if f.get("focus_run_id") == "run-1"]
    assert len(sweep) == stats["focus_samples_inserted"]
    assert all(f["format"] == "import_luma" and f["focuser_stepsize_um"] == 1.0 for f in sweep)
    assert {f["sweep_direction"] for f in sweep} == {"increasing"}
    assert all(f["analysis_input"]["representation"] == B.LUMA_REPRESENTATION for f in m["frames"])
    single = next(f for f in m["frames"] if f.get("repeat_group"))
    assert single["optical_train"] == "native" and single["focal_ratio"] == 3.95
    assert single["night_id"] == "2026-11-14" and single["ambient_temp_c"] == 4.5
    assert any("raw_without_daemon_analysis_bundle" in e["reason"] for e in m["excluded"])
    image, masks = MF.load_frame(sweep[0], tmp_path)  # read with import_session.load_luma_with_container
    assert image.shape == (192, 256) and image.dtype == np.float32
    # The DB was opened read-only: a write through the builder's connection would fail.
    with pytest.raises(sqlite3.OperationalError):
        B._ro_connect(db).execute("DELETE FROM frames")


# --------------------------------------------------------------------------
# (c) Daemon DB + capture archive
# --------------------------------------------------------------------------


def _daemon_frame(con, root, fid, ctype, started, img, rig="z114", night="2026-11-14", acquisition=None,
                  position=None, cap=None):
    rel = f"{rig}/{night}/no-session/{fid}"
    bundle = T.write_bundle(root / rel, img)
    con.execute("INSERT INTO frames (id, local_path, captured_at, capture_id, rig_id, camera_name, capture_type, "
                "capture_status, exposure_started_at, exposure_s, iso, focuser_position, night_id, acquisition_json, "
                "driver_name, temperature, temperature_state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (fid, f"{rel}/analysis/luma.png", started, cap or f"cap-{fid}", rig, "Canon EOS R50", ctype,
                 "committed", started, 2.0, 1600.0, position, night, json.dumps(acquisition or {}), "gphoto2", 9.5,
                 "measured"))
    for role, rpath in (("analysis_descriptor", "analysis/analysis.json"), ("analysis_fits", "analysis/linear.fits"),
                        ("native_raw", "native/IMG.CR3")):
        p = bundle / rpath
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"raw")
        con.execute("INSERT INTO frame_assets (id, frame_id, role, variant, original_filename, relative_path, "
                    "media_type, byte_length, checksum_sha256, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (f"{fid}-{role}", fid, role, 0, f"{fid}.CR3" if role == "native_raw" else None, f"{rel}/{rpath}",
                     "application/octet-stream", p.stat().st_size, T.sha(p), started))


def test_build_from_daemon_db_focus_runs_provenance_and_sidecar_tags(tmp_path):
    db = tmp_path / "achernar.db"
    con = T.migrate(db)
    root = tmp_path / "frames" / "captures"
    con.execute("INSERT INTO rigs (id, name, focal_ratio) VALUES ('z114', 'Z114', 3.95)")
    con.execute("INSERT INTO focuser_profiles (id, rig_id, focuser_name, steps_per_micron, scale_measured_at) "
                "VALUES ('fp', 'z114', 'esp32', 0.5, '2026-11-10T00:00:00Z')")
    con.execute("INSERT INTO focus_runs (id, rig_id, start_position, state, started_at, night_id) "
                "VALUES ('run-A', 'z114', 900, 'Converged', '2026-11-15T02:00:00+00:00', '2026-11-14')")
    t0 = dt.datetime(2026, 11, 15, 2, 0, tzinfo=UTC)
    for i, p in enumerate((1100, 1050, 1000, 950)):
        fid = f"af-{i}"
        ts = T.iso(t0 + dt.timedelta(seconds=10 * i))
        img, *_ = T.sky_frame(40 + i)
        _daemon_frame(con, root, fid, "focus", ts, img, position=p)
        con.execute("INSERT INTO focus_samples (run_id, position, hfr, star_count, captured_at, frame_id, "
                    "position_readback_before, position_readback_after, moved_during_exposure, capture_id) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    ("run-A", p, 2.0, 10, ts, fid, p, p if i != 2 else p + 3, 1 if i == 3 else 0, f"cap-{fid}"))
    img, *_ = T.sky_frame(50)
    _daemon_frame(con, root, "lt-1", "light", T.iso(t0 + dt.timedelta(minutes=30)), img)
    _daemon_frame(con, root, "dk-1", "light", T.iso(t0 + dt.timedelta(minutes=40)), T.dark_frame(2))
    _daemon_frame(con, root, "sim-1", "focus", T.iso(t0), img, acquisition={"acquisition_domain": "simulation"})
    con.execute("INSERT INTO frames (id, local_path, captured_at, capture_type, capture_status, rig_id, night_id) "
                "VALUES ('raw-only', 'x', ?, 'light', 'committed', 'z114', '2026-11-14')", (T.iso(t0),))
    con.execute("INSERT INTO frame_assets (id, frame_id, role, relative_path, media_type, byte_length, "
                "checksum_sha256, created_at) VALUES ('ra', 'raw-only', 'native_raw', 'x.CR3', 'image/x-raw', 1, "
                "'0', 'now')")
    con.commit()
    con.close()
    side = tmp_path / "side.jsonl"
    side.write_text('{"defaults": {"site_utc_offset": "-05:00"}}\n'
                    '{"t_start": "2026-11-15T01:00:00Z", "t_end": "2026-11-15T04:00:00Z", "optical_train": "native"}\n'
                    '{"frame_id": "lt-1", "repeat_group": "rg1", "iso": 800}\n'
                    '{"frame_id": "dk-1", "frame_type": "dark"}\n')
    m = B.build_from_daemon_db(db, root, SidecarLog.read(side))
    by_id = {f["provenance"]["frame_id"]: f for f in m["frames"]}
    assert set(by_id) == {"af-0", "af-1", "af-2", "af-3", "lt-1", "dk-1"}
    af = by_id["af-0"]
    assert af["focus_run_id"] == "run-A" and af["focus_position"] == 1100.0 and af["sweep_direction"] == "decreasing"
    assert af["focuser_stepsize_um"] == 2.0 and af["focal_ratio"] == 3.95 and af["optical_train"] == "native"
    assert af["format"] == "daemon_bundle" and af["analysis_input"]["representation"] == REP
    assert af["sensor_temp_c"] == 9.5
    assert "moved_during_exposure" in by_id["af-3"]["flags"]
    assert "position_readback_mismatch" in by_id["af-2"]["flags"]
    lt = by_id["lt-1"]
    assert lt["repeat_group"] == "rg1" and lt["iso"] == 1600.0  # DB wins over the sidecar for ISO
    assert lt["provenance"]["sidecar_conflicts"][0]["field"] == "iso"
    assert by_id["dk-1"]["frame_type"] == "dark" and "frame_type_source" in by_id["dk-1"]["provenance"]
    reasons = [e["reason"] for e in m["excluded"]]
    assert any("acquisition_domain simulation" in r for r in reasons)
    assert any("raw_without_daemon_analysis_bundle" in r for r in reasons)
    assert m["builder"]["fields_missing_from_db_and_sidecar"]["repeat_group"] >= 1
    image, masks = MF.load_frame(af, tmp_path)
    assert image.shape == (192, 256) and set(masks) == {"bad_pixel", "invalid", "edge", "native_saturation"}


# --------------------------------------------------------------------------
# Rust variants through a mocked bridge
# --------------------------------------------------------------------------


def _mock_bridge(echo=True, status="completed", gate_echo=True):
    calls = []

    def run_bridge(manifest_path, out_dir, repo_root, cargo_target_dir, log_path, jobs=4):
        man = json.loads(manifest_path.read_text())
        calls.append(man)
        if status != "completed":
            return {"status": status, "reason": f"{status}: rust bridge unavailable (mock)"}
        out_dir.mkdir(parents=True, exist_ok=True)
        backend = man["backends"][0]
        for case in man["cases"]:
            h, w = case["height"], case["width"]
            image = np.fromfile(manifest_path.parent / case["pixels"], dtype="<f4").reshape(h, w)
            level = float(np.median(image))
            spatial = backend == "sep"
            cfg = {"method": "sep_spatial_mesh" if spatial else "legacy_global_clip"}
            if echo and spatial:
                cfg["noise_policy"] = man["settings"]["sep"]["noise_policy"]
            quality = {"status": "ok", "hfr_px": 2.0, "used_stars": [{"x": 10.0, "y": 12.0, "hfr_px": 2.0}]}
            if gate_echo:
                quality["star_gates_config"] = {"blend_test": man["star_gates"]["blend_test"]}
            payload = {"case": case["id"], "backend": backend, "error": None,
                       "background": {"method": cfg["method"], "level_status": "measured", "noise_status": "measured",
                                      "level": level, "rms": 10.0, "config": cfg, "config_sha256": "h"},
                       "candidates": [{"x": 10.0, "y": 12.0, "xmin": 8, "xmax": 12, "ymin": 10, "ymax": 14, "npix": 9,
                                       "flux": 100.0, "peak": 50.0, "deblended": False}],
                       "catalog": None, "quality": quality}
            stem = out_dir / f"{case['id']}__{backend}"
            stem.with_suffix(".json").write_text(json.dumps(payload))
            if spatial:
                np.full((h, w), level, "<f4").tofile(str(stem) + ".back.f32")
                np.full((h, w), 10.0, "<f4").tofile(str(stem) + ".rms.f32")
        return {"status": "completed", "reason": None, "revision": {"commit": "mock"}}

    return run_bridge, calls


def test_rust_runner_one_manifest_per_variant_with_overlays_and_echo_checks(tmp_path):
    run_bridge, calls = _mock_bridge()
    runner = M.RustBridgeRunner(tmp_path, tmp_path / "io", tmp_path / "target", run_bridge=run_bridge)
    img, *_ = T.sky_frame(3)
    case = M.Case("qabc_o", img, {"bad_pixel": np.zeros(img.shape, np.uint8)})
    ids = ["rust_legacy__g1", "rust_legacy__g2", "rust_sep__n1g1", "rust_sep__n2g2"]
    out = runner.run([case], ids)
    assert [c["backends"] for c in calls] == [["legacy_global_clip"], ["legacy_global_clip"], ["sep"], ["sep"]]
    assert calls[3]["settings"] == {"sep": {"noise_policy": "masked_local_v2"}}
    assert calls[3]["star_gates"] == {"blend_test": "dip_annulus_v2", "regroup_deblended": True}
    assert calls[1]["star_gates"] == {"blend_test": "dip_annulus_v2"} and calls[0]["settings"] == {}
    assert "masks" in calls[0]["cases"][0] and calls[0]["cases"][0]["masks"] == {}  # all-zero masks not written
    sep2 = out[("qabc_o", "rust_sep__n2g2")]
    assert sep2.usable and isinstance(sep2.back, np.memmap) and sep2.frame_hfr == 2.0
    assert sep2.detections[0]["npix"] == 9 and sep2.accepted[0]["hfr_px"] == 2.0
    legacy = out[("qabc_o", "rust_legacy__g1")]
    assert legacy.usable and np.isscalar(legacy.back) and legacy.rms == 10.0
    runner.cleanup()
    assert not any((tmp_path / "io").iterdir())


def test_rust_variant_without_echo_is_unavailable_not_default(tmp_path):
    run_bridge, _ = _mock_bridge(echo=False, gate_echo=False)
    runner = M.RustBridgeRunner(tmp_path, tmp_path / "io", tmp_path / "t", run_bridge=run_bridge)
    img, *_ = T.sky_frame(4)
    out = runner.run([M.Case("qx_o", img, {})], ["rust_sep__n1g1", "rust_sep__n2g1", "rust_legacy__g2"])
    assert out[("qx_o", "rust_sep__n1g1")].usable  # baseline: the pre-v2 daemon is that behaviour
    assert out[("qx_o", "rust_sep__n2g1")].status == "unavailable"
    assert "noise_policy" in out[("qx_o", "rust_sep__n2g1")].reason
    assert out[("qx_o", "rust_legacy__g2")].status == "unavailable"
    skipped, _ = _mock_bridge(status="skipped")
    out = M.RustBridgeRunner(tmp_path, tmp_path / "io2", tmp_path / "t", run_bridge=skipped).run(
        [M.Case("qy_o", img, {})], ["rust_sep__n2g2"])
    assert out[("qy_o", "rust_sep__n2g2")].status == "unavailable"
    assert out[("qy_o", "rust_sep__n2g2")].reason.startswith("method_unavailable")


def test_star_gate_records_match_spec_and_hash_echo():
    rec = M.star_gates_record("dip_annulus_v2", True)
    assert rec["annulus"]["min_coverage"] == 0.625 and rec["dip_fraction"] == 0.15
    ok, _ = M.variant_echo("rust_sep__n2g2", {"background": {"config": {"noise_policy": "masked_local_v2"}},
                                              "quality": {"star_gates_config_sha256":
                                                          next(iter(M.expected_gate_hashes("dip_annulus_v2")))}})
    assert ok
    ok, why = M.variant_echo("rust_sep__n2g2", {"background": {"config": {"noise_policy": "frame_wide_v1"}}})
    assert not ok and "frame_wide_v1" in why


def test_python_variant_probe_uses_keyword_api_when_present(monkeypatch):
    from ml.eval.bge import python_methods as PM

    monkeypatch.delattr(PM, "PYTHON_SEP_VARIANTS", raising=False)
    assert M.MethodSuite(["python_sep__n2g2"]).availability()["python_sep__n2g2"].startswith("method_unavailable")
    seen = {}
    real_bg, real_det = PM.python_sep_background, PM.detect_and_measure

    def bg(image, masks, params=None, noise_policy="frame_wide_v1"):
        seen["noise_policy"] = noise_policy
        return real_bg(image, masks, params)

    def det(image, masks, est, det_params=None, policy=None, blend_test="peak_count_v1", regroup_deblended=False):
        seen.update(blend_test=blend_test, regroup=regroup_deblended)
        return real_det(image, masks, est, det_params, policy)

    monkeypatch.setattr(PM, "python_sep_background", bg)
    monkeypatch.setattr(PM, "detect_and_measure", det)
    img, *_ = T.sky_frame(5)
    out = M.run_python("python_sep__n2g2", M.Case("qp_o", img, {}))
    assert out.usable and seen == {"noise_policy": "masked_local_v2", "blend_test": "dip_annulus_v2", "regroup": True}


def test_python_variant_table_routes_through_run_python_method(monkeypatch):
    from ml.eval.bge import python_methods as PM

    calls = []

    def fake_run(method, scenario, bg_params=None, det_params=None):
        calls.append((method, scenario.metadata["saturation_adu"]))
        est = PM.python_sep_background(scenario.image, scenario.masks)
        return {"estimate": est, "measurement": PM.detect_and_measure(scenario.image, scenario.masks, est)}

    table = {"python_sep__n2g1": {"noise_policy": "masked_local_v2", "blend_test": "peak_count_v1",
                                  "regroup_deblended": False},
             "python_sep__n1g2": {"noise_policy": "frame_wide_v1", "blend_test": "peak_count_v1",
                                  "regroup_deblended": True}}
    monkeypatch.setattr(PM, "PYTHON_SEP_VARIANTS", table, raising=False)
    monkeypatch.setattr(PM, "run_python_method", fake_run)
    img, *_ = T.sky_frame(6)
    out = M.run_python("python_sep__n2g1", M.Case("qv_o", img, {}, saturation_adu=16383.0))
    assert out.usable and calls == [("python_sep__n2g1", 16383.0)]
    bad = M.run_python("python_sep__n1g2", M.Case("qv_o", img, {}))  # table disagrees with spec section 4
    assert bad.status == "unavailable" and "!= spec" in bad.reason


def test_python_v2_variants_of_the_harness_run_on_a_frame_when_present():
    from ml.eval.bge import python_methods as PM

    if not isinstance(getattr(PM, "PYTHON_SEP_VARIANTS", None), dict):
        pytest.skip("python_methods has no PYTHON_SEP_VARIANTS at this revision")
    img, *_ = T.sky_frame(7)
    for mid in ("python_sep__n1g1", "python_sep__n2g2"):
        out = M.run_python(mid, M.Case("qr_o", img, {}))
        assert out.status == "completed" and out.usable, (mid, out.reason)
        assert out.accepted is not None and out.frame_hfr is not None


def test_new_bridge_echo_format_is_recognised():
    rec = M.star_gates_record("dip_annulus_v2", True)
    payload = {"background": {"config": {"noise_policy": "masked_local_v2"}}, "star_gates": rec,
               "star_gates_config_sha256": M.config_sha256(rec), "quality": {}}
    assert M.variant_echo("rust_sep__n2g2", payload) == (True, None)
    assert M.variant_echo("rust_legacy__g2", {"star_gates": rec})[0]
    assert not M.variant_echo("rust_legacy__g1", {"star_gates": rec})[0]  # asked for g1, got g2


def test_manifest_cli_builds_from_a_directory(tmp_path, capsys):
    import importlib.util

    from ml.eval.bge.qualification import REPO_ROOT

    spec = importlib.util.spec_from_file_location("build_bge_real_manifest",
                                                  REPO_ROOT / "tools" / "build_bge_real_manifest.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    img, *_ = T.sky_frame(9)
    T.write_bundle(tmp_path / "frames" / "b1", img, None, {"frame_id": "f1", "capture_type": "dark",
                                                           "exposure_started_at": "2026-11-15T03:00:00+00:00"})
    assert tool.main(["dir", "--frames", str(tmp_path / "frames"), "--set", "site_utc_offset=-05:00",
                      "--out", str(tmp_path / "m.json")]) == 0
    m = json.loads((tmp_path / "m.json").read_text())
    assert m["frames"][0]["frame_type"] == "dark" and m["frames"][0]["night_id"] == "2026-11-14"
    assert "1 frames" in capsys.readouterr().out
    assert tool.main(["template"]) == 0 and "repeat_group" in capsys.readouterr().out
