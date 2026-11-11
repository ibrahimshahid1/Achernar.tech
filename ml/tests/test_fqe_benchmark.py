"""Classical FQE development benchmark: metric math, matching, statuses and the seal (fixtures only).

Native observations are generated here next to fqe_sources fixtures (the same
layout as the public sources); the Rust bridge is replaced by a fake runner
that writes bridge-format outputs, so no cargo build runs in these tests.
"""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import fqe_sources as fs  # noqa: E402
from data_pipeline.fqe_sources import SealedPartitionError, load_source, write_fixture_source  # noqa: E402
from eval import fqe_benchmark as fb  # noqa: E402
from eval.bge import bridge  # noqa: E402
from preprocess.normalize import robust_normalize  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_npz(path):
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k].copy() for k in npz.files}


def _read_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_records(path, records):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in records))


def _repin(fx, arrays, records, extra_files=()):
    """Rewrite fqe.npz/records, re-bind registry + manifest, and return a contract pinning every file."""
    d = fx.directory
    np.savez_compressed(d / "fqe.npz", **arrays)
    _write_records(d / "records.jsonl", records)
    sha = _sha(d / "fqe.npz")
    registry = json.loads((d / "fqe.splits.json").read_text())
    registry["dataset_sha256"] = sha
    (d / "fqe.splits.json").write_text(json.dumps(registry, indent=2))
    manifest = json.loads((d / "manifest.json").read_text())
    manifest["artifacts"] = [{"path": "fqe.npz", "sha256": sha}]
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    entry = dict(fx.entry)
    names = ("fqe.npz", "fqe.splits.json", "records.jsonl", "manifest.json", *extra_files)
    entry["files"] = {name: _sha(d / name) for name in names}
    return entry


def _gauss(shape, center, sigma, flux):
    y, x = np.indices(shape, dtype=np.float64)
    return flux * np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma**2)) / (2 * math.pi * sigma**2)


def add_sim_native(fx, contract, *, corrupt_rows=()):
    """observations.npz + oracle records for a sim-style fixture (crops = robust_normalize(raw))."""
    d = fx.directory
    arrays = _read_npz(d / "fqe.npz")
    records = _read_records(d / "records.jsonl")
    n = arrays["crops"].shape[0]
    rng = np.random.default_rng(7)
    raw_all = np.zeros((2 * n, 64, 64), dtype=np.float32)
    flags = np.zeros((2 * n, 64, 64), dtype=np.uint8)
    groups = np.full(2 * n, "filler", dtype="U64")
    for rec in records:
        r = rec.get("fqe_index")
        if r is None:
            continue
        ai = rec["array_index"]
        sigma = float(arrays["reference_hfr_px"][r]) / math.sqrt(2 * math.log(2))
        center = [31.2 + 0.1 * (r % 3), 31.4]
        noise = {"sky_electrons": 50.0 + r, "gradient_x_per_frame": 2.0, "gradient_y_per_frame": -1.0,
                 "read_noise_electrons": 3.0}
        background = fb.sim_oracle_background(noise)
        expected = _gauss((64, 64), center, sigma, 40000.0) + background
        raw = np.rint(np.clip(rng.poisson(expected) + rng.normal(0, 3, expected.shape) + 500, 0, 65535)).astype(np.float32)
        raw_all[ai] = raw
        groups[ai] = rec["split_group"]
        arrays["crops"][r] = robust_normalize(raw)[0]
        measured, why = fb.sep_aperture_hfr(raw - background - 500.0, center, 24.0)
        assert why is None
        arrays["measured_hfr_px"][r] = np.float32(measured)
        rec["parameters"]["center"] = center
        rec["noise"] = noise
        rec["bias_numeric_level"] = 500.0
        if r in corrupt_rows:
            raw_all[ai][5, 5] += 999.0  # native no longer reproduces the loaded crop
    np.savez_compressed(d / "observations.npz", raw_crops=raw_all, pixel_flags=flags, split_group=groups)
    entry = _repin(fx, arrays, records, extra_files=("observations.npz",))
    return contract.with_source(fx.source_id, entry)


def add_hst_native(fx, contract, *, frame_shape=(160, 200), masked_near_row=None):
    """exposures/<frame>/ for the val exposures of an hst-style fixture; targets = the teacher chain."""
    import sep

    d = fx.directory
    arrays = _read_npz(d / "fqe.npz")
    records = _read_records(d / "records.jsonl")
    rng = np.random.default_rng(11)
    by_frame = {}
    for rec in records:
        if rec.get("fqe_index") is not None:
            by_frame.setdefault(rec["frame_id"], []).append(rec)
    for frame_id, recs in sorted(by_frame.items()):
        split = arrays["split"][recs[0]["fqe_index"]]
        if split != "validation":
            continue  # train/test exposures are never created: opening one would fail loudly
        h, w = frame_shape
        background = np.full(frame_shape, 100.0, dtype=np.float32)
        science = (background + rng.normal(0, 3, frame_shape)).astype(np.float64)
        positions = []
        for k, rec in enumerate(recs):
            x, y = 40.3 + 30.0 * k, 50.6 + 40.0 * (k % 2)
            positions.append((x, y))
            science += _gauss(frame_shape, (x, y), 1.0 + 0.3 * k, 20000.0)
        science = science.astype(np.float32)
        dq = np.zeros(frame_shape, dtype=np.uint16)
        dq[5, 5] = 16
        if masked_near_row is not None:
            r_idx = [rec["fqe_index"] for rec in recs].index(masked_near_row) if masked_near_row in [
                rec["fqe_index"] for rec in recs] else None
            if r_idx is not None:
                x, y = positions[r_idx]
                dq[int(round(y)) + 9, int(round(x)) + 9] = 16  # inside the 32 px measurement crop, outside the aperture
        valid = dq == 0
        mask = ~valid
        cleaned = science.copy()
        cleaned[mask] = float(np.median(science[~mask]))
        residual = np.ascontiguousarray(cleaned - background)
        xs = np.asarray([p[0] for p in positions])
        ys = np.asarray([p[1] for p in positions])
        hfr, flags = sep.flux_radius(residual, xs, ys, np.full(len(recs), 6.0), 0.5, mask=mask, subpix=5)
        raw_crops = [np.zeros((64, 64), dtype=np.float32)]  # a filler so observation_index != k
        exposure_records = []
        for k, rec in enumerate(recs):
            x, y = positions[k]
            cx, cy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
            x0, y0 = cx - 32, cy - 32
            crop = science[y0:y0 + 64, x0:x0 + 64]
            assert crop.shape == (64, 64)
            imputed = crop.copy()
            imputed[~valid[y0:y0 + 64, x0:x0 + 64]] = background[y0:y0 + 64, x0:x0 + 64][~valid[y0:y0 + 64, x0:x0 + 64]]
            r = rec["fqe_index"]
            arrays["crops"][r] = robust_normalize(imputed)[0]
            value = np.float32(hfr[k])
            for name in ("hfr_px", "reference_hfr_px", "measured_hfr_px"):
                arrays[name][r] = value
            arrays["log_hfr"][r] = np.log(value)
            rec.update(x_native_pixel=x, y_native_pixel=y, crop_origin_xy=[x0, y0], observation_index=len(raw_crops),
                       hfr_aperture_radius_px=6.0)
            raw_crops.append(crop.copy())
            exposure_records.append({"x_native_pixel": x, "y_native_pixel": y})
        exposure_records.append({"x_native_pixel": positions[0][0] + 10.0, "y_native_pixel": positions[0][1]})
        exposure = d / "exposures" / frame_id
        exposure.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(exposure / "frame.npz", science_electrons=science, error_pipeline_electrons=np.ones_like(science),
                            dq=dq, valid_mask=valid, sep_background_estimate_electrons=background)
        np.savez_compressed(exposure / "observations.npz", raw_crops64=np.stack(raw_crops))
        _write_records(exposure / "records.jsonl", exposure_records)
        (exposure / "manifest.json").write_text(json.dumps({"split": "test", "artifacts": [
            {"path": "frame.npz", "sha256": _sha(exposure / "frame.npz")},
            {"path": "observations.npz", "sha256": _sha(exposure / "observations.npz")}]}))
    entry = _repin(fx, arrays, records)
    return contract.with_source(fx.source_id, entry)


def fake_runner(*, hfr_by_method=None, status="completed"):
    """A stand-in for the cargo bridge: bridge-format outputs from a simple peak finder.

    SEP methods report frame_smaller_than_mesh_grid below 128 px per axis, as
    the daemon does; candidates touching an invalid pixel within 9 px are
    rejected as masked_support.
    """
    from scipy import ndimage

    hfr_by_method = hfr_by_method or {"rust_legacy__g1": 1.5, "rust_legacy__g2": 1.6, "rust_sep__n1g1": 1.7,
                                      "rust_sep__n2g2": 1.8}
    calls = []

    def run(manifest_path, out_dir, log_path):
        calls.append(str(manifest_path))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("fake bridge\n")
        if status != "completed":
            return {"status": status, "reason": "skipped: rust bridge unavailable: build failed (fake)"}
        manifest = bridge.read_manifest(manifest_path)
        for case in manifest["cases"]:
            image, masks = bridge.load_case(manifest_path, case)
            invalid = masks.get("invalid", np.zeros_like(image, dtype=np.uint8))
            for backend in manifest["backends"]:
                record = {"case": case["id"], "backend": backend, "error": None,
                          "timing_ms": {"background": 0.5, "quality": 0.25, "trailing": 0.1}}
                small = case["width"] < 128 or case["height"] < 128
                if bridge.backend_of(backend) == "sep" and small:
                    record.update(background={"method": "sep_spatial_mesh", "level_status": "unavailable",
                                              "noise_status": "unavailable",
                                              "reasons": ["frame_smaller_than_mesh_grid"]},
                                  detection=None, candidates=[],
                                  quality={"status": "BackgroundUnavailable", "used_stars": [], "star_rejections": {}})
                else:
                    med = float(np.median(image))
                    mad = float(np.median(np.abs(image - med))) * 1.4826 or 1.0
                    labels, n = ndimage.label(image > med + 10 * mad)
                    cands = []
                    for lab in range(1, n + 1):
                        yy, xx = np.nonzero(labels == lab)
                        wts = image[yy, xx] - med
                        cands.append({"x": float(np.sum(xx * wts) / wts.sum()), "y": float(np.sum(yy * wts) / wts.sum()),
                                      "flux": float(wts.sum()), "peak": float(image[yy, xx].max())})
                    cands.sort(key=lambda c: -c["flux"])
                    used, tallies = [], {}
                    for c in cands[:48]:
                        if len(used) >= 16:
                            break
                        cx, cy = int(round(c["x"])), int(round(c["y"]))
                        box = invalid[max(0, cy - 9):cy + 10, max(0, cx - 9):cx + 10]
                        if box.any():
                            tallies["masked_support"] = tallies.get("masked_support", 0) + 1
                            continue
                        used.append({"x": c["x"], "y": c["y"], "hfr_px": hfr_by_method[backend]})
                    record.update(background={"method": "legacy_global_clip" if bridge.backend_of(backend) != "sep"
                                              else "sep_spatial_mesh", "level_status": "measured",
                                              "noise_status": "measured", "reasons": []},
                                  detection={"backbone": "fake"}, candidates=cands,
                                  quality={"status": "Valid" if used else "NoUsableStars", "used_stars": used,
                                           "star_rejections": tallies, "star_count_detected": len(cands),
                                           "hfr_px": float(np.median([u["hfr_px"] for u in used])) if used else None})
                (Path(out_dir) / f"{case['id']}__{backend}.json").write_text(json.dumps(record))
        return {"status": "completed", "reason": None}

    run.calls = calls
    return run


@pytest.fixture
def two_sources(tmp_path):
    data = tmp_path / "data"
    sim = write_fixture_source(data, "fixsim", n_groups=12, rows_per_group=3, style="sim")
    hst = write_fixture_source(data, "fixhst", n_groups=6, rows_per_group=4, style="hst",
                               group_splits=["train", "train", "train", "val", "val", "test"],
                               calibration="unavailable", base_contract=sim.contract)
    contract = add_sim_native(sim, hst.contract)
    contract = add_hst_native(hst, contract)
    return {"root": data, "sim": sim, "hst": hst, "contract": contract}


def _snapshot(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in Path(root).rglob("*"))


def _rows(result, comparator_id, source=None):
    return [r for r in result["rows"] if r["comparator_id"] == comparator_id and (source is None or r["source"] == source)]


# --------------------------------------------------------------------------- #
# Partition refusal and output guards
# --------------------------------------------------------------------------- #
def test_cli_and_api_refuse_the_test_partition(tmp_path, capsys):
    assert fb.main(["--partition", "test", "--sources", "stellar_sim_v1", "--out", str(tmp_path / "o")]) == 2
    assert "refused" in capsys.readouterr().err
    assert fb.main(["--partition", "train_fit", "--sources", "stellar_sim_v1", "--out", str(tmp_path / "o")]) == 2
    with pytest.raises(SealedPartitionError):
        fb.run_benchmark(["stellar_sim_v1"], out_dir=tmp_path / "o", partition="test")
    assert not (tmp_path / "o").exists()


def test_output_directory_guards(two_sources, tmp_path):
    c = two_sources
    with pytest.raises(ValueError, match="public data root"):
        fb.run_benchmark(["fixsim"], out_dir=c["root"] / "out", methods=["constant"], contract=c["contract"],
                         data_root=c["root"])
    with pytest.raises(ValueError, match="production model registry"):
        fb._guard_out_dir(fb.REPO_ROOT / "models" / "x", c["root"])
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "old.json").write_text("{}")
    with pytest.raises(ValueError, match="not empty"):
        fb.run_benchmark(["fixsim"], out_dir=busy, methods=["constant"], contract=c["contract"], data_root=c["root"])


# --------------------------------------------------------------------------- #
# Metric math (hand-computed expectations)
# --------------------------------------------------------------------------- #
def test_error_metrics_hand_values():
    m = fb.error_metrics([1.0, 2.0, 4.0], [2.0, 2.0, 2.0])
    assert m["n"] == 3
    assert m["hfr_mae_px"] == pytest.approx(1.0)
    assert m["signed_bias_px"] == pytest.approx(1.0 / 3.0)
    assert m["log_hfr_mae"] == pytest.approx(2.0 * math.log(2.0) / 3.0)
    assert m["median_relative_error"] == pytest.approx(0.5)
    empty = fb.error_metrics([], [])
    assert empty["n"] == 0 and empty["hfr_mae_px"] == fb.UNAVAILABLE
    with pytest.raises(ValueError):
        fb.error_metrics([0.0], [1.0])  # an unmeasured width can never enter as 0


def test_weighted_median_and_mixed_arm_hand_values():
    assert fb.weighted_median([4.0, 1.0, 3.0, 2.0]) == pytest.approx(2.5)
    assert fb.weighted_median([1.0, 2.0, 10.0], [1.0, 1.0, 2.0]) == pytest.approx(6.0)
    assert fb.weighted_median([5.0, 1.0, 3.0]) == pytest.approx(3.0)
    # Source-balanced: A = [1, 3] (1/4 each), B = [10, 20, 30, 40] (1/8 each).
    values = [1.0, 3.0, 10.0, 20.0, 30.0, 40.0]
    weights = [0.25, 0.25, 0.125, 0.125, 0.125, 0.125]
    assert fb.weighted_median(values, weights) == pytest.approx(6.5)
    with pytest.raises(ValueError):
        fb.weighted_median([1.0, np.nan])


def test_loglog_mapping_hand_values():
    mapping = fb.fit_loglog([1.0, math.e], [math.e**2, math.e])
    assert mapping["b"] == pytest.approx(-1.0) and mapping["a"] == pytest.approx(2.0)
    assert fb.apply_loglog(mapping, math.e**2) == pytest.approx(1.0)
    assert fb.apply_loglog(mapping, 0.0) is None and fb.apply_loglog(mapping, float("nan")) is None
    with pytest.raises(fb.MappingUnavailable):
        fb.fit_loglog([2.0, 2.0], [1.0, 3.0])


def _row(uid, prediction, target, status=None, **strata):
    status = status or ("measured" if prediction is not None else "unmeasured")
    return {"sample_uid": uid, "group_uid": "g", "prediction": prediction, "target": target, "status": status,
            "reason": None if status == "measured" else "why", "secondary_targets": {}, "strata": strata,
            "timing_ms": 1.0, "timing_scope": "row"}


def test_coverage_counts_every_unmeasured_row():
    rows = [_row("a", 1.0, 1.0), _row("b", 2.0, 1.0), _row("c", None, 1.0), _row("d", None, 1.0, status="unsupported")]
    s = fb.summarize_rows(rows)
    assert s["population"] == 4 and s["measured"] == 2 and s["coverage"] == pytest.approx(0.5)
    assert s["status_counts"] == {"measured": 2, "unmeasured": 1, "unsupported": 1, "failed": 0}
    assert s["metrics"]["hfr_mae_px"] == pytest.approx(0.5)  # errors 0 and 1 on measured rows only
    assert fb.coverage([]) == fb.UNAVAILABLE


def test_make_row_enforces_status_contract(two_sources):
    c = two_sources
    arrays = load_source("fixsim", partitions=("train_fit", "val"), contract=c["contract"], data_root=c["root"])
    spec = fb.NORMALIZED_SPEC
    with pytest.raises(ValueError):
        fb.make_row(spec, arrays, 0, prediction=0.0, status="measured")
    with pytest.raises(ValueError):
        fb.make_row(spec, arrays, 0, prediction=1.0, status="unmeasured", reason="x")
    with pytest.raises(ValueError):
        fb.make_row(spec, arrays, 0, prediction=None, status="failed")
    with pytest.raises(ValueError):
        fb.make_row(spec, arrays, 0, prediction=None, status="skipped", reason="x")
    rows = fb.prediction_rows(spec, arrays, [0, 1, 2], [1.2, float("nan"), -1.0])
    assert [r["status"] for r in rows] == ["measured", "unmeasured", "unmeasured"]
    assert [r["prediction"] for r in rows] == [pytest.approx(1.2), None, None]
    row = rows[0]
    for key in ("sample_uid", "group_uid", "source", "partition", "target", "prediction", "status", "reason",
                "processing_mode", "input_definition", "reference_definition", "timing_ms", "stages"):
        assert key in row
    assert set(row["stages"]) == set(fb.STAGES)


def test_paired_comparison_uses_the_common_valid_population():
    a = [_row("1", 1.0, 1.0), _row("2", 1.5, 1.0), _row("3", 3.0, 2.0), _row("4", None, 2.0)]
    b = [_row("1", None, 1.0), _row("2", 1.0, 1.0), _row("3", 2.5, 2.0), _row("4", 2.0, 2.0)]
    out = fb.paired_comparison({"A": a, "B": b}, ["A", "B"])
    assert out["population"] == 4 and out["common_valid"] == 2
    assert out["per_comparator"]["A"]["coverage_full_population"] == pytest.approx(0.75)
    assert out["per_comparator"]["B"]["coverage_full_population"] == pytest.approx(0.75)
    # Common rows 2 and 3: A errors 0.5, 1.0; B errors 0.0, 0.5.
    assert out["per_comparator"]["A"]["common_valid_metrics"]["hfr_mae_px"] == pytest.approx(0.75)
    assert out["per_comparator"]["B"]["common_valid_metrics"]["hfr_mae_px"] == pytest.approx(0.25)
    pair = out["pairwise"]["A vs B"]
    assert pair["n"] == 2 and pair["mean_abs_error_difference_px"] == pytest.approx(0.5)
    assert pair["fraction_a_closer"] == pytest.approx(0.0)
    restricted = fb.paired_comparison({"A": a, "B": b}, ["A", "B"], restrict_to={"3", "4"})
    assert restricted["population"] == 2 and restricted["common_valid"] == 1
    with pytest.raises(ValueError):
        fb.paired_comparison({"A": a + a[:1]}, ["A"])


def test_strata_bins_and_categories():
    rows = [_row("1", 0.6, 0.65, star_snr=10.0, motion="none"), _row("2", None, 0.9, star_snr=60.0, motion="linear"),
            _row("3", 3.0, 3.0, star_snr=None, motion="none")]
    strata = fb.standard_strata(rows)
    assert strata["width_bin"]["[0, 0.7)"]["population"] == 1
    assert strata["width_bin"]["[0.7, 1)"]["coverage"] == pytest.approx(0.0)
    assert strata["width_bin"]["[2.5, 4)"]["coverage"] == pytest.approx(1.0)
    assert strata["star_snr_bin"]["missing"]["population"] == 1
    assert strata["motion"]["none"]["population"] == 2 and strata["motion"]["linear"]["measured"] == 0


# --------------------------------------------------------------------------- #
# Matching and end-to-end stage classification
# --------------------------------------------------------------------------- #
def test_mutual_nearest_matching_and_ambiguity():
    out = fb.match_mutual_nearest([(0.0, 0.0), (10.0, 10.0), (50.0, 50.0)], [(0.5, 0.0), (10.2, 10.0), (11.0, 10.0)],
                                  radius=1.0, ambiguity_radius=3.0)
    assert out[0]["star_index"] == 0 and out[0]["distance_px"] == pytest.approx(0.5) and not out[0]["ambiguous"]
    assert out[1]["star_index"] == 1 and out[1]["ambiguous"] and out[1]["n_stars_within_ambiguity"] == 2
    assert out[2]["star_index"] is None and out[2]["distance_px"] is None
    claimed = fb.match_mutual_nearest([(0.0, 0.0), (0.3, 0.0)], [(0.2, 0.0)], radius=1.0, ambiguity_radius=3.0)
    assert claimed[0]["star_index"] is None and claimed[0]["claimed_by_other_reference"]
    assert claimed[1]["star_index"] == 0 and claimed[1]["ambiguous"]  # another reference within 3 px of the star
    beyond = fb.match_mutual_nearest([(0.0, 0.0)], [(1.2, 0.0)], radius=1.0, ambiguity_radius=3.0)
    assert beyond[0]["star_index"] is None and beyond[0]["nearest_star_distance_px"] == pytest.approx(1.2)
    assert fb.match_mutual_nearest([(0.0, 0.0)], [], radius=1.0, ambiguity_radius=3.0)[0]["star_index"] is None


def test_reference_stage_classification_against_caps():
    cands = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (40.0, 0.0)]
    used_two = [(0.0, 0.0), (10.0, 0.0)]
    assert fb.examined_count(cands, used_two, max_examined=3, max_used=2) == 2
    kw = dict(radius=1.0, max_examined=3, max_used=2)
    assert fb.classify_reference((20.0, 0.3), cands, used_two, **kw)["stage"] == "after_used_cap"
    assert fb.classify_reference((40.0, 0.0), cands, used_two, **kw)["stage"] == "beyond_examined_cap"
    assert fb.classify_reference((100.0, 0.0), cands, used_two, **kw)["stage"] == "not_detected"
    one = [(0.0, 0.0)]
    assert fb.examined_count(cands, one, max_examined=3, max_used=2) == 3
    rejected = fb.classify_reference((10.0, 0.5), cands, one, **kw)
    assert rejected["stage"] == "gate_rejected" and rejected["candidate_rank"] == 1
    assert fb.classify_reference((0.2, 0.0), cands, one, **kw)["stage"] == "used_star_claimed_elsewhere"
    assert fb.classify_reference((0.0, 0.0), [], [], **kw)["stage"] == "not_detected"


def _payload(*, level="measured", reasons=(), candidates=(), used=(), tallies=None, status="Valid", detection=True):
    return {"background": {"level_status": level, "reasons": list(reasons), "method": "m"},
            "detection": {"x": 1} if detection else None,
            "candidates": [{"x": x, "y": y, "flux": 1.0} for x, y in candidates],
            "quality": {"status": status, "used_stars": [{"x": x, "y": y, "hfr_px": h} for x, y, h in used],
                        "star_rejections": tallies or {}}}


def test_reference_outcomes_statuses():
    small = fb.reference_outcomes(_payload(level="unavailable", reasons=["frame_smaller_than_mesh_grid"], detection=False,
                                           status="BackgroundUnavailable"), [(31.0, 31.0)], radius=1.0)[0]
    assert small["status"] == "unsupported" and small["prediction"] is None
    assert small["reason"].startswith("frame_smaller_than_mesh_grid")
    other_bg = fb.reference_outcomes(_payload(level="unavailable", reasons=["no_usable_pixels"], detection=False),
                                     [(1.0, 1.0)], radius=1.0)[0]
    assert other_bg["status"] == "unmeasured" and "no_usable_pixels" in other_bg["reason"]
    fallback = fb.reference_outcomes(_payload(reasons=["fallback_used"], used=[(31.0, 31.0, 2.0)],
                                              candidates=[(31.0, 31.0)]), [(31.0, 31.0)], radius=1.0)[0]
    assert fallback["status"] == "unmeasured" and fallback["reason"].startswith("fallback_used")
    no_detection = fb.reference_outcomes(_payload(status="DetectionUnavailable"), [(1.0, 1.0)], radius=1.0)[0]
    assert no_detection["status"] == "unmeasured" and no_detection["reason"].startswith("detection_unavailable")
    gated = fb.reference_outcomes(_payload(candidates=[(31.2, 31.0)], tallies={"masked_support": 1, "low_snr": 0},
                                           status="NoUsableStars"), [(31.0, 31.0)], radius=1.0)[0]
    assert gated["status"] == "unmeasured" and gated["reason"] == "gate_rejected: masked_support"
    assert gated["stages"]["acceptance"]["status"] == "rejected"
    shared = fb.reference_outcomes(_payload(candidates=[(31.2, 31.0), (5.0, 5.0)], tallies={"masked_support": 2},
                                            status="NoUsableStars"), [(31.0, 31.0)], radius=1.0)[0]
    assert "not attributable" in shared["reason"]
    measured = fb.reference_outcomes(_payload(candidates=[(31.2, 31.0)], used=[(31.2, 31.0, 1.7)]), [(31.0, 31.0)],
                                     radius=1.0)[0]
    assert measured["status"] == "measured" and measured["prediction"] == pytest.approx(1.7)
    assert measured["match"]["distance_px"] == pytest.approx(0.2)


def test_rust_methods_must_be_section4_ids():
    assert fb.canonical_rust_methods(["rust_legacy_global_clip", "rust_sep", "rust_legacy__g2"]) == [
        "rust_legacy__g1", "rust_sep__n1g1", "rust_legacy__g2"]
    for plain in ("legacy_global_clip", "sep"):
        with pytest.raises(ValueError, match="current defaults"):
            fb.canonical_rust_methods([plain])
    with pytest.raises(KeyError):
        fb.canonical_rust_methods(["rust_sep__n3g1"])


# --------------------------------------------------------------------------- #
# Fitted comparators use train_fit rows only
# --------------------------------------------------------------------------- #
def test_constant_and_mapping_fit_on_train_fit_only(two_sources, tmp_path):
    c = two_sources
    result = fb.run_benchmark(["fixsim", "fixhst"], out_dir=tmp_path / "o", methods=["constant", "sharpness"],
                              contract=c["contract"], data_root=c["root"])
    sim = load_source("fixsim", partitions=("train_fit", "calibration", "val"), contract=c["contract"],
                      data_root=c["root"])
    hst = load_source("fixhst", partitions=("train_fit", "val"), contract=c["contract"], data_root=c["root"])
    assert (sim.partition == "calibration").any()
    fit_sim = sim.target_hfr_px[sim.partition == "train_fit"].astype(np.float64)
    fit_hst = hst.target_hfr_px[hst.partition == "train_fit"].astype(np.float64)
    fits = result["config"]["fits"]["constant"]
    assert fits["fixsim_only"]["median_hfr_px"] == pytest.approx(float(np.median(fit_sim)))
    assert fits["fixsim_only"]["n_fit"] == fit_sim.size
    assert fits["fixhst_only"]["median_hfr_px"] == pytest.approx(float(np.median(fit_hst)))
    weights = np.concatenate([np.full(fit_hst.size, 0.5 / fit_hst.size), np.full(fit_sim.size, 0.5 / fit_sim.size)])
    expected_mixed = fb.weighted_median(np.concatenate([fit_hst, fit_sim]), weights)
    assert fits["mixed"]["median_hfr_px"] == pytest.approx(expected_mixed)
    # Every val row of both sources gets each arm's constant; no calibration or val target entered a fit.
    for arm in ("fixsim_only", "fixhst_only", "mixed"):
        rows = _rows(result, f"constant__{arm}")
        assert len(rows) == int((sim.partition == "val").sum() + (hst.partition == "val").sum())
        assert {r["partition"] for r in rows} == {"val"}
        assert all(r["prediction"] == pytest.approx(fits[arm]["median_hfr_px"]) for r in rows)
    mapping = result["config"]["fits"]["sharpness"]["tenengrad__fixsim_only"]
    assert mapping["n_fit"] + mapping["n_dropped"] == fit_sim.size
    ten = np.array([fb.baselines.tenengrad(x) for x in sim.crops[sim.partition == "train_fit"]])
    assert mapping["b"] == pytest.approx(fb.fit_loglog(ten, fit_sim)["b"])
    summary = result["summary"]["fixsim"]["comparators"]["sharpness__tenengrad__fixsim_only"]
    assert summary["sweep_rank_metrics"].startswith("unavailable")


# --------------------------------------------------------------------------- #
# Native comparators, statuses, joins and the seal (fake bridge)
# --------------------------------------------------------------------------- #
def test_full_benchmark_with_native_fixtures_and_fake_bridge(two_sources, tmp_path):
    c = two_sources
    before = _snapshot(c["root"])
    runner = fake_runner()
    result = fb.run_benchmark(["fixsim", "fixhst"], out_dir=tmp_path / "o", contract=c["contract"],
                              data_root=c["root"], bridge_runner=runner)
    # Nothing was written next to any fixture NPZ (no registry, no cache, no outputs).
    assert _snapshot(c["root"]) == before
    for name in ("config.json", "rows.jsonl", "summary.json", "populations.json", "summary.md"):
        assert (tmp_path / "o" / name).is_file()
    sim_val = 6  # 2 val families x 3 rows
    # Normalized-input, constant, sharpness, SEP and Rust rows exist for every member of each population.
    g1 = _rows(result, "native_rust__rust_legacy__g1__sim_observation64__pixel_flags")
    assert len(g1) == sim_val and all(r["status"] == "measured" for r in g1)
    assert all(r["prediction"] == pytest.approx(1.5) for r in g1)
    assert all(r["match"]["distance_px"] <= fb.FROZEN["match"]["sim_radius_px"] for r in g1)
    sep_small = _rows(result, "native_rust__rust_sep__n1g1__sim_observation64__pixel_flags")
    assert {r["status"] for r in sep_small} == {"unsupported"}
    assert all(r["prediction"] is None and r["reason"].startswith("frame_smaller_than_mesh_grid") for r in sep_small)
    oracle = _rows(result, "sep_flux_radius__sim_oracle_bg_true_centre_r24")
    assert len(oracle) == sim_val and all(r["status"] == "measured" for r in oracle)
    assert max(r["recomputation_rel_diff"] for r in oracle) < 1e-6
    # HST: teacher reproduction is identity on the parent frames (declared as such).
    teacher = _rows(result, "sep_flux_radius__hst_teacher_reproduction")
    assert len(teacher) == 8 and all(r["status"] == "measured" for r in teacher)
    assert max(r["abs_diff_to_teacher_px"] for r in teacher) < 1e-5
    assert {r["measurement_role"] for r in teacher} == {"teacher_identity"}
    crop_sep = _rows(result, "native_rust__rust_sep__n2g2__hst_crop64__teacher_dq")
    assert {r["status"] for r in crop_sep} == {"unsupported"}
    frame_rows = _rows(result, "native_rust__rust_sep__n1g1__hst_parent_frame__teacher_dq")
    assert len(frame_rows) == 8 and all(r["population"] == fb.POPULATION_END_TO_END for r in frame_rows)
    assert all(r["status"] == "measured" for r in frame_rows) and all(r["timing_scope"] == "frame" for r in frame_rows)
    populations = result["populations"]
    key = "fixhst/native_rust__rust_sep__n1g1__hst_parent_frame__teacher_dq"
    assert populations[key]["population"] == 8 and len(populations[key]["matched"]) == 8
    e2e = result["summary"]["fixhst"]["end_to_end"]["native_rust__rust_sep__n1g1__hst_parent_frame__teacher_dq"]
    assert e2e["full_population_coverage"] == pytest.approx(1.0)
    assert "normalized_input__direct_hfr" in e2e["per_crop_comparators_on_matched_population"]
    sel = result["selections"]["fixhst/native_rust__rust_legacy__g1__hst_parent_frame__teacher_dq"]
    assert len(sel) == 2 and all(s["references"] == 4 for s in sel)
    # Neighbour stratum comes from the exposure records and is attached to every comparator's rows.
    norm = _rows(result, "normalized_input__direct_hfr", "fixhst")
    assert any(r["strata"].get("neighbour_within_24px") for r in norm)
    # Only parent frames of val exposures were opened; train/test exposures do not even exist here.
    frames = result["config"]["native_provenance"]["fixhst"]["frames"]
    assert sorted(frames) == ["exp0003", "exp0004"]
    # rows.jsonl is strict JSON with one row per comparator x population member.
    lines = (tmp_path / "o" / "rows.jsonl").read_text().splitlines()
    assert len(lines) == len(result["rows"])
    json.loads(lines[0])
    # Rust methods were run with section-4 ids only, never plain backend strings.
    for info in result["config"]["bridge_runs"].values():
        manifest = json.loads(Path(info["manifest"]).read_text())
        assert set(manifest["backends"]) <= set(bridge.RUST_VARIANTS)
        assert manifest["write_maps"] is False


def test_masked_support_is_unmeasured_and_imputed_policy_differs(tmp_path):
    data = tmp_path / "data"
    hst = write_fixture_source(data, "fixhst", n_groups=4, rows_per_group=2, style="hst",
                               group_splits=["train", "train", "val", "test"], calibration="unavailable")
    contract = add_hst_native(hst, hst.contract, masked_near_row=4)
    result = fb.run_benchmark(["fixhst"], out_dir=tmp_path / "o", methods=["rust_legacy__g1"], contract=contract,
                              data_root=data, bridge_runner=fake_runner(), parent_frames=False)
    teacher = {r["sample_uid"]: r for r in _rows(result, "native_rust__rust_legacy__g1__hst_crop64__teacher_dq")}
    imputed = {r["sample_uid"]: r for r in _rows(result, "native_rust__rust_legacy__g1__hst_crop64__imputed_saturation_only")}
    masked_uid = "fixhst:FIX:exp0002:SCI1:source00000"
    assert teacher[masked_uid]["status"] == "unmeasured" and teacher[masked_uid]["prediction"] is None
    assert teacher[masked_uid]["reason"] == "gate_rejected: masked_support"
    assert imputed[masked_uid]["status"] == "measured"
    summary = result["summary"]["fixhst"]["comparators"]
    assert summary["native_rust__rust_legacy__g1__hst_crop64__teacher_dq"]["summary"]["target"]["coverage"] == \
        pytest.approx(0.5)


def test_bridge_failure_fails_rows_never_zero(two_sources, tmp_path):
    c = two_sources
    result = fb.run_benchmark(["fixsim"], out_dir=tmp_path / "o", methods=["rust_legacy__g1"], contract=c["contract"],
                              data_root=c["root"], bridge_runner=fake_runner(status="skipped"))
    rows = _rows(result, "native_rust__rust_legacy__g1__sim_observation64__pixel_flags")
    assert rows and all(r["status"] == "failed" and r["prediction"] is None for r in rows)
    assert all("rust bridge unavailable" in r["reason"] for r in rows)
    s = result["summary"]["fixsim"]["comparators"]["native_rust__rust_legacy__g1__sim_observation64__pixel_flags"]
    assert s["summary"]["target"]["coverage"] == pytest.approx(0.0)
    assert s["summary"]["target"]["metrics"]["hfr_mae_px"] == fb.UNAVAILABLE


def test_native_join_mismatch_fails_the_row(tmp_path):
    data = tmp_path / "data"
    sim = write_fixture_source(data, "fixsim", n_groups=12, rows_per_group=3, style="sim")
    probe = load_source("fixsim", partitions=("val",), contract=sim.contract, data_root=data)
    bad = int(probe.npz_row[0])
    contract = add_sim_native(sim, sim.contract, corrupt_rows=(bad,))
    result = fb.run_benchmark(["fixsim"], out_dir=tmp_path / "o", methods=["sep_flux_radius", "rust_legacy__g1"],
                              contract=contract, data_root=data, bridge_runner=fake_runner())
    for cid in ("sep_flux_radius__sim_oracle_bg_true_centre_r24",
                "native_rust__rust_legacy__g1__sim_observation64__pixel_flags"):
        rows = _rows(result, cid)
        failed = [r for r in rows if r["status"] == "failed"]
        assert len(failed) == 1 and failed[0]["reason"].startswith("native_join_mismatch")
        assert failed[0]["prediction"] is None
    assert result["config"]["notes"]


def test_hst_native_refuses_non_development_exposures(two_sources):
    c = two_sources
    arrays = load_source("fixhst", partitions=("train_fit", "val"), contract=c["contract"], data_root=c["root"])
    val = arrays.partition_indices("val")
    with pytest.raises(SealedPartitionError):
        next(fb.iter_hst_native(arrays, val, contract=c["contract"], data_root=c["root"],
                                allowed_partitions=("train_fit",)))


def test_hst_exposure_hash_mismatch_is_refused(two_sources):
    c = two_sources
    exposure = c["hst"].directory / "exposures" / "exp0003"
    manifest = json.loads((exposure / "manifest.json").read_text())
    manifest["artifacts"][0]["sha256"] = "0" * 64
    (exposure / "manifest.json").write_text(json.dumps(manifest))
    arrays = load_source("fixhst", partitions=("val",), contract=c["contract"], data_root=c["root"])
    with pytest.raises(fs.HashMismatch):
        next(fb.iter_hst_native(arrays, arrays.partition_indices("val"), contract=c["contract"], data_root=c["root"]))
