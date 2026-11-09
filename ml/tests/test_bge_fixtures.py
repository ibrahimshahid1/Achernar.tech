"""Fixtures, truth, metrics, bridge and status handling of ml/eval/bge.

These are correctness checks of the evaluation harness itself under its own
simulated assumptions. They do not establish real-camera accuracy.

Run: ``python -m pytest -q ml/tests/test_bge_fixtures.py`` from the repo root.
"""

import json
import math

import numpy as np
import pytest

from ml.eval.bge import bridge, config_sha256, sha256_array
from ml.eval.bge import metrics as MT
from ml.eval.bge import python_methods as PM
from ml.eval.bge import real_frames as RF
from ml.eval.bge import run_benchmark as RB
from ml.eval.bge import scenarios as SC

sep = pytest.importorskip("sep", reason="SEP 1.4.1 is a required prerequisite of the Python reference")


def _hashes(sc):
    return sc.input_hashes()


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["sparse_gaussian_stars", "bad_pixels_invalid", "crowded_field",
                                  "quantization_coarse", "native_clipping", "amp_glow_corner"])
def test_same_seed_gives_identical_image_and_mask_hashes(name):
    a, b = SC.build_scenario(name), SC.build_scenario(name)
    assert _hashes(a) == _hashes(b)
    assert [s.to_json() for s in a.sources] == [s.to_json() for s in b.sources]
    c = SC.build_scenario(name, seed=a.metadata["seed"] + 1000)
    assert _hashes(c)["image"] != _hashes(a)["image"]


def test_sweep_frames_are_deterministic_and_share_the_star_field():
    s1, s2 = SC.sweep_moffat_fixed(), SC.sweep_moffat_fixed()
    assert [f.input_hashes() for f in s1.frames] == [f.input_hashes() for f in s2.frames]
    xy0 = [(s.x, s.y, s.flux) for s in s1.frames[0].sources]
    assert all([(s.x, s.y, s.flux) for s in f.sources] == xy0 for f in s1.frames)
    # Background is constant across the sweep.
    assert all(np.array_equal(f.background, s1.frames[0].background) for f in s1.frames)


def test_masks_are_uint8_zero_one_and_flag_what_they_claim():
    sc = SC.build_scenario("bad_pixels_invalid")
    for name, m in sc.masks.items():
        assert m.dtype == np.uint8 and set(np.unique(m)) <= {0, 1}, name
    assert np.all(sc.masks["invalid"][~np.isfinite(sc.image)] == 1)
    clip = SC.build_scenario("native_clipping")
    sat = clip.metadata["saturation_adu"]
    assert np.all(clip.image <= sat)
    assert np.array_equal(clip.masks["native_saturation"].astype(bool), clip.image >= sat)
    assert any(s.saturated for s in clip.sources) and not all(s.saturated for s in clip.sources)


# --------------------------------------------------------------------------
# Truth consistency
# --------------------------------------------------------------------------


def test_exact_circle_pixel_overlap():
    assert SC.circle_pixel_overlap(-10, 10, -10, 10, 3.0) == pytest.approx(math.pi * 9, rel=1e-12)
    assert SC.circle_pixel_overlap(0, 10, 0, 10, 3.0) == pytest.approx(math.pi * 9 / 4, rel=1e-12)
    assert SC.circle_pixel_overlap(-0.5, 0.5, -0.5, 0.5, 0.1) == pytest.approx(math.pi * 0.01, rel=1e-12)
    assert SC.circle_pixel_overlap(-0.5, 0.5, -0.5, 0.5, 1.0) == pytest.approx(1.0, rel=1e-12)
    # Half-disk straddling the axis: box [0, 5] x [-5, 5].
    assert SC.circle_pixel_overlap(0, 5, -5, 5, 2.0) == pytest.approx(2 * math.pi, rel=1e-12)


def _pixel_response_eps(sigma):
    # Pixel integration (binning) and the uniform-within-pixel assumption of
    # the curve of growth each add 1/12 px^2 of variance per axis, so to
    # leading order HFR_num = 1.17741*sqrt(sigma^2 + 1/6) = analytic*(1 + eps).
    return math.sqrt(1.0 + 1.0 / (6.0 * sigma**2)) - 1.0


@pytest.mark.parametrize("sigma", [1.5, 2.0, 3.0, 5.0])
@pytest.mark.parametrize("phase", [(0.0, 0.0), (0.3, 0.7), (0.5, 0.5), (0.25, 0.0)])
def test_numerical_hfr_of_rendered_gaussian_matches_analytic_within_pixel_bound(sigma, phase):
    # Tolerance: the numerical HFR must exceed the continuous 1.17741*sigma by
    # at least 0 and at most 2*eps, eps the leading pixel-response term above;
    # the factor 2 covers the pixel-phase-dependent remainder, which is of the
    # same order (measured +-0.5% of HFR at sigma 1.5-3). The exact overlap
    # integration itself is accurate to the 1e-6 px root tolerance.
    x0, y0 = 50.0 + phase[0], 50.0 + phase[1]
    stamp = SC.render_gaussian(x0, y0, sigma, 1e4)
    hfr, r_ap = SC.numerical_hfr(stamp, x0, y0)
    analytic = SC.GAUSSIAN_HFR_PER_SIGMA * sigma
    assert r_ap == pytest.approx(6.0 * sigma)
    assert 0.0 <= hfr - analytic <= 2.0 * _pixel_response_eps(sigma) * analytic


@pytest.mark.parametrize("fwhm", [3.0, 5.0, 8.0, 16.0])
def test_numerical_hfr_of_rendered_moffat_matches_truncated_analytic(fwhm):
    # The stamp encloses MOFFAT_ENCLOSED of the model flux; the analytic HFR
    # of that truncated profile solves 1-(1+x^2)^(1-beta) = 0.5*0.999. Same
    # pixel-response bound as the Gaussian with sigma = FWHM/2.3548.
    beta = 3.0
    x0, y0 = 60.3, 60.7
    stamp = SC.render_moffat(x0, y0, fwhm, beta, 1e5)
    hfr, _ = SC.numerical_hfr(stamp, x0, y0)
    x = math.sqrt((1 - 0.5 * SC.MOFFAT_ENCLOSED) ** (1 / (1 - beta)) - 1)
    analytic = SC.moffat_alpha(fwhm, beta) * x
    eps = _pixel_response_eps(fwhm / 2.354820045)
    assert 0.0 <= hfr - analytic <= 2.0 * eps * analytic


def test_rendered_flux_is_conserved():
    g = SC.render_gaussian(40.2, 41.7, 2.0, 1234.0)
    assert g.data.sum() == pytest.approx(1234.0, rel=1e-9)
    m = SC.render_moffat(80.0, 80.0, 5.0, 3.0, 1000.0)
    assert m.data.sum() == pytest.approx(1000.0 * SC.MOFFAT_ENCLOSED, rel=2e-3)
    d = SC.render_annulus(80.0, 80.0, 12.0, 4.0, 1.3, 1000.0)
    assert d.data.sum() == pytest.approx(1000.0, rel=1e-6)


@pytest.mark.parametrize("builder", [SC.sweep_moffat_fixed, SC.sweep_donut])
def test_focus_sweep_truth_optimum(builder):
    # Noiseless truth: per-star HFR depends only on |p - p0|, so the truth
    # curve is minimal at the sample nearest p0 and an exact hyperbola up to
    # pixel-phase scatter (+-0.5% per star, reduced by the median over the
    # field). Allowed fitted shift: 2 focuser steps (2% of the 100-step
    # sampling interval).
    sw = builder()
    curve = [float(np.median([s.hfr_true_px for s in f.sources])) for f in sw.frames]
    nearest = sw.positions[int(np.argmin(np.abs(np.asarray(sw.positions) - sw.p0_true)))]
    assert sw.positions[int(np.argmin(curve))] == nearest
    fit = MT.fit_hyperbola(sw.positions, curve)
    assert fit["status"] == "completed"
    assert abs(fit["p0"] - sw.p0_true) <= 2.0


def test_moffat_sweep_width_model():
    assert SC.moffat_sweep_fwhm(SC.SWEEP_P0) == pytest.approx(3.2)
    assert SC.moffat_sweep_fwhm(SC.SWEEP_P0 + 100) == pytest.approx(math.sqrt(3.2**2 + 4.0**2))


# --------------------------------------------------------------------------
# Metrics on toy inputs (hand-computed answers)
# --------------------------------------------------------------------------


def test_background_map_metrics_hand_computed():
    b_true = np.zeros((4, 4))
    b_hat = np.ones((4, 4))
    b_hat[0, 0] = 5.0
    usable = np.ones((4, 4), bool)
    m = MT.background_map_metrics(b_hat, b_true, usable, block=2)
    assert m["n_pixels"] == 16
    assert m["bias_median_adu"] == 1.0
    assert m["bias_mean_adu"] == pytest.approx(20 / 16)
    assert m["mae_adu"] == pytest.approx(20 / 16)
    assert m["rmse_adu"] == pytest.approx(math.sqrt((15 + 25) / 16))
    # Residual about the median: one pixel at +4, the rest 0.
    assert m["residual_rms_about_median_adu"] == pytest.approx(math.sqrt(16 / 16))
    # 2x2 blocks: means minus median = [1, 0, 0, 0] -> RMS 0.5.
    assert m["large_scale_rms_adu"] == pytest.approx(0.5)
    # A scalar estimate is evaluated as a constant map; masked pixels ignored.
    usable[0, 0] = False
    m2 = MT.background_map_metrics(3.0, b_true, usable, block=2)
    assert m2["n_pixels"] == 15 and m2["bias_median_adu"] == 3.0 and m2["rmse_adu"] == pytest.approx(3.0)
    assert m2["residual_slope_x_adu_per_px"] == pytest.approx(0.0, abs=1e-12)


def test_background_slope_of_a_planar_error():
    yy, xx = np.mgrid[0:20, 0:30].astype(float)
    m = MT.background_map_metrics(0.5 * xx - 0.25 * yy, np.zeros((20, 30)), np.ones((20, 30), bool), block=10)
    assert m["residual_slope_x_adu_per_px"] == pytest.approx(0.5)
    assert m["residual_slope_y_adu_per_px"] == pytest.approx(-0.25)


def test_noise_map_metrics_hand_computed():
    r_true = np.full((3, 3), 2.0)
    r_true[0, 0] = 0.0  # truth undefined there: excluded
    m = MT.noise_map_metrics(3.0, r_true, np.ones((3, 3), bool))
    assert m["n_pixels"] == 8 and m["rel_error_median"] == pytest.approx(0.5)
    assert m["fraction_nonpositive"] == 0.0


def test_match_greedy_is_one_to_one_closest_first():
    det = [(0.0, 0.0), (10.0, 0.0), (0.4, 0.0)]
    truth = [(0.5, 0.0), (9.0, 0.0), (50.0, 50.0)]
    pairs = MT.match_greedy(det, truth, [2.0, 2.0, 2.0])
    assert sorted((i, j) for i, j, _ in pairs) == [(1, 1), (2, 0)]


def _truth(x, y, hfr=2.0, kind="star", group=0, saturated=False, peak=1000.0, sid=0):
    return {"id": sid, "x": x, "y": y, "kind": kind, "hfr_true_px": hfr, "blend_group": group,
            "saturated": saturated, "peak_adu": peak}


def test_detection_and_blend_metrics_hand_computed():
    truth = [_truth(10, 10, group=0, sid=0), _truth(13, 10, group=0, sid=1), _truth(40, 40, group=1, sid=2),
             _truth(70, 20, group=2, sid=3)]
    cands = [
        {"x": 10.2, "y": 10.0, "xmin": 7, "xmax": 16, "ymin": 7, "ymax": 13},  # covers both members of group 0
        {"x": 40.0, "y": 40.5, "xmin": 37, "xmax": 43, "ymin": 37, "ymax": 43},
        {"x": 41.5, "y": 41.5, "xmin": 40, "xmax": 43, "ymin": 40, "ymax": 43},  # second piece of star 2
        {"x": 90.0, "y": 5.0, "xmin": 89, "xmax": 91, "ymin": 4, "ymax": 6},  # pure false detection
    ]
    noise = np.full((60, 100), 10.0)
    src = np.zeros((60, 100))
    src[38:44, 38:44] = 100.0  # footprint of star 2
    d = MT.detection_metrics(cands, truth, 100, 60, src, noise)
    assert d["n_matched"] == 2
    assert d["completeness"] == pytest.approx(2 / 4)
    assert d["purity"] == pytest.approx(2 / 4)
    assert d["false_detections"] == 1 and d["fragments_on_sources"] == 1
    assert d["false_detections_by_quadrant_tl_tr_bl_br"] == [0, 1, 0, 0]
    assert d["completeness_bright_stars"] == pytest.approx(2 / 4)  # all peaks 1000 >= 20*10
    b = MT.blend_metrics(cands, truth, d["_matches"])
    assert b["n_multi_member_groups"] == 1 and b["merged_groups"] == 1
    assert b["split_sources"] == 1  # star 2 has two candidates within 3 px


def test_hfr_metrics_hand_computed():
    truth = [_truth(10, 10, hfr=2.0, group=0, sid=0), _truth(30, 30, hfr=4.0, group=1, sid=1),
             _truth(50, 50, hfr=2.0, group=2, saturated=True, sid=2),
             _truth(70, 70, hfr=2.0, group=3, sid=3), _truth(71, 70, hfr=2.0, group=3, sid=4)]
    accepted = [{"x": 10.1, "y": 10.0, "hfr_px": 2.2}, {"x": 30.0, "y": 30.0, "hfr_px": 3.6},
                {"x": 50.0, "y": 50.0, "hfr_px": 2.0}, {"x": 70.0, "y": 70.0, "hfr_px": 2.5},
                {"x": 99.0, "y": 99.0, "hfr_px": 1.0}]
    m = MT.hfr_metrics(accepted, truth)
    assert m["n_accepted"] == 5 and m["n_matched"] == 4 and m["n_unmatched_accepted"] == 1
    assert m["saturated_leakage"] == 1 and m["blend_leakage"] == 1 and m["n_clean"] == 2
    assert m["clean_bias_median_px"] == pytest.approx((0.2 - 0.4) / 2)
    assert m["clean_mae_px"] == pytest.approx(0.3)
    assert m["clean_rel_bias_median"] == pytest.approx((0.1 - 0.1) / 2, abs=1e-12)
    assert m["clean_outlier_fraction"] == 0.0


def test_hyperbola_and_parabola_fits_recover_known_minima():
    p = np.arange(4000, 5201, 100, dtype=float)
    y = SC_hyper = MT.hyperbola(p, 2.5, 4630.0, 80.0)
    h = MT.fit_hyperbola(p, SC_hyper)
    assert h["status"] == "completed" and h["p0"] == pytest.approx(4630.0, abs=1e-6)
    assert h["a_px"] == pytest.approx(2.5, rel=1e-8) and h["b_steps"] == pytest.approx(80.0, rel=1e-6)
    par = MT.fit_parabola(p, 3.0 + 1e-5 * (p - 4555.0) ** 2)
    assert par["status"] == "completed" and par["p0"] == pytest.approx(4555.0, abs=1e-6)
    shift = MT.focus_shift(p, y, 4600.0)
    assert shift["hyperbola"]["p0_shift"] == pytest.approx(30.0, abs=1e-6)
    assert shift["n_frames_valid"] == 13


def test_focus_minimum_needs_valid_frames_on_both_sides():
    p = np.arange(4000, 5201, 100, dtype=float)
    y = MT.hyperbola(p, 2.5, 4630.0, 80.0)
    one_sided = np.where(p < 4600, np.nan, y)  # one valid frame left of p0
    res = MT.focus_shift(p, one_sided, 4630.0)
    assert res["hyperbola"]["status"] == "failed" and "1/6 valid frames left/right" in res["hyperbola"]["reason"]
    assert "p0_shift" not in res["hyperbola"]
    beyond = MT.focus_shift(p, np.where(p <= 4700, np.nan, y), 4630.0)  # minimum outside the valid range
    assert beyond["hyperbola"]["status"] == "failed" and "p0_shift" not in beyond["hyperbola"]
    ok = MT.focus_shift(p, y, 4630.0)
    assert ok["hyperbola"]["valid_frames_left_right"] == [7, 6]


def test_focus_fits_report_failures_instead_of_numbers():
    p = np.arange(4000, 5201, 100, dtype=float)
    mono = 2.0 + 0.001 * (p - 3000.0)
    assert MT.fit_parabola(p, -((p - 4600.0) ** 2))["status"] == "failed"
    assert MT.fit_hyperbola(p, mono)["status"] == "failed"
    few = [np.nan] * 10 + [1.0, 2.0, 3.0]
    assert MT.fit_hyperbola(p, few)["reason"].startswith("3 valid points")
    assert MT.focus_shift(p, few, 4600.0)["n_frames_valid"] == 3


# --------------------------------------------------------------------------
# Python methods: contract hashes and explicit unsupported statuses
# --------------------------------------------------------------------------


def test_python_sep_configuration_hashes_equal_the_daemon_contract():
    assert config_sha256(PM.sep_background_config_json()) == PM.EXPECTED_BACKGROUND_CONFIG_SHA256
    assert config_sha256(PM.sep_detection_config_json()) == PM.EXPECTED_DETECTION_CONFIG_SHA256
    assert config_sha256(PM.sep_background_config_json()) == \
        "80eab0fbae4abf9f60f348adb7de3004067a3873cca29f996289e9dc14040b84"
    assert PM.sep_detection_config_json()["filter_type"] == "conv"
    assert sep.__version__ == "1.4.1"


@pytest.mark.parametrize("name, method, reason", [
    ("tiny_frame", "python_sep", "frame_smaller_than_mesh_grid"),
    ("all_masked", "python_sep", "no_usable_pixels"),
    ("all_masked", "python_global_scalar", "no_usable_pixels"),
    ("all_nonfinite", "python_sep", "no_usable_pixels"),
    ("zero_size", "python_sep", "malformed_geometry"),
    ("zero_size", "python_global_scalar", "malformed_geometry"),
])
def test_unsupported_inputs_report_unavailable_not_numbers(name, method, reason):
    res = PM.run_python_method(method, SC.build_scenario(name))
    est = res["estimate"]
    assert est.status == "unsupported" and est.level_status == "unavailable"
    assert reason in est.reasons
    assert est.level_summary is None and est.rms_summary is None and est.back is None
    assert res["measurement"]["detection"]["status"] == "unsupported"
    assert res["measurement"]["frame_hfr_median_px"] is None


def test_sep_insufficient_mesh_support_is_detected_before_sep_fabricates_values():
    sc = SC.build_scenario("uniform_gaussian_sky")
    masks = {"invalid": np.zeros(sc.image.shape, np.uint8)}
    masks["invalid"][:, ::2] = 1  # every mesh exactly 50% usable: SEP keeps them
    masks["invalid"][:, 1::4] = 1  # now 25% usable everywhere: SEP would return back=0, rms=1
    est = PM.python_sep_background(sc.image, masks)
    assert est.status == "unsupported" and "insufficient_support" in est.reasons
    assert est.diagnostics["meshes_below_support"] == est.diagnostics["mesh_nx"] * est.diagnostics["mesh_ny"]


def test_zero_scatter_plateau_leaves_noise_unresolved_and_detection_unrun():
    for method in ("python_sep", "python_global_scalar"):
        res = PM.run_python_method(method, SC.build_scenario("quantization_plateau"))
        est = res["estimate"]
        assert est.level_status == "measured" and est.noise_status == "unresolved"
        assert "zero_empirical_scatter" in est.reasons and est.rms_summary is None
        assert res["measurement"]["detection"]["status"] == "unsupported"


def test_negative_rms_spline_is_reported_as_invalid_rms_map():
    est = PM.python_sep_background(*(lambda s: (s.image, s.masks))(SC.build_scenario("steep_ridge_background")))
    assert est.noise_status == "unresolved" and "invalid_rms_map" in est.reasons
    assert est.diagnostics["rms_min_usable"] < 0
    assert PM.rms_map_reasons(np.array([1.0, 0.0])) == ["zero_empirical_scatter"]
    assert PM.rms_map_reasons(np.array([1.0, -0.1, np.nan])) == ["invalid_rms_map"]
    assert PM.rms_map_reasons(np.array([1.0, 2.0])) == []


def test_python_sep_recovers_a_flat_sky_and_finds_every_bright_star():
    sc = SC.build_scenario("sparse_gaussian_stars")
    rec, _ = RB.run_python(sc, "python_sep")
    assert rec["status"] == "completed"
    bg = rec["metrics"]["background"]
    # Bound from the mesh noise: 1000 ADU sky, sigma 10, 64 px meshes ->
    # per-mesh mode error ~ 10/sqrt(4096)*1.25 ~ 0.2 ADU; allow 5x.
    assert abs(bg["bias_median_adu"]) < 1.0
    assert rec["metrics"]["detection"]["completeness_bright_stars"] == 1.0
    assert rec["metrics"]["detection"]["false_detections"] == 0


# --------------------------------------------------------------------------
# Bridge v1 round trip and statuses
# --------------------------------------------------------------------------


def test_bridge_manifest_round_trip(tmp_path):
    scs = [SC.build_scenario("bad_pixels_invalid"), SC.build_scenario("native_clipping")]
    cases = bridge.write_inputs(scs, tmp_path)
    path = bridge.write_manifest(cases, tmp_path, settings={"sep": {"bw": 32, "bh": 32}})
    manifest = bridge.read_manifest(path)
    assert manifest["schema"] == "bge-eval-bridge-v1" and manifest["backends"] == ["legacy_global_clip", "sep"]
    assert manifest["settings"] == {"sep": {"bw": 32, "bh": 32}} and manifest["write_maps"] is True
    for sc, case in zip(scs, manifest["cases"]):
        assert case["width"] == sc.width and case["height"] == sc.height and case["units"] == "adu"
        assert set(case["masks"]) == set(sc.masks)
        assert case["saturation_adu"] == sc.metadata["saturation_adu"]
        image, masks = bridge.load_case(path, case)
        assert image.dtype == np.dtype("<f4")
        assert sha256_array(image) == sc.input_hashes()["image"]
        np.testing.assert_array_equal(image, sc.image)  # NaN-aware
        for k in masks:
            np.testing.assert_array_equal(masks[k], sc.masks[k])
        for rel, sha in case["extras"]["sha256"].items():
            assert manifest["extras"]["input_sha256"][rel] == sha
    raw = (tmp_path / manifest["cases"][0]["pixels"]).read_bytes()
    assert raw[:4] == np.asarray(scs[0].image.ravel()[:1], dtype="<f4").tobytes()  # little-endian row-major
    (tmp_path / manifest["cases"][0]["pixels"]).write_bytes(b"\0" * len(raw))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        bridge.load_case(path, manifest["cases"][0])


def test_bridge_rejects_zero_size_and_unsafe_ids(tmp_path):
    with pytest.raises(ValueError):
        bridge.write_inputs([SC.build_scenario("zero_size")], tmp_path)
    assert not bridge.valid_case_id("a__b") and not bridge.valid_case_id("A-b") and bridge.valid_case_id("a_b2")


def _fake_bridge_output(out_dir, sc, backend, back=None, rms=None, level=None, error=None, n_map=None):
    spatial = back is not None
    payload = {
        "case": sc.id, "backend": backend, "error": error,
        "background": {"method": "sep_spatial_mesh" if spatial else "legacy_global_clip", "level_status": "measured",
                       "noise_status": "measured", "reasons": [], "level": level, "rms": 10.0, "config": {},
                       "config_sha256": "x"},
        "detection": None,
        "catalog": None,
        "candidates": [{"x": s.x, "y": s.y, "xmin": int(s.x) - 3, "xmax": int(s.x) + 3, "ymin": int(s.y) - 3,
                        "ymax": int(s.y) + 3, "npix": 30, "flux": s.flux, "peak": s.peak_adu} for s in sc.sources],
        "quality": {"status": "ok", "rejection_reasons": [], "hfr_px": 2.4, "hfd_px": 4.8,
                    "star_count_detected": len(sc.sources), "star_count_used": 1, "star_rejections": {},
                    "used_stars": [{"x": sc.sources[0].x, "y": sc.sources[0].y, "hfr_px": sc.sources[0].hfr_true_px}]},
        "trailing": {"severity": None, "sensor_pa_deg": None, "policy": "x"},
        "timing_ms": {"background": 1.0, "quality": 2.0, "trailing": 0.5},
    }
    stem = bridge.output_stem(out_dir, sc.id, backend)
    stem.with_suffix(".json").write_text(json.dumps(payload))
    if spatial:
        n = n_map or back.size
        np.asarray(back, dtype="<f4").ravel()[:n].tofile(str(stem) + ".back.f32")
        np.asarray(rms, dtype="<f4").ravel().tofile(str(stem) + ".rms.f32")


def test_bridge_output_reader_and_scoring_on_fake_outputs(tmp_path):
    sc = SC.build_scenario("sparse_gaussian_stars")
    _fake_bridge_output(tmp_path, sc, "sep", back=sc.background, rms=sc.noise_std_sky)
    _fake_bridge_output(tmp_path, sc, "legacy_global_clip", level=1002.0)
    out = bridge.read_output(tmp_path, sc.id, "sep", sc.width, sc.height)
    assert out["status"] == "completed" and out["back"].shape == (sc.height, sc.width)
    rec, _ = RB.run_rust_parse(sc, tmp_path, "sep")
    assert rec["status"] == "completed"
    assert rec["metrics"]["background"]["rmse_adu"] == pytest.approx(0.0, abs=1e-3)
    assert rec["metrics"]["noise"]["rel_error_median"] == pytest.approx(0.0, abs=1e-6)
    assert rec["metrics"]["detection"]["completeness"] == 1.0
    assert rec["metrics"]["hfr"]["n_matched"] == 1 and rec["metrics"]["hfr"]["clean_mae_px"] == pytest.approx(0.0)
    legacy, _ = RB.run_rust_parse(sc, tmp_path, "legacy_global_clip")
    assert legacy["metrics"]["background"]["bias_median_adu"] == pytest.approx(2.0)
    assert bridge.read_output(tmp_path, sc.id, "nope", sc.width, sc.height)["status"] == "missing"
    _fake_bridge_output(tmp_path, sc, "sep", back=sc.background, rms=sc.noise_std_sky, n_map=10)
    assert bridge.read_output(tmp_path, sc.id, "sep", sc.width, sc.height)["status"] == "failed"
    _fake_bridge_output(tmp_path, sc, "legacy_global_clip", level=1.0, error="boom")
    failed, _ = RB.run_rust_parse(sc, tmp_path, "legacy_global_clip")
    assert failed["status"] == "failed" and failed["reason"] == "boom"


def test_rust_revision_without_sep_adapter_is_skipped_not_scored(tmp_path):
    sc = SC.build_scenario("sparse_gaussian_stars")
    _fake_bridge_output(tmp_path, sc, "sep", level=None)
    stem = bridge.output_stem(tmp_path, sc.id, "sep").with_suffix(".json")
    payload = json.loads(stem.read_text())
    payload["background"].update(level_status="unavailable", noise_status="unavailable", reasons=["backend_failure"],
                                 backend_error="the SEP backend is not built into this revision")
    stem.write_text(json.dumps(payload))
    rec, _ = RB.run_rust_parse(sc, tmp_path, "sep", "55afec11abcdef")
    assert rec["status"] == "skipped" and "not integrated at 55afec11abcd" in rec["reason"]
    assert rec["metrics"] == {}


def test_missing_rust_bridge_is_reported_as_skipped_not_passed(tmp_path):
    res = bridge.run_bridge(tmp_path / "m.json", tmp_path / "out", tmp_path, tmp_path / "t", tmp_path / "log")
    assert res["status"] == "skipped" and res["reason"].startswith(bridge.UNAVAILABLE)
    (tmp_path / "achernar-daemon" / "src").mkdir(parents=True)
    (tmp_path / "achernar-daemon" / "Cargo.toml").write_text("[package]\nname='x'\n")
    (tmp_path / "achernar-daemon" / "src" / "main.rs").write_text("fn main() {}\n")
    res = bridge.run_bridge(tmp_path / "m.json", tmp_path / "out", tmp_path, tmp_path / "t", tmp_path / "log")
    assert res["status"] == "skipped" and res["reason"].startswith(bridge.UNAVAILABLE)


def test_benchmark_command_writes_explicit_statuses(tmp_path):
    out = tmp_path / "run"
    assert RB.main(["--out", str(out), "--scenarios", "uniform_gaussian_sky", "tiny_frame", "zero_size",
                    "all_masked"]) == 0
    doc = json.loads((out / "results.json").read_text())
    rows = {(r["scenario"], r["method"]): r for r in doc["results"]}
    assert len(rows) == 4 * 5
    assert all(r["status"] in ("completed", "unsupported", "skipped", "failed") and
               (r["status"] == "completed" or r["reason"]) for r in rows.values())
    assert rows[("tiny_frame", "python_sep")]["status"] == "unsupported"
    assert rows[("all_masked", "python_sep")]["metrics"]["honest_absence"] is True
    assert rows[("uniform_gaussian_sky", "rust_sep")]["status"] == "skipped"
    assert doc["real_frames"]["status"] == "not_run" and doc["contract_check"]["background_hash_matches_daemon_default"]
    summary = (out / "summary.md").read_text()
    for heading in (RB.SECTION_TRUTH, RB.SECTION_PARITY, RB.SECTION_REAL):
        assert heading in summary


# --------------------------------------------------------------------------
# Real-frame adapter refuses unsupported claims
# --------------------------------------------------------------------------


def _real_manifest(tmp_path):
    img = SC.build_scenario("sparse_gaussian_stars").image
    img.astype("<f4").tofile(tmp_path / "dev.f32")
    manifest = {"schema": RF.MANIFEST_SCHEMA, "frames": [
        {"id": "dev", "path": "dev.f32", "format": "f32", "width": img.shape[1], "height": img.shape[0],
         "split": "development", "group": "g"},
        {"id": "hold", "path": "does-not-exist.f32", "format": "f32", "width": 4, "height": 4, "split": "holdout"},
    ]}
    path = tmp_path / "real.json"
    path.write_text(json.dumps(manifest))
    return path


def test_real_frames_emit_observations_only_and_never_open_holdout(tmp_path):
    path = _real_manifest(tmp_path)
    res = RF.evaluate(path)  # the hold-out file does not exist: opening it would raise
    assert res["status"] == "observations_only" and res["pass_fail"] is None and res["accuracy"] is None
    assert res["n_frames_evaluated"] == 1 and res["n_holdout_frames_skipped"] == 1
    with pytest.raises(RF.RealFrameRefusal):
        RF.evaluate(path, include_holdout=True)


def test_real_frame_thresholds_must_be_predeclared_for_this_manifest(tmp_path):
    path = _real_manifest(tmp_path)
    sha = RF._sha256(path)
    future = {"schema": RF.THRESHOLDS_SCHEMA, "predeclared_utc": "2999-01-01T00:00:00+00:00",
              "manifest_sha256": sha, "metrics": {"status_failure_rate": {"max": 0.0}}}
    (tmp_path / "t.json").write_text(json.dumps(future))
    with pytest.raises(RF.RealFrameRefusal, match="predeclared"):
        RF.evaluate(path, tmp_path / "t.json")
    wrong = dict(future, predeclared_utc="2026-01-01T00:00:00+00:00", manifest_sha256="0" * 64)
    (tmp_path / "t.json").write_text(json.dumps(wrong))
    with pytest.raises(RF.RealFrameRefusal, match="different manifest"):
        RF.evaluate(path, tmp_path / "t.json")
    ok = dict(wrong, manifest_sha256=sha)
    (tmp_path / "t.json").write_text(json.dumps(ok))
    res = RF.evaluate(path, tmp_path / "t.json")
    assert res["pass_fail"]["status_failure_rate"]["status"] == "pass" and res["accuracy"] is None
