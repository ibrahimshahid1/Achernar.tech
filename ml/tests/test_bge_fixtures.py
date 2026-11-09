"""Fixtures, truth, metrics, bridge and status handling of ml/eval/bge.

These are correctness checks of the evaluation harness itself under its own
simulated assumptions. They do not establish real-camera accuracy.

Run: ``python -m pytest -q ml/tests/test_bge_fixtures.py`` from the repo root.

Tests that run SEP take the ``sep_required`` fixture: without the pinned
``sep==1.4.1`` they ERROR with "missing prerequisite: sep==1.4.1" instead of
being skipped. Every other test (truth, metrics, bridge, statuses) runs
without SEP.
"""

import json
import math
import os
import stat
import sys
from unittest import mock

import numpy as np
import pytest

from ml.eval.bge import bridge, config_sha256, sha256_array
from ml.eval.bge import metrics as MT
from ml.eval.bge import python_methods as PM
from ml.eval.bge import real_frames as RF
from ml.eval.bge import run_benchmark as RB
from ml.eval.bge import scenarios as SC


@pytest.fixture
def sep_required():
    """The pinned SEP module; raises MissingPrerequisite (test ERROR) if absent."""
    return PM.sep_module()


def test_missing_sep_is_a_hard_prerequisite_error_not_a_skip():
    with mock.patch.dict(sys.modules, {"sep": None}):
        with pytest.raises(PM.MissingPrerequisite, match=r"missing prerequisite: sep==1\.4\.1"):
            PM.sep_module()
        sc = SC.build_scenario("sparse_gaussian_stars")
        with pytest.raises(PM.MissingPrerequisite):
            PM.python_sep_background(sc.image, sc.masks)
        with pytest.raises(PM.MissingPrerequisite):
            RB.run_python(sc, "python_sep")  # never recorded as a failed or skipped row
        # Refusals that need no SEP still work without it.
        est = PM.python_sep_background(*(lambda t: (t.image, t.masks))(SC.build_scenario("tiny_frame")))
        assert "frame_smaller_than_mesh_grid" in est.reasons


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


def test_python_sep_configuration_hashes_equal_the_daemon_contract(sep_required):
    sep = sep_required
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


def test_zero_scatter_plateau_leaves_noise_unresolved_and_detection_unrun(sep_required):
    for method in ("python_sep", "python_global_scalar"):
        res = PM.run_python_method(method, SC.build_scenario("quantization_plateau"))
        est = res["estimate"]
        assert est.level_status == "measured" and est.noise_status == "unresolved"
        assert "zero_empirical_scatter" in est.reasons and est.rms_summary is None
        assert res["measurement"]["detection"]["status"] == "unsupported"


def test_negative_rms_spline_is_reported_as_invalid_rms_map(sep_required):
    est = PM.python_sep_background(*(lambda s: (s.image, s.masks))(SC.build_scenario("steep_ridge_background")))
    assert est.noise_status == "unresolved" and "invalid_rms_map" in est.reasons
    assert est.diagnostics["rms_map_validity"]["min_rms"] < 0


def test_positive_rms_undershoot_is_invalid_rms_map_d018(sep_required):
    sc = SC.build_scenario("steep_ridge_amp600")
    assert not sc.sources
    res = PM.run_python_method("python_sep", sc)
    est = res["estimate"]
    v = est.diagnostics["rms_map_validity"]
    assert est.noise_status == "unresolved" and "invalid_rms_map" in est.reasons
    assert v["min_rms"] > 0 and v["count_below_bound"] > 0  # positive, yet below 0.5 x min mesh sigma
    assert v["bound"] == pytest.approx(0.5 * v["min_positive_mesh_sigma"])
    assert res["measurement"]["detection"]["status"] == "unsupported" and res["measurement"]["candidates"] is None


def test_rms_validity_rules_are_the_shared_labeler_rules():
    bw = bh = 64
    usable = np.ones((128, 128), bool)
    flat = np.full((128, 128), 5.0, np.float32)
    assert PM.rms_map_validity(flat, usable, bw, bh)[0] == []
    plateau = np.zeros((128, 128), np.float32)  # every mesh sigma 0: no positive bound exists
    assert PM.rms_map_validity(plateau, usable, bw, bh)[0] == ["zero_empirical_scatter"]
    zero = flat.copy()
    zero[3, 3] = 0.0  # zero scatter, and also below 0.5 x the 5 ADU mesh sigma (D-018)
    assert PM.rms_map_validity(zero, usable, bw, bh)[0] == ["zero_empirical_scatter", "invalid_rms_map"]
    bad = flat.copy()
    bad[5, 5], bad[6, 6] = -0.1, np.nan
    assert PM.rms_map_validity(bad, usable, bw, bh)[0] == ["invalid_rms_map"]


def test_python_sep_recovers_a_flat_sky_and_finds_every_bright_star(sep_required):
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
        "detection": {"method": "sep_extract" if spatial else "legacy_connected_components",
                      "noise_floor_applied": False},
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


def test_rust_detection_not_run_is_not_scored_as_zero_completeness(tmp_path):
    sc = SC.build_scenario("sparse_gaussian_stars")
    _fake_bridge_output(tmp_path, sc, "sep", back=sc.background, rms=sc.noise_std_sky)
    stem = bridge.output_stem(tmp_path, sc.id, "sep").with_suffix(".json")
    payload = json.loads(stem.read_text())
    payload["background"].update(noise_status="unresolved", reasons=["invalid_rms_map"])
    payload.update(detection=None, candidates=[], catalog=None, catalog_error="noise_unresolved")
    payload["quality"].update(status="DetectionUnavailable", used_stars=[], hfr_px=None)
    stem.write_text(json.dumps(payload))
    rec, art = RB.run_rust_parse(sc, tmp_path, "sep")
    assert rec["status"] == "unsupported"
    assert rec["stages"]["detection"]["status"] == "unsupported" and "DetectionUnavailable" in rec["stages"]["detection"]["reason"]
    assert "detection" not in rec["metrics"] and "hfr" not in rec["metrics"] and "noise" not in rec["metrics"]
    assert art["candidates"] is None and "background" in rec["metrics"]


def test_catalog_parity_against_d006_bounds():
    row = {"x": 10.0, "y": 20.0, "npix": 30, "xmin": 7, "xmax": 13, "ymin": 17, "ymax": 23, "flags": 0,
           "flux": 1000.0, "a": 2.0, "b": 1.5, "theta": 0.3}
    same = RB.catalog_parity([row], [dict(row, x=10.00005)])
    assert same["catalog_within_d006"] and same["n_matched_within_0p5px"] == 1
    off = RB.catalog_parity([row], [dict(row, flux=1000.1)])
    assert not off["catalog_within_d006"] and off["max_abs_diff"]["flux_rel"] == pytest.approx(1e-4)
    npix = RB.catalog_parity([row], [dict(row, npix=31)])
    assert not npix["catalog_within_d006"] and npix["integer_field_mismatches"]["npix"] == 1
    assert not RB.catalog_parity([row], [row, dict(row, x=50.0)])["catalog_within_d006"]


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


def test_benchmark_command_writes_explicit_statuses(tmp_path, sep_required):
    out = tmp_path / "run"
    assert RB.main(["--out", str(out), "--scenarios", "uniform_gaussian_sky", "tiny_frame", "zero_size",
                    "all_masked", "quantization_plateau"]) == 0
    doc = json.loads((out / "results.json").read_text())
    rows = {(r["scenario"], r["method"]): r for r in doc["results"]}
    assert len(rows) == 5 * 5
    plateau = rows[("quantization_plateau", "python_sep")]
    assert plateau["status"] == "unsupported" and plateau["partial"] and plateau["scored_stages"] == ["background"]
    assert rows[("uniform_gaussian_sky", "python_sep")]["status"] == "completed"
    assert all(not r["partial"] for r in rows.values() if r["status"] in ("completed", "skipped", "failed"))
    assert all(r["status"] in ("completed", "unsupported", "skipped", "failed") and
               (r["status"] == "completed" or r["reason"]) for r in rows.values())
    assert rows[("tiny_frame", "python_sep")]["status"] == "unsupported"
    assert rows[("all_masked", "python_sep")]["metrics"]["honest_absence"] is True
    assert rows[("uniform_gaussian_sky", "rust_sep")]["status"] == "skipped"
    assert doc["real_frames"]["status"] == "not_run" and doc["contract_check"]["background_hash_matches_daemon_default"]
    summary = (out / "summary.md").read_text()
    for heading in (RB.SECTION_TRUTH, RB.SECTION_PARITY, RB.SECTION_REAL):
        assert heading in summary
    assert "no number was scored" not in summary and "| `python_sep` | partial: " in summary


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


def test_real_frames_emit_observations_only_and_never_open_holdout(tmp_path, sep_required):
    path = _real_manifest(tmp_path)
    res = RF.evaluate(path)  # the hold-out file does not exist: opening it would raise
    assert res["status"] == "observations_only" and res["pass_fail"] is None and res["accuracy"] is None
    assert res["n_frames_evaluated"] == 1 and res["n_holdout_frames_skipped"] == 1
    with pytest.raises(RF.RealFrameRefusal):
        RF.evaluate(path, include_holdout=True)


def test_real_frame_thresholds_must_be_predeclared_for_this_manifest(tmp_path, sep_required):
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


# --------------------------------------------------------------------------
# Review fixes: partial rows (F1), cargo classification (F2), digitized truth
# (F3), hfr stage and sweep status (F4), parity coverage (F5), timing (F6)
# --------------------------------------------------------------------------


def test_partial_rows_are_marked_and_name_their_scored_stages():
    sc = SC.build_scenario("quantization_plateau")
    stages = {"background": {"status": "completed", "reason": None},
              "noise": {"status": "unsupported", "reason": "noise unresolved"},
              "detection": {"status": "unsupported", "reason": "noise unresolved"},
              "hfr": {"status": "unsupported", "reason": "detection not run"}}
    rec = {"metrics": RB.score(sc, stages, np.float64(200.0), None, None, None, None)}
    RB.finish_record(rec, stages)
    assert rec["status"] == "unsupported" and rec["partial"] is True
    assert rec["scored_stages"] == ["background"] and rec["unscored_stages"] == ["noise", "detection", "hfr"]
    assert stages["background"]["scored"] and not stages["noise"]["scored"]
    assert RB._row_status(rec).startswith("partial: ") and "scored: background" in RB._row_status(rec)
    # Quantization-limited level: scored against the expected digitized level; analog bias is secondary.
    assert "quantization-limited" in rec["metrics"]["background"]["truth"]
    assert rec["metrics"]["background"]["bias_median_adu"] == pytest.approx(0.0, abs=1e-6)
    assert rec["metrics"]["background_vs_analog"]["bias_median_adu"] == pytest.approx(-0.2, abs=1e-4)
    full = {k: {"status": "completed", "reason": None} for k in ("background", "noise", "detection")}
    full["hfr"] = {"status": RB.NOT_APPLICABLE, "reason": "scenario has no truth stars"}
    uni = SC.build_scenario("uniform_gaussian_sky")
    rec2 = {"metrics": RB.score(uni, full, np.float64(1000.0), np.float64(10.0), [], None, None)}
    RB.finish_record(rec2, full)
    assert rec2["status"] == "completed" and rec2["partial"] is False and "hfr" not in rec2["unscored_stages"]


def test_digitized_moments_are_exact():
    from scipy import special

    mean, std = SC.digitized_moments(np.array([3.2]), 0.0, 1.0)
    assert mean[0] == 3.0 and std[0] == 0.0
    mean, std = SC.digitized_moments(np.array([0.3]), 3.0, 1.0)  # sigma >> step: Sheppard holds
    assert std[0] == pytest.approx(math.sqrt(9.0 + 1.0 / 12.0), rel=1e-6) and mean[0] == pytest.approx(0.3, abs=1e-6)
    k = np.arange(-20, 21, dtype=float)  # direct definition for mu = 200.3 (shifted), sigma = 0.5
    p = special.ndtr((k + 0.5 - 0.3) / 0.5) - special.ndtr((k - 0.5 - 0.3) / 0.5)
    ref_mean = np.sum(p * k)
    ref_std = math.sqrt(np.sum(p * (k - ref_mean) ** 2))
    mean, std = SC.digitized_moments(np.array([200.3]), 0.5, 1.0)
    assert mean[0] - 200.0 == pytest.approx(ref_mean, abs=1e-9) and std[0] == pytest.approx(ref_std, rel=1e-9)


def test_quantized_scenarios_use_the_effective_digitized_noise_as_truth():
    coarse = SC.build_scenario("quantization_coarse")
    off = coarse.source_image < 1e-3
    observed = float(np.std(coarse.image[off]))
    eff = coarse.metadata["effective_sky_std_adu"]
    # ~1.9e5 off-source pixels: the std's standard error is ~0.16%; allow 1%.
    assert observed == pytest.approx(eff, rel=0.01) and eff / 0.5 > 1.1
    assert np.allclose(coarse.noise_std_sky, eff) and np.allclose(coarse.noise_std_sky_analog, 0.5)
    assert not coarse.metadata["quantization_limited_level"] and coarse.metadata["noise_scoreable"]
    assert coarse.scoring_background() is coarse.background
    plateau = SC.build_scenario("quantization_plateau")
    assert plateau.metadata["quantization_limited_level"] and not plateau.metadata["noise_scoreable"]
    assert np.allclose(plateau.scoring_background(), 200.0) and np.all(plateau.image[plateau.source_image < 1e-3] == 200)


def test_oracle_detects_with_the_effective_noise_and_refuses_a_swallowed_one():
    coarse = SC.build_scenario("quantization_coarse")
    est = PM.oracle_background(coarse)
    assert est.noise_status == "measured" and np.allclose(est.rms, coarse.metadata["effective_sky_std_adu"])
    plateau = SC.build_scenario("quantization_plateau")
    est = PM.oracle_background(plateau)
    assert est.noise_status == "unresolved" and np.allclose(est.back, 200.0)
    meas = PM.detect_and_measure(plateau.image, plateau.masks, est)  # returns before SEP is needed
    assert meas["detection"]["status"] == "unsupported"


def test_hfr_stage_distinguishes_no_usable_stars_from_not_applicable():
    stars, empty = SC.build_scenario("sparse_gaussian_stars"), SC.build_scenario("uniform_gaussian_sky")
    no_stars = RB.hfr_stage(stars, True, 0, None, "rejections {'blended': 3}")
    assert no_stars["status"] == "unsupported" and no_stars["reason"].startswith("no usable stars")
    assert RB.hfr_stage(empty, True, 0, None, "x")["status"] == RB.NOT_APPLICABLE
    assert RB.hfr_stage(stars, True, 3, 2.4, "x")["status"] == "completed"
    assert RB.hfr_stage(stars, False, None, None, "x")["status"] == "unsupported"
    assert RB.overall_status({"background": {"status": "completed"},
                              "hfr": {"status": RB.NOT_APPLICABLE}}) == ("completed", None)


def _sweep_rec(hfr, hfr_status="completed", status="completed"):
    return {"status": status, "reason": None, "frame_hfr_median_px": hfr, "n_accepted": 1, "metrics": {},
            "stages": {"hfr": {"status": hfr_status}}}


def test_sweep_with_frames_without_hfr_is_partial_not_completed():
    p = np.arange(4000, 5201, 100, dtype=float)
    y = list(MT.hyperbola(p, 2.5, 4630.0, 80.0))
    full = RB.summarize_sweep_method([_sweep_rec(v) for v in y], p, 4630.0, 6, y)
    assert full["status"] == "completed" and full["frame_counts"]["frames_with_hfr"] == 13
    recs = [_sweep_rec(v) for v in y]
    recs[0] = _sweep_rec(None, hfr_status="unsupported", status="unsupported")
    recs[12] = _sweep_rec(None, hfr_status="failed", status="failed")
    part = RB.summarize_sweep_method(recs, p, 4630.0, 6, y)
    assert part["status"] == "partial" and "11/13" in part["reason"]
    assert part["frame_counts"] == {"frames": 13, "frames_with_hfr": 11, "frames_failed": 1, "frames_without_hfr": 2}
    assert part["fit"]["hyperbola"]["p0_shift"] == pytest.approx(0.0, abs=1e-6)


def test_parity_compares_raw_rms_maps_and_globals_whatever_the_noise_status():
    back = np.full((4, 4), 1000.0, np.float32)
    rms = np.full((4, 4), 5.0, np.float32)
    rms[0, 0] = -0.95  # a negative spline undershoot: not a noise measurement, still compared
    rec = {"status": "unsupported", "background_result": {"level_status": "measured", "noise_status": "unresolved"}}
    py = {"back": back, "rms": rms, "globalback": 1000.0, "globalrms": 37.6, "candidates": None}
    rs = {"back": back.copy(), "rms": rms.copy(), "globalback": 1000.004, "globalrms": None, "candidates": None}
    p = RB.parity(py, rs, rec, rec)
    assert p["rms_max_abs_diff"] == 0.0 and p["rms_within_d006_bound"] is True
    assert p["globalback_within_d006_bound"] is True and "globalrms_abs_diff" not in p
    assert "segmentation" in p["not_compared"]
    off = RB.parity(py, dict(rs, globalback=1000.5), rec, rec)
    assert off["globalback_within_d006_bound"] is False
    t = RB.parity_tally([p, off])
    assert t["rms"] == {"within": 2, "compared": 2} and t["globalback"] == {"within": 1, "compared": 2}
    assert t["globalrms"]["compared"] == 0


def test_python_sep_keeps_the_raw_rms_map_for_parity_when_noise_is_unresolved(sep_required):
    rec, art = RB.run_python(SC.build_scenario("steep_ridge_background"), "python_sep")
    assert rec["partial"] and "noise" not in rec["metrics"]
    assert isinstance(art["rms"], np.ndarray) and float(art["rms"].min()) < 0
    assert art["globalrms"] == pytest.approx(37.6, rel=0.01)


RUNTIME_PANIC = """   Compiling achernar-daemon v0.1.0
    Finished `release` profile [optimized] target(s) in 1.0s
     Running unittests src/main.rs (/nonexistent/achernar_daemon-abc)

running 1 test
thread 'vision::bge_eval::bge_eval_bridge' panicked at src/vision/bge_eval.rs:199:5:
unsupported manifest schema
test vision::bge_eval::bge_eval_bridge ... FAILED

test result: FAILED. 0 passed; 1 failed; 0 ignored

error: test failed, to rerun pass `--bin achernar-daemon`
"""
COMPILE_ERROR = """error[E0308]: mismatched types
   --> src/vision/vision_loop.rs:993:23
error: could not compile `achernar-daemon` (bin "achernar-daemon" test) due to 2 previous errors
"""
SIGNAL_EXIT = """     Running unittests src/main.rs (/nonexistent/achernar_daemon-abc)
error: test failed, to rerun pass `--bin achernar-daemon`

Caused by:
  process didn't exit successfully: `/nonexistent/achernar_daemon-abc bge_eval_bridge` (signal: 11, SIGSEGV: invalid memory reference)
"""


def test_cargo_runtime_failures_are_failed_and_only_unbuilt_tests_are_skipped():
    r = bridge.classify_cargo_test(101, RUNTIME_PANIC)
    assert r["status"] == "failed" and "panic at src/vision/bge_eval.rs:199:5" in r["reason"]
    assert r["binary"] == "/nonexistent/achernar_daemon-abc"
    s = bridge.classify_cargo_test(101, SIGNAL_EXIT)
    assert s["status"] == "failed" and "SIGSEGV" in s["reason"]
    c = bridge.classify_cargo_test(101, COMPILE_ERROR)
    assert c["status"] == "skipped" and c["reason"] == f"{bridge.UNAVAILABLE}: build failed (see log)"
    n = bridge.classify_cargo_test(101, "error: could not find `Cargo.toml`\n")  # no binary ran
    assert n["status"] == "skipped"
    ok = bridge.classify_cargo_test(0, "     Running unittests src/main.rs (/x)\nrunning 1 test\ntest result: ok.\n")
    assert ok["status"] == "completed" and ok["binary"] == "/x"


def _fake_executable(path, stderr_text, exit_code, write_json=None):
    body = "#!/bin/sh\n"
    if write_json:
        body += f"printf '%s' '{write_json}' > \"$ACHERNAR_BGE_EVAL_OUT/{{name}}\"\n"
    body += f"cat <<'TXT' 1>&2\n{stderr_text}TXT\nexit {exit_code}\n"
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_run_bridge_with_fake_cargo_separates_runtime_and_build_failures(tmp_path, monkeypatch):
    daemon = tmp_path / "repo" / "achernar-daemon"
    (daemon / "src").mkdir(parents=True)
    (daemon / "Cargo.toml").write_text("[package]\nname = 'achernar-daemon'\n")
    (daemon / "src" / "bge_eval.rs").write_text("#[test] #[ignore] fn bge_eval_bridge() {}\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    args = (tmp_path / "m.json", tmp_path / "out", tmp_path / "repo", tmp_path / "target", tmp_path / "log.txt")
    _fake_executable(bindir / "cargo", RUNTIME_PANIC, 101)
    res = bridge.run_bridge(*args)
    assert res["status"] == "failed" and res["returncode"] == 101 and "runtime" in res["reason"]
    _fake_executable(bindir / "cargo", COMPILE_ERROR, 101)
    res = bridge.run_bridge(*args)
    assert res["status"] == "skipped" and "build failed" in res["reason"]


def test_rust_timing_clears_stale_outputs_and_checks_the_return_code(tmp_path):
    frame = SC.build_scenario("tiny_frame")
    payload = json.dumps({"case": frame.id, "backend": "sep", "error": None, "background": {"backend_error": None},
                          "timing_ms": {"background": 5.0, "quality": 1.0, "trailing": 0.5}})
    stale = tmp_path / "timing" / "out_sep_0"
    stale.mkdir(parents=True)
    (stale / f"{frame.id}__sep.json").write_text(payload)  # left over from an earlier run
    (tmp_path / "logs").mkdir()
    failing = _fake_executable(tmp_path / "fail.sh", "boom\n", 101)
    runs = RB.rust_timing_runs(str(failing), tmp_path / "m.json", tmp_path / "timing", "sep", frame, 2,
                               tmp_path / "logs")
    assert [r["repeat"] for r in runs] == ["warmup", 0, 1] and not any(r["ok"] for r in runs)
    summary = RB.summarize_rust_timing(runs)
    assert summary["status"] == "failed" and summary["bridge_total_ms_median"] is None and "2/2" in summary["reason"]
    good = tmp_path / "good.sh"
    _fake_executable(good, "", 0, write_json=payload)
    good.write_text(good.read_text().replace("{name}", f"{frame.id}__sep.json"))
    summary = RB.summarize_rust_timing(RB.rust_timing_runs(str(good), tmp_path / "m.json", tmp_path / "timing", "sep",
                                                           frame, 2, tmp_path / "logs"))
    assert summary["status"] == "completed" and summary["repeats_ok"] == 2
    assert summary["bridge_total_ms_median"] == pytest.approx(6.5) and "cold process" in summary["repeat_semantics"]
