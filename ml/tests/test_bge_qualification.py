"""Truth-free metrics, injection-recovery, dark trials, thresholds, hold-out guard and the
runner of ml/eval/bge/qualification. Synthetic FITS fixtures in tmp dirs only; the oracle
(truth B and R) is the method whose expected behaviour is known exactly.

Run: ``python -m pytest -p no:cacheprovider -q ml/tests/test_bge_qualification.py``
"""

import datetime as dt
import json

import numpy as np
import pytest

from ml.eval.bge.qualification import DAEMON_ANALYSIS_REPRESENTATION as REP
from ml.eval.bge.qualification import DEFAULT_SPLIT_RULE, DEFAULT_THRESHOLDS, REPO_ROOT
from ml.eval.bge.qualification import builders as B
from ml.eval.bge.qualification import dark_injection as DI
from ml.eval.bge.qualification import holdout as HO
from ml.eval.bge.qualification import injection as INJ
from ml.eval.bge.qualification import masks as MK
from ml.eval.bge.qualification import methods as M
from ml.eval.bge.qualification import realmetrics as RM
from ml.eval.bge.qualification import run_qualification as RQ
from ml.eval.bge.qualification import stats as ST
from ml.eval.bge.qualification import testing as T
from ml.eval.bge.qualification import thresholds as TH
from ml.eval.bge.qualification.sidecar import SidecarLog


def _oracle_case(seed, shape=(256, 320), **kw):
    img, b, r, sources = T.sky_frame(seed, shape=shape, **kw)
    return M.Case(f"qo{seed}_o", img, {}, None, {"background": b, "noise": r}), sources


# --------------------------------------------------------------------------
# (a) pull, (c) negated residual
# --------------------------------------------------------------------------


def test_oracle_pull_is_standard_normal_on_gaussian_frames():
    case, _ = _oracle_case(1, shape=(384, 512), n_stars=25)
    out = M.run_oracle(case)
    assert out.usable and out.detections
    excl = M.unusable_mask(case.image, case.masks) | MK.source_mask(out.detections, case.image.shape)
    pull = RM.frame_pull(case.image, out.back_map(case.image.shape), out.rms_map(case.image.shape), excl)
    assert pull["status"] == "completed" and pull["n_pixels"] > 150_000
    assert abs(pull["clipped_mean"]) < 0.01 and abs(pull["robust_std"] - 1.0) < 0.01
    assert abs(pull["frac_abs_gt3"] - RM.TAIL_REFERENCE["p_abs_gt3"]) < 0.0006
    assert pull["frac_abs_gt5"] < 1e-4
    assert all(abs(p - 1.0) < 0.02 for p in pull["robust_std_by_cfa_phase_00_01_10_11"])
    # A 10 % noise underestimate shows as a robust std of ~1.11.
    biased = RM.frame_pull(case.image, out.back_map(case.image.shape), 0.9 * out.rms_map(case.image.shape), excl)
    assert abs(biased["robust_std"] - 1 / 0.9) < 0.015


def test_difference_pull_of_two_noise_frames_has_unit_std():
    a, _ = _oracle_case(2, n_stars=0)
    b, _ = _oracle_case(3, n_stars=0)
    none = np.zeros(a.image.shape, bool)
    dp = RM.difference_pull(a.image, a.truth["noise"], none, b.image, b.truth["noise"], none)
    assert abs(dp["robust_std"] - 1.0) < 0.02 and abs(dp["median"]) < 0.02


def test_negated_residual_detections_are_near_zero_on_pure_noise():
    case, _ = _oracle_case(4, shape=(512, 512), n_stars=0)
    out = M.run_oracle(case)
    bad = M.unusable_mask(case.image, case.masks)
    neg = RM.negated_detections(case.image, out.back_map(case.image.shape), out.rms_map(case.image.shape), bad,
                                out.detections, out.frame_hfr)
    assert neg["status"] == "completed" and neg["rate_per_mpx"] <= 1.0
    # An over-subtracted ring around a bright star is found and attributed to it.
    star, _ = _oracle_case(5, shape=(256, 256), n_stars=1, flux=(2e4, 2e4))
    o = M.run_oracle(star)
    yy, xx = np.mgrid[0:256, 0:256]
    s = o.detections[0]
    ring = np.exp(-0.5 * ((np.hypot(xx - s["x"], yy - s["y"]) - 5.0) / 1.5) ** 2) * 150.0
    neg = RM.negated_detections(star.image, star.truth["background"] + ring, star.truth["noise"],
                                np.zeros((256, 256), bool), o.detections, o.frame_hfr)
    assert neg["n_detections"] >= 1 and neg["fraction_near_bright"] == 1.0


# --------------------------------------------------------------------------
# (d) injection-recovery
# --------------------------------------------------------------------------


def test_injection_recovery_oracle_is_complete_and_unbiased():
    case, _ = _oracle_case(6, shape=(512, 512), n_stars=30, flux=(3e3, 2e5))
    ref = M.run_python(M.REFERENCE_METHOD, case)
    psf = INJ.estimate_psf(case.image, case.masks, ref)
    assert psf["status"] == "completed" and psf["psf"]["model"] == "moffat" and psf["n_fits"] >= 3
    distance, _n = INJ.avoidance_distance(case.image, case.masks, ref)
    orig = M.run_oracle(case)
    records = []
    for r in range(4):
        rng = np.random.default_rng(r)
        stars, img, info = INJ.plan_injections(rng, case.image, ref, psf["psf"], distance, 40)
        assert info["placed"] == 40
        INJ.true_aperture_flux(img, stars)
        inj, masks, _ = INJ.injected_image(case.image, case.masks, img, rng)
        for s in stars:  # injected stars avoid every real source
            assert distance[int(round(s["y"])), int(round(s["x"]))] > info["r_avoid_px"]
        out = M.run_oracle(M.Case(f"qi{r}", inj, masks, None, case.truth))
        records += INJ.score_round(stars, inj, out, orig)
    agg = INJ.aggregate(records)
    for b in agg["bins"]:
        if b["lo"] >= 20:
            assert b["completeness"] >= 0.97, b
    hi = agg["snr_ge_20"]
    assert abs(hi["median_rel_flux"]) < 0.01 and abs(hi["median_rel_hfr_total_flux_truth"]) < 0.02
    # The oracle measures HFR with the harness estimator (sep.flux_radius), which matches the total-flux
    # truth; against the daemon crop-estimator truth (R7) it reads a few percent high by definition.
    assert hi["n_hfr"] == hi["n_hfr_total_flux_truth"] > 0 and 0.0 < hi["median_rel_hfr"] < 0.10
    assert hi["median_centroid_px"] < 0.10
    assert agg["delta_b"]["median_sigma"] == 0.0  # the oracle's B does not move under injected stars
    # A background that absorbs the stars shows up as dB > 0 and a negative flux bias.
    absorb = M.MethodOutput(method="absorbing", status="completed", level_ok=True, noise_ok=True,
                            back=case.truth["background"] + 0.3 * img, rms=case.truth["noise"],
                            detections=out.detections, accepted=out.accepted)
    recs = INJ.score_round(stars, inj, absorb, orig)
    assert np.median([x["delta_b_sigma"] for x in recs]) > 0 and np.median([x["flux_rel"] for x in recs]) < -0.2


def test_r7_truth_is_the_daemon_crop_estimator_on_the_noiseless_stamp():
    pytest.importorskip("ml.labels.psf_measure")
    from ml.eval.bge import scenarios as SC
    from ml.labels import psf_measure as LM

    if not hasattr(LM, "rust_compute_hfr_with_sky"):
        pytest.skip("psf_measure.rust_compute_hfr_with_sky not present at this revision")
    psf = {"model": "moffat", "fwhm_px": 4.0, "beta": 3.0}
    stamp = SC.render_psf(psf, 100.3, 80.7, 5e4)
    hfr, info = INJ.crop_estimator_truth(stamp, 100.3, 80.7, 300, 200, 10.0)
    # Independent recomputation of the decision's recipe.
    above = np.argwhere(stamp.data > 50.0)
    extent = max(np.ptp(above[:, 1]) + 1, np.ptp(above[:, 0]) + 1)
    size = int(np.clip(2 * extent, 32, 128))
    x0, y0, cw, ch = LM.rust_extract_crop(300, 200, 100.3, 80.7, size)
    frame = np.zeros((200, 300))
    INJ._paste(frame, stamp.data, stamp.ix0, stamp.iy0)
    expect = LM.rust_compute_hfr_with_sky(frame[y0:y0 + ch, x0:x0 + cw].astype(np.float32), level=0.0, sigma=10.0)
    assert info["crop_px"] == size and abs(hfr - expect) < 1e-12
    total, _ = SC.numerical_hfr(stamp, 100.3, 80.7)
    assert hfr < total  # the bounded crop aperture sees less of the Moffat wings than the total-flux truth
    assert INJ.crop_estimator_truth(SC.render_psf(psf, 50.0, 50.0, 10.0), 50.0, 50.0, 300, 200, 10.0)[0] is None


def test_daemon_like_variant_is_unbiased_against_the_crop_truth_when_present():
    from ml.eval.bge import python_methods as PM

    if not isinstance(getattr(PM, "PYTHON_SEP_VARIANTS", None), dict):
        pytest.skip("python_methods has no PYTHON_SEP_VARIANTS at this revision")
    case, _ = _oracle_case(16, shape=(512, 512), n_stars=10, flux=(5e4, 2e5))
    ref = M.run_python(M.REFERENCE_METHOD, case)
    psf = INJ.estimate_psf(case.image, case.masks, ref)
    distance, _n = INJ.avoidance_distance(case.image, case.masks, ref)
    mid = "python_sep__n2g2"
    orig = M.run_python(mid, case)
    records = []
    for r in range(3):
        rng = np.random.default_rng(50 + r)
        stars, img, _info = INJ.plan_injections(rng, case.image, ref, psf["psf"], distance, 20,
                                                flux_range=(2e4, 1e5))
        INJ.true_aperture_flux(img, stars)
        inj, masks, _ = INJ.injected_image(case.image, case.masks, img, rng)
        records += INJ.score_round(stars, inj, M.run_python(mid, M.Case(f"qd{r}", inj, masks)), orig)
    hi = INJ.aggregate(records)["snr_ge_20"]
    assert hi["n_hfr"] >= 12  # the daemon uses at most 16 stars per frame (FOCUS_MAX_STARS_USED)
    assert abs(hi["median_rel_hfr"]) < 0.03  # R7: background/noise effects only
    assert hi["median_rel_hfr_total_flux_truth"] < hi["median_rel_hfr"]  # R7-obs carries the aperture definition


# --------------------------------------------------------------------------
# (e) dark + injection: exact differential truth
# --------------------------------------------------------------------------


def test_dark_injection_differential_truth_and_metrics():
    dark = T.dark_frame(7, shape=(256, 320))
    other = T.dark_frame(8, shape=(256, 320))
    sigma, how = DI.sigma_dark(dark, {}, other)
    assert how == "pair_difference" and abs(sigma - 5.0) < 0.15
    sky = DI.sky_models(320, 256, 500.0)
    assert set(sky) == {"linear_gradient", "vignetting"}
    rng = np.random.default_rng(1)
    img, truth = DI.build_trial(dark, {}, sky["linear_gradient"], DI.DEFAULT_PSF, sigma, rng)
    usable = np.ones(dark.shape, bool)
    # A method that follows the added sky exactly scores zero error.
    b_dark = np.full(dark.shape, 500.0, np.float32)
    perfect_d = M.MethodOutput("p", "completed", level_ok=True, noise_ok=True, back=b_dark, rms=5.0, detections=[],
                               accepted=[])
    perfect_t = M.MethodOutput("p", "completed", level_ok=True, noise_ok=True, back=b_dark + sky["linear_gradient"],
                               rms=5.0, detections=[], accepted=[])
    s = DI.score_trial(perfect_d, perfect_t, sky["linear_gradient"], truth, dark.shape, usable, sigma)
    assert s["background"]["rmse_adu"] < 1e-3 and abs(s["bias_sigma"]) < 1e-6
    # A constant (scalar) background cannot follow the gradient: large RMSE.
    flat_t = M.MethodOutput("f", "completed", level_ok=True, noise_ok=True,
                            back=float(np.median(b_dark + sky["linear_gradient"])), rms=5.0)
    flat_d = M.MethodOutput("f", "completed", level_ok=True, noise_ok=True, back=500.0, rms=5.0)
    s2 = DI.score_trial(flat_d, flat_t, sky["linear_gradient"], truth, dark.shape, usable, sigma)
    assert s2["background"]["rmse_adu"] > 8 * sigma
    # The real Python SEP reference on the real-noise trial: scored with the synthetic metrics.
    d_out = M.run_python(M.REFERENCE_METHOD, M.Case("qd_o", dark, {}))
    t_out = M.run_python(M.REFERENCE_METHOD, M.Case("qd_t", img, {}))
    s3 = DI.score_trial(d_out, t_out, sky["linear_gradient"], truth, dark.shape, usable, sigma)
    assert s3["status"] == "completed" and s3["background"]["rmse_adu"] < s2["background"]["rmse_adu"]
    assert s3["detection"]["n_truth"] == truth["n_stars"] and "completeness" in s3["detection"]


# --------------------------------------------------------------------------
# (f) focus
# --------------------------------------------------------------------------


def test_sweep_fit_and_cfz():
    pos = np.arange(800, 1201, 50)
    hfr = [2.0 * np.sqrt(1 + ((p - 1013.0) / 90.0) ** 2) for p in pos]
    fit = RM.fit_sweep(pos, hfr)
    assert fit["status"] == "completed" and abs(fit["p0"] - 1013.0) < 0.5
    assert fit["valid_frames_left_right"] == [5, 4]
    fit2 = RM.fit_sweep(pos[:5], hfr[:5])
    assert fit2["status"] == "failed"
    cfz = RM.cfz_steps(3.95, 2.0)
    assert abs(cfz - 4 * 0.55 * 3.95 ** 2 / 2.0) < 1e-9 and abs(cfz * 2.0 - 34.3) < 0.1
    assert RM.cfz_steps(3.95, None) is None


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------


def _night(sep=None, legacy=None, gate=None):
    base = {"availability": "available", "n_lights": 20, "status_failure_rate": 0.0,
            "pull": {"n_frames": 10, "median_abs_clipped_mean": 0.01, "median_robust_std": 0.99,
                     "median_frac_abs_gt5": 1e-6},
            "diff_pull": {"n_pairs": 5, "median_robust_std": 1.0},
            "negated": {"n_frames": 10, "rate_per_mpx": 0.5},
            "injection": {"bins": [{"bin": "10-20", "lo": 10.0, "n_injected": 500, "completeness": 0.8},
                                   {"bin": "20-50", "lo": 20.0, "n_injected": 500, "completeness": 0.99}],
                          "snr_ge_20": {"n_hfr": 300, "median_rel_hfr": -0.01, "n_flux": 300, "median_rel_flux": 0.005,
                                        "n_centroid": 300, "median_centroid_px": 0.05},
                          "delta_b": {"n": 600, "median_sigma": 0.02}},
            "dark_injection": {"trials": {"d1/lin": {"status": "completed", "bias_sigma": 0.01,
                                                     "background": {"rmse_adu": 1.0}},
                                          "d2/lin": {"status": "completed", "bias_sigma": -0.02,
                                                     "background": {"rmse_adu": 1.2}}}},
            "repeatability": {"groups": {"n/g1": {"hfr_cv": 0.03}, "n/g2": {"hfr_cv": 0.04}}},
            "focus": {"n_fits": 4, "p0_std": 5.0, "cfz_steps": 34.3,
                      "sweeps": {f"r{i}": {"status": "completed", "p0": 1000.0 + i, "rms_residual_px": 0.05}
                                 for i in range(4)},
                      "coverage": {"n_frames": 40, "fraction": 0.95}}}

    def merge(over):
        out = json.loads(json.dumps(base))
        for k, v in (over or {}).items():
            node = out
            *path, last = k.split(".")
            for p in path:
                node = node[p]
            node[last] = v
        return out

    return {"rust_sep__n2g2": merge(sep), "rust_legacy__g2": merge(legacy), "rust_legacy__g1": merge(gate)}


def test_threshold_file_encodes_the_coordinator_table():
    th = TH.load_thresholds(DEFAULT_THRESHOLDS)
    c = {x["id"]: x for x in th["criteria"]}
    assert th["variants"] == {"sep_v2": "rust_sep__n2g2", "legacy_background": "rust_legacy__g2",
                              "legacy_gate": "rust_legacy__g1"}
    assert c["R1"]["value"] == 0.05 and c["R2"]["tolerance"] == 0.07 and c["R3"]["value"] == 1e-4
    assert c["R4"]["tolerance"] == 0.10 and c["R5"]["floor_per_mpx"] == 1.0
    assert (c["R6"]["margin"], c["R6"]["min_bin_lower_edge"], c["R6"]["min_injections_per_bin"]) == (0.03, 10, 400)
    assert (c["R7"]["value"], c["R7"]["comparator_margin"], c["R7"]["min_snr"]) == (0.05, 0.01, 20)
    assert c["R8"]["value"] == 0.02 and c["R9"]["value"] == 0.10 and c["R10"]["value"] == 0.1
    assert c["R11"]["value"] == 0.05 and (c["R12"]["value"], c["R12"]["comparator_margin"]) == (0.02, 0.01)
    assert c["R13"]["ratio"] == 1.10 and (c["R14"]["ratio"], c["R14"]["max_cfz_fraction"]) == (1.10, 0.5)
    assert c["R14"]["min_data"]["sweeps"] == 4 and c["R15"]["max_cfz_fraction"] == 0.5 and c["R16"]["ratio"] == 1.10
    # Amendment (b): R17 is the pure gate change on the legacy backend; R17b is SEP v2 alone.
    assert (c["R17"]["subject"], c["R17"]["comparator"], c["R17"]["value"]) == ("legacy_background", "legacy_gate", 0.90)
    assert c["R17b"]["subject"] == "sep_v2" and "comparator" not in c["R17b"] and c["R17b"]["value"] == 0.90
    assert all(x["comparator"] == "legacy_background" for k, x in c.items() if "comparator" in x and k != "R17")
    assert all(x.get("provenance") for x in c.values())
    # Amendment (a): R7 truth is the daemon crop estimator; the total-flux bias is an observation.
    assert c["R7"]["truth_function"] == "ml.labels.psf_measure.rust_compute_hfr_with_sky"
    assert [o["id"] for o in th["observations"]] == ["R7-obs"]
    # Amendment (c): within-night statistics stated explicitly.
    assert "pooled" in c["R5"]["aggregation_within_night"] and "trials" in c["R11"]["aggregation_within_night"]
    assert "groups" in c["R13"]["aggregation_within_night"] and "sweeps" in c["R16"]["aggregation_within_night"]
    assert [a["id"] for a in th["amendments"]] == ["a", "b", "c"]


def test_thresholds_must_be_predeclared_and_complete(tmp_path):
    th = json.loads(DEFAULT_THRESHOLDS.read_text())
    p = tmp_path / "t.json"
    p.write_text(json.dumps(dict(th, predeclared_utc="2999-01-01T00:00:00+00:00")))
    with pytest.raises(TH.ThresholdsError, match="not predeclared"):
        TH.load_thresholds(p)
    p.write_text(json.dumps(dict(th, criteria=th["criteria"][:-1])))
    with pytest.raises(TH.ThresholdsError, match="exactly"):
        TH.load_thresholds(p)


def test_threshold_evaluation_pass_fail_and_not_evaluable():
    th = TH.load_thresholds(DEFAULT_THRESHOLDS)
    s0 = {"rust_sep__n2g2": {"value": 0.98, "calibrated": True}}
    res = TH.evaluate(th, {"2026-11-15": _night()}, s0)
    assert all(res[k]["status"] == "pass" for k in TH.CRITERIA), {k: v["per_night"] for k, v in res.items()
                                                                   if v["status"] != "pass"}
    assert "1 evaluable night: margin claim only" in res["R1"]["statistical_power"]
    bad = _night(sep={"pull.median_abs_clipped_mean": 0.08, "pull.median_robust_std": 1.10,
                      "negated.rate_per_mpx": 2.0, "status_failure_rate": 0.05, "focus.p0_std": 20.0,
                      "focus.coverage.fraction": 0.85})
    res = TH.evaluate(th, {"n1": bad}, s0)
    for k in ("R1", "R2", "R5", "R12", "R14", "R17b"):
        assert res[k]["status"] == "fail", k
    assert res["R17"]["status"] == "pass"  # the legacy gate pair is untouched by SEP v2's coverage
    res = TH.evaluate(th, {"n1": _night(legacy={"focus.coverage.fraction": 0.80})}, s0)
    assert res["R17"]["status"] == "fail" and res["R17"]["subject"] == "rust_legacy__g2"
    assert res["R17"]["comparator"] == "rust_legacy__g1" and res["R17b"]["status"] == "pass"
    # R5 compares with max(1 per Mpx, legacy): a worse legacy lifts the bound.
    res = TH.evaluate(th, {"n1": _night(sep={"negated.rate_per_mpx": 2.0}, legacy={"negated.rate_per_mpx": 3.0})}, s0)
    assert res["R5"]["status"] == "pass"
    # Not evaluable: too few injections, too few sweeps, no repeat groups, unavailable comparator, s0 flag.
    thin = _night(sep={"injection.bins": [{"bin": "10-20", "lo": 10.0, "n_injected": 120, "completeness": 0.8}],
                       "focus.n_fits": 3, "repeatability.groups": {}})
    thin_legacy = {"availability": "method_unavailable: Rust variants need --rust"}
    res = TH.evaluate(th, {"n1": dict(thin, rust_legacy__g2=thin_legacy)}, {})
    assert res["R6"]["status"] == "not_evaluable" and "comparator" in res["R6"]["per_night"]["n1"]["reason"]
    res = TH.evaluate(th, {"n1": thin}, {})
    assert res["R6"]["status"] == "not_evaluable" and "120 injections < 400" in res["R6"]["per_night"]["n1"]["reason"]
    assert res["R14"]["status"] == "not_evaluable" and res["R13"]["status"] == "not_evaluable"
    assert res["R2"]["per_night"]["n1"]["values"]["flag"].startswith("s0_uncalibrated")
    assert "no evaluable hold-out night" in res["R6"]["statistical_power"]
    # Two nights: one failing night fails the criterion; a not-evaluable night does not rescue it.
    res = TH.evaluate(th, {"a": _night(), "b": bad}, s0)
    assert res["R1"]["status"] == "fail" and res["R1"]["n_nights_evaluable"] == 2


def test_paired_comparisons_and_bootstrap_need_enough_units():
    nm = {f"n{i}": _night(sep={"pull.median_robust_std": 1.0 + 0.01 * i}) for i in range(6)}
    out = TH.paired_comparisons(nm, ["rust_sep__n2g2", "rust_legacy__g2", "rust_legacy__g1"], "rust_legacy__g2",
                                "rust_legacy__g1")
    row = out["rust_sep__n2g2"]["pull.median_robust_std"]
    assert row["n_units"] == 6 and row["bootstrap_ci95_median_difference"] is not None
    assert out["rust_legacy__g2"]["focus.coverage.fraction"]["comparator"] == "rust_legacy__g1"
    few = TH.paired_comparisons({"n1": _night()}, ["rust_sep__n2g2", "rust_legacy__g2"], "rust_legacy__g2")
    assert "too few" in few["rust_sep__n2g2"]["pull.median_robust_std"]["ci_note"]
    assert ST.bootstrap_ci([1.0, 2.0])[0] is None


# --------------------------------------------------------------------------
# Hold-out guard and ledger (git mocked)
# --------------------------------------------------------------------------


def _git(tracked=True, dirty_paths=(), head="a" * 40):
    def fn(args, repo):
        if args[0] == "ls-files":
            return (0 if tracked else 1), ""
        if args[0] == "status":
            paths = args[3:]
            hits = [f" M {d}" for d in dirty_paths if any(d.startswith(p) for p in paths)]
            return 0, "\n".join(hits)
        if args[0] == "rev-parse":
            return 0, head + "\n"
        raise AssertionError(args)
    return fn


def _guard(tmp_path, git_fn, **kw):
    args = dict(thresholds_path=DEFAULT_THRESHOLDS, split_rule_path=DEFAULT_SPLIT_RULE,
                ledger_path=tmp_path / "ledger.jsonl", manifest_sha="m" * 64, config_hashes={"rust_sep__n2g2": "c"},
                holdout_nights=["2026-11-15"], split_problems=[], git_fn=git_fn)
    args.update(kw)
    return HO.guard(**args)


def test_holdout_guard_refusals_and_ledger(tmp_path):
    with pytest.raises(HO.HoldoutRefusal, match="not tracked"):
        _guard(tmp_path, _git(tracked=False))
    rel = DEFAULT_THRESHOLDS.relative_to(REPO_ROOT).as_posix()
    with pytest.raises(HO.HoldoutRefusal, match="differs from HEAD"):
        _guard(tmp_path, _git(dirty_paths=[rel]))
    with pytest.raises(HO.HoldoutRefusal, match="code is not frozen"):
        _guard(tmp_path, _git(dirty_paths=["ml/eval/bge/python_methods.py"]))
    with pytest.raises(HO.HoldoutRefusal, match="split rule"):
        _guard(tmp_path, _git(), split_problems=["night x: manifest says development, rule says holdout"])
    entry = _guard(tmp_path, _git())
    assert entry["event"] == "started" and not entry["repeat"] and entry["git_head"] == "a" * 40
    with pytest.raises(HO.HoldoutRefusal, match="already consumed"):
        _guard(tmp_path, _git())
    with pytest.raises(HO.HoldoutRefusal, match="consumed hold-out night"):  # changed code after the hold-out
        _guard(tmp_path, _git(head="b" * 40))
    again = _guard(tmp_path, _git(), acknowledge_repeat=True)
    assert again["repeat"] and again["acknowledge_repeat"] and again["previous_same_key"] == 1
    HO.complete(tmp_path / "ledger.jsonl", again, "r" * 64, {"R1": "pass"})
    events = [e["event"] for e in HO.read_ledger(tmp_path / "ledger.jsonl")]
    assert events == ["started", "started", "completed"]


def test_ledger_exemption_only_covers_the_ledger():
    # The ledger itself may be dirty (every hold-out run appends to it); nothing else may.
    ledger_rel = "ml/eval/bge/qualification/holdout_ledger.jsonl"
    HO.check_code_clean(HO.CODE_PATHS, REPO_ROOT / ledger_rel, git_fn=_git(dirty_paths=[ledger_rel]))
    with pytest.raises(HO.HoldoutRefusal):
        HO.check_code_clean(HO.CODE_PATHS, REPO_ROOT / ledger_rel,
                            git_fn=_git(dirty_paths=[ledger_rel, "ml/eval/bge/qualification/engine.py"]))


# --------------------------------------------------------------------------
# Runner end to end (development and hold-out) on a synthetic campaign
# --------------------------------------------------------------------------


def _campaign(tmp_path, n_nights=4):
    """Four nights of synthetic frames: 3-frame repeat group, a 9-position sweep, 2 darks, 1 bias."""
    frames_dir = tmp_path / "frames"
    rows = ["file,frame_type,timestamp_utc,analysis_representation,repeat_group,focus_run_id,focus_position,"
            "focuser_stepsize_um,focal_ratio,rig_id,camera,optical_train,iso,exposure_s"]
    truth = {}
    k = 0
    for n in range(n_nights):
        t0 = dt.datetime(2026, 11, 15 + 2 * n, 2, 0, tzinfo=dt.timezone.utc)

        def add(name, image, ftype, ts, extra="", b=None, r=None):
            T.write_fits(frames_dir / name, image)
            if b is not None:
                T.write_fits(frames_dir / f"truth/{name}.b.fits", b)
                T.write_fits(frames_dir / f"truth/{name}.r.fits", r)
                truth[name] = (f"truth/{name}.b.fits", f"truth/{name}.r.fits")
            rows.append(f"{name},{ftype},{ts.isoformat()},{REP},{extra},z114,Canon EOS R50,native,1600,2")

        for i in range(3):
            k += 1
            img, b, r, _ = T.sky_frame(100 + k, shape=(192, 256), n_stars=10)
            add(f"n{n}_rg_{i}.fits", img, "light", t0 + dt.timedelta(seconds=30 * i), f"g{n},,,,", b, r)
        for j, p in enumerate(range(800, 1201, 50)):
            k += 1
            img, b, r, _ = T.sky_frame(100 + k, shape=(192, 256), n_stars=8, psf_sigma=T.focus_hfr_sigma(p),
                                       flux=(4e4, 2e5))
            add(f"n{n}_sw_{j}.fits", img, "light", t0 + dt.timedelta(minutes=5, seconds=20 * j),
                f",run{n},{p},2.0,3.95", b, r)
        for d in range(2):
            add(f"n{n}_dark_{d}.fits", T.dark_frame(500 + 10 * n + d), "dark",
                t0 + dt.timedelta(minutes=30, seconds=d), ",,,,")
        add(f"n{n}_bias.fits", T.dark_frame(600 + n, sigma=3.0, hot=()), "bias", t0 + dt.timedelta(minutes=31), ",,,,")
    (tmp_path / "log.csv").write_text("\n".join(rows) + "\n")
    m = B.build_from_directory(frames_dir, SidecarLog.read(tmp_path / "log.csv").with_defaults(
        {"site_utc_offset": "-05:00"}))  # host-timezone independent nights
    for f in m["frames"]:
        name = f["provenance"]["relative_path"]
        if name in truth:
            f["truth"] = {"background": str(frames_dir / truth[name][0]), "noise": str(frames_dir / truth[name][1])}
    path = tmp_path / "manifest.json"
    B.write_manifest(m, path)
    return path, m


def _args(tmp_path, manifest, split, **kw):
    a = RQ.parse_args(["--manifest", str(manifest), "--split", split, "--out", str(tmp_path / f"out_{split}"),
                       "--methods", "oracle_true_background,python_sep__n1g1", "--thresholds", str(DEFAULT_THRESHOLDS),
                       "--ledger", str(tmp_path / "ledger.jsonl"), "--injections-per-round", "8",
                       "--injection-rounds", "1", "--dark-trials", "1"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_runner_development_and_holdout_end_to_end(tmp_path):
    manifest, m = _campaign(tmp_path)
    assert [n["split"] for n in m["nights"]] == ["development", "holdout", "development", "holdout"]
    roles = ["sep_v2=oracle_true_background", "legacy_background=python_sep__n1g1", "legacy_gate=python_sep__n1g1"]
    dev = RQ.run(_args(tmp_path, manifest, "development", role=roles), log=lambda m: None)
    assert dev["status"] == "development_observations_with_informational_criteria"
    assert dev["manifest"]["evaluated_nights"] == ["2026-11-14", "2026-11-18"]
    night = dev["nights"]["2026-11-14"]["methods"]
    o = night["oracle_true_background"]
    assert o["n_lights"] == 12 and o["status_failure_rate"] == 0.0
    assert abs(o["pull"]["median_robust_std"] - 1.0) < 0.03 and o["pull"]["median_abs_clipped_mean"] < 0.03
    assert o["diff_pull"]["n_pairs"] == 2 and abs(o["diff_pull"]["median_robust_std"] - 1.0) < 0.05
    assert o["focus"]["n_fits"] == 1 and abs(o["focus"]["sweeps"]["run0"]["p0"] - 1000.0) < 25
    assert abs(o["focus"]["cfz_steps"] - 4 * 0.55 * 3.95 ** 2 / 2.0) < 1e-9
    assert o["injection"]["n_injected"] > 0
    py = night["python_sep__n1g1"]
    assert py["dark_injection"]["n_trials"] == 2 and py["calibration_pulls"]
    assert dev["s0"]["python_sep__n1g1"]["calibrated"] and dev["s0"]["python_sep__n1g1"]["n_frames"] == 6
    assert dev["s0"]["oracle_true_background"]["calibrated"] is False  # no truth for darks: s0 = 1.0, flagged
    c = dev["criteria"]
    assert set(c) == set(TH.CRITERIA) and "R7-obs" in dev["observations"]
    assert c["R14"]["status"] == "not_evaluable"  # 1 sweep per night < 4
    assert c["R6"]["status"] == "not_evaluable"  # far fewer than 400 injections per bin
    assert c["R1"]["status"] in ("pass", "fail") and c["R1"]["n_nights_evaluable"] == 2
    assert (tmp_path / "out_development" / "summary.md").exists()
    rows = [json.loads(x) for x in (tmp_path / "out_development" / "frames.jsonl").read_text().splitlines()]
    assert {r["method"] for r in rows} == {"oracle_true_background", "python_sep__n1g1"}
    # Hold-out: role overrides are development-only.
    with pytest.raises(HO.HoldoutRefusal, match="development-only"):
        RQ.run(_args(tmp_path, manifest, "holdout", role=roles), git_fn=_git(), log=lambda m: None)
    # Without --rust the predeclared Rust roles are unavailable: every criterion is not evaluable, honestly.
    ho = RQ.run(_args(tmp_path, manifest, "holdout"), git_fn=_git(), log=lambda m: None)
    assert ho["status"] == "holdout_evaluated" and ho["manifest"]["evaluated_nights"] == ["2026-11-16", "2026-11-20"]
    assert all(v["status"] == "not_evaluable" for v in ho["criteria"].values())
    assert ho["ledger_entry"]["holdout_nights"] == ["2026-11-16", "2026-11-20"]
    assert [e["event"] for e in HO.read_ledger(tmp_path / "ledger.jsonl")] == ["started", "completed"]
    assert ho["s0"]["python_sep__n1g1"]["calibrated"]  # from development darks/bias only
    with pytest.raises(HO.HoldoutRefusal, match="already consumed"):
        RQ.run(_args(tmp_path, manifest, "holdout"), git_fn=_git(), log=lambda m: None)
    # A manifest whose recorded split disagrees with the rule is refused for the hold-out.
    bad = json.loads(manifest.read_text())
    for f in bad["frames"]:
        if f["night_id"] == "2026-11-16":
            f["split"] = "development"
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(HO.HoldoutRefusal, match="split rule"):
        RQ.run(_args(tmp_path, tmp_path / "bad.json", "holdout", ledger=tmp_path / "l2.jsonl"), git_fn=_git(),
               log=lambda m: None)


def test_runner_with_mocked_rust_bridge_evaluates_predeclared_roles(tmp_path):
    from test_bge_qualification_builders import _mock_bridge

    manifest, _ = _campaign(tmp_path, n_nights=2)
    run_bridge, calls = _mock_bridge()
    rust = M.RustBridgeRunner(tmp_path, tmp_path / "io", tmp_path / "t", run_bridge=run_bridge)
    methods = ["rust_legacy__g1", "rust_legacy__g2", "rust_sep__n2g2"]
    suite = M.MethodSuite(methods, rust_runner=rust)
    args = _args(tmp_path, manifest, "holdout", methods=",".join(methods), rust=True)
    res = RQ.run(args, suite=suite, git_fn=_git(), log=lambda m: None)
    night = res["nights"]["2026-11-16"]["methods"]
    assert night["rust_sep__n2g2"]["availability"] == "available" and night["rust_sep__n2g2"]["n_lights"] == 12
    assert res["criteria"]["R12"]["status"] == "pass"  # mocked outputs always measure level and noise
    assert res["criteria"]["R14"]["status"] == "not_evaluable"
    assert {c["backends"][0] for c in calls} == {"legacy_global_clip", "sep"}
    assert not any((tmp_path / "io").iterdir())  # bridge inputs/outputs removed after each frame


def test_real_frames_adapter_still_reads_v2_manifests(tmp_path):
    from ml.eval.bge import real_frames as RF

    manifest, _ = _campaign(tmp_path, n_nights=2)
    res = RF.evaluate(manifest)
    assert res["status"] == "observations_only" and res["n_frames_evaluated"] == 12
