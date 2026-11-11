"""BGE v2 harness: held-out and development generators (rendering and truth
bookkeeping only), the held-out guard, the method matrix and bridge
plumbing (mocked Rust outputs), acceptance metrics and the evaluator.

Held-out (ho_*) frames are rendered here only to check pixels and truth; no
gate, background or benchmark runs on them (docs/bge-v2-spec.md §6): the
SEP entry points are patched to fail while they are built.
"""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from ml.eval.bge import bridge
from ml.eval.bge import metrics as MT
from ml.eval.bge import python_methods as PM
from ml.eval.bge import run_benchmark as RB
from ml.eval.bge import scenarios as SC
from ml.eval.bge.acceptance import evaluate as EV
from ml.labels import psf_measure as pm

REPO = Path(__file__).resolve().parents[2]
ACCEPTANCE = REPO / "ml" / "eval" / "bge" / "acceptance" / "synthetic-v2.json"


def _no_sep(*_a, **_k):
    raise AssertionError("SEP must not run on held-out frames in tests")


@pytest.fixture(scope="module")
def holdout():
    """Every held-out set and sweep, rendered with SEP disabled."""
    with mock.patch.object(pm, "require_sep", _no_sep), mock.patch.object(PM, "sep_module", _no_sep):
        sets = {k: f() for k, f in SC.HOLDOUT_SETS.items()}
        sweeps = {k: f() for k, f in SC.HOLDOUT_SWEEPS.items()}
    return sets, sweeps


# --------------------------------------------------------------------------
# Seeds, ids and splits
# --------------------------------------------------------------------------


def test_holdout_ids_match_the_committed_criteria_and_seeds_are_disjoint():
    criteria = json.loads(ACCEPTANCE.read_text())
    assert sorted(SC.HOLDOUT_IDS) == sorted(criteria["heldout_scenarios"])
    ho, dev = set(SC.HOLDOUT_SEEDS.values()), set(SC.DEV_V2_SEEDS.values())
    assert len(ho) == len(SC.HOLDOUT_SEEDS) and len(dev) == len(SC.DEV_V2_SEEDS)
    assert not ho & dev and not ho & set(SC.PRE_EXISTING_SEEDS) and not dev & set(SC.PRE_EXISTING_SEEDS)
    used = {SC.SCENARIOS[n].__defaults__[0] for n in SC.SCENARIOS} | {SC.SWEEPS[n].__defaults__[0] for n in SC.SWEEPS}
    assert used <= set(SC.PRE_EXISTING_SEEDS) | {301, 401}
    assert set(SC.PRE_EXISTING_SCENARIOS) == set(SC.SCENARIOS) and len(SC.PRE_EXISTING_SCENARIOS) == 22


def test_holdout_frames_are_marked_and_bridge_safe(holdout):
    sets, sweeps = holdout
    for sid, st in sets.items():
        assert st.split == "holdout" and st.frames
        for fr in st.frames:
            assert fr.id.startswith(sid) and bridge.valid_case_id(fr.id)
            assert fr.metadata["split"] == "holdout" and fr.metadata["scenario_set"] == sid
            assert not fr.metadata["pre_existing"]
    for sid, sw in sweeps.items():
        assert sw.metadata["split"] == "holdout" and sw.metadata["step"] == 100
        assert all(bridge.valid_case_id(fr.id) for fr in sw.frames)


def test_spider_donut_truth_and_geometry(holdout):
    st = holdout[0]["ho_newtonian_spider_donuts"]
    vanes = {fr.metadata["params"]["n_vanes"] for fr in st.frames}
    assert vanes == {3, 4}
    r_outs = []
    for fr in st.frames:
        p = fr.metadata["params"]
        assert 0.25 <= p["obstruction"] <= 0.35 and 0.005 <= p["vane_width_frac"] <= 0.015
        assert 1.0 <= p["sigma_seeing_px"] <= 1.8 and 6.0 <= p["r_out_px"] <= 32.0
        assert len(fr.sources) <= SC.GATE_MAX_STARS_USED
        angles = {s.psf["vane_angle_rad"] for s in fr.sources}
        assert len(angles) == 1  # one vane angle per frame
        for s in fr.sources:
            t = s.to_json()
            assert t["r_out_px"] == p["r_out_px"] and t["vane_width_frac"] == p["vane_width_frac"]
            assert t["n_vanes"] == p["n_vanes"] and t["isolated"] is True
            assert 5e4 <= s.flux <= 1e6 and s.flux_rendered == pytest.approx(s.flux, rel=1e-6)
            assert s.hfr_true_px > 0
        r_outs.append(p["r_out_px"])
    assert min(r_outs) < 10 and max(r_outs) > 28  # stratified over 6-32 px


def test_pupil_shadow_area_matches_the_vane_geometry():
    for r_out, obstruction, n_vanes, frac in ((20.0, 0.3, 4, 0.03), (12.0, 0.25, 3, 0.01), (8.0, 0.0, 0, 0.0)):
        cover, *_ = SC.pupil_shadow_coverage(50.3, 50.7, r_out, obstruction, 1.0, n_vanes, frac, 0.4)
        w_img = frac * 2.0 * r_out
        expected = math.pi * r_out ** 2 * (1 - obstruction ** 2) - n_vanes * w_img * r_out * (1 - obstruction)
        assert cover.sum() == pytest.approx(expected, rel=2e-3)
    stamp = SC.render_pupil_shadow(40.0, 40.0, 15.0, 0.3, 1.2, 1e5, n_vanes=4, vane_width_frac=0.05)
    assert stamp.data.sum() == pytest.approx(1e5, rel=1e-6)
    # On a vane (angle 0, +x) the ring is darker than half-way between vanes.
    iy, ix = 40 - stamp.iy0, 50 - stamp.ix0
    on_vane = stamp.data[iy, ix]
    off = stamp.data[int(round(40 + 10 * math.sin(math.pi / 4))) - stamp.iy0,
                     int(round(40 + 10 * math.cos(math.pi / 4))) - stamp.ix0]
    assert on_vane < 0.8 * off


def test_refractor_comatic_doubles_sweeps_ridges_truth(holdout):
    sets, sweeps = holdout
    for fr in sets["ho_refractor_disks"].frames:
        assert fr.metadata["params"]["obstruction"] == 0.0 and 6.0 <= fr.metadata["params"]["r_out_px"] <= 24.0
    for fr in sets["ho_comatic_donuts"].frames:
        for s in fr.sources:
            assert s.psf["modulation"] == 0.4 and s.to_json()["hole_offset_frac"] <= 0.2
            assert s.to_json()["hfr_centre"] == "flux_centroid"
    pairs = 0
    for fr in sets["ho_close_doubles"].frames:
        assert 3.0 <= fr.metadata["params"]["psf"]["fwhm_px"] <= 6.0
        singles = [s for s in fr.sources if s.extra["role"] == "single"]
        members = [s for s in fr.sources if s.extra["role"] != "single"]
        assert len(singles) + len(members) // 2 <= SC.GATE_MAX_STARS_USED
        for s in members:
            assert 3.0 <= s.extra["separation_px"] <= 10.0 and 0.2 <= s.extra["flux_ratio"] <= 1.0
        pairs += len(members) // 2
        by_pair = {}
        for s in members:
            by_pair.setdefault(s.extra["pair_id"], []).append(s)
        for pair in by_pair.values():
            a, b = pair
            assert math.hypot(a.x - b.x, a.y - b.y) == pytest.approx(a.extra["separation_px"])
            assert b.flux / a.flux == pytest.approx(a.extra["flux_ratio"])
    assert pairs == 60
    ridge = sets["ho_amp_glow_ridge"].frames
    assert sum(fr.metadata["starless"] for fr in ridge) == 6 and len(ridge) == 12
    assert all(150 <= fr.metadata["params"]["ridge_amplitude_adu"] <= 900 for fr in ridge)
    assert all(not fr.sources for fr in ridge if fr.metadata["starless"])
    sw = sweeps["ho_sweep_newtonian"]
    assert sw.p0_true not in sw.positions and min(sw.positions) < sw.p0_true < max(sw.positions)
    assert all(s.psf["n_vanes"] == 4 for fr in sw.frames for s in fr.sources)
    wings = sweeps["ho_sweep_defocused_wings"]
    assert wings.metadata["seed"] not in SC.PRE_EXISTING_SEEDS and len(wings.frames) == 13
    dense = sets["ho_dense_defocused_field"]
    assert all(len(fr.sources) == 80 for fr in dense.frames)


def test_dev_variants_are_development_split():
    st = SC.DEV_V2_SETS["dev_close_doubles"]()
    assert st.split == "dev" and all(fr.metadata["split"] == "dev" for fr in st.frames)
    assert SC.DEV_V2_SWEEPS["dev_sweep_newtonian"]().metadata["split"] == "dev"


# --------------------------------------------------------------------------
# Held-out guard and splits
# --------------------------------------------------------------------------


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(cwd), "PATH": "/usr/bin:/bin:/usr/local/bin"})


def test_holdout_guard_requires_a_tracked_unchanged_criteria_file(tmp_path):
    rel = RB.ACCEPTANCE_RELPATH
    _git(tmp_path, "init", "-q")
    target = tmp_path / rel
    target.parent.mkdir(parents=True)
    target.write_text('{"schema": "x"}')
    untracked = RB.holdout_guard(tmp_path)
    assert not untracked["ok"] and "not tracked" in untracked["reason"]
    _git(tmp_path, "add", rel)
    _git(tmp_path, "commit", "-q", "-m", "criteria")
    ok = RB.holdout_guard(tmp_path)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    assert ok["ok"] and ok["commit"] == head
    assert ok["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    target.write_text('{"schema": "y"}')
    changed = RB.holdout_guard(tmp_path)
    assert not changed["ok"] and "differs from HEAD" in changed["reason"]


def test_holdout_split_refuses_before_building_any_holdout_frame(tmp_path):
    def forbidden():
        raise AssertionError("held-out frame built despite a failed guard")

    refuse = {"ok": False, "reason": "criteria modified", "file": RB.ACCEPTANCE_RELPATH}
    with mock.patch.object(RB, "holdout_guard", return_value=refuse), \
            mock.patch.dict(SC.HOLDOUT_SETS, {k: forbidden for k in SC.HOLDOUT_SETS}), \
            mock.patch.dict(SC.HOLDOUT_SWEEPS, {k: forbidden for k in SC.HOLDOUT_SWEEPS}):
        assert RB.main(["--out", str(tmp_path / "run"), "--split", "holdout"]) == 2
    assert not (tmp_path / "run" / "results.json").exists()


def test_dev_split_never_names_holdout_scenarios():
    args = argparse.Namespace(split="dev", scenarios=["ho_close_doubles"], holdout_only=False)
    with pytest.raises(SystemExit, match="held-out"):
        RB.build_cases(args)
    args = argparse.Namespace(split="dev", scenarios=["dev_close_doubles", "tiny_frame"], holdout_only=False)
    core, sweeps, sets = RB.build_cases(args)
    assert [sc.id for sc in core] == ["tiny_frame"] and [s.id for s in sets] == ["dev_close_doubles"]


# --------------------------------------------------------------------------
# Method matrix and bridge plumbing
# --------------------------------------------------------------------------


def test_method_resolution_and_matrix():
    assert RB.resolve_methods(None) == list(RB.DEFAULT_METHODS) and len(RB.DEFAULT_METHODS) == 5
    matrix = RB.resolve_methods(["matrix"])
    assert len(matrix) == 13 and matrix[3:7] == ["python_sep__n1g1", "python_sep__n2g1", "python_sep__n1g2",
                                                  "python_sep__n2g2"]
    assert RB.resolve_methods(["python_sep", "python_variants", "python_sep"])[:2] == ["python_sep", "python_sep__n1g1"]
    with pytest.raises(SystemExit):
        RB.resolve_methods(["rust_sep__n3g1"])
    assert PM.sep_background_config_for("frame_wide_v1") == PM.sep_background_config_json()
    assert pm.config_sha256(PM.sep_background_config_for("masked_local_v2")) == \
        "97bd28741f79a184aa55d8e64e8f368ffd1eb71e07b2338cc0b99ed60832a6f4"
    assert PM.variant_star_gates("rust_legacy__g2") == pm.StarGateParams()
    assert PM.variant_star_gates("python_sep__n2g1").config_sha256() == pm.StarGateParams(
        blend_test="peak_count_v1", regroup_deblended=False).config_sha256()


def test_bridge_manifests_use_method_ids_never_plain_backends(tmp_path):
    sc = SC.build_scenario("tiny_frame")
    cases = bridge.write_inputs([sc], tmp_path)
    path = bridge.write_methods_manifest(cases, tmp_path, ["rust_sep", "rust_sep__n1g1", "rust_legacy_global_clip",
                                                           "rust_sep__n2g2"], settings={"sep": {"bw": 32}})
    m = json.loads(path.read_text())
    assert m["backends"] == ["rust_sep__n1g1", "rust_legacy__g1", "rust_sep__n2g2"]
    assert m["settings"] == {"sep": {"bw": 32}} and "star_gates" not in m
    one = json.loads(bridge.write_variant_manifest(cases, tmp_path, "rust_sep", settings={"sep": {"bw": 32}})
                     .read_text())
    assert one["backends"] == ["rust_sep__n1g1"]
    assert one["settings"] == {"sep": {"bw": 32, "noise_policy": "frame_wide_v1"}}
    assert one["star_gates"] == {"blend_test": "peak_count_v1", "regroup_deblended": False}
    assert bridge.backend_of("rust_legacy__g2") == "legacy_global_clip" and bridge.backend_of("sep") == "sep"
    assert bridge.rust_variant("rust_sep")[0] == "rust_sep__n1g1"


def _python_row(sc, method):
    rec, art = RB.run_python(sc, method)
    return rec, art


def _mock_rust_output(out_dir, sc, method, py_rec, py_art, refinement=None, tallies=None, used=None):
    payload = {
        "case": sc.id, "backend": method, "error": None,
        "background": {"method": "sep_spatial_mesh", "level_status": py_rec["background_result"]["level_status"],
                       "noise_status": py_rec["background_result"]["noise_status"],
                       "reasons": py_rec["background_result"]["reasons"], "level": 1000.0, "rms": 10.0,
                       "config": {}, "config_sha256": "x", "refinement": refinement,
                       "support": {"usable_pixels": py_art["usable_pixels"]}},
        "detection": {"method": "sep_extract", "noise_floor_applied": False},
        "catalog": py_art["candidates"], "candidates": py_art["focus_candidates"],
        "quality": {"status": "Valid", "hfr_px": py_rec["frame_hfr_median_px"],
                    "star_count_detected": len(py_art["focus_candidates"]),
                    "star_count_used": py_rec["n_accepted"],
                    "star_rejections": tallies if tallies is not None else py_rec["rejections"],
                    "gate_diagnostics": py_rec["gate_diagnostics"],
                    "star_gates_config_sha256": py_rec["star_gates_config_sha256"],
                    "used_stars": used if used is not None else [
                        {"x": a["x"], "y": a["y"], "hfr_px": a["hfr_px"]} for a in py_rec["accepted_stars"]]},
        "trailing": {}, "timing_ms": {"background": 1.0, "quality": 1.0, "trailing": 1.0},
    }
    stem = bridge.output_stem(out_dir, sc.id, method)
    stem.with_suffix(".json").write_text(json.dumps(payload))
    py_art["back"].astype("<f4").tofile(str(stem) + ".back.f32")
    py_art["rms"].astype("<f4").tofile(str(stem) + ".rms.f32")


def test_mocked_rust_variant_rows_and_v2_parity(tmp_path):
    PM.sep_module()
    sc = SC.build_scenario("sparse_gaussian_stars")
    py_rec, py_art = _python_row(sc, "python_sep__n2g2")
    assert py_rec["n_accepted"] == 16 and py_rec["refinement"]["pass2_used"]
    _mock_rust_output(tmp_path, sc, "rust_sep__n2g2", py_rec, py_art, refinement=py_rec["refinement"])
    rs_rec, rs_art = RB.run_rust_parse(sc, tmp_path, "rust_sep__n2g2")
    assert rs_rec["status"] == "completed" and rs_rec["refinement"] == py_rec["refinement"]
    assert rs_rec["gate_diagnostics"] == py_rec["gate_diagnostics"] and len(rs_rec["accepted_stars"]) == 16
    assert rs_rec["metrics"]["detection"]["n_candidates"] == len(py_art["candidates"])  # scored on the catalog
    agree = RB.parity_v2(py_rec, rs_rec, py_art, rs_art)
    assert agree["all_agree"] and agree["accepted"]["max_position_diff_px"] == 0.0
    # invalid_rms_pixels may differ by up to 0.1% of usable pixels, no more.
    usable = py_art["usable_pixels"]
    near = dict(py_rec["refinement"], invalid_rms_pixels=int(0.001 * usable))
    far = dict(py_rec["refinement"], invalid_rms_pixels=int(0.001 * usable) + 1)
    for refinement, expected in ((near, True), (far, False)):
        _mock_rust_output(tmp_path, sc, "rust_sep__n2g2", py_rec, py_art, refinement=refinement)
        rs_rec, rs_art = RB.run_rust_parse(sc, tmp_path, "rust_sep__n2g2")
        assert RB.parity_v2(py_rec, rs_rec, py_art, rs_art)["refinement_agree"] is expected
    _mock_rust_output(tmp_path, sc, "rust_sep__n2g2", py_rec, py_art, refinement=py_rec["refinement"],
                      tallies=dict(py_rec["rejections"], blended=1), used=[])
    rs_rec, rs_art = RB.run_rust_parse(sc, tmp_path, "rust_sep__n2g2")
    bad = RB.parity_v2(py_rec, rs_rec, py_art, rs_art)
    assert not bad["all_agree"] and bad["tally_mismatches"] == {"blended": [0, 1]} and not bad["accepted"]["agree"]
    tally = RB.parity_v2_tally({"a": agree, "b": bad})
    assert tally["all_agree"] == 1 and tally["disagreeing_cases"] == ["b"]


def test_sweep_summary_reports_noise_and_step_fraction():
    recs = [{"status": "completed", "frame_hfr_median_px": h, "stages": {"hfr": {"status": "completed"}},
             "background_result": {"noise_status": "measured" if k != 3 else "unresolved"},
             "metrics": {"hfr": {"matched_truth_ids": [1]}}}
            for k, h in enumerate(MT.hyperbola(np.arange(4000, 5201, 100), 3.0, 4630.0, 120.0))]
    entry = RB.summarize_sweep_method(recs, list(range(4000, 5201, 100)), 4630.0, 6, [1.0] * 13, step=100.0)
    assert entry["frames_noise_measured"] == 12 and entry["step"] == 100.0
    assert abs(entry["hyperbola_p0_shift_fraction_of_step"]) < 1e-3
    assert entry["valid_frames_left_right"] == [7, 6]


# --------------------------------------------------------------------------
# Acceptance metrics (hand-computed)
# --------------------------------------------------------------------------


def _t(sid, x, y, **extra):
    return dict({"id": sid, "x": x, "y": y, "kind": "star", "hfr_true_px": 4.0}, **extra)


def test_acceptance_metrics_hand_computed():
    truth = [_t(0, 10, 10, r_out_px=8.0, isolated=True), _t(1, 50, 50, r_out_px=5.0, isolated=True),
             _t(2, 90, 90, r_out_px=12.0, isolated=False), _t(3, 130, 130, r_out_px=20.0, isolated=True)]
    accepted = [{"x": 10.5, "y": 10.0, "hfr_px": 4.4}, {"x": 90.0, "y": 90.0, "hfr_px": 3.8},
                {"x": 200.0, "y": 200.0, "hfr_px": 3.0}]
    frames = [{"truth": truth, "accepted": accepted, "detection_ran": True},
              {"truth": [_t(10, 10, 10, r_out_px=9.0, isolated=True)], "accepted": None, "detection_ran": False}]
    rate = MT.acceptance_rate(frames, lambda s: s["isolated"] and s["r_out_px"] >= 6)
    assert (rate["n_population"], rate["n_accepted"], rate["frames_without_detection"]) == (3, 1, 1)
    bias = MT.accepted_rel_hfr_bias(frames)
    assert bias["n_matched"] == 2 and bias["median_rel_bias"] == pytest.approx((0.1 - 0.05) / 2)
    pair = {"pair_id": 0, "separation_px": 6.0, "flux_ratio": 0.8, "pair_mid_x": 100.0, "pair_mid_y": 100.0}
    weak = {"pair_id": 1, "separation_px": 4.0, "flux_ratio": 0.9, "pair_mid_x": 300.0, "pair_mid_y": 100.0}
    pairs = [_t(0, 97, 100, **pair), _t(1, 103, 100, **pair), _t(2, 298, 100, **weak), _t(3, 302, 100, **weak)]
    accepted_pair = [{"x": 104.0, "y": 100.5, "hfr_px": 2.0}]  # within 3 + 2 px of the midpoint
    both = MT.close_double_rejection([{"truth": pairs, "accepted": accepted_pair}], 5.0, 0.5)
    assert (both["n_pairs"], both["n_rejected"]) == (1, 0)
    assert MT.close_double_rejection([{"truth": pairs, "accepted": []}], 0.0, 0.0)["rate"] == 1.0
    fd = MT.false_detections_per_frame([{"false_detections": 3, "detection_ran": True},
                                        {"false_detections": None, "detection_ran": False}])
    assert fd["per_frame"] == 1.5 and fd["frames_without_detection"] == 1


# --------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------


def _toy_doc(accept_v2=True):
    """A fabricated results document over toy (non-ho) set ids."""
    truth = [_t(k, 20.0 + 40 * k, 20.0, r_out_px=10.0, isolated=True, vane_width_frac=0.01, n_vanes=4)
             for k in range(4)]
    scen = {"toy_f0": {"group": "set:toy_spider", "split": "dev", "pre_existing": False,
                       "metadata": {"starless": False}, "sources": truth},
            "toy_core": {"group": "core", "split": "dev", "pre_existing": True, "metadata": {}, "sources": truth}}
    rows = []
    for method, accepted in (("python_sep__n2g2", truth[:3] if accept_v2 else truth[:1]),
                             ("python_sep__n1g1", truth[:1])):
        acc = [{"x": s["x"], "y": s["y"], "hfr_px": 4.0} for s in accepted]
        rows.append({"scenario": "toy_f0", "method": method, "status": "completed", "accepted_stars": acc,
                     "detection_ran": True, "metrics": {"detection": {"false_detections": 0}}})
        rows.append({"scenario": "toy_core", "method": method, "status": "completed", "accepted_stars": acc,
                     "detection_ran": True,
                     "metrics": {"detection": {"completeness_stars": 1.0 if method.endswith("2") else 0.99,
                                               "false_detections": 0},
                                 "hfr": {"clean_rel_bias_median": 0.01}}})
    return {"results": rows, "scenarios": scen,
            "scenario_sets": {"toy_spider": {"frames": ["toy_f0"], "split": "dev"}},
            "focus_sweeps": [], "cross_implementation": {"v2": {"n2g2": {"status": "completed", "cases": {
                "toy_f0": {"all_agree": True}}}}}}


def test_evaluator_takes_thresholds_from_the_criteria_and_reports_three_states():
    variants = json.loads(ACCEPTANCE.read_text())["variants"]
    crit = {"schema": "bge-synthetic-acceptance-v1", "name": "toy", "variants": variants, "criteria": [
        {"id": "T1", "metric": "donut_acceptance_rate", "scenarios": ["toy_spider"], "backend": "sep",
         "population": "isolated truth sources with r_out_px >= 6", "variant": "v2", "op": ">=", "value": 0.70},
        {"id": "T2", "metric": "donut_acceptance_rate", "scenarios": ["toy_spider"], "backend": "sep",
         "population": "isolated truth sources with r_out_px >= 6", "variant": "v2", "op": ">=", "value": 0.80},
        {"id": "T3", "metric": "no_regression", "scenarios": "all pre-existing development scenarios",
         "backend": "sep", "compare": ["completeness v2 >= v1 - 0.02", "abs_median_hfr_bias v2 <= abs_v1 + 0.005"]},
        {"id": "T4", "metric": "donut_acceptance_rate", "scenarios": ["toy_missing"], "backend": "sep",
         "population": "", "variant": "v2", "op": ">=", "value": 0.1},
        {"id": "T5", "metric": "rust_python_parity", "scenarios": "all", "compare": "x"}]}
    verdict = EV.evaluate(crit, [_toy_doc()])
    by = {c["id"]: c for c in verdict["criteria"]}
    python = lambda cid: next(e for e in by[cid]["entries"] if e["method"] == "python_sep__n2g2")  # noqa: E731
    assert python("T1")["status"] == "pass" and python("T1")["value"] == 0.75
    assert python("T2")["status"] == "fail"  # same data, the file's stricter value decides
    assert by["T1"]["status"] == "not_evaluable"  # the Rust implementation is absent
    assert python("T3")["status"] == "pass"
    assert by["T4"]["status"] == "not_evaluable"
    t5 = {e["method"]: e["status"] for e in by["T5"]["entries"]}
    assert t5["sep__n2g2"] == "pass" and t5["sep__n1g1"] == "not_evaluable"
    worse = EV.evaluate(crit, [_toy_doc(accept_v2=False)])
    assert next(e for e in worse["criteria"][0]["entries"] if e["method"] == "python_sep__n2g2")["status"] == "fail"
    assert EV.parse_compare("v2 <= 2.5 * v1")["mult"] == 2.5
    assert EV.parse_compare("v2 >= v1 - 0.05")["offset"] == -0.05
    assert EV.parse_compare("abs_median_hfr_bias v2 <= abs_v1 + 0.005")["abs"]


def test_evaluator_reads_the_committed_criteria_without_writing_it(tmp_path):
    before = hashlib.sha256(ACCEPTANCE.read_bytes()).hexdigest()
    results = tmp_path / "results.json"
    results.write_text(json.dumps(_toy_doc()))
    assert EV.main(["--results", str(results), "--out", str(tmp_path / "verdict.json")]) == 0
    assert hashlib.sha256(ACCEPTANCE.read_bytes()).hexdigest() == before
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert [c["id"] for c in verdict["criteria"]] == [f"A{k}" for k in range(1, 12)]
    assert verdict["summary"]["pass"] + verdict["summary"]["fail"] + verdict["summary"]["not_evaluable"] == 11
