"""BGE v2 star gates (docs/bge-v2-spec.md §1-§2, §5): dip_annulus_v2,
SEP deblend regrouping, the configuration record and the Rust-parity gates.

Seeded synthetic crops/frames; correctness checks of the reference
implementation, not on-sky accuracy. SEP-backed tests require sep==1.4.1
(MissingPrerequisite otherwise). No held-out (ho_*) scenario is used.
"""

import json
import math

import numpy as np
import pytest

from ml.eval.bge import scenarios as SC
from ml.labels import psf_measure as pm

V2 = pm.StarGateParams()
V1 = pm.StarGateParams(blend_test="peak_count_v1", regroup_deblended=False)
SKY, NOISE = 1000.0, 10.0

# Pinned cross-language hashes (spec §1.6; the Rust tests pin the same).
STAR_GATES_V2_SHA256 = "77034b3d3e0fc0dc9661a822a8ae7baa0ef644eeaa2273ddd2fcf8ec60f93c6b"
STAR_GATES_V1_SHA256 = "b517dd32a276a9e645c00d584cf9386bef2817584db91bf2087322d8e39e4460"
STAR_GATES_V1_REGROUP_SHA256 = "bb8329236a23e7337e8766d9f7032fa722d4c6535ad32a8c535adf918348b9ad"
STAR_GATES_V2_JSON = (
    '{"accept_deblended":false,"annulus":{"bin_fraction":0.5,"bins":16,"half_width":0.25,"hole_fraction":0.6,'
    '"hole_radius":0.35,"min_coverage":0.625,"min_radius_px":2.5,"radius_tol":0.35},"blend_test":"dip_annulus_v2",'
    '"dip_fraction":0.15,"dip_min_sigma":4.0,"peak_window_half":3,"regroup_deblended":true,"threshold_sigma":5.0}'
)


@pytest.fixture(scope="module")
def sep():
    return pm.require_sep()


def _crop(size=96, seed=1):
    return np.random.default_rng(seed).normal(0.0, NOISE, (size, size))


def _ring(size, cx, cy, r_out, n_vanes=0, vane=0.01, flux=3e5, obstruction=0.3, seeing=1.3, angle=0.3):
    img = np.zeros((size, size))
    SC._paste(img, SC.render_pupil_shadow(cx, cy, r_out, obstruction, seeing, flux, n_vanes=n_vanes,
                                          vane_width_frac=vane, vane_angle_rad=angle))
    return img


def _gauss(size, cx, cy, sigma, flux):
    ys, xs = np.mgrid[0:size, 0:size]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma * sigma))
    return flux * g / g.sum()


# --------------------------------------------------------------------------
# Configuration record
# --------------------------------------------------------------------------


def test_star_gate_records_and_pinned_hashes():
    assert pm.canonical_json(V2.config()) == STAR_GATES_V2_JSON
    assert V2.config_sha256() == STAR_GATES_V2_SHA256
    assert V1.config() == {"blend_test": "peak_count_v1", "peak_window_half": 3, "threshold_sigma": 5.0,
                           "regroup_deblended": False, "accept_deblended": False}
    assert V1.config_sha256() == STAR_GATES_V1_SHA256
    assert pm.StarGateParams(blend_test="peak_count_v1").config_sha256() == STAR_GATES_V1_REGROUP_SHA256
    assert pm.star_gates("peak_count_v1", False) == V1
    with pytest.raises(ValueError):
        pm.StarGateParams(blend_test="nope")
    # accept_deblended disables regrouping (spec §2.5).
    assert V2.regroups and not pm.StarGateParams(accept_deblended=True).regroups and not V1.regroups


# --------------------------------------------------------------------------
# dip_annulus_v2 on crops (spec §5 unit cases)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_vanes", [0, 3, 4])
def test_ring_with_0_3_4_vanes_is_one_source(n_vanes):
    res = _ring(96, 47.3, 48.1, 14.0, n_vanes=n_vanes) + _crop()
    v2 = pm.blend_gate(res, NOISE, V2)
    assert v2.n_peaks >= 3 and not v2.blended
    # v1 counts every maximum along the ring as a separate core (D-014).
    assert pm.blend_gate(res, NOISE, V1).blended


def test_spider_cut_ring_is_accepted_through_the_annulus_exemption():
    """Wide vanes cut the ring into arcs separated by real (significant) dips;
    the arcs share one circle, cover the azimuth and leave a hole."""
    res = _ring(96, 47.3, 48.1, 20.0, n_vanes=4, vane=0.03, flux=3e5, obstruction=0.35, seeing=1.2) + _crop()
    result = pm.blend_gate(res, NOISE, V2)
    assert result.flagged and result.n_components >= 2
    assert result.exempt and not result.blended
    assert result.annulus["passed"] and result.annulus["coverage"] >= 0.625
    assert result.annulus["hole_mean"] <= 0.6 * result.annulus["top"]


def test_exemption_fails_the_common_circle_on_a_broad_flat_annulus():
    """Recorded behaviour (development evidence, not tuned): on a large,
    weakly obstructed donut cut by vanes, each arc's maximum is a noise peak
    anywhere across the broad annulus, so the significant peaks need not share
    one circle within ANN_RADIUS_TOL * r + 1."""
    res = _ring(96, 47.3, 48.1, 30.0, n_vanes=4, vane=0.015, flux=1e6) + _crop()
    result = pm.blend_gate(res, NOISE, V2)
    assert result.flagged and result.blended and result.annulus["step"] == "common_circle"
    r = result.annulus["radius_px"]
    assert max(abs(d - r) for d in result.annulus["peak_distances_px"]) > 0.35 * r + 1.0


def test_close_double_at_6px_is_rejected():
    res = _gauss(96, 45.0, 48.0, 1.8, 2e4) + _gauss(96, 51.0, 48.0, 1.8, 2e4) + _crop()
    result = pm.blend_gate(res, NOISE, V2)
    assert result.flagged and not result.exempt and result.blended
    assert result.annulus["step"] == "coverage" and result.annulus["coverage"] < 0.625
    assert pm.blend_gate(res, NOISE, V1).blended


def test_neighbouring_star_in_the_crop_is_rejected():
    res = _gauss(96, 48.0, 48.0, 1.8, 2e4) + _gauss(96, 80.0, 70.0, 1.8, 1e4) + _crop()
    result = pm.blend_gate(res, NOISE, V2)
    assert result.blended
    disjoint = [r for r in result.resolved if r["kind"] == "disjoint"]
    assert disjoint and disjoint[0]["significant"]


def test_disjoint_noise_bump_just_above_threshold_is_ignored():
    """A bump 1-2 sigma above the 5-sigma threshold, disjoint from the star:
    v1 counts a second core; v2's disjoint dip p - t (< 4 sigma) is not
    significant (a deliberate change from v1)."""
    res = _gauss(96, 48.0, 48.0, 1.8, 2e4)
    res[20, 75] = 6.5 * NOISE  # a lone pixel: above t = 50, dip 15 < max(0.15 p, 40)
    assert pm.blend_gate(res, NOISE, V1).blended
    result = pm.blend_gate(res, NOISE, V2)
    assert result.n_peaks == 2 and result.n_components == 1 and not result.flagged
    assert result.resolved[0]["kind"] == "disjoint" and not result.resolved[0]["significant"]


def test_saddle_merge_resolves_the_fainter_peak_at_the_saddle_level():
    res = np.zeros((9, 15), dtype=np.float32)
    res[4, 2] = 100.0   # bright peak
    res[4, 12] = 80.0   # faint peak, 10 px away (outside the 7x7 window)
    # A sloped ridge above t (no plateau maxima) joins them at a saddle of 60.
    res[4, 3:12] = [95, 90, 85, 75, 60, 70, 72, 75, 78]
    result = pm.blend_gate(res, 1.0, V2)  # t = 5
    (rec,) = result.resolved
    assert rec["kind"] == "saddle" and rec["dip"] == pytest.approx(20.0)
    # dip 20 > max(0.15 * 80, 4 * 1): significant; the pair is no annulus.
    assert rec["significant"] and result.blended
    shallow = res.copy()
    shallow[4, 3:12] = [95, 90, 85, 80, 75, 76, 77, 78, 79]  # dip 5 < 0.15 * 80 = 12: one component
    shallow_result = pm.blend_gate(shallow, 1.0, V2)
    assert shallow_result.n_peaks == 2 and not shallow_result.flagged


def test_peaks_ties_and_plateaus_follow_the_spec():
    res = np.zeros((12, 12), dtype=np.float32)
    res[2, 2] = res[2, 3] = 50.0  # one plateau peak, representative = lowest index
    res[9, 9] = 50.0               # an equal peak: the higher representative is the fainter
    peaks = pm.find_blend_peaks(res, np.float32(5.0))
    assert [p.rep for p in peaks] == [2 * 12 + 2, 9 * 12 + 9]
    records, brightest = pm.saddle_merge(res, np.float32(5.0), peaks)
    assert brightest == 0 and records[0]["peak"] == 1 and records[0]["kind"] == "disjoint"


def test_crop_path_measure_star_uses_v2_by_default_and_keeps_v1():
    crop = SKY + _ring(64, 31.6, 32.2, 10.0, n_vanes=4, flux=1e5) + _crop(64, seed=4)
    v2 = pm.measure_star(crop, sky=SKY, sky_rms=NOISE, saturation_level=65535.0, fit_fwhm=False)
    v1 = pm.measure_star(crop, sky=SKY, sky_rms=NOISE, saturation_level=65535.0, fit_fwhm=False,
                         blend_test="peak_count_v1")
    assert "blended" not in v2.rejections and v2.blend["n_peaks"] >= 2
    assert "blended" in v1.rejections and v1.blend is None
    double = SKY + _gauss(64, 29.0, 32.0, 1.6, 2e4) + _gauss(64, 35.0, 32.0, 1.6, 2e4) + _crop(64, seed=5)
    for test in ("dip_annulus_v2", "peak_count_v1"):
        m = pm.measure_star(double, sky=SKY, sky_rms=NOISE, saturation_level=65535.0, fit_fwhm=False,
                            blend_test=test)
        assert "blended" in m.rejections
    with pytest.raises(ValueError):
        pm.measure_star(crop, sky=SKY, sky_rms=NOISE, saturation_level=65535.0, blend_test="x")


# --------------------------------------------------------------------------
# Families (spec §2) and the Rust-parity gates
# --------------------------------------------------------------------------


def test_deblend_families_unite_8_connected_children_only():
    seg = np.zeros((6, 8), dtype=np.int32)
    seg[1, 1] = 1          # child 0
    seg[2, 2] = 2          # child 1, diagonal neighbour of child 0
    seg[1, 5] = 3          # parent-less isolated object 2 (not a child)
    seg[4, 6] = 4          # child 3, alone
    seg[4, 5] = 5          # child 4 touches child 3 from the left
    seg[1, 4] = 6          # child 5 touches non-child 2 only
    merged = [True, True, False, True, True, True]
    assert pm.deblend_families(seg, merged) == [[0, 1], [3, 4], [5]]
    assert pm.deblend_families(seg, [False] * 6) == []


def _row(i, x, y, npix, flux, peak, bbox, deblended):
    return {"x": x, "y": y, "xmin": bbox[0], "xmax": bbox[1], "ymin": bbox[2], "ymax": bbox[3], "npix": npix,
            "flux": flux, "peak_raw": peak, "deblended": deblended, "truncated": False}


def test_family_candidate_is_the_union_with_a_flux_weighted_centre():
    rows = [_row(0, 10.0, 20.0, 30, 300.0, 50.0, (5, 15, 15, 25), True),
            _row(1, 14.0, 20.0, 20, 100.0, 70.0, (12, 18, 16, 24), True),
            _row(2, 50.0, 50.0, 10, 50.0, 40.0, (48, 52, 48, 52), False),
            _row(3, 18.0, 22.0, 5, -5.0, 10.0, (17, 19, 21, 23), True)]
    seg = np.zeros((60, 60), dtype=np.int32)
    seg[20, 10:14] = 1
    seg[20, 14:17] = 2
    seg[21, 17:19] = 4
    seg[50, 50] = 3
    cands, families = pm.focus_candidates_from_catalog(rows, seg, V2)
    assert families == [[0, 1, 3]]
    fam, single = cands
    assert (fam.family_size, fam.members, fam.deblended) == (3, (0, 1, 3), True)
    assert fam.x == pytest.approx((10 * 300 + 14 * 100) / 400) and fam.y == pytest.approx(20.0)
    assert (fam.area, fam.flux, fam.peak) == (55, pytest.approx(395.0), 70.0)
    assert (fam.min_x, fam.max_x, fam.min_y, fam.max_y) == (5, 19, 15, 25)
    assert single.family_size == 1 and not single.deblended
    # Without regrouping the children stay children (v1).
    cands_v1, families_v1 = pm.focus_candidates_from_catalog(rows, seg, V1)
    assert families_v1 == [] and [c.family_size for c in cands_v1] == [1, 1, 1]  # flux <= 0 child filtered


def _frame_with(image_fn, size=256, seed=11):
    rng = np.random.default_rng(seed)
    return SKY + image_fn(size) + rng.normal(0.0, NOISE, (size, size))


def _variant(image, gates, noise_policy="masked_local_v2"):
    from ml.eval.bge import python_methods as PM

    est = PM.python_sep_variant_background(image, {}, noise_policy)
    return PM.detect_and_gate(image, {}, est, gates)


def test_family_regroup_of_a_spider_cut_ring_is_accepted(sep):
    """SEP splits a ring cut by wide vanes into arcs (OBJ_MERGED children):
    v1 rejects every child as deblended; v2 regroups them into one family,
    which gate 5 flags and the annulus exemption accepts."""
    image = _frame_with(lambda n: _ring(n, 128.3, 127.6, 20.0, n_vanes=4, vane=0.03, flux=3e5, obstruction=0.35,
                                        seeing=1.2))
    v1 = _variant(image, V1)
    assert v1["detection"]["objects_deblended"] >= 2
    assert v1["n_accepted"] == 0 and v1["rejections"]["deblended"] >= 2
    v2 = _variant(image, V2)
    assert v2["families"]["multi_member"] == 1 and v2["n_accepted"] == 1
    assert v2["gate_diagnostics"] == {"annulus_exempt": 1, "regrouped_accepted": 1}
    assert v2["accepted"][0]["family_size"] >= 2 and v2["accepted"][0]["annulus_exempt"]
    assert v2["star_count_detected"] == 1  # the family replaces its children
    # The labeller regroups and accepts the same family.
    sources, label = pm.measure_frame_sep(image, fit_fwhm=False)
    family = [s for s in sources if s.family_size >= 2]
    assert len(family) == 1 and family[0].valid and family[0].blend["exempt"]
    assert label.gate_diagnostics == {"annulus_exempt": 1, "regrouped_accepted": 1}
    sources, label = pm.measure_frame_sep(image, fit_fwhm=False, blend_test="peak_count_v1", regroup_deblended=False)
    assert not any(s.valid for s in sources) and label.rejection_counts.get("deblended", 0) >= 2


def test_family_of_two_stars_is_rejected_as_deblended_not_blended(sep):
    image = _frame_with(lambda n: _gauss(n, 124.0, 128.0, 1.6, 3e4) + _gauss(n, 132.0, 128.0, 1.6, 3e4), seed=17)
    v2 = _variant(image, V2)
    assert v2["families"]["multi_member"] == 1 and v2["n_accepted"] == 0
    assert v2["rejections"]["deblended"] == 1 and v2["rejections"]["blended"] == 0
    v1 = _variant(image, V1)
    assert v1["rejections"]["deblended"] == 2


def test_rust_port_gate_order_caps_and_crop(sep):
    rng = np.random.default_rng(2)
    ys, xs = np.mgrid[0:256, 0:256]
    image = SKY + rng.normal(0, NOISE, xs.shape)
    for k in range(20):
        cx, cy = 20 + (k % 5) * 50, 25 + (k // 5) * 55
        image += _gauss(256, cx, cy, 1.8, 2e4 + 1e3 * k)
    out = _variant(image, V2)
    assert out["n_accepted"] == pm.FOCUS_MAX_STARS_USED  # 16 used, the rest never examined
    assert sum(out["rejections"].values()) == 0
    assert out["star_gates_config_sha256"] == STAR_GATES_V2_SHA256
    assert pm.rust_extract_crop(256, 256, 3.4, 250.6, 40) == (0, 216, 40, 40)
    assert pm.rust_extract_crop(256, 256, 128.5, 128.49, 32) == (113, 112, 32, 32)


def test_legacy_crop_sky_and_median_conventions():
    assert pm.rust_local_sky_stats(np.array([1.0, 2.0, 3.0, 4.0], np.float32))[0] == pytest.approx(2.5)
    assert pm.rust_legacy_crop_sky(np.array([1.0, 2.0, 3.0, 4.0], np.float32))[0] == 3.0  # nearest rank
    crop = (SKY + _gauss(64, 32.0, 32.0, 2.0, 1e5) + _crop(64, seed=8)).astype(np.float32)
    hfr = pm.rust_compute_hfr_with_sky(crop, level=SKY, sigma=NOISE)
    # Pixel-centre sampling of the curve of growth reads a sigma = 2 px
    # Gaussian a few percent below its continuous HFR.
    assert hfr == pytest.approx(2.0 * math.sqrt(2.0 * math.log(2.0)), rel=0.1)
    level = np.full(crop.shape, SKY, np.float32)
    assert pm.rust_compute_hfr_with_sky(crop, level=level, sigma=np.full(crop.shape, NOISE, np.float32)) \
        == pytest.approx(hfr, rel=1e-6)
    bad = np.full(crop.shape, NOISE, np.float32)
    bad[0, 0] = 0.0
    assert math.isnan(pm.rust_compute_hfr_with_sky(crop, level=level, sigma=bad))


def test_gate_record_is_json_serialisable():
    res = _ring(96, 47.3, 48.1, 30.0, n_vanes=4, vane=0.015, flux=1e6) + _crop()
    json.dumps(pm.blend_gate(res, NOISE, V2).to_json())
