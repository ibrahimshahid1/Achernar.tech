"""BGE v2 noise map masked_local_v2 (docs/bge-v2-spec.md §3, §5; amendment
A-1): cell floor, source mask and pass 2, local validity, refinement record,
configuration hashes.

Seeded synthetic frames; correctness checks against SEP 1.4.1 and the spec,
not on-sky accuracy. No held-out (ho_*) scenario is used.
"""

import numpy as np
import pytest

from ml.labels import psf_measure as pm

V1 = pm.SepBackgroundParams(noise_policy="frame_wide_v1")
V2 = pm.SepBackgroundParams()
V1_SHA256 = "80eab0fbae4abf9f60f348adb7de3004067a3873cca29f996289e9dc14040b84"
V2_SHA256 = "97bd28741f79a184aa55d8e64e8f368ffd1eb71e07b2338cc0b99ed60832a6f4"
V2_JSON = (
    '{"bh":64,"bw":64,"fh":3,"fthresh":0.0,"fw":3,"library":"sep","library_version":"1.4.1","maskthresh":0.0,'
    '"method":"sep_spatial_mesh","method_version":1,"min_meshes_per_axis":2,"noise_policy":"masked_local_v2",'
    '"rms_validity":{"fraction":0.5,"max_invalid_fraction":0.25,"rule":"local_cell_min"},'
    '"source_mask":{"ellipse_scale":3.0,"max_fraction":0.5,"minarea":5,"thresh_sigma":2.0}}'
)
REFINEMENT_KEYS = {"noise_policy", "pass2_used", "pass2_skipped_reason", "source_mask_objects", "source_mask_pixels",
                   "source_mask_fraction", "invalid_rms_pixels", "invalid_rms_fraction", "detection_area_fraction"}


@pytest.fixture(scope="module")
def sep():
    return pm.require_sep()


def _ridge(amplitude=600.0, seed=5, size=512):
    ys, xs = np.mgrid[0:size, 0:size]
    rng = np.random.default_rng(seed)
    return 1000.0 + amplitude * np.exp(-(((xs - 256) / 60.0) ** 2)) + rng.normal(0, 5.0, xs.shape), xs, ys


def _star(xs, ys, cx, cy, amp=3000.0, sigma=1.8):
    return amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma * sigma))


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_v1_hash_unchanged_and_v2_hash_pinned():
    assert V1.config_sha256() == V1_SHA256
    assert pm.canonical_json(V2.config()) == V2_JSON
    assert V2.config_sha256() == V2_SHA256
    with pytest.raises(ValueError):
        pm.SepBackgroundParams(noise_policy="local")
    with pytest.raises(ValueError):
        pm.SepBackgroundParams(rms_max_invalid_fraction=1.5)


# --------------------------------------------------------------------------
# Cell floor (spec §3.2)
# --------------------------------------------------------------------------


def test_cell_lower_node_formula_at_frame_edges():
    lower = pm._cell_lower_nodes(256, 64, 4)
    # Nodes at 32, 96, 160, 224: left of the first node -> cell 0; the last
    # cell (2) covers everything right of node 2, clamped to nx - 2.
    assert list(lower[[0, 31, 32, 95, 96, 159, 160, 224, 255]]) == [0, 0, 0, 0, 1, 1, 2, 2, 2]
    # A partial last mesh (w = 200, nx = 4): still clamped to nx - 2.
    assert pm._cell_lower_nodes(200, 64, 4)[-1] == 2
    # Single-node axis: always 0.
    assert set(pm._cell_lower_nodes(50, 64, 1)) == {0}


def test_cell_floor_takes_the_cell_minimum_and_falls_back_globally():
    sigmas = np.array([[5.0, 6.0, 7.0],
                       [8.0, 0.0, 9.0]])  # ny = 2, nx = 3
    floor = pm.cell_floor_map(sigmas, (128, 192), (64, 64))
    assert floor.dtype == np.float32 and floor.shape == (128, 192)
    assert floor[0, 0] == 5.0             # cell (0,0): nodes 5, 6, 8, 0 -> 5 (0 is not positive)
    assert floor[127, 191] == 6.0         # cell (1,0)... x right of node 1: nodes 6, 7, 0, 9 -> 6
    nonpositive = np.array([[0.0, 0.0, 4.0], [-1.0, np.nan, 3.0]])
    floor = pm.cell_floor_map(nonpositive, (128, 192), (64, 64))
    assert floor[0, 0] == 3.0             # no positive node in the cell: global minimum positive
    assert floor[0, 191] == 3.0
    assert pm.cell_floor_map(np.zeros((2, 3)), (128, 192), (64, 64)) is None


def test_cell_floor_on_single_mesh_axes():
    sigmas = np.array([[4.0, 2.0, 6.0]])  # ny = 1: the +1 row is dropped
    floor = pm.cell_floor_map(sigmas, (40, 192), (64, 64))
    assert np.all(floor[:, :96] == 2.0) and np.all(floor[:, 160:] == 2.0)
    floor = pm.cell_floor_map(np.array([[3.0], [1.5]]), (128, 50), (64, 64))  # nx = 1
    assert np.all(floor == 1.5)


# --------------------------------------------------------------------------
# Background passes and validity
# --------------------------------------------------------------------------


def test_v1_policy_is_byte_identical_and_has_no_refinement(sep):
    image, xs, ys = _ridge(0.0)
    image = image + _star(xs, ys, 100, 100)
    a = pm.estimate_spatial_background(image, params=V1)
    raw = sep.Background(np.ascontiguousarray(image, dtype=np.float32), bw=64, bh=64, fw=3, fh=3)
    np.testing.assert_array_equal(a.background_map, raw.back(dtype=np.float32))
    assert a.refinement is None and a.invalid_mask is None and a.noise_policy == "frame_wide_v1"
    assert a.provenance()["refinement"] is None


def test_source_mask_and_pass2(sep):
    image, xs, ys = _ridge(0.0, seed=3)
    for cx, cy in ((100, 100), (300, 200), (400, 400)):
        image += _star(xs, ys, cx, cy, amp=20000.0, sigma=3.0)
    bg = pm.estimate_spatial_background(image)
    r = bg.refinement
    assert set(r) == REFINEMENT_KEYS and r["pass2_used"] and r["pass2_skipped_reason"] is None
    assert r["source_mask_objects"] == 3 and r["source_mask_pixels"] > 3 * 50
    assert r["source_mask_fraction"] == pytest.approx(r["source_mask_pixels"] / image.size)
    assert (r["invalid_rms_pixels"], r["detection_area_fraction"]) == (0, 1.0)
    # Reproduce the spec steps by hand: R_mask = fmax(R1, floor), extraction
    # at 2 sigma without deblending/cleaning, M_src = segmentation + SEP
    # ellipses at r = 3, pass 2 = SEP with that mask.
    v1 = pm.estimate_spatial_background(image, params=V1)
    floor = pm.cell_floor_map(pm.mesh_sigma_grid(v1.raw_rms_map, (64, 64))[0], image.shape, (64, 64))
    data_sub = np.ascontiguousarray(image - v1.background_map, dtype=np.float32)
    objects, seg = sep.extract(data_sub, 2.0, err=np.fmax(v1.raw_rms_map, floor).astype(np.float32), minarea=5,
                               filter_kernel=pm.SepDetectionParams().kernel_array(), filter_type="conv",
                               deblend_cont=1.0, clean=False, segmentation_map=True)
    mask = pm.build_source_mask(sep, objects, seg, 3.0)
    ellipses = np.zeros(seg.shape, np.uint8)
    sep.mask_ellipse(ellipses, objects["x"], objects["y"], objects["a"], objects["b"], objects["theta"], r=3.0)
    assert len(objects) == 3 and ellipses.sum() > 0
    np.testing.assert_array_equal(mask, (seg > 0) | ellipses.astype(bool))
    assert int(mask.sum()) == r["source_mask_pixels"]
    b2 = sep.Background(np.ascontiguousarray(np.where(mask, 0, image), dtype=np.float32),
                        mask=mask.astype(np.uint8), maskthresh=0.0, bw=64, bh=64, fw=3, fh=3)
    np.testing.assert_array_equal(bg.background_map, b2.back(dtype=np.float32))
    np.testing.assert_array_equal(bg.rms_map, b2.rms(dtype=np.float32))


def test_pass2_is_skipped_for_a_giant_source_mask(sep):
    """A dense field (529 stars every 11 px): segmentation plus 3x ellipses
    cover more than half of the frame."""
    ys, xs = np.mgrid[0:256, 0:256]
    rng = np.random.default_rng(9)
    image = 1000.0 + rng.normal(0, 5.0, xs.shape)
    for cx in range(6, 256, 11):
        for cy in range(6, 256, 11):
            image = image + _star(xs, ys, cx, cy)
    bg = pm.estimate_spatial_background(image)
    r = bg.refinement
    assert not r["pass2_used"] and r["pass2_skipped_reason"] == "source_mask_too_large"
    assert r["source_mask_fraction"] > 0.5 and "source_mask_too_large" in bg.reasons
    v1 = pm.estimate_spatial_background(image, params=V1)
    np.testing.assert_array_equal(bg.background_map, v1.background_map)  # pass 1 kept
    # The max-fraction parameter is what decides (here a small frame of stars).
    small, sxs, sys_ = _ridge(0.0, seed=4, size=256)
    small = small + _star(sxs, sys_, 128, 128, amp=2e4, sigma=3.0)
    tight = pm.SepBackgroundParams(source_mask_max_fraction=1e-4)
    assert pm.estimate_spatial_background(small, params=tight).refinement["pass2_skipped_reason"] == \
        "source_mask_too_large"


def test_source_mask_failure_keeps_pass1(sep):
    image, xs, ys = _ridge(0.0, seed=6, size=256)
    image = image + _star(xs, ys, 128, 128, amp=2e4, sigma=3.0)
    tiny = pm.SepDetectionParams(pixstack=10)
    bg = pm.estimate_spatial_background(image, detection_params=tiny)
    assert bg.refinement["pass2_skipped_reason"] == "source_mask_failed" and "source_mask_failed" in bg.reasons
    assert bg.refinement["source_mask_objects"] is None and bg.noise_measured


def test_undefined_floor_is_noise_unresolved(sep):
    image = np.full((256, 256), 200.0)
    bg = pm.estimate_spatial_background(image)
    assert bg.refinement["pass2_skipped_reason"] == "noise_unresolved"
    assert bg.noise_status == "unresolved" and "zero_empirical_scatter" in bg.reasons
    assert bg.refinement["invalid_rms_pixels"] is None


def test_local_invalid_pixels_are_masked_and_tallied(sep):
    """amp-600 ridge (D-018 fixture): the undershoot is two narrow bands; the
    noise is measured, those pixels join the extraction mask, and a star
    whose measurement support touches them is invalid_noise_support."""
    image, xs, ys = _ridge(600.0)
    bands = pm.estimate_spatial_background(image).invalid_mask
    assert bands is not None and bands[:, 452:462].all() and not bands[:, 470:].any()
    image = image + _star(xs, ys, 472, 256) + _star(xs, ys, 256, 128)
    sources, label = pm.measure_frame_sep(image, fit_fwhm=False)
    assert label.background["noise_status"] == "measured" and "rms_map_locally_invalid" in label.reasons
    near = [s for s in sources if abs(s.x - 472) < 2]
    clean = [s for s in sources if abs(s.x - 256) < 2]
    assert near and near[0].rejections == [pm.Rejection.INVALID_NOISE_SUPPORT.value]
    assert clean and clean[0].valid
    refinement = label.background["refinement"]
    assert refinement["invalid_rms_fraction"] < 0.25
    assert refinement["detection_area_fraction"] == pytest.approx(1.0 - refinement["invalid_rms_fraction"])
    # The Rust-parity gates tally the same star.
    from ml.eval.bge import python_methods as PM

    est = PM.python_sep_variant_background(image, {}, "masked_local_v2")
    out = PM.detect_and_gate(image, {}, est, pm.StarGateParams())
    assert out["rejections"]["invalid_noise_support"] == 1 and out["n_accepted"] == 1


def test_frame_wide_fallback_above_the_invalid_fraction(sep):
    image, _, _ = _ridge(800.0)
    measured = pm.estimate_spatial_background(image)
    frac = measured.refinement["invalid_rms_fraction"]
    assert measured.noise_measured and 0.0 < frac < 0.25
    strict = pm.SepBackgroundParams(rms_max_invalid_fraction=frac / 2)
    bg = pm.estimate_spatial_background(image, params=strict)
    assert bg.noise_status == "unresolved" and bg.reasons == ("invalid_rms_map",)
    assert bg.rms_map is None and bg.invalid_mask is None
    assert bg.refinement["invalid_rms_fraction"] == pytest.approx(frac)
    sources, label = pm.measure_frame_sep(image, background_params=strict, fit_fwhm=False)
    assert sources == [] and label.status == "unavailable"


def test_local_bound_is_against_the_cell_floor_not_the_global_minimum():
    rms = np.full((128, 192), 6.0, dtype=np.float32)
    rms[10, 10] = 2.9    # below 0.5 * 6 (its cell floor): invalid
    rms[10, 180] = 2.9   # cell floor there is 6 as well
    rms[100, 100] = 3.1  # above 0.5 * 6
    sigmas = np.array([[6.0, 6.0, 6.0], [6.0, 6.0, 6.0]])
    v = pm.local_rms_validity(rms, np.ones(rms.shape, bool), sigmas, (64, 64), 0.5)
    assert v["count_below_local_bound"] == 2 and v["invalid_mask"][10, 10] and not v["invalid_mask"][100, 100]
    sigmas[0, 0] = 1.0   # one quiet mesh lowers only its own cells' floor
    v = pm.local_rms_validity(rms, np.ones(rms.shape, bool), sigmas, (64, 64), 0.5)
    assert not v["invalid_mask"][10, 10] and v["invalid_mask"][10, 180]


def test_amendment_a1_extraction_noise_and_the_sep_first_row_defect(sep):
    """SEP 1.4.1 thresholds masked pixels of the first image row against
    their own (negative) noise value; A-1 hands it the cell floor there."""
    rng = np.random.default_rng(0)
    data = rng.normal(0, 5, (200, 200)).astype(np.float32)
    noise = np.full(data.shape, 5.0, np.float32)
    mask = np.zeros(data.shape, np.uint8)
    mask[0:3, 20:60] = mask[100:103, 20:60] = 1
    noise[mask.astype(bool)] = -1.0
    data[mask.astype(bool)] = 0.0
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32)
    junk = sep.extract(data, 5.0, err=noise, mask=mask, maskthresh=0.0, minarea=5, filter_kernel=kernel,
                       filter_type="conv")
    assert len(junk) == 1 and int(junk["ymin"][0]) == int(junk["ymax"][0]) == 0
    floored = np.where(mask.astype(bool), np.float32(5.0), noise)
    assert len(sep.extract(data, 5.0, err=floored, mask=mask, maskthresh=0.0, minarea=5, filter_kernel=kernel,
                           filter_type="conv")) == 0
    # The SpatialBackground hands SEP the floor on every masked pixel and
    # keeps the reported R map unchanged.
    image, _, _ = _ridge(800.0)
    bg = pm.estimate_spatial_background(image)
    err = bg.extraction_rms()
    masked = bg.extraction_mask()
    np.testing.assert_array_equal(err[masked], bg.rms_floor_map[masked])
    np.testing.assert_array_equal(err[~masked], bg.rms_map[~masked])
    assert (bg.rms_map[masked] < 0).any() and (err > 0).all()
    assert pm.estimate_spatial_background(image, params=V1).extraction_rms() is None  # v1: unresolved here
