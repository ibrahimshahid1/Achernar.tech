"""
Spatial SEP background, local-noise detection, masks and label provenance
(BGE-02, BGE-07, BGE-15, BGE-16, BGE-17, BGE-20 — Python reference).

Every input is a fixed, seeded synthetic frame. These are correctness checks
of the reference implementation against known simulated truth or against the
pinned SEP library itself; they are not measured on-sky accuracy.

SEP-backed tests call ``require_sep()`` in ``setUpClass``: without
``sep==1.4.1`` they ERROR with "missing prerequisite: sep==1.4.1" instead of
skipping.
"""
import hashlib
import json
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labels import psf_measure as pm  # noqa: E402
from labels.psf_measure import (  # noqa: E402
    MissingPrerequisite,
    filtered_mesh_sigmas,
    rms_map_validity,
    Rejection,
    SepBackgroundParams,
    SepDetectionParams,
    StarMeasurement,
    aggregate_frame,
    estimate_spatial_background,
    fit_moffat_fwhm,
    measure_frame_sep,
    require_sep,
)

DEFAULT_BACKGROUND_SHA256 = "80eab0fbae4abf9f60f348adb7de3004067a3873cca29f996289e9dc14040b84"
DEFAULT_DETECTION_SHA256 = "4f4499cda7067107c0fe6d64319684c13d1ec62f590b4bee4b346c1459dc99df"

# Non-default, non-dyadic and small values (review F3). These are the exact
# Python json.dumps strings; the Rust canonical_json must reproduce them.
NONDEFAULT_DETECTION_JSON = (
    '{"accept_deblended_for_focus":false,"clean":true,"clean_param":1.0,"deblend_cont":1e-05,'
    '"deblend_nthresh":32,"filter_kernel":[[1.0,2.0,1.0],[2.0,4.0,2.0],[1.0,2.0,1.0]],'
    '"filter_type":"conv","library":"sep","library_version":"1.4.1","method":"sep_extract",'
    '"minarea":5,"noise":"background_rms_map","pixstack":300000,"thresh_sigma":1.2}'
)
NONDEFAULT_DETECTION_SHA256 = "fd6c2088782a5017c1a69a18cf4f4aee73665b029927083266e664f5f1718554"
NONDEFAULT_BACKGROUND_JSON = (
    '{"bh":64,"bw":64,"fh":3,"fthresh":0.1,"fw":3,"library":"sep","library_version":"1.4.1",'
    '"maskthresh":0.0,"method":"sep_spatial_mesh","method_version":1,"min_meshes_per_axis":2}'
)
NONDEFAULT_BACKGROUND_SHA256 = "1d87974ba87e717a55e01ee098d2892de8bc2f77807e56823f36265e6cbf48a1"


def _gaussian(xs, ys, cx, cy, amp, sigma):
    return amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2))


def _matched(detected_xy, truth_xy, tol=2.0):
    """Indices of truth positions that have a detection within ``tol`` px."""
    detected = np.asarray(detected_xy, dtype=float).reshape(-1, 2)
    found = set()
    for k, (tx, ty) in enumerate(truth_xy):
        if detected.size and np.min(np.hypot(detected[:, 0] - tx, detected[:, 1] - ty)) <= tol:
            found.add(k)
    return found


def _legacy_scalar_extract(sep, image):
    """The pre-v2 labeler's detection call: library-default mesh background,
    scalar ``err=globalrms`` (which SEP runs as CONV), no mask."""
    data = np.ascontiguousarray(image, dtype=np.float32)
    bkg = sep.Background(data)
    return sep.extract(data - bkg.back(), 5.0, err=bkg.globalrms, minarea=5, deblend_nthresh=32, deblend_cont=0.005)


class SepTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Raises MissingPrerequisite("missing prerequisite: sep==1.4.1 ...").
        cls.sep = require_sep()


class TestConfigContract(unittest.TestCase):
    def test_default_background_config_keys_and_hash(self):
        config = SepBackgroundParams().config()
        self.assertEqual(
            set(config),
            {"method", "method_version", "library", "library_version", "bw", "bh", "fw", "fh",
             "fthresh", "maskthresh", "min_meshes_per_axis"},
        )
        self.assertEqual(config["method"], "sep_spatial_mesh")
        self.assertEqual(config["library_version"], "1.4.1")
        self.assertEqual(SepBackgroundParams().config_sha256(), DEFAULT_BACKGROUND_SHA256)

    def test_default_detection_config_keys_and_hash(self):
        config = SepDetectionParams().config()
        self.assertEqual(
            set(config),
            {"method", "library", "library_version", "thresh_sigma", "noise", "minarea", "filter_type",
             "filter_kernel", "deblend_nthresh", "deblend_cont", "clean", "clean_param", "pixstack",
             "accept_deblended_for_focus"},
        )
        # The pre-v2 labeler's threshold value, kept with its CONV semantics.
        self.assertEqual(config["thresh_sigma"], 5.0)
        self.assertEqual(config["filter_type"], "conv")
        self.assertEqual(config["noise"], "background_rms_map")
        self.assertEqual(config["filter_kernel"], [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
        self.assertEqual(SepDetectionParams().config_sha256(), DEFAULT_DETECTION_SHA256)

    def test_nondefault_configs_pin_the_cross_language_strings(self):
        detection = SepDetectionParams(thresh_sigma=1.2, deblend_cont=1e-5).config()
        self.assertEqual(pm.canonical_json(detection), NONDEFAULT_DETECTION_JSON)
        self.assertEqual(pm.config_sha256(detection), NONDEFAULT_DETECTION_SHA256)
        background = SepBackgroundParams(fthresh=0.1).config()
        self.assertEqual(pm.canonical_json(background), NONDEFAULT_BACKGROUND_JSON)
        self.assertEqual(pm.config_sha256(background), NONDEFAULT_BACKGROUND_SHA256)

    def test_hash_is_sha256_of_canonical_json(self):
        config = SepDetectionParams(filter_type="matched", accept_deblended_for_focus=True).config()
        expected = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(pm.config_sha256(config), expected)
        self.assertNotEqual(expected, DEFAULT_DETECTION_SHA256)

    def test_parameters_are_validated(self):
        for kwargs in ({"bw": 0}, {"bh": -1}, {"fw": 0}, {"fh": 0}, {"fthresh": float("nan")},
                       {"min_meshes_per_axis": 0}, {"bw": 1.5}):
            with self.assertRaises(ValueError, msg=kwargs):
                SepBackgroundParams(**kwargs)
        for kwargs in ({"thresh_sigma": 0.0}, {"minarea": 0}, {"deblend_nthresh": 0}, {"deblend_cont": 0.0},
                       {"deblend_cont": 1.5}, {"filter_type": "gauss"}, {"filter_kernel": ((1.0, 2.0),)},
                       {"filter_kernel": ((0.0,),)}, {"pixstack": 0}, {"clean_param": -1.0}):
            with self.assertRaises(ValueError, msg=kwargs):
                SepDetectionParams(**kwargs)


class TestPrerequisite(unittest.TestCase):
    def test_missing_sep_is_a_clear_failure(self):
        with mock.patch.dict(sys.modules, {"sep": None}):
            with self.assertRaises(MissingPrerequisite) as ctx:
                require_sep()
        self.assertIn("missing prerequisite: sep==1.4.1", str(ctx.exception))

    def test_unpinned_sep_version_is_a_missing_prerequisite(self):
        fake = types.ModuleType("sep")
        fake.__version__ = "1.2.1"
        with mock.patch.dict(sys.modules, {"sep": fake}):
            with self.assertRaises(MissingPrerequisite) as ctx:
                measure_frame_sep(np.zeros((256, 256)))
        self.assertIn("missing prerequisite: sep==1.4.1", str(ctx.exception))
        self.assertIn("1.2.1", str(ctx.exception))


class TestSpatialBackgroundStatus(SepTestCase):
    def test_maps_are_the_library_maps_with_the_same_mask(self):
        rng = np.random.default_rng(0)
        ys, xs = np.mgrid[0:256, 0:320]
        image = (500.0 + 0.8 * xs + rng.normal(0, 6.0, xs.shape)).astype(np.float32)
        mask = np.zeros(image.shape, dtype=bool)
        mask[40:44, 100:180] = True
        bg = estimate_spatial_background(image, detector_mask=mask)
        reference = self.sep.Background(np.where(mask, 0.0, image).astype(np.float32), mask=mask.astype(np.uint8),
                                        bw=64, bh=64, fw=3, fh=3, fthresh=0.0)
        self.assertEqual(bg.level_status, "measured")
        self.assertEqual(bg.noise_status, "measured")
        self.assertEqual(bg.noise_source, "empirical_mesh_rms")
        self.assertEqual(bg.background_map.dtype, np.float32)
        self.assertEqual(bg.background_map.shape, image.shape)
        np.testing.assert_array_equal(bg.background_map, reference.back(dtype=np.float32))
        np.testing.assert_array_equal(bg.rms_map, reference.rms(dtype=np.float32))
        self.assertEqual(bg.global_background, float(reference.globalback))
        self.assertEqual(bg.global_rms, float(reference.globalrms))
        self.assertEqual(bg.library_version, self.sep.__version__)
        self.assertEqual(bg.config_sha256, SepBackgroundParams().config_sha256(self.sep.__version__))
        self.assertEqual(bg.mesh_grid, (5, 4))
        self.assertAlmostEqual(bg.usable_fraction, 1.0 - mask.mean())
        np.testing.assert_array_equal(bg.detector_mask, mask)

    def test_constant_frame_noise_is_unresolved_not_one(self):
        image = np.full((256, 256), 100.0)
        # SEP itself reports a fabricated global RMS of exactly 1.0 here
        # (background.c: no positive mesh sigma -> globalrms = 1.0).
        self.assertEqual(float(self.sep.Background(image.astype(np.float32)).globalrms), 1.0)
        bg = estimate_spatial_background(image)
        self.assertEqual(bg.level_status, "measured")
        self.assertEqual(bg.global_background, 100.0)
        self.assertEqual(bg.noise_status, "unresolved")
        self.assertIn("zero_empirical_scatter", bg.reasons)
        self.assertIsNone(bg.rms_map)
        self.assertIsNone(bg.global_rms)
        self.assertEqual(bg.noise_source, "unavailable")
        sources, label = measure_frame_sep(image)
        self.assertEqual(sources, [])
        self.assertEqual(label.status, "unavailable")
        self.assertIn("zero_empirical_scatter", label.reasons)
        self.assertIsNone(label.sky_rms_adu)
        self.assertIsNone(label.hfr_px_median)
        self.assertIsNone(label.saturated_star_fraction)  # no detection ran: not 0

    def test_frame_smaller_than_mesh_grid_is_unavailable(self):
        rng = np.random.default_rng(1)
        image = 200.0 + rng.normal(0, 5.0, (96, 96))
        # SEP would happily return numbers for this frame: one full and one
        # partial 64-px mesh per axis, i.e. effectively a global estimate.
        self.assertTrue(np.isfinite(self.sep.Background(image.astype(np.float32)).globalback))
        bg = estimate_spatial_background(image)
        self.assertEqual(bg.level_status, "unavailable")
        self.assertEqual(bg.noise_status, "unavailable")
        self.assertEqual(bg.reasons, ("frame_smaller_than_mesh_grid",))
        self.assertIsNone(bg.background_map)
        self.assertIsNone(bg.global_background)
        sources, label = measure_frame_sep(image)
        self.assertEqual((sources, label.status, label.sky_median_adu), ([], "unavailable", None))
        # The rule is per axis: 128 x 96 still fails on the short axis.
        self.assertEqual(estimate_spatial_background(rng.normal(0, 1, (96, 128))).reasons,
                         ("frame_smaller_than_mesh_grid",))
        self.assertEqual(estimate_spatial_background(200 + rng.normal(0, 1, (128, 128))).level_status, "measured")

    def test_no_supported_mesh_is_insufficient_support_not_zero(self):
        rng = np.random.default_rng(2)
        image = (300.0 + rng.normal(0, 5.0, (256, 256))).astype(np.float32)
        mask = np.zeros(image.shape, dtype=bool)
        mask[:, ::5] = False
        mask[:, (np.arange(256) % 5) < 3] = True  # 60% of every mesh masked
        # SEP: no mesh has >= 50% valid pixels -> back 0.0, rms 1.0, RETURN_OK.
        raw = self.sep.Background(image, mask=mask.astype(np.uint8))
        self.assertEqual((float(raw.globalback), float(raw.globalrms)), (0.0, 1.0))
        bg = estimate_spatial_background(image, detector_mask=mask)
        self.assertEqual(bg.level_status, "unavailable")
        self.assertEqual(bg.reasons, ("insufficient_support",))
        self.assertEqual(bg.meshes_supported, 0)
        self.assertIsNone(bg.global_background)

    def test_all_masked_and_all_nan_have_no_usable_pixels(self):
        image = np.full((256, 256), np.nan)
        self.assertEqual(estimate_spatial_background(image).reasons, ("no_usable_pixels",))
        finite = np.full((256, 256), 10.0)
        self.assertEqual(
            estimate_spatial_background(finite, detector_mask=np.ones(finite.shape, bool)).reasons,
            ("no_usable_pixels",),
        )

    def test_unsupported_mesh_is_reported_as_interpolated(self):
        rng = np.random.default_rng(3)
        image = 300.0 + rng.normal(0, 5.0, (256, 256))
        mask = np.zeros(image.shape, dtype=bool)
        mask[64:128, 64:128] = True
        bg = estimate_spatial_background(image, detector_mask=mask)
        self.assertEqual(bg.level_status, "measured")
        self.assertIn("meshes_interpolated", bg.reasons)
        self.assertEqual((bg.meshes_supported, bg.meshes_total), (15, 16))

    def test_malformed_geometry(self):
        self.assertEqual(estimate_spatial_background(np.ones(300)).reasons, ("malformed_geometry",))
        self.assertEqual(estimate_spatial_background(np.ones((0, 300))).reasons, ("malformed_geometry",))
        self.assertEqual(
            estimate_spatial_background(np.ones((256, 256)), detector_mask=np.zeros((256, 255), bool)).reasons,
            ("malformed_geometry",),
        )
        _, label = measure_frame_sep(np.ones((256, 256)), native_saturation_mask=np.zeros((10, 10), bool))
        self.assertEqual(label.reasons, ["malformed_geometry"])

    def test_filtered_mesh_sigmas_are_recovered_exactly(self):
        # bw odd and bh even put every spline node exactly on a pixel (x node at
        # (k+0.5)*bw-0.5, y node at (j+0.5)*bh), so R there IS the node value:
        # an independent check of the spline transcription and inversion.
        rng = np.random.default_rng(31)
        ys, xs = np.mgrid[0:320, 0:315]
        image = (300.0 + rng.normal(0.0, 1.0, xs.shape) * (3.0 + 0.02 * xs + 0.01 * ys)).astype(np.float32)
        rms = self.sep.Background(image, bw=63, bh=64).rms()
        grid, diagnostics = filtered_mesh_sigmas(rms, (63, 64))
        nodes = rms[np.ix_(32 + 64 * np.arange(5), 31 + 63 * np.arange(5))]
        self.assertEqual(diagnostics["mesh_sigma_method"], "spline_inversion")
        self.assertLess(diagnostics["reconstruction_max_rel_error"], 1e-6)
        # Tolerance: float32 storage of the map (~6e-8 relative) and the
        # float32 dx accumulation inside SEP; 1e-6 relative leaves margin.
        np.testing.assert_allclose(grid, nodes, rtol=1e-6)

    def test_positive_spline_undershoot_is_invalid_and_not_extracted(self):
        """D-018 fixture (review B1): amp-600 ridge, starless. The map stays
        positive, yet dips far below every mesh sigma; extraction on it finds
        stars that do not exist. The labeler must stop before sep.extract."""
        rng = np.random.default_rng(5)
        ys, xs = np.mgrid[0:512, 0:512]
        image = 1000.0 + 600.0 * np.exp(-(((xs - 256) / 60.0) ** 2)) + rng.normal(0, 5.0, xs.shape)
        raw = self.sep.Background(image.astype(np.float32))
        rms = raw.rms()
        self.assertGreater(float(rms.min()), 0.0)
        false = self.sep.extract((image - raw.back()).astype(np.float32), 5.0, err=rms, filter_type="conv", minarea=5)
        self.assertGreater(len(false), 20)  # the defect: a starless frame "has" stars

        reasons, diagnostics = rms_map_validity(rms, np.ones(rms.shape, bool), SepBackgroundParams())
        self.assertEqual(reasons, ["invalid_rms_map"])
        self.assertEqual(diagnostics["mesh_sigma_method"], "spline_inversion")
        self.assertLess(diagnostics["min_rms"], diagnostics["bound"])
        self.assertAlmostEqual(diagnostics["bound"], 0.5 * diagnostics["min_positive_mesh_sigma"])
        self.assertGreater(diagnostics["count_below_bound"], 0)

        calls = []
        original = self.sep.extract

        def recording(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        with mock.patch.object(self.sep, "extract", recording):
            sources, label = measure_frame_sep(image, fit_fwhm=False)
        self.assertEqual(calls, [])
        self.assertEqual(sources, [])
        self.assertEqual(label.status, "unavailable")
        self.assertIn("invalid_rms_map", label.reasons)
        self.assertEqual(label.background["rms_validity"]["count_below_bound"], diagnostics["count_below_bound"])

    def test_smooth_heteroscedastic_field_stays_measured(self):
        # Flat-fielded vignetting: sigma 5 -> 59 smoothly. No undershoot.
        rng = np.random.default_rng(21)
        ys, xs = np.mgrid[0:512, 0:512]
        sigma = 5.0 + 27.0 * ((xs - 255.5) ** 2 + (ys - 255.5) ** 2) / (255.5**2)
        image = 800.0 + rng.normal(0.0, 1.0, xs.shape) * sigma
        bg = estimate_spatial_background(image)
        self.assertEqual((bg.level_status, bg.noise_status, bg.reasons), ("measured", "measured", ()))
        self.assertEqual(bg.rms_validity["count_below_bound"], 0)
        self.assertGreaterEqual(bg.rms_validity["min_rms"], bg.rms_validity["bound"])
        self.assertEqual(bg.rms_validity["mesh_sigma_method"], "spline_inversion")

    def test_negative_spline_rms_is_invalid_not_consumed(self):
        # A narrow bright ridge inflates the clipped sigma of a few meshes; SEP's
        # bicubic RMS spline then undershoots below zero between meshes.
        rng = np.random.default_rng(5)
        ys, xs = np.mgrid[0:512, 0:512]
        image = 1000.0 + 800.0 * np.exp(-(((xs - 256) / 60.0) ** 2)) + rng.normal(0, 5.0, xs.shape)
        raw = self.sep.Background(image.astype(np.float32))
        self.assertLess(float(raw.rms().min()), 0.0)
        bg = estimate_spatial_background(image)
        self.assertEqual(bg.level_status, "measured")
        self.assertEqual(bg.noise_status, "unresolved")
        self.assertIn("invalid_rms_map", bg.reasons)
        self.assertIsNone(bg.rms_map)


class TestLocalNoiseDetection(SepTestCase):
    """Detection uses err=R(x,y). With CONV the per-pixel threshold is
    thresh_sigma * R(x,y); with the pre-v2 scalar err it was one number."""

    def test_flat_noise_detection_matches_the_scalar_semantics(self):
        rng = np.random.default_rng(0)
        ys, xs = np.mgrid[0:512, 0:512]
        image = 300.0 + rng.normal(0, 10.0, xs.shape)
        positions = [(x, y) for y in range(40, 512, 64) for x in range(40, 512, 64)]
        for k, (cx, cy) in enumerate(positions):
            image += _gaussian(xs, ys, cx, cy, 5.0 + 2.0 * k, 1.5)
        legacy = _legacy_scalar_extract(self.sep, image)
        sources, _ = measure_frame_sep(image, fit_fwhm=False)
        found_local = _matched([(s.x, s.y) for s in sources], positions)
        found_legacy = _matched(np.column_stack([legacy["x"], legacy["y"]]), positions)
        # Same threshold value and semantics on flat noise: at most one marginal
        # source (near threshold) may differ.
        self.assertLessEqual(len(found_local ^ found_legacy), 1)
        # MATCHED at the same number would be ~2.67x more sensitive (see
        # SepDetectionParams); the contract default deliberately is not.
        matched, _ = measure_frame_sep(image, fit_fwhm=False,
                                       detection_params=SepDetectionParams(filter_type="matched"))
        self.assertGreater(len(_matched([(s.x, s.y) for s in matched], positions)), len(found_local) + 5)

    def test_sharp_noise_step_undershoots_and_is_invalid(self):
        """An 8x noise step (sigma 40 | 5) between neighbouring meshes. SEP's
        natural-cubic RMS spline undershoots into the quiet side to ~1.6, a
        third of the true sigma 5 there, while staying positive: a 5-sigma
        threshold on that map would really be ~1.6 sigma. Decision D-018 makes
        this invalid_rms_map (this fixture was, before D-018, the
        equal-significance test, which now uses a 3x step)."""
        rng = np.random.default_rng(11)
        ys, xs = np.mgrid[0:512, 0:512]
        sigma = np.where(xs < 320, 40.0, 5.0)
        image = 1000.0 + rng.normal(0.0, 1.0, xs.shape) * sigma
        raw = self.sep.Background(image.astype(np.float32)).rms()
        quiet = xs >= 320
        self.assertGreater(float(raw[quiet].min()), 0.0)   # the old zero/negative rules do not fire
        self.assertLess(float(raw[quiet].min()), 0.5 * 5.0)  # but R is < half the true sigma
        bg = estimate_spatial_background(image)
        self.assertEqual((bg.level_status, bg.noise_status), ("measured", "unresolved"))
        self.assertEqual(bg.reasons, ("invalid_rms_map",))
        self.assertGreater(bg.rms_validity["count_below_bound"], 0)

    def test_heteroscedastic_halves_equal_significance(self):
        """Noisy region (sigma 15) on 5/8 of the frame, quiet (sigma 5) on 3/8.
        Every star has the same peak significance, 12 x its local sigma. A 3x
        step keeps SEP's RMS spline above the D-018 bound (the 8x step of the
        pre-D-018 version undershoots: test_sharp_noise_step_undershoots...)."""
        rng = np.random.default_rng(11)
        ys, xs = np.mgrid[0:512, 0:512]
        sigma = np.where(xs < 320, 15.0, 5.0)
        image = 1000.0 + rng.normal(0.0, 1.0, xs.shape) * sigma
        noisy = [(x, y) for y in (64, 160, 256, 352, 448) for x in (64, 160, 256)]
        quiet = [(x, y) for y in (64, 160, 256, 352, 448) for x in (416, 464)]
        for cx, cy in noisy:
            image += _gaussian(xs, ys, cx, cy, 12.0 * 15.0, 1.5)
        for cx, cy in quiet:
            image += _gaussian(xs, ys, cx, cy, 12.0 * 5.0, 1.5)

        self.assertEqual(estimate_spatial_background(image).noise_status, "measured")
        sources, label = measure_frame_sep(image, fit_fwhm=False)
        xy = [(s.x, s.y) for s in sources]
        self.assertEqual(len(_matched(xy, noisy)), len(noisy))
        self.assertEqual(len(_matched(xy, quiet)), len(quiet))
        false_local = [p for p in xy if not _matched([p], noisy + quiet, tol=3.0) and min(
            np.hypot(p[0] - tx, p[1] - ty) for tx, ty in noisy + quiet) > 3.0]
        self.assertLessEqual(len(false_local), 2)

        # Pre-v2 scalar err: globalrms is the median mesh sigma (~15 here), so
        # the quiet region's threshold is 3x too high and its stars vanish.
        legacy = _legacy_scalar_extract(self.sep, image)
        legacy_xy = np.column_stack([legacy["x"], legacy["y"]])
        self.assertEqual(len(_matched(legacy_xy, noisy)), len(noisy))
        self.assertEqual(len(_matched(legacy_xy, quiet)), 0)

    def test_smooth_noise_gradient_flat_fielded_vignetting(self):
        """Flat-fielding a vignetted frame leaves the sky level flat but
        amplifies noise toward the corners: R(x,y) varies smoothly (5 -> ~60)
        while B(x,y) is constant. Equal-significance stars across the field."""
        rng = np.random.default_rng(21)
        ys, xs = np.mgrid[0:512, 0:512]
        r2 = ((xs - 255.5) ** 2 + (ys - 255.5) ** 2) / (255.5**2)
        sigma = 5.0 + 27.0 * r2
        image = 800.0 + rng.normal(0.0, 1.0, xs.shape) * sigma
        positions = [(x, y) for y in range(48, 512, 104) for x in range(48, 512, 104)]
        for cx, cy in positions:
            local = 5.0 + 27.0 * ((cx - 255.5) ** 2 + (cy - 255.5) ** 2) / (255.5**2)
            image += _gaussian(xs, ys, cx, cy, 20.0 * local, 1.5)
        sources, _ = measure_frame_sep(image, fit_fwhm=False)
        self.assertEqual(len(_matched([(s.x, s.y) for s in sources], positions)), len(positions))
        legacy = _legacy_scalar_extract(self.sep, image)
        legacy_found = _matched(np.column_stack([legacy["x"], legacy["y"]]), positions)
        # The scalar threshold is 5 x globalrms (the median mesh sigma) everywhere,
        # so it loses exactly the stars whose local sigma is far below it: the
        # quiet centre. Completeness then depends on position, not significance.
        globalrms = float(self.sep.Background(np.ascontiguousarray(image, dtype=np.float32)).globalrms)
        missed = set(range(len(positions))) - legacy_found
        centre = positions.index((256, 256))
        self.assertIn(centre, missed)
        for k in missed:
            cx, cy = positions[k]
            self.assertLess(5.0 + 27.0 * ((cx - 255.5) ** 2 + (cy - 255.5) ** 2) / (255.5**2), globalrms)


class TestMaskPropagation(SepTestCase):
    def _frame(self):
        rng = np.random.default_rng(7)
        ys, xs = np.mgrid[0:256, 0:256]
        image = 400.0 + rng.normal(0, 6.0, xs.shape)
        stars = [(64.0, 64.0), (192.0, 64.0), (64.0, 192.0), (192.0, 192.0)]
        for cx, cy in stars:
            image += _gaussian(xs, ys, cx, cy, 3000.0, 1.8)
        return image, stars

    def test_one_detector_mask_reaches_every_sep_call(self):
        image, _ = self._frame()
        image[10, 10] = np.nan
        image[200, 30] = np.inf
        bad = np.zeros(image.shape, dtype=bool)
        bad[120:123, 120:200] = True
        expected = bad | ~np.isfinite(image)

        seen = {}
        sep = self.sep
        originals = {name: getattr(sep, name) for name in ("Background", "extract", "flux_radius", "sum_circle")}

        def recorder(name):
            def wrapped(*args, **kwargs):
                seen.setdefault(name, []).append(np.asarray(kwargs.get("mask")).astype(bool))
                return originals[name](*args, **kwargs)
            return wrapped

        with mock.patch.multiple(sep, **{name: recorder(name) for name in originals}):
            sources, label = measure_frame_sep(image, detector_mask=bad, fit_fwhm=False)
        self.assertEqual(set(seen), set(originals))
        for name, masks in seen.items():
            for received in masks:
                np.testing.assert_array_equal(received, expected, err_msg=name)
        self.assertEqual(label.status, "measured")
        self.assertTrue(all(np.isfinite(s.hfr_px) for s in sources if s.valid))
        self.assertTrue(np.isfinite(label.sky_median_adu) and np.isfinite(label.sky_rms_adu))
        self.assertEqual(label.star_count_used, 4)

    def test_star_whose_wing_touches_a_bad_pixel_is_rejected(self):
        image, stars = self._frame()
        bad = np.zeros(image.shape, dtype=bool)
        bad[64, 69] = True  # 5 px into the wing of the first star; its core is clean
        sources, label = measure_frame_sep(image, detector_mask=bad, fit_fwhm=False)
        by_star = {k: s for s in sources for k, (cx, cy) in enumerate(stars) if np.hypot(s.x - cx, s.y - cy) < 2}
        self.assertEqual(set(by_star), {0, 1, 2, 3})
        self.assertIn(Rejection.MASKED_SUPPORT.value, by_star[0].rejections)
        self.assertFalse(by_star[0].valid)
        self.assertTrue(all(by_star[k].valid for k in (1, 2, 3)))
        self.assertEqual(label.rejection_counts.get("masked_support"), 1)


class TestSaturationFootprint(SepTestCase):
    def _clipped_star_frame(self):
        rng = np.random.default_rng(9)
        ys, xs = np.mgrid[0:256, 0:256]
        image = 500.0 + rng.normal(0, 8.0, xs.shape)
        image += _gaussian(xs, ys, 80.0, 80.0, 3000.0, 1.8)          # clean reference star
        image += _gaussian(xs, ys, 176.0, 176.0, 80000.0, 1.8)       # true peak far above full well
        full_well = 16383.0
        saturated = image >= full_well
        image = np.minimum(image, full_well)
        return image, saturated, full_well

    def test_clipped_star_with_masked_core_is_rejected(self):
        image, saturated, full_well = self._clipped_star_frame()
        self.assertGreater(int(saturated.sum()), 4)
        # Mask the core away entirely (some pipelines zero it); only the
        # apparently clean wings remain visible to SEP. With the core masked
        # (zero data, infinite noise inside sep.extract) the wings form a ring
        # that SEP deblends into several children around the hole. The blend
        # and shape gates are opened here so that only the footprint
        # saturation check can reject them: it must, for every child.
        image = image.copy()
        image[saturated] = 0.0
        sources, label = measure_frame_sep(
            image,
            native_saturation_mask=saturated,
            fit_fwhm=False,
            max_ellipticity=1.0,
            detection_params=SepDetectionParams(accept_deblended_for_focus=True),
        )
        near = [s for s in sources if np.hypot(s.x - 176.0, s.y - 176.0) < 8.0]
        self.assertGreaterEqual(len(near), 1)
        for s in near:
            self.assertIn(Rejection.SATURATED.value, s.rejections)
            self.assertEqual(s.saturation_evidence, ["native_mask"])
            self.assertFalse(s.valid)
            self.assertLess(s.peak, full_well)  # the visible wings alone look unsaturated
        self.assertEqual(label.saturation_gate, "native_mask_applied")
        self.assertIsNone(label.saturation_threshold)
        clean = min(sources, key=lambda s: np.hypot(s.x - 80.0, s.y - 80.0))
        self.assertTrue(clean.valid)
        self.assertEqual(label.star_count_used, 1)

    def test_without_evidence_the_clipped_star_escapes_and_the_gate_says_so(self):
        # Demonstrates why the footprint check matters: with no native mask and
        # an unknown saturation level the flat-topped star becomes a "valid"
        # (artificially small) width, and the label must record that no gate ran.
        image, _, _ = self._clipped_star_frame()
        sources, label = measure_frame_sep(image, fit_fwhm=False)
        bright = min(sources, key=lambda s: np.hypot(s.x - 176.0, s.y - 176.0))
        self.assertTrue(bright.valid)
        self.assertEqual(label.saturation_gate, "not_performed")
        self.assertIsNone(label.saturation_threshold)
        # A declared level gates it through the raw footprint peak instead.
        sources, label = measure_frame_sep(image, saturation_level=0.98 * 16383.0, fit_fwhm=False)
        bright = min(sources, key=lambda s: np.hypot(s.x - 176.0, s.y - 176.0))
        self.assertEqual(bright.saturation_evidence, ["scalar_threshold"])
        self.assertEqual(label.saturation_gate, "applied")


class TestLocalMoffatFit(SepTestCase):
    def test_moffat_on_gradient_uses_local_background(self):
        """25 Moffat stars (alpha 3, beta 2.5, peak 3000) on a 4 ADU/px sky
        gradient, sigma 5 noise.

        Tolerance: each local fit within 5% of the analytic FWHM. This is a
        regression bound set after one exploratory run of this exact fixture
        (worst local error 3.7%), not a derived accuracy limit; the residual
        error comes from star wings contaminating SEP's 64-px meshes. The
        comparison that matters is relative: the pre-v2 path (raw crop minus
        globalback, free constant offset) cannot remove a tilt that varies by
        ~190 ADU across the 48-px crop, and its median error must be several
        times the local one.
        """
        alpha, beta = 3.0, 2.5
        truth = 2.0 * alpha * np.sqrt(2.0 ** (1.0 / beta) - 1.0)
        rng = np.random.default_rng(5)
        ys, xs = np.mgrid[0:512, 0:512]
        image = 1000.0 + 4.0 * xs
        positions = [(x, y) for y in range(64, 512, 96) for x in range(64, 512, 96)]
        for cx, cy in positions:
            image = image + 3000.0 * (1.0 + ((xs - cx) ** 2 + (ys - cy) ** 2) / alpha**2) ** (-beta)
        image = image + rng.normal(0, 5.0, image.shape)

        sources, label = measure_frame_sep(image)
        fitted = [s for s in sources if s.fwhm_model == "moffat"]
        self.assertEqual(len(fitted), len(positions))
        self.assertTrue(all(s.fit_converged and s.moffat_offset_mode == "fixed_zero"
                            and s.moffat_residual_offset is None for s in fitted))
        local_err = np.array([abs(s.fwhm_px / truth - 1.0) for s in fitted])
        self.assertLess(local_err.max(), 0.05)
        self.assertEqual(label.fwhm_moffat_count, len(positions))
        self.assertAlmostEqual(label.fwhm_px_median, truth, delta=0.05 * truth)

        globalback = float(self.sep.Background(image.astype(np.float32)).globalback)
        legacy_err = []
        for s in fitted:
            ix, iy = int(np.floor(s.x + 0.5)), int(np.floor(s.y + 0.5))
            fwhm, _ = fit_moffat_fwhm(image[iy - 24:iy + 24, ix - 24:ix + 24], globalback)
            legacy_err.append(abs(fwhm / truth - 1.0) if fwhm is not None else np.inf)
        self.assertGreater(np.median(legacy_err), 3.0 * np.median(local_err))
        self.assertGreater(np.max(legacy_err), 0.2)

    def test_free_offset_is_reported(self):
        rng = np.random.default_rng(6)
        ys, xs = np.mgrid[0:256, 0:256]
        image = 700.0 + rng.normal(0, 5.0, xs.shape) + _gaussian(xs, ys, 128.0, 128.0, 4000.0, 2.0)
        sources, _ = measure_frame_sep(image, moffat_fit_offset=True)
        star = min(sources, key=lambda s: np.hypot(s.x - 128.0, s.y - 128.0))
        self.assertEqual(star.moffat_offset_mode, "free")
        self.assertIsNotNone(star.moffat_residual_offset)
        self.assertLess(abs(star.moffat_residual_offset), 20.0)


class TestGainAndMetadata(SepTestCase):
    def _field(self):
        rng = np.random.default_rng(13)
        ys, xs = np.mgrid[0:256, 0:256]
        image = 300.0 + rng.normal(0, 7.0, xs.shape)
        for cx, cy in ((60, 60), (190, 70), (70, 190), (180, 180)):
            image += _gaussian(xs, ys, cx, cy, 2500.0, 1.7)
        return image

    def test_gain_adds_source_poisson_noise_only(self):
        image = self._field()
        without, label_without = measure_frame_sep(image, fit_fwhm=False)
        with_gain, label_with = measure_frame_sep(image, fit_fwhm=False, gain_e_per_adu=2.0)
        self.assertTrue(label_without.snr_definition.startswith("background_noise_only_proxy"))
        self.assertTrue(label_with.snr_definition.startswith("aperture_snr_rms_map_plus_source_poisson"))
        self.assertEqual(label_with.gain_e_per_adu, 2.0)
        self.assertIsNone(label_without.gain_e_per_adu)
        for a, b in zip(without, with_gain):
            self.assertAlmostEqual(a.flux, b.flux, places=3)
            var_rms = (a.flux / a.snr) ** 2
            var_total = (b.flux / b.snr) ** 2
            # sep.sum_circle adds flux/gain and nothing else (no read noise).
            self.assertAlmostEqual(var_total - var_rms, a.flux / 2.0, delta=1e-3 * var_total)

    def test_frame_label_carries_provenance(self):
        sources, label = measure_frame_sep(self._field())
        payload = json.loads(json.dumps(label.to_dict()))
        self.assertEqual(payload["label_schema_version"], pm.LABEL_SCHEMA_VERSION)
        self.assertEqual(payload["label_method"], "sep_spatial_mesh+sep_extract+sep_flux_radius")
        self.assertEqual(payload["status"], "measured")
        bg = payload["background"]
        self.assertEqual(bg["producer"], "python_sep_label")
        self.assertEqual(bg["schema"], "achernar.python_sep_background_provenance.v1")
        self.assertNotIn("schema_version", bg)  # that key belongs to the Rust BackgroundResult
        self.assertEqual(bg["method"], "sep_spatial_mesh")
        self.assertEqual(bg["library_version"], "1.4.1")
        self.assertEqual(bg["config_sha256"], DEFAULT_BACKGROUND_SHA256)
        self.assertEqual(bg["noise_source"], "empirical_mesh_rms")
        self.assertEqual((bg["level_status"], bg["noise_status"]), ("measured", "measured"))
        self.assertIn("median", bg["level_definition"])
        self.assertEqual(payload["detection"]["config_sha256"], DEFAULT_DETECTION_SHA256)
        self.assertEqual(payload["detection"]["config"]["filter_type"], "conv")
        self.assertEqual(set(payload["width_definition"]), {"hfr", "hfd", "fwhm"})
        self.assertEqual(payload["hfd_px_median"], 2.0 * payload["hfr_px_median"])
        self.assertEqual(payload["units"], "adu")
        self.assertEqual(payload["saturation_gate"], "not_performed")
        self.assertIsInstance(payload["rejection_counts"], dict)
        self.assertIsInstance(payload["sep_flag_counts"], dict)
        for s in sources:
            self.assertEqual(s.hfd_px, 2.0 * s.hfr_px if s.hfr_px is not None else None)
            self.assertGreater(s.local_rms, 0.0)
            self.assertIsNotNone(s.local_background)

    def test_deblending_and_acceptance_are_separate(self):
        rng = np.random.default_rng(17)
        ys, xs = np.mgrid[0:256, 0:256]
        image = 300.0 + rng.normal(0, 5.0, xs.shape)
        image += _gaussian(xs, ys, 124.0, 128.0, 3000.0, 1.6) + _gaussian(xs, ys, 132.0, 128.0, 3000.0, 1.6)
        sources, label = measure_frame_sep(image, fit_fwhm=False)
        children = [s for s in sources if s.deblended]
        self.assertEqual(len(children), 2)
        # A policy-rejected SEP child is "deblended" (Rust StarRejectionTallies
        # .deblended); "blended" stays the crop two-peak heuristic.
        self.assertTrue(all(s.rejections == [Rejection.DEBLENDED.value] for s in children))
        self.assertEqual(label.rejection_counts.get("deblended"), 2)
        self.assertNotIn("blended", label.rejection_counts)
        self.assertEqual(label.sep_flag_counts.get("obj_merged"), 2)
        accepting = SepDetectionParams(accept_deblended_for_focus=True)
        sources, label = measure_frame_sep(image, fit_fwhm=False, detection_params=accepting)
        children = [s for s in sources if s.deblended]
        self.assertEqual(len(children), 2)
        self.assertFalse(any(Rejection.DEBLENDED.value in s.rejections for s in children))
        self.assertEqual(label.detection["config_sha256"], accepting.config_sha256())

    def test_source_mask_pass_is_off_by_default_and_never_enters_the_detector_mask(self):
        image = self._field()
        _, label = measure_frame_sep(image, fit_fwhm=False)
        self.assertFalse(label.background["source_mask_applied"])
        self.assertFalse(label.detection["source_mask_background"]["enabled"])

        seen = []
        original = self.sep.extract

        def recording_extract(*args, **kwargs):
            seen.append(np.asarray(kwargs["mask"]).astype(bool))
            return original(*args, **kwargs)

        with mock.patch.object(self.sep, "extract", recording_extract):
            _, label = measure_frame_sep(image, fit_fwhm=False, source_mask_background=True)
        self.assertTrue(label.background["source_mask_applied"])
        self.assertGreater(label.background["source_masked_fraction"], 0.0)
        self.assertEqual(len(seen), 2)
        for received in seen:
            self.assertFalse(received.any())  # detector mask only: nothing is bad here


class TestRmsValidityFallback(unittest.TestCase):
    def test_map_that_is_not_a_sep_spline_uses_the_node_sample_approximation(self):
        rng = np.random.default_rng(0)
        rms = (5.0 + rng.random((256, 256))).astype(np.float32)
        reasons, diagnostics = rms_map_validity(rms, np.ones(rms.shape, bool), (64, 64))
        self.assertEqual(diagnostics["mesh_sigma_method"], "node_pixel_sample")
        self.assertGreater(diagnostics["reconstruction_max_rel_error"], 1e-4)
        self.assertEqual(reasons, [])  # every value lies in [5, 6): above half of any node
        rms[100, 100] = 2.0  # below 0.5 x min node (>= 5)
        reasons, diagnostics = rms_map_validity(rms, np.ones(rms.shape, bool), {"bw": 64, "bh": 64})
        self.assertEqual((reasons, diagnostics["count_below_bound"]), (["invalid_rms_map"], 1))


class TestFwhmPooling(unittest.TestCase):
    def test_fallback_fwhm_is_never_pooled_with_moffat(self):
        def star(fwhm, model):
            return StarMeasurement(0, 0, 1, 1, 10.0, 2.0, fwhm, 2.5 if model == "moffat" else None, 0.0, True, [],
                                   fwhm_model=model, fit_converged=model == "moffat")
        stars = [star(3.0, "moffat"), star(3.2, "moffat"), star(3.1, "moffat"),
                 star(9.0, "gaussian_moment_fallback"), star(9.5, "gaussian_moment_fallback")]
        label = aggregate_frame(stars, sky_median=100.0, sky_rms=5.0)
        self.assertAlmostEqual(label.fwhm_px_median, 3.1)
        self.assertEqual((label.fwhm_moffat_count, label.fwhm_fallback_count), (3, 2))
        self.assertAlmostEqual(label.fwhm_fallback_px_median, 9.25)


if __name__ == "__main__":
    unittest.main(verbosity=2)
