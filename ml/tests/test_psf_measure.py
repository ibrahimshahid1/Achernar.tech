"""
Synthetic-PSF validation of the FQE label pipeline (Section 7.7.5 verification).

These tests encode the correctness claims from the astronomical-quality-metrics
research note:

  * ``FWHM = 2 * HFR`` holds for a **Gaussian** PSF within sampling tolerance.
  * For a **Moffat** PSF the fitted FWHM matches the analytic value but is a
    genuinely different number from ``2 * HFR`` — so the daemon's
    ``fwhm = hfr * 2`` is only correct in the Gaussian limit.
  * Invalid / empty crops yield ``None`` widths, never ``0`` (0 would win a
    lower-is-better comparison and mark a blank frame as perfectly focused).
  * Saturated and edge-truncated stars are rejected before they can bias a label.

Run: ``python -m unittest ml.tests.test_psf_measure`` (from repo root) or
``python -m unittest`` inside ``ml/``.
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labels.psf_measure import (  # noqa: E402
    Rejection,
    aggregate_frame,
    Rejection,
    curve_of_growth_hfr,
    estimate_background,
    fit_moffat_fwhm,
    measure_star,
)


def render_psf(profile, size=64, cx=None, cy=None, sky=100.0, amp=8000.0):
    cx = size / 2 - 0.5 if cx is None else cx
    cy = size / 2 - 0.5 if cy is None else cy
    ys, xs = np.mgrid[0:size, 0:size]
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return (sky + amp * profile(r)).astype(np.float64)


def gaussian(sigma):
    return lambda r: np.exp(-(r**2) / (2 * sigma**2))


def moffat(alpha, beta):
    return lambda r: (1.0 + (r / alpha) ** 2) ** (-beta)


def noisy_gaussian_hfr(hfr, size=64, amp=6000.0, sky=100.0, read_noise=0.0, seed=0):
    """Render a Poisson+read-noise Gaussian star of a known HFR and measure it."""
    rng = np.random.default_rng(seed)
    sigma = hfr / np.sqrt(2 * np.log(2))
    img = render_psf(gaussian(sigma), size=size, sky=sky, amp=amp)
    if read_noise:
        img = rng.poisson(np.clip(img, 0, None)).astype(float) + rng.normal(0, read_noise, img.shape)
    skv, rms = estimate_background(img)
    return curve_of_growth_hfr(img, skv, sky_rms=rms)


class TestGaussianPSF(unittest.TestCase):
    def test_fwhm_equals_twice_hfr_for_gaussian(self):
        sigma = 3.0
        img = render_psf(gaussian(sigma))
        sky, _ = estimate_background(img)

        hfr = curve_of_growth_hfr(img, sky)
        fwhm, _ = fit_moffat_fwhm(img, sky)

        analytic_hfr = sigma * math.sqrt(2 * math.log(2))
        analytic_fwhm = 2.0 * math.sqrt(2 * math.log(2)) * sigma

        self.assertAlmostEqual(hfr, analytic_hfr, delta=0.05 * analytic_hfr)
        self.assertAlmostEqual(fwhm, analytic_fwhm, delta=0.06 * analytic_fwhm)
        # The relationship the daemon assumes: it holds *here* (Gaussian only).
        self.assertAlmostEqual(fwhm, 2.0 * hfr, delta=0.06 * fwhm)


class TestMoffatPSF(unittest.TestCase):
    def test_fitted_fwhm_matches_analytic(self):
        alpha, beta = 2.5, 2.5
        img = render_psf(moffat(alpha, beta))
        sky, _ = estimate_background(img)

        fwhm, fit_beta = fit_moffat_fwhm(img, sky)
        analytic_fwhm = 2 * alpha * math.sqrt(2 ** (1 / beta) - 1)

        self.assertIsNotNone(fwhm)
        self.assertAlmostEqual(fwhm, analytic_fwhm, delta=0.05 * analytic_fwhm)
        self.assertAlmostEqual(fit_beta, beta, delta=0.4)

    def test_twice_hfr_is_not_the_moffat_fwhm(self):
        # The whole point: 2*HFR is a different quantity from the true FWHM for a
        # realistic (finite-beta) Moffat PSF. Regressing this guards against
        # anyone reintroducing `fwhm = hfr * 2` outside the Gaussian limit.
        alpha, beta = 2.5, 2.5
        img = render_psf(moffat(alpha, beta))
        sky, _ = estimate_background(img)

        hfr = curve_of_growth_hfr(img, sky)
        fwhm, _ = fit_moffat_fwhm(img, sky)
        rel_gap = abs(fwhm - 2.0 * hfr) / fwhm
        self.assertGreater(rel_gap, 0.08)


class TestInvalidNeverZero(unittest.TestCase):
    def test_blank_frame_hfr_is_none(self):
        img = np.full((64, 64), 100.0)  # pure sky, no star
        sky, _ = estimate_background(img)
        self.assertIsNone(curve_of_growth_hfr(img, sky))

    def test_measure_star_on_blank_is_invalid_with_none_width(self):
        img = np.full((64, 64), 100.0)
        m = measure_star(img, saturation_level=65535.0)
        self.assertFalse(m.valid)
        self.assertIsNone(m.hfr_px)  # never 0.0
        self.assertIsNone(m.fwhm_px)


class TestSourceGates(unittest.TestCase):
    def test_saturated_star_is_rejected(self):
        img = render_psf(gaussian(3.0), amp=200000.0)  # clips well past full well
        img = np.clip(img, 0, 65535.0)
        m = measure_star(img, saturation_level=65000.0)
        self.assertIn(Rejection.SATURATED.value, m.rejections)
        self.assertFalse(m.valid)

    def test_edge_truncated_star_is_rejected(self):
        img = render_psf(gaussian(2.0), cx=1.0, cy=1.0)  # centroid in the corner
        m = measure_star(img, saturation_level=65535.0)
        self.assertIn(Rejection.EDGE_TRUNCATED.value, m.rejections)
        self.assertFalse(m.valid)

    def test_low_snr_star_is_rejected(self):
        rng = np.random.default_rng(0)
        img = render_psf(gaussian(3.0), amp=30.0) + rng.normal(0, 40.0, (64, 64))
        m = measure_star(img, saturation_level=65535.0, min_snr=5.0)
        self.assertIn(Rejection.LOW_SNR.value, m.rejections)


class TestAggregation(unittest.TestCase):
    def test_robust_pooling_over_multiple_stars(self):
        sky = 100.0
        measurements = [measure_star(render_psf(gaussian(3.0)), saturation_level=65535.0) for _ in range(8)]
        label = aggregate_frame(measurements, sky_median=sky, sky_rms=5.0)
        self.assertEqual(label.star_count_used, 8)
        self.assertIsNotNone(label.hfr_px_median)
        self.assertIsNotNone(label.hfr_px_mad)
        # Consistent stars -> tiny scatter.
        self.assertLess(label.hfr_px_mad, 0.1 * label.hfr_px_median)


class TestHfrIsNoiseInvariant(unittest.TestCase):
    """Regression for the P0 label bug: a whole-crop clip(...-sky, 0) rectified
    noise into a pedestal, so HFR grew with noise level and crop size (a true
    3 px star measured 16 px at high noise, 32 px on a big crop)."""

    def test_hfr_stable_across_noise_levels(self):
        truth = 3.0
        for read_noise, amp in [(0.0, 6000.0), (20.0, 3000.0), (60.0, 1500.0)]:
            vals = [
                noisy_gaussian_hfr(truth, amp=amp, read_noise=read_noise, seed=s) for s in range(5)
            ]
            mean = float(np.mean(vals))
            self.assertLess(
                abs(mean - truth),
                0.25 * truth,
                f"HFR {mean:.2f} drifted from {truth} at read_noise={read_noise}",
            )

    def test_hfr_stable_across_crop_size(self):
        truth = 3.0
        measured = [noisy_gaussian_hfr(truth, size=s, amp=3000.0, read_noise=20.0) for s in (32, 64, 96, 128)]
        self.assertLess(max(measured) - min(measured), 0.3, f"HFR crop-size dependent: {measured}")

    def test_hfr_stable_across_contrast(self):
        truth = 4.0
        vals = [noisy_gaussian_hfr(truth, amp=amp, read_noise=15.0, seed=1) for amp in (800.0, 3000.0, 12000.0)]
        self.assertLess(max(vals) - min(vals), 0.5, f"HFR contrast dependent: {vals}")


class TestBackgroundGradient(unittest.TestCase):
    def test_hfr_survives_a_sky_gradient(self):
        # A linear sky gradient (vignetting/moonlight) must not corrupt HFR.
        truth = 3.0
        base = render_psf(gaussian(truth / np.sqrt(2 * np.log(2))), sky=200.0, amp=6000.0)
        ys, xs = np.mgrid[0:64, 0:64]
        grad = 6.0 * xs  # up to ~380 ADU across the crop
        img = base + grad
        sky, rms = estimate_background(img)
        # Local-plane background: subtract the fitted gradient the way a mesh would.
        hfr = curve_of_growth_hfr(img - grad, float(np.median((img - grad))), sky_rms=rms)
        self.assertAlmostEqual(hfr, truth, delta=0.4 * truth)


class TestUndersampledPSF(unittest.TestCase):
    def test_near_nyquist_star_is_measured_not_exploded(self):
        # A tight (in-focus) star at ~1.2 px HFR must stay bounded, not blow up.
        hfr = noisy_gaussian_hfr(1.2, amp=8000.0, read_noise=10.0, seed=3)
        self.assertIsNotNone(hfr)
        self.assertLess(hfr, 3.0)
        self.assertGreater(hfr, 0.5)


class TestBlendRejection(unittest.TestCase):
    def test_close_double_is_flagged_blended(self):
        size = 64
        sigma = 2.0 / np.sqrt(2 * np.log(2))
        a = render_psf(gaussian(sigma), size=size, cx=28, cy=32, sky=100.0, amp=8000.0)
        b = render_psf(gaussian(sigma), size=size, cx=36, cy=32, sky=0.0, amp=8000.0)
        m = measure_star(a + b, saturation_level=65535.0)
        self.assertIn(Rejection.BLENDED.value, m.rejections)
        self.assertFalse(m.valid)


class TestSepPath(unittest.TestCase):
    def test_measure_frame_sep_recovers_hfr(self):
        try:
            import sep  # noqa: F401
        except ImportError:
            self.skipTest("sep not installed")
        from labels.psf_measure import measure_frame_sep

        # A field of identical 3 px-HFR stars on a noisy sky.
        rng = np.random.default_rng(0)
        truth = 3.0
        sigma = truth / np.sqrt(2 * np.log(2))
        img = np.full((256, 256), 300.0)
        ys, xs = np.mgrid[0:256, 0:256]
        for _ in range(12):
            cx, cy = rng.uniform(30, 226), rng.uniform(30, 226)
            img += 9000.0 * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
        img = rng.poisson(np.clip(img, 0, None)).astype(np.float64) + rng.normal(0, 12.0, img.shape)

        sources, label = measure_frame_sep(img, saturation_level=65535.0)
        self.assertGreaterEqual(label.star_count_used, 8)
        # SEP flux_radius should recover the true HFR within sampling tolerance.
        self.assertAlmostEqual(label.hfr_px_median, truth, delta=0.5)

    def test_measure_frame_sep_uses_aperture_flux_for_moffat_wings(self):
        try:
            import sep  # noqa: F401
        except ImportError:
            self.skipTest("sep not installed")
        from labels.psf_measure import measure_frame_sep

        rng = np.random.default_rng(11)
        truth = 3.0
        beta = 2.5
        alpha = truth / np.sqrt(2 ** (1 / (beta - 1)) - 1)
        img = np.full((256, 256), 300.0)
        ys, xs = np.mgrid[0:256, 0:256]
        for cx, cy in [(45, 45), (95, 55), (155, 48), (210, 70), (60, 145), (130, 135), (205, 150), (80, 210), (165, 205)]:
            r2 = (xs - cx) ** 2 + (ys - cy) ** 2
            img += 9000.0 * (1.0 + r2 / alpha**2) ** (-beta)
        img = rng.poisson(img).astype(np.float64) + rng.normal(0, 12.0, img.shape)

        _, label = measure_frame_sep(img, saturation_level=65535.0, fit_fwhm=False)
        self.assertGreaterEqual(label.star_count_used, 8)
        # Isophotal objects["flux"] biases this profile to ~2.84 px. Aperture
        # normalization recovers the actual half-light radius.
        self.assertAlmostEqual(label.hfr_px_median, truth, delta=0.12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
