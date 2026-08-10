#!/usr/bin/env python3
"""
Tests for tools/photon_transfer.py against simulated sensors.

The point of simulating is that the answer is known. We generate frames from a
sensor with a chosen gain, read noise, black level and full well, then check the
tool recovers those numbers — which is the only way to know the estimator is
right before pointing it at a camera whose true values nobody knows.

Noise model, with gain g in e-/ADU:

    ADU = black + (Poisson(S) + Normal(0, sigma_read)) / g

so   signal_ADU  = S / g
     var_ADU     = signal_ADU / g + sigma_read^2 / g^2

giving a PTC of slope 1/g, which is what `fit_linear_region` inverts.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from photon_transfer import (  # noqa: E402
    fit_linear_region,
    load_cfa_plane,
    measure,
    pair_statistics,
)

TRUE_GAIN = 0.80          # e-/ADU
TRUE_READ_E = 4.0         # electrons
TRUE_BLACK = 2048.0       # ADU
TRUE_FULL_WELL_E = 32000.0
SATURATION_ADU = TRUE_BLACK + TRUE_FULL_WELL_E / TRUE_GAIN   # 42048 ADU


def synth_frame(signal_e: float, rng: np.random.Generator, size: int = 256) -> np.ndarray:
    electrons = rng.poisson(signal_e, size=(size, size)).astype(np.float64)
    electrons += rng.normal(0.0, TRUE_READ_E, size=(size, size))
    adu = TRUE_BLACK + electrons / TRUE_GAIN
    return np.clip(adu, 0, SATURATION_ADU).astype(np.uint16)


def write16(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


class TestFit(unittest.TestCase):
    def test_recovers_slope_on_clean_line(self):
        signal = np.array([100.0, 200.0, 400.0, 800.0, 1600.0])
        gain = 0.5
        variance = signal / gain + 25.0
        recovered, intercept, n = fit_linear_region(signal, variance)
        self.assertAlmostEqual(recovered, gain, places=4)
        self.assertAlmostEqual(intercept, 25.0, places=3)
        self.assertEqual(n, 5)

    def test_stops_at_the_knee(self):
        signal = np.array([100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0])
        variance = signal / 0.5 + 25.0
        variance[-1] = 50.0          # turnover: pixels pinned at full well
        _, _, n = fit_linear_region(signal, variance)
        self.assertLess(n, 6)

    def test_rejects_a_non_increasing_curve(self):
        signal = np.array([100.0, 200.0, 400.0])
        variance = np.array([500.0, 400.0, 300.0])
        with self.assertRaises(RuntimeError):
            fit_linear_region(signal, variance)

    def test_pair_statistics_difference_cancels_fixed_pattern(self):
        rng = np.random.default_rng(0)
        base = rng.normal(1000.0, 50.0, size=(128, 128))     # fixed pattern
        a = base + rng.normal(0.0, 10.0, size=(128, 128))    # temporal noise
        b = base + rng.normal(0.0, 10.0, size=(128, 128))
        _, diff_var = pair_statistics(a, b)
        # Difference variance sees only temporal noise: 2 * 10^2 = 200,
        # not the ~2500 of the fixed pattern.
        self.assertLess(diff_var, 400)
        self.assertGreater(diff_var, 100)


class TestMeasure(unittest.TestCase):
    def _build(self, d: Path, levels_e):
        bias_dir, flat_dir = d / "bias", d / "flats"
        bias_dir.mkdir()
        flat_dir.mkdir()
        rng = np.random.default_rng(42)
        for i in range(4):
            write16(bias_dir / f"bias_{i:02d}.png", synth_frame(0.0, rng))
        for i, level in enumerate(levels_e):
            write16(flat_dir / f"lvl{i:02d}_a.png", synth_frame(level, rng))
            write16(flat_dir / f"lvl{i:02d}_b.png", synth_frame(level, rng))
        return sorted(bias_dir.iterdir()), sorted(flat_dir.iterdir())

    def test_recovers_the_simulated_sensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            levels = [500, 1500, 3000, 6000, 10000, 16000, 24000]
            bias, flats = self._build(Path(tmp), levels)
            r = measure(bias, flats, crop=128)

            self.assertAlmostEqual(r["gain_e_per_adu"], TRUE_GAIN, delta=0.06)
            self.assertAlmostEqual(r["read_noise_e"], TRUE_READ_E, delta=0.8)
            self.assertAlmostEqual(r["black_level_adu"], TRUE_BLACK, delta=2.0)

    def test_black_level_is_measured_not_assumed(self):
        """Canon bodies sit near 2048; the tool must not hard-code that."""
        with tempfile.TemporaryDirectory() as tmp:
            bias, flats = self._build(Path(tmp), [500, 1500, 3000, 6000])
            r = measure(bias, flats, crop=128)
            self.assertNotEqual(r["black_level_adu"], 0.0)
            self.assertAlmostEqual(r["black_level_adu"], TRUE_BLACK, delta=2.0)

    def test_flags_a_ladder_that_never_saturated(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Top level is well under full well.
            bias, flats = self._build(Path(tmp), [500, 1500, 3000, 6000, 9000])
            r = measure(bias, flats, crop=128)
            self.assertTrue(r["saturation_is_lower_bound"])

    def test_detects_saturation_when_the_ladder_reaches_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            levels = [1000, 4000, 8000, 16000, 26000, 31000, 33000, 36000]
            bias, flats = self._build(Path(tmp), levels)
            r = measure(bias, flats, crop=128)
            self.assertFalse(r["saturation_is_lower_bound"])
            # The recovered clip point should be near the true one, and must
            # NOT be the 16-bit container ceiling.
            self.assertLess(r["saturation_adu"], 65535)
            self.assertGreater(r["saturation_adu"], TRUE_BLACK)

    def test_rejects_too_few_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            bias, flats = self._build(Path(tmp), [500, 1500])
            with self.assertRaises(RuntimeError):
                measure(bias, flats, crop=128)          # only 4 flats
            with self.assertRaises(RuntimeError):
                measure(bias[:1], flats, crop=128)      # only 1 bias

    def test_rejects_demosaiced_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rgb.png"
            Image.new("RGB", (32, 32), (5, 5, 5)).save(path)
            with self.assertRaises(RuntimeError) as ctx:
                load_cfa_plane(path)
            self.assertIn("demosaiced", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
