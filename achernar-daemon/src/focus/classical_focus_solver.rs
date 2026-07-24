//! Classical focus solver — Section 7.3.4.
//! HFD/HFR computation + parabola/V-curve fitting.

use serde::{Deserialize, Serialize};

/// Result of a classical curve fit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveFitResult {
    pub optimal_position: f64,
    pub predicted_hfr: f64,
    pub r_squared: f64,
    pub coefficients: Vec<f64>,
    pub method: CurveFitMethod,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CurveFitMethod {
    Parabola,
    VCurve,
    Hyperbola,
}

/// Classical focus metric solver using HFD/HFR.
pub struct ClassicalFocusSolver;

impl ClassicalFocusSolver {
    /// Compute HFR (Half-Flux Radius, px) from a star image crop.
    ///
    /// Two properties this guarantees (matching the corrected Python labeler in
    /// `ml/labels/psf_measure.py`, and unlike the previous implementation):
    ///
    /// * **No noise pedestal.** The sky-subtracted flux is integrated *without*
    ///   clipping negatives to zero, so background noise cancels in the mean. The
    ///   old `(pixel - sky).max(0.0)` rectified every positive noise excursion
    ///   into a pedestal that inflated HFR with the noise level.
    /// * **Bounded aperture.** Integration is limited to a radius adapted to the
    ///   star (~4x an initial estimate), so a larger crop cannot accumulate more
    ///   noise-only annuli into the total.
    ///
    /// Returns `NaN` when no positive stellar flux is present.
    pub fn compute_hfr(image_data: &[f32], width: usize, height: usize) -> f64 {
        if width == 0 || height == 0 || image_data.len() != width * height {
            return f64::NAN;
        }

        let background = percentile(image_data, 0.5) as f64;
        let sky_sigma = robust_sigma(image_data, background);

        // Centroid from the thresholded core only. Clipping is fine for
        // *locating* the star; the bug was clipping when *integrating* its flux.
        let threshold = 3.0 * sky_sigma.max(1e-6);
        let mut core_total = 0.0_f64;
        let mut x_weighted = 0.0_f64;
        let mut y_weighted = 0.0_f64;
        for y in 0..height {
            for x in 0..width {
                let core = (image_data[y * width + x] as f64 - background - threshold).max(0.0);
                if core > 0.0 {
                    core_total += core;
                    x_weighted += x as f64 * core;
                    y_weighted += y as f64 * core;
                }
            }
        }
        if core_total <= f64::EPSILON {
            return f64::NAN;
        }
        let cx = x_weighted / core_total;
        let cy = y_weighted / core_total;

        // (radius, UNCLIPPED residual) for every pixel, sorted by radius.
        let mut radial: Vec<(f64, f64)> = Vec::with_capacity(image_data.len());
        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let resid = image_data[y * width + x] as f64 - background;
                radial.push(((dx * dx + dy * dy).sqrt(), resid));
            }
        }
        radial.sort_by(|a, b| a.0.total_cmp(&b.0));

        let hard_cap = (width.min(height) as f64) / 2.0 - 1.0;
        // Pass 1: rough HFR within a modest cap. Pass 2: bound the aperture to
        // ~4x that estimate so noise-only outer annuli stay out of the integral.
        let est = match cog_half_radius(&radial, hard_cap.min(20.0)) {
            Some((hfr, total)) if total > 0.0 => hfr,
            _ => return f64::NAN,
        };
        let aperture = (4.0 * est).max(6.0).min(hard_cap);
        match cog_half_radius(&radial, aperture) {
            Some((hfr, _)) => hfr,
            None => f64::NAN,
        }
    }

    /// Compute HFD (Half-Flux Diameter) from a star image crop.
    pub fn compute_hfd(image_data: &[f32], width: usize, height: usize) -> f64 {
        Self::compute_hfr(image_data, width, height) * 2.0
    }

    /// Fit a parabola to position-HFR samples and find the minimum.
    pub fn fit_parabola(positions: &[i32], hfr_values: &[f64]) -> Option<CurveFitResult> {
        if positions.len() != hfr_values.len() || positions.len() < 3 {
            return None;
        }

        let samples: Vec<(f64, f64)> = positions
            .iter()
            .zip(hfr_values.iter())
            .filter_map(|(x, y)| y.is_finite().then_some((*x as f64, *y)))
            .collect();
        if samples.len() < 3 {
            return None;
        }

        let origin = samples.iter().map(|(x, _)| *x).sum::<f64>() / samples.len() as f64;
        let mut sx = 0.0;
        let mut sx2 = 0.0;
        let mut sx3 = 0.0;
        let mut sx4 = 0.0;
        let mut sy = 0.0;
        let mut sxy = 0.0;
        let mut sx2y = 0.0;

        for (position, hfr) in &samples {
            let x = *position - origin;
            let x2 = x * x;
            sx += x;
            sx2 += x2;
            sx3 += x2 * x;
            sx4 += x2 * x2;
            sy += hfr;
            sxy += x * hfr;
            sx2y += x2 * hfr;
        }

        let n = samples.len() as f64;
        let [a, b, c] = solve_3x3(
            [[sx4, sx3, sx2], [sx3, sx2, sx], [sx2, sx, n]],
            [sx2y, sxy, sy],
        )?;

        if !a.is_finite() || a <= 0.0 {
            return None;
        }

        let optimal_position = origin - b / (2.0 * a);
        let min_position = samples
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::INFINITY, f64::min);
        let max_position = samples
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::NEG_INFINITY, f64::max);
        if optimal_position <= min_position
            || optimal_position >= max_position
            || !has_two_sided_support(&samples, optimal_position)
        {
            return None;
        }

        let predicted_hfr =
            a * (optimal_position - origin).powi(2) + b * (optimal_position - origin) + c;
        let mean_y = sy / n;
        let ss_tot = samples
            .iter()
            .map(|(_, y)| (y - mean_y).powi(2))
            .sum::<f64>();
        let ss_res = samples
            .iter()
            .map(|(position, y)| {
                let x = *position - origin;
                let predicted = a * x * x + b * x + c;
                (y - predicted).powi(2)
            })
            .sum::<f64>();
        let r_squared = if ss_tot <= f64::EPSILON {
            1.0
        } else {
            (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
        };

        Some(CurveFitResult {
            optimal_position,
            predicted_hfr,
            r_squared,
            coefficients: vec![a, b, c, origin],
            method: CurveFitMethod::Parabola,
        })
    }

    /// Fit a V-curve (two linear segments) to position-HFR samples.
    pub fn fit_v_curve(positions: &[i32], hfr_values: &[f64]) -> Option<CurveFitResult> {
        if positions.len() != hfr_values.len() || positions.len() < 4 {
            return None;
        }

        let mut samples: Vec<(f64, f64)> = positions
            .iter()
            .zip(hfr_values.iter())
            .filter_map(|(position, hfr)| hfr.is_finite().then_some((*position as f64, *hfr)))
            .collect();
        samples.sort_by(|a, b| a.0.total_cmp(&b.0));
        if samples.len() < 4
            || samples
                .windows(2)
                .any(|pair| (pair[0].0 - pair[1].0).abs() <= f64::EPSILON)
        {
            return None;
        }

        let mean_y = samples.iter().map(|(_, y)| *y).sum::<f64>() / samples.len() as f64;
        let ss_tot = samples
            .iter()
            .map(|(_, y)| (y - mean_y).powi(2))
            .sum::<f64>();
        if ss_tot <= f64::EPSILON {
            return None;
        }

        let mut best: Option<CurveFitResult> = None;
        // The sample at `split` belongs to both arms. This anchors both lines at
        // the measured valley while still requiring at least two points per arm.
        for split in 1..samples.len() - 1 {
            let Some((left_slope, left_intercept)) = fit_line(&samples[..=split]) else {
                continue;
            };
            let Some((right_slope, right_intercept)) = fit_line(&samples[split..]) else {
                continue;
            };
            if left_slope >= 0.0 || right_slope <= 0.0 {
                continue;
            }
            let denominator = left_slope - right_slope;
            if denominator.abs() <= f64::EPSILON {
                continue;
            }
            let optimum = (right_intercept - left_intercept) / denominator;
            let min_x = samples.first()?.0;
            let max_x = samples.last()?.0;
            if !optimum.is_finite() || optimum <= min_x || optimum >= max_x {
                continue;
            }
            if !has_two_sided_support(&samples, optimum) {
                continue;
            }
            let predicted_hfr = left_slope * optimum + left_intercept;
            if !predicted_hfr.is_finite() || predicted_hfr <= 0.0 {
                continue;
            }

            let ss_res = samples
                .iter()
                .map(|(x, y)| {
                    let predicted = if *x <= optimum {
                        left_slope * *x + left_intercept
                    } else {
                        right_slope * *x + right_intercept
                    };
                    (y - predicted).powi(2)
                })
                .sum::<f64>();
            let r_squared = (1.0 - ss_res / ss_tot).clamp(0.0, 1.0);
            let candidate = CurveFitResult {
                optimal_position: optimum,
                predicted_hfr,
                r_squared,
                coefficients: vec![left_slope, left_intercept, right_slope, right_intercept],
                method: CurveFitMethod::VCurve,
            };
            if best
                .as_ref()
                .map(|current| candidate.r_squared > current.r_squared)
                .unwrap_or(true)
            {
                best = Some(candidate);
            }
        }
        best
    }

    /// Fit a hyperbola to position-HFR samples.
    pub fn fit_hyperbola(positions: &[i32], hfr_values: &[f64]) -> Option<CurveFitResult> {
        let _ = (positions, hfr_values);
        // Do not relabel a different estimator as a hyperbola. A bounded nonlinear
        // fit can be added once real sweeps justify its extra degrees of freedom.
        None
    }
}

fn fit_line(samples: &[(f64, f64)]) -> Option<(f64, f64)> {
    if samples.len() < 2 {
        return None;
    }
    let n = samples.len() as f64;
    let mean_x = samples.iter().map(|(x, _)| *x).sum::<f64>() / n;
    let mean_y = samples.iter().map(|(_, y)| *y).sum::<f64>() / n;
    let denominator = samples
        .iter()
        .map(|(x, _)| (x - mean_x).powi(2))
        .sum::<f64>();
    if denominator <= f64::EPSILON {
        return None;
    }
    let slope = samples
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>()
        / denominator;
    Some((slope, mean_y - slope * mean_x))
}

/// A focus minimum is actionable only when the sweep genuinely brackets it.
/// Two distinct measurements on each side keep a noisy endpoint/shoulder from
/// looking like a high-R2 minimum with no observed rising arm.
fn has_two_sided_support(samples: &[(f64, f64)], optimum: f64) -> bool {
    samples.iter().filter(|(x, _)| *x < optimum).count() >= 2
        && samples.iter().filter(|(x, _)| *x > optimum).count() >= 2
}

fn percentile(values: &[f32], q: f64) -> f32 {
    let mut finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return 0.0;
    }
    finite.sort_by(|a, b| a.total_cmp(b));
    let idx = ((finite.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    finite[idx]
}

/// Robust sky sigma via the median absolute deviation (MAD x 1.4826). Used only
/// to set the centroid detection threshold, so star pixels barely bias it.
fn robust_sigma(values: &[f32], median: f64) -> f64 {
    let mut deviations: Vec<f64> = values
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| (*v as f64 - median).abs())
        .collect();
    if deviations.is_empty() {
        return 0.0;
    }
    deviations.sort_by(|a, b| a.total_cmp(b));
    deviations[deviations.len() / 2] * 1.4826
}

/// One curve-of-growth pass within `max_radius`. `radial` is `(radius, residual)`
/// sorted ascending by radius. Returns `(half_flux_radius, total_flux)` using an
/// UNCLIPPED cumulative sum so zero-mean sky noise cancels instead of forming a
/// positive pedestal. The half-flux crossing lies in the steeply-rising stellar
/// core, so it is robust to noise wiggles in the wings.
fn cog_half_radius(radial: &[(f64, f64)], max_radius: f64) -> Option<(f64, f64)> {
    let mut radii: Vec<f64> = Vec::new();
    let mut cumulative: Vec<f64> = Vec::new();
    let mut running = 0.0_f64;
    for &(r, v) in radial {
        if r > max_radius {
            break; // radial is sorted ascending, so nothing further qualifies
        }
        running += v;
        radii.push(r);
        cumulative.push(running);
    }
    if radii.len() < 4 {
        return None;
    }
    let total = *cumulative.last().unwrap();
    if total <= 0.0 {
        return None;
    }

    let half = 0.5 * total;
    let idx = cumulative.iter().position(|&c| c >= half)?;
    if idx == 0 {
        return Some((radii[0], total));
    }
    let (c0, c1) = (cumulative[idx - 1], cumulative[idx]);
    let (r0, r1) = (radii[idx - 1], radii[idx]);
    let hfr = if (c1 - c0).abs() < f64::EPSILON {
        r1
    } else {
        r0 + (half - c0) / (c1 - c0) * (r1 - r0)
    };
    Some((hfr, total))
}

fn solve_3x3(mut matrix: [[f64; 3]; 3], mut rhs: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let pivot =
            (col..3).max_by(|a, b| matrix[*a][col].abs().total_cmp(&matrix[*b][col].abs()))?;
        if matrix[pivot][col].abs() <= f64::EPSILON {
            return None;
        }
        matrix.swap(col, pivot);
        rhs.swap(col, pivot);

        let divisor = matrix[col][col];
        for c in col..3 {
            matrix[col][c] /= divisor;
        }
        rhs[col] /= divisor;

        for row in 0..3 {
            if row == col {
                continue;
            }
            let factor = matrix[row][col];
            for c in col..3 {
                matrix[row][c] -= factor * matrix[col][c];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    Some(rhs)
}

#[cfg(test)]
mod tests {
    use super::{ClassicalFocusSolver, CurveFitMethod};

    #[test]
    fn hfr_is_finite_for_centered_star_crop() {
        let width = 9;
        let height = 9;
        let mut image = vec![10.0_f32; width * height];
        let center = (height / 2) * width + (width / 2);
        image[center] = 110.0;
        image[center - 1] = 60.0;
        image[center + 1] = 60.0;
        image[center - width] = 60.0;
        image[center + width] = 60.0;

        let hfr = ClassicalFocusSolver::compute_hfr(&image, width, height);

        assert!(hfr.is_finite());
        assert!((0.0..=1.0).contains(&hfr));
    }

    // Deterministic pseudo-noise (no rand dependency in tests).
    fn lcg_noise(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map high bits to roughly [-1, 1).
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn gaussian_star(
        size: usize,
        hfr: f32,
        sky: f32,
        amp: f32,
        read_noise: f32,
        seed: u64,
    ) -> Vec<f32> {
        let sigma = hfr / (2.0 * std::f32::consts::LN_2).sqrt();
        let c = size as f32 / 2.0 - 0.5;
        let mut state = seed.max(1);
        let mut pixels = vec![0.0_f32; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - c;
                let dy = y as f32 - c;
                let star = amp * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                pixels[y * size + x] = sky + star + read_noise * lcg_noise(&mut state);
            }
        }
        pixels
    }

    #[test]
    fn hfr_is_stable_across_noise_and_crop_size() {
        // A true 3 px HFR star must measure ~3 px regardless of noise level and
        // crop size — the property the old clipped, whole-crop integrator broke
        // (it drifted to 16 px at high noise and 21 px on a 128 px crop).
        let truth = 3.0_f32;
        for (amp, noise, seed) in [(6000.0, 0.0, 1), (3000.0, 25.0, 2), (1500.0, 60.0, 3)] {
            let img = gaussian_star(64, truth, 100.0, amp, noise, seed);
            let hfr = ClassicalFocusSolver::compute_hfr(&img, 64, 64) as f32;
            assert!(
                (hfr - truth).abs() < 0.6,
                "noise={noise}: HFR {hfr} strayed from {truth}"
            );
        }
        for size in [32usize, 64, 96, 128] {
            let img = gaussian_star(size, truth, 100.0, 3000.0, 25.0, 7);
            let hfr = ClassicalFocusSolver::compute_hfr(&img, size, size) as f32;
            assert!(
                (hfr - truth).abs() < 0.6,
                "crop={size}: HFR {hfr} strayed from {truth}"
            );
        }
    }

    #[test]
    fn parabola_fit_finds_known_minimum() {
        let positions = [900, 950, 1000, 1050, 1100];
        let hfr_values: Vec<f64> = positions
            .iter()
            .map(|position| 1.0 + ((*position - 1000) as f64 / 100.0).powi(2))
            .collect();

        let fit = ClassicalFocusSolver::fit_parabola(&positions, &hfr_values).unwrap();

        assert!(matches!(fit.method, CurveFitMethod::Parabola));
        assert!((fit.optimal_position - 1000.0).abs() < 0.001);
        assert!((fit.predicted_hfr - 1.0).abs() < 0.001);
        assert!(fit.r_squared > 0.999);
    }

    #[test]
    fn parabola_rejects_minimum_without_two_sided_support() {
        // This is an exact upward parabola, but its vertex lies between the first
        // two samples. The fit alone is excellent; the sweep did not bracket the
        // minimum well enough to command a move.
        let positions = [900, 950, 1000, 1050, 1100];
        let hfr_values: Vec<f64> = positions
            .iter()
            .map(|position| 1.0 + ((*position - 925) as f64 / 100.0).powi(2))
            .collect();

        assert!(ClassicalFocusSolver::fit_parabola(&positions, &hfr_values).is_none());
    }

    #[test]
    fn v_curve_fits_two_arms_and_intersection() {
        let fit = ClassicalFocusSolver::fit_v_curve(
            &[800, 900, 1000, 1100, 1200],
            &[5.0, 3.0, 1.0, 3.0, 5.0],
        )
        .unwrap();

        assert!(matches!(fit.method, CurveFitMethod::VCurve));
        assert!((fit.optimal_position - 1000.0).abs() < 1e-6);
        assert!((fit.predicted_hfr - 1.0).abs() < 1e-6);
        assert!(fit.r_squared > 0.999);
    }

    #[test]
    fn v_curve_rejects_unbracketed_monotonic_sweep() {
        assert!(ClassicalFocusSolver::fit_v_curve(
            &[800, 900, 1000, 1100, 1200],
            &[5.0, 4.0, 3.0, 2.0, 1.0],
        )
        .is_none());
    }
}
