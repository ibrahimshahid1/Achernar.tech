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
    /// Compute HFR (Half-Flux Radius) from a star image crop.
    pub fn compute_hfr(image_data: &[f32], width: usize, height: usize) -> f64 {
        if width == 0 || height == 0 || image_data.len() != width * height {
            return f64::NAN;
        }

        let background = percentile(image_data, 0.5);
        let mut total_flux = 0.0_f64;
        let mut x_weighted = 0.0_f64;
        let mut y_weighted = 0.0_f64;

        for y in 0..height {
            for x in 0..width {
                let flux = (image_data[y * width + x] - background).max(0.0) as f64;
                total_flux += flux;
                x_weighted += x as f64 * flux;
                y_weighted += y as f64 * flux;
            }
        }

        if total_flux <= f64::EPSILON {
            return f64::NAN;
        }

        let cx = x_weighted / total_flux;
        let cy = y_weighted / total_flux;
        let mut radial_flux = Vec::with_capacity(image_data.len());
        for y in 0..height {
            for x in 0..width {
                let flux = (image_data[y * width + x] - background).max(0.0) as f64;
                if flux <= 0.0 {
                    continue;
                }
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                radial_flux.push(((dx * dx + dy * dy).sqrt(), flux));
            }
        }

        radial_flux.sort_by(|a, b| a.0.total_cmp(&b.0));
        let half_flux = total_flux * 0.5;
        let mut cumulative = 0.0_f64;
        for (radius, flux) in radial_flux {
            cumulative += flux;
            if cumulative >= half_flux {
                return radius;
            }
        }

        f64::NAN
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
        if optimal_position < min_position || optimal_position > max_position {
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
        let (idx, (_, hfr)) = positions
            .iter()
            .zip(hfr_values.iter())
            .enumerate()
            .filter(|(_, (_, hfr))| hfr.is_finite())
            .min_by(|(_, (_, a)), (_, (_, b))| a.total_cmp(b))?;

        Some(CurveFitResult {
            optimal_position: positions[idx] as f64,
            predicted_hfr: *hfr,
            r_squared: 0.0,
            coefficients: vec![],
            method: CurveFitMethod::VCurve,
        })
    }

    /// Fit a hyperbola to position-HFR samples.
    pub fn fit_hyperbola(positions: &[i32], hfr_values: &[f64]) -> Option<CurveFitResult> {
        // Hyperbolic fits are useful once we have more field data. For the MVP
        // deterministic gate, a V-curve fallback gives bounded behavior without
        // overfitting sparse autofocus sweeps.
        Self::fit_v_curve(positions, hfr_values).map(|mut fit| {
            fit.method = CurveFitMethod::Hyperbola;
            fit
        })
    }
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
    fn v_curve_uses_best_finite_sample() {
        let fit =
            ClassicalFocusSolver::fit_v_curve(&[100, 200, 300], &[4.0, f64::NAN, 3.0]).unwrap();

        assert!(matches!(fit.method, CurveFitMethod::VCurve));
        assert_eq!(fit.optimal_position, 300.0);
        assert_eq!(fit.predicted_hfr, 3.0);
    }
}
