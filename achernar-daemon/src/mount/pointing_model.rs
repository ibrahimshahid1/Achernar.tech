//! Pointing model — Section 8.3 (M-4).
//! Thin-plate spline correction for mount pointing errors.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointingCorrection {
    pub ra_correction_arcsec: f64,
    pub dec_correction_arcsec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointingSample {
    pub commanded_ra: f64,
    pub commanded_dec: f64,
    pub actual_ra: f64,
    pub actual_dec: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct PointingModel {
    samples: Vec<PointingSample>,
}

impl PointingModel {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    pub fn add_sample(&mut self, sample: PointingSample) {
        self.samples.push(sample);
    }

    /// Compute pointing correction using thin-plate spline interpolation.
    pub fn correct(&self, ra: f64, dec: f64) -> PointingCorrection {
        const MAX_CORRECTION_DEG: f64 = 0.5;

        let (ra_error_deg, dec_error_deg) = if self.samples.len() >= 5 {
            (
                tps_predict(&self.samples, ra, dec, Axis::Ra)
                    .unwrap_or_else(|| idw_predict(&self.samples, ra, dec, Axis::Ra)),
                tps_predict(&self.samples, ra, dec, Axis::Dec)
                    .unwrap_or_else(|| idw_predict(&self.samples, ra, dec, Axis::Dec)),
            )
        } else {
            (
                idw_predict(&self.samples, ra, dec, Axis::Ra),
                idw_predict(&self.samples, ra, dec, Axis::Dec),
            )
        };

        PointingCorrection {
            ra_correction_arcsec: (ra_error_deg.clamp(-MAX_CORRECTION_DEG, MAX_CORRECTION_DEG))
                * 3600.0,
            dec_correction_arcsec: (dec_error_deg.clamp(-MAX_CORRECTION_DEG, MAX_CORRECTION_DEG))
                * 3600.0,
        }
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

#[derive(Debug, Clone, Copy)]
enum Axis {
    Ra,
    Dec,
}

fn tps_predict(samples: &[PointingSample], ra: f64, dec: f64, axis: Axis) -> Option<f64> {
    let n = samples.len();
    let size = n + 3;
    let mut matrix = vec![vec![0.0_f64; size]; size];
    let mut rhs = vec![0.0_f64; size];

    for (row, sample_i) in samples.iter().enumerate() {
        for (col, sample_j) in samples.iter().enumerate() {
            let r = angular_distance_deg(
                sample_i.commanded_ra,
                sample_i.commanded_dec,
                sample_j.commanded_ra,
                sample_j.commanded_dec,
            );
            matrix[row][col] = tps_phi(r);
        }
        matrix[row][row] += 1.0e-4;
        matrix[row][n] = 1.0;
        matrix[row][n + 1] = sample_i.commanded_ra;
        matrix[row][n + 2] = sample_i.commanded_dec;

        matrix[n][row] = 1.0;
        matrix[n + 1][row] = sample_i.commanded_ra;
        matrix[n + 2][row] = sample_i.commanded_dec;
        rhs[row] = error_for_axis(sample_i, axis);
    }

    let solution = solve_linear(&matrix, &rhs)?;
    let weights = &solution[..n];
    let affine = &solution[n..];
    let mut predicted = affine[0] + affine[1] * ra + affine[2] * dec;

    for (weight, sample) in weights.iter().zip(samples.iter()) {
        let r = angular_distance_deg(ra, dec, sample.commanded_ra, sample.commanded_dec);
        predicted += weight * tps_phi(r);
    }

    predicted.is_finite().then_some(predicted)
}

fn idw_predict(samples: &[PointingSample], ra: f64, dec: f64, axis: Axis) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }

    let mut weighted = 0.0_f64;
    let mut weight_sum = 0.0_f64;
    for sample in samples {
        let distance = angular_distance_deg(ra, dec, sample.commanded_ra, sample.commanded_dec);
        let weight = 1.0 / (distance * distance + 1.0e-6);
        weighted += weight * error_for_axis(sample, axis);
        weight_sum += weight;
    }

    if weight_sum <= f64::EPSILON {
        0.0
    } else {
        weighted / weight_sum
    }
}

fn error_for_axis(sample: &PointingSample, axis: Axis) -> f64 {
    match axis {
        Axis::Ra => sample.actual_ra - sample.commanded_ra,
        Axis::Dec => sample.actual_dec - sample.commanded_dec,
    }
}

fn angular_distance_deg(ra_a: f64, dec_a: f64, ra_b: f64, dec_b: f64) -> f64 {
    let dra = ((ra_a - ra_b + 180.0).rem_euclid(360.0) - 180.0) * dec_a.to_radians().cos();
    let ddec = dec_a - dec_b;
    (dra * dra + ddec * ddec).sqrt()
}

fn tps_phi(r: f64) -> f64 {
    if r <= f64::EPSILON {
        0.0
    } else {
        let r2 = r * r;
        r2 * r.ln()
    }
}

fn solve_linear(matrix: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }

    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();

    for col in 0..n {
        let pivot = (col..n)
            .max_by(|a_idx, b_idx| a[*a_idx][col].abs().total_cmp(&a[*b_idx][col].abs()))?;
        if a[pivot][col].abs() <= f64::EPSILON {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);

        let divisor = a[col][col];
        for c in col..n {
            a[col][c] /= divisor;
        }
        b[col] /= divisor;

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for c in col..n {
                a[row][c] -= factor * a[col][c];
            }
            b[row] -= factor * b[col];
        }
    }

    Some(b)
}
