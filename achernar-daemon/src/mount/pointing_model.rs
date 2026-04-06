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
        Self { samples: Vec::new() }
    }

    pub fn add_sample(&mut self, sample: PointingSample) {
        self.samples.push(sample);
    }

    /// Compute pointing correction using thin-plate spline interpolation.
    pub fn correct(&self, _ra: f64, _dec: f64) -> PointingCorrection {
        todo!("Implement thin-plate spline correction — Section 8.3 M-4")
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}
