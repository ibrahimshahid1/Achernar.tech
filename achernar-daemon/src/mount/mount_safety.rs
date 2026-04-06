//! Mount safety — Section 8.3.
//! MountSafetyConfig, horizon model, hour-angle limits.

use serde::{Deserialize, Serialize};

/// Horizon model for altitude checking.
pub struct HorizonModel {
    /// Azimuth-altitude pairs defining the horizon profile
    profile: Vec<(f64, f64)>,
}

impl HorizonModel {
    pub fn flat(min_altitude: f64) -> Self {
        Self { profile: vec![(0.0, min_altitude), (360.0, min_altitude)] }
    }

    pub fn from_file(_path: &str) -> crate::error::Result<Self> {
        todo!("Load horizon model from file")
    }

    /// Get minimum allowed altitude at a given azimuth.
    pub fn min_altitude_at(&self, _azimuth: f64) -> f64 {
        todo!("Interpolate horizon profile")
    }
}

/// Full mount safety validator.
pub struct MountSafetyValidator {
    config: crate::config::MountSafetyConfig,
    horizon: HorizonModel,
}

impl MountSafetyValidator {
    pub fn new(config: crate::config::MountSafetyConfig) -> Self {
        let horizon = HorizonModel::flat(config.min_altitude);
        Self { config, horizon }
    }

    pub fn is_target_safe(&self, _alt: f64, _az: f64, _ha: f64) -> bool {
        todo!("Validate target against horizon model and HA limits")
    }
}
