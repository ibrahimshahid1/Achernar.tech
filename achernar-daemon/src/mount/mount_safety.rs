//! Mount safety — Section 8.3.
//! MountSafetyConfig, horizon model, hour-angle limits.

use std::fs;

/// Horizon model for altitude checking.
pub struct HorizonModel {
    /// Azimuth-altitude pairs defining the horizon profile
    profile: Vec<(f64, f64)>,
}

impl HorizonModel {
    pub fn flat(min_altitude: f64) -> Self {
        Self {
            profile: vec![(0.0, min_altitude), (360.0, min_altitude)],
        }
    }

    pub fn from_file(path: &str) -> crate::error::Result<Self> {
        let contents = fs::read_to_string(path)?;
        let mut profile = Vec::new();

        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line
                .split(|ch: char| ch == ',' || ch.is_whitespace())
                .filter(|part| !part.is_empty())
                .collect();
            if parts.len() < 2 {
                continue;
            }
            let azimuth = parts[0].parse::<f64>().map_err(|err| {
                crate::error::ConfigError::ParseError(format!(
                    "Invalid horizon azimuth '{line}': {err}"
                ))
            })?;
            let altitude = parts[1].parse::<f64>().map_err(|err| {
                crate::error::ConfigError::ParseError(format!(
                    "Invalid horizon altitude '{line}': {err}"
                ))
            })?;
            profile.push((azimuth.rem_euclid(360.0), altitude));
        }

        if profile.is_empty() {
            return Err(crate::error::ConfigError::ParseError(
                "Horizon model file did not contain azimuth/altitude pairs".to_string(),
            )
            .into());
        }

        profile.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ok(Self { profile })
    }

    /// Get minimum allowed altitude at a given azimuth.
    pub fn min_altitude_at(&self, azimuth: f64) -> f64 {
        if self.profile.is_empty() {
            return 0.0;
        }
        if self.profile.len() == 1 {
            return self.profile[0].1;
        }

        let azimuth = azimuth.rem_euclid(360.0);
        for window in self.profile.windows(2) {
            let (az_a, alt_a) = window[0];
            let (az_b, alt_b) = window[1];
            if azimuth >= az_a && azimuth <= az_b {
                let t = if (az_b - az_a).abs() <= f64::EPSILON {
                    0.0
                } else {
                    (azimuth - az_a) / (az_b - az_a)
                };
                return alt_a + (alt_b - alt_a) * t;
            }
        }

        let (last_az, last_alt) = *self.profile.last().unwrap();
        let (first_az, first_alt) = self.profile[0];
        let wrapped_first_az = first_az + 360.0;
        let adjusted_az = if azimuth < first_az {
            azimuth + 360.0
        } else {
            azimuth
        };
        let t = if (wrapped_first_az - last_az).abs() <= f64::EPSILON {
            0.0
        } else {
            (adjusted_az - last_az) / (wrapped_first_az - last_az)
        };
        last_alt + (first_alt - last_alt) * t
    }
}

/// Full mount safety validator.
pub struct MountSafetyValidator {
    config: crate::config::MountSafetyConfig,
    horizon: HorizonModel,
}

impl MountSafetyValidator {
    pub fn new(config: crate::config::MountSafetyConfig) -> Self {
        let horizon = config
            .horizon_model_path
            .as_deref()
            .and_then(|path| HorizonModel::from_file(path).ok())
            .unwrap_or_else(|| HorizonModel::flat(config.min_altitude));
        Self { config, horizon }
    }

    pub fn is_target_safe(&self, alt: f64, az: f64, ha: f64) -> bool {
        if !alt.is_finite() || !az.is_finite() || !ha.is_finite() {
            return false;
        }
        let horizon_alt = self
            .horizon
            .min_altitude_at(az)
            .max(self.config.min_altitude);
        alt >= horizon_alt && ha.abs() <= self.config.max_hour_angle
    }
}
