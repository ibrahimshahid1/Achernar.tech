//! Filter wheel safety — Section 8.2.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterWheelSafetyConfig {
    pub max_switch_timeout_ms: u64,
    pub verify_position_after_switch: bool,
    pub block_during_exposure: bool,
}

impl Default for FilterWheelSafetyConfig {
    fn default() -> Self {
        Self {
            max_switch_timeout_ms: 10000,
            verify_position_after_switch: true,
            block_during_exposure: true,
        }
    }
}

pub struct FilterWheelSafetyGuard {
    config: FilterWheelSafetyConfig,
}

impl FilterWheelSafetyGuard {
    pub fn new(config: FilterWheelSafetyConfig) -> Self {
        Self { config }
    }

    pub fn validate_switch(&self, is_exposing: bool) -> Result<(), String> {
        if self.config.max_switch_timeout_ms == 0 {
            return Err("Filter switch timeout must be greater than zero".to_string());
        }
        if self.config.block_during_exposure && is_exposing {
            return Err("Filter switch blocked during active exposure".to_string());
        }
        Ok(())
    }
}
