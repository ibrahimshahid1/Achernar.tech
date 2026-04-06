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

    pub fn validate_switch(&self, _is_exposing: bool) -> Result<(), String> {
        todo!("Implement filter switch safety validation")
    }
}
