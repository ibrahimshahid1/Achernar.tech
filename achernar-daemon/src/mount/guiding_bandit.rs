//! Guiding bandit — Section 8.3 (M-1).
//! LinUCB aggressiveness tuner for RA and Dec guiding.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidingBanditState {
    pub ra_aggressiveness: f64,
    pub dec_aggressiveness: f64,
    pub ra_rms: f64,
    pub dec_rms: f64,
}

pub struct GuidingBandit {
    ra_alpha: f64,
    dec_alpha: f64,
}

impl GuidingBandit {
    pub fn new(alpha: f64) -> Self {
        Self { ra_alpha: alpha, dec_alpha: alpha }
    }

    pub fn select_aggressiveness(&self, _context: &[f64]) -> GuidingBanditState {
        todo!("Implement LinUCB aggressiveness selection — Section 8.3 M-1")
    }

    pub fn update(&mut self, _reward: f64, _axis: &str, _context: &[f64]) {
        todo!("Implement LinUCB update")
    }
}
