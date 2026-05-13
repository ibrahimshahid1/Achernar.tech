//! Filter bandit — Section 8.2 (FW-2).
//! LinUCB contextual bandit for filter arm selection.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterBanditArm {
    pub filter_name: String,
    pub estimated_reward: f64,
    pub uncertainty: f64,
    pub ucb_score: f64,
    pub times_pulled: u32,
}

pub struct FilterBandit {
    alpha: f64,
    arms: Vec<FilterBanditArm>,
}

impl FilterBandit {
    pub fn new(filter_names: &[String], alpha: f64) -> Self {
        let arms = filter_names
            .iter()
            .map(|name| FilterBanditArm {
                filter_name: name.clone(),
                estimated_reward: 0.0,
                uncertainty: 1.0,
                ucb_score: f64::MAX,
                times_pulled: 0,
            })
            .collect();
        Self { alpha, arms }
    }

    /// Select the best filter arm using LinUCB.
    pub fn select_arm(&self, _context: &[f64]) -> &FilterBanditArm {
        todo!("Implement LinUCB arm selection — Section 8.2 FW-2")
    }

    /// Update arm with observed reward.
    pub fn update(&mut self, _filter_name: &str, _reward: f64, _context: &[f64]) {
        todo!("Implement LinUCB update")
    }
}
