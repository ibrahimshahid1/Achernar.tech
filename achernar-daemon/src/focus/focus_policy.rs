//! Focus policy — Section 7.3.5.
//! Deterministic policy layer that combines classical and ML outputs.

use serde::{Deserialize, Serialize};

/// Decision from the focus policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusDecision {
    pub proposed_step: i32,
    pub confidence: f64,
    pub source: DecisionSource,
    pub reasoning: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DecisionSource {
    Classical,
    MlModel,
    Hybrid,
    Fallback,
}

/// The deterministic policy layer that decides the next focus action.
pub struct FocusPolicy {
    use_ml: bool,
    fallback_step_size: i32,
}

impl FocusPolicy {
    pub fn new(use_ml: bool, fallback_step_size: i32) -> Self {
        Self {
            use_ml,
            fallback_step_size,
        }
    }

    /// Determine the next focus action given classical and (optionally) ML outputs.
    pub fn decide(
        &self,
        _classical_fit: Option<&super::classical_focus_solver::CurveFitResult>,
        _ml_prediction: Option<&super::focus_inference::FocusOffsetOutput>,
        _current_position: i32,
        _samples: &[super::focus_agent::FocusSample],
    ) -> FocusDecision {
        todo!("Implement hybrid decision logic — spec Section 7.3.5")
    }

    /// Fallback decision when both classical and ML fail.
    pub fn fallback_decision(&self, direction: i32) -> FocusDecision {
        FocusDecision {
            proposed_step: direction * self.fallback_step_size,
            confidence: 0.3,
            source: DecisionSource::Fallback,
            reasoning: "Using fallback step — insufficient data".to_string(),
        }
    }
}
