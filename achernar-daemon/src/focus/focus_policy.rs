//! Focus policy — Section 7.3.5.
//! Deterministic policy layer that combines classical and ML outputs.

use serde::{Deserialize, Serialize};

/// What a Focus Offset Predictor run returns — Section 7.6.2.
///
/// Declared here, where it is consumed, rather than in a runner module that
/// does not exist yet. It previously lived in `focus_inference`, whose only
/// other contents were a `todo!()` reimplementation of the FQE inference that
/// `vision/fqe_runner.rs` already does; that module is gone and this type
/// outlived it because the policy layer genuinely needs a shape to accept.
///
/// The three fields are the heads of the distilled MLP in Section 7.6.2:
/// signed delta, log-uncertainty, and the direction logits reduced to a choice.
/// When the FOP runner lands it should own this the way `FqeRunner` owns
/// `FqeOutput`, and this declaration should move there.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusOffsetPrediction {
    /// Predicted signed steps from the current position to best focus.
    pub delta_steps: f64,
    /// Predictive standard deviation of `delta_steps`, in steps.
    pub uncertainty_steps: f64,
    /// -1 inward / 0 stop / +1 outward, from the direction head.
    pub direction: i8,
}

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
        _ml_prediction: Option<&FocusOffsetPrediction>,
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
