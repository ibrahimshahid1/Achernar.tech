//! WebSocket handler — Section 7.5.5.
//! WS telemetry + all FOCUS_* and VisionUpdate events.

use serde::{Deserialize, Serialize};

/// All WebSocket event types emitted by the daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WsEvent {
    // Telemetry
    TelemetryUpdate {
        payload: serde_json::Value,
    },

    // Focus events — Section 7.5.5
    FocusRunStarted {
        run_id: String,
        method: String,
    },
    FocusRunStateChanged {
        run_id: String,
        old_state: String,
        new_state: String,
    },
    FocusSampleCaptured {
        run_id: String,
        position: i32,
        hfr: f64,
        star_count: u32,
    },
    FocusPredictionMade {
        run_id: String,
        model: String,
        predicted_offset: f64,
        confidence: f64,
    },
    FocusActionProposed {
        run_id: String,
        proposed_step: i32,
        source: String,
        confidence: f64,
    },
    FocusActionValidated {
        run_id: String,
        verdict: String,
        step: i32,
    },
    FocusActionExecuted {
        run_id: String,
        step: i32,
        new_position: i32,
    },
    FocusActionRejected {
        run_id: String,
        reason: String,
    },
    FocusFallbackTriggered {
        run_id: String,
        reason: String,
    },
    FocusRunConverged {
        run_id: String,
        best_position: i32,
        best_hfr: f64,
    },
    FocusRunFailed {
        run_id: String,
        error: String,
    },
    /// Never emitted yet — no publisher constructs this. Kept in the schema,
    /// but shaped to match `DriftPrediction`: a drift prediction always has a
    /// magnitude, and has a confidence and a recommendation only when a refocus
    /// threshold is known. A `bool` here would force a publisher to invent
    /// `false` for a rig whose optics were never declared, which reads as "we
    /// checked and it is fine".
    FocusDriftAlert {
        predicted_drift: f64,
        confidence: Option<f64>,
        refocus_recommended: Option<bool>,
    },

    // Vision events — Section 8.1
    VisionUpdate {
        observation: serde_json::Value,
    },
}

impl WsEvent {
    pub fn event_name(&self) -> &'static str {
        match self {
            Self::TelemetryUpdate { .. } => "TELEMETRY_UPDATE",
            Self::FocusRunStarted { .. } => "FOCUS_RUN_STARTED",
            Self::FocusRunStateChanged { .. } => "FOCUS_RUN_STATE_CHANGED",
            Self::FocusSampleCaptured { .. } => "FOCUS_SAMPLE_CAPTURED",
            Self::FocusPredictionMade { .. } => "FOCUS_PREDICTION_MADE",
            Self::FocusActionProposed { .. } => "FOCUS_ACTION_PROPOSED",
            Self::FocusActionValidated { .. } => "FOCUS_ACTION_VALIDATED",
            Self::FocusActionExecuted { .. } => "FOCUS_ACTION_EXECUTED",
            Self::FocusActionRejected { .. } => "FOCUS_ACTION_REJECTED",
            Self::FocusFallbackTriggered { .. } => "FOCUS_FALLBACK_TRIGGERED",
            Self::FocusRunConverged { .. } => "FOCUS_RUN_CONVERGED",
            Self::FocusRunFailed { .. } => "FOCUS_RUN_FAILED",
            Self::FocusDriftAlert { .. } => "FOCUS_DRIFT_ALERT",
            Self::VisionUpdate { .. } => "VISION_UPDATE",
        }
    }
}
