//! Control supervisor — Section 7.3.1.
//! Manages AutonomyMode enum and mode downgrade logic.

use serde::{Deserialize, Serialize};

/// Autonomy modes for the control system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutonomyMode {
    /// Full AI-driven control — all decisions automated
    FullAuto,
    /// AI suggests, human approves
    Supervised,
    /// Human drives, AI monitors and alerts only
    Manual,
    /// Emergency stop — all hardware frozen
    Emergency,
}

/// Reason for a mode change or downgrade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModeChangeReason {
    UserRequested,
    SafetyViolation(String),
    HardwareFailure(String),
    CommunicationLoss,
    WeatherAlert,
    ConfidenceBelowThreshold { confidence: f64, threshold: f64 },
}

/// The control supervisor manages the overall autonomy state.
pub struct ControlSupervisor {
    current_mode: AutonomyMode,
    mode_history: Vec<(
        AutonomyMode,
        ModeChangeReason,
        chrono::DateTime<chrono::Utc>,
    )>,
}

impl ControlSupervisor {
    pub fn new() -> Self {
        Self {
            current_mode: AutonomyMode::Manual,
            mode_history: Vec::new(),
        }
    }

    pub fn current_mode(&self) -> AutonomyMode {
        self.current_mode
    }

    /// Attempt to change the autonomy mode.
    pub fn set_mode(&mut self, mode: AutonomyMode, reason: ModeChangeReason) {
        let prev = self.current_mode;
        self.current_mode = mode;
        self.mode_history.push((prev, reason, chrono::Utc::now()));
        tracing::info!("Control mode changed: {:?} → {:?}", prev, mode);
    }

    /// Downgrade autonomy level due to an issue.
    /// FullAuto → Supervised → Manual → Emergency
    pub fn downgrade(&mut self, reason: ModeChangeReason) {
        let new_mode = match self.current_mode {
            AutonomyMode::FullAuto => AutonomyMode::Supervised,
            AutonomyMode::Supervised => AutonomyMode::Manual,
            AutonomyMode::Manual | AutonomyMode::Emergency => AutonomyMode::Emergency,
        };
        self.set_mode(new_mode, reason);
    }

    /// Emergency stop — immediately freeze all hardware.
    pub fn emergency_stop(&mut self) {
        self.set_mode(AutonomyMode::Emergency, ModeChangeReason::UserRequested);
    }
}
