//! Focus agent — Section 7.3.4.
//! FocusRunState machine and run_autofocus() orchestration.

use serde::{Deserialize, Serialize};

/// State machine for an autofocus run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FocusRunState {
    Idle,
    Initializing,
    Sweeping,
    Measuring,
    Evaluating,
    Moving,
    Settling,
    Converged,
    Failed,
    Aborted,
}

/// Metadata for a focus run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusRunContext {
    pub run_id: uuid::Uuid,
    pub rig_id: String,
    pub filter_name: Option<String>,
    pub temperature: Option<f64>,
    pub start_position: i32,
    pub state: FocusRunState,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub samples: Vec<FocusSample>,
    pub best_position: Option<i32>,
    pub best_hfr: Option<f64>,
}

/// A single focus measurement sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusSample {
    pub position: i32,
    pub hfr: f64,
    pub fwhm: Option<f64>,
    pub star_count: u32,
    pub quality_score: Option<f64>,
    pub temperature: Option<f64>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// The focus agent orchestrates an autofocus run.
pub struct FocusAgent {
    state: FocusRunState,
    current_run: Option<FocusRunContext>,
}

impl FocusAgent {
    pub fn new() -> Self {
        Self {
            state: FocusRunState::Idle,
            current_run: None,
        }
    }

    pub fn state(&self) -> FocusRunState {
        self.state
    }

    pub fn current_run(&self) -> Option<&FocusRunContext> {
        self.current_run.as_ref()
    }

    /// Start a new autofocus run.
    pub async fn run_autofocus(
        &mut self,
        _config: &crate::config::FocusConfig,
    ) -> crate::error::Result<FocusRunContext> {
        todo!("Implement autofocus run orchestration — spec Section 7.3.4")
    }

    /// Abort the current focus run.
    pub fn abort(&mut self) {
        self.state = FocusRunState::Aborted;
        tracing::warn!("Focus run aborted");
    }

    fn transition(&mut self, new_state: FocusRunState) {
        tracing::info!("Focus state: {:?} → {:?}", self.state, new_state);
        self.state = new_state;
    }
}
