//! Focus agent — Section 7.3.4.
//! FocusRunState machine and run_autofocus() orchestration.

use serde::{Deserialize, Serialize};
use std::future::Future;

use super::classical_focus_solver::ClassicalFocusSolver;

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
    history: Vec<FocusRunContext>,
}

impl FocusAgent {
    pub fn new() -> Self {
        Self {
            state: FocusRunState::Idle,
            current_run: None,
            history: Vec::new(),
        }
    }

    pub fn state(&self) -> FocusRunState {
        self.state
    }

    pub fn current_run(&self) -> Option<&FocusRunContext> {
        self.current_run.as_ref()
    }

    pub fn history(&self) -> &[FocusRunContext] {
        &self.history
    }

    /// Start a new autofocus run.
    pub async fn run_autofocus<F, Fut>(
        &mut self,
        config: &crate::config::FocusConfig,
        filter_name: Option<String>,
        start_position: i32,
        mut measure_at: F,
    ) -> crate::error::Result<FocusRunContext>
    where
        F: FnMut(i32) -> Fut,
        Fut: Future<Output = crate::error::Result<FocusSample>>,
    {
        if matches!(
            self.state,
            FocusRunState::Initializing
                | FocusRunState::Sweeping
                | FocusRunState::Measuring
                | FocusRunState::Moving
                | FocusRunState::Settling
                | FocusRunState::Evaluating
        ) {
            return Err(crate::error::FocusError::RunFailed(
                "Autofocus is already running".to_string(),
            )
            .into());
        }

        let mut run = FocusRunContext {
            run_id: uuid::Uuid::new_v4(),
            rig_id: "local-rig".to_string(),
            filter_name,
            temperature: None,
            start_position,
            state: FocusRunState::Initializing,
            started_at: chrono::Utc::now(),
            samples: Vec::new(),
            best_position: None,
            best_hfr: None,
        };
        self.transition(FocusRunState::Initializing);
        self.current_run = Some(run.clone());

        self.transition(FocusRunState::Sweeping);
        run.state = FocusRunState::Sweeping;
        self.current_run = Some(run.clone());

        for position in sweep_positions(config, start_position) {
            self.transition(FocusRunState::Moving);
            run.state = FocusRunState::Moving;
            self.current_run = Some(run.clone());

            let sample = match measure_at(position).await {
                Ok(sample) => sample,
                Err(err) => {
                    self.transition(FocusRunState::Failed);
                    run.state = FocusRunState::Failed;
                    self.current_run = Some(run.clone());
                    self.history.push(run);
                    return Err(err);
                }
            };

            self.transition(FocusRunState::Measuring);
            run.state = FocusRunState::Measuring;
            run.temperature = sample.temperature;
            run.samples.push(sample);
            self.current_run = Some(run.clone());
        }

        self.transition(FocusRunState::Evaluating);
        run.state = FocusRunState::Evaluating;
        self.current_run = Some(run.clone());

        let positions: Vec<i32> = run.samples.iter().map(|sample| sample.position).collect();
        let hfr_values: Vec<f64> = run.samples.iter().map(|sample| sample.hfr).collect();
        let fit = ClassicalFocusSolver::fit_parabola(&positions, &hfr_values)
            .or_else(|| ClassicalFocusSolver::fit_v_curve(&positions, &hfr_values));

        if let Some(fit) = fit {
            let best_position = fit.optimal_position.round() as i32;
            run.best_position =
                Some(best_position.clamp(config.safety.min_position, config.safety.max_position));
            run.best_hfr = Some(fit.predicted_hfr);
        } else if let Some(best_sample) = run
            .samples
            .iter()
            .filter(|sample| sample.hfr.is_finite())
            .min_by(|a, b| a.hfr.total_cmp(&b.hfr))
        {
            run.best_position = Some(best_sample.position);
            run.best_hfr = Some(best_sample.hfr);
        } else {
            self.transition(FocusRunState::Failed);
            run.state = FocusRunState::Failed;
            self.current_run = Some(run.clone());
            self.history.push(run);
            return Err(crate::error::FocusError::RunFailed(
                "No valid focus samples were collected".to_string(),
            )
            .into());
        }

        self.transition(FocusRunState::Converged);
        run.state = FocusRunState::Converged;
        self.current_run = Some(run.clone());
        self.history.push(run.clone());
        Ok(run)
    }

    /// Abort the current focus run.
    pub fn abort(&mut self) {
        self.state = FocusRunState::Aborted;
        if let Some(run) = self.current_run.as_mut() {
            run.state = FocusRunState::Aborted;
        }
        tracing::warn!("Focus run aborted");
    }

    fn transition(&mut self, new_state: FocusRunState) {
        tracing::info!("Focus state: {:?} → {:?}", self.state, new_state);
        self.state = new_state;
    }
}

fn sweep_positions(config: &crate::config::FocusConfig, start_position: i32) -> Vec<i32> {
    let step = config
        .initial_step_size
        .abs()
        .max(1)
        .min(config.safety.max_step_single.max(1));
    let max_radius = (config.safety.max_total_travel_per_run / 2)
        .max(step)
        .min(step * 3);
    let mut positions = (-3..=3)
        .map(|offset| start_position + offset * step)
        .filter(|position| (position - start_position).abs() <= max_radius)
        .map(|position| position.clamp(config.safety.min_position, config.safety.max_position))
        .collect::<Vec<_>>();
    positions.sort_unstable();
    positions.dedup();
    positions
}

#[cfg(test)]
mod tests {
    use super::{sweep_positions, FocusAgent, FocusRunState, FocusSample};
    use crate::config::FocusConfig;

    #[tokio::test]
    async fn autofocus_converges_on_synthetic_u_curve() {
        let mut config = FocusConfig::default();
        config.initial_step_size = 50;
        config.safety.min_position = 0;
        config.safety.max_position = 20_000;
        config.safety.max_step_single = 500;
        config.safety.max_total_travel_per_run = 1_000;

        let mut agent = FocusAgent::new();
        let run = agent
            .run_autofocus(
                &config,
                Some("L".to_string()),
                10_000,
                |position| async move {
                    let hfr = 1.2 + ((position - 10_000) as f64 / 100.0).powi(2);
                    Ok(FocusSample {
                        position,
                        hfr,
                        fwhm: Some(hfr * 2.0),
                        star_count: 1,
                        quality_score: Some(1.0 / hfr),
                        temperature: Some(5.0),
                        timestamp: chrono::Utc::now(),
                    })
                },
            )
            .await
            .unwrap();

        assert_eq!(run.state, FocusRunState::Converged);
        assert_eq!(run.filter_name.as_deref(), Some("L"));
        assert_eq!(run.best_position, Some(10_000));
        assert_eq!(run.samples.len(), 7);
        assert_eq!(agent.state(), FocusRunState::Converged);
        assert_eq!(agent.history().len(), 1);
    }

    #[test]
    fn sweep_positions_are_sorted_deduped_and_safety_clamped() {
        let mut config = FocusConfig::default();
        config.initial_step_size = 50;
        config.safety.min_position = 9_900;
        config.safety.max_position = 10_100;
        config.safety.max_step_single = 500;
        config.safety.max_total_travel_per_run = 300;

        let positions = sweep_positions(&config, 10_000);

        assert_eq!(positions, vec![9_900, 9_950, 10_000, 10_050, 10_100]);
    }
}
