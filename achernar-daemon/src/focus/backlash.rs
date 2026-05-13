//! Backlash compensation — Section 7.3.5.
//! Handles focuser backlash by always approaching from the same direction.

use serde::{Deserialize, Serialize};

/// Direction of focuser movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FocusDirection {
    Inward,  // Decreasing position
    Outward, // Increasing position
}

/// Backlash compensator state.
pub struct BacklashCompensator {
    enabled: bool,
    compensation_steps: i32,
    approach_direction: FocusDirection,
    last_direction: Option<FocusDirection>,
}

impl BacklashCompensator {
    pub fn new(config: &crate::config::BacklashConfig) -> Self {
        let approach_direction = match config.always_approach_from.as_deref() {
            Some("inward") => FocusDirection::Inward,
            _ => FocusDirection::Outward,
        };

        Self {
            enabled: config.enabled,
            compensation_steps: config.compensation_steps,
            approach_direction,
            last_direction: None,
        }
    }

    /// Compute the actual move sequence including backlash compensation.
    /// Returns a list of (steps, is_compensation) moves.
    pub fn compensate_move(&mut self, desired_step: i32) -> Vec<(i32, bool)> {
        if !self.enabled || desired_step == 0 {
            return vec![(desired_step, false)];
        }

        let desired_direction = if desired_step > 0 {
            FocusDirection::Outward
        } else {
            FocusDirection::Inward
        };

        let needs_compensation = self
            .last_direction
            .map(|last| last != desired_direction)
            .unwrap_or(false);

        self.last_direction = Some(desired_direction);

        if needs_compensation {
            // Overshoot and come back
            let overshoot = match desired_direction {
                FocusDirection::Outward => -self.compensation_steps,
                FocusDirection::Inward => self.compensation_steps,
            };
            vec![(desired_step + overshoot, false), (-overshoot, true)]
        } else {
            vec![(desired_step, false)]
        }
    }

    /// Reset backlash state (e.g., at start of new run).
    pub fn reset(&mut self) {
        self.last_direction = None;
    }
}
