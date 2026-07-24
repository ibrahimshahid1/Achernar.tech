//! Safety guard — Section 7.3.3.
//! Hardware firewall that validates all actions before motor commands.
//! This module MUST sit between AI output and hardware calls.

use crate::config::FocusSafetyConfig;
use crate::error::AchernarError;
use serde::{Deserialize, Serialize};

/// Result of a safety validation check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyVerdict {
    /// Action is safe to execute
    Approved,
    /// Action modified to stay within limits
    Modified {
        original: i32,
        clamped: i32,
        reason: String,
    },
    /// Action rejected entirely
    Rejected { reason: String },
}

/// Validates a proposed focus move against safety constraints.
pub struct FocusSafetyGuard {
    config: FocusSafetyConfig,
    total_travel_this_run: i32,
    consecutive_failures: u32,
}

impl FocusSafetyGuard {
    pub fn new(config: FocusSafetyConfig) -> Self {
        Self {
            config,
            total_travel_this_run: 0,
            consecutive_failures: 0,
        }
    }

    /// Reset counters for a new focus run.
    pub fn reset_run(&mut self) {
        self.total_travel_this_run = 0;
        self.consecutive_failures = 0;
    }

    /// Record a failed improvement attempt.
    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
    }

    /// Record a successful improvement.
    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
    }

    /// Validate a focus action before it reaches the hardware.
    /// This is the critical safety gate — every focus move MUST pass through here.
    pub fn validate_focus_action(
        &mut self,
        current_position: i32,
        proposed_step: i32,
        confidence: f64,
    ) -> Result<SafetyVerdict, AchernarError> {
        // Check confidence threshold
        if confidence < self.config.min_confidence_to_act {
            return Ok(SafetyVerdict::Rejected {
                reason: format!(
                    "Confidence {:.3} below threshold {:.3}",
                    confidence, self.config.min_confidence_to_act
                ),
            });
        }

        // Check consecutive failures
        if self.consecutive_failures >= self.config.max_failed_improvements {
            return Ok(SafetyVerdict::Rejected {
                reason: format!(
                    "Max consecutive failures ({}) reached",
                    self.config.max_failed_improvements
                ),
            });
        }

        // Clamp single step size
        let max_step = self.config.max_step_single.saturating_abs().max(1);
        let clamped_step = proposed_step.clamp(-max_step, max_step);

        // Check target position bounds
        let target = current_position + clamped_step;
        if target < self.config.min_position || target > self.config.max_position {
            return Ok(SafetyVerdict::Rejected {
                reason: format!(
                    "Target position {} outside bounds [{}, {}]",
                    target, self.config.min_position, self.config.max_position
                ),
            });
        }

        // Check total travel budget
        let new_total = self.total_travel_this_run + clamped_step.abs();
        if new_total > self.config.max_total_travel_per_run.max(0) {
            return Ok(SafetyVerdict::Rejected {
                reason: format!(
                    "Total travel {} would exceed run budget {}",
                    new_total,
                    self.config.max_total_travel_per_run.max(0)
                ),
            });
        }

        // Update tracking
        self.total_travel_this_run += clamped_step.abs();

        if clamped_step != proposed_step {
            Ok(SafetyVerdict::Modified {
                original: proposed_step,
                clamped: clamped_step,
                reason: format!("Clamped to max step size {max_step}"),
            })
        } else {
            Ok(SafetyVerdict::Approved)
        }
    }

    /// Get the configured settle time after a focus move.
    pub fn settle_ms(&self) -> u64 {
        self.config.settle_ms
    }

    /// Get backlash compensation steps.
    pub fn backlash_steps(&self) -> i32 {
        self.config.backlash_steps
    }
}

/// Mount safety guard — Section 8.3.
pub struct MountSafetyGuard {
    config: crate::config::MountSafetyConfig,
}

impl MountSafetyGuard {
    pub fn new(config: crate::config::MountSafetyConfig) -> Self {
        Self { config }
    }

    /// Validate a slew target against altitude and hour-angle limits.
    pub fn validate_slew(&self, alt: f64, hour_angle: f64) -> SafetyVerdict {
        if alt < self.config.min_altitude {
            return SafetyVerdict::Rejected {
                reason: format!(
                    "Altitude {:.1}° below minimum {:.1}°",
                    alt, self.config.min_altitude
                ),
            };
        }

        if hour_angle.abs() > self.config.max_hour_angle {
            return SafetyVerdict::Rejected {
                reason: format!(
                    "Hour angle {:.2}h exceeds limit {:.2}h",
                    hour_angle, self.config.max_hour_angle
                ),
            };
        }

        SafetyVerdict::Approved
    }
}
