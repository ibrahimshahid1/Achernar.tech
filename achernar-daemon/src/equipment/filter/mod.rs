//! Filter wheel behaviour (spec §3.4, §2.9, H8). Owner: WP1A-6.
//!
//! - [`reconcile`] (W-b): slot names and focus offsets reconciled with the
//!   identity profile at connect. The `[filter_wheel]` TOML arrays are seeds
//!   only and are applied solely through an explicit import (D17).
//! - [`motion`]: the worker-level wheel operation (`FilterMoveOp`, C-4/C-5)
//!   and the snapshot publisher that tracks the verified slot.
//! - [`check`]: the filter `SafetyCheck` plug-in (§2.9 step 3) with the
//!   switch history, the frequency limits and the plan lock.
//! - [`change`] (W-c): the v6 §8.2.1 filter-change handshake behind the gate,
//!   with resume blocks on timeout, wrong slot or an incomplete handshake.
//!
//! The drivers live in `alpaca/filterwheel.rs` and `sim/filterwheel.rs`.
//! Slots are 0-based everywhere inside the layer (W-a).

pub mod change;
pub mod check;
pub mod motion;
pub mod reconcile;

#[cfg(test)]
pub(crate) mod test_support;

use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::config::{EquipmentConfigs, FilterWheelConfig, FocusSafetyConfig, GuiderConfig};

use super::events::EquipmentEvent;
use super::snapshot::ResumeBlock;

/// Rolling cap on switches requested by automatic sources (policies,
/// triggers, sequencing) within one hour. Operators are bound only by
/// `min_switch_interval_s`. A behaviour default, not a rig value; `[filter_wheel]`
/// has no key for it yet (change note to WP0's config owner).
pub const DEFAULT_MAX_AUTOMATIC_SWITCHES_PER_HOUR: u32 = 60;

/// Poll interval while the wheel moves (§2.7: 4 Hz).
pub const MOVE_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Extra time a handshake waits for a worker beyond the operation's own
/// deadline before it gives up on the ticket (the worker should resolve it
/// at the deadline; this only bounds a worker that never does).
pub const WATCHDOG_GRACE: Duration = Duration::from_secs(5);

/// The reconciled wheel shared by the handshake, the snapshot publisher and
/// the connect path. Readers clone what they need and never hold the lock
/// across an `.await`.
pub type SharedWheel = Arc<RwLock<reconcile::ReconciledWheel>>;

/// A new shared wheel model.
pub fn shared_wheel(wheel: reconcile::ReconciledWheel) -> SharedWheel {
    Arc::new(RwLock::new(wheel))
}

/// A clone of the current reconciled wheel (a poisoned lock still yields its
/// data: the model is plain values).
pub fn read_wheel(shared: &SharedWheel) -> reconcile::ReconciledWheel {
    shared.read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Replace the shared wheel model.
pub fn write_wheel(shared: &SharedWheel, wheel: reconcile::ReconciledWheel) {
    *shared.write().unwrap_or_else(|e| e.into_inner()) = wheel;
}

/// Where the filter layer publishes events and raises resume blocks. The hub
/// (WP1A-2) implements it: it sequences events onto the bus and keeps the
/// resume-block registry (H22), which emits `resume_blocked`.
pub trait FilterEventSink: Send + Sync {
    fn publish(&self, event: EquipmentEvent);
    fn raise_resume_block(&self, block: ResumeBlock);
}

/// Behaviour settings of the filter layer, taken from `[filter_wheel]`, the
/// focuser move timeout (`[focus.safety] move_timeout_ms`) and the guider
/// timeouts (`[guider]`). No rig value lives here (D17): per-wheel settle
/// times and offsets come from the identity profile.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterSettings {
    /// Settle after a verified slot change; a profile `settle_ms` wins.
    pub settle_ms: u64,
    /// Deadline of one wheel move (C-4).
    pub max_switch_timeout_ms: u64,
    /// Minimum time between two commanded switches, every source.
    pub min_switch_interval_s: f64,
    /// Rolling one-hour cap for automatic sources; `None` disables it.
    pub max_automatic_switches_per_hour: Option<u32>,
    /// Apply reconciled (profile or device) offsets; never the TOML seeds.
    pub use_focus_offsets: bool,
    /// Pause PHD2 while a non-zero offset is applied (W-c step 4).
    pub pause_guiding_on_offset: bool,
    /// Start the offset move together with the wheel move (NINA style).
    pub parallel_offset_move: bool,
    /// Filter used for autofocus, when configured.
    pub af_filter: Option<String>,
    /// Budget for the focuser offset move.
    pub focuser_move_timeout_ms: u64,
    /// Budget for a guider pause or resume RPC.
    pub guider_rpc_timeout_s: u64,
    /// Budget for the guider to settle after resuming.
    pub guider_settle_timeout_s: f64,
}

impl Default for FilterSettings {
    fn default() -> Self {
        Self::from_parts(
            &FilterWheelConfig::default(),
            &FocusSafetyConfig::default(),
            &GuiderConfig::default(),
        )
    }
}

impl FilterSettings {
    pub fn from_parts(
        wheel: &FilterWheelConfig,
        focus_safety: &FocusSafetyConfig,
        guider: &GuiderConfig,
    ) -> Self {
        let af_filter = Some(wheel.af_filter.trim().to_string()).filter(|name| !name.is_empty());
        Self {
            settle_ms: wheel.settle_ms,
            max_switch_timeout_ms: wheel.max_switch_timeout_ms,
            min_switch_interval_s: wheel.min_switch_interval_s,
            max_automatic_switches_per_hour: Some(DEFAULT_MAX_AUTOMATIC_SWITCHES_PER_HOUR),
            use_focus_offsets: wheel.use_focus_offsets,
            pause_guiding_on_offset: wheel.pause_guiding_on_offset,
            parallel_offset_move: wheel.parallel_offset_move,
            af_filter,
            focuser_move_timeout_ms: focus_safety.move_timeout_ms,
            guider_rpc_timeout_s: guider.rpc_timeout_s,
            guider_settle_timeout_s: guider.settle_timeout_s,
        }
    }

    pub fn from_configs(configs: &EquipmentConfigs) -> Self {
        Self::from_parts(
            &configs.filter_wheel,
            &configs.focus_safety,
            &configs.guider,
        )
    }

    /// Deadline of one wheel move.
    pub fn switch_timeout(&self) -> Duration {
        Duration::from_millis(self.max_switch_timeout_ms.max(1))
    }

    /// Settle time after arrival: the profile value wins (W-c step 8).
    pub fn settle(&self, profile_settle_ms: Option<u64>) -> Duration {
        Duration::from_millis(profile_settle_ms.unwrap_or(self.settle_ms))
    }

    pub fn min_switch_interval(&self) -> Duration {
        if self.min_switch_interval_s.is_finite() && self.min_switch_interval_s > 0.0 {
            Duration::from_secs_f64(self.min_switch_interval_s)
        } else {
            Duration::ZERO
        }
    }

    pub fn focuser_move_timeout(&self) -> Duration {
        Duration::from_millis(self.focuser_move_timeout_ms.max(1))
    }

    pub fn guider_rpc_timeout(&self) -> Duration {
        Duration::from_secs(self.guider_rpc_timeout_s.max(1))
    }

    pub fn guider_settle_timeout(&self) -> Duration {
        if self.guider_settle_timeout_s.is_finite() && self.guider_settle_timeout_s > 0.0 {
            Duration::from_secs_f64(self.guider_settle_timeout_s)
        } else {
            Duration::from_secs(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_follow_the_config_defaults_and_the_profile_wins() {
        let settings = FilterSettings::default();
        assert_eq!(settings.settle_ms, 500);
        assert_eq!(settings.max_switch_timeout_ms, 10_000);
        assert!(settings.use_focus_offsets && settings.pause_guiding_on_offset);
        assert!(!settings.parallel_offset_move);
        assert_eq!(settings.af_filter, None);
        assert_eq!(settings.focuser_move_timeout_ms, 60_000);
        assert_eq!(settings.guider_rpc_timeout(), Duration::from_secs(10));
        assert_eq!(settings.settle(None), Duration::from_millis(500));
        assert_eq!(settings.settle(Some(1200)), Duration::from_millis(1200));
        assert_eq!(settings.min_switch_interval(), Duration::ZERO);
        assert_eq!(
            settings.max_automatic_switches_per_hour,
            Some(DEFAULT_MAX_AUTOMATIC_SWITCHES_PER_HOUR)
        );

        let wheel = FilterWheelConfig {
            af_filter: "  L ".into(),
            min_switch_interval_s: 2.5,
            ..FilterWheelConfig::default()
        };
        let settings = FilterSettings::from_parts(
            &wheel,
            &FocusSafetyConfig::default(),
            &GuiderConfig::default(),
        );
        assert_eq!(settings.af_filter.as_deref(), Some("L"));
        assert_eq!(settings.min_switch_interval(), Duration::from_millis(2500));
    }

    #[test]
    fn the_shared_wheel_hands_out_clones() {
        let shared = shared_wheel(reconcile::ReconciledWheel::default());
        assert!(read_wheel(&shared).is_empty());
        let wheel = reconcile::reconcile(
            &[crate::equipment::snapshot::SlotInfo {
                index0: 0,
                device_name: Some("L".into()),
                device_focus_offset: None,
            }],
            None,
        );
        write_wheel(&shared, wheel);
        assert_eq!(read_wheel(&shared).len(), 1);
    }
}
