//! Long operations (spec §2.4, §2.5, §3.1 C-4/C-5). Frozen by WP0.
//!
//! Device work packages implement [`Operation`] for each long operation of
//! their kind; the worker (WP1A-2) drives every operation the same way:
//!
//! 1. `start` sends the initiator (for example `SlewToCoordinatesAsync`).
//! 2. After every poll the worker calls `step`, which evaluates the
//!    completion property (never position-versus-target, C-5), then settles,
//!    then verifies.
//! 3. A passed `deadline`, a cancel or a preemption calls `cancel`, which
//!    **halts the hardware** (C-4). The outcome is then `TimedOut` or
//!    `Aborted { by }`.
//!
//! Only one operation is active per device.

use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::traits::Clock;
use super::types::{CommandId, DeviceResult, OpOutcome};

/// Every operation kind the layer runs or audits. Serde names are the
/// `operation` values of the `equipment_overheads` view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpKind {
    // Mount
    Slew,
    SlewAltAz,
    AbortSlew,
    Sync,
    SetTracking,
    SetTrackingRate,
    Park,
    Unpark,
    SetPark,
    Home,
    MoveAxis,
    PulseGuide,
    SetGuideRates,
    SetSite,
    SetUtc,
    SetSideOfPier,
    DestinationSideOfPier,
    // Focuser
    FocuserMove,
    FocuserHalt,
    FocuserTempComp,
    FocuserSync,
    FocuserSetPosition,
    // Filter wheel
    FilterMove,
    FilterHandshake,
    FilterImportSeed,
    // Guider
    GuiderStart,
    GuiderStop,
    GuiderPause,
    GuiderResume,
    Dither,
    SetAlgoParam,
    GuiderRefresh,
    // Rotator
    RotatorMove,
    RotatorSync,
    RotatorHalt,
    RotatorReverse,
    // Cover calibrator
    CoverOpen,
    CoverClose,
    CoverHalt,
    CalibratorOn,
    CalibratorOff,
    // Switch
    SwitchSet,
    // Dome
    DomeOpenShutter,
    DomeCloseShutter,
    DomePark,
    DomeSlew,
    DomeAbort,
    DomeSlave,
    // Environment
    Refresh,
    // Compound and cross-cutting
    Center,
    Flip,
    Autofocus,
    Capture,
    SiteSync,
    ClearResumeBlock,
    Abort,
    #[serde(rename = "estop")]
    EStop,
    Connect,
    Disconnect,
}

impl OpKind {
    /// True for operations that move hardware and therefore halt it on cancel.
    pub fn moves_hardware(self) -> bool {
        matches!(
            self,
            OpKind::Slew
                | OpKind::SlewAltAz
                | OpKind::Park
                | OpKind::Unpark
                | OpKind::Home
                | OpKind::MoveAxis
                | OpKind::PulseGuide
                | OpKind::FocuserMove
                | OpKind::FilterMove
                | OpKind::FilterHandshake
                | OpKind::Dither
                | OpKind::RotatorMove
                | OpKind::CoverOpen
                | OpKind::CoverClose
                | OpKind::DomeOpenShutter
                | OpKind::DomeCloseShutter
                | OpKind::DomePark
                | OpKind::DomeSlew
                | OpKind::Center
                | OpKind::Flip
                | OpKind::Autofocus
        )
    }
}

/// Phase of the active operation, published in snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpPhase {
    #[default]
    Starting,
    InProgress,
    Settling,
    Verifying,
    Cancelling,
}

/// The device's active operation, as published in its snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActiveOp {
    pub command_id: CommandId,
    pub kind: OpKind,
    pub phase: OpPhase,
    pub started_at: DateTime<Utc>,
    pub started_mono_ns: u64,
    pub deadline_mono_ns: u64,
}

/// What `step` tells the worker after each poll.
#[derive(Debug, Clone, PartialEq)]
pub enum OpProgress {
    /// Not complete; poll again after `next_poll` (the active cadence).
    Continue { next_poll: Duration },
    /// Complete and settling until the monotonic time given.
    Settling { until_mono_ns: u64 },
    /// Finished. `Unknown` raises a resume block (C-6).
    Done(OpOutcome),
}

/// A long operation on a driver of type `D` (for example
/// `dyn MountDriver`). Called only on the device's worker thread.
pub trait Operation<D: ?Sized>: Send {
    /// Slew, Park, Unpark, Home, FocuserMove, FilterMove, ...
    fn kind(&self) -> OpKind;
    /// Total budget from `start`; on expiry the worker calls `cancel` and
    /// resolves `TimedOut`.
    fn deadline(&self) -> Duration;
    /// Sends the initiator.
    fn start(&mut self, drv: &mut D, clock: &dyn Clock) -> DeviceResult<()>;
    /// Completion predicate, settle and verification.
    fn step(&mut self, drv: &mut D, clock: &dyn Clock) -> OpProgress;
    /// Halts the hardware (C-4). A device with no halt (a filter wheel)
    /// returns `Ok` and the worker resolves the outcome `Unknown`.
    fn cancel(&mut self, drv: &mut D) -> DeviceResult<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::DeviceError;

    /// A toy driver: a counter that "moves" one step per poll.
    struct Counter {
        position: i32,
        target: Option<i32>,
        halted: bool,
    }

    struct MoveTo {
        target: i32,
    }

    impl Operation<Counter> for MoveTo {
        fn kind(&self) -> OpKind {
            OpKind::FocuserMove
        }
        fn deadline(&self) -> Duration {
            Duration::from_secs(5)
        }
        fn start(&mut self, drv: &mut Counter, _clock: &dyn Clock) -> DeviceResult<()> {
            drv.target = Some(self.target);
            Ok(())
        }
        fn step(&mut self, drv: &mut Counter, clock: &dyn Clock) -> OpProgress {
            match drv.target {
                Some(t) if drv.position < t => {
                    drv.position += 1;
                    OpProgress::Continue {
                        next_poll: Duration::from_millis(100),
                    }
                }
                Some(_) => {
                    drv.target = None;
                    OpProgress::Settling {
                        until_mono_ns: clock.now_mono_ns() + 1_000,
                    }
                }
                None => OpProgress::Done(OpOutcome::Succeeded),
            }
        }
        fn cancel(&mut self, drv: &mut Counter) -> DeviceResult<()> {
            drv.halted = true;
            drv.target = None;
            Err(DeviceError::cancelled("halted"))
        }
    }

    #[test]
    fn an_operation_is_driven_through_start_step_and_cancel() {
        let clock = ManualClock::new();
        let mut drv = Counter {
            position: 0,
            target: None,
            halted: false,
        };
        let mut op: Box<dyn Operation<Counter>> = Box::new(MoveTo { target: 2 });
        assert_eq!(op.kind(), OpKind::FocuserMove);
        assert!(op.kind().moves_hardware());
        op.start(&mut drv, &clock).unwrap();
        assert!(matches!(
            op.step(&mut drv, &clock),
            OpProgress::Continue { .. }
        ));
        assert!(matches!(
            op.step(&mut drv, &clock),
            OpProgress::Continue { .. }
        ));
        assert!(matches!(
            op.step(&mut drv, &clock),
            OpProgress::Settling { .. }
        ));
        assert_eq!(
            op.step(&mut drv, &clock),
            OpProgress::Done(OpOutcome::Succeeded)
        );
        assert!(op.cancel(&mut drv).is_err());
        assert!(drv.halted);
    }

    #[test]
    fn op_kinds_use_overhead_view_names() {
        assert_eq!(
            serde_json::to_value(OpKind::FilterHandshake).unwrap(),
            serde_json::json!("filter_handshake")
        );
        assert_eq!(
            serde_json::to_value(OpKind::EStop).unwrap(),
            serde_json::json!("estop")
        );
    }
}
