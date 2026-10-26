//! The wheel operation and snapshot publishing (spec §3.1 C-4/C-5, §3.4
//! W-a, §2.5). Owner: WP1A-6.
//!
//! [`FilterMoveOp`] is the `Operation` the worker (WP1A-2) runs for
//! `FilterWheelCmd::SetPosition`; [`operation_for`] builds it. Completion
//! comes only from the completion property: Alpaca Position stops reading
//! -1 (INDI: `FILTER_SLOT` leaves Busy). It is never inferred by comparing
//! position with target. Verification then needs a second, stable read: a
//! stable read equal to the target succeeds, a stable different read is
//! `Failed{wrong_slot}`. A wheel has no halt, so `cancel` sends nothing and
//! a deadline yields `Unknown` (C-4), which raises a resume block.
//!
//! [`FilterWheelPublisher`] turns each poll into a [`FilterWheelSnapshot`]
//! and tracks the verified slot (W-c step 7): set by a successful move or by
//! the first stationary read after connect, cleared by any unexplained move,
//! a failed or unknown move, or a reconnect (§2.6: "a filter wheel's
//! verified slot is re-read before any move").

use std::time::Duration;

use crate::equipment::clock::duration_ns;
use crate::equipment::commands::FilterWheelCmd;
use crate::equipment::ops::{ActiveOp, OpKind, OpProgress, Operation};
use crate::equipment::snapshot::{FilterSlot, FilterWheelSnapshot};
use crate::equipment::traits::{Clock, FilterWheelDriver, PollMethod, RawPoll};
use crate::equipment::types::{
    DeviceError, DeviceResult, ErrorClass, ObsState, Observed, OpOutcome, Source,
};

use super::{FilterSettings, MOVE_POLL_INTERVAL};

/// Message prefix of a wrong-slot failure (W-c step 7).
pub const WRONG_SLOT: &str = "wrong_slot";

/// Message prefix of a wheel that never reported arrival (W-c step 6).
pub const ARRIVAL_TIMEOUT: &str = "timeout";

/// A wrong-slot failure.
pub fn wrong_slot_error(reported: u32, commanded: u32) -> DeviceError {
    DeviceError::new(
        ErrorClass::Driver,
        format!("{WRONG_SLOT}: the wheel reports slot {reported}, commanded {commanded}"),
    )
}

/// True for a failure produced by [`wrong_slot_error`].
pub fn is_wrong_slot(outcome: &OpOutcome) -> bool {
    matches!(outcome, OpOutcome::Failed(err) if err.message.starts_with(WRONG_SLOT))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MovePhase {
    /// Waiting for the completion property.
    Moving,
    /// Completion seen; waiting for a second, stable read.
    Verifying { last: u32 },
    /// Verified; optional worker-level settle.
    Settling { until_mono_ns: u64 },
}

/// `FilterWheelCmd::SetPosition` as a worker operation.
#[derive(Debug, Clone)]
pub struct FilterMoveOp {
    target: u32,
    deadline: Duration,
    settle: Duration,
    started_mono_ns: Option<u64>,
    phase: MovePhase,
}

impl FilterMoveOp {
    /// A move to `target` with the `max_switch_timeout_ms` deadline. It does
    /// not settle: the handshake settles (W-c step 8) and records the actual
    /// settle time.
    pub fn new(target: u32, settings: &FilterSettings) -> Self {
        Self {
            target,
            deadline: settings.switch_timeout(),
            settle: Duration::ZERO,
            started_mono_ns: None,
            phase: MovePhase::Moving,
        }
    }

    /// Settle inside the operation, for callers that use the wheel without
    /// the handshake.
    pub fn with_settle(mut self, settle: Duration) -> Self {
        self.settle = settle;
        self
    }

    pub fn target(&self) -> u32 {
        self.target
    }

    fn timed_out(&self, now: u64) -> bool {
        self.started_mono_ns
            .is_some_and(|start| now.saturating_sub(start) >= duration_ns(self.deadline))
    }

    fn keep_polling() -> OpProgress {
        OpProgress::Continue {
            next_poll: MOVE_POLL_INTERVAL,
        }
    }
}

/// The operation for a wheel command (for the worker's dispatch).
pub fn operation_for(cmd: &FilterWheelCmd, settings: &FilterSettings) -> FilterMoveOp {
    match cmd {
        FilterWheelCmd::SetPosition(target) => FilterMoveOp::new(*target, settings),
    }
}

impl<D: FilterWheelDriver + ?Sized> Operation<D> for FilterMoveOp {
    fn kind(&self) -> OpKind {
        OpKind::FilterMove
    }

    fn deadline(&self) -> Duration {
        self.deadline
    }

    fn start(&mut self, drv: &mut D, clock: &dyn Clock) -> DeviceResult<()> {
        self.started_mono_ns = Some(clock.now_mono_ns());
        self.phase = MovePhase::Moving;
        drv.set_position(self.target)
    }

    fn step(&mut self, drv: &mut D, clock: &dyn Clock) -> OpProgress {
        let now = clock.now_mono_ns();
        if let MovePhase::Settling { until_mono_ns } = self.phase {
            return if now >= until_mono_ns {
                OpProgress::Done(OpOutcome::Succeeded)
            } else {
                OpProgress::Settling { until_mono_ns }
            };
        }
        if self.timed_out(now) {
            return OpProgress::Done(OpOutcome::Unknown(DeviceError::outcome_unknown(format!(
                "{ARRIVAL_TIMEOUT}: the wheel did not report arrival at slot {} within {} ms; \
                 a filter wheel has no halt, so its position is unknown",
                self.target,
                self.deadline.as_millis()
            ))));
        }
        let read = match drv.position() {
            Ok(read) => read,
            Err(err)
                if matches!(
                    err.class,
                    ErrorClass::NotConnected | ErrorClass::Disconnected
                ) =>
            {
                return OpProgress::Done(OpOutcome::Unknown(DeviceError::outcome_unknown(
                    format!(
                        "the wheel connection was lost during the move: {}",
                        err.message
                    ),
                )));
            }
            // A failed read is retried on the next poll, until the deadline.
            Err(_) => return Self::keep_polling(),
        };
        match (self.phase, read) {
            (_, None) => {
                self.phase = MovePhase::Moving;
                Self::keep_polling()
            }
            (MovePhase::Moving, Some(slot)) => {
                self.phase = MovePhase::Verifying { last: slot };
                Self::keep_polling()
            }
            (MovePhase::Verifying { .. }, Some(slot)) if slot == self.target => {
                if self.settle.is_zero() {
                    OpProgress::Done(OpOutcome::Succeeded)
                } else {
                    let until_mono_ns = now.saturating_add(duration_ns(self.settle));
                    self.phase = MovePhase::Settling { until_mono_ns };
                    OpProgress::Settling { until_mono_ns }
                }
            }
            (MovePhase::Verifying { last }, Some(slot)) if slot == last => {
                OpProgress::Done(OpOutcome::Failed(wrong_slot_error(slot, self.target)))
            }
            (MovePhase::Verifying { .. }, Some(slot)) => {
                // Still changing (some drivers report intermediate slots).
                self.phase = MovePhase::Verifying { last: slot };
                Self::keep_polling()
            }
            (MovePhase::Settling { until_mono_ns }, Some(_)) => {
                OpProgress::Settling { until_mono_ns }
            }
        }
    }

    /// A filter wheel has no halt: nothing is sent, and the worker resolves
    /// the outcome `Unknown` (ops.rs contract, C-4).
    fn cancel(&mut self, _drv: &mut D) -> DeviceResult<()> {
        Ok(())
    }
}

fn source_for(method: PollMethod) -> Source {
    match method {
        PollMethod::DeviceState => Source::DeviceState,
        PollMethod::PerMember | PollMethod::InProcess => Source::Property,
        PollMethod::PushCache => Source::IndiPush,
    }
}

/// Builds wheel snapshots from polls and tracks the verified slot.
#[derive(Debug, Clone, Default)]
pub struct FilterWheelPublisher {
    verified_slot: Option<u32>,
    /// An unexplained move, a wrong slot or an unknown outcome: nothing is
    /// verified until a move succeeds.
    suspect: bool,
}

impl FilterWheelPublisher {
    pub fn new() -> Self {
        Self::default()
    }

    /// Connect or reconnect: the slot is re-read before any move.
    pub fn on_connect(&mut self) {
        self.verified_slot = None;
        self.suspect = false;
    }

    /// A wheel operation resolved.
    pub fn on_op_done(&mut self, target: u32, outcome: &OpOutcome) {
        if outcome.is_success() {
            self.verified_slot = Some(target);
            self.suspect = false;
        } else {
            self.verified_slot = None;
            self.suspect = true;
        }
    }

    pub fn verified_slot(&self) -> Option<u32> {
        self.verified_slot
    }

    pub fn is_suspect(&self) -> bool {
        self.suspect
    }

    /// The snapshot after one poll. `position` is the driver's accessor
    /// result; `raw` supplies the read's timing.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        &mut self,
        prev: &FilterWheelSnapshot,
        position: DeviceResult<Option<u32>>,
        raw: Option<&RawPoll>,
        slots: Vec<FilterSlot>,
        active_op: Option<ActiveOp>,
        clock: &dyn Clock,
    ) -> FilterWheelSnapshot {
        let member = raw.and_then(|raw| {
            raw.members
                .iter()
                .find(|m| m.name.eq_ignore_ascii_case("position"))
                .or_else(|| raw.members.first())
        });
        let source = raw.map_or(Source::Property, |raw| source_for(raw.method));
        let observed = |value: Option<u32>| Observed {
            value,
            state: ObsState::Fresh,
            observed_at: Some(
                member
                    .and_then(|m| m.observed_at)
                    .unwrap_or_else(|| clock.now_utc()),
            ),
            mono_ns: Some(
                member
                    .and_then(|m| m.mono_ns)
                    .unwrap_or_else(|| clock.now_mono_ns()),
            ),
            device_time: raw.and_then(|raw| raw.device_time),
            latency_ms: member.and_then(|m| m.latency_ms),
            source,
        };
        let op_running = active_op.is_some();
        let (position, moving) = match position {
            Ok(Some(slot)) => {
                match self.verified_slot {
                    Some(verified) if verified != slot => {
                        // The wheel moved without a verified move.
                        self.verified_slot = None;
                        self.suspect = true;
                    }
                    None if !self.suspect && !op_running => self.verified_slot = Some(slot),
                    _ => {}
                }
                (observed(Some(slot)), false)
            }
            Ok(None) => {
                self.verified_slot = None;
                if !op_running {
                    self.suspect = true;
                }
                (observed(None), true)
            }
            Err(err) if err.is_not_set() => (Observed::not_set(source), false),
            Err(_) => (prev.position.clone().into_stale(), prev.moving),
        };
        let verified = position.is_fresh()
            && !moving
            && !self.suspect
            && position.value.is_some()
            && position.value == self.verified_slot;
        FilterWheelSnapshot {
            connection_id: prev.connection_id,
            position,
            moving,
            slots,
            verified_slot: self.verified_slot,
            verified,
            active_op,
            published_at: clock.now_utc(),
            published_mono_ns: clock.now_mono_ns(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::sim::filterwheel::{SimFilterWheel, SimFilterWheelConfig};
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::{Device, PollPlan};
    use std::sync::Arc;

    fn wheel(clock: &Arc<ManualClock>) -> SimFilterWheel {
        let mut sim = SimFilterWheel::new(
            SimFilterWheelConfig::default(),
            Arc::clone(clock) as Arc<dyn Clock>,
        );
        sim.connect(Duration::from_secs(1)).unwrap();
        sim
    }

    /// Drive an op the way the worker does: poll, then step.
    fn drive(
        op: &mut FilterMoveOp,
        sim: &mut SimFilterWheel,
        clock: &ManualClock,
        max_ticks: usize,
    ) -> (OpOutcome, usize) {
        for tick in 0..max_ticks {
            clock.advance(MOVE_POLL_INTERVAL);
            let _ = sim.poll(&PollPlan::default());
            if let OpProgress::Done(outcome) = op.step(sim, clock) {
                return (outcome, tick + 1);
            }
        }
        panic!("the operation did not finish in {max_ticks} ticks");
    }

    #[test]
    fn a_move_completes_from_the_completion_property_then_verifies() {
        let clock = Arc::new(ManualClock::new());
        let mut sim = wheel(&clock);
        let mut op = FilterMoveOp::new(3, &FilterSettings::default());
        assert_eq!(Operation::<SimFilterWheel>::kind(&op), OpKind::FilterMove);
        op.start(&mut sim, clock.as_ref()).unwrap();
        assert_eq!(sim.control().commanded(), vec![3]);
        // Right after the command the wheel counts as moving until re-read.
        assert_eq!(sim.position().unwrap(), None);
        let (outcome, ticks) = drive(&mut op, &mut sim, &clock, 100);
        assert_eq!(outcome, OpOutcome::Succeeded);
        // 3 slots at 600 ms plus 200 ms overhead = 2000 ms of travel, then
        // one extra verification read.
        assert!(ticks >= 9, "{ticks}");
        assert_eq!(sim.position().unwrap(), Some(3));
    }

    #[test]
    fn a_stable_wrong_slot_fails_and_a_jam_times_out_unknown() {
        let clock = Arc::new(ManualClock::new());
        let mut sim = wheel(&clock);
        sim.control().wrong_slot_next_move(2);
        let mut op = FilterMoveOp::new(4, &FilterSettings::default());
        op.start(&mut sim, clock.as_ref()).unwrap();
        let (outcome, _) = drive(&mut op, &mut sim, &clock, 100);
        assert!(is_wrong_slot(&outcome), "{outcome:?}");
        assert!(matches!(&outcome, OpOutcome::Failed(e) if e.message.contains("slot 2")));

        sim.control().jam_next_move();
        let settings = FilterSettings {
            max_switch_timeout_ms: 3000,
            ..FilterSettings::default()
        };
        let mut op = FilterMoveOp::new(5, &settings);
        assert_eq!(
            Operation::<SimFilterWheel>::deadline(&op),
            Duration::from_secs(3)
        );
        op.start(&mut sim, clock.as_ref()).unwrap();
        let (outcome, ticks) = drive(&mut op, &mut sim, &clock, 100);
        assert_eq!(ticks, 12, "3 s at 4 Hz");
        match outcome {
            OpOutcome::Unknown(err) => {
                assert_eq!(err.class, ErrorClass::OutcomeUnknown);
                assert!(err.message.starts_with(ARRIVAL_TIMEOUT));
            }
            other => panic!("expected Unknown, got {other:?}"),
        }
        // No halt exists, so cancel sends nothing.
        let before = sim.control().commanded();
        assert!(Operation::<SimFilterWheel>::cancel(&mut op, &mut sim).is_ok());
        assert_eq!(sim.control().commanded(), before);
    }

    #[test]
    fn intermediate_reads_keep_verifying_and_failed_reads_are_retried() {
        let clock = Arc::new(ManualClock::new());
        let mut sim = wheel(&clock);
        let mut op = FilterMoveOp::new(1, &FilterSettings::default())
            .with_settle(Duration::from_millis(500));
        op.start(&mut sim, clock.as_ref()).unwrap();
        sim.control().fail_next_polls(3);
        let (outcome, _) = drive(&mut op, &mut sim, &clock, 100);
        assert_eq!(outcome, OpOutcome::Succeeded);

        // A lost connection mid-move is an unknown outcome.
        let mut op = FilterMoveOp::new(2, &FilterSettings::default());
        op.start(&mut sim, clock.as_ref()).unwrap();
        sim.control().drop_connection();
        let (outcome, _) = drive(&mut op, &mut sim, &clock, 10);
        assert!(matches!(outcome, OpOutcome::Unknown(_)), "{outcome:?}");
    }

    #[test]
    fn the_publisher_tracks_the_verified_slot() {
        let clock = ManualClock::new();
        let mut publisher = FilterWheelPublisher::new();
        let prev = FilterWheelSnapshot::default();
        // First stationary read after connect is a verification read.
        let snap = publisher.build(&prev, Ok(Some(2)), None, Vec::new(), None, &clock);
        assert!(snap.verified && snap.verified_slot == Some(2));
        assert_eq!(snap.position.value, Some(2));
        assert!(!snap.moving);

        // An unexplained move clears it and makes the wheel suspect.
        let snap = publisher.build(&snap, Ok(None), None, Vec::new(), None, &clock);
        assert!(snap.moving && !snap.verified);
        assert_eq!(snap.position.state, ObsState::Fresh);
        let snap = publisher.build(&snap, Ok(Some(4)), None, Vec::new(), None, &clock);
        assert!(!snap.verified, "suspect until a verified move");

        // A successful move verifies again.
        publisher.on_op_done(4, &OpOutcome::Succeeded);
        let snap = publisher.build(&snap, Ok(Some(4)), None, Vec::new(), None, &clock);
        assert!(snap.verified);

        // A failed read keeps the last value but never as fresh.
        let snap = publisher.build(
            &snap,
            Err(DeviceError::timeout("slow")),
            None,
            Vec::new(),
            None,
            &clock,
        );
        assert_eq!(snap.position.state, ObsState::Stale);
        assert_eq!(snap.position.value, Some(4));
        assert!(!snap.verified);

        // An unknown outcome clears verification; a reconnect resets it.
        publisher.on_op_done(1, &OpOutcome::Unknown(DeviceError::outcome_unknown("t")));
        assert!(publisher.is_suspect() && publisher.verified_slot().is_none());
        publisher.on_connect();
        let snap = publisher.build(&snap, Ok(Some(1)), None, Vec::new(), None, &clock);
        assert!(snap.verified);
    }

    #[test]
    fn snapshots_take_timing_from_the_poll() {
        let clock = Arc::new(ManualClock::new());
        let mut sim = wheel(&clock);
        clock.advance_ms(40);
        let raw = sim.poll(&PollPlan::default()).unwrap();
        let mut publisher = FilterWheelPublisher::new();
        let snap = publisher.build(
            &FilterWheelSnapshot::default(),
            sim.position(),
            Some(&raw),
            Vec::new(),
            None,
            clock.as_ref(),
        );
        assert_eq!(snap.position.mono_ns, raw.members[0].mono_ns);
        assert_eq!(snap.position.source, Source::Property);
        let op = operation_for(&FilterWheelCmd::SetPosition(6), &FilterSettings::default());
        assert_eq!(op.target(), 6);
    }
}
