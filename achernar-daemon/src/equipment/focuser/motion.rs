//! Focuser operations (spec §3.3 F-a, F-b, F-c, F-h), driven by the device
//! worker through the frozen [`Operation`] trait.
//!
//! A move ([`FocuserMoveOp`]):
//!
//! 1. refuses to start while the camera integrates an exposure (A06; the
//!    gate checks this first, the operation re-checks it);
//! 2. resolves the logical target: `MoveTo` needs a reliable origin;
//!    `MoveBy` on an unknown origin is limited to `max_step_single`;
//! 3. clamps the target to the travel bounds (`[focus.safety]`, the profile
//!    and `[0, MaxStep]`);
//! 4. plans the backlash legs (F-e) and splits each leg into chunks of at
//!    most `MaxIncrement` steps, only when `MaxIncrement` is smaller than the
//!    leg;
//! 5. switches TempComp off when configured (F-d);
//! 6. sends each chunk and waits for `IsMoving` false from a sample taken
//!    after the chunk was sent: completion is never inferred from position
//!    (C-5). While moving it watches for a stalled position on an absolute
//!    focuser (`stuck_timeout_s`, then halt and fail `stuck`) and for the
//!    per-chunk `move_timeout_ms`;
//! 7. settles (`settle_ms` from the profile or `[focus.safety]`), reads the
//!    position from a poll taken after the settle, and re-issues an
//!    off-target arrival at most `max_reissue` times;
//! 8. restores TempComp, updates the logical position, persists a relative
//!    focuser's position, and writes exactly one `focuser_moves` row.
//!
//! Cancel, abort, preemption and timeout all halt the hardware (F-c). An
//! interrupted move of a relative focuser loses its origin.
//!
//! Events: `focuser_move_commanded`, `focuser_move_started` (again for each
//! re-issue), `focuser_move_stopped{position, outcome}` once when the move
//! resolves, `focuser_settled` after a successful verification,
//! `focuser_halted`, `tempcomp_changed`, `focuser_position_reset`.

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use super::backlash::{self, BacklashPlan, LegPurpose};
use super::tempcomp::TempCompGuard;
use super::{CloseState, SessionInner, ACTIVE_POLL_INTERVAL};
use crate::equipment::clock::duration_ns;
use crate::equipment::events::{EventKind, FocuserMoveRecord, GroundTruthRecord};
use crate::equipment::ops::{OpKind, OpProgress, Operation};
use crate::equipment::snapshot::{FocuserState, MoveMode, PositionSource};
use crate::equipment::traits::{Clock, FocuserDriver};
use crate::equipment::types::{
    CommandId, CommandSource, DeviceError, DeviceResult, ErrorClass, ObsState, OpOutcome,
};

/// Split a raw move into absolute targets at most `max_increment` apart.
/// Chunking applies only when `MaxIncrement` is smaller than the move (F-a).
pub fn chunk_targets(from: i32, to: i32, max_increment: Option<i32>) -> Vec<i32> {
    if from == to {
        return Vec::new();
    }
    let span = (i64::from(to) - i64::from(from)).abs();
    let step = match max_increment.filter(|m| *m > 0) {
        Some(max) if i64::from(max) < span => i64::from(max),
        _ => return vec![to],
    };
    let sign = if to > from { 1 } else { -1 };
    let mut out = Vec::new();
    let mut at = i64::from(from);
    while (i64::from(to) - at).abs() > step {
        at += sign * step;
        out.push(at as i32);
    }
    out.push(to);
    out
}

/// A commanded move: a logical absolute target, or a relative step count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveRequest {
    To(i32),
    By(i32),
}

impl MoveRequest {
    fn mode(self) -> MoveMode {
        match self {
            MoveRequest::To(_) => MoveMode::Absolute,
            MoveRequest::By(_) => MoveMode::Relative,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentCommand {
    Absolute(i32),
    Relative(i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Segment {
    command: SegmentCommand,
    /// Pause for the settle time after this chunk (the end of an overshoot
    /// leg), before the return leg.
    pause_after: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    NotStarted,
    Moving {
        sent_mono: u64,
        last_raw: Option<i32>,
        last_change_mono: u64,
    },
    Pausing {
        until: u64,
    },
    Settling {
        until: u64,
    },
    Finished,
}

fn now_after(clock: &dyn Clock, duration: Duration) -> u64 {
    clock.now_mono_ns().saturating_add(duration_ns(duration))
}

/// An initiator that failed in transit may still have reached the device:
/// its outcome is unknown, not failed (C-6).
fn initiator_error(err: DeviceError) -> DeviceError {
    match err.class {
        ErrorClass::Timeout | ErrorClass::Transport | ErrorClass::Disconnected => DeviceError {
            class: ErrorClass::OutcomeUnknown,
            code: err.code,
            message: format!("move command outcome unknown: {}", err.message),
        },
        _ => err,
    }
}

fn outcome_for(err: &DeviceError) -> OpOutcome {
    if err.class == ErrorClass::OutcomeUnknown {
        OpOutcome::Unknown(err.clone())
    } else {
        OpOutcome::Failed(err.clone())
    }
}

/// True when the observation was taken at or after `mono` (so it reflects
/// the command sent at `mono`).
fn sampled_since<T>(value: &crate::equipment::types::Observed<T>, mono: u64) -> bool {
    value.state == ObsState::Fresh && value.mono_ns.is_some_and(|at| at >= mono)
}

// ---------------------------------------------------------------------------
// Move
// ---------------------------------------------------------------------------

/// `FocuserCmd::MoveTo` / `MoveBy` (F-a, F-b, F-d, F-e).
pub struct FocuserMoveOp {
    inner: Arc<SessionInner>,
    move_id: Uuid,
    command_id: CommandId,
    source: CommandSource,
    run_id: Option<String>,
    request: MoveRequest,
    commanded_at: DateTime<Utc>,
    started_mono: Option<u64>,
    tempcomp: TempCompGuard,
    plan: Option<BacklashPlan>,
    segments: Vec<Segment>,
    next_segment: usize,
    phase: Phase,
    position_source: PositionSource,
    logical_start: Option<i32>,
    logical_target: Option<i32>,
    raw_start: Option<i32>,
    /// Relative focusers: steps of the chunks confirmed complete.
    executed_steps: i64,
    /// A chunk was sent and has not been confirmed complete.
    in_flight: bool,
    /// The profile row was marked `moving` and must be closed.
    marked_moving: bool,
    reissues: u32,
    temperature_start: Option<f64>,
    started_at: Option<DateTime<Utc>>,
    stopped_at: Option<DateTime<Utc>>,
    settled_at: Option<DateTime<Utc>>,
    outcome: Option<OpOutcome>,
}

impl FocuserMoveOp {
    pub(super) fn new(
        inner: Arc<SessionInner>,
        request: MoveRequest,
        command_id: CommandId,
        source: CommandSource,
        run_id: Option<String>,
    ) -> Self {
        let tempcomp = if run_id.is_some() {
            TempCompGuard::for_autofocus()
        } else {
            TempCompGuard::new(inner.settings.tempcomp_during_moves)
        };
        let commanded_at = inner.clock.now_utc();
        let position_source = inner.lock().source;
        Self {
            inner,
            move_id: Uuid::new_v4(),
            command_id,
            source,
            run_id,
            request,
            commanded_at,
            started_mono: None,
            tempcomp,
            plan: None,
            segments: Vec::new(),
            next_segment: 0,
            phase: Phase::NotStarted,
            position_source,
            logical_start: None,
            logical_target: None,
            raw_start: None,
            executed_steps: 0,
            in_flight: false,
            marked_moving: false,
            reissues: 0,
            temperature_start: None,
            started_at: None,
            stopped_at: None,
            settled_at: None,
            outcome: None,
        }
    }

    /// The `focuser_moves.id` / event `move_id` of this move.
    pub fn move_id(&self) -> Uuid {
        self.move_id
    }

    fn absolute(&self) -> bool {
        self.inner.caps.absolute
    }

    fn commanded_target(&self) -> i32 {
        match self.request {
            MoveRequest::To(target) => target,
            // Relative: the logical target when the origin is known, else the
            // step count (mode `rel` with no logical start says which).
            MoveRequest::By(steps) => self
                .logical_start
                .map(|start| start.saturating_add(steps))
                .unwrap_or(steps),
        }
    }

    fn event(&self, kind: EventKind) {
        self.inner.event(kind, Some(self.command_id));
    }

    /// Steps the planned moves need in the worst case, before `start`.
    fn estimated_segments(&self) -> u32 {
        let caps = &self.inner.caps;
        let span = match self.request {
            MoveRequest::By(steps) => i64::from(steps).abs(),
            MoveRequest::To(_) => i64::from(caps.max_step.unwrap_or(0).max(0)),
        };
        let per = i64::from(caps.max_increment.filter(|m| *m > 0).unwrap_or(i32::MAX));
        let chunks = (span / per.max(1) + 1).min(10_000) as u32;
        chunks + 2
    }

    fn planned_segments(&self) -> u32 {
        if self.started_mono.is_some() {
            u32::try_from(self.segments.len())
                .unwrap_or(u32::MAX)
                .max(1)
        } else {
            self.estimated_segments()
        }
    }

    fn prepare(&mut self, drv: &mut dyn FocuserDriver, clock: &dyn Clock) -> DeviceResult<()> {
        let inner = Arc::clone(&self.inner);
        let settings = &inner.settings;
        if inner.camera_exposing() {
            return Err(DeviceError::new(
                ErrorClass::SafetyRejected,
                "the camera is integrating an exposure; focuser moves wait for the exposure \
                 boundary (A06)",
            ));
        }
        let state = drv.state()?;
        self.temperature_start = state.temperature_c.known().copied();
        let (backlash_state, logical_known) = {
            let s = inner.lock();
            self.position_source = s.source;
            (s.backlash, s.logical)
        };
        let reliable = self.position_source.is_reliable();

        // Logical start.
        if self.absolute() {
            let raw = state.position.fresh_value().copied().ok_or_else(|| {
                DeviceError::invalid_operation(
                    "the focuser position is unknown; an absolute move cannot be planned",
                )
            })?;
            self.raw_start = Some(raw);
            self.logical_start = Some(raw - backlash_state.offset);
        } else {
            self.logical_start = if reliable { logical_known } else { None };
        }

        // Logical target, F-b restrictions, clamp (F-a).
        let bounds = inner.travel_bounds();
        let target = match self.request {
            MoveRequest::To(target) => {
                if !reliable {
                    return Err(DeviceError::invalid_operation(
                        "absolute moves are refused while the focuser origin is unknown; use a \
                         relative move or set the position",
                    ));
                }
                Some(target)
            }
            MoveRequest::By(steps) => {
                if !reliable && steps.unsigned_abs() > settings.max_step_single.unsigned_abs() {
                    return Err(DeviceError::invalid_value(format!(
                        "relative move of {steps} steps exceeds max_step_single {} while the \
                         focuser origin is unknown",
                        settings.max_step_single
                    )));
                }
                self.logical_start.map(|start| start.saturating_add(steps))
            }
        };
        let target = match (target, self.logical_start) {
            (Some(target), Some(_)) => {
                let (lo, hi) = bounds.ok_or_else(|| {
                    DeviceError::invalid_value(
                        "the focuser travel bounds are empty; check [focus.safety] and the profile",
                    )
                })?;
                let clamped = target.clamp(lo, hi);
                if clamped != target {
                    tracing::info!(
                        requested = target,
                        clamped,
                        lo,
                        hi,
                        "focuser target clamped to the travel bounds"
                    );
                }
                Some(clamped)
            }
            _ => None,
        };
        self.logical_target = target;

        // Backlash legs in raw (absolute) or step-counter (relative) space.
        let base = self.logical_start.unwrap_or(0);
        let goal = match (target, self.request) {
            (Some(target), _) => target,
            (None, MoveRequest::By(steps)) => base.saturating_add(steps),
            (None, MoveRequest::To(target)) => target,
        };
        let plan_bounds = if self.logical_start.is_some() {
            bounds
        } else {
            None
        };
        let plan = backlash::plan(
            settings.backlash_model,
            settings.backlash_owner,
            backlash_state,
            base,
            goal,
            plan_bounds,
        );
        let mut segments = Vec::new();
        let mut at = plan.raw_start;
        for leg in &plan.legs {
            let chunks = chunk_targets(at, leg.raw_target, inner.caps.max_increment);
            let n = chunks.len();
            for (i, chunk) in chunks.into_iter().enumerate() {
                let command = if self.absolute() {
                    SegmentCommand::Absolute(chunk)
                } else {
                    SegmentCommand::Relative(chunk - at)
                };
                segments.push(Segment {
                    command,
                    pause_after: leg.purpose == LegPurpose::Overshoot && i + 1 == n,
                });
                at = chunk;
            }
        }
        self.segments = segments;
        self.plan = Some(plan);

        // F-d.
        if let Some(change) = self.tempcomp.before_move(
            drv,
            inner.caps.temp_comp_available,
            state.temp_comp.known().copied(),
        )? {
            self.event(EventKind::TempcompChanged {
                from: change.from,
                to: change.to,
                reason: Some(change.reason.into()),
            });
        }

        // A relative position is unclean from here until the move resolves.
        if !self.absolute() && self.logical_start.is_some() {
            inner.persist_relative(self.logical_start, CloseState::Moving);
            self.marked_moving = true;
        }

        self.started_at = Some(clock.now_utc());
        if self.segments.is_empty() {
            // Already there: verify from the next poll without a settle.
            self.stopped_at = Some(clock.now_utc());
            self.phase = Phase::Settling {
                until: clock.now_mono_ns(),
            };
        } else {
            self.send_next(drv, clock)?;
        }
        self.event(EventKind::FocuserMoveStarted {
            move_id: self.move_id,
            from: self.logical_start,
        });
        Ok(())
    }

    fn send_next(&mut self, drv: &mut dyn FocuserDriver, clock: &dyn Clock) -> DeviceResult<()> {
        let segment = self.segments[self.next_segment];
        let result = match segment.command {
            SegmentCommand::Absolute(target) => drv.move_absolute(target),
            SegmentCommand::Relative(steps) => drv.move_relative(steps),
        };
        if let Err(err) = result {
            // A transport failure may have left a chunk running.
            self.in_flight = !matches!(
                initiator_error(err.clone()).class,
                ErrorClass::InvalidValue
                    | ErrorClass::InvalidOperation
                    | ErrorClass::Unsupported
                    | ErrorClass::NotConnected
            );
            return Err(initiator_error(err));
        }
        self.in_flight = true;
        let now = clock.now_mono_ns();
        let last_raw = drv
            .state()
            .ok()
            .and_then(|s| s.position.fresh_value().copied());
        self.phase = Phase::Moving {
            sent_mono: now,
            last_raw,
            last_change_mono: now,
        };
        Ok(())
    }

    fn set_settling(&self, settling: bool) {
        self.inner.lock().settling = settling;
    }

    fn step_moving(
        &mut self,
        drv: &mut dyn FocuserDriver,
        clock: &dyn Clock,
        sent_mono: u64,
        mut last_raw: Option<i32>,
        mut last_change_mono: u64,
    ) -> OpProgress {
        let (move_timeout, stuck_timeout, settle) = {
            let s = &self.inner.settings;
            (s.move_timeout, s.stuck_timeout, s.settle)
        };
        let now = clock.now_mono_ns();
        let state = drv.state().unwrap_or_default();

        if Duration::from_nanos(now.saturating_sub(sent_mono)) >= move_timeout {
            return self.halt_and_finish(drv, OpOutcome::TimedOut);
        }

        let moving = if sampled_since(&state.is_moving, sent_mono) {
            state.is_moving.value
        } else {
            None
        };

        // Stuck detection: Position frozen while IsMoving says true.
        if self.absolute() {
            if let Some(raw) = state.position.fresh_value().copied() {
                if sampled_since(&state.position, sent_mono) && Some(raw) != last_raw {
                    last_raw = Some(raw);
                    last_change_mono = state.position.mono_ns.unwrap_or(now);
                }
            }
            if moving == Some(true)
                && Duration::from_nanos(now.saturating_sub(last_change_mono)) >= stuck_timeout
            {
                let err = DeviceError::driver(format!(
                    "stuck: the focuser position stayed at {} for {} s while IsMoving was true",
                    last_raw.map_or_else(|| "unknown".to_string(), |r| r.to_string()),
                    stuck_timeout.as_secs()
                ));
                return self.halt_and_finish(drv, OpOutcome::Failed(err));
            }
        }
        self.phase = Phase::Moving {
            sent_mono,
            last_raw,
            last_change_mono,
        };

        if moving != Some(false) {
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL_INTERVAL,
            };
        }

        // This chunk completed.
        self.in_flight = false;
        let segment = self.segments[self.next_segment];
        if let SegmentCommand::Relative(steps) = segment.command {
            self.executed_steps += i64::from(steps);
        }
        self.next_segment += 1;
        if self.next_segment < self.segments.len() {
            if segment.pause_after {
                let until = now_after(clock, settle);
                self.phase = Phase::Pausing { until };
                self.set_settling(true);
                return OpProgress::Settling {
                    until_mono_ns: until,
                };
            }
            return match self.send_next(drv, clock) {
                Ok(()) => OpProgress::Continue {
                    next_poll: ACTIVE_POLL_INTERVAL,
                },
                Err(err) => self.finish_progress(drv, outcome_for(&err)),
            };
        }
        self.stopped_at = Some(clock.now_utc());
        let until = now_after(clock, settle);
        self.phase = Phase::Settling { until };
        self.set_settling(true);
        OpProgress::Settling {
            until_mono_ns: until,
        }
    }

    fn verify(&mut self, drv: &mut dyn FocuserDriver, clock: &dyn Clock, until: u64) -> OpProgress {
        let state = drv.state().unwrap_or_default();
        // The verification read must come from a poll taken after the settle.
        let fresh = sampled_since(&state.is_moving, until)
            && (!self.absolute() || sampled_since(&state.position, until));
        if !fresh {
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL_INTERVAL,
            };
        }
        if state.is_moving.value == Some(true) {
            // Something moved it again after the settle (an external
            // controller): verify once it stops. The operation deadline
            // still bounds the wait.
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL_INTERVAL,
            };
        }
        if self.absolute() {
            let target = self
                .plan
                .as_ref()
                .map(|p| p.raw_target)
                .or(self.raw_start)
                .unwrap_or_default();
            let Some(raw) = state.position.fresh_value().copied() else {
                return self.finish_progress(
                    drv,
                    OpOutcome::Unknown(DeviceError::outcome_unknown(
                        "the focuser position could not be read after the move",
                    )),
                );
            };
            let error = raw - target;
            if error.abs() > self.inner.settings.arrival_tolerance_steps {
                if self.reissues < self.inner.settings.max_reissue {
                    self.reissues += 1;
                    let chunks = chunk_targets(raw, target, self.inner.caps.max_increment);
                    let first = self.segments.len();
                    self.segments
                        .extend(chunks.into_iter().map(|chunk| Segment {
                            command: SegmentCommand::Absolute(chunk),
                            pause_after: false,
                        }));
                    self.next_segment = first;
                    self.set_settling(false);
                    self.event(EventKind::FocuserMoveStarted {
                        move_id: self.move_id,
                        from: Some(raw - self.plan.as_ref().map_or(0, |p| p.state_after.offset)),
                    });
                    return match self.send_next(drv, clock) {
                        Ok(()) => OpProgress::Continue {
                            next_poll: ACTIVE_POLL_INTERVAL,
                        },
                        Err(err) => self.finish_progress(drv, outcome_for(&err)),
                    };
                }
                let err = DeviceError::driver(format!(
                    "arrived {error} steps off target {target} after {} re-issue(s)",
                    self.reissues
                ));
                return self.finish_progress(drv, OpOutcome::Failed(err));
            }
        }
        self.settled_at = Some(clock.now_utc());
        self.finish_progress(drv, OpOutcome::Succeeded)
    }

    fn halt_and_finish(&mut self, drv: &mut dyn FocuserDriver, outcome: OpOutcome) -> OpProgress {
        if let Err(err) = drv.halt() {
            tracing::warn!(error = %err, "focuser halt failed");
        }
        let position = self.inner.lock().logical;
        self.event(EventKind::FocuserHalted { position });
        self.finish_progress(drv, outcome)
    }

    fn finish_progress(&mut self, drv: &mut dyn FocuserDriver, outcome: OpOutcome) -> OpProgress {
        self.finish(drv, outcome.clone());
        OpProgress::Done(self.outcome.clone().unwrap_or(outcome))
    }

    /// Resolve the move exactly once: TempComp restore, logical position,
    /// profile, events and the `focuser_moves` row.
    fn finish(&mut self, drv: &mut dyn FocuserDriver, outcome: OpOutcome) {
        if self.outcome.is_some() {
            return;
        }
        let inner = Arc::clone(&self.inner);
        let now_utc = inner.clock.now_utc();
        match self.tempcomp.restore(drv) {
            Ok(Some(change)) => self.event(EventKind::TempcompChanged {
                from: change.from,
                to: change.to,
                reason: Some(change.reason.into()),
            }),
            Ok(None) => {}
            Err(err) => tracing::warn!(error = %err, "focuser TempComp restore failed"),
        }
        let state: FocuserState = drv.state().unwrap_or_default();
        let raw_end = if self.absolute() {
            state.position.known().copied()
        } else {
            None
        };
        let success = outcome.is_success();
        let plan = self.plan.clone();
        let mut origin_lost_from = None;
        let (logical_end, persist_relative, persist_backlash) = {
            let mut s = inner.lock();
            s.op_active = false;
            s.settling = false;
            if self.absolute() {
                if success {
                    if let Some(plan) = &plan {
                        s.backlash = plan.state_after;
                    }
                } else {
                    // Partial travel leaves the slop state unknown.
                    s.backlash.last_direction = None;
                }
                if let Some(raw) = raw_end {
                    if state.position.state == ObsState::Fresh {
                        s.logical = Some(raw - s.backlash.offset);
                    }
                }
                // After a halt the motor may coast past the last reading, so
                // only a verified arrival seeds external-change detection.
                s.last_idle_raw = if success { raw_end } else { None };
                (s.logical, None, success.then_some((s.backlash, raw_end)))
            } else {
                let origin_lost = self.started_at.is_some()
                    && (self.in_flight || matches!(outcome, OpOutcome::Unknown(_)));
                if origin_lost {
                    if s.source.is_reliable() || s.logical.is_some() {
                        origin_lost_from = Some(s.logical);
                    }
                    s.logical = None;
                    s.source = PositionSource::EmulatedOriginUnknown;
                    s.backlash = Default::default();
                    (
                        None,
                        self.marked_moving.then_some((None, CloseState::Unknown)),
                        None,
                    )
                } else {
                    let offset_before = s.backlash.offset;
                    let offset_after = if success {
                        plan.as_ref()
                            .map_or(offset_before, |p| p.state_after.offset)
                    } else {
                        offset_before
                    };
                    if success {
                        if let Some(plan) = &plan {
                            s.backlash = plan.state_after;
                        }
                    } else {
                        s.backlash.last_direction = None;
                    }
                    if let Some(start) = self.logical_start.filter(|_| s.source.is_reliable()) {
                        let end = i64::from(start) + i64::from(offset_before) + self.executed_steps
                            - i64::from(offset_after);
                        s.logical = i32::try_from(end).ok();
                    }
                    let persist = (self.marked_moving || success) && s.source.is_reliable();
                    (
                        s.logical,
                        persist.then_some((s.logical, CloseState::Clean)),
                        None,
                    )
                }
            }
        };
        if let Some(from) = origin_lost_from {
            self.event(EventKind::FocuserPositionReset {
                from,
                to: None,
                cause: "relative_move_interrupted".into(),
            });
        }
        if let Some((logical, close)) = persist_relative {
            inner.persist_relative(logical, close);
        }
        if let Some((backlash, raw)) = persist_backlash {
            inner.persist_backlash(backlash, raw);
        }

        self.event(EventKind::FocuserMoveStopped {
            move_id: self.move_id,
            position: logical_end,
            outcome: outcome.clone(),
        });
        if success {
            self.event(EventKind::FocuserSettled {
                move_id: Some(self.move_id),
                position: logical_end,
                raw_position: raw_end,
            });
        }
        if self.stopped_at.is_none() && self.started_at.is_some() {
            self.stopped_at = Some(now_utc);
        }
        let record = FocuserMoveRecord {
            id: self.move_id,
            command_id: Some(self.command_id),
            connection_id: inner.connection_id,
            mode: self.request.mode(),
            commanded_target: self.commanded_target(),
            logical_start: self.logical_start,
            logical_end,
            raw_start: self.raw_start,
            raw_end,
            position_source: self.position_source,
            direction: plan.as_ref().and_then(|p| p.direction),
            reversal: plan.as_ref().and_then(|p| p.reversal),
            backlash_model: plan
                .as_ref()
                .map_or_else(|| inner.settings.effective_backlash(), |p| p.model),
            backlash_owner: inner.settings.backlash_owner,
            tempcomp_before: self.tempcomp.before(),
            tempcomp_after: self.tempcomp.after(state.temp_comp.known().copied()),
            temperature_start_c: self.temperature_start,
            temperature_end_c: state.temperature_c.known().copied(),
            commanded_at: self.commanded_at,
            started_at: self.started_at,
            stopped_at: self.stopped_at,
            settled_at: self.settled_at,
            outcome: outcome.clone(),
            source: self.source.clone(),
            run_id: self.run_id.clone(),
        };
        inner
            .sink
            .record(GroundTruthRecord::FocuserMove(Box::new(record)));
        self.outcome = Some(outcome);
        self.phase = Phase::Finished;
    }
}

impl Operation<dyn FocuserDriver> for FocuserMoveOp {
    fn kind(&self) -> OpKind {
        OpKind::FocuserMove
    }

    /// Exact once `start` has planned the chunks; before that a worst-case
    /// estimate. The per-chunk `move_timeout_ms` and the stuck timer are the
    /// operation's own, tighter guards.
    fn deadline(&self) -> Duration {
        let settings = &self.inner.settings;
        let legs = self.planned_segments().saturating_add(settings.max_reissue);
        settings
            .move_timeout
            .saturating_mul(legs)
            .saturating_add(settings.settle.saturating_mul(3))
            .saturating_add(Duration::from_secs(5))
    }

    fn start(
        &mut self,
        drv: &mut (dyn FocuserDriver + 'static),
        clock: &dyn Clock,
    ) -> DeviceResult<()> {
        self.started_mono = Some(clock.now_mono_ns());
        let logical = {
            let mut s = self.inner.lock();
            s.op_active = true;
            s.logical.filter(|_| s.source.is_reliable())
        };
        // `target` is the logical target; a relative move on an unknown
        // origin carries only its step count.
        let (target, steps) = match self.request {
            MoveRequest::To(target) => (target, None),
            MoveRequest::By(steps) => (
                logical.map_or(steps, |at| at.saturating_add(steps)),
                Some(steps),
            ),
        };
        self.event(EventKind::FocuserMoveCommanded {
            move_id: self.move_id,
            mode: self.request.mode(),
            target,
            steps,
        });
        match self.prepare(drv, clock) {
            Ok(()) => Ok(()),
            Err(err) => {
                if self.in_flight {
                    if let Err(halt) = drv.halt() {
                        tracing::warn!(error = %halt, "focuser halt after a failed start failed");
                    }
                }
                self.finish(drv, outcome_for(&err));
                Err(err)
            }
        }
    }

    fn step(&mut self, drv: &mut (dyn FocuserDriver + 'static), clock: &dyn Clock) -> OpProgress {
        match self.phase {
            Phase::Finished => {
                OpProgress::Done(self.outcome.clone().unwrap_or_else(|| {
                    OpOutcome::Unknown(DeviceError::outcome_unknown("finished"))
                }))
            }
            Phase::NotStarted => OpProgress::Continue {
                next_poll: ACTIVE_POLL_INTERVAL,
            },
            Phase::Moving {
                sent_mono,
                last_raw,
                last_change_mono,
            } => self.step_moving(drv, clock, sent_mono, last_raw, last_change_mono),
            Phase::Pausing { until } => {
                if clock.now_mono_ns() < until {
                    return OpProgress::Settling {
                        until_mono_ns: until,
                    };
                }
                self.set_settling(false);
                match self.send_next(drv, clock) {
                    Ok(()) => OpProgress::Continue {
                        next_poll: ACTIVE_POLL_INTERVAL,
                    },
                    Err(err) => self.finish_progress(drv, outcome_for(&err)),
                }
            }
            Phase::Settling { until } => {
                if clock.now_mono_ns() < until {
                    return OpProgress::Settling {
                        until_mono_ns: until,
                    };
                }
                self.verify(drv, clock, until)
            }
        }
    }

    /// Halts the focuser (F-c). The outcome is `TimedOut` once the deadline
    /// has passed, otherwise `Aborted` (the worker knows the aborting id).
    fn cancel(&mut self, drv: &mut (dyn FocuserDriver + 'static)) -> DeviceResult<()> {
        if self.outcome.is_some() {
            return Ok(());
        }
        let result = drv.halt();
        let position = self.inner.lock().logical;
        self.event(EventKind::FocuserHalted { position });
        let expired = self.started_mono.is_some_and(|start| {
            self.inner.clock.now_mono_ns() >= start.saturating_add(duration_ns(self.deadline()))
        });
        let outcome = if expired {
            OpOutcome::TimedOut
        } else {
            OpOutcome::Aborted { by: None }
        };
        self.finish(drv, outcome);
        result
    }
}

// ---------------------------------------------------------------------------
// Halt, TempComp, sync
// ---------------------------------------------------------------------------

/// `FocuserCmd::Halt` when no move is active (an active move is cancelled
/// instead). Completes when a later poll reads `IsMoving` false.
pub struct FocuserHaltOp {
    inner: Arc<SessionInner>,
    command_id: CommandId,
    sent_mono: u64,
}

impl FocuserHaltOp {
    pub(super) fn new(inner: Arc<SessionInner>, command_id: CommandId) -> Self {
        Self {
            inner,
            command_id,
            sent_mono: 0,
        }
    }
}

impl Operation<dyn FocuserDriver> for FocuserHaltOp {
    fn kind(&self) -> OpKind {
        OpKind::FocuserHalt
    }

    fn deadline(&self) -> Duration {
        Duration::from_secs(10)
    }

    fn start(
        &mut self,
        drv: &mut (dyn FocuserDriver + 'static),
        clock: &dyn Clock,
    ) -> DeviceResult<()> {
        self.sent_mono = clock.now_mono_ns();
        drv.halt()?;
        let position = self.inner.lock().logical;
        self.inner
            .event(EventKind::FocuserHalted { position }, Some(self.command_id));
        Ok(())
    }

    fn step(&mut self, drv: &mut (dyn FocuserDriver + 'static), _clock: &dyn Clock) -> OpProgress {
        let state = drv.state().unwrap_or_default();
        if sampled_since(&state.is_moving, self.sent_mono) && state.is_moving.value == Some(false) {
            OpProgress::Done(OpOutcome::Succeeded)
        } else {
            OpProgress::Continue {
                next_poll: ACTIVE_POLL_INTERVAL,
            }
        }
    }

    fn cancel(&mut self, drv: &mut (dyn FocuserDriver + 'static)) -> DeviceResult<()> {
        drv.halt()
    }
}

/// `FocuserCmd::SetTempComp`. Verified from the next TempComp read.
pub struct FocuserTempCompOp {
    inner: Arc<SessionInner>,
    command_id: CommandId,
    on: bool,
    sent_mono: u64,
}

impl FocuserTempCompOp {
    pub(super) fn new(inner: Arc<SessionInner>, command_id: CommandId, on: bool) -> Self {
        Self {
            inner,
            command_id,
            on,
            sent_mono: 0,
        }
    }
}

impl Operation<dyn FocuserDriver> for FocuserTempCompOp {
    fn kind(&self) -> OpKind {
        OpKind::FocuserTempComp
    }

    fn deadline(&self) -> Duration {
        Duration::from_secs(15)
    }

    fn start(
        &mut self,
        drv: &mut (dyn FocuserDriver + 'static),
        clock: &dyn Clock,
    ) -> DeviceResult<()> {
        let from = drv.state().ok().and_then(|s| s.temp_comp.known().copied());
        drv.set_temp_comp(self.on)?;
        self.sent_mono = clock.now_mono_ns();
        self.inner.event(
            EventKind::TempcompChanged {
                from,
                to: self.on,
                reason: Some("command".into()),
            },
            Some(self.command_id),
        );
        Ok(())
    }

    fn step(&mut self, drv: &mut (dyn FocuserDriver + 'static), _clock: &dyn Clock) -> OpProgress {
        let state = drv.state().unwrap_or_default();
        if state.temp_comp.state == ObsState::Unsupported {
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        if !sampled_since(&state.temp_comp, self.sent_mono) {
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL_INTERVAL,
            };
        }
        match state.temp_comp.value {
            Some(value) if value == self.on => OpProgress::Done(OpOutcome::Succeeded),
            Some(value) => OpProgress::Done(OpOutcome::Failed(DeviceError::driver(format!(
                "TempComp reads {value} after writing {}",
                self.on
            )))),
            None => OpProgress::Continue {
                next_poll: ACTIVE_POLL_INTERVAL,
            },
        }
    }

    fn cancel(&mut self, _drv: &mut (dyn FocuserDriver + 'static)) -> DeviceResult<()> {
        Ok(())
    }
}

/// `FocuserCmd::Sync` on a driver with position sync (INDI `FOCUS_SYNC`,
/// the simulator). Emits `focuser_position_reset{cause: "sync"}` so FDF and
/// FOP re-baseline (F-h).
pub struct FocuserSyncOp {
    inner: Arc<SessionInner>,
    command_id: CommandId,
    position: i32,
    sent_mono: u64,
}

impl FocuserSyncOp {
    pub(super) fn new(inner: Arc<SessionInner>, command_id: CommandId, position: i32) -> Self {
        Self {
            inner,
            command_id,
            position,
            sent_mono: 0,
        }
    }
}

impl Operation<dyn FocuserDriver> for FocuserSyncOp {
    fn kind(&self) -> OpKind {
        OpKind::FocuserSync
    }

    fn deadline(&self) -> Duration {
        Duration::from_secs(15)
    }

    fn start(
        &mut self,
        drv: &mut (dyn FocuserDriver + 'static),
        clock: &dyn Clock,
    ) -> DeviceResult<()> {
        drv.sync_position(self.position)?;
        self.sent_mono = clock.now_mono_ns();
        let absolute = self.inner.caps.absolute;
        let from = {
            let mut s = self.inner.lock();
            let from = s.logical;
            s.logical = Some(self.position);
            s.backlash = Default::default();
            s.last_idle_raw = None;
            if !absolute {
                s.source = PositionSource::Emulated;
            }
            from
        };
        self.inner.event(
            EventKind::FocuserPositionReset {
                from,
                to: Some(self.position),
                cause: "sync".into(),
            },
            Some(self.command_id),
        );
        if absolute {
            self.inner
                .persist_backlash(Default::default(), Some(self.position));
        } else {
            self.inner
                .persist_relative(Some(self.position), CloseState::Clean);
        }
        Ok(())
    }

    fn step(&mut self, drv: &mut (dyn FocuserDriver + 'static), _clock: &dyn Clock) -> OpProgress {
        if !self.inner.caps.absolute {
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        let state = drv.state().unwrap_or_default();
        if !sampled_since(&state.position, self.sent_mono) {
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL_INTERVAL,
            };
        }
        match state.position.value {
            Some(position) if position == self.position => OpProgress::Done(OpOutcome::Succeeded),
            Some(position) => OpProgress::Done(OpOutcome::Failed(DeviceError::driver(format!(
                "sync to {} not applied: the focuser reads {position}",
                self.position
            )))),
            None => OpProgress::Continue {
                next_poll: ACTIVE_POLL_INTERVAL,
            },
        }
    }

    fn cancel(&mut self, _drv: &mut (dyn FocuserDriver + 'static)) -> DeviceResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_only_when_max_increment_is_smaller_than_the_move() {
        assert_eq!(chunk_targets(0, 350, Some(100)), vec![100, 200, 300, 350]);
        assert_eq!(chunk_targets(350, 0, Some(100)), vec![250, 150, 50, 0]);
        assert_eq!(chunk_targets(0, 300, Some(100)), vec![100, 200, 300]);
        assert_eq!(chunk_targets(0, 100, Some(100)), vec![100]);
        assert_eq!(chunk_targets(0, 99, Some(100)), vec![99]);
        assert_eq!(chunk_targets(0, 5000, None), vec![5000]);
        assert_eq!(chunk_targets(0, 5000, Some(0)), vec![5000]);
        assert!(chunk_targets(7, 7, Some(1)).is_empty());
    }

    #[test]
    fn initiator_transport_failures_become_outcome_unknown() {
        let err = initiator_error(DeviceError::timeout("no reply"));
        assert_eq!(err.class, ErrorClass::OutcomeUnknown);
        assert!(matches!(outcome_for(&err), OpOutcome::Unknown(_)));
        let err = initiator_error(DeviceError::invalid_value("too far"));
        assert_eq!(err.class, ErrorClass::InvalidValue);
        assert!(matches!(outcome_for(&err), OpOutcome::Failed(_)));
    }
}
