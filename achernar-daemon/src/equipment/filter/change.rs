//! The filter-change handshake (spec §3.4 W-c, v6 §8.2.1, H8). Owner:
//! WP1A-6.
//!
//! [`FilterChanger::change`] runs behind the gate: the caller passes the
//! `Permit` the gate issued for `GatedOp::FilterHandshake` (domains: wheel,
//! focuser, guider). The steps, in the v6 §8.2.1 order:
//!
//! 1. Preconditions: an exposure boundary (the camera is not exposing), the
//!    minimum interval and the frequency limit (`check::evaluate`, the same
//!    rules the gate plug-in runs).
//! 2. Ownership: the permit must hold the wheel; the focuser and guider are
//!    used only when the permit holds them too.
//! 3. Read and verify the current slot. The offset the focuser holds is
//!    known only for a slot the wheel has verified; an unknown or unverified
//!    slot (after a wrong slot, an unknown outcome or a move nobody
//!    commanded) gives `prev_offset_unknown`, no offset move, and focus
//!    verification is required. The wheel move itself still goes ahead, even
//!    to the slot the wheel already reports, so the slot is verified again.
//!    A verified slot equal to the target is a no-op that publishes no
//!    `filter_*` event (H2 and the window builder reset on those).
//! 4. If PHD2 is guiding, the offset Δ ≠ 0 and `pause_guiding_on_offset` is
//!    set: pause guiding (`set_paused [true]`; NINA stops guiding, pausing is
//!    lighter). The pause is awaited to its outcome even across an abort, and
//!    a pause that was sent and not definitely refused is always followed by
//!    a resume; a resume that fails raises a resume block.
//! 5. Command the position (the worker polls at 4 Hz). An acceptance error
//!    that may come after the initiator reached the wheel (timeout,
//!    transport, driver, a worker that stopped) is `OutcomeUnknown` with a
//!    resume block, not a refusal.
//! 6. On timeout: `OutcomeUnknown` and a `filter_handshake_incomplete`
//!    resume block. Unlike NINA, a timeout is never silent.
//! 7. Verify Position == target; otherwise `Failed{wrong_slot}` plus a
//!    resume block.
//! 8. Settle for `settle_ms` (the profile value wins).
//! 9. Apply the focuser **relative** move Δ = offset[new] − offset[prev],
//!    through the focuser's own move path and its safety checks. Sequential
//!    by default; `parallel_offset_move` starts it with the wheel move.
//! 10. If an offset is uncertain (unmeasured or estimated), emit
//!     `focus_verification_required`.
//! 11. Resume guiding and wait until the guider has settled.
//! 12. Release ownership (the permit is dropped on every path).
//! 13. Write the `filter_switch_events` row.
//!
//! Sub-commands get their own command ids (so `equipment_overheads` keeps
//! one row per operation) and permits that hold no ownership: the parent
//! permit holds every domain for the whole handshake. Every focuser
//! sub-command is first run through [`ChildCommandCheck`], which the gate
//! implements with its registered safety checks, so an offset move obeys the
//! focuser's travel bounds exactly as a manual move does.
//!
//! Waits use [`HandshakeTimer`]: settle waits are injectable (D22), and a
//! watchdog bounds a worker that never resolves a ticket. The operation's
//! own deadline still comes from the worker.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use tokio::sync::{watch, Mutex as AsyncMutex};
use uuid::Uuid;

use crate::config::FilterWheelConfig;
use crate::equipment::commands::{
    Domain, FilterTarget, FilterWheelCmd, FocuserCmd, GateError, GatedCommand, GatedOp, GuiderCmd,
};
use crate::equipment::events::{EquipmentEvent, EventKind, FilterSwitchRecord, GroundTruthRecord};
use crate::equipment::handle::{FilterWheelHandle, FocuserHandle, GuiderHandle, OpTicket, Permit};
use crate::equipment::ops::OpKind;
use crate::equipment::snapshot::{
    FilterWheelSnapshot, GuiderAppState, ResumeBlock, ResumeBlockKind,
};
use crate::equipment::traits::{CameraActivity, Clock, RecorderSink};
use crate::equipment::types::{
    AutonomyMode, CommandId, CommandSource, DeviceError, DeviceKind, ErrorClass, HealthState,
    Observed, OpOutcome, Priority, Source,
};

use super::check::{self, FilterSafetyCheck, PlanLock, SwitchContext, SwitchHistory, SwitchLimits};
use super::motion::{is_wrong_slot, ARRIVAL_TIMEOUT};
use super::reconcile::{self, default_slot_name, ReconciledWheel, SeedImportResult};
use super::{
    read_wheel, write_wheel, FilterEventSink, FilterSettings, SharedWheel, WATCHDOG_GRACE,
};

// ---------------------------------------------------------------------------
// Injected services
// ---------------------------------------------------------------------------

pub type TimerFuture = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

/// Waits of the handshake (D22).
pub trait HandshakeTimer: Send + Sync {
    /// A settle wait. Tests advance a `ManualClock` instead of sleeping.
    fn sleep(&self, duration: Duration) -> TimerFuture;
    /// Last-resort bound on a worker that never resolves a ticket.
    fn watchdog(&self, duration: Duration) -> TimerFuture;
}

/// Production timer: tokio sleeps on the system clock.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokioTimer;

impl HandshakeTimer for TokioTimer {
    fn sleep(&self, duration: Duration) -> TimerFuture {
        Box::pin(tokio::time::sleep(duration))
    }

    fn watchdog(&self, duration: Duration) -> TimerFuture {
        Box::pin(tokio::time::sleep(duration))
    }
}

/// Runs the gate's safety checks for a sub-command of a compound operation
/// (ownership is already held by the parent permit). The gate (WP1A-2)
/// implements it over its registered `SafetyCheck`s and current snapshot.
pub trait ChildCommandCheck: Send + Sync {
    fn check(&self, cmd: &GatedCommand) -> Result<(), GateError>;
}

/// Accepts every sub-command (tests, and an inert hub).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoChildChecks;

impl ChildCommandCheck for NoChildChecks {
    fn check(&self, _cmd: &GatedCommand) -> Result<(), GateError> {
        Ok(())
    }
}

/// What the handshake publishes to and reads time from.
#[derive(Clone)]
pub struct HandshakeServices {
    pub clock: Arc<dyn Clock>,
    pub timer: Arc<dyn HandshakeTimer>,
    pub events: Arc<dyn FilterEventSink>,
    pub recorder: Arc<dyn RecorderSink>,
    pub child_checks: Arc<dyn ChildCommandCheck>,
}

/// The other devices of the change, taken from their slots per call (they
/// connect and reconnect independently of the wheel).
#[derive(Clone, Default)]
pub struct Peers {
    pub focuser: Option<FocuserHandle>,
    pub guider: Option<GuiderHandle>,
    pub camera: Option<Arc<dyn CameraActivity>>,
}

// ---------------------------------------------------------------------------
// Requests and reports
// ---------------------------------------------------------------------------

/// FW-2 (or another policy) attribution recorded with the switch.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PolicyAttribution {
    pub policy_version: Option<String>,
    pub arm: Option<String>,
    /// Only when the choice was randomised.
    pub propensity: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HandshakeRequest {
    pub target: FilterTarget,
    /// `filter_switch_events.trigger`; the command source label when unset.
    pub trigger: Option<String>,
    pub policy: Option<PolicyAttribution>,
    /// Sky background before the change (FW-2 reward context).
    pub background_before: Option<f64>,
    pub session_id: Option<String>,
}

impl HandshakeRequest {
    pub fn new(target: FilterTarget) -> Self {
        Self {
            target,
            trigger: None,
            policy: None,
            background_before: None,
            session_id: None,
        }
    }
}

/// The result of one handshake.
#[derive(Debug, Clone, PartialEq)]
pub struct HandshakeReport {
    pub command_id: CommandId,
    pub outcome: OpOutcome,
    /// Succeeded and no resume block.
    pub handshake_ok: bool,
    /// The verified current slot already was the target; nothing moved and
    /// no `filter_*` event was published.
    pub noop: bool,
    /// The slot the wheel reported before the change (it may be unverified;
    /// see `from_verified`).
    pub from_slot: Option<u32>,
    /// `from_slot` was verified, so the focuser holds its offset.
    pub from_verified: bool,
    pub to_slot: Option<u32>,
    /// The slot the wheel reported at the end.
    pub arrived_slot: Option<u32>,
    pub offset_from: Option<i32>,
    pub offset_to: Option<i32>,
    /// The relative move applied; `Some(0)` when both offsets were known
    /// and equal, `None` when nothing could be applied.
    pub offset_applied_steps: Option<i32>,
    /// Command id of the focuser offset move (`focuser_moves.command_id`).
    pub focuser_command_id: Option<CommandId>,
    /// PHD2 was paused, or a pause was sent and may have taken effect
    /// (`guider_pause_unconfirmed`). Either way a resume is sent on every
    /// path.
    pub guiding_paused: bool,
    /// Why focus must be verified (W-c step 10); empty when it need not be.
    pub focus_verification: Vec<String>,
    pub resume_block: Option<ResumeBlock>,
    pub record: Option<FilterSwitchRecord>,
    /// Audited notes, e.g. `exposure_state_unknown`.
    pub notes: Vec<String>,
    pub duration_ms: u64,
}

impl HandshakeReport {
    fn new(command_id: CommandId) -> Self {
        Self {
            command_id,
            outcome: OpOutcome::Succeeded,
            handshake_ok: false,
            noop: false,
            from_slot: None,
            from_verified: false,
            to_slot: None,
            arrived_slot: None,
            offset_from: None,
            offset_to: None,
            offset_applied_steps: None,
            focuser_command_id: None,
            guiding_paused: false,
            focus_verification: Vec::new(),
            resume_block: None,
            record: None,
            notes: Vec::new(),
            duration_ms: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// What the handshake needs from its permit. The permit itself stays owned
/// by `change` (it is `Send` but not `Sync`) and is dropped at step 12.
#[derive(Debug, Clone)]
struct PermitInfo {
    command_id: CommandId,
    priority: Priority,
    source: CommandSource,
    mode: AutonomyMode,
    domains: Vec<Domain>,
}

impl PermitInfo {
    fn of(permit: &Permit) -> Self {
        Self {
            command_id: permit.command_id(),
            priority: permit.priority(),
            source: permit.source().clone(),
            mode: permit.mode(),
            domains: permit.domains().to_vec(),
        }
    }

    fn owns(&self, kind: DeviceKind) -> bool {
        self.domains.contains(&Domain::Device(kind))
    }

    /// A sub-command id and a permit that holds no ownership.
    fn child(&self) -> (CommandId, Permit) {
        let id = CommandId::new();
        (
            id,
            Permit::unmanaged(id, self.priority, self.source.clone()),
        )
    }
}

/// Why the steps stopped early.
struct Stop {
    outcome: OpOutcome,
    /// A resume block is raised with this reason.
    block: Option<String>,
}

impl Stop {
    /// Refused before anything moved: no resume block.
    fn refused(err: DeviceError) -> Self {
        Self {
            outcome: OpOutcome::Failed(err),
            block: None,
        }
    }

    fn blocked(outcome: OpOutcome, reason: impl Into<String>) -> Self {
        Self {
            outcome,
            block: Some(reason.into()),
        }
    }
}

enum Waited {
    Done(OpOutcome),
    Watchdog,
    Cancelled(CommandId),
}

struct RunState {
    cmd: CommandId,
    source: CommandSource,
    started_mono_ns: u64,
    requested_at: DateTime<Utc>,
    report: HandshakeReport,
    /// `filter_change_requested` was emitted, so `filter_change_ended` is due.
    requested: bool,
    /// `operation_commanded` was emitted, so `operation_ended` is due.
    commanded: bool,
    /// A `filter_switch_events` row is due.
    record: bool,
    wheel_arrived: bool,
    verification_emitted: bool,
    guider_resumed: bool,
    wheel_done_at: Option<DateTime<Utc>>,
    verified_at: Option<DateTime<Utc>>,
    settle_ms_actual: Option<u64>,
    block_reason: Option<String>,
}

/// The longest a guider pause or resume may take (G-e: the guider waits up
/// to 60 s for `Paused`, or for the state to leave it). The watchdog on a
/// guider command never fires before this.
const GUIDER_PAUSE_WAIT: Duration = Duration::from_secs(60);

/// The slot the wheel reports: a fresh, stationary read. It may be
/// unverified.
fn current_slot(snap: &FilterWheelSnapshot, health: &HealthState) -> Option<u32> {
    if !health.state.is_connected() || snap.moving {
        return None;
    }
    snap.position.fresh_value().copied()
}

/// The current slot only when the wheel has verified it (W-c step 3): the
/// read equals the slot that the last successful move, or the first
/// stationary read after connect, established. After a wrong slot, an
/// unknown outcome or a move nobody commanded the wheel stays unverified,
/// and the focuser cannot be assumed to hold that slot's offset.
fn verified_current_slot(snap: &FilterWheelSnapshot, health: &HealthState) -> Option<u32> {
    current_slot(snap, health).filter(|slot| snap.verified && snap.verified_slot == Some(*slot))
}

/// True when a command that `DeviceHandle::send` refused with `err` may still
/// have reached the device (§2.8 `outcome_unknown`, C-6). Only errors raised
/// before anything is sent (validation, ownership, safety, a retired handle)
/// and a device's definite refusal are excluded. A timeout, transport or
/// driver error on the initiator, or a worker that stopped while starting
/// the operation (`Disconnected`), leaves the outcome unknown.
fn may_have_reached_device(err: &DeviceError) -> bool {
    !matches!(
        err.class,
        ErrorClass::NotConnected
            | ErrorClass::Unsupported
            | ErrorClass::InvalidValue
            | ErrorClass::InvalidWhileParked
            | ErrorClass::InvalidOperation
            | ErrorClass::SafetyRejected
            | ErrorClass::OwnershipConflict
    )
}

/// True for an outcome that says the command did not take effect: a failure
/// the device answered with, not a lost answer.
fn definitely_not_applied(outcome: &OpOutcome) -> bool {
    matches!(
        outcome,
        OpOutcome::Failed(err) if !matches!(
            err.class,
            ErrorClass::Timeout
                | ErrorClass::Transport
                | ErrorClass::Disconnected
                | ErrorClass::OutcomeUnknown
                | ErrorClass::Cancelled
        )
    )
}

/// What a guider command did (steps 4 and 11).
enum GuiderReply {
    Done,
    /// It certainly did not take effect.
    Refused(String),
    /// It may have taken effect: the answer was lost or never came.
    Unconfirmed(String),
}

fn is_guiding(guider: &GuiderHandle) -> bool {
    guider.health().state.is_connected()
        && guider.snapshot().app_state.known() == Some(&GuiderAppState::Guiding)
}

fn cancelled_now(rx: &watch::Receiver<Option<CommandId>>) -> Option<CommandId> {
    *rx.borrow()
}

async fn cancelled(rx: &mut watch::Receiver<Option<CommandId>>) -> CommandId {
    loop {
        let current = *rx.borrow_and_update();
        if let Some(by) = current {
            return by;
        }
        if rx.changed().await.is_err() {
            return std::future::pending().await;
        }
    }
}

async fn maybe_cancelled(rx: Option<&mut watch::Receiver<Option<CommandId>>>) -> CommandId {
    match rx {
        Some(rx) => cancelled(rx).await,
        None => std::future::pending().await,
    }
}

fn outcome_text(outcome: &OpOutcome) -> String {
    match outcome {
        OpOutcome::Succeeded => "succeeded".into(),
        OpOutcome::Failed(err) | OpOutcome::Unknown(err) => format!("{}: {err}", outcome.label()),
        OpOutcome::Aborted { .. } | OpOutcome::TimedOut => outcome.label().into(),
    }
}

/// The offsets of a change and why focus must be verified afterwards.
struct OffsetPlan {
    from: Option<i32>,
    to: Option<i32>,
    delta: Option<i32>,
    reasons: Vec<String>,
}

impl OffsetPlan {
    /// `from_slot` is the slot the wheel reports; its offset is used only
    /// when `from_verified` (W-c step 3).
    fn new(
        settings: &FilterSettings,
        wheel: &ReconciledWheel,
        from_slot: Option<u32>,
        from_verified: bool,
        to_slot: u32,
    ) -> Self {
        let reported = from_slot;
        let from_slot = from_slot.filter(|_| from_verified);
        let to_entry = wheel.slot(to_slot);
        let from_entry = from_slot.and_then(|slot| wheel.slot(slot));
        let to = to_entry.and_then(|s| s.slot.offset);
        let from = from_entry.and_then(|s| s.slot.offset);
        let mut reasons = Vec::new();
        if !settings.use_focus_offsets {
            return Self {
                from,
                to,
                delta: None,
                reasons,
            };
        }
        let delta = match (from_slot, from, to) {
            (None, _, _) => {
                reasons.push(match reported {
                    Some(slot) => {
                        format!("prev_offset_unknown: slot {} is not verified", slot + 1)
                    }
                    None => "prev_offset_unknown".to_string(),
                });
                None
            }
            (Some(slot), None, _) => {
                reasons.push(format!("offset_unknown: slot {} has no offset", slot + 1));
                None
            }
            (Some(_), Some(_), None) => {
                reasons.push(format!(
                    "offset_unknown: slot {} has no offset",
                    to_slot + 1
                ));
                None
            }
            (Some(_), Some(a), Some(b)) => match b.checked_sub(a) {
                Some(delta) => Some(delta),
                None => {
                    reasons.push("offset_overflow".to_string());
                    None
                }
            },
        };
        if delta.is_some() {
            let certain = [from_entry, to_entry]
                .iter()
                .all(|entry| entry.is_some_and(|e| e.offset_quality.is_certain()));
            if !certain {
                reasons.push("offset_uncertain: an offset is estimated, not measured".to_string());
            }
        }
        Self {
            from,
            to,
            delta,
            reasons,
        }
    }
}

// ---------------------------------------------------------------------------
// The changer
// ---------------------------------------------------------------------------

/// Runs filter changes for one connected wheel. One handshake at a time.
pub struct FilterChanger {
    wheel: FilterWheelHandle,
    model: SharedWheel,
    settings: FilterSettings,
    limits: SwitchLimits,
    services: HandshakeServices,
    history: Arc<SwitchHistory>,
    plan: Arc<PlanLock>,
    running: AsyncMutex<()>,
    cancel: watch::Sender<Option<CommandId>>,
}

impl FilterChanger {
    pub fn new(
        wheel: FilterWheelHandle,
        model: SharedWheel,
        settings: FilterSettings,
        services: HandshakeServices,
    ) -> Self {
        let (cancel, _) = watch::channel(None);
        Self {
            wheel,
            model,
            limits: SwitchLimits::from_settings(&settings),
            settings,
            services,
            history: SwitchHistory::new(),
            plan: PlanLock::new(),
            running: AsyncMutex::new(()),
            cancel,
        }
    }

    /// Share the switch history and plan lock with an existing check (for
    /// example across a wheel reconnect).
    pub fn with_shared(mut self, history: Arc<SwitchHistory>, plan: Arc<PlanLock>) -> Self {
        self.history = history;
        self.plan = plan;
        self
    }

    pub fn wheel(&self) -> &FilterWheelHandle {
        &self.wheel
    }

    pub fn model(&self) -> SharedWheel {
        Arc::clone(&self.model)
    }

    pub fn settings(&self) -> &FilterSettings {
        &self.settings
    }

    pub fn history(&self) -> Arc<SwitchHistory> {
        Arc::clone(&self.history)
    }

    pub fn plan(&self) -> Arc<PlanLock> {
        Arc::clone(&self.plan)
    }

    /// The gate plug-in for this wheel, sharing history and plan lock.
    pub fn safety_check(&self) -> FilterSafetyCheck {
        FilterSafetyCheck::new(self.limits, self.history(), self.plan())
    }

    /// `POST filter-wheel/abort`: cancel the running handshake. The wheel has
    /// no halt; its operation is cancelled and its position becomes unknown,
    /// which the handshake reports with a resume block.
    pub fn abort(&self) -> CommandId {
        let by = CommandId::new();
        self.cancel.send_replace(Some(by));
        let _ = self.wheel.abort();
        by
    }

    /// `POST filter-wheel/import-seed` (D17): copy the `[filter_wheel]` seeds
    /// into the wheel's profile, record the patch and re-reconcile.
    pub fn import_seed(
        &self,
        seeds: &FilterWheelConfig,
        permit: &Permit,
    ) -> Result<SeedImportResult, DeviceError> {
        if !PermitInfo::of(permit).owns(DeviceKind::FilterWheel) {
            return Err(DeviceError::new(
                ErrorClass::OwnershipConflict,
                "the import permit does not own the filter wheel",
            ));
        }
        let current = read_wheel(&self.model);
        let identity = self.wheel.identity();
        let result = reconcile::import_seed(
            seeds,
            &current,
            &identity.identity_key,
            self.services.clock.now_utc(),
        )?;
        self.services
            .recorder
            .record(GroundTruthRecord::Profile(result.update.clone()));
        let wheel = reconcile::reconcile(&current.device_slots, Some(&result.profile));
        if let Some(kind) = wheel.mismatch_event() {
            self.emit(kind, permit.command_id());
        }
        write_wheel(&self.model, wheel);
        Ok(result)
    }

    fn event(&self, kind: EventKind, cmd: CommandId) -> EquipmentEvent {
        EquipmentEvent::new(kind, self.services.clock.as_ref())
            .for_device(DeviceKind::FilterWheel, self.wheel.connection_id())
            .with_command(cmd)
    }

    fn emit(&self, kind: EventKind, cmd: CommandId) {
        self.services.events.publish(self.event(kind, cmd));
    }

    fn elapsed_ms(&self, since_mono_ns: u64) -> u64 {
        self.services
            .clock
            .now_mono_ns()
            .saturating_sub(since_mono_ns)
            / 1_000_000
    }

    fn mark_commanded(&self, st: &mut RunState) {
        if !st.commanded {
            st.commanded = true;
            self.emit(
                EventKind::OperationCommanded {
                    op: OpKind::FilterHandshake,
                    source: st.source.clone(),
                },
                st.cmd,
            );
        }
    }

    async fn wait(
        &self,
        ticket: OpTicket,
        budget: Duration,
        cancel: Option<&mut watch::Receiver<Option<CommandId>>>,
    ) -> Waited {
        let watchdog = self.services.timer.watchdog(budget);
        tokio::select! {
            biased;
            outcome = ticket.wait() => Waited::Done(outcome),
            by = maybe_cancelled(cancel) => Waited::Cancelled(by),
            () = watchdog => Waited::Watchdog,
        }
    }

    /// Run a guider command and wait for its outcome (step 4 pause, step 11
    /// resume). An abort does not cut the wait short: a pause that was sent
    /// must be followed by a resume, and a resume sent while the pause is
    /// still outstanding would be refused or overtaken by it. Nothing moves
    /// meanwhile. A watchdog bounds the acceptance and the outcome together.
    async fn guider_command(
        &self,
        guider: &GuiderHandle,
        cmd: GuiderCmd,
        info: &PermitInfo,
    ) -> GuiderReply {
        let (_, permit) = info.child();
        let budget = self.settings.guider_rpc_timeout().max(GUIDER_PAUSE_WAIT) + WATCHDOG_GRACE;
        let mut watchdog = self.services.timer.watchdog(budget);
        let sent = tokio::select! {
            biased;
            sent = guider.send(cmd, permit) => sent,
            () = &mut watchdog => {
                return GuiderReply::Unconfirmed(
                    "the guider worker did not accept the command".into(),
                );
            }
        };
        let ticket = match sent {
            Ok(ticket) => ticket,
            Err(err) if may_have_reached_device(&err) => {
                return GuiderReply::Unconfirmed(err.to_string())
            }
            Err(err) => return GuiderReply::Refused(err.to_string()),
        };
        tokio::select! {
            biased;
            outcome = ticket.wait() => match outcome {
                OpOutcome::Succeeded => GuiderReply::Done,
                other if definitely_not_applied(&other) => {
                    GuiderReply::Refused(outcome_text(&other))
                }
                other => GuiderReply::Unconfirmed(outcome_text(&other)),
            },
            () = &mut watchdog => GuiderReply::Unconfirmed("the guider did not answer".into()),
        }
    }

    /// Step 11: resume, then wait for the guider to report it has settled.
    async fn resume_guider(&self, guider: &GuiderHandle, info: &PermitInfo) -> Result<(), String> {
        match self.guider_command(guider, GuiderCmd::Resume, info).await {
            GuiderReply::Done => {}
            GuiderReply::Refused(why) => return Err(why),
            GuiderReply::Unconfirmed(why) => return Err(format!("unconfirmed: {why}")),
        }
        let mut rx = guider.watch_snapshot();
        let mut watchdog = self
            .services
            .timer
            .watchdog(self.settings.guider_settle_timeout());
        loop {
            let settling = rx.borrow_and_update().settling;
            if !settling {
                return Ok(());
            }
            tokio::select! {
                changed = rx.changed() => {
                    if changed.is_err() {
                        return Err("the guider went away while settling".into());
                    }
                }
                () = &mut watchdog => return Err("the guider did not settle in time".into()),
            }
        }
    }

    /// Start the focuser offset move after its safety checks.
    async fn start_offset_move(
        &self,
        focuser: &FocuserHandle,
        steps: i32,
        info: &PermitInfo,
    ) -> Result<(CommandId, OpTicket), DeviceError> {
        let (id, permit) = info.child();
        let mut gated = GatedCommand::new(
            GatedOp::Focuser(FocuserCmd::MoveBy(steps)),
            info.source.clone(),
            info.mode,
            info.priority,
        );
        gated.command_id = id;
        self.services
            .child_checks
            .check(&gated)
            .map_err(|err| err.to_device_error())?;
        let ticket = focuser.send(FocuserCmd::MoveBy(steps), permit).await?;
        Ok((id, ticket))
    }

    /// Run one handshake. Never panics on device errors; the outcome, the
    /// records and any resume block are in the report.
    pub async fn change(
        &self,
        peers: Peers,
        req: HandshakeRequest,
        permit: Permit,
    ) -> HandshakeReport {
        let info = PermitInfo::of(&permit);
        let clock = Arc::clone(&self.services.clock);
        let mut st = RunState {
            cmd: info.command_id,
            source: info.source.clone(),
            started_mono_ns: clock.now_mono_ns(),
            requested_at: clock.now_utc(),
            report: HandshakeReport::new(info.command_id),
            requested: false,
            commanded: false,
            record: false,
            wheel_arrived: false,
            verification_emitted: false,
            guider_resumed: false,
            wheel_done_at: None,
            verified_at: None,
            settle_ms_actual: None,
            block_reason: None,
        };
        let Ok(_running) = self.running.try_lock() else {
            st.report.outcome = OpOutcome::Failed(DeviceError::invalid_operation(
                "another filter change is already running on this wheel",
            ));
            return st.report;
        };
        self.cancel.send_replace(None);
        let mut cancel = self.cancel.subscribe();
        let model = read_wheel(&self.model);

        if let Err(stop) = self
            .run(&peers, &req, &info, &model, &mut st, &mut cancel)
            .await
        {
            st.report.outcome = stop.outcome;
            if let Some(reason) = stop.block {
                if st.block_reason.is_none() {
                    st.block_reason = Some(reason);
                }
            }
        }

        // Step 11 on every other path: never leave PHD2 paused. A resume that
        // fails or cannot be confirmed leaves guiding in doubt, so it blocks
        // automatic resumption until the operator clears it.
        if st.report.guiding_paused && !st.guider_resumed {
            if let Some(guider) = &peers.guider {
                st.guider_resumed = true;
                if let Err(why) = self.resume_guider(guider, &info).await {
                    st.report.notes.push(format!("guider_resume_failed: {why}"));
                    if st.block_reason.is_none() {
                        st.block_reason = Some(format!(
                            "guiding did not resume after the filter change: {why}"
                        ));
                    }
                }
            }
        }

        // Step 12: release ownership on every path.
        drop(permit);
        self.finish(st, &req, &model)
    }

    async fn run(
        &self,
        peers: &Peers,
        req: &HandshakeRequest,
        info: &PermitInfo,
        model: &ReconciledWheel,
        st: &mut RunState,
        cancel: &mut watch::Receiver<Option<CommandId>>,
    ) -> Result<(), Stop> {
        let cmd = st.cmd;
        let clock = Arc::clone(&self.services.clock);

        // Target.
        let to_slot = match model.resolve(&req.target) {
            Ok(slot) => slot,
            Err(err) => {
                self.mark_commanded(st);
                return Err(Stop::refused(err));
            }
        };
        st.report.to_slot = Some(to_slot);

        // Step 3: read and verify the current slot (first, so the request
        // names it).
        let wheel_snap = self.wheel.snapshot();
        let wheel_health = self.wheel.health();
        let from_slot = current_slot(&wheel_snap, &wheel_health);
        let verified_from = verified_current_slot(&wheel_snap, &wheel_health);
        st.report.from_slot = from_slot;
        st.report.from_verified = verified_from.is_some();
        if verified_from == Some(to_slot) {
            // Nothing moves, so nothing is published: every `filter_*` event
            // resets observation baselines and discards inference (H2,
            // §7.2.1), and a sequencer may assert the filter before every
            // frame. NINA does nothing here either.
            st.report.noop = true;
            st.report.arrived_slot = Some(to_slot);
            return Ok(());
        }
        st.requested = true;
        self.emit(
            EventKind::FilterChangeRequested {
                from_slot,
                to_slot,
                to_name: model.name_of(to_slot).map(str::to_string),
            },
            cmd,
        );
        self.mark_commanded(st);
        st.record = true;

        // Step 2: ownership.
        if !info.owns(DeviceKind::FilterWheel) {
            return Err(Stop::refused(DeviceError::new(
                ErrorClass::OwnershipConflict,
                "the handshake permit does not own the filter wheel",
            )));
        }

        // Step 1: exposure boundary, interval and frequency limits.
        let exposing = peers.camera.as_ref().map_or_else(
            || Observed::not_set(Source::Derived),
            |camera| camera.exposing(),
        );
        let ctx = SwitchContext {
            source: &st.source,
            priority: info.priority,
            exposing: &exposing,
            now_mono_ns: clock.now_mono_ns(),
        };
        match check::evaluate(&ctx, &self.limits, &self.history, &self.plan) {
            Ok(notes) => st.report.notes.extend(notes),
            Err(rejection) => {
                return Err(Stop::refused(DeviceError::new(
                    ErrorClass::SafetyRejected,
                    rejection.to_string(),
                )))
            }
        }

        // Offsets and focus-verification reasons (steps 9 and 10).
        let offsets = OffsetPlan::new(
            &self.settings,
            model,
            from_slot,
            st.report.from_verified,
            to_slot,
        );
        st.report.offset_from = offsets.from;
        st.report.offset_to = offsets.to;
        st.report.focus_verification.extend(offsets.reasons);
        if offsets.delta == Some(0) {
            st.report.offset_applied_steps = Some(0);
        }
        let mut focuser = None;
        if let Some(delta) = offsets.delta.filter(|d| *d != 0) {
            match &peers.focuser {
                None => st.report.focus_verification.push(format!(
                    "offset_not_applied: no focuser is connected ({delta} steps)"
                )),
                Some(_) if !info.owns(DeviceKind::Focuser) => {
                    st.report.focus_verification.push(format!(
                    "offset_not_applied: the handshake does not own the focuser ({delta} steps)"
                ))
                }
                Some(handle) => focuser = Some((handle.clone(), delta)),
            }
        }

        self.emit(
            EventKind::OperationStarted {
                op: OpKind::FilterHandshake,
            },
            cmd,
        );

        // Step 4: pause guiding only when an offset will move the focuser.
        if let (Some(guider), Some(_)) = (&peers.guider, &focuser) {
            if self.settings.pause_guiding_on_offset && is_guiding(guider) {
                if info.owns(DeviceKind::Guider) {
                    match self
                        .guider_command(guider, GuiderCmd::Pause { full: false }, info)
                        .await
                    {
                        GuiderReply::Done => st.report.guiding_paused = true,
                        GuiderReply::Refused(why) => {
                            st.report.notes.push(format!("guider_pause_failed: {why}"))
                        }
                        GuiderReply::Unconfirmed(why) => {
                            // PHD2 may have applied it: resume on every path.
                            st.report.guiding_paused = true;
                            st.report
                                .notes
                                .push(format!("guider_pause_unconfirmed: {why}"));
                        }
                    }
                } else {
                    st.report
                        .notes
                        .push("guider_not_owned: guiding was not paused".into());
                }
            }
        }
        if let Some(by) = cancelled_now(cancel) {
            return Err(Stop {
                outcome: OpOutcome::Aborted { by: Some(by) },
                block: None,
            });
        }

        // Step 5: command the position.
        let (_, wheel_permit) = info.child();
        let wheel_ticket = match self
            .wheel
            .send(FilterWheelCmd::SetPosition(to_slot), wheel_permit)
            .await
        {
            Ok(ticket) => ticket,
            Err(err) if may_have_reached_device(&err) => {
                // The initiator may be in flight: the wheel may be moving or
                // at another slot, and no offset is applied.
                self.history.record(clock.now_mono_ns());
                self.emit(EventKind::FilterMoving { to_slot }, cmd);
                self.emit(
                    EventKind::FilterArrived {
                        slot: None,
                        verified: false,
                    },
                    cmd,
                );
                let outcome = OpOutcome::Unknown(DeviceError::outcome_unknown(format!(
                    "the command to slot {to_slot} may have reached the wheel ({}: {}); \
                     its position is unknown",
                    err.class.as_str(),
                    err.message
                )));
                let reason = format!(
                    "the filter wheel did not verify slot {to_slot}: {}",
                    outcome_text(&outcome)
                );
                return Err(Stop::blocked(outcome, reason));
            }
            Err(err) => return Err(Stop::refused(err)),
        };
        self.history.record(clock.now_mono_ns());
        self.emit(EventKind::FilterMoving { to_slot }, cmd);

        let mut parallel: Option<OpTicket> = None;
        let mut offset_error: Option<DeviceError> = None;
        if self.settings.parallel_offset_move {
            if let Some((handle, delta)) = &focuser {
                match self.start_offset_move(handle, *delta, info).await {
                    Ok((id, ticket)) => {
                        st.report.focuser_command_id = Some(id);
                        parallel = Some(ticket);
                    }
                    Err(err) => offset_error = Some(err),
                }
            }
        }

        // Steps 6 and 7: arrival, verified by the wheel operation.
        let budget = self.settings.switch_timeout() + WATCHDOG_GRACE;
        let waited = self.wait(wheel_ticket, budget, Some(&mut *cancel)).await;
        st.wheel_done_at = Some(clock.now_utc());
        let mut outcome = match waited {
            Waited::Done(OpOutcome::TimedOut) => OpOutcome::Unknown(DeviceError::outcome_unknown(
                format!("{ARRIVAL_TIMEOUT}: the wheel move timed out; its position is unknown"),
            )),
            Waited::Done(outcome) => outcome,
            Waited::Watchdog => {
                let _ = self.wheel.abort();
                OpOutcome::Unknown(DeviceError::outcome_unknown(format!(
                    "{ARRIVAL_TIMEOUT}: the wheel worker did not resolve the move within {} ms",
                    budget.as_millis()
                )))
            }
            Waited::Cancelled(by) => OpOutcome::Aborted { by: Some(by) },
        };
        if let Some(by) = cancelled_now(cancel) {
            if !outcome.is_success() {
                outcome = OpOutcome::Aborted { by: Some(by) };
            }
        }
        if !outcome.is_success() {
            let seen = current_slot(&self.wheel.snapshot(), &self.wheel.health());
            st.report.arrived_slot = seen;
            self.emit(
                EventKind::FilterArrived {
                    slot: seen,
                    verified: false,
                },
                cmd,
            );
            if parallel.take().is_some() {
                if let Some((handle, _)) = &focuser {
                    let _ = handle.abort();
                }
                st.report
                    .notes
                    .push("offset_move_halted: the wheel did not arrive".into());
            }
            let reason = if is_wrong_slot(&outcome) {
                format!(
                    "wrong slot after a filter change: {}",
                    outcome_text(&outcome)
                )
            } else {
                format!(
                    "the filter wheel did not verify slot {to_slot}: {}",
                    outcome_text(&outcome)
                )
            };
            return Err(Stop::blocked(outcome, reason));
        }
        st.wheel_arrived = true;
        st.report.arrived_slot = Some(to_slot);
        st.verified_at = Some(clock.now_utc());
        self.emit(
            EventKind::FilterArrived {
                slot: Some(to_slot),
                verified: true,
            },
            cmd,
        );

        // Step 8: settle (the profile value wins).
        let settle = self.settings.settle(model.settle_ms);
        let settle_start = clock.now_mono_ns();
        let sleep = self.services.timer.sleep(settle);
        let settled = tokio::select! {
            biased;
            () = sleep => None,
            by = cancelled(cancel) => Some(by),
        };
        let settle_ms = self.elapsed_ms(settle_start);
        st.settle_ms_actual = Some(settle_ms);
        if let Some(by) = settled {
            if parallel.take().is_some() {
                if let Some((handle, _)) = &focuser {
                    let _ = handle.abort();
                }
            }
            return Err(Stop::blocked(
                OpOutcome::Aborted { by: Some(by) },
                "the filter change was aborted while settling",
            ));
        }
        self.emit(
            EventKind::FilterSettled {
                slot: to_slot,
                settle_ms,
            },
            cmd,
        );

        // Step 9: the relative offset move.
        if let Some((handle, delta)) = &focuser {
            let started = match (parallel.take(), offset_error.take()) {
                (Some(ticket), _) => Ok(ticket),
                (None, Some(err)) => Err(err),
                (None, None) => {
                    self.start_offset_move(handle, *delta, info)
                        .await
                        .map(|(id, ticket)| {
                            st.report.focuser_command_id = Some(id);
                            ticket
                        })
                }
            };
            let ticket = match started {
                Ok(ticket) => ticket,
                Err(err) => {
                    st.report
                        .focus_verification
                        .push(format!("offset_not_applied: {err}"));
                    return Err(Stop::blocked(
                        OpOutcome::Failed(err),
                        "the focus offset for the new filter could not be applied",
                    ));
                }
            };
            let budget = self.settings.focuser_move_timeout() + WATCHDOG_GRACE;
            let outcome = match self.wait(ticket, budget, Some(&mut *cancel)).await {
                Waited::Done(outcome) => outcome,
                Waited::Watchdog => {
                    let _ = handle.abort();
                    OpOutcome::Unknown(DeviceError::outcome_unknown(
                        "the focuser worker did not resolve the offset move",
                    ))
                }
                Waited::Cancelled(by) => {
                    let _ = handle.abort();
                    OpOutcome::Aborted { by: Some(by) }
                }
            };
            if !outcome.is_success() {
                st.report
                    .focus_verification
                    .push(format!("offset_not_applied: {}", outcome_text(&outcome)));
                return Err(Stop::blocked(
                    outcome,
                    "the focus offset move for the new filter did not complete",
                ));
            }
            st.report.offset_applied_steps = Some(*delta);
            self.emit(
                EventKind::FilterOffsetApplied {
                    steps: *delta,
                    focuser_move_id: st.report.focuser_command_id.map(|id| id.0),
                },
                cmd,
            );
        }

        // Step 10.
        self.emit_verification(st);

        // Step 11: resume guiding and wait for it to settle.
        if st.report.guiding_paused {
            if let Some(guider) = &peers.guider {
                st.guider_resumed = true;
                if let Err(why) = self.resume_guider(guider, info).await {
                    return Err(Stop::blocked(
                        OpOutcome::Failed(DeviceError::driver(format!(
                            "guider_resume_failed: {why}"
                        ))),
                        "guiding did not resume after the filter change",
                    ));
                }
            }
        }
        Ok(())
    }

    fn emit_verification(&self, st: &mut RunState) {
        if !st.verification_emitted && !st.report.focus_verification.is_empty() {
            st.verification_emitted = true;
            self.emit(
                EventKind::FocusVerificationRequired {
                    reason: st.report.focus_verification.join("; "),
                },
                st.cmd,
            );
        }
    }

    /// Steps 12-13 bookkeeping: the resume block, the record and the end
    /// events.
    fn finish(
        &self,
        mut st: RunState,
        req: &HandshakeRequest,
        model: &ReconciledWheel,
    ) -> HandshakeReport {
        let clock = Arc::clone(&self.services.clock);
        let duration_ms = self.elapsed_ms(st.started_mono_ns);
        st.report.duration_ms = duration_ms;

        if st.wheel_arrived {
            self.emit_verification(&mut st);
        }
        if let Some(reason) = st.block_reason.take() {
            let block = ResumeBlock {
                id: Uuid::new_v4(),
                kind: ResumeBlockKind::FilterHandshakeIncomplete,
                raised_at: clock.now_utc(),
                raised_mono_ns: clock.now_mono_ns(),
                device_kind: Some(DeviceKind::FilterWheel),
                connection_id: Some(self.wheel.connection_id()),
                command_id: Some(st.cmd),
                reason,
            };
            self.services.events.raise_resume_block(block.clone());
            st.report.resume_block = Some(block);
        }
        st.report.handshake_ok = st.report.outcome.is_success() && st.report.resume_block.is_none();

        if st.record {
            if let Some(to_slot) = st.report.to_slot {
                let name = |slot: u32| {
                    model
                        .name_of(slot)
                        .map_or_else(|| default_slot_name(slot), str::to_string)
                };
                let policy = req.policy.clone().unwrap_or_default();
                let source_version = match &st.source {
                    CommandSource::Policy { version, .. } => Some(version.clone()),
                    _ => None,
                };
                let record = FilterSwitchRecord {
                    id: Uuid::new_v4(),
                    session_id: req.session_id.clone(),
                    from_filter: st.report.from_slot.map(name),
                    to_filter: name(to_slot),
                    focus_offset: st.report.offset_to,
                    duration_ms: Some(duration_ms),
                    success: st.report.handshake_ok,
                    command_id: Some(st.cmd),
                    from_slot: st.report.from_slot,
                    to_slot,
                    requested_at: st.requested_at,
                    wheel_done_at: st.wheel_done_at,
                    verified_at: st.verified_at,
                    settle_ms_actual: st.settle_ms_actual,
                    offset_from: st.report.offset_from,
                    offset_to: st.report.offset_to,
                    offset_applied_steps: st.report.offset_applied_steps,
                    focuser_move_id: st.report.focuser_command_id.map(|id| id.0),
                    guiding_paused: st.report.guiding_paused,
                    handshake_ok: st.report.handshake_ok,
                    outcome: st.report.outcome.clone(),
                    trigger: req.trigger.clone().or_else(|| Some(st.source.label())),
                    policy_version: policy.policy_version.or(source_version),
                    arm: policy.arm,
                    propensity: policy.propensity,
                    background_before: req.background_before,
                    background_after: None,
                };
                self.services
                    .recorder
                    .record(GroundTruthRecord::FilterSwitch(Box::new(record.clone())));
                st.report.record = Some(record);
            }
        }

        if st.requested {
            self.emit(
                EventKind::FilterChangeEnded {
                    outcome: st.report.outcome.clone(),
                    duration_ms,
                },
                st.cmd,
            );
        }
        if st.commanded {
            self.emit(
                EventKind::OperationEnded {
                    op: OpKind::FilterHandshake,
                    outcome: st.report.outcome.clone(),
                    duration_ms,
                },
                st.cmd,
            );
        }
        st.report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::SafetyRejection;
    use crate::equipment::filter::reconcile::{
        reconcile, FilterWheelProfile, OffsetQuality, ProfileSlot,
    };
    use crate::equipment::filter::shared_wheel;
    use crate::equipment::filter::test_support::{
        entries, push, spawn_wheel_worker, timeline, CollectingSink, ManualTimer, Timeline,
        WheelWorker,
    };
    use crate::equipment::sim::filterwheel::{
        SimFilterWheel, SimFilterWheelConfig, SimFilterWheelControl,
    };
    use crate::equipment::snapshot::{ExposureRef, FocuserSnapshot, GuiderSnapshot, SlotInfo};
    use crate::equipment::testing::fake_handles::{
        fake_filter_wheel, fake_focuser, fake_guider, FakeCameraActivity, FakeDevice, FakeResponse,
        RecordingSink,
    };
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::{Device, FilterWheelDriver};
    use crate::equipment::types::ObsState;
    use std::time::Instant;

    const NAMES: [&str; 7] = ["L", "R", "G", "B", "Ha", "OIII", "SII"];

    fn measured_profile(offsets: [i32; 7]) -> FilterWheelProfile {
        FilterWheelProfile {
            slots: NAMES
                .iter()
                .zip(offsets)
                .enumerate()
                .map(|(i, (name, offset))| ProfileSlot {
                    index: i as u32,
                    name: Some((*name).into()),
                    offset: Some(offset),
                    offset_quality: Some(OffsetQuality::Measured),
                    ..ProfileSlot::default()
                })
                .collect(),
            ..FilterWheelProfile::default()
        }
    }

    fn operator() -> CommandSource {
        CommandSource::Operator { overriding: false }
    }

    fn permit(target: &FilterTarget, source: CommandSource, priority: Priority) -> Permit {
        let cmd = GatedCommand::new(
            GatedOp::FilterHandshake {
                target: target.clone(),
            },
            source,
            AutonomyMode::Manual,
            priority,
        );
        Permit::new(&cmd, None)
    }

    fn wait_for(what: &str, cond: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !cond() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    struct Options {
        settings: FilterSettings,
        profile: Option<FilterWheelProfile>,
        guiding: bool,
        child_checks: Arc<dyn ChildCommandCheck>,
    }

    impl Default for Options {
        fn default() -> Self {
            Self {
                settings: FilterSettings::default(),
                profile: Some(measured_profile([0, 12, 8, 15, 120, 118, 122])),
                guiding: true,
                child_checks: Arc::new(NoChildChecks),
            }
        }
    }

    struct Rig {
        clock: Arc<ManualClock>,
        timeline: Timeline,
        events: Arc<CollectingSink>,
        recorder: Arc<RecordingSink>,
        timer: Arc<ManualTimer>,
        camera: Arc<FakeCameraActivity>,
        focuser: FocuserHandle,
        focuser_fake: FakeDevice<FocuserCmd, FocuserSnapshot>,
        guider: GuiderHandle,
        guider_fake: FakeDevice<GuiderCmd, GuiderSnapshot>,
        sim: Option<SimFilterWheelControl>,
        wheel_fake: Option<FakeDevice<FilterWheelCmd, FilterWheelSnapshot>>,
        changer: Arc<FilterChanger>,
        _worker: Option<WheelWorker>,
    }

    enum Wheel {
        Sim(SimFilterWheelConfig),
        Fake(FilterWheelSnapshot),
    }

    fn rig(wheel: Wheel, options: Options) -> Rig {
        let clock = Arc::new(ManualClock::new());
        let timeline = timeline();
        let events = CollectingSink::with_timeline(&timeline);
        let recorder = RecordingSink::new();
        let timer = ManualTimer::new(&clock);
        let camera = FakeCameraActivity::new();

        let (handle, sim, wheel_fake, worker, model) = match wheel {
            Wheel::Sim(config) => {
                let mut sim = SimFilterWheel::new(config, Arc::clone(&clock) as Arc<dyn Clock>);
                sim.connect(Duration::from_secs(1)).unwrap();
                let control = sim.control();
                let model =
                    shared_wheel(reconcile(&sim.slots().unwrap(), options.profile.as_ref()));
                let (handle, worker) = spawn_wheel_worker(
                    Box::new(sim),
                    Arc::clone(&clock),
                    options.settings.clone(),
                    Arc::clone(&model),
                );
                wait_for("the first wheel snapshot", || {
                    handle.snapshot().position.is_fresh()
                });
                (handle, Some(control), None, Some(worker), model)
            }
            Wheel::Fake(initial) => {
                let slots: Vec<SlotInfo> = NAMES
                    .iter()
                    .enumerate()
                    .map(|(i, name)| SlotInfo {
                        index0: i as u32,
                        device_name: Some((*name).into()),
                        device_focus_offset: Some(0),
                    })
                    .collect();
                let model = shared_wheel(reconcile(&slots, options.profile.as_ref()));
                let (handle, fake) = fake_filter_wheel(initial);
                (handle, None, Some(fake), None, model)
            }
        };

        let (focuser, focuser_fake) = fake_focuser(FocuserSnapshot::default());
        let log = Arc::clone(&timeline);
        focuser_fake.set_responder(move |cmd, _| {
            push(&log, format!("focuser:{cmd:?}"));
            FakeResponse::succeed()
        });
        let state = if options.guiding {
            GuiderAppState::Guiding
        } else {
            GuiderAppState::Looping
        };
        let (guider, guider_fake) = fake_guider(GuiderSnapshot {
            app_state: Observed::fresh(state, Source::Phd2Event, clock.now_utc(), 0),
            ..GuiderSnapshot::default()
        });
        let log = Arc::clone(&timeline);
        let at = clock.now_utc();
        guider_fake.set_responder(move |cmd, snap| {
            match cmd {
                GuiderCmd::Pause { .. } => {
                    push(&log, "guider:pause");
                    snap.app_state =
                        Observed::fresh(GuiderAppState::Paused, Source::Phd2Event, at, 1);
                }
                GuiderCmd::Resume => {
                    push(&log, "guider:resume");
                    snap.app_state =
                        Observed::fresh(GuiderAppState::Guiding, Source::Phd2Event, at, 2);
                }
                other => push(&log, format!("guider:{other:?}")),
            }
            FakeResponse::succeed()
        });

        let services = HandshakeServices {
            clock: Arc::clone(&clock) as Arc<dyn Clock>,
            timer: Arc::clone(&timer) as Arc<dyn HandshakeTimer>,
            events: Arc::clone(&events) as Arc<dyn FilterEventSink>,
            recorder: Arc::clone(&recorder) as Arc<dyn RecorderSink>,
            child_checks: options.child_checks,
        };
        let changer = Arc::new(FilterChanger::new(
            handle,
            model,
            options.settings,
            services,
        ));
        Rig {
            clock,
            timeline,
            events,
            recorder,
            timer,
            camera,
            focuser,
            focuser_fake,
            guider,
            guider_fake,
            sim,
            wheel_fake,
            changer,
            _worker: worker,
        }
    }

    fn sim_rig(options: Options) -> Rig {
        rig(Wheel::Sim(SimFilterWheelConfig::default()), options)
    }

    impl Rig {
        fn peers(&self) -> Peers {
            Peers {
                focuser: Some(self.focuser.clone()),
                guider: Some(self.guider.clone()),
                camera: Some(Arc::clone(&self.camera) as Arc<dyn CameraActivity>),
            }
        }

        async fn change_as(
            &self,
            target: FilterTarget,
            source: CommandSource,
            priority: Priority,
            peers: Peers,
        ) -> HandshakeReport {
            let permit = permit(&target, source, priority);
            self.changer
                .change(peers, HandshakeRequest::new(target), permit)
                .await
        }

        async fn change(&self, name: &str) -> HandshakeReport {
            self.change_as(
                FilterTarget::Name(name.into()),
                operator(),
                Priority::Exposure,
                self.peers(),
            )
            .await
        }

        fn sim(&self) -> &SimFilterWheelControl {
            self.sim.as_ref().unwrap()
        }

        fn switch_records(&self) -> Vec<FilterSwitchRecord> {
            self.recorder
                .records()
                .into_iter()
                .filter_map(|rec| match rec {
                    GroundTruthRecord::FilterSwitch(rec) => Some(*rec),
                    _ => None,
                })
                .collect()
        }

        fn clear_timeline(&self) {
            self.timeline
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clear();
        }
    }

    fn fresh_slot(slot: u32) -> FilterWheelSnapshot {
        FilterWheelSnapshot {
            position: Observed::fresh(slot, Source::Property, chrono::Utc::now(), 0),
            verified_slot: Some(slot),
            verified: true,
            ..FilterWheelSnapshot::default()
        }
    }

    /// Acceptance (§5.3, §5.7 item 5): the v6 §8.2.1 order, shown by one
    /// timeline across the wheel, guider and focuser, with the profile offset
    /// applied as a relative move.
    #[tokio::test]
    async fn the_handshake_runs_in_the_v6_order_and_applies_the_profile_offset() {
        let rig = sim_rig(Options::default());
        let report = rig.change("Ha").await;
        assert_eq!(report.outcome, OpOutcome::Succeeded, "{report:?}");
        assert!(report.handshake_ok && !report.noop);
        assert_eq!(
            entries(&rig.timeline),
            vec![
                "event:filter_change_requested",
                "event:operation_commanded",
                "event:operation_started",
                "guider:pause",
                "event:filter_moving",
                "event:filter_arrived",
                "event:filter_settled",
                "focuser:MoveBy(120)",
                "event:filter_offset_applied",
                "guider:resume",
                "event:filter_change_ended",
                "event:operation_ended",
            ]
        );
        assert_eq!(rig.sim().commanded(), vec![4]);
        assert_eq!((report.from_slot, report.to_slot), (Some(0), Some(4)));
        assert_eq!(report.offset_applied_steps, Some(120));
        assert!(report.guiding_paused);
        assert!(report.focus_verification.is_empty(), "measured offsets");
        assert!(report.resume_block.is_none() && rig.events.blocks().is_empty());
        assert_eq!(
            report.focuser_command_id,
            rig.focuser_fake.command_ids().first().copied()
        );

        let events = rig.events.events();
        assert!(events
            .iter()
            .all(|e| e.command_id == Some(report.command_id)
                && e.device_kind == Some(DeviceKind::FilterWheel)));
        assert!(events.iter().any(|e| matches!(
            &e.event,
            EventKind::FilterArrived {
                slot: Some(4),
                verified: true
            }
        )));
        assert!(events.iter().any(|e| matches!(
            &e.event,
            EventKind::FilterOffsetApplied {
                steps: 120,
                focuser_move_id: Some(_)
            }
        )));
        assert!(events.iter().any(|e| matches!(
            &e.event,
            EventKind::OperationEnded {
                op: OpKind::FilterHandshake,
                outcome: OpOutcome::Succeeded,
                ..
            }
        )));

        let records = rig.switch_records();
        assert_eq!(records.len(), 1);
        let rec = &records[0];
        assert_eq!(rec.from_filter.as_deref(), Some("L"));
        assert_eq!(rec.to_filter, "Ha");
        assert_eq!((rec.from_slot, rec.to_slot), (Some(0), 4));
        assert_eq!((rec.offset_from, rec.offset_to), (Some(0), Some(120)));
        assert_eq!(rec.focus_offset, Some(120));
        assert_eq!(rec.settle_ms_actual, Some(500));
        assert!(rec.success && rec.handshake_ok && rec.guiding_paused);
        assert!(rec.wheel_done_at.is_some() && rec.verified_at.is_some());
        assert!(rec.requested_at <= rec.wheel_done_at.unwrap());
        assert_eq!(rec.trigger.as_deref(), Some("operator"));
        assert_eq!(
            rec.focuser_move_id,
            report.focuser_command_id.map(|id| id.0)
        );
        assert_eq!(rec.command_id, Some(report.command_id));

        let snap = rig.changer.wheel().snapshot();
        assert!(snap.verified && snap.verified_slot == Some(4));
        assert_eq!(snap.slots.len(), 7);

        // The next change applies the difference of the two offsets,
        // relatively, never as an absolute target.
        rig.clear_timeline();
        let report = rig.change("R").await;
        assert!(report.handshake_ok, "{report:?}");
        assert_eq!(report.offset_applied_steps, Some(12 - 120));
        assert_eq!(
            rig.focuser_fake.commands(),
            vec![FocuserCmd::MoveBy(120), FocuserCmd::MoveBy(-108)]
        );
        assert_eq!(rig.switch_records().len(), 2);
    }

    #[tokio::test]
    async fn a_wheel_timeout_is_outcome_unknown_with_a_resume_block() {
        let rig = sim_rig(Options {
            settings: FilterSettings {
                max_switch_timeout_ms: 2000,
                ..FilterSettings::default()
            },
            ..Options::default()
        });
        rig.sim().jam_next_move();
        let started = rig.clock.now_mono_ns();
        let report = rig.change("Ha").await;
        match &report.outcome {
            OpOutcome::Unknown(err) => {
                assert_eq!(err.class, ErrorClass::OutcomeUnknown);
                assert!(err.message.starts_with(ARRIVAL_TIMEOUT), "{err}");
            }
            other => panic!("expected OutcomeUnknown, got {other:?}"),
        }
        assert!(rig.clock.now_mono_ns() - started >= 2_000_000_000);
        let block = report.resume_block.clone().expect("a resume block");
        assert_eq!(block.kind, ResumeBlockKind::FilterHandshakeIncomplete);
        assert_eq!(block.command_id, Some(report.command_id));
        assert_eq!(rig.events.blocks(), vec![block.clone()]);
        assert!(!report.handshake_ok);
        // No focus offset for a wheel whose position is unknown, and guiding
        // was resumed anyway.
        assert!(rig.focuser_fake.commands().is_empty());
        assert_eq!(
            rig.guider_fake.commands(),
            vec![GuiderCmd::Pause { full: false }, GuiderCmd::Resume]
        );
        let names = rig.events.names();
        assert!(!names.contains(&"filter_settled"));
        assert!(rig.events.events().iter().any(|e| matches!(
            &e.event,
            EventKind::FilterArrived {
                verified: false,
                ..
            }
        )));
        assert_eq!(names.last(), Some(&"operation_ended"));
        let rec = &rig.switch_records()[0];
        assert!(!rec.success && !rec.handshake_ok);
        assert_eq!(rec.outcome.label(), "unknown");
        assert!(rec.verified_at.is_none());
        // While the block is active the gate answers automatic captures 423.
        assert_eq!(
            GateError::ResumeBlocked {
                blocks: vec![block.kind.clone()]
            }
            .http_status(),
            423
        );
        assert!(block.kind.operator_clears());
    }

    #[tokio::test]
    async fn a_wrong_slot_is_detected_and_blocks_resumption() {
        let rig = sim_rig(Options::default());
        rig.sim().wrong_slot_next_move(2);
        let report = rig.change("Ha").await;
        assert!(is_wrong_slot(&report.outcome), "{report:?}");
        assert_eq!(report.arrived_slot, Some(2));
        assert!(rig.events.events().iter().any(|e| matches!(
            &e.event,
            EventKind::FilterArrived {
                slot: Some(2),
                verified: false
            }
        )));
        let block = report.resume_block.expect("a resume block");
        assert!(block.reason.contains("wrong slot"));
        assert!(rig.focuser_fake.commands().is_empty());
        assert!(!rig.changer.wheel().snapshot().verified);
    }

    #[tokio::test]
    async fn guiding_is_paused_only_for_a_nonzero_offset() {
        // Equal offsets: Δ = 0, no pause and no focuser move.
        let rig = sim_rig(Options {
            profile: Some(measured_profile([0, 0, 8, 15, 120, 118, 122])),
            ..Options::default()
        });
        let report = rig.change("R").await;
        assert!(report.handshake_ok, "{report:?}");
        assert_eq!(report.offset_applied_steps, Some(0));
        assert!(!report.guiding_paused);
        assert!(rig.guider_fake.commands().is_empty());
        assert!(rig.focuser_fake.commands().is_empty());
        assert!(!rig.events.names().contains(&"filter_offset_applied"));

        // Δ ≠ 0 but PHD2 is only looping: the offset moves, guiding is left.
        let rig = sim_rig(Options {
            guiding: false,
            ..Options::default()
        });
        let report = rig.change("Ha").await;
        assert!(report.handshake_ok);
        assert_eq!(rig.focuser_fake.commands(), vec![FocuserCmd::MoveBy(120)]);
        assert!(rig.guider_fake.commands().is_empty());

        // Δ ≠ 0 and guiding, but pausing is disabled.
        let rig = sim_rig(Options {
            settings: FilterSettings {
                pause_guiding_on_offset: false,
                ..FilterSettings::default()
            },
            ..Options::default()
        });
        assert!(rig.change("Ha").await.handshake_ok);
        assert!(rig.guider_fake.commands().is_empty());
    }

    #[tokio::test]
    async fn a_change_is_refused_while_exposing() {
        let rig = sim_rig(Options::default());
        rig.camera.start_exposure(ExposureRef {
            duration_s: Some(300.0),
            ..ExposureRef::default()
        });
        let report = rig.change("Ha").await;
        match &report.outcome {
            OpOutcome::Failed(err) => {
                assert_eq!(err.class, ErrorClass::SafetyRejected);
                assert!(err.message.contains("exposing"));
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        assert!(rig.sim().commanded().is_empty(), "the wheel never moved");
        assert!(rig.focuser_fake.commands().is_empty());
        assert!(rig.guider_fake.commands().is_empty());
        assert!(
            report.resume_block.is_none(),
            "nothing moved, nothing blocks"
        );
        assert_eq!(
            rig.events.names(),
            vec![
                "filter_change_requested",
                "operation_commanded",
                "filter_change_ended",
                "operation_ended",
            ]
        );
        assert!(!rig.switch_records()[0].success);

        // Without any camera hook the exposure state is unknown: automatic
        // sources are refused, an operator change proceeds with a note.
        rig.camera.finish_exposure();
        let no_camera = Peers {
            camera: None,
            ..rig.peers()
        };
        let report = rig
            .change_as(
                FilterTarget::Name("Ha".into()),
                CommandSource::Policy {
                    model: "fw2".into(),
                    version: "1".into(),
                },
                Priority::Recommendation,
                no_camera.clone(),
            )
            .await;
        assert!(
            matches!(&report.outcome, OpOutcome::Failed(e) if e.message.contains("exposure_state_unknown"))
        );
        let report = rig
            .change_as(
                FilterTarget::Name("Ha".into()),
                operator(),
                Priority::Exposure,
                no_camera,
            )
            .await;
        assert!(report.handshake_ok, "{report:?}");
        assert_eq!(report.notes, vec!["exposure_state_unknown".to_string()]);
    }

    #[tokio::test]
    async fn toml_seeds_are_applied_only_after_an_explicit_import() {
        // No profile: the sim wheel's own offsets are zero, so the TOML seed
        // offsets (which exist in config) are never used.
        let rig = sim_rig(Options {
            profile: None,
            ..Options::default()
        });
        let seeds = FilterWheelConfig {
            filter_names: NAMES.iter().map(|n| n.to_string()).collect(),
            focus_offsets: vec![0, 12, 8, 15, 120, 118, 122],
            ..FilterWheelConfig::default()
        };
        let report = rig.change("Ha").await;
        assert!(report.handshake_ok, "{report:?}");
        assert_eq!(report.offset_applied_steps, Some(0));
        assert!(rig.focuser_fake.commands().is_empty());
        assert!(report
            .focus_verification
            .iter()
            .any(|r| r.starts_with("offset_uncertain")));
        assert!(rig.events.names().contains(&"focus_verification_required"));

        // The operator imports the seeds into the profile (audited by the
        // gate); only now do they drive the focuser.
        let import = permit(&FilterTarget::Slot(0), operator(), Priority::Exposure);
        let result = rig.changer.import_seed(&seeds, &import).unwrap();
        assert_eq!(result.applied_slots, 7);
        assert!(rig.recorder.records().iter().any(|rec| matches!(
            rec,
            GroundTruthRecord::Profile(update) if update.patch.get("seed_import").is_some()
        )));
        let report = rig.change("L").await;
        assert!(report.handshake_ok, "{report:?}");
        assert_eq!(rig.focuser_fake.commands(), vec![FocuserCmd::MoveBy(-120)]);
        assert!(
            !report.focus_verification.is_empty(),
            "imported seeds are estimates and still need verification"
        );
        let unowned = Permit::unmanaged(CommandId::new(), Priority::Exposure, operator());
        assert_eq!(
            rig.changer.import_seed(&seeds, &unowned).unwrap_err().class,
            ErrorClass::OwnershipConflict
        );
    }

    #[tokio::test]
    async fn the_verified_current_slot_is_a_silent_noop() {
        let rig = sim_rig(Options::default());
        let report = rig.change("L").await;
        assert!(report.noop && report.handshake_ok, "{report:?}");
        assert!(report.from_verified);
        assert_eq!(report.arrived_slot, Some(0));
        assert!(rig.sim().commanded().is_empty());
        // No `filter_*` event: H2 and the window builder reset on those, and
        // a sequencer may assert the filter before every frame.
        assert!(rig.events.names().is_empty(), "{:?}", rig.events.names());
        assert!(rig.switch_records().is_empty());
        assert!(rig.focuser_fake.commands().is_empty());
        assert!(rig.guider_fake.commands().is_empty());
    }

    /// Review: after a failed handshake the wheel reports a slot it has not
    /// verified, and the focuser still holds the offset of the slot before.
    /// The next change must not compute Δ from the unverified slot.
    #[tokio::test]
    async fn an_unverified_current_slot_gives_no_offset_and_requires_verification() {
        let rig = sim_rig(Options::default());
        rig.sim().wrong_slot_next_move(2);
        let first = rig.change("Ha").await;
        assert!(is_wrong_slot(&first.outcome), "{first:?}");
        assert!(rig.focuser_fake.commands().is_empty());
        let snap = rig.changer.wheel().snapshot();
        assert_eq!(snap.position.value, Some(2));
        assert!(!snap.verified);

        rig.clear_timeline();
        let second = rig.change("R").await;
        assert!(second.handshake_ok, "the wheel move itself succeeds");
        assert_eq!(second.from_slot, Some(2), "the reported slot is kept");
        assert!(!second.from_verified);
        assert_eq!(second.offset_from, None, "the held offset is unknown");
        assert_eq!(second.offset_to, Some(12));
        assert_eq!(second.offset_applied_steps, None);
        assert_eq!(
            second.focus_verification,
            vec!["prev_offset_unknown: slot 3 is not verified".to_string()]
        );
        assert!(rig.focuser_fake.commands().is_empty(), "no offset move");
        assert!(entries(&rig.timeline).contains(&"event:focus_verification_required".to_string()));
        // No offset move, so guiding was not paused for this change.
        assert_eq!(
            rig.guider_fake.commands(),
            vec![GuiderCmd::Pause { full: false }, GuiderCmd::Resume]
        );
        let snap = rig.changer.wheel().snapshot();
        assert!(snap.verified && snap.verified_slot == Some(1));
        let rec = rig.switch_records().pop().unwrap();
        assert_eq!((rec.from_slot, rec.offset_from), (Some(2), None));
        assert_eq!(rec.offset_applied_steps, None);
    }

    /// Review: a target equal to the unverified current slot is moved to and
    /// verified, not treated as Δ = 0.
    #[tokio::test]
    async fn a_target_equal_to_the_unverified_current_slot_is_reverified() {
        let rig = sim_rig(Options::default());
        rig.sim().wrong_slot_next_move(2);
        assert!(is_wrong_slot(&rig.change("Ha").await.outcome));

        let report = rig.change("G").await;
        assert!(!report.noop, "an unverified slot is never a no-op");
        assert!(report.handshake_ok, "{report:?}");
        assert_eq!((report.from_slot, report.to_slot), (Some(2), Some(2)));
        assert_eq!(rig.sim().commanded(), vec![4, 2]);
        assert_eq!(report.offset_applied_steps, None, "not Some(0)");
        assert!(report
            .focus_verification
            .iter()
            .any(|r| r.starts_with("prev_offset_unknown")));
        assert!(rig.events.names().contains(&"focus_verification_required"));
        assert!(rig.focuser_fake.commands().is_empty());
        let snap = rig.changer.wheel().snapshot();
        assert!(snap.verified && snap.verified_slot == Some(2));

        // Now verified: asking for it again is a no-op.
        let before = rig.events.names().len();
        let again = rig.change("G").await;
        assert!(again.noop && again.from_verified);
        assert_eq!(rig.events.names().len(), before);
    }

    fn holding_guider(rig: &Rig, resume: FakeResponse) {
        let log = Arc::clone(&rig.timeline);
        rig.guider_fake.set_responder(move |cmd, _| match cmd {
            GuiderCmd::Pause { .. } => {
                push(&log, "guider:pause");
                FakeResponse::hold()
            }
            GuiderCmd::Resume => {
                push(&log, "guider:resume");
                resume.clone()
            }
            _ => FakeResponse::succeed(),
        });
    }

    fn spawn_change(rig: &Rig, name: &str) -> tokio::task::JoinHandle<HandshakeReport> {
        let changer = Arc::clone(&rig.changer);
        let peers = rig.peers();
        let target = FilterTarget::Name(name.into());
        tokio::spawn(async move {
            let permit = permit(&target, operator(), Priority::Exposure);
            changer
                .change(peers, HandshakeRequest::new(target), permit)
                .await
        })
    }

    async fn until_pending<C, S>(fake: &FakeDevice<C, S>)
    where
        C: Clone + Send + 'static,
        S: Clone + Send + Sync + 'static,
    {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !fake.has_pending() {
            assert!(Instant::now() < deadline, "the command never arrived");
            tokio::task::yield_now().await;
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Review: an abort while the pause is outstanding must not leave PHD2
    /// paused. The handshake waits for the pause to resolve, never commands
    /// the wheel, and resumes.
    #[tokio::test]
    async fn an_abort_during_the_pause_still_resumes_guiding() {
        let rig = sim_rig(Options::default());
        holding_guider(&rig, FakeResponse::succeed());
        let task = spawn_change(&rig, "Ha");
        until_pending(&rig.guider_fake).await;
        let by = rig.changer.abort();
        // The handshake sees the abort but keeps waiting for the pause.
        for _ in 0..20 {
            tokio::task::yield_now().await;
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(!task.is_finished(), "the pause outcome is awaited");
        // PHD2 applies set_paused [true] after the abort.
        assert!(rig.guider_fake.complete_pending(OpOutcome::Succeeded));
        let report = task.await.unwrap();
        assert!(
            matches!(report.outcome, OpOutcome::Aborted { by: Some(id) } if id == by),
            "{report:?}"
        );
        assert!(report.guiding_paused);
        assert_eq!(
            rig.guider_fake.commands(),
            vec![GuiderCmd::Pause { full: false }, GuiderCmd::Resume]
        );
        assert!(rig.sim().commanded().is_empty(), "the wheel never moved");
        assert!(rig.focuser_fake.commands().is_empty());
        assert!(report.resume_block.is_none(), "nothing moved, PHD2 resumed");

        // The same abort when the resume then fails: guiding is in doubt,
        // so automatic resumption is blocked.
        let rig = sim_rig(Options::default());
        holding_guider(
            &rig,
            FakeResponse::finish(OpOutcome::Failed(DeviceError::driver(
                "PHD2 refused set_paused",
            ))),
        );
        let task = spawn_change(&rig, "Ha");
        until_pending(&rig.guider_fake).await;
        rig.changer.abort();
        assert!(rig.guider_fake.complete_pending(OpOutcome::Succeeded));
        let report = task.await.unwrap();
        assert!(matches!(report.outcome, OpOutcome::Aborted { .. }));
        let block = report.resume_block.expect("a resume block");
        assert_eq!(block.kind, ResumeBlockKind::FilterHandshakeIncomplete);
        assert!(block.reason.contains("guiding did not resume"), "{block:?}");
        assert!(report
            .notes
            .iter()
            .any(|n| n.starts_with("guider_resume_failed")));
    }

    /// Review: a pause whose answer is lost may still have taken effect, so
    /// it is followed by a resume like a confirmed one.
    #[tokio::test]
    async fn an_unconfirmed_pause_is_resumed() {
        let rig = sim_rig(Options::default());
        holding_guider(&rig, FakeResponse::succeed());
        let task = spawn_change(&rig, "Ha");
        until_pending(&rig.guider_fake).await;
        assert!(rig.guider_fake.complete_pending(OpOutcome::TimedOut));
        let report = task.await.unwrap();
        assert!(report.handshake_ok, "{report:?}");
        assert!(report.guiding_paused);
        assert!(report
            .notes
            .iter()
            .any(|n| n.starts_with("guider_pause_unconfirmed")));
        assert_eq!(rig.focuser_fake.commands(), vec![FocuserCmd::MoveBy(120)]);
        let timeline = entries(&rig.timeline);
        let applied = timeline
            .iter()
            .position(|e| e == "event:filter_offset_applied")
            .unwrap();
        let resumed = timeline.iter().position(|e| e == "guider:resume").unwrap();
        assert!(applied < resumed, "{timeline:?}");

        // A pause PHD2 refused outright is not followed by a resume.
        let rig = sim_rig(Options::default());
        rig.guider_fake.set_responder(|cmd, _| match cmd {
            GuiderCmd::Pause { .. } => FakeResponse::finish(OpOutcome::Failed(
                DeviceError::driver("PHD2 is not guiding"),
            )),
            _ => FakeResponse::succeed(),
        });
        let report = rig.change("Ha").await;
        assert!(report.handshake_ok, "{report:?}");
        assert!(!report.guiding_paused);
        assert_eq!(
            rig.guider_fake.commands(),
            vec![GuiderCmd::Pause { full: false }]
        );
    }

    /// Review: an acceptance error that may come after the initiator reached
    /// the wheel is an unknown outcome with a resume block, not a refusal.
    #[tokio::test]
    async fn an_initiator_that_may_have_reached_the_wheel_blocks_resumption() {
        for err in [
            DeviceError::timeout("PUT position timed out"),
            DeviceError::transport("connection reset"),
            DeviceError::disconnected("the worker stopped while starting"),
            DeviceError::driver("HTTP 500"),
        ] {
            let class = err.class;
            let rig = rig(Wheel::Fake(fresh_slot(0)), Options::default());
            let fake = rig.wheel_fake.as_ref().unwrap();
            let reject = err.clone();
            fake.set_responder(move |_, _| FakeResponse::reject(reject.clone()));
            let report = rig.change("Ha").await;
            match &report.outcome {
                OpOutcome::Unknown(e) => {
                    assert_eq!(e.class, ErrorClass::OutcomeUnknown);
                    assert!(e.message.contains(class.as_str()), "{e}");
                }
                other => panic!("{class:?}: expected Unknown, got {other:?}"),
            }
            let block = report.resume_block.clone().expect("a resume block");
            assert_eq!(block.kind, ResumeBlockKind::FilterHandshakeIncomplete);
            assert_eq!(rig.events.blocks(), vec![block]);
            assert!(rig.events.events().iter().any(|e| matches!(
                &e.event,
                EventKind::FilterArrived {
                    slot: None,
                    verified: false
                }
            )));
            assert!(rig.events.names().contains(&"filter_moving"));
            assert_eq!(report.arrived_slot, None);
            assert!(rig.focuser_fake.commands().is_empty(), "no offset");
            assert_eq!(
                rig.guider_fake.commands(),
                vec![GuiderCmd::Pause { full: false }, GuiderCmd::Resume]
            );
            let rec = &rig.switch_records()[0];
            assert!(!rec.handshake_ok);
            assert_eq!(rec.outcome.label(), "unknown");
        }

        // A refusal raised before anything is sent stays a refusal.
        let rig = rig(Wheel::Fake(fresh_slot(0)), Options::default());
        rig.wheel_fake
            .as_ref()
            .unwrap()
            .set_responder(|_, _| FakeResponse::reject(DeviceError::invalid_value("no slot 9")));
        let report = rig.change("Ha").await;
        assert!(
            matches!(&report.outcome, OpOutcome::Failed(e) if e.class == ErrorClass::InvalidValue),
            "{report:?}"
        );
        assert!(report.resume_block.is_none());
        assert!(!rig.events.names().contains(&"filter_moving"));
    }

    #[test]
    fn acceptance_errors_are_sorted_by_whether_the_initiator_may_have_been_sent() {
        for class in [
            ErrorClass::NotConnected,
            ErrorClass::Unsupported,
            ErrorClass::InvalidValue,
            ErrorClass::InvalidWhileParked,
            ErrorClass::InvalidOperation,
            ErrorClass::SafetyRejected,
            ErrorClass::OwnershipConflict,
        ] {
            assert!(
                !may_have_reached_device(&DeviceError::new(class, "x")),
                "{class:?}"
            );
        }
        for class in [
            ErrorClass::Disconnected,
            ErrorClass::Timeout,
            ErrorClass::OutcomeUnknown,
            ErrorClass::Transport,
            ErrorClass::Driver,
            ErrorClass::Cancelled,
        ] {
            assert!(
                may_have_reached_device(&DeviceError::new(class, "x")),
                "{class:?}"
            );
        }
        assert!(definitely_not_applied(&OpOutcome::Failed(
            DeviceError::driver("no")
        )));
        assert!(!definitely_not_applied(&OpOutcome::Failed(
            DeviceError::timeout("lost")
        )));
        assert!(!definitely_not_applied(&OpOutcome::TimedOut));
        assert!(!definitely_not_applied(&OpOutcome::Aborted { by: None }));
        assert!(!definitely_not_applied(&OpOutcome::Succeeded));
    }

    #[tokio::test]
    async fn an_unknown_starting_slot_requires_focus_verification() {
        let rig = rig(
            Wheel::Fake(FilterWheelSnapshot {
                position: Observed::not_set(Source::Property),
                ..FilterWheelSnapshot::default()
            }),
            Options::default(),
        );
        rig.wheel_fake.as_ref().unwrap().set_responder(|cmd, snap| {
            let FilterWheelCmd::SetPosition(slot) = cmd;
            *snap = FilterWheelSnapshot {
                connection_id: snap.connection_id,
                ..fresh_slot(*slot)
            };
            FakeResponse::succeed()
        });
        let report = rig.change("Ha").await;
        assert!(report.handshake_ok, "{report:?}");
        assert_eq!(report.from_slot, None);
        assert_eq!(report.offset_applied_steps, None);
        assert_eq!(
            report.focus_verification,
            vec!["prev_offset_unknown".to_string()]
        );
        assert!(rig.focuser_fake.commands().is_empty());
        assert!(rig.events.names().contains(&"focus_verification_required"));
        assert_eq!(rig.switch_records()[0].from_slot, None);
    }

    #[tokio::test]
    async fn a_hung_wheel_worker_is_bounded_by_the_watchdog() {
        let rig = rig(Wheel::Fake(fresh_slot(0)), Options::default());
        let fake = rig.wheel_fake.as_ref().unwrap();
        fake.set_responder(|_, _| FakeResponse::hold());
        rig.timer.arm_watchdog();
        let peers = Peers {
            focuser: None,
            guider: None,
            camera: Some(Arc::clone(&rig.camera) as Arc<dyn CameraActivity>),
        };
        let report = rig
            .change_as(FilterTarget::Slot(3), operator(), Priority::Exposure, peers)
            .await;
        assert!(
            matches!(&report.outcome, OpOutcome::Unknown(e) if e.class == ErrorClass::OutcomeUnknown)
        );
        assert!(report.resume_block.is_some());
        assert!(
            fake.wait_for_cancels(1, Duration::from_secs(5)),
            "the worker op is cancelled"
        );
        // Without a focuser the would-be offset is reported, not invented.
        assert!(report
            .focus_verification
            .iter()
            .any(|r| r.contains("no focuser")));
    }

    #[tokio::test]
    async fn abort_cancels_the_running_handshake_and_a_second_change_is_refused() {
        let rig = rig(Wheel::Fake(fresh_slot(0)), Options::default());
        let fake = rig.wheel_fake.as_ref().unwrap();
        fake.set_responder(|_, _| FakeResponse::hold());
        let changer = Arc::clone(&rig.changer);
        let peers = rig.peers();
        let task = tokio::spawn(async move {
            let target = FilterTarget::Slot(1);
            let permit = permit(&target, operator(), Priority::Exposure);
            changer
                .change(peers, HandshakeRequest::new(target), permit)
                .await
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        while !fake.has_pending() {
            assert!(Instant::now() < deadline, "the wheel never got the command");
            tokio::task::yield_now().await;
            std::thread::sleep(Duration::from_millis(1));
        }
        let second = rig.change("G").await;
        assert!(
            matches!(&second.outcome, OpOutcome::Failed(e) if e.class == ErrorClass::InvalidOperation)
        );
        let by = rig.changer.abort();
        let report = task.await.unwrap();
        assert!(
            matches!(report.outcome, OpOutcome::Aborted { by: Some(id) } if id == by),
            "{report:?}"
        );
        assert!(
            report.resume_block.is_some(),
            "the wheel position is unknown"
        );
        assert!(fake.wait_for_cancels(1, Duration::from_secs(5)));
        // Guiding was paused for the offset and is resumed on this path too.
        assert_eq!(
            rig.guider_fake.commands(),
            vec![GuiderCmd::Pause { full: false }, GuiderCmd::Resume]
        );
    }

    struct RejectFocuser;

    impl ChildCommandCheck for RejectFocuser {
        fn check(&self, cmd: &GatedCommand) -> Result<(), GateError> {
            match cmd.op {
                GatedOp::Focuser(_) => Err(GateError::SafetyRejected(SafetyRejection::new(
                    "focuser_travel",
                    "the offset leaves the travel bounds",
                ))),
                _ => Ok(()),
            }
        }
    }

    #[tokio::test]
    async fn an_offset_refused_by_the_focuser_checks_leaves_a_block() {
        let rig = sim_rig(Options {
            child_checks: Arc::new(RejectFocuser),
            ..Options::default()
        });
        let report = rig.change("Ha").await;
        assert!(
            matches!(&report.outcome, OpOutcome::Failed(e) if e.class == ErrorClass::SafetyRejected),
            "{report:?}"
        );
        assert_eq!(report.arrived_slot, Some(4), "the wheel itself arrived");
        assert!(report.resume_block.is_some());
        assert!(rig.focuser_fake.commands().is_empty());
        assert!(report
            .focus_verification
            .iter()
            .any(|r| r.starts_with("offset_not_applied")));
        let names = rig.events.names();
        let verify = names
            .iter()
            .position(|n| *n == "focus_verification_required");
        let ended = names.iter().position(|n| *n == "filter_change_ended");
        assert!(verify.unwrap() < ended.unwrap());
        assert_eq!(
            rig.guider_fake.commands(),
            vec![GuiderCmd::Pause { full: false }, GuiderCmd::Resume]
        );
    }

    #[tokio::test]
    async fn a_parallel_offset_move_starts_with_the_wheel() {
        let rig = sim_rig(Options {
            settings: FilterSettings {
                parallel_offset_move: true,
                ..FilterSettings::default()
            },
            ..Options::default()
        });
        let report = rig.change("Ha").await;
        assert!(report.handshake_ok, "{report:?}");
        let timeline = entries(&rig.timeline);
        let moved = timeline
            .iter()
            .position(|e| e == "focuser:MoveBy(120)")
            .unwrap();
        let arrived = timeline
            .iter()
            .position(|e| e == "event:filter_arrived")
            .unwrap();
        let applied = timeline
            .iter()
            .position(|e| e == "event:filter_offset_applied")
            .unwrap();
        assert!(moved < arrived && arrived < applied, "{timeline:?}");
    }

    #[tokio::test]
    async fn a_guider_that_does_not_resume_leaves_the_handshake_incomplete() {
        let rig = sim_rig(Options::default());
        rig.guider_fake.set_responder(|cmd, _| match cmd {
            GuiderCmd::Resume => FakeResponse::finish(OpOutcome::Failed(DeviceError::driver(
                "PHD2 refused set_paused",
            ))),
            _ => FakeResponse::succeed(),
        });
        let report = rig.change("Ha").await;
        assert!(
            matches!(&report.outcome, OpOutcome::Failed(e) if e.message.contains("guider_resume_failed"))
        );
        assert_eq!(report.offset_applied_steps, Some(120));
        assert!(report.resume_block.is_some());
        assert_eq!(rig.guider_fake.commands().len(), 2, "resume is not retried");
    }

    #[tokio::test]
    async fn the_permit_must_own_the_wheel_and_targets_must_exist() {
        let rig = sim_rig(Options::default());
        let unowned = Permit::unmanaged(CommandId::new(), Priority::Exposure, operator());
        let report = rig
            .changer
            .change(
                rig.peers(),
                HandshakeRequest::new(FilterTarget::Slot(2)),
                unowned,
            )
            .await;
        assert!(
            matches!(&report.outcome, OpOutcome::Failed(e) if e.class == ErrorClass::OwnershipConflict)
        );
        assert!(rig.sim().commanded().is_empty());

        let report = rig.change("Halpha").await;
        assert!(
            matches!(&report.outcome, OpOutcome::Failed(e) if e.class == ErrorClass::InvalidValue)
        );
        assert!(report.record.is_none());
        let names = rig.events.names();
        assert_eq!(
            &names[names.len() - 2..],
            &["operation_commanded", "operation_ended"]
        );
    }

    #[test]
    fn the_gate_plug_in_shares_the_changer_history() {
        let clock = Arc::new(ManualClock::new());
        let (handle, _fake) = fake_filter_wheel(FilterWheelSnapshot::default());
        let settings = FilterSettings {
            min_switch_interval_s: 60.0,
            ..FilterSettings::default()
        };
        let services = HandshakeServices {
            clock: Arc::clone(&clock) as Arc<dyn Clock>,
            timer: Arc::new(TokioTimer),
            events: CollectingSink::new() as Arc<dyn FilterEventSink>,
            recorder: RecordingSink::new() as Arc<dyn RecorderSink>,
            child_checks: Arc::new(NoChildChecks),
        };
        let changer = FilterChanger::new(
            handle,
            shared_wheel(ReconciledWheel::default()),
            settings,
            services,
        );
        let check = changer.safety_check();
        changer.history().record(10_000_000_000);
        let cmd = GatedCommand::new(
            GatedOp::FilterHandshake {
                target: FilterTarget::Slot(1),
            },
            operator(),
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        let mut snap =
            crate::equipment::snapshot::EquipmentSnapshot::empty(clock.now_utc(), 20_000_000_000);
        snap.camera.exposing = Observed::fresh(false, Source::Derived, clock.now_utc(), 0);
        use crate::equipment::traits::SafetyCheck;
        assert!(check
            .check(&cmd, &snap)
            .unwrap_err()
            .reason
            .starts_with("min_switch_interval"));
        assert_eq!(snap.camera.exposing.state, ObsState::Fresh);
    }
}
