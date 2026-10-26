//! The device worker (spec §2.5, §2.6, §2.8, §2.15, D3, D4, D19, D22).
//! Owner: WP1A-2.
//!
//! One std thread per connected device, named `eq-{kind}-{first 8
//! characters of the connection id}`, owns the driver. Nothing else ever
//! calls it: handlers hold a [`DeviceHandle`](super::handle::DeviceHandle)
//! and the worker serves its queue.
//!
//! The loop waits on the command queue until the next poll is due, by the
//! injected [`Clock`] (each real-time wait is capped at
//! [`WorkerConfig::max_wait`], so a manual clock in a test is noticed
//! promptly). Each tick:
//!
//! 1. polls the driver and builds the typed snapshot (the kind adapter);
//! 2. publishes it to the handle's watch and to the hub ring, and emits the
//!    adapter's edge events;
//! 3. steps the active operation: completion predicate, then settle, then
//!    verification (C-5); a passed deadline cancels it (halting the
//!    hardware, C-4) and resolves `TimedOut`, or `Unknown` on a device with
//!    no halt (a filter wheel), since nothing stopped it.
//!
//! Commands are planned by the kind adapter ([`WorkerKind::plan`]) into an
//! instant action, a long [`Operation`], a halt, or a concurrent action.
//! One operation is active per device; a conflicting command fails with
//! `InvalidOperation` unless it outranks the active operation's permit (the
//! gate's preemption rule), in which case the active operation is cancelled
//! first and resolves `Aborted { by }`.
//!
//! Abort (D4, D19) never waits for this thread: the handle runs the driver's
//! out-of-band `AbortPath` on a blocking thread and puts a `Cancel` at the
//! head of the queue. The worker then cancels the operation and resolves it
//! `Aborted` once a poll reads that motion stopped (a poll that cannot read
//! motion confirms nothing; the confirmation timeout then ends it
//! `Unknown`). An abort also stops what waits behind it: the motion
//! commands still queued when the cancel is handled are refused, and so is
//! any motion command whose permit the gate issued before the abort (the
//! gate's abort epochs, [`WorkerDeps::aborts`]). They resolve
//! `Aborted { by }` without starting.
//!
//! Health (§2.8): one failed or slow poll degrades the device; loss needs
//! `lost_after_failures` consecutive failures or no good poll for
//! `lost_after_s`, then one reconnect attempt (NINA's reconnect-once). The
//! [`Watchdog`] covers a worker blocked inside a driver call: on loss it
//! starts the abort path on a blocking thread and resolves the operation
//! `Unknown` at once, without waiting for the hung call. A lost device
//! stays lost for the connection: its last values are published as stale,
//! its operation resolves `Unknown` (raising an `outcome_unknown` resume
//! block), the abort path is attempted once, and later commands fail until
//! a disconnect retires it.
//!
//! Daemon shutdown (§2.15, [`halts_at_shutdown`]) halts focuser and rotator
//! moves, stops a jog and aborts an Achernar slew. Park, unpark, home,
//! dome, cover and wheel operations are left running and resolve
//! `Unknown`.
//!
//! Every driver call runs under `catch_unwind`. On a panic the worker runs
//! the abort path, resolves the ticket `Unknown`, marks the device `Lost`
//! (`driver_panic`) and drops the driver; the permit is released by drop.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use tokio::sync::oneshot;

use crate::control::action_arbiter::outranks;

use super::commands::{
    CoverCalibratorCmd, DomeCmd, FilterWheelCmd, FocuserCmd, GuiderCmd, MountCmd, RotatorCmd,
    SafetyCmd, SwitchCmd, WeatherCmd,
};
use super::events::{
    ConnectionEnd, DeviceConnectionRecord, EquipmentEvent, EventKind, GroundTruthRecord,
};
use super::gate::{run_detached, AbortEpochs};
use super::handle::{Envelope, EnvelopeBody, Permit, WorkerEnd};
use super::health_bridge::{
    mark_lost, on_poll_failed, on_poll_ok, on_stale, HealthChange, HealthPolicy, Watchdog, Watched,
};
use super::hub::EventSink;
use super::ops::{ActiveOp, OpKind, OpPhase, OpProgress, Operation};
use super::ring::{DeviceSnapshot, Ring};
use super::traits::{AbortPath, Clock, ConnectReport, Device, PollPlan, RawPoll, RecorderSink};
use super::types::{
    kind_name, CommandId, CommandSource, DeviceError, DeviceIdentity, DeviceKind, DeviceResult,
    ErrorClass, FailureClass, HealthState, LinkState, OpOutcome, Priority,
};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

fn ns(duration: Duration) -> u64 {
    super::clock::duration_ns(duration)
}

// ---------------------------------------------------------------------------
// Cadence (§2.7)
// ---------------------------------------------------------------------------

/// Poll intervals for one kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cadence {
    pub idle: Duration,
    /// Only while an Achernar-owned operation runs (D20: PHD2's pulses never
    /// switch the mount to the active cadence).
    pub active: Duration,
}

/// The §2.7 table. The mount's 10 Hz during an Achernar pulse comes from the
/// pulse operation's `Continue { next_poll }` hint.
pub fn default_cadence(kind: DeviceKind) -> Cadence {
    let (idle_ms, active_ms) = match kind {
        DeviceKind::Mount => (1000, 250),
        DeviceKind::Focuser => (2000, 100),
        DeviceKind::FilterWheel => (2000, 250),
        DeviceKind::Rotator => (2000, 250),
        DeviceKind::CoverCalibrator => (2000, 500),
        DeviceKind::Switch => (5000, 500),
        DeviceKind::Weather => (10_000, 10_000),
        DeviceKind::SafetyMonitor => (2000, 2000),
        DeviceKind::Dome => (1000, 500),
        // PHD2 is event-driven (GuiderService); a nominal cadence for
        // staleness and window grace.
        DeviceKind::Guider => (2000, 1000),
        DeviceKind::Camera => (2000, 2000),
    };
    Cadence {
        idle: Duration::from_millis(idle_ms),
        active: Duration::from_millis(active_ms),
    }
}

/// The shortest poll interval the worker uses.
const MIN_INTERVAL: Duration = Duration::from_millis(50);
/// Adaptive cadence (§2.7): at most 4x the idle interval.
const MAX_SLOWDOWN: u32 = 4;
/// Fast ticks needed before the slowdown is halved again.
const RECOVERY_TICKS: u32 = 20;

// ---------------------------------------------------------------------------
// The kind adapter
// ---------------------------------------------------------------------------

/// A command enum the worker can run.
pub trait DeviceCommand: Send + 'static {
    fn op_kind(&self) -> OpKind;

    /// True for a command that starts (or keeps up) motion. An abort or
    /// e-stop refuses such a command when it was authorised or queued
    /// before the abort (D19). Stops, halts, reads and plain sets never
    /// are refused this way.
    fn starts_motion(&self) -> bool {
        self.op_kind().moves_hardware()
    }
}

macro_rules! device_command {
    ($($ty:ty),* $(,)?) => {
        $(impl DeviceCommand for $ty {
            fn op_kind(&self) -> OpKind {
                <$ty>::op_kind(self)
            }
        })*
    };
}

device_command!(FilterWheelCmd, RotatorCmd, CoverCalibratorCmd, SwitchCmd);

impl DeviceCommand for MountCmd {
    fn op_kind(&self) -> OpKind {
        MountCmd::op_kind(self)
    }

    /// Tracking on drives the axes, and setting SideOfPier flips the mount
    /// (ASCOM). MoveAxis at rate 0 is a stop.
    fn starts_motion(&self) -> bool {
        match self {
            MountCmd::SetTracking(on) => *on,
            MountCmd::MoveAxis { rate_deg_s, .. } => rate_deg_s.abs() > 0.0,
            MountCmd::SetSideOfPier(_) => true,
            other => other.op_kind().moves_hardware(),
        }
    }
}

impl DeviceCommand for FocuserCmd {
    fn op_kind(&self) -> OpKind {
        FocuserCmd::op_kind(self)
    }

    /// Temperature compensation on moves the focuser by itself.
    fn starts_motion(&self) -> bool {
        match self {
            FocuserCmd::SetTempComp(on) => *on,
            other => other.op_kind().moves_hardware(),
        }
    }
}

impl DeviceCommand for GuiderCmd {
    fn op_kind(&self) -> OpKind {
        GuiderCmd::op_kind(self)
    }

    /// Guiding (started or resumed) pulses the mount; a dither moves it.
    fn starts_motion(&self) -> bool {
        matches!(
            self,
            GuiderCmd::Start { .. } | GuiderCmd::Resume | GuiderCmd::Dither { .. }
        )
    }
}

impl DeviceCommand for DomeCmd {
    fn op_kind(&self) -> OpKind {
        DomeCmd::op_kind(self)
    }

    /// A slaved dome follows the mount.
    fn starts_motion(&self) -> bool {
        match self {
            DomeCmd::SetSlaved(on) => *on,
            other => other.op_kind().moves_hardware(),
        }
    }
}

impl DeviceCommand for WeatherCmd {
    fn op_kind(&self) -> OpKind {
        OpKind::Refresh
    }

    fn starts_motion(&self) -> bool {
        false
    }
}

impl DeviceCommand for SafetyCmd {
    fn op_kind(&self) -> OpKind {
        OpKind::Refresh
    }

    fn starts_motion(&self) -> bool {
        false
    }
}

/// Daemon shutdown (§2.15) halts only these: focuser and rotator moves, a
/// jog, and an Achernar slew. Park, unpark, home, dome, cover and wheel
/// operations keep running, and tracking and PHD2 are left alone.
pub fn halts_at_shutdown(op: OpKind) -> bool {
    matches!(
        op,
        OpKind::FocuserMove
            | OpKind::RotatorMove
            | OpKind::MoveAxis
            | OpKind::Slew
            | OpKind::SlewAltAz
    )
}

/// O11 e: an operation's initiator failed in a way that does not prove the
/// device ignored it (a timeout, a transport error, or an already unknown
/// outcome), on an operation that moves hardware. Refusals that come back
/// from the device itself (invalid value, parked, not implemented) are plain
/// failures.
pub fn initiator_outcome_unknown(op: OpKind, err: &DeviceError) -> bool {
    op.moves_hardware()
        && matches!(
            err.class,
            ErrorClass::Timeout | ErrorClass::Transport | ErrorClass::OutcomeUnknown
        )
}

/// A one-shot driver action.
pub type Action<D> = Box<dyn FnOnce(&mut D) -> DeviceResult<()> + Send>;

/// What the worker does with a command.
pub enum Plan<D: ?Sized> {
    /// Runs now; conflicts with an active operation unless it preempts.
    /// Its error is the command's refusal (an HTTP error on accept).
    Instant(Action<D>),
    /// Runs now, alongside an active operation (a jog renewal, a query).
    Concurrent(Action<D>),
    /// A halt or abort command: cancels the active operation (`Aborted {
    /// by: this command }`), then runs the action. Never refused.
    Halt(Action<D>),
    /// A long operation driven by the worker.
    Operation(Box<dyn Operation<D>>),
    /// Refused without touching the device.
    Reject(DeviceError),
}

/// Events an operation emits while it runs (its `slew_started`,
/// `focuser_settled`, ...). The worker drains them after every call and
/// publishes them with the device and command ids.
#[derive(Clone, Default)]
pub struct OpEvents(Arc<Mutex<Vec<EventKind>>>);

impl std::fmt::Debug for OpEvents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("OpEvents")
            .field(&lock(&self.0).len())
            .finish()
    }
}

impl OpEvents {
    pub fn push(&self, event: EventKind) {
        lock(&self.0).push(event);
    }

    pub fn drain(&self) -> Vec<EventKind> {
        std::mem::take(&mut *lock(&self.0))
    }
}

/// What the adapter knows when planning a command.
pub struct PlanContext<'a, S> {
    pub command_id: CommandId,
    pub source: &'a CommandSource,
    pub priority: Priority,
    pub clock: &'a dyn Clock,
    /// The latest published snapshot.
    pub last: Option<&'a S>,
    /// The kind of the active operation, if any.
    pub active: Option<OpKind>,
    /// Hand this to the operation so it can emit its own events.
    pub events: OpEvents,
}

/// What the adapter knows when building a snapshot.
pub struct SnapshotContext<'a, S> {
    pub identity: &'a DeviceIdentity,
    pub clock: &'a dyn Clock,
    pub tick: u64,
    pub previous: Option<&'a S>,
    pub active_op: Option<&'a ActiveOp>,
    pub health: &'a HealthState,
}

/// The per-kind half of a worker, implemented by the device packages
/// (mount WP1A-4, focuser WP1A-5, wheel WP1A-6, ...).
pub trait WorkerKind: Send + 'static {
    type Driver: ?Sized + Device;
    type Cmd: DeviceCommand;
    type Snap: DeviceSnapshot;

    fn kind(&self) -> DeviceKind;

    fn cadence(&self) -> Cadence {
        default_cadence(self.kind())
    }

    /// False for a device with no halt (a filter wheel): a cancelled
    /// operation then resolves `Unknown` (C-4).
    fn has_halt(&self) -> bool {
        true
    }

    /// Slow members due on this tick (§2.7 "Extra members").
    fn extra_members(&mut self, _tick: u64) -> Vec<String> {
        Vec::new()
    }

    /// Capabilities for `device_connected` and `device_connections`.
    fn capabilities_json(&mut self, _drv: &mut Self::Driver) -> serde_json::Value {
        serde_json::Value::Null
    }

    fn plan(
        &mut self,
        cmd: Self::Cmd,
        drv: &mut Self::Driver,
        ctx: &PlanContext<'_, Self::Snap>,
    ) -> Plan<Self::Driver>;

    /// The typed snapshot after a successful poll. The worker fills in the
    /// connection id, publication time and active operation.
    fn snapshot(
        &mut self,
        drv: &mut Self::Driver,
        poll: &RawPoll,
        ctx: &SnapshotContext<'_, Self::Snap>,
    ) -> DeviceResult<Self::Snap>;

    /// The snapshot to publish after a failed poll: normally the previous one
    /// with every value marked stale (C-2). `None` publishes nothing.
    fn stale_snapshot(
        &mut self,
        _previous: &Self::Snap,
        _ctx: &SnapshotContext<'_, Self::Snap>,
    ) -> Option<Self::Snap> {
        None
    }

    /// Edge events between two published snapshots (Slewing, AtPark,
    /// SideOfPier, IsSafe, position, ...).
    fn edges(&mut self, _previous: &Self::Snap, _next: &Self::Snap) -> Vec<EventKind> {
        Vec::new()
    }

    /// After the single reconnect (§2.6): a relative focuser's origin becomes
    /// `unknown_after_reconnect`, a wheel's verified slot is re-read.
    fn after_reconnect(&mut self, _drv: &mut Self::Driver) {}

    /// Before a disconnect: halt focuser and rotator moves.
    fn halt_on_disconnect(&mut self, _drv: &mut Self::Driver) -> DeviceResult<()> {
        Ok(())
    }

    /// A mount slew in progress refuses a plain disconnect (§2.6).
    fn refuses_disconnect(&self, _last: Option<&Self::Snap>) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Configuration and the handle to a running worker
// ---------------------------------------------------------------------------

/// Worker behaviour.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerConfig {
    pub policy: HealthPolicy,
    /// `[equipment] connect_timeout_s`, for connect and the one reconnect.
    pub connect_timeout: Duration,
    /// Cap on one real-time queue wait.
    pub max_wait: Duration,
    /// How long a cancelled operation may keep moving before it resolves
    /// `Unknown`.
    pub cancel_confirm_timeout: Duration,
    /// Connect inside the worker before the first poll.
    pub connect_on_start: bool,
    /// Emit `device_connected` and record the `device_connections` row at
    /// start.
    pub announce_connected: bool,
    pub session_id: Option<String>,
    pub night_id: Option<String>,
    /// O11 b: refuse a motion command whose permit the gate did not issue
    /// (no abort-epoch stamp in [`WorkerDeps::aborts`]). The spine turns it
    /// on for every production worker; stops, halts, reads and sets are never
    /// refused this way, so e-stop steps still run.
    pub require_gate_permits: bool,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            policy: HealthPolicy::default(),
            connect_timeout: Duration::from_secs(30),
            max_wait: Duration::from_millis(100),
            cancel_confirm_timeout: Duration::from_secs(10),
            connect_on_start: false,
            announce_connected: true,
            session_id: None,
            night_id: None,
            require_gate_permits: false,
        }
    }
}

/// Where a worker publishes.
pub struct WorkerDeps<S> {
    pub clock: Arc<dyn Clock>,
    pub events: Arc<dyn EventSink>,
    pub recorder: Arc<dyn RecorderSink>,
    /// The hub ring for this kind (`EquipmentHub::attach`).
    pub ring: Option<Arc<Ring<S>>>,
    pub watchdog: Option<Arc<Watchdog>>,
    /// The gate's abort epochs (`CommandGate::abort_epochs`). With them, a
    /// motion command authorised before an abort or e-stop is refused even
    /// when it was already queued (D19). Without them, only the motion
    /// commands queued when the cancel is handled are refused.
    pub aborts: Option<Arc<AbortEpochs>>,
}

#[derive(Debug, Default)]
struct WorkerControl {
    stop: AtomicBool,
}

/// The running worker. Dropping it asks the worker to stop (as at
/// shutdown) without waiting.
pub struct WorkerHandle {
    thread: Option<JoinHandle<()>>,
    name: String,
    control: Arc<WorkerControl>,
    connected: Mutex<Option<oneshot::Receiver<DeviceResult<ConnectReport>>>>,
}

impl std::fmt::Debug for WorkerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerHandle")
            .field("name", &self.name)
            .field("finished", &self.is_finished())
            .finish()
    }
}

impl WorkerHandle {
    /// The thread name, `eq-{kind}-{connection id prefix}`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Daemon shutdown (§2.15): halt moves, stop jogs, abort Achernar
    /// slews, never park, close the connection record as `shutdown`.
    pub fn stop(&self) {
        self.control.stop.store(true, Ordering::Release);
    }

    pub fn is_finished(&self) -> bool {
        self.thread.iter().all(JoinHandle::is_finished)
    }

    /// Wait (real time) for the thread to end. True when it ended.
    pub fn join_timeout(mut self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while !self.is_finished() {
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        true
    }

    /// The connect result, when the worker connects at start.
    pub async fn connected(&self) -> DeviceResult<ConnectReport> {
        let rx = lock(&self.connected).take();
        match rx {
            Some(rx) => rx.await.unwrap_or_else(|_| {
                Err(DeviceError::disconnected(
                    "the worker stopped before reporting the connect result",
                ))
            }),
            None => Ok(ConnectReport::default()),
        }
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        self.control.stop.store(true, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// State shared with the watchdog
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct SharedFlags {
    lost_announced: bool,
    unknown_announced: Option<CommandId>,
    /// The operation whose abort path already ran (once per operation).
    abort_ran: Option<CommandId>,
    finished: bool,
}

/// The active operation's ticket. It lives here, not on the worker thread,
/// so the watchdog can resolve it while the worker is stuck in a driver
/// call; whichever side takes it first resolves it, exactly once.
struct PendingDone {
    command_id: CommandId,
    op: OpKind,
    started_mono_ns: u64,
    tx: oneshot::Sender<OpOutcome>,
}

struct WorkerShared<C, S> {
    end: Arc<WorkerEnd<C, S>>,
    interval_ns: AtomicU64,
    policy: HealthPolicy,
    clock: Arc<dyn Clock>,
    events: Arc<dyn EventSink>,
    abort_path: Option<Arc<dyn AbortPath>>,
    done: Mutex<Option<PendingDone>>,
    flags: Mutex<SharedFlags>,
}

impl<C, S: DeviceSnapshot> WorkerShared<C, S> {
    fn emit(&self, event: EventKind, command: Option<CommandId>) {
        let identity = &self.end.identity;
        let mut event = EquipmentEvent::new(event, self.clock.as_ref())
            .for_device(identity.kind, identity.connection_id);
        if let Some(command) = command {
            event = event.with_command(command);
        }
        self.events.emit(event);
    }

    /// Announce `Lost` once, from whichever side saw it first.
    fn announce_lost(&self, class: FailureClass, reason: Option<String>) -> bool {
        let first = {
            let mut flags = lock(&self.flags);
            !std::mem::replace(&mut flags.lost_announced, true)
        };
        if first {
            self.emit(EventKind::DeviceLost { class, reason }, None);
        }
        first
    }

    /// Announce an unknown outcome once per command.
    fn announce_unknown(&self, command_id: CommandId, op: OpKind, reason: String) {
        let first = {
            let mut flags = lock(&self.flags);
            if flags.unknown_announced == Some(command_id) {
                false
            } else {
                flags.unknown_announced = Some(command_id);
                true
            }
        };
        if first {
            self.emit(EventKind::OutcomeUnknown { op, reason }, Some(command_id));
        }
    }

    /// Claim the abort path for `command_id`: true the first time only.
    fn claim_abort(&self, command_id: Option<CommandId>) -> bool {
        let Some(command_id) = command_id else {
            return true;
        };
        let mut flags = lock(&self.flags);
        if flags.abort_ran == Some(command_id) {
            false
        } else {
            flags.abort_ran = Some(command_id);
            true
        }
    }

    /// Resolve the active ticket with `outcome`, unless the other side
    /// already did. Emits `operation_ended` (and `outcome_unknown` for an
    /// unknown outcome) with it.
    fn resolve_done(&self, outcome: &OpOutcome) -> bool {
        let Some(pending) = lock(&self.done).take() else {
            return false;
        };
        let duration_ms = self
            .clock
            .now_mono_ns()
            .saturating_sub(pending.started_mono_ns)
            / 1_000_000;
        let _ = pending.tx.send(outcome.clone());
        self.emit(
            EventKind::OperationEnded {
                op: pending.op,
                outcome: outcome.clone(),
                duration_ms,
            },
            Some(pending.command_id),
        );
        if let OpOutcome::Unknown(err) = outcome {
            self.announce_unknown(pending.command_id, pending.op, err.message.clone());
        }
        true
    }

    /// The device was lost while the worker may be stuck inside the driver
    /// (§2.15, D4): start the abort path on a blocking thread now, and
    /// resolve the operation `Unknown` without waiting for the hung call.
    fn abandon_active(&self, reason: &str) {
        let command_id = lock(&self.done).as_ref().map(|p| p.command_id);
        let Some(command_id) = command_id else {
            return;
        };
        if let Some(path) = self.abort_path.clone() {
            if self.claim_abort(Some(command_id)) {
                run_detached("eq-abort", move || {
                    match catch_unwind(AssertUnwindSafe(|| path.abort())) {
                        Ok(Ok(())) => {}
                        Ok(Err(err)) => tracing::warn!("abort path after loss failed: {err}"),
                        Err(_) => tracing::warn!("abort path after loss panicked"),
                    }
                });
            }
        }
        self.resolve_done(&OpOutcome::Unknown(DeviceError::disconnected(reason)));
    }
}

impl<C: Send + 'static, S: DeviceSnapshot> Watched for WorkerShared<C, S> {
    fn check(&self, now_mono_ns: u64) -> bool {
        if lock(&self.flags).finished {
            return false;
        }
        let interval = Duration::from_nanos(self.interval_ns.load(Ordering::Acquire));
        let mut change = HealthChange::Unchanged;
        self.end.health.send_if_modified(|health| {
            change = on_stale(health, interval, &self.policy, now_mono_ns);
            change != HealthChange::Unchanged
        });
        match change {
            HealthChange::Degraded { class, reason } => {
                self.emit(EventKind::DeviceDegraded { class, reason }, None);
            }
            HealthChange::Lost { class, reason } => {
                self.announce_lost(class, reason.clone());
                // The worker may be stuck inside the driver: run the abort
                // path and resolve the operation unknown now (§2.15), which
                // raises its outcome_unknown block.
                let reason = reason.unwrap_or_else(|| "device lost during the operation".into());
                if let Some(op) = self.end.current().active_op().cloned() {
                    self.announce_unknown(op.command_id, op.kind, reason.clone());
                }
                self.abandon_active(&reason);
            }
            _ => {}
        }
        true
    }
}

// ---------------------------------------------------------------------------
// The worker
// ---------------------------------------------------------------------------

struct Active<D: ?Sized> {
    op: Box<dyn Operation<D>>,
    info: ActiveOp,
    permit: Permit,
    events: OpEvents,
    settle_until: Option<u64>,
    /// `Some` once cancelled: who cancelled it, and since when.
    cancel: Option<(Option<CommandId>, u64)>,
    hint: Option<Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Running,
    /// Lost (sticky for the connection); the driver is kept for a polite
    /// disconnect.
    Lost,
    /// The driver panicked and was dropped.
    Dead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    Continue,
    Exit,
}

struct Worker<K: WorkerKind> {
    kind: K,
    kind_id: DeviceKind,
    drv: Option<Box<K::Driver>>,
    end: Arc<WorkerEnd<K::Cmd, K::Snap>>,
    shared: Arc<WorkerShared<K::Cmd, K::Snap>>,
    config: WorkerConfig,
    clock: Arc<dyn Clock>,
    recorder: Arc<dyn RecorderSink>,
    ring: Option<Arc<Ring<K::Snap>>>,
    control: Arc<WorkerControl>,
    abort_path: Option<Arc<dyn AbortPath>>,
    aborts: Option<Arc<AbortEpochs>>,
    /// The latest abort this worker handled (`Aborted { by }` for commands
    /// it refuses as authorised before it).
    last_abort: Option<CommandId>,
    cadence: Cadence,
    tick: u64,
    next_poll_at: u64,
    slowdown: u32,
    fast_ticks: u32,
    active: Option<Active<K::Driver>>,
    last: Option<Arc<K::Snap>>,
    mode: Mode,
    reconnect_used: bool,
    connection_closed: bool,
    connected_tx: Option<oneshot::Sender<DeviceResult<ConnectReport>>>,
}

/// Spawn the worker thread for one connected device.
pub fn spawn_worker<K: WorkerKind>(
    kind: K,
    driver: Box<K::Driver>,
    end: WorkerEnd<K::Cmd, K::Snap>,
    config: WorkerConfig,
    deps: WorkerDeps<K::Snap>,
) -> std::io::Result<WorkerHandle> {
    let identity = Arc::clone(&end.identity);
    let id = identity.connection_id.to_string();
    let name = format!("eq-{}-{}", kind_name(identity.kind), &id[..id.len().min(8)]);
    let cadence = kind.cadence();
    let end = Arc::new(end);
    let abort_path = driver.abort_path();
    let shared = Arc::new(WorkerShared {
        end: Arc::clone(&end),
        interval_ns: AtomicU64::new(ns(cadence.idle)),
        policy: config.policy,
        clock: Arc::clone(&deps.clock),
        events: Arc::clone(&deps.events),
        abort_path: abort_path.clone(),
        done: Mutex::new(None),
        flags: Mutex::new(SharedFlags::default()),
    });
    if let Some(watchdog) = &deps.watchdog {
        let weak: Weak<WorkerShared<K::Cmd, K::Snap>> = Arc::downgrade(&shared);
        watchdog.register(weak as Weak<dyn Watched>);
    }
    let control = Arc::new(WorkerControl::default());
    let (connected_tx, connected_rx) = oneshot::channel();
    let worker = Worker {
        kind_id: identity.kind,
        kind,
        drv: Some(driver),
        end,
        shared,
        clock: deps.clock,
        recorder: deps.recorder,
        ring: deps.ring,
        control: Arc::clone(&control),
        abort_path,
        aborts: deps.aborts,
        last_abort: None,
        cadence,
        tick: 0,
        next_poll_at: 0,
        slowdown: 1,
        fast_ticks: 0,
        active: None,
        last: None,
        mode: Mode::Running,
        reconnect_used: false,
        connection_closed: false,
        connected_tx: Some(connected_tx),
        config,
    };
    let thread = std::thread::Builder::new()
        .name(name.clone())
        .spawn(move || worker.run())?;
    Ok(WorkerHandle {
        thread: Some(thread),
        name,
        control,
        connected: Mutex::new(Some(connected_rx)),
    })
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "driver panicked".into()
    }
}

impl<K: WorkerKind> Worker<K> {
    fn now(&self) -> (DateTime<Utc>, u64) {
        (self.clock.now_utc(), self.clock.now_mono_ns())
    }

    fn emit(&self, event: EventKind, command: Option<CommandId>) {
        self.shared.emit(event, command);
    }

    fn update_health(&self, f: impl FnOnce(&mut HealthState) -> HealthChange) -> HealthChange {
        let mut change = HealthChange::Unchanged;
        let mut f = Some(f);
        self.end.health.send_if_modified(|health| {
            if let Some(f) = f.take() {
                change = f(health);
            }
            change != HealthChange::Unchanged
        });
        change
    }

    fn current_interval(&self) -> Duration {
        let interval = match &self.active {
            Some(active) => {
                let hint = active.hint.unwrap_or(self.cadence.active);
                self.cadence.active.min(hint)
            }
            None => self.cadence.idle.saturating_mul(self.slowdown),
        };
        interval.max(MIN_INTERVAL)
    }

    fn close_connection(&mut self, end_class: ConnectionEnd) {
        if self.connection_closed {
            return;
        }
        self.connection_closed = true;
        self.recorder.record(GroundTruthRecord::ConnectionClosed {
            connection_id: self.end.identity.connection_id,
            disconnected_at: self.clock.now_utc(),
            end_class,
        });
    }

    // -- the loop ------------------------------------------------------------------

    fn run(mut self) {
        if self.config.connect_on_start && !self.connect_phase() {
            lock(&self.shared.flags).finished = true;
            return;
        }
        if let Some(tx) = self.connected_tx.take() {
            let _ = tx.send(Ok(ConnectReport::default()));
        }
        if self.config.announce_connected {
            self.announce_connected();
        }
        self.poll_now();
        loop {
            if self.control.stop.load(Ordering::Acquire) {
                self.shutdown();
                break;
            }
            let wait = self.wait_budget();
            match self.end.commands.pop_timeout(wait) {
                Some(envelope) => {
                    if self.handle_envelope(envelope) == Flow::Exit {
                        break;
                    }
                }
                None if self.end.commands.is_closed() => {
                    self.shutdown();
                    break;
                }
                None => {}
            }
            self.sync_lost();
            if self.mode == Mode::Running && self.clock.now_mono_ns() >= self.next_poll_at {
                self.poll_now();
            }
        }
        lock(&self.shared.flags).finished = true;
    }

    fn wait_budget(&self) -> Duration {
        if self.mode != Mode::Running {
            return self.config.max_wait;
        }
        let now = self.clock.now_mono_ns();
        Duration::from_nanos(self.next_poll_at.saturating_sub(now)).min(self.config.max_wait)
    }

    fn connect_phase(&mut self) -> bool {
        let now = self.clock.now_mono_ns();
        self.end.set_health(HealthState::connecting(now));
        let timeout = self.config.connect_timeout;
        let result = match self.drv.as_mut() {
            Some(drv) => catch_unwind(AssertUnwindSafe(|| drv.connect(timeout))),
            None => Ok(Err(DeviceError::driver("no driver"))),
        };
        let result = match result {
            Ok(result) => result,
            Err(panic) => Err(DeviceError::driver(format!(
                "driver panicked while connecting: {}",
                panic_message(panic.as_ref())
            ))),
        };
        match result {
            Ok(report) => {
                if let Some(tx) = self.connected_tx.take() {
                    let _ = tx.send(Ok(report));
                }
                true
            }
            Err(err) => {
                tracing::warn!(device = %self.end.identity.name, "connect failed: {err}");
                if let Some(tx) = self.connected_tx.take() {
                    let _ = tx.send(Err(err));
                }
                self.end.retire(self.clock.now_mono_ns());
                false
            }
        }
    }

    fn announce_connected(&mut self) {
        let capabilities = match self.drv.as_mut() {
            Some(drv) => {
                let kind = &mut self.kind;
                catch_unwind(AssertUnwindSafe(|| kind.capabilities_json(drv)))
                    .unwrap_or(serde_json::Value::Null)
            }
            None => serde_json::Value::Null,
        };
        let identity = (*self.end.identity).clone();
        self.emit(
            EventKind::DeviceConnected {
                identity: Box::new(identity.clone()),
                capabilities: capabilities.clone(),
            },
            None,
        );
        self.recorder
            .record(GroundTruthRecord::ConnectionOpened(Box::new(
                DeviceConnectionRecord {
                    identity,
                    capabilities,
                    connected_at: self.clock.now_utc(),
                    session_id: self.config.session_id.clone(),
                    night_id: self.config.night_id.clone(),
                },
            )));
    }

    /// The watchdog may have declared the device lost while this thread was
    /// inside the driver.
    fn sync_lost(&mut self) {
        if self.mode == Mode::Running && self.end.health.borrow().state == LinkState::Lost {
            let health = self.end.health.borrow().clone();
            self.enter_lost(
                health.failure_class.unwrap_or(FailureClass::Timeout),
                health.reason,
            );
        }
    }

    // -- polling -------------------------------------------------------------------

    fn snapshot_ctx_parts(&self) -> (Option<ActiveOp>, HealthState) {
        (
            self.active.as_ref().map(|a| a.info.clone()),
            self.end.health.borrow().clone(),
        )
    }

    fn poll_now(&mut self) {
        if self.mode != Mode::Running {
            return;
        }
        let started = self.clock.now_mono_ns();
        self.tick += 1;
        let interval = self.current_interval();
        self.shared
            .interval_ns
            .store(ns(interval), Ordering::Release);
        let plan = PollPlan {
            tick: self.tick,
            interval,
            active: self.active.is_some(),
            prefer_device_state: true,
            extra_members: self.kind.extra_members(self.tick),
        };
        let (active_op, health) = self.snapshot_ctx_parts();
        let previous = self.last.clone();
        let identity = Arc::clone(&self.end.identity);
        let tick = self.tick;
        let clock = Arc::clone(&self.clock);
        let Some(drv) = self.drv.as_mut() else {
            return;
        };
        let kind = &mut self.kind;
        let result = catch_unwind(AssertUnwindSafe(|| {
            let raw = drv.poll(&plan)?;
            let ctx = SnapshotContext {
                identity: &identity,
                clock: clock.as_ref(),
                tick,
                previous: previous.as_deref(),
                active_op: active_op.as_ref(),
                health: &health,
            };
            let snap = kind.snapshot(drv, &raw, &ctx)?;
            Ok::<_, DeviceError>((raw, snap))
        }));
        let result = match result {
            Ok(result) => result,
            Err(panic) => {
                self.on_panic(&panic_message(panic.as_ref()));
                return;
            }
        };
        // The watchdog may have declared the device lost while this call
        // blocked: then the result is not trusted to step the operation.
        self.sync_lost();
        if self.mode != Mode::Running {
            return;
        }
        match result {
            Err(err) => self.on_poll_error(err),
            Ok((raw, snap)) => self.on_poll_ok(&raw, snap),
        }
        let finished = self.clock.now_mono_ns();
        self.adapt_cadence(started, finished, interval);
        let mut next = started
            .saturating_add(ns(self.current_interval()))
            .max(finished);
        if let Some(until) = self.active.as_ref().and_then(|a| a.settle_until) {
            next = next.min(until.max(finished));
        }
        self.next_poll_at = next;
    }

    fn adapt_cadence(&mut self, started: u64, finished: u64, interval: Duration) {
        if self.active.is_some() {
            return;
        }
        let took = finished.saturating_sub(started);
        if took > ns(interval) / 2 && self.slowdown < MAX_SLOWDOWN {
            self.slowdown = (self.slowdown * 2).min(MAX_SLOWDOWN);
            self.fast_ticks = 0;
            let reduced = self.cadence.idle.saturating_mul(self.slowdown);
            self.emit(
                EventKind::PollCadenceReduced {
                    interval_ms: u64::try_from(reduced.as_millis()).unwrap_or(u64::MAX),
                    factor: self.slowdown,
                },
                None,
            );
        } else if took < ns(self.cadence.idle) / 4 && self.slowdown > 1 {
            self.fast_ticks += 1;
            if self.fast_ticks >= RECOVERY_TICKS {
                self.slowdown /= 2;
                self.fast_ticks = 0;
            }
        }
    }

    fn publish(&mut self, mut snap: K::Snap) -> Arc<K::Snap> {
        let (utc, mono) = self.now();
        snap.set_connection_id(self.end.identity.connection_id);
        snap.set_published(utc, mono);
        snap.set_active_op(self.active.as_ref().map(|a| a.info.clone()));
        let snap = Arc::new(snap);
        self.end.publish_arc(Arc::clone(&snap));
        if let Some(ring) = &self.ring {
            ring.push(Arc::clone(&snap));
        }
        snap
    }

    /// Re-publish the latest snapshot with the current operation state.
    fn republish(&mut self) {
        if let Some(last) = self.last.clone() {
            let snap = self.publish((*last).clone());
            self.last = Some(snap);
        }
    }

    fn on_poll_ok(&mut self, raw: &RawPoll, snap: K::Snap) {
        let (utc, mono) = self.now();
        self.update_health(|h| on_poll_ok(h, utc, mono));
        self.reconnect_used = false;
        let previous = self.last.clone();
        let snap = self.publish(snap);
        if let Some(previous) = previous {
            for event in self.kind.edges(&previous, &snap) {
                let mut event = EquipmentEvent::new(event, self.clock.as_ref())
                    .for_device(self.kind_id, self.end.identity.connection_id);
                if let Some(device_time) = raw.device_time {
                    event = event.with_device_time(device_time);
                }
                self.shared.events.emit(event);
            }
        }
        self.last = Some(Arc::clone(&snap));
        self.after_poll(Some(&snap));
    }

    fn on_poll_error(&mut self, err: DeviceError) {
        let mono = self.clock.now_mono_ns();
        let interval = self.current_interval();
        let policy = self.config.policy;
        let change = self.update_health(|h| on_poll_failed(h, &err, &policy, interval, mono));
        match change {
            HealthChange::Degraded { class, reason } => {
                self.emit(EventKind::DeviceDegraded { class, reason }, None);
            }
            HealthChange::LostCandidate { class, reason } => {
                if !self.reconnect_used && self.try_reconnect(&err) {
                    return;
                }
                self.enter_lost(class, reason);
                return;
            }
            _ => {}
        }
        self.publish_stale();
        // Time passes even while polls fail.
        self.after_poll(None);
    }

    /// Publish the adapter's stale copy of the latest snapshot (C-2, D5):
    /// after a failed poll, and when the device is lost, so its last values
    /// are never presented as fresh.
    fn publish_stale(&mut self) {
        let Some(previous) = self.last.clone() else {
            return;
        };
        let (active_op, health) = self.snapshot_ctx_parts();
        let ctx = SnapshotContext {
            identity: &self.end.identity,
            clock: self.clock.as_ref(),
            tick: self.tick,
            previous: Some(previous.as_ref()),
            active_op: active_op.as_ref(),
            health: &health,
        };
        let kind = &mut self.kind;
        let stale = match catch_unwind(AssertUnwindSafe(|| kind.stale_snapshot(&previous, &ctx))) {
            Ok(stale) => stale,
            Err(panic) => {
                tracing::warn!(
                    device = %self.end.identity.name,
                    "stale snapshot panicked: {}",
                    panic_message(panic.as_ref())
                );
                None
            }
        };
        if let Some(stale) = stale {
            let snap = self.publish(stale);
            self.last = Some(snap);
        }
    }

    fn try_reconnect(&mut self, cause: &DeviceError) -> bool {
        self.reconnect_used = true;
        let timeout = self.config.connect_timeout;
        let Some(drv) = self.drv.as_mut() else {
            return false;
        };
        let result = catch_unwind(AssertUnwindSafe(|| drv.connect(timeout)));
        match result {
            Ok(Ok(_)) => {
                let kind = &mut self.kind;
                let _ = catch_unwind(AssertUnwindSafe(|| kind.after_reconnect(drv)));
                let (utc, mono) = self.now();
                self.update_health(|h| {
                    h.consecutive_failures = 0;
                    h.last_ok_at = Some(utc);
                    h.last_ok_mono_ns = Some(mono);
                    h.state = LinkState::Ready;
                    h.failure_class = None;
                    h.reason = None;
                    h.changed_mono_ns = mono;
                    HealthChange::Ready {
                        from: LinkState::Degraded,
                    }
                });
                self.emit(
                    EventKind::DeviceReconnected {
                        reason: Some(cause.to_string()),
                    },
                    None,
                );
                self.next_poll_at = self.clock.now_mono_ns();
                true
            }
            Ok(Err(err)) => {
                tracing::warn!(device = %self.end.identity.name, "reconnect failed: {err}");
                false
            }
            Err(panic) => {
                self.on_panic(&panic_message(panic.as_ref()));
                true
            }
        }
    }

    /// Run the abort path on this thread (the device is lost or the driver
    /// panicked; there is nothing else to wait for). Skipped when the
    /// watchdog already ran it for the active operation.
    fn best_effort_abort(&self) {
        let active = self.active.as_ref().map(|a| a.info.command_id);
        if !self.shared.claim_abort(active) {
            return;
        }
        if let Some(path) = &self.abort_path {
            match catch_unwind(AssertUnwindSafe(|| path.abort())) {
                Ok(Ok(())) => {}
                Ok(Err(err)) => tracing::warn!("abort path failed: {err}"),
                Err(_) => tracing::warn!("abort path panicked"),
            }
        }
    }

    fn enter_lost(&mut self, class: FailureClass, reason: Option<String>) {
        if self.mode != Mode::Running {
            return;
        }
        self.mode = Mode::Lost;
        let mono = self.clock.now_mono_ns();
        let lost_reason = reason.clone();
        self.update_health(|h| {
            if h.state == LinkState::Lost {
                return HealthChange::Unchanged;
            }
            mark_lost(h, class, lost_reason, mono);
            HealthChange::Lost {
                class,
                reason: None,
            }
        });
        self.shared.announce_lost(class, reason.clone());
        self.close_connection(ConnectionEnd::Lost);
        // Nothing more will be read: the last values are stale from now on.
        self.publish_stale();
        if self.active.is_some() {
            self.best_effort_abort();
            let message =
                reason.unwrap_or_else(|| "the device was lost during the operation".into());
            self.resolve(OpOutcome::Unknown(DeviceError::disconnected(message)));
        }
    }

    fn on_panic(&mut self, message: &str) {
        tracing::error!(device = %self.end.identity.name, "driver panic contained: {message}");
        self.best_effort_abort();
        if self.active.is_some() {
            self.resolve(OpOutcome::Unknown(DeviceError::driver(format!(
                "the driver panicked during the operation: {message}"
            ))));
        }
        let mono = self.clock.now_mono_ns();
        let reason = Some(format!("driver panic: {message}"));
        let lost_reason = reason.clone();
        self.update_health(|h| {
            mark_lost(h, FailureClass::DriverPanic, lost_reason, mono);
            HealthChange::Lost {
                class: FailureClass::DriverPanic,
                reason: None,
            }
        });
        self.shared.announce_lost(FailureClass::DriverPanic, reason);
        self.close_connection(ConnectionEnd::Failed);
        self.publish_stale();
        // Dropping the driver may panic too.
        let drv = self.drv.take();
        let _ = catch_unwind(AssertUnwindSafe(move || drop(drv)));
        self.mode = Mode::Dead;
    }

    // -- operations ------------------------------------------------------------------

    fn drain_op_events(&mut self) {
        let Some(active) = &self.active else {
            return;
        };
        let command_id = active.info.command_id;
        for event in active.events.drain() {
            self.emit(event, Some(command_id));
        }
    }

    fn resolve(&mut self, outcome: OpOutcome) {
        self.drain_op_events();
        let Some(active) = self.active.take() else {
            return;
        };
        // The watchdog may already have resolved it (device lost while this
        // thread was inside the driver); then it is not resolved twice.
        self.shared.resolve_done(&outcome);
        // Releases the domains (on every path, including panics).
        drop(active);
        self.republish();
    }

    fn after_poll(&mut self, snap: Option<&Arc<K::Snap>>) {
        if self.mode != Mode::Running || self.active.is_none() {
            return;
        }
        let now = self.clock.now_mono_ns();
        let (cancel, deadline, settle_until) = {
            let active = self.active.as_ref().expect("checked above");
            (
                active.cancel,
                active.info.deadline_mono_ns,
                active.settle_until,
            )
        };
        if let Some((by, since)) = cancel {
            // Only a poll that read "not moving" confirms the stop; a poll
            // that could not read motion leaves it to the timeout (§2.5).
            let stopped = snap.map(|s| s.device_moving() == Some(false));
            match stopped {
                Some(true) => self.resolve(OpOutcome::Aborted { by }),
                _ if now.saturating_sub(since) >= ns(self.config.cancel_confirm_timeout) => {
                    self.resolve(OpOutcome::Unknown(DeviceError::outcome_unknown(
                        "motion did not stop after the cancel",
                    )));
                }
                _ => {}
            }
            return;
        }
        if now >= deadline {
            self.cancel_op();
            if self.kind.has_halt() {
                self.resolve(OpOutcome::TimedOut);
            } else {
                // Nothing stopped it (C-4): its final state is unknown, which
                // blocks automatic resumption (v6 §8.2.1).
                self.resolve(OpOutcome::Unknown(DeviceError::outcome_unknown(
                    "the deadline passed on a device with no halt; its final position is unknown",
                )));
            }
            return;
        }
        if snap.is_none() {
            return;
        }
        if let Some(until) = settle_until {
            if now < until {
                return;
            }
            if let Some(active) = self.active.as_mut() {
                active.settle_until = None;
                active.info.phase = OpPhase::Verifying;
            }
        }
        let clock = Arc::clone(&self.clock);
        let (Some(drv), Some(active)) = (self.drv.as_mut(), self.active.as_mut()) else {
            return;
        };
        let op = &mut active.op;
        let progress = catch_unwind(AssertUnwindSafe(|| op.step(drv, clock.as_ref())));
        match progress {
            Err(panic) => self.on_panic(&panic_message(panic.as_ref())),
            Ok(OpProgress::Continue { next_poll }) => {
                if let Some(active) = self.active.as_mut() {
                    if active.info.phase != OpPhase::Verifying {
                        active.info.phase = OpPhase::InProgress;
                    }
                    active.hint = Some(next_poll);
                }
                self.drain_op_events();
            }
            Ok(OpProgress::Settling { until_mono_ns }) => {
                if let Some(active) = self.active.as_mut() {
                    active.info.phase = OpPhase::Settling;
                    active.settle_until = Some(until_mono_ns);
                }
                self.drain_op_events();
            }
            Ok(OpProgress::Done(outcome)) => self.resolve(outcome),
        }
    }

    /// `Operation::cancel` (halts the hardware). Returns false on a panic.
    fn cancel_op(&mut self) -> bool {
        let (Some(drv), Some(active)) = (self.drv.as_mut(), self.active.as_mut()) else {
            return true;
        };
        let op = &mut active.op;
        match catch_unwind(AssertUnwindSafe(|| op.cancel(drv))) {
            Ok(Ok(())) => {
                self.drain_op_events();
                true
            }
            Ok(Err(err)) => {
                // The halt itself failed; the outcome is decided by the
                // confirmation poll (or its timeout).
                tracing::warn!(device = %self.end.identity.name, "cancel failed: {err}");
                self.drain_op_events();
                true
            }
            Err(panic) => {
                self.on_panic(&panic_message(panic.as_ref()));
                false
            }
        }
    }

    /// Cancel the active operation for `by`. A preemption resolves at once
    /// (the preempting command starts next); an abort waits for a poll that
    /// confirms motion stopped.
    fn cancel_active(&mut self, by: Option<CommandId>, wait_for_stop: bool) {
        if self.active.is_none() {
            return;
        }
        if !self.cancel_op() || self.active.is_none() {
            return;
        }
        if !self.kind.has_halt() {
            self.resolve(OpOutcome::Unknown(DeviceError::outcome_unknown(
                "the device has no halt; its final position is unknown",
            )));
            return;
        }
        if !wait_for_stop {
            self.resolve(OpOutcome::Aborted { by });
            return;
        }
        let now = self.clock.now_mono_ns();
        if let Some(active) = self.active.as_mut() {
            active.cancel = Some((by, now));
            active.info.phase = OpPhase::Cancelling;
        }
        self.republish();
        // Confirm promptly.
        self.next_poll_at = now;
    }

    // -- envelopes -------------------------------------------------------------------

    fn handle_envelope(&mut self, envelope: Envelope<K::Cmd>) -> Flow {
        let command_id = envelope.command_id;
        match envelope.body {
            EnvelopeBody::Command {
                cmd,
                permit,
                accepted,
                done,
            } => {
                self.on_command(command_id, cmd, permit, accepted, done);
                Flow::Continue
            }
            EnvelopeBody::Cancel { by } => {
                self.last_abort = Some(by);
                if self.mode == Mode::Running {
                    let already = self
                        .active
                        .as_ref()
                        .is_some_and(|active| active.cancel.is_some());
                    if !already {
                        self.cancel_active(Some(by), true);
                    }
                }
                self.refuse_queued_motion();
                Flow::Continue
            }
            EnvelopeBody::Disconnect { force, done } => self.on_disconnect(command_id, force, done),
        }
    }

    fn refuse(&self, accepted: oneshot::Sender<DeviceResult<()>>, err: DeviceError) {
        let _ = accepted.send(Err(err));
    }

    /// The gate stamped `permit` before an abort of this kind (D19): the
    /// command was authorised before the abort and must not start.
    fn authorised_before_abort(&self, permit: &Permit) -> bool {
        let Some(aborts) = &self.aborts else {
            return false;
        };
        aborts
            .issued_epoch(permit.command_id(), self.kind_id)
            .is_some_and(|issued| issued < aborts.epoch(self.kind_id))
    }

    /// O11 b: the gate issued `permit` (its abort-epoch stamp is live). Only
    /// checked when [`WorkerConfig::require_gate_permits`] is set and the gate's
    /// epochs are installed.
    fn issued_by_gate(&self, permit: &Permit) -> bool {
        if !self.config.require_gate_permits {
            return true;
        }
        let Some(aborts) = &self.aborts else {
            return true;
        };
        aborts
            .issued_epoch(permit.command_id(), self.kind_id)
            .is_some()
    }

    /// The gate stamped `permit` with the current abort epoch: authorised
    /// after every abort of this kind so far.
    fn authorised_after_abort(&self, permit: &Permit) -> bool {
        let Some(aborts) = &self.aborts else {
            return false;
        };
        aborts
            .issued_epoch(permit.command_id(), self.kind_id)
            .is_some_and(|issued| issued >= aborts.epoch(self.kind_id))
    }

    /// Refuse a motion command that an abort superseded: it never starts
    /// and is recorded as `Aborted { by }`.
    fn refuse_superseded(
        &self,
        command_id: CommandId,
        op: OpKind,
        source: CommandSource,
        accepted: oneshot::Sender<DeviceResult<()>>,
    ) {
        let by = self.last_abort;
        self.emit(
            EventKind::OperationCommanded { op, source },
            Some(command_id),
        );
        self.emit_ended(op, command_id, OpOutcome::Aborted { by }, 0);
        let message = match by {
            Some(by) => format!("abort {by} was issued after this command; it was not started"),
            None => "an abort was issued after this command; it was not started".into(),
        };
        self.refuse(accepted, DeviceError::cancelled(message));
    }

    /// An abort also stops what waits behind it (D19). Every motion command
    /// still queued when the cancel is handled was queued before the abort,
    /// or raced it, so it is refused, unless the gate stamped its permit
    /// with the current abort epoch (authorised after the abort). Stops,
    /// halts, reads and the e-stop's own tracking-off stay queued, in order.
    fn refuse_queued_motion(&mut self) {
        let queued = self.end.commands.len();
        let mut kept = Vec::new();
        for _ in 0..queued {
            let Some(Envelope { command_id, body }) = self.end.commands.try_pop() else {
                break;
            };
            match body {
                EnvelopeBody::Command {
                    cmd,
                    permit,
                    accepted,
                    done,
                } if cmd.starts_motion() && !self.authorised_after_abort(&permit) => {
                    let source = permit.source().clone();
                    // Released before the caller hears of the refusal.
                    drop(permit);
                    drop(done);
                    self.refuse_superseded(command_id, cmd.op_kind(), source, accepted);
                }
                body => kept.push(Envelope { command_id, body }),
            }
        }
        // Back at the head, in their original order. A queue closed in the
        // meantime drops them, which releases their permits and fails their
        // callers.
        for envelope in kept.into_iter().rev() {
            let _ = self.end.commands.push_front(envelope);
        }
    }

    fn on_command(
        &mut self,
        command_id: CommandId,
        cmd: K::Cmd,
        permit: Permit,
        accepted: oneshot::Sender<DeviceResult<()>>,
        done: oneshot::Sender<OpOutcome>,
    ) {
        match self.mode {
            Mode::Running => {}
            Mode::Lost => {
                return self.refuse(
                    accepted,
                    DeviceError::disconnected(format!(
                        "{} {} is lost; reconnect before issuing commands",
                        kind_name(self.kind_id),
                        self.end.identity.name
                    )),
                )
            }
            Mode::Dead => {
                return self.refuse(
                    accepted,
                    DeviceError::disconnected("the driver panicked and was stopped"),
                )
            }
        }
        let op_kind = cmd.op_kind();
        if cmd.starts_motion() && self.authorised_before_abort(&permit) {
            let source = permit.source().clone();
            drop(permit);
            return self.refuse_superseded(command_id, op_kind, source, accepted);
        }
        if cmd.starts_motion() && !self.issued_by_gate(&permit) {
            let err = DeviceError::invalid_operation(
                "the command's permit was not issued by the command gate; motion needs a \
                 gate permit (CommandGate::authorize or child_permit)",
            );
            self.emit_ended(op_kind, command_id, OpOutcome::Failed(err.clone()), 0);
            return self.refuse(accepted, err);
        }
        self.emit(
            EventKind::OperationCommanded {
                op: op_kind,
                source: permit.source().clone(),
            },
            Some(command_id),
        );
        let events = OpEvents::default();
        let last = self.last.clone();
        let source = permit.source().clone();
        let ctx = PlanContext {
            command_id,
            source: &source,
            priority: permit.priority(),
            clock: self.clock.as_ref(),
            last: last.as_deref(),
            active: self.active.as_ref().map(|a| a.info.kind),
            events: events.clone(),
        };
        let Some(drv) = self.drv.as_mut() else {
            return self.refuse(accepted, DeviceError::disconnected("no driver"));
        };
        let kind = &mut self.kind;
        let plan = match catch_unwind(AssertUnwindSafe(|| kind.plan(cmd, drv, &ctx))) {
            Ok(plan) => plan,
            Err(panic) => {
                let message = panic_message(panic.as_ref());
                self.on_panic(&message);
                return self.refuse(
                    accepted,
                    DeviceError::driver(format!("driver panic: {message}")),
                );
            }
        };
        for event in events.drain() {
            self.emit(event, Some(command_id));
        }
        match plan {
            Plan::Reject(err) => {
                self.emit_ended(op_kind, command_id, OpOutcome::Failed(err.clone()), 0);
                self.refuse(accepted, err);
            }
            Plan::Concurrent(action) => {
                self.run_action(command_id, op_kind, action, accepted, done);
            }
            Plan::Halt(action) => {
                self.cancel_active(Some(command_id), true);
                if self.mode != Mode::Running {
                    return self.refuse(accepted, DeviceError::disconnected("the device was lost"));
                }
                self.run_action(command_id, op_kind, action, accepted, done);
            }
            Plan::Instant(action) => {
                if !self.make_room(&permit, command_id) {
                    return self.refuse_busy(op_kind, command_id, accepted);
                }
                self.run_action(command_id, op_kind, action, accepted, done);
                drop(permit);
            }
            Plan::Operation(op) => {
                if !self.make_room(&permit, command_id) {
                    return self.refuse_busy(op_kind, command_id, accepted);
                }
                self.start_operation(command_id, op, permit, events, accepted, done);
            }
        }
    }

    /// A conflicting command (§2.5): `InvalidOperation`, recorded as ended.
    fn refuse_busy(
        &self,
        op: OpKind,
        command_id: CommandId,
        accepted: oneshot::Sender<DeviceResult<()>>,
    ) {
        let err = DeviceError::invalid_operation("another operation is active on this device");
        self.emit_ended(op, command_id, OpOutcome::Failed(err.clone()), 0);
        self.refuse(accepted, err);
    }

    /// Make room for a new command: true when no operation is active, or
    /// when the new permit outranks the active one, which is then cancelled
    /// (`Aborted { by: new command }`).
    fn make_room(&mut self, permit: &Permit, command_id: CommandId) -> bool {
        let Some(active) = &self.active else {
            return true;
        };
        if !outranks(permit.rank(), permit.source(), active.permit.rank()) {
            return false;
        }
        self.cancel_active(Some(command_id), false);
        self.mode == Mode::Running
    }

    fn emit_ended(&self, op: OpKind, command_id: CommandId, outcome: OpOutcome, duration_ms: u64) {
        self.emit(
            EventKind::OperationEnded {
                op,
                outcome,
                duration_ms,
            },
            Some(command_id),
        );
    }

    fn run_action(
        &mut self,
        command_id: CommandId,
        op_kind: OpKind,
        action: Action<K::Driver>,
        accepted: oneshot::Sender<DeviceResult<()>>,
        done: oneshot::Sender<OpOutcome>,
    ) {
        let Some(drv) = self.drv.as_mut() else {
            return self.refuse(accepted, DeviceError::disconnected("no driver"));
        };
        let started = self.clock.now_mono_ns();
        let result = catch_unwind(AssertUnwindSafe(|| action(drv)));
        let duration_ms = self.clock.now_mono_ns().saturating_sub(started) / 1_000_000;
        match result {
            Err(panic) => {
                let message = panic_message(panic.as_ref());
                self.on_panic(&message);
                self.refuse(
                    accepted,
                    DeviceError::driver(format!("driver panic: {message}")),
                );
            }
            Ok(Err(err)) => {
                self.emit_ended(
                    op_kind,
                    command_id,
                    OpOutcome::Failed(err.clone()),
                    duration_ms,
                );
                self.refuse(accepted, err);
            }
            Ok(Ok(())) => {
                let _ = accepted.send(Ok(()));
                self.emit(
                    EventKind::OperationStarted { op: op_kind },
                    Some(command_id),
                );
                let _ = done.send(OpOutcome::Succeeded);
                self.emit_ended(op_kind, command_id, OpOutcome::Succeeded, duration_ms);
                // A successful set invalidates the member: re-read it now
                // (C-2).
                self.next_poll_at = self.clock.now_mono_ns();
            }
        }
    }

    fn start_operation(
        &mut self,
        command_id: CommandId,
        mut op: Box<dyn Operation<K::Driver>>,
        permit: Permit,
        events: OpEvents,
        accepted: oneshot::Sender<DeviceResult<()>>,
        done: oneshot::Sender<OpOutcome>,
    ) {
        let op_kind = op.kind();
        let deadline = op.deadline();
        let clock = Arc::clone(&self.clock);
        let Some(drv) = self.drv.as_mut() else {
            return self.refuse(accepted, DeviceError::disconnected("no driver"));
        };
        let started = catch_unwind(AssertUnwindSafe(|| op.start(drv, clock.as_ref())));
        for event in events.drain() {
            self.emit(event, Some(command_id));
        }
        match started {
            Err(panic) => {
                let message = panic_message(panic.as_ref());
                self.on_panic(&message);
                self.refuse(
                    accepted,
                    DeviceError::driver(format!("driver panic: {message}")),
                );
            }
            Ok(Err(err)) if initiator_outcome_unknown(op_kind, &err) => {
                // O11 e: a Timeout or Transport error on the initiator means
                // the command may have reached the device. It is accepted and
                // resolved `Unknown` (which raises the hub's outcome_unknown
                // resume block), and the hardware is halted best effort (C-4)
                // so nothing keeps moving unobserved.
                let unknown = DeviceError::outcome_unknown(format!(
                    "the {op_kind:?} initiator may have reached the device: {err}"
                ));
                let _ = accepted.send(Ok(()));
                self.emit(
                    EventKind::OperationStarted { op: op_kind },
                    Some(command_id),
                );
                if let Some(drv) = self.drv.as_mut() {
                    if let Ok(Err(cancel_err)) = catch_unwind(AssertUnwindSafe(|| op.cancel(drv))) {
                        tracing::warn!(
                            device = %self.end.identity.name,
                            "halt after an unconfirmed initiator failed: {cancel_err}"
                        );
                    }
                }
                for event in events.drain() {
                    self.emit(event, Some(command_id));
                }
                // The events (and the resume block they raise) come before
                // the outcome, so whoever awaits the ticket sees the block.
                let outcome = OpOutcome::Unknown(unknown.clone());
                self.emit_ended(op_kind, command_id, outcome.clone(), 0);
                self.shared
                    .announce_unknown(command_id, op_kind, unknown.message);
                let _ = done.send(outcome);
                self.next_poll_at = self.clock.now_mono_ns();
            }
            Ok(Err(err)) => {
                self.emit_ended(op_kind, command_id, OpOutcome::Failed(err.clone()), 0);
                self.refuse(accepted, err);
            }
            Ok(Ok(())) => {
                let (utc, mono) = self.now();
                *lock(&self.shared.done) = Some(PendingDone {
                    command_id,
                    op: op_kind,
                    started_mono_ns: mono,
                    tx: done,
                });
                self.active = Some(Active {
                    op,
                    info: ActiveOp {
                        command_id,
                        kind: op_kind,
                        phase: OpPhase::Starting,
                        started_at: utc,
                        started_mono_ns: mono,
                        deadline_mono_ns: mono.saturating_add(ns(deadline)),
                    },
                    permit,
                    events,
                    settle_until: None,
                    cancel: None,
                    hint: None,
                });
                let _ = accepted.send(Ok(()));
                self.emit(
                    EventKind::OperationStarted { op: op_kind },
                    Some(command_id),
                );
                self.republish();
                // Poll at the active cadence from now on.
                self.next_poll_at = mono.saturating_add(ns(self.current_interval()));
            }
        }
    }

    fn on_disconnect(
        &mut self,
        command_id: CommandId,
        force: bool,
        done: oneshot::Sender<DeviceResult<()>>,
    ) -> Flow {
        if self.mode == Mode::Running
            && !force
            && self.kind.refuses_disconnect(self.last.as_deref())
        {
            let _ = done.send(Err(DeviceError::invalid_operation(
                "a slew is in progress; disconnect with force=true to abort it first",
            )));
            return Flow::Continue;
        }
        if self.mode == Mode::Running {
            self.cancel_active(Some(command_id), false);
        }
        if let Some(drv) = self.drv.as_mut() {
            let kind = &mut self.kind;
            if let Ok(Err(err)) = catch_unwind(AssertUnwindSafe(|| kind.halt_on_disconnect(drv))) {
                tracing::warn!("halt before disconnect failed: {err}");
            }
            if let Ok(Err(err)) = catch_unwind(AssertUnwindSafe(|| drv.disconnect())) {
                tracing::warn!("disconnect failed: {err}");
            }
        }
        self.emit(
            EventKind::DeviceDisconnected {
                end_class: ConnectionEnd::Operator,
                reason: None,
            },
            Some(command_id),
        );
        self.close_connection(ConnectionEnd::Operator);
        self.end.retire(self.clock.now_mono_ns());
        let _ = done.send(Ok(()));
        Flow::Exit
    }

    /// §2.15: halt focuser and rotator moves, stop a jog and abort an
    /// Achernar slew; leave park, unpark, home, dome, cover and wheel
    /// operations running (their outcome is then unknown); never park;
    /// close politely.
    fn shutdown(&mut self) {
        if self.mode == Mode::Running && self.active.is_some() {
            let op = self.active.as_ref().map(|a| a.info.kind);
            match op {
                Some(op) if halts_at_shutdown(op) => {
                    self.cancel_op();
                    self.resolve(OpOutcome::Aborted { by: None });
                }
                Some(op) if op.moves_hardware() => {
                    self.resolve(OpOutcome::Unknown(DeviceError::outcome_unknown(format!(
                        "the daemon shut down during {op:?}; it was left running and its \
                         result was not observed"
                    ))));
                }
                _ => self.resolve(OpOutcome::Aborted { by: None }),
            }
        }
        if let Some(drv) = self.drv.as_mut() {
            if let Ok(Err(err)) = catch_unwind(AssertUnwindSafe(|| drv.disconnect())) {
                tracing::warn!("disconnect at shutdown failed: {err}");
            }
        }
        if !self.end.is_retired() {
            self.emit(
                EventKind::DeviceDisconnected {
                    end_class: ConnectionEnd::Shutdown,
                    reason: None,
                },
                None,
            );
        }
        self.close_connection(ConnectionEnd::Shutdown);
        self.end.retire(self.clock.now_mono_ns());
    }
}

// ---------------------------------------------------------------------------
// Helpers for device packages
// ---------------------------------------------------------------------------

/// An [`Operation`] built from closures, for simple long operations and
/// tests: `start` sends the initiator, `done` is the completion predicate
/// (never position versus target, C-5), `cancel` halts.
pub struct FnOperation<D: ?Sized> {
    pub kind: OpKind,
    pub deadline: Duration,
    pub settle: Duration,
    #[allow(clippy::type_complexity)]
    pub start: Option<Box<dyn FnOnce(&mut D) -> DeviceResult<()> + Send>>,
    #[allow(clippy::type_complexity)]
    pub done: Box<dyn FnMut(&mut D) -> DeviceResult<bool> + Send>,
    #[allow(clippy::type_complexity)]
    pub cancel: Box<dyn FnMut(&mut D) -> DeviceResult<()> + Send>,
    pub poll: Duration,
    settled: bool,
}

impl<D: ?Sized> FnOperation<D> {
    pub fn new(
        kind: OpKind,
        deadline: Duration,
        start: impl FnOnce(&mut D) -> DeviceResult<()> + Send + 'static,
        done: impl FnMut(&mut D) -> DeviceResult<bool> + Send + 'static,
        cancel: impl FnMut(&mut D) -> DeviceResult<()> + Send + 'static,
    ) -> Self {
        Self {
            kind,
            deadline,
            settle: Duration::ZERO,
            start: Some(Box::new(start)),
            done: Box::new(done),
            cancel: Box::new(cancel),
            poll: default_cadence(DeviceKind::Mount).active,
            settled: false,
        }
    }

    pub fn with_settle(mut self, settle: Duration) -> Self {
        self.settle = settle;
        self
    }

    pub fn with_poll(mut self, poll: Duration) -> Self {
        self.poll = poll;
        self
    }
}

impl<D: ?Sized> Operation<D> for FnOperation<D> {
    fn kind(&self) -> OpKind {
        self.kind
    }

    fn deadline(&self) -> Duration {
        self.deadline
    }

    fn start(&mut self, drv: &mut D, _clock: &dyn Clock) -> DeviceResult<()> {
        match self.start.take() {
            Some(start) => start(drv),
            None => Ok(()),
        }
    }

    fn step(&mut self, drv: &mut D, clock: &dyn Clock) -> OpProgress {
        if self.settled {
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        match (self.done)(drv) {
            Ok(false) => OpProgress::Continue {
                next_poll: self.poll,
            },
            Ok(true) if self.settle.is_zero() => OpProgress::Done(OpOutcome::Succeeded),
            Ok(true) => {
                self.settled = true;
                OpProgress::Settling {
                    until_mono_ns: clock.now_mono_ns().saturating_add(ns(self.settle)),
                }
            }
            Err(err) => OpProgress::Done(OpOutcome::Failed(err)),
        }
    }

    fn cancel(&mut self, drv: &mut D) -> DeviceResult<()> {
        (self.cancel)(drv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::{GatedCommand, GatedOp};
    use crate::equipment::gate::tests::gate_with_ctx;
    use crate::equipment::handle::{DeviceHandle, FocuserHandle};
    use crate::equipment::hub::{EquipmentHub, HubConfig};
    use crate::equipment::snapshot::{FocuserCapabilities, FocuserSnapshot, FocuserState};
    use crate::equipment::testing::fake_handles::{fake_identity, FakeAbortPath, RecordingSink};
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::{FocuserDriver, PollMethod};
    use crate::equipment::types::{AutonomyMode, ErrorClass, ObsState, Observed, Source};
    use std::sync::Condvar;

    // -- a scripted focuser driver ---------------------------------------------------

    #[derive(Default)]
    struct Script {
        position: i32,
        target: Option<i32>,
        steps_per_poll: i32,
        moving: bool,
        halted: u32,
        polls: u32,
        /// Fail the next N polls.
        fail_polls: u32,
        fail_with: Option<DeviceError>,
        /// Panic on the next poll.
        panic_poll: bool,
        /// Advance the manual clock by this much inside each poll.
        poll_takes: Duration,
        /// Block (real time) inside the next poll until released.
        block_next_poll: bool,
        blocked: bool,
        reconnects: u32,
        reconnect_fails: bool,
        disconnected: bool,
        ignore_halt: bool,
        /// IsMoving cannot be read (the member errors).
        moving_unknown: bool,
        /// The next move's initiator fails with this error.
        move_fails_with: Option<DeviceError>,
    }

    struct Shared {
        script: Mutex<Script>,
        released: Condvar,
    }

    impl Shared {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                script: Mutex::new(Script {
                    steps_per_poll: 10,
                    ..Script::default()
                }),
                released: Condvar::new(),
            })
        }

        fn with<R>(&self, f: impl FnOnce(&mut Script) -> R) -> R {
            f(&mut lock(&self.script))
        }

        fn release(&self) {
            self.with(|s| s.block_next_poll = false);
            self.released.notify_all();
        }
    }

    struct TestDriver {
        identity: DeviceIdentity,
        shared: Arc<Shared>,
        clock: Arc<ManualClock>,
        abort: Arc<FakeAbortPath>,
    }

    impl Device for TestDriver {
        fn identity(&self) -> &DeviceIdentity {
            &self.identity
        }
        fn connect(&mut self, _timeout: Duration) -> DeviceResult<ConnectReport> {
            self.shared.with(|s| {
                s.reconnects += 1;
                if s.reconnect_fails {
                    Err(DeviceError::transport("connection refused"))
                } else {
                    Ok(ConnectReport::default())
                }
            })
        }
        fn disconnect(&mut self) -> DeviceResult<()> {
            self.shared.with(|s| s.disconnected = true);
            Ok(())
        }
        fn is_connected(&mut self) -> DeviceResult<bool> {
            Ok(true)
        }
        fn poll(&mut self, _plan: &PollPlan) -> DeviceResult<RawPoll> {
            let started = self.clock.now_mono_ns();
            {
                let mut s = lock(&self.shared.script);
                s.polls += 1;
                if s.block_next_poll {
                    s.blocked = true;
                    let deadline = Instant::now() + Duration::from_secs(5);
                    while s.block_next_poll && Instant::now() < deadline {
                        s = self
                            .shared
                            .released
                            .wait_timeout(s, Duration::from_millis(50))
                            .unwrap_or_else(|e| e.into_inner())
                            .0;
                    }
                    s.blocked = false;
                }
                if s.panic_poll {
                    s.panic_poll = false;
                    drop(s);
                    panic!("scripted driver panic");
                }
                if !s.poll_takes.is_zero() {
                    self.clock.advance(s.poll_takes);
                }
                if s.fail_polls > 0 {
                    s.fail_polls -= 1;
                    return Err(s
                        .fail_with
                        .clone()
                        .unwrap_or_else(|| DeviceError::transport("connection refused")));
                }
                if let Some(target) = s.target {
                    let step = s.steps_per_poll;
                    let delta = (target - s.position).clamp(-step, step);
                    s.position += delta;
                    s.moving = s.position != target;
                    if !s.moving {
                        s.target = None;
                    }
                }
            }
            Ok(RawPoll {
                method: PollMethod::InProcess,
                started_mono_ns: started,
                finished_mono_ns: self.clock.now_mono_ns(),
                ..RawPoll::default()
            })
        }
        fn supported_actions(&mut self) -> DeviceResult<Vec<String>> {
            Ok(Vec::new())
        }
        fn action(&mut self, _name: &str, _params: &str) -> DeviceResult<String> {
            Err(DeviceError::unsupported("no actions"))
        }
        fn abort_path(&self) -> Option<Arc<dyn AbortPath>> {
            Some(self.abort.clone() as Arc<dyn AbortPath>)
        }
    }

    impl FocuserDriver for TestDriver {
        fn capabilities(&mut self) -> DeviceResult<FocuserCapabilities> {
            Ok(FocuserCapabilities {
                absolute: true,
                max_step: Some(50_000),
                ..FocuserCapabilities::default()
            })
        }
        fn state(&mut self) -> DeviceResult<FocuserState> {
            let (position, moving, moving_unknown) = self
                .shared
                .with(|s| (s.position, s.moving, s.moving_unknown));
            let clock = self.clock.as_ref();
            Ok(FocuserState {
                position: Observed::fresh_at(position, Source::Property, clock),
                is_moving: if moving_unknown {
                    Observed::error(Source::Property)
                } else {
                    Observed::fresh_at(moving, Source::Property, clock)
                },
                temperature_c: Observed::unsupported(Source::Property),
                temp_comp: Observed::unsupported(Source::Property),
            })
        }
        fn move_absolute(&mut self, target: i32) -> DeviceResult<()> {
            self.shared.with(|s| {
                if let Some(err) = s.move_fails_with.take() {
                    return Err(err);
                }
                s.target = Some(target);
                s.moving = true;
                Ok(())
            })
        }
        fn move_relative(&mut self, _steps: i32) -> DeviceResult<()> {
            Err(DeviceError::unsupported("absolute focuser"))
        }
        fn halt(&mut self) -> DeviceResult<()> {
            self.shared.with(|s| {
                s.halted += 1;
                if !s.ignore_halt {
                    s.target = None;
                    s.moving = false;
                }
            });
            Ok(())
        }
        fn set_temp_comp(&mut self, _on: bool) -> DeviceResult<()> {
            Err(DeviceError::unsupported("no tempcomp"))
        }
        fn sync_position(&mut self, _position: i32) -> DeviceResult<()> {
            Err(DeviceError::unsupported("Alpaca has no sync"))
        }
    }

    /// The driver type of the test kind (`dyn FocuserDriver + 'static`).
    type Drv = dyn FocuserDriver + 'static;

    /// Settle after arrival, scripted per test.
    struct TestKind {
        settle: Duration,
        deadline: Duration,
        panic_on_step: bool,
    }

    impl WorkerKind for TestKind {
        type Driver = dyn FocuserDriver;
        type Cmd = FocuserCmd;
        type Snap = FocuserSnapshot;

        fn kind(&self) -> DeviceKind {
            DeviceKind::Focuser
        }

        fn plan(
            &mut self,
            cmd: FocuserCmd,
            _drv: &mut Drv,
            ctx: &PlanContext<'_, FocuserSnapshot>,
        ) -> Plan<Drv> {
            match cmd {
                FocuserCmd::MoveTo(target) => {
                    let events = ctx.events.clone();
                    let panic_on_step = self.panic_on_step;
                    let op = FnOperation::new(
                        OpKind::FocuserMove,
                        self.deadline,
                        move |drv: &mut Drv| drv.move_absolute(target),
                        move |drv: &mut Drv| {
                            if panic_on_step {
                                panic!("scripted operation panic");
                            }
                            let state = drv.state()?;
                            let moving = state.is_moving.known().copied().unwrap_or(true);
                            if !moving {
                                events.push(EventKind::FocuserSettled {
                                    move_id: None,
                                    position: state.position.known().copied(),
                                    raw_position: state.position.known().copied(),
                                });
                            }
                            Ok(!moving)
                        },
                        |drv: &mut Drv| drv.halt(),
                    )
                    .with_settle(self.settle)
                    .with_poll(Duration::from_millis(100));
                    Plan::Operation(Box::new(op))
                }
                FocuserCmd::Halt => Plan::Halt(Box::new(|drv: &mut Drv| drv.halt())),
                FocuserCmd::SetTempComp(on) => {
                    Plan::Instant(Box::new(move |drv: &mut Drv| drv.set_temp_comp(on)))
                }
                FocuserCmd::MoveBy(_) => Plan::Concurrent(Box::new(|_drv| Ok(()))),
                FocuserCmd::Sync(_) => Plan::Reject(DeviceError::unsupported("no sync")),
            }
        }

        fn snapshot(
            &mut self,
            drv: &mut Drv,
            _poll: &RawPoll,
            _ctx: &SnapshotContext<'_, FocuserSnapshot>,
        ) -> DeviceResult<FocuserSnapshot> {
            let state = drv.state()?;
            Ok(FocuserSnapshot {
                logical_position: state.position.known().copied(),
                state,
                ..FocuserSnapshot::default()
            })
        }

        fn stale_snapshot(
            &mut self,
            previous: &FocuserSnapshot,
            _ctx: &SnapshotContext<'_, FocuserSnapshot>,
        ) -> Option<FocuserSnapshot> {
            let mut stale = previous.clone();
            stale.state.position = stale.state.position.into_stale();
            stale.state.is_moving = stale.state.is_moving.into_stale();
            Some(stale)
        }

        fn edges(&mut self, previous: &FocuserSnapshot, next: &FocuserSnapshot) -> Vec<EventKind> {
            let was = previous.state.is_moving.known().copied();
            let now = next.state.is_moving.known().copied();
            if was == Some(false) && now == Some(true) {
                vec![EventKind::FocuserMoveStarted {
                    move_id: uuid::Uuid::new_v4(),
                    from: previous.logical_position,
                }]
            } else {
                Vec::new()
            }
        }
    }

    struct Rig {
        handle: FocuserHandle,
        worker: Option<WorkerHandle>,
        shared: Arc<Shared>,
        clock: Arc<ManualClock>,
        hub: EquipmentHub,
        sink: Arc<RecordingSink>,
        abort: Arc<FakeAbortPath>,
    }

    impl Drop for Rig {
        fn drop(&mut self) {
            self.shared.release();
            if let Some(worker) = self.worker.take() {
                worker.stop();
                worker.join_timeout(Duration::from_secs(5));
            }
        }
    }

    fn rig_with(kind: TestKind, policy: HealthPolicy) -> Rig {
        rig_kind(kind, policy)
    }

    fn rig_kind<K>(kind: K, policy: HealthPolicy) -> Rig
    where
        K: WorkerKind<Driver = dyn FocuserDriver, Cmd = FocuserCmd, Snap = FocuserSnapshot>,
    {
        rig_kind_with(kind, policy, false)
    }

    fn rig_kind_with<K>(kind: K, policy: HealthPolicy, require_gate_permits: bool) -> Rig
    where
        K: WorkerKind<Driver = dyn FocuserDriver, Cmd = FocuserCmd, Snap = FocuserSnapshot>,
    {
        let clock = Arc::new(ManualClock::new());
        let sink = RecordingSink::new();
        let hub = EquipmentHub::new(HubConfig::default(), clock.clone(), sink.clone());
        let shared = Shared::new();
        let abort = FakeAbortPath::new();
        let identity = fake_identity(DeviceKind::Focuser);
        let driver = TestDriver {
            identity: identity.clone(),
            shared: Arc::clone(&shared),
            clock: Arc::clone(&clock),
            abort: Arc::clone(&abort),
        };
        let (handle, end) = DeviceHandle::new_pair(
            identity,
            Some(abort.clone() as Arc<dyn AbortPath>),
            FocuserSnapshot::default(),
            HealthState::ready(clock.now_utc(), clock.now_mono_ns()),
        );
        let ring = hub.attach(handle.clone());
        let worker = spawn_worker(
            kind,
            Box::new(driver) as Box<dyn FocuserDriver>,
            end,
            WorkerConfig {
                policy,
                max_wait: Duration::from_millis(2),
                connect_timeout: Duration::from_millis(10),
                require_gate_permits,
                ..WorkerConfig::default()
            },
            WorkerDeps {
                clock: clock.clone(),
                events: hub.event_sink(),
                recorder: sink.clone(),
                ring: Some(ring),
                watchdog: Some(hub.watchdog()),
                aborts: Some(hub.gate().abort_epochs()),
            },
        )
        .unwrap();
        Rig {
            handle,
            worker: Some(worker),
            shared,
            clock,
            hub,
            sink,
            abort,
        }
    }

    fn rig() -> Rig {
        rig_with(
            TestKind {
                settle: Duration::ZERO,
                deadline: Duration::from_secs(60),
                panic_on_step: false,
            },
            HealthPolicy::default(),
        )
    }

    /// Wait (real time) for a condition while the manual clock advances in
    /// `step` increments.
    fn drive(rig: &Rig, step: Duration, mut done: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if done() {
                return true;
            }
            rig.clock.advance(step);
            std::thread::sleep(Duration::from_millis(3));
        }
        done()
    }

    fn permit(source: CommandSource, priority: Priority) -> Permit {
        Permit::unmanaged(CommandId::new(), priority, source)
    }

    fn operator() -> Permit {
        permit(
            CommandSource::Operator { overriding: false },
            Priority::Exposure,
        )
    }

    async fn wait_outcome(rig: &Rig, ticket: crate::equipment::handle::OpTicket) -> OpOutcome {
        let mut done = ticket.done;
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match done.try_recv() {
                Ok(outcome) => return outcome,
                Err(oneshot::error::TryRecvError::Closed) => panic!("ticket dropped"),
                Err(oneshot::error::TryRecvError::Empty) => {}
            }
            assert!(Instant::now() < deadline, "operation never resolved");
            rig.clock.advance_ms(100);
            tokio::time::sleep(Duration::from_millis(3)).await;
        }
    }

    #[test]
    fn the_worker_thread_is_named_after_the_kind_and_connection() {
        let rig = rig();
        let name = rig.worker.as_ref().unwrap().name().to_string();
        let id = rig.handle.connection_id().to_string();
        assert_eq!(name, format!("eq-focuser-{}", &id[..8]));
        assert!(drive(&rig, Duration::from_millis(100), || rig
            .sink
            .event_names()
            .contains(&"device_connected")));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_operation_completes_settles_and_emits_its_records() {
        let rig = rig_with(
            TestKind {
                settle: Duration::from_millis(500),
                deadline: Duration::from_secs(60),
                panic_on_step: false,
            },
            HealthPolicy::default(),
        );
        let ticket = rig
            .handle
            .send(FocuserCmd::MoveTo(40), operator())
            .await
            .unwrap();
        let id = ticket.command_id;
        assert_eq!(wait_outcome(&rig, ticket).await, OpOutcome::Succeeded);
        assert_eq!(rig.shared.with(|s| s.position), 40);
        let names = rig.sink.event_names();
        let order: Vec<_> = names
            .iter()
            .filter(|n| {
                matches!(
                    **n,
                    "operation_commanded"
                        | "operation_started"
                        | "focuser_settled"
                        | "operation_ended"
                )
            })
            .copied()
            .collect();
        assert_eq!(
            order,
            vec![
                "operation_commanded",
                "operation_started",
                "focuser_settled",
                "operation_ended"
            ]
        );
        assert!(
            names.contains(&"focuser_move_started"),
            "edge event from the adapter"
        );
        // The ring holds published snapshots, and the last has no active op.
        assert!(rig.hub.rings().focuser.len() >= 3);
        assert!(rig.handle.snapshot().active_op.is_none());
        let ended = rig.sink.records().into_iter().find_map(|r| match r {
            GroundTruthRecord::Event(e) if e.name() == "operation_ended" => Some(e),
            _ => None,
        });
        assert_eq!(ended.unwrap().command_id, Some(id));
        // Instant, concurrent and rejected commands.
        assert_eq!(
            rig.handle
                .send(FocuserCmd::SetTempComp(true), operator())
                .await
                .unwrap_err()
                .class,
            ErrorClass::Unsupported
        );
        assert!(rig
            .handle
            .send(FocuserCmd::Sync(5), operator())
            .await
            .is_err());
        let ok = rig
            .handle
            .send(FocuserCmd::MoveBy(1), operator())
            .await
            .unwrap();
        assert_eq!(ok.wait().await, OpOutcome::Succeeded);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_operation_that_overruns_its_deadline_is_halted_and_times_out() {
        let rig = rig_with(
            TestKind {
                settle: Duration::ZERO,
                deadline: Duration::from_secs(2),
                panic_on_step: false,
            },
            HealthPolicy::default(),
        );
        rig.shared.with(|s| s.steps_per_poll = 1);
        let ticket = rig
            .handle
            .send(FocuserCmd::MoveTo(10_000), operator())
            .await
            .unwrap();
        assert_eq!(wait_outcome(&rig, ticket).await, OpOutcome::TimedOut);
        assert!(
            rig.shared.with(|s| s.halted) >= 1,
            "timing out halts the hardware"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abort_cancels_after_a_poll_confirms_motion_stopped() {
        let rig = rig();
        rig.shared.with(|s| s.steps_per_poll = 1);
        let ticket = rig
            .handle
            .send(FocuserCmd::MoveTo(10_000), operator())
            .await
            .unwrap();
        let abort = rig.handle.abort();
        let abort_id = abort.command_id;
        assert!(abort.abort_path_ran && abort.cancel_sent);
        assert!(abort.outcome().await.is_ok());
        let outcome = wait_outcome(&rig, ticket).await;
        assert_eq!(outcome, OpOutcome::Aborted { by: Some(abort_id) });
        assert_eq!(rig.abort.calls(), 1);
        assert!(
            rig.shared.with(|s| s.halted) >= 1,
            "cancel halts the hardware"
        );
        // A cancelled move that keeps moving resolves unknown after the
        // confirmation timeout.
        rig.shared.with(|s| s.ignore_halt = true);
        let ticket = rig
            .handle
            .send(FocuserCmd::MoveTo(-10_000), operator())
            .await
            .unwrap();
        rig.handle.abort();
        assert!(matches!(
            wait_outcome(&rig, ticket).await,
            OpOutcome::Unknown(_)
        ));
        assert!(rig.sink.event_names().contains(&"outcome_unknown"));
        assert!(drive(&rig, Duration::from_millis(10), || !rig
            .hub
            .resume()
            .active()
            .is_empty()));
        // A halt command cancels the active operation in its own name.
        rig.shared.with(|s| s.ignore_halt = false);
        let ticket = rig
            .handle
            .send(FocuserCmd::MoveTo(5_000), operator())
            .await
            .unwrap();
        let halt = rig.handle.send(FocuserCmd::Halt, operator()).await.unwrap();
        let halt_id = halt.command_id;
        assert_eq!(halt.wait().await, OpOutcome::Succeeded);
        assert_eq!(
            wait_outcome(&rig, ticket).await,
            OpOutcome::Aborted { by: Some(halt_id) }
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_higher_rank_preempts_and_a_conflict_is_refused() {
        let rig = rig();
        rig.shared.with(|s| s.steps_per_poll = 1);
        let policy = CommandSource::Policy {
            model: "af".into(),
            version: "1".into(),
        };
        let held = rig
            .handle
            .send(
                FocuserCmd::MoveTo(10_000),
                permit(policy.clone(), Priority::Autofocus),
            )
            .await
            .unwrap();
        // Equal rank and an operator without override are refused.
        let err = rig
            .handle
            .send(FocuserCmd::MoveTo(1), permit(policy, Priority::Autofocus))
            .await
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::InvalidOperation);
        assert!(rig
            .handle
            .send(FocuserCmd::MoveTo(1), operator())
            .await
            .is_err());
        // An operator override preempts: the held op resolves Aborted by it.
        let over = permit(
            CommandSource::Operator { overriding: true },
            Priority::Exposure,
        );
        let over_id = over.command_id();
        let ticket = rig.handle.send(FocuserCmd::MoveTo(3), over).await.unwrap();
        assert_eq!(
            wait_outcome(&rig, held).await,
            OpOutcome::Aborted { by: Some(over_id) }
        );
        assert_eq!(wait_outcome(&rig, ticket).await, OpOutcome::Succeeded);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_driver_panic_is_contained_and_the_permit_released() {
        let rig = rig_with(
            TestKind {
                settle: Duration::ZERO,
                deadline: Duration::from_secs(60),
                panic_on_step: true,
            },
            HealthPolicy::default(),
        );
        let (gate, _ctx) = gate_with_ctx();
        let cmd = GatedCommand::new(
            GatedOp::Focuser(FocuserCmd::MoveTo(100)),
            CommandSource::Operator { overriding: false },
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        let domain = crate::equipment::commands::Domain::Device(DeviceKind::Focuser);
        let permit = gate.authorize(&cmd).unwrap();
        assert!(gate.holder(domain).is_some());
        let ticket = rig
            .handle
            .send(FocuserCmd::MoveTo(100), permit)
            .await
            .unwrap();
        let outcome = wait_outcome(&rig, ticket).await;
        assert!(matches!(&outcome, OpOutcome::Unknown(e) if e.message.contains("panicked")));
        assert!(
            gate.holder(domain).is_none(),
            "the permit was released by drop"
        );
        assert_eq!(rig.handle.health().state, LinkState::Lost);
        assert_eq!(
            rig.handle.health().failure_class,
            Some(FailureClass::DriverPanic)
        );
        assert!(rig.abort.calls() >= 1, "the abort path ran");
        let names = rig.sink.event_names();
        assert!(names.contains(&"device_lost") && names.contains(&"outcome_unknown"));
        // Commands now fail fast; abort still reaches the abort path (D19).
        let err = rig
            .handle
            .send(FocuserCmd::MoveTo(1), operator())
            .await
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::Disconnected);
        let abort = rig.handle.abort();
        assert!(abort.outcome().await.is_ok());
        // Disconnect retires the handle; after that even abort is 412.
        rig.handle.disconnect(false).await.unwrap();
        assert!(rig.handle.is_retired());
        assert_eq!(
            rig.handle.abort().outcome().await.unwrap_err().class,
            ErrorClass::NotConnected
        );
    }

    #[test]
    fn a_poll_panic_is_contained_too() {
        let rig = rig();
        rig.shared.with(|s| s.panic_poll = true);
        assert!(drive(&rig, Duration::from_millis(500), || rig
            .handle
            .health()
            .state
            == LinkState::Lost));
        assert!(rig.sink.records().iter().any(|r| matches!(
            r,
            GroundTruthRecord::ConnectionClosed {
                end_class: ConnectionEnd::Failed,
                ..
            }
        )));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abort_returns_within_100_ms_while_the_worker_is_blocked() {
        let rig = rig();
        rig.shared.with(|s| s.steps_per_poll = 1);
        let ticket = rig
            .handle
            .send(FocuserCmd::MoveTo(10_000), operator())
            .await
            .unwrap();
        // The next poll blocks the worker inside the driver (a 5 s call).
        rig.shared.with(|s| s.block_next_poll = true);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !rig.shared.with(|s| s.blocked) {
            assert!(Instant::now() < deadline, "the worker never blocked");
            rig.clock.advance_ms(100);
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        let started = Instant::now();
        let abort = rig.handle.abort();
        let abort_id = abort.command_id;
        assert!(abort.outcome().await.is_ok());
        let latency = started.elapsed();
        assert!(
            latency < Duration::from_millis(100),
            "abort took {latency:?}"
        );
        assert_eq!(rig.abort.calls(), 1);
        assert!(
            rig.shared.with(|s| s.blocked),
            "the worker is still blocked"
        );
        // Once the call returns, the queued cancel resolves the operation.
        rig.shared.release();
        assert_eq!(
            wait_outcome(&rig, ticket).await,
            OpOutcome::Aborted { by: Some(abort_id) }
        );
    }

    #[test]
    fn lost_after_n_failures_not_after_one_slow_read() {
        let rig = rig();
        assert!(drive(&rig, Duration::from_millis(100), || rig
            .shared
            .with(|s| s.polls)
            >= 2));
        // One slow read (4 s, with a 1 s mount-like... here a 2 s focuser
        // cadence): degraded at most, never lost.
        rig.shared.with(|s| s.poll_takes = Duration::from_secs(4));
        let polls = rig.shared.with(|s| s.polls);
        assert!(drive(&rig, Duration::from_millis(100), || rig
            .shared
            .with(|s| s.polls)
            > polls));
        rig.shared.with(|s| s.poll_takes = Duration::ZERO);
        let polls = rig.shared.with(|s| s.polls);
        assert!(drive(&rig, Duration::from_millis(100), || rig
            .shared
            .with(|s| s.polls)
            > polls + 2));
        assert_eq!(rig.handle.health().state, LinkState::Ready);
        assert!(rig.sink.event_names().contains(&"poll_cadence_reduced"));
        assert!(!rig.sink.event_names().contains(&"device_lost"));
        // Two failures degrade; the third (with a failed reconnect) loses.
        rig.shared.with(|s| {
            s.fail_polls = 100;
            s.reconnect_fails = true;
        });
        assert!(drive(&rig, Duration::from_millis(250), || rig
            .handle
            .health()
            .state
            == LinkState::Degraded));
        assert!(drive(&rig, Duration::from_millis(250), || rig
            .handle
            .health()
            .state
            == LinkState::Lost));
        let health = rig.handle.health();
        assert!(health.consecutive_failures >= 3);
        assert_eq!(health.failure_class, Some(FailureClass::Disconnected));
        assert_eq!(rig.shared.with(|s| s.reconnects), 1, "reconnect once");
        let names = rig.sink.event_names();
        assert!(names.contains(&"device_degraded") && names.contains(&"device_lost"));
        // Stale values are published, never presented as fresh.
        assert_eq!(rig.handle.snapshot().state.position.state, ObsState::Stale);
        // Abort on the lost (not retired) handle still reaches the path.
        let abort = rig.handle.abort();
        assert!(abort.abort_path_ran);
        assert!(drive(&rig, Duration::from_millis(1), || rig.abort.calls() == 1));
    }

    #[test]
    fn a_reconnect_restores_a_device_that_said_not_connected() {
        let rig = rig();
        assert!(drive(&rig, Duration::from_millis(100), || rig
            .shared
            .with(|s| s.polls)
            >= 1));
        rig.shared.with(|s| {
            s.fail_polls = 1;
            s.fail_with = Some(DeviceError::not_connected("Connected=false"));
        });
        assert!(drive(&rig, Duration::from_millis(250), || rig
            .sink
            .event_names()
            .contains(&"device_reconnected")));
        assert!(drive(&rig, Duration::from_millis(250), || rig
            .handle
            .health()
            .state
            == LinkState::Ready));
        assert_eq!(rig.shared.with(|s| s.reconnects), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_watchdog_loses_a_worker_stuck_in_a_driver_call() {
        let rig = rig();
        rig.shared.with(|s| s.steps_per_poll = 1);
        let ticket = rig
            .handle
            .send(FocuserCmd::MoveTo(10_000), operator())
            .await
            .unwrap();
        rig.shared.with(|s| s.block_next_poll = true);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !rig.shared.with(|s| s.blocked) {
            assert!(Instant::now() < deadline);
            rig.clock.advance_ms(100);
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        // 3 s of silence degrades; 10 s loses and raises outcome_unknown.
        rig.clock.advance_ms(3_000);
        rig.hub.watchdog().tick(rig.clock.now_mono_ns());
        assert_eq!(rig.handle.health().state, LinkState::Degraded);
        rig.clock.advance_ms(10_000);
        rig.hub.watchdog().tick(rig.clock.now_mono_ns());
        assert_eq!(rig.handle.health().state, LinkState::Lost);
        assert!(!rig.hub.resume().active().is_empty());
        rig.shared.release();
        assert!(matches!(
            wait_outcome(&rig, ticket).await,
            OpOutcome::Unknown(_)
        ));
        let unknowns = rig
            .sink
            .event_names()
            .into_iter()
            .filter(|n| *n == "outcome_unknown")
            .count();
        assert_eq!(unknowns, 1, "announced once, by the watchdog");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn disconnect_halts_retires_and_closes_the_connection_row() {
        let rig = rig();
        rig.shared.with(|s| s.steps_per_poll = 1);
        let ticket = rig
            .handle
            .send(FocuserCmd::MoveTo(10_000), operator())
            .await
            .unwrap();
        rig.handle.disconnect(false).await.unwrap();
        assert!(matches!(ticket.wait().await, OpOutcome::Aborted { .. }));
        assert!(rig.handle.is_retired());
        assert!(rig.shared.with(|s| s.disconnected && s.halted >= 1));
        let records = rig.sink.records();
        assert!(records
            .iter()
            .any(|r| matches!(r, GroundTruthRecord::ConnectionOpened(_))));
        assert!(records.iter().any(|r| matches!(
            r,
            GroundTruthRecord::ConnectionClosed {
                end_class: ConnectionEnd::Operator,
                ..
            }
        )));
        assert!(rig.sink.event_names().contains(&"device_disconnected"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_aborts_moves_and_records_shutdown() {
        let mut rig = rig();
        rig.shared.with(|s| s.steps_per_poll = 1);
        let ticket = rig
            .handle
            .send(FocuserCmd::MoveTo(10_000), operator())
            .await
            .unwrap();
        let worker = rig.worker.take().unwrap();
        worker.stop();
        assert!(worker.join_timeout(Duration::from_secs(5)));
        assert_eq!(ticket.wait().await, OpOutcome::Aborted { by: None });
        assert!(rig.shared.with(|s| s.halted >= 1));
        assert!(rig.handle.is_retired());
        assert!(rig.sink.records().iter().any(|r| matches!(
            r,
            GroundTruthRecord::ConnectionClosed {
                end_class: ConnectionEnd::Shutdown,
                ..
            }
        )));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn connect_on_start_reports_the_result() {
        let clock = Arc::new(ManualClock::new());
        let shared = Shared::new();
        shared.with(|s| s.reconnect_fails = true);
        let identity = fake_identity(DeviceKind::Focuser);
        let driver = TestDriver {
            identity: identity.clone(),
            shared: Arc::clone(&shared),
            clock: Arc::clone(&clock),
            abort: FakeAbortPath::new(),
        };
        let (handle, end): (FocuserHandle, _) = DeviceHandle::new_pair(
            identity,
            None,
            FocuserSnapshot::default(),
            HealthState::connecting(0),
        );
        let hub = EquipmentHub::new(HubConfig::default(), clock.clone(), RecordingSink::new());
        let worker = spawn_worker(
            TestKind {
                settle: Duration::ZERO,
                deadline: Duration::from_secs(1),
                panic_on_step: false,
            },
            Box::new(driver) as Box<dyn FocuserDriver>,
            end,
            WorkerConfig {
                connect_on_start: true,
                max_wait: Duration::from_millis(2),
                ..WorkerConfig::default()
            },
            WorkerDeps {
                clock,
                events: hub.event_sink(),
                recorder: hub.recorder(),
                ring: None,
                watchdog: None,
                aborts: None,
            },
        )
        .unwrap();
        let err = worker.connected().await.unwrap_err();
        assert_eq!(err.class, ErrorClass::Transport);
        assert!(worker.join_timeout(Duration::from_secs(5)));
        assert!(handle.is_retired(), "a failed connect retires the handle");
    }

    // -- variants of the test kind ---------------------------------------------------

    /// The test kind, with its moves run as another operation kind and an
    /// optional halt.
    struct Variant {
        base: TestKind,
        op: OpKind,
        halt: bool,
    }

    impl Variant {
        fn new(op: OpKind, halt: bool, deadline: Duration) -> Self {
            Self {
                base: TestKind {
                    settle: Duration::ZERO,
                    deadline,
                    panic_on_step: false,
                },
                op,
                halt,
            }
        }
    }

    impl WorkerKind for Variant {
        type Driver = dyn FocuserDriver;
        type Cmd = FocuserCmd;
        type Snap = FocuserSnapshot;

        fn kind(&self) -> DeviceKind {
            DeviceKind::Focuser
        }

        fn has_halt(&self) -> bool {
            self.halt
        }

        fn plan(
            &mut self,
            cmd: FocuserCmd,
            drv: &mut Drv,
            ctx: &PlanContext<'_, FocuserSnapshot>,
        ) -> Plan<Drv> {
            match cmd {
                FocuserCmd::MoveTo(target) => {
                    let halt = self.halt;
                    let op = FnOperation::new(
                        self.op,
                        self.base.deadline,
                        move |drv: &mut Drv| drv.move_absolute(target),
                        |drv: &mut Drv| {
                            Ok(!drv.state()?.is_moving.known().copied().unwrap_or(true))
                        },
                        move |drv: &mut Drv| if halt { drv.halt() } else { Ok(()) },
                    )
                    .with_poll(Duration::from_millis(100));
                    Plan::Operation(Box::new(op))
                }
                other => self.base.plan(other, drv, ctx),
            }
        }

        fn snapshot(
            &mut self,
            drv: &mut Drv,
            poll: &RawPoll,
            ctx: &SnapshotContext<'_, FocuserSnapshot>,
        ) -> DeviceResult<FocuserSnapshot> {
            self.base.snapshot(drv, poll, ctx)
        }

        fn stale_snapshot(
            &mut self,
            previous: &FocuserSnapshot,
            ctx: &SnapshotContext<'_, FocuserSnapshot>,
        ) -> Option<FocuserSnapshot> {
            self.base.stale_snapshot(previous, ctx)
        }
    }

    /// Wait (real time) until the worker is blocked inside a poll.
    async fn wait_blocked(rig: &Rig) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !rig.shared.with(|s| s.blocked) {
            assert!(Instant::now() < deadline, "the worker never blocked");
            rig.clock.advance_ms(100);
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }

    /// The `operation_ended` outcomes recorded for `command_id`.
    fn ended(rig: &Rig, command_id: CommandId) -> Vec<OpOutcome> {
        rig.sink
            .records()
            .into_iter()
            .filter_map(|r| match r {
                GroundTruthRecord::Event(e) if e.command_id == Some(command_id) => match &e.event {
                    EventKind::OperationEnded { outcome, .. } => Some(outcome.clone()),
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    // -- abort and e-stop stop queued motion (D19) ------------------------------------

    fn test_kind() -> TestKind {
        TestKind {
            settle: Duration::ZERO,
            deadline: Duration::from_secs(60),
            panic_on_step: false,
        }
    }

    /// O11 e: an initiator that timed out (or hit a transport error) may
    /// have reached the device. The worker accepts the command, halts the
    /// hardware best effort and resolves it `Unknown`, which raises the
    /// hub's outcome_unknown resume block. A definite refusal stays a
    /// refusal.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_initiator_timeout_is_an_unknown_outcome_with_a_resume_block() {
        let rig = rig();
        assert!(drive(&rig, Duration::from_millis(100), || rig
            .shared
            .with(|s| s.polls)
            >= 1));
        rig.shared
            .with(|s| s.move_fails_with = Some(DeviceError::timeout("no reply to PUT move")));
        let ticket = rig
            .handle
            .send(FocuserCmd::MoveTo(500), operator())
            .await
            .expect("an unconfirmed initiator is accepted, not refused");
        let outcome = ticket.wait().await;
        assert!(
            matches!(&outcome, OpOutcome::Unknown(e) if e.class == ErrorClass::OutcomeUnknown),
            "{outcome:?}"
        );
        assert!(rig.shared.with(|s| s.halted) >= 1, "halted best effort");
        let blocks = rig.hub.resume().active();
        assert!(
            blocks.iter().any(|b| matches!(
                b.kind,
                crate::equipment::snapshot::ResumeBlockKind::OutcomeUnknown {
                    device: DeviceKind::Focuser,
                    ..
                }
            )),
            "{blocks:?}"
        );
        // A refusal from the device itself is a plain failure.
        rig.shared
            .with(|s| s.move_fails_with = Some(DeviceError::invalid_value("out of range")));
        let err = rig
            .handle
            .send(FocuserCmd::MoveTo(500), operator())
            .await
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::InvalidValue);
        assert!(initiator_outcome_unknown(
            OpKind::FocuserMove,
            &DeviceError::transport("socket reset")
        ));
        assert!(!initiator_outcome_unknown(
            OpKind::Refresh,
            &DeviceError::timeout("slow read")
        ));
    }

    /// O11 b: with `require_gate_permits`, motion needs a permit the gate
    /// issued; stops and halts never do.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn motion_needs_a_gate_permit_when_the_spine_requires_one() {
        let rig = rig_kind_with(test_kind(), HealthPolicy::default(), true);
        assert!(drive(&rig, Duration::from_millis(100), || rig
            .shared
            .with(|s| s.polls)
            >= 1));
        let err = rig
            .handle
            .send(FocuserCmd::MoveTo(300), operator())
            .await
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::InvalidOperation);
        assert!(
            err.message.contains("not issued by the command gate"),
            "{err}"
        );
        // A halt with an unmanaged permit still runs.
        rig.handle
            .send(FocuserCmd::Halt, operator())
            .await
            .expect("halts are never refused for their permit");
        // The gate's permit is accepted.
        let cmd = GatedCommand::new(
            GatedOp::Focuser(FocuserCmd::MoveTo(300)),
            CommandSource::Operator { overriding: false },
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        let permit = rig.hub.gate().authorize(&cmd).unwrap();
        let ticket = rig
            .handle
            .send(FocuserCmd::MoveTo(300), permit)
            .await
            .unwrap();
        assert!(drive(&rig, Duration::from_millis(100), || rig
            .shared
            .with(|s| s.position)
            == 300));
        assert_eq!(ticket.wait().await, OpOutcome::Succeeded);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_move_queued_before_an_estop_never_starts() {
        let rig = rig();
        rig.shared.with(|s| s.steps_per_poll = 1);
        assert!(drive(&rig, Duration::from_millis(100), || rig
            .shared
            .with(|s| s.polls)
            >= 1));
        // The worker is stuck in a slow poll while a move is queued.
        rig.shared.with(|s| s.block_next_poll = true);
        wait_blocked(&rig).await;
        let handle = rig.handle.clone();
        let queued =
            tokio::spawn(async move { handle.send(FocuserCmd::MoveTo(3_000), operator()).await });
        // Let `send` queue the envelope; it then waits for acceptance.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !queued.is_finished(),
            "the worker cannot accept while blocked"
        );
        let report = crate::equipment::estop::run_estop(
            &rig.hub,
            crate::equipment::estop::EstopRequest {
                abort_exposure: false,
                source: CommandSource::Operator { overriding: false },
            },
        )
        .await;
        assert!(report.all_ok(), "{report:?}");
        assert!(rig.shared.with(|s| s.blocked), "still inside the slow call");
        // The slow call returns: the cancel is handled, then the queued move
        // is refused, never started.
        rig.shared.release();
        let err = queued.await.unwrap().unwrap_err();
        assert_eq!(err.class, ErrorClass::Cancelled, "{err}");
        let polls = rig.shared.with(|s| s.polls);
        assert!(drive(&rig, Duration::from_millis(100), || rig
            .shared
            .with(|s| s.polls)
            > polls + 3));
        assert_eq!(rig.shared.with(|s| (s.position, s.moving)), (0, false));
        assert!(rig
            .sink
            .records()
            .iter()
            .any(|r| matches!(r, GroundTruthRecord::Event(e) if matches!(
                &e.event,
                EventKind::OperationEnded { op: OpKind::FocuserMove, outcome: OpOutcome::Aborted { by: Some(_) }, .. }
            ))));
        // A move sent after the e-stop runs.
        let ticket = rig
            .handle
            .send(FocuserCmd::MoveTo(3), operator())
            .await
            .unwrap();
        assert_eq!(wait_outcome(&rig, ticket).await, OpOutcome::Succeeded);
        assert_eq!(rig.shared.with(|s| s.position), 3);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_permit_issued_before_an_abort_never_starts_motion() {
        let rig = rig();
        let gate = Arc::clone(rig.hub.gate());
        let source = CommandSource::Operator { overriding: false };
        let move_to = |target: i32| {
            GatedCommand::new(
                GatedOp::Focuser(FocuserCmd::MoveTo(target)),
                source.clone(),
                AutonomyMode::Manual,
                Priority::Exposure,
            )
        };
        // Authorised, then delayed (for example by the D20 guider stop)
        // while the operator aborts the focuser.
        let early = move_to(500);
        let early_permit = gate.authorize(&early).unwrap();
        let abort = gate.abort(&rig.handle, source.clone());
        assert!(abort.outcome().await.is_ok());
        let err = rig
            .handle
            .send(FocuserCmd::MoveTo(500), early_permit)
            .await
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::Cancelled);
        assert_eq!(
            ended(&rig, early.command_id),
            vec![OpOutcome::Aborted {
                by: Some(abort_ticket_id(&rig))
            }]
        );
        assert_eq!(rig.shared.with(|s| s.position), 0);
        // A command authorised after the abort runs.
        let late = move_to(40);
        let permit = gate.authorize(&late).unwrap();
        let ticket = rig
            .handle
            .send(FocuserCmd::MoveTo(40), permit)
            .await
            .unwrap();
        assert_eq!(wait_outcome(&rig, ticket).await, OpOutcome::Succeeded);
        assert_eq!(rig.shared.with(|s| s.position), 40);
        // Stops and sets are never refused this way.
        let halt = gate
            .authorize(&GatedCommand::new(
                GatedOp::Focuser(FocuserCmd::Halt),
                source.clone(),
                AutonomyMode::Manual,
                Priority::Safety,
            ))
            .unwrap();
        let _ = gate.abort(&rig.handle, source.clone());
        let halted = rig.handle.send(FocuserCmd::Halt, halt).await.unwrap();
        assert_eq!(halted.wait().await, OpOutcome::Succeeded);
    }

    /// The id the worker resolved the latest abort with.
    fn abort_ticket_id(rig: &Rig) -> CommandId {
        rig.sink
            .records()
            .into_iter()
            .rev()
            .find_map(|r| match r {
                GroundTruthRecord::Event(e) => match &e.event {
                    EventKind::OperationEnded {
                        outcome: OpOutcome::Aborted { by: Some(by) },
                        ..
                    } => Some(*by),
                    _ => None,
                },
                _ => None,
            })
            .expect("an aborted operation")
    }

    // -- cancel confirmation, deadlines and shutdown ----------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cancel_is_confirmed_only_by_a_poll_that_read_not_moving() {
        let rig = rig();
        rig.shared.with(|s| s.steps_per_poll = 1);
        let ticket = rig
            .handle
            .send(FocuserCmd::MoveTo(10_000), operator())
            .await
            .unwrap();
        let id = ticket.command_id;
        // IsMoving can no longer be read, so no poll can confirm the halt.
        rig.shared.with(|s| s.moving_unknown = true);
        rig.handle.abort();
        let polls = rig.shared.with(|s| s.polls);
        assert!(drive(&rig, Duration::from_millis(100), || rig
            .shared
            .with(|s| s.polls)
            >= polls + 5));
        assert!(
            ended(&rig, id).is_empty(),
            "an unread IsMoving is not a confirmed stop"
        );
        // The confirmation timeout then ends it unknown.
        assert!(matches!(
            wait_outcome(&rig, ticket).await,
            OpOutcome::Unknown(_)
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_device_with_no_halt_that_overruns_its_deadline_is_unknown() {
        let rig = rig_kind(
            Variant::new(OpKind::FilterMove, false, Duration::from_secs(2)),
            HealthPolicy::default(),
        );
        rig.shared.with(|s| s.steps_per_poll = 1);
        let ticket = rig
            .handle
            .send(FocuserCmd::MoveTo(10_000), operator())
            .await
            .unwrap();
        let outcome = wait_outcome(&rig, ticket).await;
        assert!(
            matches!(&outcome, OpOutcome::Unknown(e) if e.class == ErrorClass::OutcomeUnknown),
            "{outcome:?}"
        );
        assert!(drive(&rig, Duration::from_millis(10), || rig
            .hub
            .resume()
            .active()
            .iter()
            .any(|b| matches!(
                b.kind,
                crate::equipment::snapshot::ResumeBlockKind::OutcomeUnknown {
                    op: OpKind::FilterMove,
                    ..
                }
            ))));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_halts_only_moves_jogs_and_slews() {
        for (op, halts) in [
            (OpKind::FocuserMove, true),
            (OpKind::RotatorMove, true),
            (OpKind::MoveAxis, true),
            (OpKind::Slew, true),
            (OpKind::SlewAltAz, true),
            (OpKind::Park, false),
            (OpKind::Unpark, false),
            (OpKind::Home, false),
            (OpKind::DomePark, false),
            (OpKind::DomeSlew, false),
            (OpKind::DomeOpenShutter, false),
            (OpKind::DomeCloseShutter, false),
            (OpKind::CoverOpen, false),
            (OpKind::CoverClose, false),
            (OpKind::FilterMove, false),
        ] {
            assert_eq!(halts_at_shutdown(op), halts, "{op:?}");
            let mut rig = rig_kind(
                Variant::new(op, true, Duration::from_secs(600)),
                HealthPolicy::default(),
            );
            rig.shared.with(|s| s.steps_per_poll = 1);
            let ticket = rig
                .handle
                .send(FocuserCmd::MoveTo(10_000), operator())
                .await
                .unwrap();
            let worker = rig.worker.take().unwrap();
            worker.stop();
            assert!(worker.join_timeout(Duration::from_secs(5)));
            let outcome = ticket.wait().await;
            let halted = rig.shared.with(|s| s.halted);
            if halts {
                assert_eq!(outcome, OpOutcome::Aborted { by: None }, "{op:?}");
                assert!(halted >= 1, "{op:?} is halted at shutdown");
            } else {
                assert!(
                    matches!(&outcome, OpOutcome::Unknown(e) if e.class == ErrorClass::OutcomeUnknown),
                    "{op:?}: {outcome:?}"
                );
                assert_eq!(halted, 0, "{op:?} is left running");
            }
            assert!(rig.handle.is_retired());
        }
    }

    // -- loss --------------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_watchdog_aborts_and_resolves_a_stuck_operation_at_once() {
        let rig = rig();
        rig.shared.with(|s| s.steps_per_poll = 1);
        let ticket = rig
            .handle
            .send(FocuserCmd::MoveTo(10_000), operator())
            .await
            .unwrap();
        let id = ticket.command_id;
        rig.shared.with(|s| s.block_next_poll = true);
        wait_blocked(&rig).await;
        rig.clock.advance_ms(15_000);
        rig.hub.watchdog().tick(rig.clock.now_mono_ns());
        assert_eq!(rig.handle.health().state, LinkState::Lost);
        // Without waiting for the hung call: the abort path runs and the
        // ticket resolves unknown.
        let deadline = Instant::now() + Duration::from_secs(2);
        while rig.abort.calls() == 0 {
            assert!(Instant::now() < deadline, "the abort path never ran");
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        let mut done = ticket.done;
        assert!(matches!(done.try_recv(), Ok(OpOutcome::Unknown(_))));
        assert!(rig.shared.with(|s| s.blocked), "the call is still hung");
        assert!(!rig.hub.resume().active().is_empty());
        // When the call returns, nothing is resolved or aborted twice.
        rig.shared.release();
        assert!(drive(&rig, Duration::from_millis(10), || rig
            .handle
            .snapshot()
            .state
            .position
            .state
            == ObsState::Stale));
        assert_eq!(rig.abort.calls(), 1);
        assert_eq!(ended(&rig, id).len(), 1);
        let unknowns = rig
            .sink
            .event_names()
            .into_iter()
            .filter(|n| *n == "outcome_unknown")
            .count();
        assert_eq!(unknowns, 1);
    }

    #[test]
    fn a_lost_device_publishes_its_last_values_as_stale() {
        let rig = rig();
        assert!(drive(&rig, Duration::from_millis(100), || rig
            .shared
            .with(|s| s.polls)
            >= 2));
        assert_eq!(rig.handle.snapshot().state.position.state, ObsState::Fresh);
        // NotConnected, and the single reconnect fails: lost at once.
        rig.shared.with(|s| {
            s.fail_polls = 1000;
            s.fail_with = Some(DeviceError::not_connected("Connected=false"));
            s.reconnect_fails = true;
        });
        assert!(drive(&rig, Duration::from_millis(250), || rig
            .handle
            .health()
            .state
            == LinkState::Lost));
        rig.clock.advance(Duration::from_secs(600));
        let snap = rig.handle.snapshot();
        assert_eq!(snap.state.position.state, ObsState::Stale);
        assert_eq!(snap.state.position.value, Some(0), "the last value is kept");
        assert_eq!(
            rig.hub
                .rings()
                .focuser
                .latest()
                .unwrap()
                .state
                .position
                .state,
            ObsState::Stale
        );
    }
}
