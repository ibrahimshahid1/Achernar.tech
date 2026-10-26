//! The per-kind halves of the device worker (spec §2.5, O11 i). Owner:
//! WP1B-1 (the spine).
//!
//! Wave A built each device's behaviour as a runtime of its own:
//! [`MountRuntime`], [`FocuserSession`], the filter-wheel operation and
//! publisher, and the weather and safety snapshot builders. These adapters
//! put each of them behind [`WorkerKind`], so every polled device runs on the
//! one [`DeviceWorker`](super::worker) and gets the same health, abort,
//! panic-containment and shutdown behaviour. The PHD2 guider keeps its own
//! tokio service (`guider/service.rs`), which honours the same handle
//! contract.
//!
//! The adapters add no device behaviour of their own; each one only
//! translates between the worker's calls and its runtime's API.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use super::commands::{FilterWheelCmd, FocuserCmd, MountCmd, SafetyCmd, WeatherCmd};
use super::events::{ConnectionEnd, EventKind};
use super::filter::motion::{FilterMoveOp, FilterWheelPublisher};
use super::filter::{read_wheel, FilterSettings, SharedWheel};
use super::focuser::{FocuserSession, TEMPCOMP_MEMBER, TEMPCOMP_REFRESH};
use super::hub::EventSink;
use super::mount::{stale_state, CommandMeta, MountPlan, MountRuntime};
use super::ops::{OpKind, OpProgress, Operation};
use super::safety::{safety_snapshot, weather_snapshot};
use super::snapshot::{
    FilterWheelSnapshot, FocuserSnapshot, MountSnapshot, SafetySnapshot, WeatherReading,
    WeatherSnapshot,
};
use super::traits::{
    Clock, FilterWheelDriver, FocuserDriver, MountDriver, RawPoll, SafetyDriver, WeatherDriver,
};
use super::types::{DeviceError, DeviceKind, DeviceResult, Observed, OpOutcome, Source};
use super::worker::{default_cadence, Cadence, Plan, PlanContext, SnapshotContext, WorkerKind};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

fn ns(duration: Duration) -> u64 {
    super::clock::duration_ns(duration)
}

/// A plan that resolves at once with `outcome` (a command the runtime
/// absorbed into the running operation).
fn absorbed<D: ?Sized + 'static>(outcome: OpOutcome) -> Plan<D> {
    Plan::Concurrent(Box::new(move |_drv: &mut D| match outcome {
        OpOutcome::Failed(err) | OpOutcome::Unknown(err) => Err(err),
        _ => Ok(()),
    }))
}

// ---------------------------------------------------------------------------
// Mount
// ---------------------------------------------------------------------------

/// The mount behind the worker: [`MountRuntime`] plans every command, builds
/// every snapshot and runs the limit watchdog; its event outbox is drained
/// into the hub after every call.
pub struct MountKind {
    runtime: MountRuntime,
    events: Arc<dyn EventSink>,
    clock: Arc<dyn Clock>,
    cadence: Cadence,
}

impl MountKind {
    pub fn new(runtime: MountRuntime, events: Arc<dyn EventSink>, clock: Arc<dyn Clock>) -> Self {
        // The runtime's events (its operations' `slew_settled`, `parked`,
        // the watchdog's `limit_reached`) reach the hub as they are raised.
        let sink = Arc::clone(&events);
        runtime
            .shared()
            .events()
            .set_forward(Some(Arc::new(move |event| {
                sink.emit(event);
            })));
        Self {
            runtime,
            events,
            clock,
            cadence: default_cadence(DeviceKind::Mount),
        }
    }

    /// Override the §2.7 cadence (tests).
    pub fn with_cadence(mut self, cadence: Cadence) -> Self {
        self.cadence = cadence;
        self
    }

    /// Publish what the runtime and its operations emitted.
    fn flush(&self) {
        for event in self.runtime.take_events() {
            self.events.emit(event);
        }
    }
}

impl WorkerKind for MountKind {
    type Driver = dyn MountDriver;
    type Cmd = MountCmd;
    type Snap = MountSnapshot;

    fn kind(&self) -> DeviceKind {
        DeviceKind::Mount
    }

    fn cadence(&self) -> Cadence {
        self.cadence
    }

    fn capabilities_json(&mut self, _drv: &mut Self::Driver) -> serde_json::Value {
        serde_json::to_value(self.runtime.shared().capabilities()).unwrap_or_default()
    }

    fn plan(
        &mut self,
        cmd: MountCmd,
        _drv: &mut Self::Driver,
        ctx: &PlanContext<'_, MountSnapshot>,
    ) -> Plan<Self::Driver> {
        let meta = CommandMeta::new(ctx.command_id, ctx.source.clone()).with_priority(ctx.priority);
        let plan = match self.runtime.plan(cmd, meta, ctx.active) {
            MountPlan::Start(op) => Plan::Operation(op),
            MountPlan::Absorbed(outcome) => absorbed(outcome),
            MountPlan::CancelActive(mut op) => {
                // AbortSlew while an operation runs: the worker cancels it
                // (`Aborted { by }`, confirmed by a poll that reads Slewing
                // false), then this sends AbortSlew plus MoveAxis 0 on the
                // jogging axes.
                let clock = Arc::clone(&self.clock);
                Plan::Halt(Box::new(move |drv: &mut Self::Driver| {
                    op.start(drv, clock.as_ref())
                }))
            }
            MountPlan::Refused(err) => Plan::Reject(err),
        };
        self.flush();
        plan
    }

    fn snapshot(
        &mut self,
        drv: &mut Self::Driver,
        _poll: &RawPoll,
        ctx: &SnapshotContext<'_, MountSnapshot>,
    ) -> DeviceResult<MountSnapshot> {
        let snapshot = self.runtime.after_poll(drv, ctx.active_op.cloned());
        self.flush();
        Ok(snapshot)
    }

    fn stale_snapshot(
        &mut self,
        previous: &MountSnapshot,
        _ctx: &SnapshotContext<'_, MountSnapshot>,
    ) -> Option<MountSnapshot> {
        self.flush();
        let mut stale = previous.clone();
        stale.state = stale_state(&previous.state);
        Some(stale)
    }

    fn edges(&mut self, _previous: &MountSnapshot, _next: &MountSnapshot) -> Vec<EventKind> {
        // The runtime emits its own edge events (M-j); they were flushed with
        // the snapshot.
        self.flush();
        Vec::new()
    }

    fn after_reconnect(&mut self, drv: &mut Self::Driver) {
        // Capabilities and the site/time check again (M-a, M-b); connecting
        // never moves hardware (D16).
        if let Err(err) = self.runtime.on_connected(drv) {
            tracing::warn!("mount post-connect after the reconnect failed: {err}");
        }
        self.flush();
    }

    fn halt_on_disconnect(&mut self, drv: &mut Self::Driver) -> DeviceResult<()> {
        // Reached only for a forced disconnect when a slew is in progress
        // (`refuses_disconnect` refuses a plain one): abort it and stop any
        // jog. Disconnect never parks.
        let slewing = self
            .runtime
            .last_snapshot()
            .and_then(|s| s.state.slewing.known().copied())
            == Some(true);
        let result = if slewing || self.runtime.shared().jog_active() {
            self.runtime.prepare_disconnect(drv)
        } else {
            Ok(())
        };
        self.flush();
        result
    }

    fn refuses_disconnect(&self, last: Option<&MountSnapshot>) -> bool {
        let active = last.and_then(|s| s.active_op.as_ref().map(|op| op.kind));
        self.runtime.check_disconnect(active, false).is_err()
    }
}

// ---------------------------------------------------------------------------
// Focuser
// ---------------------------------------------------------------------------

/// The focuser behind the worker: [`FocuserSession`] builds the operations
/// and the snapshot; its events go to the hub through the session's sink.
pub struct FocuserKind {
    session: FocuserSession,
    clock: Arc<dyn Clock>,
    next_tempcomp_mono_ns: u64,
    cadence: Cadence,
}

impl FocuserKind {
    pub fn new(session: FocuserSession, clock: Arc<dyn Clock>) -> Self {
        Self {
            session,
            clock,
            next_tempcomp_mono_ns: 0,
            cadence: default_cadence(DeviceKind::Focuser),
        }
    }

    pub fn with_cadence(mut self, cadence: Cadence) -> Self {
        self.cadence = cadence;
        self
    }

    pub fn session(&self) -> &FocuserSession {
        &self.session
    }
}

impl WorkerKind for FocuserKind {
    type Driver = dyn FocuserDriver;
    type Cmd = FocuserCmd;
    type Snap = FocuserSnapshot;

    fn kind(&self) -> DeviceKind {
        DeviceKind::Focuser
    }

    fn cadence(&self) -> Cadence {
        self.cadence
    }

    /// TempComp is not in DeviceState; it is read every 10 s (§2.7).
    fn extra_members(&mut self, _tick: u64) -> Vec<String> {
        let now = self.clock.now_mono_ns();
        if now >= self.next_tempcomp_mono_ns {
            self.next_tempcomp_mono_ns = now.saturating_add(ns(TEMPCOMP_REFRESH));
            vec![TEMPCOMP_MEMBER.to_string()]
        } else {
            Vec::new()
        }
    }

    fn capabilities_json(&mut self, _drv: &mut Self::Driver) -> serde_json::Value {
        serde_json::to_value(self.session.capabilities()).unwrap_or_default()
    }

    fn plan(
        &mut self,
        cmd: FocuserCmd,
        _drv: &mut Self::Driver,
        ctx: &PlanContext<'_, FocuserSnapshot>,
    ) -> Plan<Self::Driver> {
        if cmd == FocuserCmd::Halt && ctx.active.is_some() {
            // The running move's cancel halts and records it; the halt is
            // sent again so a move another client started stops too.
            return Plan::Halt(Box::new(|drv: &mut Self::Driver| drv.halt()));
        }
        match self
            .session
            .operation(&cmd, ctx.command_id, ctx.source.clone())
        {
            Ok(op) => Plan::Operation(op),
            Err(err) => Plan::Reject(err),
        }
    }

    fn snapshot(
        &mut self,
        drv: &mut Self::Driver,
        poll: &RawPoll,
        ctx: &SnapshotContext<'_, FocuserSnapshot>,
    ) -> DeviceResult<FocuserSnapshot> {
        Ok(self
            .session
            .after_poll(drv, Ok(poll), ctx.active_op.cloned()))
    }

    fn stale_snapshot(
        &mut self,
        previous: &FocuserSnapshot,
        _ctx: &SnapshotContext<'_, FocuserSnapshot>,
    ) -> Option<FocuserSnapshot> {
        let mut stale = previous.clone();
        let state = &mut stale.state;
        state.position = std::mem::take(&mut state.position).into_stale();
        state.is_moving = std::mem::take(&mut state.is_moving).into_stale();
        state.temperature_c = std::mem::take(&mut state.temperature_c).into_stale();
        state.temp_comp = std::mem::take(&mut state.temp_comp).into_stale();
        Some(stale)
    }

    fn after_reconnect(&mut self, _drv: &mut Self::Driver) {
        self.session.on_reconnect();
    }

    fn halt_on_disconnect(&mut self, drv: &mut Self::Driver) -> DeviceResult<()> {
        let moving = drv
            .state()
            .ok()
            .and_then(|s| s.is_moving.known().copied())
            .unwrap_or(true);
        let result = if moving { drv.halt() } else { Ok(()) };
        self.session.on_disconnect(ConnectionEnd::Operator);
        result
    }
}

// ---------------------------------------------------------------------------
// Filter wheel
// ---------------------------------------------------------------------------

/// The wheel driver type the worker drives (the alias fixes the trait
/// object's lifetime to `'static`, as in `Operation<dyn FilterWheelDriver>`).
type DynWheel = dyn FilterWheelDriver;

/// A wheel move that reports its result to the publisher, which tracks the
/// verified slot (W-c step 7).
struct TrackedFilterMove {
    op: FilterMoveOp,
    publisher: Arc<Mutex<FilterWheelPublisher>>,
}

impl Operation<DynWheel> for TrackedFilterMove {
    fn kind(&self) -> OpKind {
        <FilterMoveOp as Operation<DynWheel>>::kind(&self.op)
    }

    fn deadline(&self) -> Duration {
        <FilterMoveOp as Operation<DynWheel>>::deadline(&self.op)
    }

    fn start(&mut self, drv: &mut DynWheel, clock: &dyn Clock) -> DeviceResult<()> {
        let result = self.op.start(drv, clock);
        if let Err(err) = &result {
            // The initiator failed: the slot is not verified any more.
            lock(&self.publisher).on_op_done(self.op.target(), &OpOutcome::Failed(err.clone()));
        }
        result
    }

    fn step(&mut self, drv: &mut DynWheel, clock: &dyn Clock) -> OpProgress {
        let progress = self.op.step(drv, clock);
        if let OpProgress::Done(outcome) = &progress {
            lock(&self.publisher).on_op_done(self.op.target(), outcome);
        }
        progress
    }

    fn cancel(&mut self, drv: &mut DynWheel) -> DeviceResult<()> {
        // A wheel has no halt (C-4): its final slot is unknown.
        lock(&self.publisher).on_op_done(
            self.op.target(),
            &OpOutcome::Unknown(DeviceError::outcome_unknown(
                "the wheel move was cancelled; a filter wheel has no halt",
            )),
        );
        self.op.cancel(drv)
    }
}

/// The filter wheel behind the worker: the W-c move operation, and the
/// publisher that keeps the verified slot and the reconciled slot list.
pub struct FilterWheelKind {
    settings: FilterSettings,
    model: SharedWheel,
    publisher: Arc<Mutex<FilterWheelPublisher>>,
    clock: Arc<dyn Clock>,
    cadence: Cadence,
}

impl FilterWheelKind {
    pub fn new(settings: FilterSettings, model: SharedWheel, clock: Arc<dyn Clock>) -> Self {
        Self {
            settings,
            model,
            publisher: Arc::new(Mutex::new(FilterWheelPublisher::new())),
            clock,
            cadence: default_cadence(DeviceKind::FilterWheel),
        }
    }

    pub fn with_cadence(mut self, cadence: Cadence) -> Self {
        self.cadence = cadence;
        self
    }
}

impl WorkerKind for FilterWheelKind {
    type Driver = dyn FilterWheelDriver;
    type Cmd = FilterWheelCmd;
    type Snap = FilterWheelSnapshot;

    fn kind(&self) -> DeviceKind {
        DeviceKind::FilterWheel
    }

    fn cadence(&self) -> Cadence {
        self.cadence
    }

    fn has_halt(&self) -> bool {
        false
    }

    fn capabilities_json(&mut self, _drv: &mut Self::Driver) -> serde_json::Value {
        let wheel = read_wheel(&self.model);
        serde_json::json!({
            "slots": wheel.filter_slots(),
            "differences": wheel.differences,
        })
    }

    fn plan(
        &mut self,
        cmd: FilterWheelCmd,
        _drv: &mut Self::Driver,
        _ctx: &PlanContext<'_, FilterWheelSnapshot>,
    ) -> Plan<Self::Driver> {
        let FilterWheelCmd::SetPosition(target) = cmd;
        let slots = read_wheel(&self.model).filter_slots().len();
        if slots > 0 && usize::try_from(target).map_or(true, |t| t >= slots) {
            return Plan::Reject(DeviceError::invalid_value(format!(
                "slot {target} is outside the wheel's {slots} slots (0-based)"
            )));
        }
        Plan::Operation(Box::new(TrackedFilterMove {
            op: FilterMoveOp::new(target, &self.settings),
            publisher: Arc::clone(&self.publisher),
        }))
    }

    fn snapshot(
        &mut self,
        drv: &mut Self::Driver,
        poll: &RawPoll,
        ctx: &SnapshotContext<'_, FilterWheelSnapshot>,
    ) -> DeviceResult<FilterWheelSnapshot> {
        let previous = ctx.previous.cloned().unwrap_or_default();
        let position = drv.position();
        let slots = read_wheel(&self.model).filter_slots();
        Ok(lock(&self.publisher).build(
            &previous,
            position,
            Some(poll),
            slots,
            ctx.active_op.cloned(),
            self.clock.as_ref(),
        ))
    }

    fn stale_snapshot(
        &mut self,
        previous: &FilterWheelSnapshot,
        _ctx: &SnapshotContext<'_, FilterWheelSnapshot>,
    ) -> Option<FilterWheelSnapshot> {
        let mut stale = previous.clone();
        stale.position = std::mem::take(&mut stale.position).into_stale();
        Some(stale)
    }

    fn after_reconnect(&mut self, _drv: &mut Self::Driver) {
        // §2.6: the verified slot is re-read before any move.
        lock(&self.publisher).on_connect();
    }
}

// ---------------------------------------------------------------------------
// Weather
// ---------------------------------------------------------------------------

/// A weather reading with every value marked stale (C-2).
pub fn stale_weather(reading: &WeatherReading) -> WeatherReading {
    let mut r = reading.clone();
    for value in [
        &mut r.cloud_cover_pct,
        &mut r.dew_point_c,
        &mut r.humidity_pct,
        &mut r.pressure_hpa,
        &mut r.rain_rate_mm_h,
        &mut r.sky_brightness_lux,
        &mut r.sky_quality_mag_arcsec2,
        &mut r.sky_temperature_c,
        &mut r.star_fwhm_arcsec,
        &mut r.temperature_c,
        &mut r.wind_direction_deg,
        &mut r.wind_gust_ms,
        &mut r.wind_speed_ms,
    ] {
        *value = std::mem::take(value).into_stale();
    }
    r
}

/// An ObservingConditions device behind the worker. The only command is an
/// explicit refresh.
pub struct WeatherKind {
    clock: Arc<dyn Clock>,
    cadence: Cadence,
}

impl WeatherKind {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            clock,
            cadence: default_cadence(DeviceKind::Weather),
        }
    }

    pub fn with_cadence(mut self, cadence: Cadence) -> Self {
        self.cadence = cadence;
        self
    }
}

impl WorkerKind for WeatherKind {
    type Driver = dyn WeatherDriver;
    type Cmd = WeatherCmd;
    type Snap = WeatherSnapshot;

    fn kind(&self) -> DeviceKind {
        DeviceKind::Weather
    }

    fn cadence(&self) -> Cadence {
        self.cadence
    }

    fn plan(
        &mut self,
        cmd: WeatherCmd,
        _drv: &mut Self::Driver,
        _ctx: &PlanContext<'_, WeatherSnapshot>,
    ) -> Plan<Self::Driver> {
        match cmd {
            WeatherCmd::Refresh => {
                Plan::Concurrent(Box::new(|drv: &mut Self::Driver| drv.refresh()))
            }
        }
    }

    fn snapshot(
        &mut self,
        drv: &mut Self::Driver,
        _poll: &RawPoll,
        ctx: &SnapshotContext<'_, WeatherSnapshot>,
    ) -> DeviceResult<WeatherSnapshot> {
        weather_snapshot(drv, ctx.identity.connection_id, self.clock.as_ref())
    }

    fn stale_snapshot(
        &mut self,
        previous: &WeatherSnapshot,
        _ctx: &SnapshotContext<'_, WeatherSnapshot>,
    ) -> Option<WeatherSnapshot> {
        Some(WeatherSnapshot {
            reading: stale_weather(&previous.reading),
            ..previous.clone()
        })
    }
}

// ---------------------------------------------------------------------------
// Safety monitor
// ---------------------------------------------------------------------------

/// A SafetyMonitor behind the worker. Never cached (§3.10): a failed poll
/// publishes an unknown level in state `Error`, never the previous level.
pub struct SafetyKind {
    clock: Arc<dyn Clock>,
    cadence: Cadence,
}

impl SafetyKind {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            clock,
            cadence: default_cadence(DeviceKind::SafetyMonitor),
        }
    }

    pub fn with_cadence(mut self, cadence: Cadence) -> Self {
        self.cadence = cadence;
        self
    }
}

impl WorkerKind for SafetyKind {
    type Driver = dyn SafetyDriver;
    type Cmd = SafetyCmd;
    type Snap = SafetySnapshot;

    fn kind(&self) -> DeviceKind {
        DeviceKind::SafetyMonitor
    }

    fn cadence(&self) -> Cadence {
        self.cadence
    }

    fn plan(
        &mut self,
        cmd: SafetyCmd,
        _drv: &mut Self::Driver,
        _ctx: &PlanContext<'_, SafetySnapshot>,
    ) -> Plan<Self::Driver> {
        match cmd {
            // The next poll re-reads IsSafe; nothing is cached to refresh.
            SafetyCmd::Refresh => Plan::Concurrent(Box::new(|_drv: &mut Self::Driver| Ok(()))),
        }
    }

    fn snapshot(
        &mut self,
        drv: &mut Self::Driver,
        poll: &RawPoll,
        ctx: &SnapshotContext<'_, SafetySnapshot>,
    ) -> DeviceResult<SafetySnapshot> {
        Ok(safety_snapshot(
            drv,
            &Ok(poll.clone()),
            ctx.identity.connection_id,
            self.clock.as_ref(),
        ))
    }

    fn stale_snapshot(
        &mut self,
        previous: &SafetySnapshot,
        _ctx: &SnapshotContext<'_, SafetySnapshot>,
    ) -> Option<SafetySnapshot> {
        Some(SafetySnapshot {
            level: Observed::error(Source::Property),
            detail: Some("the safety monitor poll failed; safety is unknown".into()),
            published_at: self.clock.now_utc(),
            published_mono_ns: self.clock.now_mono_ns(),
            ..previous.clone()
        })
    }
}
