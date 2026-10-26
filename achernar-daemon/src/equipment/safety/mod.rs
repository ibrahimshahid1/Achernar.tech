//! Environment and safety (spec §3.9, §3.10, §2.8 fail-safe rule). Owner:
//! WP1A-8.
//!
//! - [`monitor_hub`]: the safety hub. With N monitors the result is safe
//!   only if every one is connected, Ready and reads safe; with none it is
//!   `none`. It detects the edges that start the on-unsafe policy (a
//!   *confirmed* Unsafe reading, or a monitor going `Lost`) and provides the
//!   [`monitor_hub::SafetyGateCheck`] plug-in that refuses new automatic
//!   actions while safety is not `Safe` or `None`.
//! - [`policy`]: the on-unsafe policy executor (through the gate at safety
//!   priority, every action audited) and the [`policy::SafetySupervisor`]
//!   that ties monitors, hub and policy together.
//! - This file: the ports the spine (WP1B-1) implements over the hub, the
//!   gate and the resume-block store (WP1A-2), plus inert implementations
//!   for tests and for use before the spine lands; the snapshot builders a
//!   worker calls after a weather or safety poll; the `environment_log`
//!   writer ([`WeatherLogger`]); and the COC corroboration helper (H14).
//!
//! Nothing here opens a device or starts a task by default (D13).

pub mod monitor_hub;
pub mod policy;

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use futures::future::BoxFuture;
use tokio::sync::watch;
use uuid::Uuid;

use crate::config::WeatherConfig;
use crate::equipment::alpaca::observingconditions::WeatherField;
use crate::equipment::commands::{
    CoverCalibratorCmd, DomeCmd, GateError, GatedCommand, GatedOp, GuiderCmd, MountCmd,
};
use crate::equipment::events::{
    AuditRecord, AuditStage, EnvironmentRecord, EquipmentEvent, EventKind, GroundTruthRecord,
};
use crate::equipment::handle::{Permit, WeatherHandle};
use crate::equipment::snapshot::{
    CocCorroboration, CorroborationLevel, ResumeBlock, ResumeBlockKind, SafetyGate, SafetyLevel,
    SafetySnapshot, SafetySummary, WeatherReading, WeatherSnapshot,
};
use crate::equipment::traits::{
    Clock, PollMethod, RawPoll, RecorderSink, SafetyDriver, WeatherDriver,
};
use crate::equipment::types::{
    CommandId, CommandSource, DeviceIdentity, DeviceKind, DeviceResult, ObsState, Observed, Source,
};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// Ports (implemented by the spine over WP1A-2's hub, gate and resume store)
// ---------------------------------------------------------------------------

/// Where environment and safety events go. The hub implements it: it
/// assigns `seq`, broadcasts to subscribers and journals the event.
pub trait EventSink: Send + Sync {
    fn emit(&self, event: EquipmentEvent);
}

/// What to raise (H22).
#[derive(Debug, Clone, PartialEq)]
pub struct BlockRequest {
    pub kind: ResumeBlockKind,
    pub reason: String,
    pub device_kind: Option<DeviceKind>,
    pub connection_id: Option<Uuid>,
    pub command_id: Option<CommandId>,
}

impl BlockRequest {
    pub fn new(kind: ResumeBlockKind, reason: impl Into<String>) -> Self {
        Self {
            kind,
            reason: reason.into(),
            device_kind: None,
            connection_id: None,
            command_id: None,
        }
    }

    pub fn for_device(mut self, kind: DeviceKind, connection_id: Uuid) -> Self {
        self.device_kind = Some(kind);
        self.connection_id = Some(connection_id);
        self
    }

    pub fn with_command(mut self, command_id: CommandId) -> Self {
        self.command_id = Some(command_id);
        self
    }
}

/// The resume-block store (`resume.rs`, WP1A-2). It emits
/// `resume_blocked` and `resume_cleared` itself.
pub trait ResumeBlockStore: Send + Sync {
    /// Raise a block. While a block of the same kind is active the store
    /// returns it instead of adding a duplicate.
    fn raise(&self, request: BlockRequest) -> ResumeBlock;
    /// Clear one block; `None` when it was not active.
    fn clear(&self, id: Uuid, by: &CommandSource) -> Option<ResumeBlock>;
    fn active(&self) -> Vec<ResumeBlock>;
}

/// The command gate as the safety policy uses it (`gate.rs`, WP1A-2).
///
/// `authorize` runs the gate's steps for a [`CommandSource::Safety`] command
/// at [`Priority::Safety`](crate::equipment::types::Priority) (rank 6, so it
/// preempts every holder below safety) and writes the `proposed` and
/// `validated` or `rejected` audit rows. Abort-type commands skip ownership,
/// arbiter and checks but are still audited (§2.9). The policy reports the
/// `executed` and `outcome` stages through [`SafetyGatePort::audit`].
pub trait SafetyGatePort: Send + Sync {
    fn authorize<'a>(&'a self, cmd: &'a GatedCommand) -> BoxFuture<'a, Result<Permit, GateError>>;
    fn audit(&self, record: AuditRecord);
}

// ---------------------------------------------------------------------------
// Inert and stand-in implementations
// ---------------------------------------------------------------------------

/// Journals events through a `RecorderSink` only (no live subscribers). For
/// tests, and for an environment layer running before the hub is wired.
pub struct RecorderEventSink {
    recorder: Arc<dyn RecorderSink>,
}

impl RecorderEventSink {
    pub fn new(recorder: Arc<dyn RecorderSink>) -> Self {
        Self { recorder }
    }
}

impl EventSink for RecorderEventSink {
    fn emit(&self, event: EquipmentEvent) {
        self.recorder
            .record(GroundTruthRecord::Event(Arc::new(event)));
    }
}

/// An in-memory resume-block store with a watch of the active list. It
/// stands in for `resume.rs` in tests and emits the same events.
pub struct MemoryResumeBlocks {
    blocks: Mutex<Vec<ResumeBlock>>,
    watch: watch::Sender<Vec<ResumeBlock>>,
    events: Arc<dyn EventSink>,
    clock: Arc<dyn Clock>,
}

impl MemoryResumeBlocks {
    pub fn new(events: Arc<dyn EventSink>, clock: Arc<dyn Clock>) -> Self {
        let (watch, _) = watch::channel(Vec::new());
        Self {
            blocks: Mutex::new(Vec::new()),
            watch,
            events,
            clock,
        }
    }

    pub fn subscribe(&self) -> watch::Receiver<Vec<ResumeBlock>> {
        self.watch.subscribe()
    }
}

impl ResumeBlockStore for MemoryResumeBlocks {
    fn raise(&self, request: BlockRequest) -> ResumeBlock {
        let mut blocks = lock(&self.blocks);
        if let Some(existing) = blocks.iter().find(|b| b.kind == request.kind) {
            return existing.clone();
        }
        let block = ResumeBlock {
            id: Uuid::new_v4(),
            kind: request.kind,
            raised_at: self.clock.now_utc(),
            raised_mono_ns: self.clock.now_mono_ns(),
            device_kind: request.device_kind,
            connection_id: request.connection_id,
            command_id: request.command_id,
            reason: request.reason,
        };
        blocks.push(block.clone());
        self.watch.send_replace(blocks.clone());
        drop(blocks);
        let mut event = EquipmentEvent::new(
            EventKind::ResumeBlocked {
                block: block.clone(),
            },
            self.clock.as_ref(),
        );
        event.command_id = block.command_id;
        self.events.emit(event);
        block
    }

    fn clear(&self, id: Uuid, by: &CommandSource) -> Option<ResumeBlock> {
        let mut blocks = lock(&self.blocks);
        let index = blocks.iter().position(|b| b.id == id)?;
        let block = blocks.remove(index);
        self.watch.send_replace(blocks.clone());
        drop(blocks);
        self.events.emit(EquipmentEvent::new(
            EventKind::ResumeCleared {
                block: block.clone(),
                by: by.clone(),
            },
            self.clock.as_ref(),
        ));
        Some(block)
    }

    fn active(&self) -> Vec<ResumeBlock> {
        lock(&self.blocks).clone()
    }
}

/// A gate stand-in for the safety policy only: it grants a permit that holds
/// no ownership and runs no checks, and audits every stage to the recorder.
/// It refuses every source other than [`CommandSource::Safety`], so it can
/// never become a bypass for ordinary commands. The spine replaces it with
/// the `CommandGate` (WP1A-2), which also preempts lower-priority holders.
pub struct AuditingSafetyGate {
    recorder: Arc<dyn RecorderSink>,
    clock: Arc<dyn Clock>,
}

impl AuditingSafetyGate {
    pub fn new(recorder: Arc<dyn RecorderSink>, clock: Arc<dyn Clock>) -> Self {
        Self { recorder, clock }
    }

    fn record(
        &self,
        cmd: &GatedCommand,
        stage: AuditStage,
        accepted: bool,
        reason: Option<String>,
    ) {
        self.audit(AuditRecord::for_command(
            cmd,
            stage,
            accepted,
            reason,
            self.clock.now_utc(),
        ));
    }
}

impl SafetyGatePort for AuditingSafetyGate {
    fn authorize<'a>(&'a self, cmd: &'a GatedCommand) -> BoxFuture<'a, Result<Permit, GateError>> {
        Box::pin(async move {
            self.record(cmd, AuditStage::Proposed, true, None);
            if cmd.source != CommandSource::Safety {
                let reason = "the safety gate stand-in only authorises safety commands".to_string();
                self.record(cmd, AuditStage::Rejected, false, Some(reason.clone()));
                return Err(GateError::Refused { reason });
            }
            self.record(cmd, AuditStage::Validated, true, None);
            Ok(Permit::new(cmd, None))
        })
    }

    fn audit(&self, record: AuditRecord) {
        self.recorder
            .record(GroundTruthRecord::Audit(Box::new(record)));
    }
}

// ---------------------------------------------------------------------------
// Snapshot builders (called by the worker after a poll)
// ---------------------------------------------------------------------------

/// The `WeatherSnapshot` a worker publishes after a weather poll. The
/// reading keeps the driver's `Observed` states, so a failed poll publishes
/// stale values, never fresh ones (C-2).
pub fn weather_snapshot(
    driver: &mut dyn WeatherDriver,
    connection_id: Uuid,
    clock: &dyn Clock,
) -> DeviceResult<WeatherSnapshot> {
    Ok(WeatherSnapshot {
        connection_id,
        reading: driver.read()?,
        published_at: clock.now_utc(),
        published_mono_ns: clock.now_mono_ns(),
    })
}

/// The `SafetySnapshot` a worker publishes after a safety poll. Never
/// cached (§3.10): a failed poll or read publishes an unknown level in state
/// `Error`, never the previous level.
pub fn safety_snapshot(
    driver: &mut dyn SafetyDriver,
    poll: &DeviceResult<RawPoll>,
    connection_id: Uuid,
    clock: &dyn Clock,
) -> SafetySnapshot {
    let published_at = clock.now_utc();
    let published_mono_ns = clock.now_mono_ns();
    let source = match poll {
        Ok(raw) if raw.method == PollMethod::DeviceState => Source::DeviceState,
        Ok(raw) if raw.method == PollMethod::PushCache => Source::IndiPush,
        _ => Source::Property,
    };
    let read = match poll {
        Ok(raw) => driver.is_safe().map(|reading| (raw, reading)),
        Err(err) => Err(err.clone()),
    };
    match read {
        Ok((raw, reading)) => {
            let member = raw
                .members
                .iter()
                .find(|m| m.name.eq_ignore_ascii_case("issafe"))
                .or_else(|| raw.members.first());
            SafetySnapshot {
                connection_id,
                level: Observed {
                    value: Some(reading.level),
                    state: ObsState::Fresh,
                    observed_at: Some(member.and_then(|m| m.observed_at).unwrap_or(published_at)),
                    mono_ns: Some(member.and_then(|m| m.mono_ns).unwrap_or(published_mono_ns)),
                    device_time: raw.device_time,
                    latency_ms: member.and_then(|m| m.latency_ms),
                    source,
                },
                detail: reading.detail,
                published_at,
                published_mono_ns,
            }
        }
        Err(err) => SafetySnapshot {
            connection_id,
            level: Observed::error(source),
            detail: Some(err.to_string()),
            published_at,
            published_mono_ns,
        },
    }
}

// ---------------------------------------------------------------------------
// Weather logging (§3.9 records)
// ---------------------------------------------------------------------------

/// A copy of `reading` in which every value that is not fresh is unknown.
/// Probe rows must never repeat an old value under a new time.
pub fn fresh_only(reading: &WeatherReading) -> WeatherReading {
    let mut fresh = reading.clone();
    for field in WeatherField::ALL {
        let observed = field.get_mut(&mut fresh);
        if !observed.is_fresh() {
            observed.value = None;
        }
    }
    fresh
        .sensor_ages_s
        .retain(|name, _| WeatherField::from_name(name).is_some_and(|f| f.get(reading).is_fresh()));
    fresh
}

/// Writes one `environment_log` probe row (`source = 'probe'`) and one
/// `weather_sample` event every `[weather] log_interval_s` (0 disables).
/// Only fresh values are written; a reading with none is skipped.
pub struct WeatherLogger {
    interval: Option<Duration>,
    last_logged_mono: Option<u64>,
    recorder: Arc<dyn RecorderSink>,
    events: Arc<dyn EventSink>,
    session_id: Option<String>,
    night_id: Option<String>,
    rig_id: Option<String>,
}

impl WeatherLogger {
    pub fn new(
        config: &WeatherConfig,
        recorder: Arc<dyn RecorderSink>,
        events: Arc<dyn EventSink>,
    ) -> Self {
        Self {
            interval: (config.log_interval_s > 0)
                .then(|| Duration::from_secs(config.log_interval_s)),
            last_logged_mono: None,
            recorder,
            events,
            session_id: None,
            night_id: None,
            rig_id: None,
        }
    }

    pub fn with_context(
        mut self,
        session_id: Option<String>,
        night_id: Option<String>,
        rig_id: Option<String>,
    ) -> Self {
        self.session_id = session_id;
        self.night_id = night_id;
        self.rig_id = rig_id;
        self
    }

    /// Offer a published snapshot. Returns the row when one was written.
    pub fn observe(
        &mut self,
        snapshot: &WeatherSnapshot,
        identity: &DeviceIdentity,
        clock: &dyn Clock,
    ) -> Option<EnvironmentRecord> {
        let interval = self.interval?;
        let now = clock.now_mono_ns();
        if let Some(last) = self.last_logged_mono {
            if now.saturating_sub(last) < crate::equipment::clock::duration_ns(interval) {
                return None;
            }
        }
        let fresh = fresh_only(&snapshot.reading);
        if WeatherField::ALL
            .iter()
            .all(|field| field.get(&fresh).value.is_none())
        {
            return None;
        }
        let mut row = EnvironmentRecord::from_weather(
            snapshot.published_at,
            Some(identity.identity_key.clone()),
            &fresh,
        );
        row.session_id = self.session_id.clone();
        row.night_id = self.night_id.clone();
        row.rig_id = self.rig_id.clone();
        self.recorder
            .record(GroundTruthRecord::Environment(Box::new(row.clone())));
        self.events.emit(
            EquipmentEvent::new(
                EventKind::WeatherSample {
                    reading: Box::new(snapshot.reading.clone()),
                },
                clock,
            )
            .for_device(DeviceKind::Weather, snapshot.connection_id),
        );
        self.last_logged_mono = Some(now);
        Some(row)
    }

    /// Follow a weather handle until its worker stops. No lock or watch
    /// guard is held across an `.await`.
    pub async fn follow(mut self, handle: WeatherHandle, clock: Arc<dyn Clock>) {
        let identity = handle.identity();
        let mut snapshots = handle.watch_snapshot();
        loop {
            let snapshot = Arc::clone(&snapshots.borrow_and_update());
            self.observe(&snapshot, &identity, clock.as_ref());
            if snapshots.changed().await.is_err() {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Model-facing helpers (H14, H22, gating)
// ---------------------------------------------------------------------------

/// The hub's combined safety as an `Observed<bool>` (COC `is_safe`): safe or
/// unsafe when confirmed, `Error` while a monitor is not Ready, and
/// `Unsupported` when no monitor is configured. Never a synthesised "safe".
pub fn is_safe_observed(summary: &SafetySummary, clock: &dyn Clock) -> Observed<bool> {
    match summary.state {
        SafetyGate::Safe => Observed::fresh_at(true, Source::Derived, clock),
        SafetyGate::Unsafe => Observed::fresh_at(false, Source::Derived, clock),
        SafetyGate::Unknown => Observed::error(Source::Derived),
        SafetyGate::None => Observed::unsupported(Source::Derived),
    }
}

/// A field for corroboration: a fresh value whose poll or sensor update is
/// older than `max_age` is downgraded to `Stale`.
fn current(observed: &Observed<f64>, clock: &dyn Clock, max_age: Duration) -> Observed<f64> {
    if !observed.is_fresh() {
        return observed.clone();
    }
    let max_ms = u64::try_from(max_age.as_millis()).unwrap_or(u64::MAX);
    let poll_old = observed
        .age_ms(clock.now_mono_ns())
        .is_some_and(|age| age > max_ms);
    let sensor_old = observed.device_time.is_some_and(|at| {
        let age = (clock.now_utc() - at).num_milliseconds();
        age > i64::try_from(max_ms).unwrap_or(i64::MAX)
    });
    if poll_old || sensor_old {
        observed.clone().into_stale()
    } else {
        observed.clone()
    }
}

/// Independent sky corroboration for COC (H14, R18-R19), with per-sensor
/// ages kept in each `Observed`. With no weather device every field is
/// `Unsupported`. Level: `None` when nothing is known; `Full` when a direct
/// sky sensor (cloud cover, sky temperature or sky quality) is current and
/// safety is confirmed; `Partial` otherwise.
pub fn coc_corroboration(
    weather: Option<&WeatherReading>,
    safety: &SafetySummary,
    clock: &dyn Clock,
    max_age: Duration,
) -> CocCorroboration {
    let field = |field: WeatherField| match weather {
        Some(reading) => current(field.get(reading), clock, max_age),
        None => Observed::unsupported(Source::Derived),
    };
    let mut out = CocCorroboration {
        level: CorroborationLevel::None,
        cloud_cover_pct: field(WeatherField::CloudCover),
        sky_temperature_c: field(WeatherField::SkyTemperature),
        rain_rate_mm_h: field(WeatherField::RainRate),
        humidity_pct: field(WeatherField::Humidity),
        dew_point_c: field(WeatherField::DewPoint),
        sky_quality_mag_arcsec2: field(WeatherField::SkyQuality),
        is_safe: is_safe_observed(safety, clock),
    };
    let direct = out.cloud_cover_pct.is_fresh()
        || out.sky_temperature_c.is_fresh()
        || out.sky_quality_mag_arcsec2.is_fresh();
    let indirect =
        out.rain_rate_mm_h.is_fresh() || out.humidity_pct.is_fresh() || out.dew_point_c.is_fresh();
    let safety_known = out.is_safe.is_fresh();
    out.level = if direct && safety_known {
        CorroborationLevel::Full
    } else if direct || indirect || safety_known {
        CorroborationLevel::Partial
    } else {
        CorroborationLevel::None
    };
    out
}

/// Operations that stop or secure the rig. They stay allowed while safety
/// is unknown or unsafe: refusing them would make the rig less safe.
pub fn is_safing_op(op: &GatedOp) -> bool {
    op.bypasses_gate_checks()
        || matches!(
            op,
            GatedOp::ClearResumeBlock { .. }
                | GatedOp::Mount(MountCmd::Park | MountCmd::SetTracking(false))
                | GatedOp::Guider(GuiderCmd::Stop | GuiderCmd::Pause { .. })
                | GatedOp::CoverCalibrator(
                    CoverCalibratorCmd::Close | CoverCalibratorCmd::CalibratorOff
                )
                | GatedOp::Dome(DomeCmd::CloseShutter | DomeCmd::Park)
        )
}

/// Whether a resume block may be cleared now. `safety_*` blocks clear only
/// on a confirmed safe reading (or once no monitor is configured); in Tier 1
/// (`resume = "manual"`) that clear is the operator's request (§2.9, §3.10).
/// Every other kind is the operator's call.
pub fn safety_block_clear_allowed(kind: &ResumeBlockKind, summary: &SafetySummary) -> bool {
    match kind {
        ResumeBlockKind::SafetyUnsafe | ResumeBlockKind::SafetyUnknown => {
            matches!(summary.state, SafetyGate::Safe | SafetyGate::None)
        }
        _ => true,
    }
}

/// True when the level counts as unsafe under `warning_is_unsafe`.
pub fn level_is_unsafe(level: SafetyLevel, warning_is_unsafe: bool) -> bool {
    match level {
        SafetyLevel::Unsafe => true,
        SafetyLevel::Warning => warning_is_unsafe,
        SafetyLevel::Safe => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::{FocuserCmd, PixelUnit, Settle};
    use crate::equipment::sim::safety::{SimSafetyControl, SimSafetyMonitor};
    use crate::equipment::sim::weather::{SimWeather, SimWeatherControl};
    use crate::equipment::snapshot::SafetySource;
    use crate::equipment::testing::fake_handles::{fake_weather, RecordingSink};
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::{Device, PollPlan};
    use crate::equipment::types::{AutonomyMode, LinkState, Priority};

    fn clock() -> Arc<ManualClock> {
        Arc::new(ManualClock::new())
    }

    #[test]
    fn safety_snapshots_are_never_cached() {
        let clock = clock();
        let control = SimSafetyControl::new();
        let mut sim = SimSafetyMonitor::new(Arc::clone(&clock) as Arc<dyn Clock>, control.clone());
        sim.connect(Duration::from_secs(1)).unwrap();
        let id = sim.identity().connection_id;

        let poll = sim.poll(&PollPlan::default());
        let snap = safety_snapshot(&mut sim, &poll, id, clock.as_ref());
        assert_eq!(snap.level.value, Some(SafetyLevel::Safe));
        assert_eq!(snap.level.state, ObsState::Fresh);
        assert_eq!(snap.level.mono_ns, Some(0));

        control.fail_next_reads(1);
        clock.advance_ms(2000);
        let poll = sim.poll(&PollPlan::default());
        let snap = safety_snapshot(&mut sim, &poll, id, clock.as_ref());
        assert_eq!(snap.level.value, None, "the earlier Safe is not reused");
        assert_eq!(snap.level.state, ObsState::Error);
        assert!(snap.detail.unwrap().contains("injected"));
    }

    #[test]
    fn weather_rows_are_written_at_the_log_interval_with_fresh_values_only() {
        let clock = clock();
        let control = SimWeatherControl::new();
        let mut sim = SimWeather::new(Arc::clone(&clock) as Arc<dyn Clock>, control.clone());
        sim.connect(Duration::from_secs(1)).unwrap();
        let identity = sim.identity().clone();
        let recorder = RecordingSink::new();
        let events = Arc::new(RecorderEventSink::new(recorder.clone()));
        let mut logger = WeatherLogger::new(
            &WeatherConfig { log_interval_s: 60 },
            recorder.clone(),
            events,
        )
        .with_context(Some("s1".into()), None, Some("rig".into()));

        // Before any poll nothing is known: no row.
        let empty = WeatherSnapshot::default();
        assert!(logger.observe(&empty, &identity, clock.as_ref()).is_none());

        control.set_age(WeatherField::Humidity, 4.0);
        sim.poll(&PollPlan::default()).unwrap();
        let snap = weather_snapshot(&mut sim, identity.connection_id, clock.as_ref()).unwrap();
        let row = logger.observe(&snap, &identity, clock.as_ref()).unwrap();
        assert_eq!(row.device_identity_key.as_deref(), Some("sim:weather"));
        assert_eq!(row.humidity_pct, Some(72.0));
        assert_eq!(row.wind_speed_ms, Some(1.8));
        assert_eq!(row.sensor_ages_s["Humidity"], 4.0);
        assert_eq!(row.session_id.as_deref(), Some("s1"));
        assert_eq!(recorder.event_names(), vec!["weather_sample"]);

        // Within the interval: no row.
        clock.advance(Duration::from_secs(10));
        sim.poll(&PollPlan::default()).unwrap();
        let snap = weather_snapshot(&mut sim, identity.connection_id, clock.as_ref()).unwrap();
        assert!(logger.observe(&snap, &identity, clock.as_ref()).is_none());

        // A failed poll leaves stale values, which are never logged.
        clock.advance(Duration::from_secs(60));
        control.fail_next_polls(1);
        assert!(sim.poll(&PollPlan::default()).is_err());
        let snap = weather_snapshot(&mut sim, identity.connection_id, clock.as_ref()).unwrap();
        assert_eq!(snap.reading.humidity_pct.state, ObsState::Stale);
        assert!(logger.observe(&snap, &identity, clock.as_ref()).is_none());

        // The next good poll logs at once.
        sim.poll(&PollPlan::default()).unwrap();
        let snap = weather_snapshot(&mut sim, identity.connection_id, clock.as_ref()).unwrap();
        assert!(logger.observe(&snap, &identity, clock.as_ref()).is_some());
        let rows = recorder
            .records()
            .into_iter()
            .filter(|r| matches!(r, GroundTruthRecord::Environment(_)))
            .count();
        assert_eq!(rows, 2);

        // log_interval_s = 0 disables logging.
        let mut off = WeatherLogger::new(
            &WeatherConfig { log_interval_s: 0 },
            recorder.clone(),
            Arc::new(RecorderEventSink::new(recorder.clone())),
        );
        assert!(off.observe(&snap, &identity, clock.as_ref()).is_none());
    }

    #[test]
    fn fresh_only_drops_stale_and_unknown_values() {
        let clock = clock();
        let mut reading = WeatherReading {
            temperature_c: Observed::fresh_at(5.0, Source::Property, clock.as_ref()),
            humidity_pct: Observed::fresh_at(80.0, Source::Property, clock.as_ref()).into_stale(),
            ..WeatherReading::default()
        };
        reading.sensor_ages_s.insert("Temperature".into(), 1.0);
        reading.sensor_ages_s.insert("Humidity".into(), 1.0);
        let fresh = fresh_only(&reading);
        assert_eq!(fresh.temperature_c.value, Some(5.0));
        assert_eq!(fresh.humidity_pct.value, None);
        assert_eq!(fresh.humidity_pct.state, ObsState::Stale);
        assert!(!fresh.sensor_ages_s.contains_key("Humidity"));
        let row = EnvironmentRecord::from_weather(clock.now_utc(), None, &fresh);
        assert_eq!(row.humidity_pct, None);
    }

    #[tokio::test]
    async fn the_logger_follows_a_weather_handle_until_its_worker_stops() {
        let clock = clock();
        let (handle, fake) = fake_weather(WeatherSnapshot::default());
        let recorder = RecordingSink::new();
        let logger = WeatherLogger::new(
            &WeatherConfig { log_interval_s: 60 },
            recorder.clone(),
            Arc::new(RecorderEventSink::new(recorder.clone())),
        );
        let task =
            tokio::spawn(logger.follow(handle.clone(), Arc::clone(&clock) as Arc<dyn Clock>));
        fake.update(|snap| {
            snap.reading.temperature_c = Observed::fresh_at(3.0, Source::Property, clock.as_ref());
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while recorder.is_empty() {
            assert!(std::time::Instant::now() < deadline, "no row written");
            tokio::task::yield_now().await;
        }
        drop(handle);
        drop(fake);
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("the logger stops with its worker")
            .unwrap();
        assert_eq!(recorder.event_names(), vec!["weather_sample"]);
    }

    #[test]
    fn corroboration_levels_and_ages() {
        let clock = clock();
        let safe = SafetySummary {
            state: SafetyGate::Safe,
            ..SafetySummary::default()
        };
        let none = coc_corroboration(
            None,
            &SafetySummary::default(),
            clock.as_ref(),
            Duration::from_secs(300),
        );
        assert_eq!(none.level, CorroborationLevel::None);
        assert_eq!(none.cloud_cover_pct.state, ObsState::Unsupported);
        assert_eq!(none.is_safe.state, ObsState::Unsupported);
        assert_eq!(none.is_safe.value, None, "no monitor is never 'safe'");

        let mut reading = WeatherReading {
            cloud_cover_pct: Observed::fresh_at(20.0, Source::Property, clock.as_ref()),
            humidity_pct: Observed::fresh_at(90.0, Source::Property, clock.as_ref()),
            ..WeatherReading::default()
        };
        let full = coc_corroboration(
            Some(&reading),
            &safe,
            clock.as_ref(),
            Duration::from_secs(300),
        );
        assert_eq!(full.level, CorroborationLevel::Full);
        assert_eq!(full.is_safe.value, Some(true));

        // A sensor that last updated 10 minutes ago is not current.
        reading.cloud_cover_pct.device_time = Some(clock.now_utc() - chrono::Duration::minutes(10));
        let partial = coc_corroboration(
            Some(&reading),
            &safe,
            clock.as_ref(),
            Duration::from_secs(300),
        );
        assert_eq!(partial.cloud_cover_pct.state, ObsState::Stale);
        assert_eq!(partial.level, CorroborationLevel::Partial);

        let unknown = SafetySummary {
            state: SafetyGate::Unknown,
            ..SafetySummary::default()
        };
        clock.advance(Duration::from_secs(600));
        let old = coc_corroboration(
            Some(&reading),
            &unknown,
            clock.as_ref(),
            Duration::from_secs(300),
        );
        assert_eq!(old.humidity_pct.state, ObsState::Stale, "poll too old");
        assert_eq!(old.is_safe.state, ObsState::Error);
        assert_eq!(old.level, CorroborationLevel::None);
    }

    #[test]
    fn safing_ops_and_block_clearing_rules() {
        assert!(is_safing_op(&GatedOp::Mount(MountCmd::Park)));
        assert!(is_safing_op(&GatedOp::Mount(MountCmd::SetTracking(false))));
        assert!(!is_safing_op(&GatedOp::Mount(MountCmd::SetTracking(true))));
        assert!(is_safing_op(&GatedOp::Mount(MountCmd::AbortSlew)));
        assert!(is_safing_op(&GatedOp::Guider(GuiderCmd::Stop)));
        assert!(is_safing_op(&GatedOp::Focuser(FocuserCmd::Halt)));
        assert!(is_safing_op(&GatedOp::Dome(DomeCmd::CloseShutter)));
        assert!(!is_safing_op(&GatedOp::Mount(MountCmd::Unpark)));
        assert!(!is_safing_op(&GatedOp::Guider(GuiderCmd::Dither {
            amount: 5.0,
            unit: PixelUnit::GuidePx,
            ra_only: false,
            settle: Settle::default(),
        })));

        let mut summary = SafetySummary {
            state: SafetyGate::Unknown,
            sources: vec![SafetySource {
                connection_id: Uuid::new_v4(),
                name: "m".into(),
                link: LinkState::Degraded,
                level: None,
            }],
            changed_at: None,
        };
        assert!(!safety_block_clear_allowed(
            &ResumeBlockKind::SafetyUnsafe,
            &summary
        ));
        assert!(safety_block_clear_allowed(
            &ResumeBlockKind::Estop,
            &summary
        ));
        summary.state = SafetyGate::Safe;
        assert!(safety_block_clear_allowed(
            &ResumeBlockKind::SafetyUnsafe,
            &summary
        ));
        assert!(level_is_unsafe(SafetyLevel::Warning, true));
        assert!(!level_is_unsafe(SafetyLevel::Warning, false));
    }

    #[tokio::test]
    async fn the_stand_in_gate_audits_and_only_authorises_safety_commands() {
        let clock = clock();
        let recorder = RecordingSink::new();
        let gate = AuditingSafetyGate::new(recorder.clone(), Arc::clone(&clock) as Arc<dyn Clock>);
        let park = GatedCommand::new(
            GatedOp::Mount(MountCmd::Park),
            CommandSource::Safety,
            AutonomyMode::Bounded,
            Priority::Safety,
        );
        let permit = gate.authorize(&park).await.unwrap();
        assert_eq!(permit.rank(), 6);
        assert_eq!(permit.command_id(), park.command_id);
        let unpark = GatedCommand::new(
            GatedOp::Mount(MountCmd::Unpark),
            CommandSource::Operator { overriding: false },
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        assert!(matches!(
            gate.authorize(&unpark).await,
            Err(GateError::Refused { .. })
        ));
        let stages: Vec<_> = recorder
            .records()
            .into_iter()
            .filter_map(|r| match r {
                GroundTruthRecord::Audit(a) => Some((a.stage, a.accepted)),
                _ => None,
            })
            .collect();
        assert_eq!(
            stages,
            vec![
                (AuditStage::Proposed, true),
                (AuditStage::Validated, true),
                (AuditStage::Proposed, true),
                (AuditStage::Rejected, false),
            ]
        );
    }

    #[test]
    fn the_memory_store_deduplicates_and_emits_block_events() {
        let clock = clock();
        let recorder = RecordingSink::new();
        let store = MemoryResumeBlocks::new(
            Arc::new(RecorderEventSink::new(recorder.clone())),
            Arc::clone(&clock) as Arc<dyn Clock>,
        );
        let watch = store.subscribe();
        let a = store.raise(BlockRequest::new(ResumeBlockKind::SafetyUnsafe, "rain"));
        let b = store.raise(BlockRequest::new(
            ResumeBlockKind::SafetyUnsafe,
            "rain again",
        ));
        assert_eq!(a.id, b.id);
        assert_eq!(watch.borrow().len(), 1);
        assert!(store
            .clear(a.id, &CommandSource::Operator { overriding: false })
            .is_some());
        assert!(store.clear(a.id, &CommandSource::Safety).is_none());
        assert!(store.active().is_empty());
        assert_eq!(
            recorder.event_names(),
            vec!["resume_blocked", "resume_cleared"]
        );
    }
}
