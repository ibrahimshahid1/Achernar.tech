//! Handles backed by scripted responders (spec §2.4 "testing/"), so wave-A
//! packages that depend on another device test against fakes:
//!
//! - [`FakeDevice`]: a worker thread behind a real [`DeviceHandle`] that
//!   records commands, answers through a responder closure, and can hold an
//!   operation open until the test completes or cancels it.
//! - [`FakeAbortPath`], [`FakeCameraActivity`], [`RecordingSink`] and
//!   [`FakeReader`] for the abort, camera, recorder and reader contracts.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use tokio::sync::{broadcast, oneshot, watch};

use crate::equipment::clock::SystemClock;
use crate::equipment::commands::{
    CoverCalibratorCmd, DomeCmd, FilterWheelCmd, FocuserCmd, GuiderCmd, MountCmd, RotatorCmd,
    SafetyCmd, SwitchCmd, WeatherCmd,
};
use crate::equipment::events::{EquipmentEvent, GroundTruthRecord};
use crate::equipment::handle::{DeviceHandle, EnvelopeBody, Permit, WorkerEnd};
use crate::equipment::snapshot::{
    BusySet, CocContext, CoverCalibratorSnapshot, DeviceWindow, DomeSnapshot, EquipmentSnapshot,
    ExposureRef, FilterWheelSnapshot, FocuserSnapshot, GuiderSnapshot, MountSnapshot, Quiescence,
    ResumeBlock, RotatorSnapshot, SafetySnapshot, SwitchSnapshot, TemperatureReading,
    WeatherSnapshot,
};
use crate::equipment::traits::{AbortPath, CameraActivity, Clock, EquipmentReader, RecorderSink};
use crate::equipment::types::{
    identity_key, kind_name, CommandId, DeviceError, DeviceIdentity, DeviceKind, DeviceResult,
    DriverId, Endpoint, ErrorClass, HealthState, LinkState, ObsState, Observed, OpOutcome, Source,
};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// A simulator identity for `kind` with a fresh connection id.
pub fn fake_identity(kind: DeviceKind) -> DeviceIdentity {
    DeviceIdentity::new(
        kind,
        DriverId::Sim,
        Endpoint::InProcess {
            name: format!("fake-{}", kind_name(kind)),
        },
        identity_key::sim(kind),
        format!("Fake {}", kind_name(kind)),
    )
}

// ---------------------------------------------------------------------------
// Abort path
// ---------------------------------------------------------------------------

/// Counts abort calls; the result and an optional real-time delay are
/// scriptable.
#[derive(Debug)]
pub struct FakeAbortPath {
    calls: AtomicUsize,
    result: Mutex<DeviceResult<()>>,
    delay: Mutex<Option<Duration>>,
}

impl FakeAbortPath {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicUsize::new(0),
            result: Mutex::new(Ok(())),
            delay: Mutex::new(None),
        })
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    pub fn set_result(&self, result: DeviceResult<()>) {
        *lock(&self.result) = result;
    }

    pub fn set_delay(&self, delay: Option<Duration>) {
        *lock(&self.delay) = delay;
    }
}

impl AbortPath for FakeAbortPath {
    fn abort(&self) -> DeviceResult<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let delay = *lock(&self.delay);
        if let Some(delay) = delay {
            std::thread::sleep(delay);
        }
        lock(&self.result).clone()
    }
}

// ---------------------------------------------------------------------------
// Fake devices
// ---------------------------------------------------------------------------

/// What a responder tells the fake worker to do with a command.
#[derive(Debug, Clone, PartialEq)]
pub struct FakeResponse {
    /// Acceptance; an error is returned from `DeviceHandle::send`.
    pub accept: DeviceResult<()>,
    /// The outcome; `None` holds the operation open until
    /// [`FakeDevice::complete_pending`] or a cancel.
    pub outcome: Option<OpOutcome>,
}

impl FakeResponse {
    pub fn succeed() -> Self {
        Self {
            accept: Ok(()),
            outcome: Some(OpOutcome::Succeeded),
        }
    }

    pub fn finish(outcome: OpOutcome) -> Self {
        Self {
            accept: Ok(()),
            outcome: Some(outcome),
        }
    }

    pub fn hold() -> Self {
        Self {
            accept: Ok(()),
            outcome: None,
        }
    }

    pub fn reject(err: DeviceError) -> Self {
        Self {
            accept: Err(err),
            outcome: None,
        }
    }
}

type Responder<C, S> = Box<dyn FnMut(&C, &mut S) -> FakeResponse + Send>;

struct Pending {
    command_id: CommandId,
    done: oneshot::Sender<OpOutcome>,
    _permit: Permit,
}

struct FakeState<C, S> {
    commands: Vec<(CommandId, C)>,
    cancels: Vec<CommandId>,
    disconnects: u32,
    pending: Option<Pending>,
    responder: Responder<C, S>,
}

struct Shared<C, S> {
    state: Mutex<FakeState<C, S>>,
    changed: Condvar,
}

/// The test side of a fake device. Dropping it stops the fake worker.
pub struct FakeDevice<C, S> {
    shared: Arc<Shared<C, S>>,
    end: Arc<WorkerEnd<C, S>>,
    abort_path: Arc<FakeAbortPath>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl<C, S> FakeDevice<C, S>
where
    C: Clone + Send + 'static,
    S: Clone + Send + Sync + 'static,
{
    /// A ready device whose responder accepts and succeeds every command.
    pub fn spawn(identity: DeviceIdentity, initial: S) -> (DeviceHandle<C, S>, Self) {
        let abort_path = FakeAbortPath::new();
        let health = HealthState::ready(Utc::now(), SystemClock.now_mono_ns());
        let (handle, end) = DeviceHandle::new_pair(
            identity,
            Some(Arc::clone(&abort_path) as Arc<dyn AbortPath>),
            initial,
            health,
        );
        let shared = Arc::new(Shared {
            state: Mutex::new(FakeState {
                commands: Vec::new(),
                cancels: Vec::new(),
                disconnects: 0,
                pending: None,
                responder: Box::new(|_: &C, _: &mut S| FakeResponse::succeed()),
            }),
            changed: Condvar::new(),
        });
        let end = Arc::new(end);
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let (shared, end, stop) = (Arc::clone(&shared), Arc::clone(&end), Arc::clone(&stop));
            std::thread::Builder::new()
                .name(format!("eq-fake-{}", kind_name(end.identity.kind)))
                .spawn(move || run_fake_worker(&shared, &end, &stop))
                .expect("spawn fake device worker")
        };
        let device = Self {
            shared,
            end,
            abort_path,
            stop,
            thread: Some(thread),
        };
        (handle, device)
    }

    /// Replace the responder. It runs on the fake worker thread with a
    /// mutable copy of the current snapshot, which is published afterwards.
    pub fn set_responder(
        &self,
        responder: impl FnMut(&C, &mut S) -> FakeResponse + Send + 'static,
    ) {
        lock(&self.shared.state).responder = Box::new(responder);
    }

    pub fn commands(&self) -> Vec<C> {
        lock(&self.shared.state)
            .commands
            .iter()
            .map(|(_, cmd)| cmd.clone())
            .collect()
    }

    pub fn command_ids(&self) -> Vec<CommandId> {
        lock(&self.shared.state)
            .commands
            .iter()
            .map(|(id, _)| *id)
            .collect()
    }

    pub fn cancels(&self) -> Vec<CommandId> {
        lock(&self.shared.state).cancels.clone()
    }

    pub fn disconnects(&self) -> u32 {
        lock(&self.shared.state).disconnects
    }

    pub fn has_pending(&self) -> bool {
        lock(&self.shared.state).pending.is_some()
    }

    pub fn abort_path(&self) -> Arc<FakeAbortPath> {
        Arc::clone(&self.abort_path)
    }

    pub fn publish(&self, snapshot: S) {
        self.end.publish(snapshot);
    }

    pub fn update(&self, f: impl FnOnce(&mut S)) {
        let mut snapshot = (*self.end.current()).clone();
        f(&mut snapshot);
        self.end.publish(snapshot);
    }

    pub fn set_health(&self, health: HealthState) {
        self.end.set_health(health);
    }

    pub fn set_link(&self, state: LinkState) {
        self.end.update_health(|health| health.state = state);
    }

    pub fn retire(&self) {
        self.end.retire(SystemClock.now_mono_ns());
    }

    /// Resolve a held operation. False when nothing was pending.
    pub fn complete_pending(&self, outcome: OpOutcome) -> bool {
        let pending = lock(&self.shared.state).pending.take();
        match pending {
            Some(pending) => {
                let _ = pending.done.send(outcome);
                self.shared.changed.notify_all();
                true
            }
            None => false,
        }
    }

    /// The id of the held operation, if any.
    pub fn pending_command_id(&self) -> Option<CommandId> {
        lock(&self.shared.state)
            .pending
            .as_ref()
            .map(|p| p.command_id)
    }

    /// Wait (real time) until at least `n` commands arrived.
    pub fn wait_for_commands(&self, n: usize, timeout: Duration) -> bool {
        self.wait_for(timeout, |state| state.commands.len() >= n)
    }

    /// Wait (real time) until at least `n` cancels arrived.
    pub fn wait_for_cancels(&self, n: usize, timeout: Duration) -> bool {
        self.wait_for(timeout, |state| state.cancels.len() >= n)
    }

    fn wait_for(&self, timeout: Duration, done: impl Fn(&FakeState<C, S>) -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        let mut state = lock(&self.shared.state);
        while !done(&state) {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            state = self
                .shared
                .changed
                .wait_timeout(state, deadline - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        true
    }
}

impl<C, S> Drop for FakeDevice<C, S> {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.end.commands.close();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn run_fake_worker<C, S>(shared: &Shared<C, S>, end: &WorkerEnd<C, S>, stop: &AtomicBool)
where
    C: Clone,
    S: Clone,
{
    while !stop.load(Ordering::Acquire) {
        let Some(envelope) = end.commands.pop_timeout(Duration::from_millis(20)) else {
            if end.commands.is_closed() {
                break;
            }
            continue;
        };
        let mut state = lock(&shared.state);
        match envelope.body {
            EnvelopeBody::Command {
                cmd,
                permit,
                accepted,
                done,
            } => {
                if state.pending.is_some() {
                    let _ = accepted.send(Err(DeviceError::invalid_operation(
                        "another operation is active on this device",
                    )));
                } else {
                    let mut snapshot = (*end.current()).clone();
                    let response = (state.responder)(&cmd, &mut snapshot);
                    end.publish(snapshot);
                    state.commands.push((envelope.command_id, cmd));
                    match response.accept {
                        Err(err) => {
                            let _ = accepted.send(Err(err));
                        }
                        Ok(()) => {
                            let _ = accepted.send(Ok(()));
                            match response.outcome {
                                Some(outcome) => {
                                    let _ = done.send(outcome);
                                }
                                None => {
                                    state.pending = Some(Pending {
                                        command_id: envelope.command_id,
                                        done,
                                        _permit: permit,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            EnvelopeBody::Cancel { by } => {
                state.cancels.push(by);
                if let Some(pending) = state.pending.take() {
                    let _ = pending.done.send(OpOutcome::Aborted { by: Some(by) });
                }
            }
            EnvelopeBody::Disconnect { done, .. } => {
                state.disconnects += 1;
                if let Some(pending) = state.pending.take() {
                    let _ = pending
                        .done
                        .send(OpOutcome::Unknown(DeviceError::disconnected(
                            "disconnected during the operation",
                        )));
                }
                end.retire(SystemClock.now_mono_ns());
                let _ = done.send(Ok(()));
            }
        }
        drop(state);
        shared.changed.notify_all();
    }
}

macro_rules! fake_constructor {
    ($(#[$meta:meta])* $name:ident, $kind:expr, $cmd:ty, $snap:ty) => {
        $(#[$meta])*
        pub fn $name(mut initial: $snap) -> (DeviceHandle<$cmd, $snap>, FakeDevice<$cmd, $snap>) {
            let identity = fake_identity($kind);
            initial.connection_id = identity.connection_id;
            FakeDevice::spawn(identity, initial)
        }
    };
}

fake_constructor!(
    /// A fake mount; the snapshot's connection id is set to the identity's.
    fake_mount,
    DeviceKind::Mount,
    MountCmd,
    MountSnapshot
);
fake_constructor!(
    fake_focuser,
    DeviceKind::Focuser,
    FocuserCmd,
    FocuserSnapshot
);
fake_constructor!(
    fake_filter_wheel,
    DeviceKind::FilterWheel,
    FilterWheelCmd,
    FilterWheelSnapshot
);
fake_constructor!(fake_guider, DeviceKind::Guider, GuiderCmd, GuiderSnapshot);
fake_constructor!(
    fake_rotator,
    DeviceKind::Rotator,
    RotatorCmd,
    RotatorSnapshot
);
fake_constructor!(
    fake_cover_calibrator,
    DeviceKind::CoverCalibrator,
    CoverCalibratorCmd,
    CoverCalibratorSnapshot
);
fake_constructor!(fake_switch, DeviceKind::Switch, SwitchCmd, SwitchSnapshot);
fake_constructor!(
    fake_weather,
    DeviceKind::Weather,
    WeatherCmd,
    WeatherSnapshot
);
fake_constructor!(
    fake_safety_monitor,
    DeviceKind::SafetyMonitor,
    SafetyCmd,
    SafetySnapshot
);
fake_constructor!(fake_dome, DeviceKind::Dome, DomeCmd, DomeSnapshot);

// ---------------------------------------------------------------------------
// Camera hook
// ---------------------------------------------------------------------------

/// A scriptable `CameraActivity`.
#[derive(Debug, Default)]
pub struct FakeCameraActivity {
    exposing: AtomicBool,
    exposure: Mutex<Option<ExposureRef>>,
    name: Mutex<Option<String>>,
    aborts: AtomicUsize,
    abort_result: Mutex<Option<DeviceError>>,
}

impl FakeCameraActivity {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn start_exposure(&self, exposure: ExposureRef) {
        *lock(&self.exposure) = Some(exposure);
        self.exposing.store(true, Ordering::SeqCst);
    }

    pub fn finish_exposure(&self) {
        self.exposing.store(false, Ordering::SeqCst);
        *lock(&self.exposure) = None;
    }

    pub fn set_name(&self, name: Option<&str>) {
        *lock(&self.name) = name.map(str::to_string);
    }

    pub fn fail_aborts(&self, err: Option<DeviceError>) {
        *lock(&self.abort_result) = err;
    }

    pub fn aborts(&self) -> usize {
        self.aborts.load(Ordering::SeqCst)
    }
}

impl CameraActivity for FakeCameraActivity {
    fn exposing(&self) -> Observed<bool> {
        Observed::fresh_at(
            self.exposing.load(Ordering::SeqCst),
            Source::Derived,
            &SystemClock,
        )
    }

    fn current_exposure(&self) -> Option<ExposureRef> {
        lock(&self.exposure).clone()
    }

    fn abort_exposure(&self) -> DeviceResult<()> {
        self.aborts.fetch_add(1, Ordering::SeqCst);
        match lock(&self.abort_result).clone() {
            Some(err) => Err(err),
            None => {
                self.finish_exposure();
                Ok(())
            }
        }
    }

    fn camera_name(&self) -> Option<String> {
        lock(&self.name).clone()
    }
}

// ---------------------------------------------------------------------------
// Recorder sink
// ---------------------------------------------------------------------------

/// Keeps every record for inspection.
#[derive(Debug, Default)]
pub struct RecordingSink {
    records: Mutex<Vec<GroundTruthRecord>>,
}

impl RecordingSink {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn records(&self) -> Vec<GroundTruthRecord> {
        lock(&self.records).clone()
    }

    pub fn len(&self) -> usize {
        lock(&self.records).len()
    }

    pub fn is_empty(&self) -> bool {
        lock(&self.records).is_empty()
    }

    /// Catalogue names of the recorded events, in order.
    pub fn event_names(&self) -> Vec<&'static str> {
        lock(&self.records)
            .iter()
            .filter_map(|record| match record {
                GroundTruthRecord::Event(event) => Some(event.name()),
                _ => None,
            })
            .collect()
    }
}

impl RecorderSink for RecordingSink {
    fn record(&self, rec: GroundTruthRecord) {
        lock(&self.records).push(rec);
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// A settable `EquipmentReader` for model and API tests.
pub struct FakeReader {
    snapshot: Mutex<Arc<EquipmentSnapshot>>,
    events: broadcast::Sender<Arc<EquipmentEvent>>,
    busy: watch::Sender<BusySet>,
    resume_blocks: watch::Sender<Vec<ResumeBlock>>,
    window: Mutex<DeviceWindow>,
    quiescence: Mutex<Quiescence>,
    ambient: Mutex<Observed<TemperatureReading>>,
    coc: Mutex<CocContext>,
}

impl Default for FakeReader {
    fn default() -> Self {
        let (events, _) = broadcast::channel(1024);
        let (busy, _) = watch::channel(BusySet::default());
        let (resume_blocks, _) = watch::channel(Vec::new());
        Self {
            snapshot: Mutex::new(Arc::new(EquipmentSnapshot::default())),
            events,
            busy,
            resume_blocks,
            window: Mutex::new(DeviceWindow::default()),
            quiescence: Mutex::new(Quiescence::default()),
            ambient: Mutex::new(Observed::not_set(Source::Derived)),
            coc: Mutex::new(CocContext::default()),
        }
    }
}

impl FakeReader {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn set_snapshot(&self, snapshot: EquipmentSnapshot) {
        *lock(&self.snapshot) = Arc::new(snapshot);
    }

    /// Broadcast an event to subscribers; returns how many received it.
    pub fn emit(&self, event: EquipmentEvent) -> usize {
        self.events.send(Arc::new(event)).unwrap_or(0)
    }

    pub fn set_busy(&self, busy: BusySet) {
        self.busy.send_replace(busy);
    }

    pub fn set_resume_blocks(&self, blocks: Vec<ResumeBlock>) {
        self.resume_blocks.send_replace(blocks);
    }

    pub fn set_window(&self, window: DeviceWindow) {
        *lock(&self.window) = window;
    }

    pub fn set_quiescence(&self, quiescence: Quiescence) {
        *lock(&self.quiescence) = quiescence;
    }

    pub fn set_ambient(&self, ambient: Observed<TemperatureReading>) {
        *lock(&self.ambient) = ambient;
    }

    pub fn set_coc(&self, context: CocContext) {
        *lock(&self.coc) = context;
    }
}

impl EquipmentReader for FakeReader {
    fn snapshot(&self) -> Arc<EquipmentSnapshot> {
        Arc::clone(&lock(&self.snapshot))
    }

    fn subscribe(&self) -> broadcast::Receiver<Arc<EquipmentEvent>> {
        self.events.subscribe()
    }

    fn window(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> DeviceWindow {
        DeviceWindow {
            start,
            end,
            ..lock(&self.window).clone()
        }
    }

    fn quiescent(
        &self,
        _kinds: &[DeviceKind],
        _start: DateTime<Utc>,
        _end: DateTime<Utc>,
    ) -> Quiescence {
        lock(&self.quiescence).clone()
    }

    fn busy(&self) -> watch::Receiver<BusySet> {
        self.busy.subscribe()
    }

    fn resume_blocks(&self) -> watch::Receiver<Vec<ResumeBlock>> {
        self.resume_blocks.subscribe()
    }

    fn ambient_temperature(&self) -> Observed<TemperatureReading> {
        lock(&self.ambient).clone()
    }

    fn coc_context(&self) -> CocContext {
        lock(&self.coc).clone()
    }
}

/// True when `outcome` is an abort issued by `by`.
pub fn aborted_by(outcome: &OpOutcome, by: CommandId) -> bool {
    matches!(outcome, OpOutcome::Aborted { by: Some(id) } if *id == by)
}

/// True when `err` is the 412 a retired handle returns.
pub fn is_not_connected(err: &DeviceError) -> bool {
    err.class == ErrorClass::NotConnected
}

/// An observed value is unknown, never synthesised.
pub fn is_unknown<T>(value: &Observed<T>) -> bool {
    value.value.is_none() && value.state != ObsState::Fresh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::events::EventKind;
    use crate::equipment::snapshot::GuiderAppState;
    use crate::equipment::types::{CommandSource, Priority};

    fn permit() -> Permit {
        Permit::unmanaged(CommandId::new(), Priority::Exposure, CommandSource::System)
    }

    #[tokio::test]
    async fn fake_devices_record_commands_and_publish_responder_changes() {
        let (handle, fake) = fake_focuser(FocuserSnapshot::default());
        assert_eq!(handle.snapshot().connection_id, handle.connection_id());
        fake.set_responder(|cmd, snap: &mut FocuserSnapshot| {
            if let FocuserCmd::MoveTo(target) = cmd {
                snap.logical_position = Some(*target);
            }
            FakeResponse::succeed()
        });
        let ticket = handle
            .send(FocuserCmd::MoveTo(1200), permit())
            .await
            .unwrap();
        assert_eq!(ticket.wait().await, OpOutcome::Succeeded);
        assert_eq!(fake.commands(), vec![FocuserCmd::MoveTo(1200)]);
        assert_eq!(handle.snapshot().logical_position, Some(1200));

        fake.set_responder(|_, _| {
            FakeResponse::reject(DeviceError::invalid_value("outside travel bounds"))
        });
        let err = handle
            .send(FocuserCmd::MoveTo(-5), permit())
            .await
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::InvalidValue);
    }

    #[tokio::test]
    async fn held_operations_resolve_on_abort_with_the_abort_id() {
        let (handle, fake) = fake_mount(MountSnapshot::default());
        fake.set_responder(|_, _| FakeResponse::hold());
        let ticket = handle
            .send(
                MountCmd::Slew {
                    target: crate::equipment::types::EqCoord::default(),
                },
                permit(),
            )
            .await
            .unwrap();
        assert!(fake.has_pending());
        // A second command conflicts with the active operation.
        let err = handle.send(MountCmd::Park, permit()).await.unwrap_err();
        assert_eq!(err.class, ErrorClass::InvalidOperation);
        // Abort while degraded still runs the path and cancels the op.
        fake.set_link(LinkState::Degraded);
        let abort = handle.abort();
        let abort_id = abort.command_id;
        assert!(abort.outcome().await.is_ok());
        assert!(aborted_by(&ticket.wait().await, abort_id));
        assert_eq!(fake.abort_path().calls(), 1);
        assert_eq!(fake.cancels(), vec![abort_id]);
    }

    #[tokio::test]
    async fn complete_pending_and_disconnect_retire_the_handle() {
        let (handle, fake) = fake_guider(GuiderSnapshot::default());
        fake.update(|snap| {
            snap.app_state =
                Observed::fresh_at(GuiderAppState::Guiding, Source::Phd2Event, &SystemClock)
        });
        assert_eq!(
            handle.snapshot().app_state.value,
            Some(GuiderAppState::Guiding)
        );
        fake.set_responder(|_, _| FakeResponse::hold());
        let ticket = handle.send(GuiderCmd::Stop, permit()).await.unwrap();
        assert!(fake.complete_pending(OpOutcome::Succeeded));
        assert_eq!(ticket.wait().await, OpOutcome::Succeeded);
        assert!(!fake.complete_pending(OpOutcome::Succeeded));
        handle.disconnect(false).await.unwrap();
        assert_eq!(fake.disconnects(), 1);
        assert!(handle.is_retired());
        assert!(is_not_connected(
            &handle.send(GuiderCmd::Refresh, permit()).await.unwrap_err()
        ));
    }

    #[test]
    fn camera_reader_and_sink_fakes_behave() {
        let camera = FakeCameraActivity::new();
        assert_eq!(camera.exposing().value, Some(false));
        camera.start_exposure(ExposureRef {
            duration_s: Some(30.0),
            ..ExposureRef::default()
        });
        assert_eq!(camera.exposing().value, Some(true));
        assert!(camera.abort_exposure().is_ok());
        assert_eq!(camera.aborts(), 1);
        assert_eq!(camera.exposing().value, Some(false));

        let reader = FakeReader::new();
        let mut events = reader.subscribe();
        let clock = crate::equipment::testing::manual_clock::ManualClock::new();
        assert_eq!(
            reader.emit(EquipmentEvent::new(EventKind::SlewStarted, &clock)),
            1
        );
        assert_eq!(events.try_recv().unwrap().name(), "slew_started");
        let mut busy = reader.busy();
        reader.set_busy(BusySet {
            exposing: true,
            ..BusySet::default()
        });
        assert!(busy.borrow_and_update().exposing);
        assert!(is_unknown(&reader.ambient_temperature()));

        let sink = RecordingSink::new();
        sink.record(GroundTruthRecord::Event(Arc::new(EquipmentEvent::new(
            EventKind::Homed,
            &clock,
        ))));
        assert_eq!(sink.event_names(), vec!["homed"]);
        assert_eq!(sink.len(), 1);
    }

    #[test]
    fn every_kind_has_a_fake_constructor() {
        let (_h1, _f1) = fake_filter_wheel(FilterWheelSnapshot::default());
        let (_h2, _f2) = fake_rotator(RotatorSnapshot::default());
        let (_h3, _f3) = fake_cover_calibrator(CoverCalibratorSnapshot::default());
        let (_h4, _f4) = fake_switch(SwitchSnapshot::default());
        let (_h5, _f5) = fake_weather(WeatherSnapshot::default());
        let (h6, _f6) = fake_safety_monitor(SafetySnapshot::default());
        let (h7, _f7) = fake_dome(DomeSnapshot::default());
        assert_eq!(h6.kind(), DeviceKind::SafetyMonitor);
        assert_eq!(h7.identity().identity_key, "sim:dome");
        assert_eq!(h7.health().state, LinkState::Ready);
    }
}
