//! Test doubles for the filter layer (tests only).
//!
//! - [`CollectingSink`]: a `FilterEventSink` that keeps events and resume
//!   blocks, and appends to a shared timeline so a test can assert the order
//!   of filter events relative to commands seen by fake focuser and guider
//!   devices.
//! - [`ManualTimer`]: settle waits advance a `ManualClock` (or never end
//!   once a test holds them); the watchdog never fires unless a test arms
//!   it.
//! - [`spawn_wheel_worker`]: a minimal stand-in for the WP1A-2 worker that
//!   drives a real `FilterWheelDriver` (the sim) and [`FilterMoveOp`] behind
//!   a real `DeviceHandle`, advancing the manual clock one 250 ms tick per
//!   poll. It exists only so the handshake can be tested end to end against
//!   the simulator before the real worker lands.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::equipment::commands::FilterWheelCmd;
use crate::equipment::events::EquipmentEvent;
use crate::equipment::handle::{DeviceHandle, EnvelopeBody, FilterWheelHandle, Permit};
use crate::equipment::ops::{ActiveOp, OpPhase, OpProgress, Operation};
use crate::equipment::snapshot::{FilterWheelSnapshot, ResumeBlock};
use crate::equipment::testing::manual_clock::ManualClock;
use crate::equipment::traits::{Clock, FilterWheelDriver, PollPlan};
use crate::equipment::types::{DeviceError, HealthState, OpOutcome};

use super::change::{HandshakeTimer, TimerFuture};
use super::motion::{FilterMoveOp, FilterWheelPublisher};
use super::{read_wheel, FilterEventSink, FilterSettings, SharedWheel, MOVE_POLL_INTERVAL};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// A shared, ordered log of what happened, across devices.
pub type Timeline = Arc<Mutex<Vec<String>>>;

pub fn timeline() -> Timeline {
    Arc::new(Mutex::new(Vec::new()))
}

pub fn push(timeline: &Timeline, entry: impl Into<String>) {
    lock(timeline).push(entry.into());
}

pub fn entries(timeline: &Timeline) -> Vec<String> {
    lock(timeline).clone()
}

/// Keeps every published event and raised block.
#[derive(Debug, Default)]
pub struct CollectingSink {
    events: Mutex<Vec<EquipmentEvent>>,
    blocks: Mutex<Vec<ResumeBlock>>,
    timeline: Option<Timeline>,
}

impl CollectingSink {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn with_timeline(timeline: &Timeline) -> Arc<Self> {
        Arc::new(Self {
            timeline: Some(Arc::clone(timeline)),
            ..Self::default()
        })
    }

    pub fn events(&self) -> Vec<EquipmentEvent> {
        lock(&self.events).clone()
    }

    pub fn names(&self) -> Vec<&'static str> {
        lock(&self.events).iter().map(|e| e.name()).collect()
    }

    pub fn blocks(&self) -> Vec<ResumeBlock> {
        lock(&self.blocks).clone()
    }
}

impl FilterEventSink for CollectingSink {
    fn publish(&self, event: EquipmentEvent) {
        if let Some(timeline) = &self.timeline {
            push(timeline, format!("event:{}", event.name()));
        }
        lock(&self.events).push(event);
    }

    fn raise_resume_block(&self, block: ResumeBlock) {
        if let Some(timeline) = &self.timeline {
            push(timeline, "resume_block");
        }
        lock(&self.blocks).push(block);
    }
}

/// Settle waits advance the manual clock at once, or never end while held
/// (so a test can abort during the settle); the watchdog fires only when
/// armed (a hung worker).
#[derive(Debug)]
pub struct ManualTimer {
    clock: Arc<ManualClock>,
    watchdog_fires: AtomicBool,
    sleeps_held: AtomicBool,
}

impl ManualTimer {
    pub fn new(clock: &Arc<ManualClock>) -> Arc<Self> {
        Arc::new(Self {
            clock: Arc::clone(clock),
            watchdog_fires: AtomicBool::new(false),
            sleeps_held: AtomicBool::new(false),
        })
    }

    pub fn arm_watchdog(&self) {
        self.watchdog_fires.store(true, Ordering::SeqCst);
    }

    /// Settle waits started from now on never end.
    pub fn hold_sleeps(&self) {
        self.sleeps_held.store(true, Ordering::SeqCst);
    }
}

impl HandshakeTimer for ManualTimer {
    fn sleep(&self, duration: Duration) -> TimerFuture {
        if self.sleeps_held.load(Ordering::SeqCst) {
            return Box::pin(std::future::pending());
        }
        self.clock.advance(duration);
        Box::pin(async {})
    }

    fn watchdog(&self, duration: Duration) -> TimerFuture {
        if self.watchdog_fires.load(Ordering::SeqCst) {
            self.clock.advance(duration);
            Box::pin(async {})
        } else {
            Box::pin(std::future::pending())
        }
    }
}

struct Active {
    op: FilterMoveOp,
    done: tokio::sync::oneshot::Sender<OpOutcome>,
    _permit: Permit,
    info: ActiveOp,
}

/// Stops the test worker on drop.
pub struct WheelWorker {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for WheelWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Run `driver` behind a real handle with a minimal worker loop.
pub fn spawn_wheel_worker(
    mut driver: Box<dyn FilterWheelDriver>,
    clock: Arc<ManualClock>,
    settings: FilterSettings,
    model: SharedWheel,
) -> (FilterWheelHandle, WheelWorker) {
    let identity = driver.identity().clone();
    let initial = FilterWheelSnapshot {
        connection_id: identity.connection_id,
        ..FilterWheelSnapshot::default()
    };
    let (handle, end) = DeviceHandle::new_pair(
        identity,
        None,
        initial,
        HealthState::ready(clock.now_utc(), clock.now_mono_ns()),
    );
    let stop = Arc::new(AtomicBool::new(false));
    let stop_flag = Arc::clone(&stop);
    let thread = std::thread::Builder::new()
        .name("eq-test-filter-wheel".into())
        .spawn(move || {
            let mut publisher = FilterWheelPublisher::new();
            let mut active: Option<Active> = None;
            let clock_ref: &dyn Clock = clock.as_ref();
            let publish = |publisher: &mut FilterWheelPublisher,
                           driver: &mut Box<dyn FilterWheelDriver>,
                           active: Option<ActiveOp>| {
                let raw = driver.poll(&PollPlan::default()).ok();
                let position = driver.position();
                let prev = end.current();
                let slots = read_wheel(&model).filter_slots();
                let snapshot =
                    publisher.build(&prev, position, raw.as_ref(), slots, active, clock_ref);
                end.publish(snapshot);
            };
            publish(&mut publisher, &mut driver, None);
            while !stop_flag.load(Ordering::SeqCst) {
                let wait = if active.is_some() {
                    Duration::ZERO
                } else {
                    Duration::from_millis(2)
                };
                if let Some(envelope) = end.commands.pop_timeout(wait) {
                    match envelope.body {
                        EnvelopeBody::Command {
                            cmd: FilterWheelCmd::SetPosition(target),
                            permit,
                            accepted,
                            done,
                        } => {
                            if active.is_some() {
                                let _ = accepted.send(Err(DeviceError::invalid_operation(
                                    "another operation is active on this device",
                                )));
                            } else {
                                let mut op = FilterMoveOp::new(target, &settings);
                                match op.start(driver.as_mut(), clock_ref) {
                                    Ok(()) => {
                                        let _ = accepted.send(Ok(()));
                                        let now = clock.now_mono_ns();
                                        active = Some(Active {
                                            info: ActiveOp {
                                                command_id: envelope.command_id,
                                                kind: crate::equipment::ops::OpKind::FilterMove,
                                                phase: OpPhase::InProgress,
                                                started_at: clock.now_utc(),
                                                started_mono_ns: now,
                                                deadline_mono_ns: now
                                                    + crate::equipment::clock::duration_ns(
                                                        settings.switch_timeout(),
                                                    ),
                                            },
                                            op,
                                            done,
                                            _permit: permit,
                                        });
                                    }
                                    Err(err) => {
                                        let _ = accepted.send(Err(err));
                                    }
                                }
                            }
                        }
                        EnvelopeBody::Cancel { .. } => {
                            if let Some(mut running) = active.take() {
                                let _ = running.op.cancel(driver.as_mut());
                                let outcome = OpOutcome::Unknown(DeviceError::outcome_unknown(
                                    "cancelled; a filter wheel has no halt",
                                ));
                                publisher.on_op_done(running.op.target(), &outcome);
                                let _ = running.done.send(outcome);
                            }
                        }
                        EnvelopeBody::Disconnect { done, .. } => {
                            end.retire(clock.now_mono_ns());
                            let _ = done.send(Ok(()));
                            break;
                        }
                    }
                }
                if let Some(mut running) = active.take() {
                    clock.advance(MOVE_POLL_INTERVAL);
                    publish(&mut publisher, &mut driver, Some(running.info.clone()));
                    match running.op.step(driver.as_mut(), clock_ref) {
                        OpProgress::Done(outcome) => {
                            publisher.on_op_done(running.op.target(), &outcome);
                            publish(&mut publisher, &mut driver, None);
                            let _ = running.done.send(outcome);
                        }
                        OpProgress::Continue { .. } | OpProgress::Settling { .. } => {
                            active = Some(running);
                        }
                    }
                }
            }
        })
        .expect("spawn the test wheel worker");
    (
        handle,
        WheelWorker {
            stop,
            thread: Some(thread),
        },
    )
}
