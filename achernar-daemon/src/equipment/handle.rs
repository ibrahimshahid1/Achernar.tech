//! Device handles (spec §2.4, §2.5, D3, D4, D19). Frozen by WP0.
//!
//! A [`DeviceHandle`] is the cheap, cloneable front of one connected device.
//! Its worker (a std thread per device, or the PHD2 `GuiderService` tokio
//! task) holds the matching [`WorkerEnd`].
//!
//! Handle semantics:
//! - `send` returns as soon as the worker accepts the command. The caller
//!   either awaits [`OpTicket::wait`] or returns immediately; HTTP routes
//!   return on accept.
//! - A call on a retired handle fails with `NotConnected` (HTTP 412).
//! - `abort` is the exception (D19): it runs the out-of-band [`AbortPath`] on
//!   a blocking thread immediately and puts a `Cancel` at the head of the
//!   queue, on a ready, degraded or lost handle, until the handle is retired.
//! - No lock is ever held across an `.await`.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tokio::sync::{oneshot, watch, Notify};
use uuid::Uuid;

use super::commands::{
    CoverCalibratorCmd, Domain, DomeCmd, FilterWheelCmd, FocuserCmd, GatedCommand, GuiderCmd,
    MountCmd, RotatorCmd, SafetyCmd, SwitchCmd, WeatherCmd,
};
use super::snapshot::{
    CoverCalibratorSnapshot, DomeSnapshot, FilterWheelSnapshot, FocuserSnapshot, GuiderSnapshot,
    MountSnapshot, RotatorSnapshot, SafetySnapshot, SwitchSnapshot, WeatherSnapshot,
};
use super::traits::AbortPath;
use super::types::{
    kind_name, AutonomyMode, CommandId, CommandSource, DeviceError, DeviceIdentity, DeviceKind,
    DeviceResult, HealthState, LinkState, OpOutcome, Priority,
};

pub type MountHandle = DeviceHandle<MountCmd, MountSnapshot>;
pub type FocuserHandle = DeviceHandle<FocuserCmd, FocuserSnapshot>;
pub type FilterWheelHandle = DeviceHandle<FilterWheelCmd, FilterWheelSnapshot>;
pub type GuiderHandle = DeviceHandle<GuiderCmd, GuiderSnapshot>;
pub type RotatorHandle = DeviceHandle<RotatorCmd, RotatorSnapshot>;
pub type CoverCalibratorHandle = DeviceHandle<CoverCalibratorCmd, CoverCalibratorSnapshot>;
pub type SwitchHandle = DeviceHandle<SwitchCmd, SwitchSnapshot>;
pub type WeatherHandle = DeviceHandle<WeatherCmd, WeatherSnapshot>;
pub type SafetyMonitorHandle = DeviceHandle<SafetyCmd, SafetySnapshot>;
pub type DomeHandle = DeviceHandle<DomeCmd, DomeSnapshot>;

// ---------------------------------------------------------------------------
// Command queue
// ---------------------------------------------------------------------------

/// A multi-producer, single-consumer queue that a std worker thread reads
/// with a timeout ([`CommandQueue::pop_timeout`]) and a tokio task reads
/// asynchronously ([`CommandQueue::pop`]). Pushing never blocks.
pub struct CommandQueue<T> {
    inner: Arc<QueueInner<T>>,
}

struct QueueInner<T> {
    items: Mutex<VecDeque<T>>,
    available: Condvar,
    notify: Notify,
    closed: AtomicBool,
}

impl<T> Clone for CommandQueue<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> Default for CommandQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> CommandQueue<T> {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(QueueInner {
                items: Mutex::new(VecDeque::new()),
                available: Condvar::new(),
                notify: Notify::new(),
                closed: AtomicBool::new(false),
            }),
        }
    }

    fn items(&self) -> MutexGuard<'_, VecDeque<T>> {
        // A panicking worker must not wedge the queue (§2.5 containment).
        self.inner.items.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn wake(&self) {
        self.inner.available.notify_all();
        self.inner.notify.notify_one();
    }

    /// Append; fails with the item once the queue is closed.
    pub fn push(&self, item: T) -> Result<(), T> {
        let mut items = self.items();
        if self.inner.closed.load(Ordering::Acquire) {
            return Err(item);
        }
        items.push_back(item);
        drop(items);
        self.wake();
        Ok(())
    }

    /// Insert at the head, ahead of queued commands (cancels; D4).
    pub fn push_front(&self, item: T) -> Result<(), T> {
        let mut items = self.items();
        if self.inner.closed.load(Ordering::Acquire) {
            return Err(item);
        }
        items.push_front(item);
        drop(items);
        self.wake();
        Ok(())
    }

    pub fn try_pop(&self) -> Option<T> {
        self.items().pop_front()
    }

    /// Blocking receive for worker threads. `None` on timeout, or once the
    /// queue is closed and drained.
    pub fn pop_timeout(&self, timeout: Duration) -> Option<T> {
        let deadline = Instant::now() + timeout;
        let mut items = self.items();
        loop {
            if let Some(item) = items.pop_front() {
                return Some(item);
            }
            if self.is_closed() {
                return None;
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            let (guard, _) = self
                .inner
                .available
                .wait_timeout(items, deadline - now)
                .unwrap_or_else(|e| e.into_inner());
            items = guard;
        }
    }

    /// Async receive for tokio services. `None` once closed and drained.
    pub async fn pop(&self) -> Option<T> {
        loop {
            if let Some(item) = self.try_pop() {
                return Some(item);
            }
            if self.is_closed() {
                return None;
            }
            // `notify_one` stores a permit when nobody waits, so a push
            // between `try_pop` and here is not lost.
            self.inner.notify.notified().await;
        }
    }

    /// Refuse further pushes and wake every waiter.
    pub fn close(&self) {
        let items = self.items();
        self.inner.closed.store(true, Ordering::Release);
        drop(items);
        self.inner.available.notify_all();
        self.inner.notify.notify_waiters();
        self.inner.notify.notify_one();
    }

    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::Acquire)
    }

    pub fn len(&self) -> usize {
        self.items().len()
    }

    pub fn is_empty(&self) -> bool {
        self.items().is_empty()
    }
}

// ---------------------------------------------------------------------------
// Permits
// ---------------------------------------------------------------------------

/// Proof that the gate authorised a command. Dropping it releases every
/// domain it holds, on success, error, timeout, cancel and worker panic
/// (v6 §7.3). The gate (WP1A-2) supplies the release callback.
pub struct Permit {
    command_id: CommandId,
    priority: Priority,
    rank: u8,
    source: CommandSource,
    mode: AutonomyMode,
    domains: Vec<Domain>,
    release: Option<Box<dyn FnOnce() + Send>>,
}

impl Permit {
    /// A permit for an authorised command. `release` runs exactly once, on
    /// drop.
    pub fn new(cmd: &GatedCommand, release: Option<Box<dyn FnOnce() + Send>>) -> Self {
        Self {
            command_id: cmd.command_id,
            priority: cmd.priority,
            rank: cmd.rank(),
            source: cmd.source.clone(),
            mode: cmd.mode,
            domains: cmd.domains.clone(),
            release,
        }
    }

    /// A permit that holds no ownership: abort and e-stop paths (which skip
    /// the gate's first three steps) and tests.
    pub fn unmanaged(command_id: CommandId, priority: Priority, source: CommandSource) -> Self {
        Self {
            command_id,
            priority,
            rank: priority.preemption_rank(&source),
            source,
            mode: AutonomyMode::Manual,
            domains: Vec::new(),
            release: None,
        }
    }

    pub fn command_id(&self) -> CommandId {
        self.command_id
    }

    pub fn priority(&self) -> Priority {
        self.priority
    }

    pub fn rank(&self) -> u8 {
        self.rank
    }

    pub fn source(&self) -> &CommandSource {
        &self.source
    }

    pub fn mode(&self) -> AutonomyMode {
        self.mode
    }

    pub fn domains(&self) -> &[Domain] {
        &self.domains
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            release();
        }
    }
}

impl fmt::Debug for Permit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Permit")
            .field("command_id", &self.command_id)
            .field("priority", &self.priority)
            .field("rank", &self.rank)
            .field("source", &self.source)
            .field("domains", &self.domains)
            .field("releases_ownership", &self.release.is_some())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Envelopes and tickets
// ---------------------------------------------------------------------------

/// What travels from a handle to its worker.
#[derive(Debug)]
pub struct Envelope<C> {
    pub command_id: CommandId,
    pub body: EnvelopeBody<C>,
}

#[derive(Debug)]
pub enum EnvelopeBody<C> {
    /// A gated command. The worker answers `accepted` at once (a conflicting
    /// command gets `InvalidOperation` unless the gate preempted), keeps the
    /// permit for the operation's lifetime and resolves `done` with the
    /// outcome.
    Command {
        cmd: C,
        permit: Permit,
        accepted: oneshot::Sender<DeviceResult<()>>,
        done: oneshot::Sender<OpOutcome>,
    },
    /// Cancel the active operation (abort or preemption). The worker calls
    /// `Operation::cancel` and resolves the op `Aborted { by }` once a poll
    /// confirms motion stopped.
    Cancel { by: CommandId },
    /// Disconnect (§2.6): halt focuser/rotator moves; refuse during a slew
    /// unless `force`, which aborts first; never park.
    Disconnect {
        force: bool,
        done: oneshot::Sender<DeviceResult<()>>,
    },
}

/// Returned once a command is accepted.
#[derive(Debug)]
pub struct OpTicket {
    pub command_id: CommandId,
    pub done: oneshot::Receiver<OpOutcome>,
}

impl OpTicket {
    /// The final outcome. A worker that dropped the operation without
    /// resolving it yields `Unknown`, which raises a resume block.
    pub async fn wait(self) -> OpOutcome {
        self.done.await.unwrap_or_else(|_| {
            OpOutcome::Unknown(DeviceError::outcome_unknown(
                "the device worker dropped the operation before it resolved",
            ))
        })
    }
}

/// Returned by [`DeviceHandle::abort`]. It does not wait for confirmation;
/// that arrives as events.
#[derive(Debug)]
pub struct AbortTicket {
    pub command_id: CommandId,
    /// A `Cancel` was queued at the head of the worker's queue.
    pub cancel_sent: bool,
    /// The out-of-band abort path was started.
    pub abort_path_ran: bool,
    result: oneshot::Receiver<DeviceResult<()>>,
}

impl AbortTicket {
    /// Result of the abort path (or of queuing the cancel when the device
    /// has no abort path).
    pub async fn outcome(self) -> DeviceResult<()> {
        self.result.await.unwrap_or_else(|_| {
            Err(DeviceError::outcome_unknown(
                "the abort path stopped without reporting a result",
            ))
        })
    }

    /// Non-blocking check, for callers that collect results within a budget
    /// (e-stop returns within 2 s).
    pub fn try_outcome(&mut self) -> Option<DeviceResult<()>> {
        match self.result.try_recv() {
            Ok(result) => Some(result),
            Err(oneshot::error::TryRecvError::Empty) => None,
            Err(oneshot::error::TryRecvError::Closed) => Some(Err(DeviceError::outcome_unknown(
                "the abort path stopped without reporting a result",
            ))),
        }
    }
}

/// Run `f` on a blocking thread: tokio's blocking pool inside a runtime, a
/// dedicated thread otherwise.
fn run_blocking(f: impl FnOnce() + Send + 'static) {
    match tokio::runtime::Handle::try_current() {
        Ok(runtime) => drop(runtime.spawn_blocking(f)),
        Err(_) => {
            // If the thread cannot start, `f` is dropped and the caller's
            // oneshot reports that the path stopped without a result.
            let _ = std::thread::Builder::new().name("eq-abort".into()).spawn(f);
        }
    }
}

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

/// The front of one connected device (D3).
pub struct DeviceHandle<C, S> {
    commands: CommandQueue<Envelope<C>>,
    snapshot: watch::Receiver<Arc<S>>,
    abort_path: Option<Arc<dyn AbortPath>>,
    identity: Arc<DeviceIdentity>,
    health: watch::Receiver<HealthState>,
    retired: Arc<AtomicBool>,
}

impl<C, S> Clone for DeviceHandle<C, S> {
    fn clone(&self) -> Self {
        Self {
            commands: self.commands.clone(),
            snapshot: self.snapshot.clone(),
            abort_path: self.abort_path.clone(),
            identity: Arc::clone(&self.identity),
            health: self.health.clone(),
            retired: Arc::clone(&self.retired),
        }
    }
}

impl<C, S> fmt::Debug for DeviceHandle<C, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceHandle")
            .field("kind", &self.identity.kind)
            .field("connection_id", &self.identity.connection_id)
            .field("name", &self.identity.name)
            .field("retired", &self.is_retired())
            .finish()
    }
}

impl<C, S> DeviceHandle<C, S> {
    /// Create a handle and the worker end that serves it.
    pub fn new_pair(
        identity: DeviceIdentity,
        abort_path: Option<Arc<dyn AbortPath>>,
        initial: S,
        health: HealthState,
    ) -> (Self, WorkerEnd<C, S>) {
        let identity = Arc::new(identity);
        let commands = CommandQueue::new();
        let (snapshot_tx, snapshot_rx) = watch::channel(Arc::new(initial));
        let (health_tx, health_rx) = watch::channel(health);
        let retired = Arc::new(AtomicBool::new(false));
        let handle = Self {
            commands: commands.clone(),
            snapshot: snapshot_rx,
            abort_path,
            identity: Arc::clone(&identity),
            health: health_rx,
            retired: Arc::clone(&retired),
        };
        let end = WorkerEnd {
            commands,
            snapshot: snapshot_tx,
            health: health_tx,
            identity,
            retired,
        };
        (handle, end)
    }

    fn retired_error(&self) -> DeviceError {
        DeviceError::not_connected(format!(
            "{} {} is disconnected; reconnect before issuing commands",
            kind_name(self.identity.kind),
            self.identity.name
        ))
    }

    /// The queue is closed: retired, or the worker stopped without retiring
    /// (a contained panic leaves the device `Lost`, §2.5).
    fn stopped_error(&self) -> DeviceError {
        if self.is_retired() {
            self.retired_error()
        } else {
            DeviceError::disconnected(format!(
                "the {} worker for {} has stopped",
                kind_name(self.identity.kind),
                self.identity.name
            ))
        }
    }

    /// Submit a gated command. Resolves when the worker accepts or refuses
    /// it; the outcome arrives on the returned ticket.
    pub async fn send(&self, cmd: C, permit: Permit) -> DeviceResult<OpTicket> {
        if self.is_retired() {
            return Err(self.retired_error());
        }
        let command_id = permit.command_id();
        let (accepted_tx, accepted_rx) = oneshot::channel();
        let (done_tx, done_rx) = oneshot::channel();
        let envelope = Envelope {
            command_id,
            body: EnvelopeBody::Command {
                cmd,
                permit,
                accepted: accepted_tx,
                done: done_tx,
            },
        };
        if self.commands.push(envelope).is_err() {
            return Err(self.stopped_error());
        }
        match accepted_rx.await {
            Ok(Ok(())) => Ok(OpTicket {
                command_id,
                done: done_rx,
            }),
            Ok(Err(err)) => Err(err),
            Err(_) => Err(DeviceError::disconnected(
                "the device worker stopped before accepting the command",
            )),
        }
    }

    /// The latest published snapshot. Never touches the device.
    pub fn snapshot(&self) -> Arc<S> {
        Arc::clone(&self.snapshot.borrow())
    }

    /// Stop now (D4, D19): start the abort path on a blocking thread and put
    /// a `Cancel` at the head of the worker's queue. Runs on ready, degraded
    /// and lost handles; only a retired handle refuses (`NotConnected`).
    pub fn abort(&self) -> AbortTicket {
        let command_id = CommandId::new();
        let (tx, rx) = oneshot::channel();
        if self.is_retired() {
            let _ = tx.send(Err(self.retired_error()));
            return AbortTicket {
                command_id,
                cancel_sent: false,
                abort_path_ran: false,
                result: rx,
            };
        }
        let mut tx = Some(tx);
        let abort_path_ran = match self.abort_path.clone() {
            Some(path) => {
                let tx = tx.take();
                run_blocking(move || {
                    let result = path.abort();
                    if let Some(tx) = tx {
                        let _ = tx.send(result);
                    }
                });
                true
            }
            None => false,
        };
        let cancel_sent = self
            .commands
            .push_front(Envelope {
                command_id,
                body: EnvelopeBody::Cancel { by: command_id },
            })
            .is_ok();
        if let Some(tx) = tx {
            let _ = tx.send(if cancel_sent {
                Ok(())
            } else {
                Err(self.stopped_error())
            });
        }
        AbortTicket {
            command_id,
            cancel_sent,
            abort_path_ran,
            result: rx,
        }
    }

    /// Ask the worker to disconnect (§2.6). A retired handle is already
    /// disconnected.
    pub async fn disconnect(&self, force: bool) -> DeviceResult<()> {
        if self.is_retired() {
            return Ok(());
        }
        let (done_tx, done_rx) = oneshot::channel();
        let envelope = Envelope {
            command_id: CommandId::new(),
            body: EnvelopeBody::Disconnect {
                force,
                done: done_tx,
            },
        };
        if self.commands.push(envelope).is_err() {
            return Ok(());
        }
        done_rx.await.unwrap_or(Ok(()))
    }

    pub fn identity(&self) -> Arc<DeviceIdentity> {
        Arc::clone(&self.identity)
    }

    /// The out-of-band abort path, for e-stop steps that must not wait for
    /// the worker (O11 j).
    pub fn abort_path(&self) -> Option<Arc<dyn AbortPath>> {
        self.abort_path.clone()
    }

    pub fn kind(&self) -> DeviceKind {
        self.identity.kind
    }

    pub fn connection_id(&self) -> Uuid {
        self.identity.connection_id
    }

    pub fn health(&self) -> HealthState {
        self.health.borrow().clone()
    }

    pub fn is_retired(&self) -> bool {
        self.retired.load(Ordering::Acquire)
    }

    /// A receiver that wakes on every snapshot publish.
    pub fn watch_snapshot(&self) -> watch::Receiver<Arc<S>> {
        self.snapshot.clone()
    }

    /// A receiver that wakes on every health change.
    pub fn watch_health(&self) -> watch::Receiver<HealthState> {
        self.health.clone()
    }
}

/// The worker's side of a handle.
pub struct WorkerEnd<C, S> {
    pub commands: CommandQueue<Envelope<C>>,
    pub snapshot: watch::Sender<Arc<S>>,
    pub health: watch::Sender<HealthState>,
    pub identity: Arc<DeviceIdentity>,
    retired: Arc<AtomicBool>,
}

impl<C, S> WorkerEnd<C, S> {
    pub fn publish(&self, snapshot: S) {
        self.snapshot.send_replace(Arc::new(snapshot));
    }

    pub fn publish_arc(&self, snapshot: Arc<S>) {
        self.snapshot.send_replace(snapshot);
    }

    pub fn current(&self) -> Arc<S> {
        Arc::clone(&self.snapshot.borrow())
    }

    pub fn set_health(&self, health: HealthState) {
        self.health.send_replace(health);
    }

    pub fn update_health(&self, f: impl FnOnce(&mut HealthState)) {
        self.health.send_modify(f);
    }

    pub fn is_retired(&self) -> bool {
        self.retired.load(Ordering::Acquire)
    }

    /// Retire the handle: every later call (abort included) fails with
    /// `NotConnected`, and queued envelopes are dropped, which releases their
    /// permits and fails their callers.
    pub fn retire(&self, now_mono_ns: u64) {
        self.retired.store(true, Ordering::Release);
        self.commands.close();
        while let Some(envelope) = self.commands.try_pop() {
            drop(envelope);
        }
        self.health.send_modify(|health| {
            health.state = LinkState::Disconnected;
            health.changed_mono_ns = now_mono_ns;
        });
    }
}

impl<C, S> Drop for WorkerEnd<C, S> {
    /// A worker that goes away closes its queue, so callers fail fast instead
    /// of waiting for an acceptance that can never come. The handle is not
    /// retired: abort still runs its out-of-band path (D19).
    fn drop(&mut self) {
        self.commands.close();
        while let Some(envelope) = self.commands.try_pop() {
            drop(envelope);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::GatedOp;
    use crate::equipment::testing::fake_handles::{fake_identity, FakeAbortPath};
    use crate::equipment::types::ErrorClass;
    use std::sync::atomic::AtomicUsize;

    fn operator_permit() -> Permit {
        Permit::unmanaged(
            CommandId::new(),
            Priority::Exposure,
            CommandSource::Operator { overriding: false },
        )
    }

    #[test]
    fn queue_serves_threads_with_timeouts_and_refuses_after_close() {
        let queue = CommandQueue::new();
        assert_eq!(queue.pop_timeout(Duration::from_millis(5)), None);
        let producer = queue.clone();
        let thread = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            producer.push(1).unwrap();
            producer.push_front(0).unwrap();
        });
        assert!(queue.pop_timeout(Duration::from_secs(5)).is_some());
        thread.join().unwrap();
        assert!(queue.pop_timeout(Duration::from_secs(1)).is_some());
        queue.push(7).unwrap();
        queue.close();
        assert_eq!(queue.push(8), Err(8));
        assert_eq!(queue.pop_timeout(Duration::from_secs(1)), Some(7));
        assert_eq!(queue.pop_timeout(Duration::from_secs(1)), None);
    }

    #[tokio::test]
    async fn queue_serves_async_tasks() {
        let queue = CommandQueue::new();
        let consumer = queue.clone();
        let task = tokio::spawn(async move { consumer.pop().await });
        tokio::task::yield_now().await;
        queue.push("x").unwrap();
        assert_eq!(task.await.unwrap(), Some("x"));
        let consumer = queue.clone();
        let task = tokio::spawn(async move { consumer.pop().await });
        queue.close();
        assert_eq!(task.await.unwrap(), None);
    }

    #[test]
    fn permits_release_ownership_exactly_once_on_drop() {
        let released = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&released);
        let cmd = GatedCommand::new(
            GatedOp::Focuser(FocuserCmd::MoveTo(100)),
            CommandSource::Operator { overriding: false },
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        let permit = Permit::new(
            &cmd,
            Some(Box::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            })),
        );
        assert_eq!(permit.rank(), 5);
        assert_eq!(permit.domains(), &[Domain::Device(DeviceKind::Focuser)]);
        let envelope = Envelope {
            command_id: permit.command_id(),
            body: EnvelopeBody::<FocuserCmd>::Cancel {
                by: permit.command_id(),
            },
        };
        drop(envelope);
        assert_eq!(released.load(Ordering::SeqCst), 0);
        drop(permit);
        assert_eq!(released.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn send_is_accepted_by_the_worker_and_resolves_its_ticket() {
        let (handle, end): (FocuserHandle, _) = DeviceHandle::new_pair(
            fake_identity(DeviceKind::Focuser),
            None,
            FocuserSnapshot::default(),
            HealthState::default(),
        );
        let worker = std::thread::spawn(move || {
            let envelope = end.commands.pop_timeout(Duration::from_secs(5)).unwrap();
            match envelope.body {
                EnvelopeBody::Command {
                    cmd,
                    accepted,
                    done,
                    permit,
                } => {
                    assert_eq!(cmd, FocuserCmd::MoveBy(25));
                    accepted.send(Ok(())).unwrap();
                    drop(permit);
                    done.send(OpOutcome::Succeeded).unwrap();
                }
                other => panic!("unexpected {other:?}"),
            }
            end
        });
        let ticket = handle
            .send(FocuserCmd::MoveBy(25), operator_permit())
            .await
            .unwrap();
        assert_eq!(ticket.wait().await, OpOutcome::Succeeded);
        let end = worker.join().unwrap();
        drop(end);
        // The worker is gone but the handle was never retired: commands fail
        // fast as a disconnect, and abort still reaches its path.
        assert!(!handle.is_retired());
        let err = handle
            .send(FocuserCmd::Halt, operator_permit())
            .await
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::Disconnected);
        assert!(!handle.abort().cancel_sent);
    }

    #[tokio::test]
    async fn retired_handles_refuse_commands_and_aborts_with_412() {
        let abort_path = FakeAbortPath::new();
        let (handle, end): (MountHandle, _) = DeviceHandle::new_pair(
            fake_identity(DeviceKind::Mount),
            Some(abort_path.clone() as Arc<dyn AbortPath>),
            MountSnapshot::default(),
            HealthState::default(),
        );
        end.retire(5);
        assert!(handle.is_retired());
        assert_eq!(handle.health().state, LinkState::Disconnected);
        let err = handle
            .send(MountCmd::Park, operator_permit())
            .await
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::NotConnected);
        assert_eq!(err.class.http_status(), 412);
        let ticket = handle.abort();
        assert!(!ticket.abort_path_ran && !ticket.cancel_sent);
        assert_eq!(
            ticket.outcome().await.unwrap_err().class,
            ErrorClass::NotConnected
        );
        assert_eq!(abort_path.calls(), 0);
        assert!(handle.disconnect(false).await.is_ok());
    }

    #[tokio::test]
    async fn abort_runs_on_a_lost_handle_and_jumps_the_queue() {
        let abort_path = FakeAbortPath::new();
        let (handle, end): (MountHandle, _) = DeviceHandle::new_pair(
            fake_identity(DeviceKind::Mount),
            Some(abort_path.clone() as Arc<dyn AbortPath>),
            MountSnapshot::default(),
            HealthState::default(),
        );
        end.update_health(|h| h.state = LinkState::Lost);
        // A command is already queued; the cancel must overtake it.
        let queued = handle.clone();
        let pending =
            tokio::spawn(async move { queued.send(MountCmd::Park, operator_permit()).await });
        while end.commands.is_empty() {
            tokio::task::yield_now().await;
        }
        let ticket = handle.abort();
        assert!(ticket.abort_path_ran && ticket.cancel_sent);
        let abort_id = ticket.command_id;
        assert!(ticket.outcome().await.is_ok());
        assert_eq!(abort_path.calls(), 1);
        let first = end.commands.try_pop().unwrap();
        assert!(matches!(first.body, EnvelopeBody::Cancel { by } if by == abort_id));
        let second = end.commands.try_pop().unwrap();
        assert!(matches!(second.body, EnvelopeBody::Command { .. }));
        drop(second);
        assert!(pending.await.unwrap().is_err());
    }

    #[test]
    fn abort_without_a_runtime_uses_a_thread() {
        let abort_path = FakeAbortPath::new();
        let (handle, _end): (FocuserHandle, _) = DeviceHandle::new_pair(
            fake_identity(DeviceKind::Focuser),
            Some(abort_path.clone() as Arc<dyn AbortPath>),
            FocuserSnapshot::default(),
            HealthState::default(),
        );
        let mut ticket = handle.abort();
        let deadline = Instant::now() + Duration::from_secs(5);
        let result = loop {
            if let Some(result) = ticket.try_outcome() {
                break result;
            }
            assert!(Instant::now() < deadline, "abort path never reported");
            std::thread::sleep(Duration::from_millis(1));
        };
        assert!(result.is_ok());
        assert_eq!(abort_path.calls(), 1);
    }

    #[test]
    fn snapshots_and_health_are_published_without_the_device() {
        let (handle, end): (FocuserHandle, _) = DeviceHandle::new_pair(
            fake_identity(DeviceKind::Focuser),
            None,
            FocuserSnapshot::default(),
            HealthState::connecting(1),
        );
        assert_eq!(handle.health().state, LinkState::Connecting);
        end.publish(FocuserSnapshot {
            settling: true,
            ..FocuserSnapshot::default()
        });
        end.set_health(HealthState::ready(chrono::Utc::now(), 2));
        assert!(handle.snapshot().settling);
        assert_eq!(handle.health().state, LinkState::Ready);
        assert_eq!(handle.kind(), DeviceKind::Focuser);
        assert_eq!(handle.connection_id(), end.identity.connection_id);
    }
}
