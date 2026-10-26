//! The ground-truth recorder (spec §4.3, H10, H19, H21, H23; D6). Owner:
//! WP1A-9.
//!
//! [`Recorder`] implements the frozen [`RecorderSink`]: `record` never blocks
//! and never awaits. It downsamples telemetry and environment rows in place,
//! then `try_send`s onto one bounded queue; a full queue drops the record and
//! counts it. One writer task drains the queue and writes each batch in a
//! single SQLite transaction through the `storage::*_repo` modules, so the
//! journal, the device tables and the profiles are written in the order they
//! were recorded.
//!
//! - A record that fails (a foreign key, a constraint) is counted and logged;
//!   the rest of its batch is still written.
//! - Events whose `seq` was never assigned by the hub are rejected, and
//!   `guide_step` events are not journaled (they go to `guide_steps`).
//! - `exposure_completed` events link the pending dithers to that frame (H10).
//! - If a focuser profile update is ever dropped, that focuser's
//!   relative-position record is rewritten as unknown once every record
//!   queued before the drop has been written, so a lost write can never
//!   resurrect a stale origin after a restart (§2.15).
//! - [`Recorder::stats`] is the drop counter for telemetry and `/status`.
//! - [`Recorder::flush`] and [`Recorder::flush_blocking`] wait until
//!   everything recorded so far is on disk.
//! - Attaching to a store closes `device_connections` rows a crashed process
//!   left open; [`Recorder::shutdown`] writes what is queued, closes the rows
//!   still open with `end_class = shutdown` and stops the writer, within a
//!   budget (§2.15 steps 6 and 7, 2 s).
//!
//! Construction is split ([`Recorder::channel`] then
//! [`RecorderWriter::spawn`]) so the queue exists before the writer runs; the
//! inert hub never builds one (D11).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::{SqliteConnection, SqlitePool};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use super::events::{
    ConnectionEnd, EnvironmentRecord, EquipmentEvent, EventKind, GroundTruthRecord, TelemetrySample,
};
use super::profile::{self, RelativePosition};
use super::traits::RecorderSink;
use super::types::DeviceKind;
use crate::config::EquipmentConfigs;
use crate::storage::{
    control_audit_repo, device_connections_repo, dither_repo, environment_repo,
    equipment_events_repo, exposure_windows_repo, filter_switch_repo, focuser_moves_repo,
    guide_steps_repo, guider_state_repo, night, pointing_samples_repo, policy_decisions_repo,
    telemetry_log_repo,
};

/// How early, as a fraction of the period, a downsampled sample may arrive
/// and still be kept (see `next_due`).
const DOWNSAMPLE_SLACK_FRACTION: f64 = 0.1;

/// Recorder settings.
#[derive(Debug, Clone, PartialEq)]
pub struct RecorderConfig {
    /// Queue bound. A full queue drops records and counts them.
    pub capacity: usize,
    /// Records written per transaction at most.
    pub max_batch: usize,
    /// `[equipment] telemetry_log_hz`: per-kind `telemetry_log` rate; 0 (or
    /// not finite) disables telemetry rows.
    pub telemetry_hz: f64,
    /// `[weather] log_interval_s`: one `environment_log` probe row per device
    /// identity per interval; 0 keeps every row.
    pub environment_interval_s: u64,
    /// A dither is linked only to a frame that started within this time of
    /// it (H10).
    pub dither_link_horizon: chrono::Duration,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            capacity: 4096,
            max_batch: 256,
            telemetry_hz: 1.0,
            environment_interval_s: 60,
            dither_link_horizon: chrono::Duration::hours(2),
        }
    }
}

impl RecorderConfig {
    /// Settings from the equipment configuration.
    pub fn from_configs(configs: &EquipmentConfigs) -> Self {
        Self {
            telemetry_hz: configs.equipment.telemetry_log_hz,
            environment_interval_s: configs.weather.log_interval_s,
            ..Self::default()
        }
    }
}

/// Counters, published in telemetry and `/status` (the drop counter of the
/// WP1A-9 contract).
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct RecorderStats {
    /// Records accepted onto the queue.
    pub enqueued: u64,
    /// Records the writer has taken off the queue.
    pub processed: u64,
    /// Records written (rows inserted or updated, or a no-op duplicate).
    pub written: u64,
    /// Records whose write failed.
    pub failed: u64,
    /// Records dropped because the queue was full or the recorder was shut
    /// down.
    pub dropped: u64,
    /// Telemetry and environment rows skipped by downsampling (not a loss).
    pub downsampled: u64,
    /// Records refused as malformed (an unsequenced event, a non-object
    /// profile patch).
    pub rejected: u64,
    /// Transactions committed.
    pub batches: u64,
    /// Dither rows linked to their next frame.
    pub dithers_linked: u64,
    /// Focuser positions marked unknown after a dropped profile write.
    pub positions_invalidated: u64,
    pub pending: u64,
    pub capacity: u64,
    pub closed: bool,
    pub last_error: Option<String>,
}

/// Why a flush or shutdown did not complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlushError {
    /// The budget passed before the writer reached the flush point.
    TimedOut,
    /// The writer is not running (never spawned, stopped, or panicked).
    WriterGone,
    /// [`Recorder::flush_blocking`] was called on an async runtime thread.
    InAsyncContext,
}

impl std::fmt::Display for FlushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            FlushError::TimedOut => "recorder flush timed out",
            FlushError::WriterGone => "recorder writer is not running",
            FlushError::InAsyncContext => "flush_blocking called on an async runtime thread",
        })
    }
}

impl std::error::Error for FlushError {}

enum Ack {
    Async(oneshot::Sender<()>),
    Blocking(std::sync::mpsc::SyncSender<()>),
}

impl Ack {
    fn send(self) {
        match self {
            Ack::Async(tx) => {
                let _ = tx.send(());
            }
            Ack::Blocking(tx) => {
                let _ = tx.try_send(());
            }
        }
    }
}

enum Queued {
    Record(GroundTruthRecord),
    Flush(Ack),
    /// Stop at the given shutdown time (used to close open connections).
    Stop(Ack, DateTime<Utc>),
}

/// A focuser whose profile update was dropped. Applied once the writer has
/// processed every record that was queued at the time of the drop.
#[derive(Debug, Clone)]
struct Invalidation {
    identity_key: String,
    after_processed: u64,
    at: DateTime<Utc>,
}

#[derive(Default)]
struct Counters {
    enqueued: AtomicU64,
    processed: AtomicU64,
    written: AtomicU64,
    failed: AtomicU64,
    dropped: AtomicU64,
    downsampled: AtomicU64,
    rejected: AtomicU64,
    batches: AtomicU64,
    dithers_linked: AtomicU64,
    positions_invalidated: AtomicU64,
}

struct Shared {
    counters: Counters,
    capacity: usize,
    closed: AtomicBool,
    telemetry: Downsampler,
    environment: IntervalThrottle,
    invalidations: Mutex<Vec<Invalidation>>,
    invalidation_pending: AtomicBool,
    session_id: Mutex<Option<String>>,
    last_error: Mutex<Option<String>>,
    /// O11 c backstop: `(command_id, kind)` of once-per-command events
    /// already accepted (bounded).
    once_seen: Mutex<std::collections::VecDeque<(uuid::Uuid, &'static str)>>,
}

/// Events that happen at most once per command; a second copy with the same
/// command id is a duplicate report (O11 c), not a second fact.
const ONCE_PER_COMMAND: &[&str] = &["park_failed"];
/// How many `(command_id, kind)` pairs the dedupe remembers.
const ONCE_SEEN_LIMIT: usize = 1024;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

impl Shared {
    fn note_error(&self, context: &str, err: &dyn std::fmt::Display) {
        let failed = self.counters.failed.fetch_add(1, Ordering::Relaxed) + 1;
        let message = format!("{context}: {err}");
        // The first failure and then every 1000th, so a broken table cannot
        // flood the log.
        if failed == 1 || failed.is_multiple_of(1000) {
            tracing::warn!("equipment recorder: {message} ({failed} failed so far)");
        }
        *lock(&self.last_error) = Some(message);
    }

    fn session_id(&self) -> Option<String> {
        lock(&self.session_id).clone()
    }

    /// True when `event` repeats a once-per-command event already accepted.
    fn is_duplicate(&self, event: &EquipmentEvent) -> bool {
        let name = event.name();
        let Some(command_id) = event.command_id else {
            return false;
        };
        if !ONCE_PER_COMMAND.contains(&name) {
            return false;
        }
        let key = (command_id.0, name);
        let mut seen = lock(&self.once_seen);
        if seen.contains(&key) {
            return true;
        }
        if seen.len() >= ONCE_SEEN_LIMIT {
            seen.pop_front();
        }
        seen.push_back(key);
        false
    }
}

/// The recorder's intake. Cheap to share as `Arc<Recorder>` /
/// `Arc<dyn RecorderSink>`; callable from worker threads and async tasks.
pub struct Recorder {
    tx: mpsc::Sender<Queued>,
    shared: Arc<Shared>,
    task: Mutex<Option<JoinHandle<()>>>,
}

/// The writer half, run by [`RecorderWriter::spawn`] or awaited directly
/// with [`RecorderWriter::run`].
pub struct RecorderWriter {
    rx: mpsc::Receiver<Queued>,
    shared: Arc<Shared>,
    config: RecorderConfig,
    recorder: std::sync::Weak<Recorder>,
}

impl Recorder {
    /// Build the queue and its writer without starting anything.
    pub fn channel(config: RecorderConfig) -> (Arc<Recorder>, RecorderWriter) {
        let capacity = config.capacity.max(1);
        let (tx, rx) = mpsc::channel(capacity);
        let shared = Arc::new(Shared {
            counters: Counters::default(),
            capacity,
            closed: AtomicBool::new(false),
            telemetry: Downsampler::from_hz(config.telemetry_hz),
            environment: IntervalThrottle::new(Duration::from_secs(config.environment_interval_s)),
            invalidations: Mutex::new(Vec::new()),
            invalidation_pending: AtomicBool::new(false),
            session_id: Mutex::new(None),
            last_error: Mutex::new(None),
            once_seen: Mutex::new(std::collections::VecDeque::new()),
        });
        let recorder = Arc::new(Recorder {
            tx,
            shared: Arc::clone(&shared),
            task: Mutex::new(None),
        });
        let writer = RecorderWriter {
            rx,
            shared,
            config,
            recorder: Arc::downgrade(&recorder),
        };
        (recorder, writer)
    }

    /// Build the recorder and start its writer on the current tokio runtime
    /// (`AppState::with_equipment`).
    pub fn start(pool: SqlitePool, config: RecorderConfig) -> Arc<Recorder> {
        let (recorder, writer) = Self::channel(config);
        writer.spawn(pool);
        recorder
    }

    /// The session that `equipment_events` and `device_connections` rows are
    /// attributed to. Those columns have no foreign key; the tables that do
    /// (`filter_switch_events`, `guider_state_log`, `environment_log`) take
    /// the session from their own records.
    pub fn set_session(&self, session_id: Option<String>) {
        *lock(&self.shared.session_id) = session_id;
    }

    pub fn stats(&self) -> RecorderStats {
        let c = &self.shared.counters;
        let enqueued = c.enqueued.load(Ordering::Relaxed);
        let processed = c.processed.load(Ordering::Relaxed);
        RecorderStats {
            enqueued,
            processed,
            written: c.written.load(Ordering::Relaxed),
            failed: c.failed.load(Ordering::Relaxed),
            dropped: c.dropped.load(Ordering::Relaxed),
            downsampled: c.downsampled.load(Ordering::Relaxed),
            rejected: c.rejected.load(Ordering::Relaxed),
            batches: c.batches.load(Ordering::Relaxed),
            dithers_linked: c.dithers_linked.load(Ordering::Relaxed),
            positions_invalidated: c.positions_invalidated.load(Ordering::Relaxed),
            pending: enqueued.saturating_sub(processed),
            capacity: self.shared.capacity as u64,
            closed: self.shared.closed.load(Ordering::Relaxed),
            last_error: lock(&self.shared.last_error).clone(),
        }
    }

    /// True when a telemetry sample of `kind` at `at` would be kept. The hub
    /// can ask before serialising a snapshot it would otherwise discard.
    pub fn telemetry_due(&self, kind: DeviceKind, at: DateTime<Utc>) -> bool {
        self.shared.telemetry.would_accept(kind, at)
    }

    /// Wait until every record recorded before this call is written.
    pub async fn flush(&self, budget: Duration) -> Result<(), FlushError> {
        let (tx, rx) = oneshot::channel();
        let send = self.tx.send(Queued::Flush(Ack::Async(tx)));
        let wait = async {
            send.await.map_err(|_| FlushError::WriterGone)?;
            rx.await.map_err(|_| FlushError::WriterGone)
        };
        tokio::time::timeout(budget, wait)
            .await
            .map_err(|_| FlushError::TimedOut)?
    }

    /// [`Recorder::flush`] for worker threads (D3), e.g. a focuser worker
    /// making its `in_motion` profile record durable before it sends a move.
    /// Refuses to run on an async runtime thread, where it would block a
    /// tokio worker.
    pub fn flush_blocking(&self, budget: Duration) -> Result<(), FlushError> {
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(FlushError::InAsyncContext);
        }
        let deadline = std::time::Instant::now() + budget;
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let mut item = Queued::Flush(Ack::Blocking(tx));
        loop {
            match self.tx.try_send(item) {
                Ok(()) => break,
                Err(mpsc::error::TrySendError::Closed(_)) => return Err(FlushError::WriterGone),
                Err(mpsc::error::TrySendError::Full(back)) => {
                    if std::time::Instant::now() >= deadline {
                        return Err(FlushError::TimedOut);
                    }
                    item = back;
                    std::thread::sleep(Duration::from_millis(2));
                }
            }
        }
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        match rx.recv_timeout(left) {
            Ok(()) => Ok(()),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err(FlushError::TimedOut),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(FlushError::WriterGone),
        }
    }

    /// Daemon shutdown (§2.15 steps 6 and 7): stop accepting records, write
    /// everything already queued, close every `device_connections` row still
    /// open with `end_class = shutdown` at `at`, and stop the writer, all
    /// within `budget` (the spec allows 2 s). Records arriving after this
    /// call are counted as dropped.
    pub async fn shutdown(
        &self,
        at: DateTime<Utc>,
        budget: Duration,
    ) -> Result<RecorderStats, FlushError> {
        self.shared.closed.store(true, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        let stop = async {
            self.tx
                .send(Queued::Stop(Ack::Async(tx), at))
                .await
                .map_err(|_| FlushError::WriterGone)?;
            rx.await.map_err(|_| FlushError::WriterGone)
        };
        let result = tokio::time::timeout(budget, stop)
            .await
            .map_err(|_| FlushError::TimedOut)
            .and_then(|inner| inner);
        let task = lock(&self.task).take();
        if result.is_ok() {
            if let Some(task) = task {
                let _ = task.await;
            }
        }
        result.map(|()| self.stats())
    }

    fn drop_record(&self, record: &GroundTruthRecord) {
        self.shared.counters.dropped.fetch_add(1, Ordering::Relaxed);
        if let GroundTruthRecord::Profile(update) = record {
            if update.kind == DeviceKind::Focuser {
                let processed_target = self.shared.counters.enqueued.load(Ordering::SeqCst);
                lock(&self.shared.invalidations).push(Invalidation {
                    identity_key: update.identity_key.clone(),
                    after_processed: processed_target,
                    at: update.updated_at,
                });
                self.shared
                    .invalidation_pending
                    .store(true, Ordering::SeqCst);
            }
        }
    }
}

impl RecorderSink for Recorder {
    fn record(&self, rec: GroundTruthRecord) {
        let shared = &self.shared;
        let keep = match &rec {
            GroundTruthRecord::Telemetry(sample) => {
                shared.telemetry.accept(sample.source, sample.at)
            }
            GroundTruthRecord::Environment(row) => shared.environment.accept(
                row.device_identity_key.as_deref().unwrap_or(""),
                row.observed_at,
            ),
            _ => true,
        };
        if !keep {
            shared.counters.downsampled.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if let GroundTruthRecord::Event(event) = &rec {
            if shared.is_duplicate(event) {
                shared.counters.rejected.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        if shared.closed.load(Ordering::SeqCst) {
            self.drop_record(&rec);
            return;
        }
        match self.tx.try_send(Queued::Record(rec)) {
            Ok(()) => {
                shared.counters.enqueued.fetch_add(1, Ordering::SeqCst);
            }
            Err(mpsc::error::TrySendError::Full(Queued::Record(rec)))
            | Err(mpsc::error::TrySendError::Closed(Queued::Record(rec))) => {
                self.drop_record(&rec);
            }
            Err(_) => {}
        }
    }
}

impl RecorderWriter {
    /// Run the writer on the current tokio runtime.
    pub fn spawn(self, pool: SqlitePool) {
        let recorder = self.recorder.clone();
        let handle = tokio::spawn(self.run(pool));
        if let Some(recorder) = recorder.upgrade() {
            *lock(&recorder.task) = Some(handle);
        }
    }

    /// Drain the queue until a stop, or until every sender is gone.
    ///
    /// Attaching to a store first closes the `device_connections` rows a
    /// previous process left open when it ended without a clean shutdown
    /// (`end_class = lost`, ending at their last journaled event), so a stale
    /// row never claims a device was connected during later frames.
    pub async fn run(mut self, pool: SqlitePool) {
        match device_connections_repo::close_stale(&pool, ConnectionEnd::Lost).await {
            Ok(0) => {}
            Ok(n) => tracing::info!("equipment recorder: closed {n} stale device connection(s)"),
            Err(err) => self.shared.note_error("close stale connections", &err),
        }
        let mut batch: Vec<GroundTruthRecord> = Vec::with_capacity(self.config.max_batch);
        loop {
            let Some(first) = self.rx.recv().await else {
                break;
            };
            let mut flush: Option<Ack> = None;
            let mut stop: Option<(Ack, DateTime<Utc>)> = None;
            match first {
                Queued::Record(rec) => batch.push(rec),
                Queued::Flush(a) => flush = Some(a),
                Queued::Stop(a, at) => stop = Some((a, at)),
            }
            while flush.is_none() && stop.is_none() && batch.len() < self.config.max_batch {
                match self.rx.try_recv() {
                    Ok(Queued::Record(rec)) => batch.push(rec),
                    Ok(Queued::Flush(a)) => flush = Some(a),
                    Ok(Queued::Stop(a, at)) => stop = Some((a, at)),
                    Err(_) => break,
                }
            }
            self.write_batch(&pool, &mut batch).await;
            if let Some(a) = flush {
                a.send();
            }
            if let Some((a, at)) = stop {
                // Write whatever raced in ahead of the close, then refuse the
                // rest (counted as dropped by `record`).
                self.rx.close();
                let mut acks = vec![a];
                while let Ok(item) = self.rx.try_recv() {
                    match item {
                        Queued::Record(rec) => batch.push(rec),
                        Queued::Flush(other) | Queued::Stop(other, _) => acks.push(other),
                    }
                }
                self.write_batch(&pool, &mut batch).await;
                if let Err(err) =
                    device_connections_repo::close_all_open(&pool, &at, ConnectionEnd::Shutdown)
                        .await
                {
                    self.shared
                        .note_error("close connections at shutdown", &err);
                }
                acks.into_iter().for_each(Ack::send);
                return;
            }
        }
        self.write_batch(&pool, &mut batch).await;
    }

    async fn write_batch(&self, pool: &SqlitePool, batch: &mut Vec<GroundTruthRecord>) {
        let pending_invalidation = self.shared.invalidation_pending.load(Ordering::SeqCst);
        if batch.is_empty() && !pending_invalidation {
            return;
        }
        let count = batch.len() as u64;
        let mut tx = match pool.begin().await {
            Ok(tx) => tx,
            Err(err) => {
                self.shared
                    .counters
                    .processed
                    .fetch_add(count, Ordering::SeqCst);
                for _ in 0..count {
                    self.shared.note_error("begin transaction", &err);
                }
                batch.clear();
                return;
            }
        };
        let session = self.shared.session_id();
        let mut written = 0_u64;
        for rec in batch.drain(..) {
            match self.write_one(&mut tx, &rec, session.as_deref()).await {
                Ok(true) => written += 1,
                Ok(false) => {}
                Err(err) => self.shared.note_error(record_name(&rec), &err),
            }
            self.shared
                .counters
                .processed
                .fetch_add(1, Ordering::SeqCst);
        }
        self.apply_invalidations(&mut tx).await;
        match tx.commit().await {
            Ok(()) => {
                self.shared
                    .counters
                    .written
                    .fetch_add(written, Ordering::Relaxed);
                self.shared.counters.batches.fetch_add(1, Ordering::Relaxed);
            }
            Err(err) => {
                for _ in 0..written {
                    self.shared.note_error("commit", &err);
                }
            }
        }
    }

    /// `Ok(true)` when the record was written, `Ok(false)` when it was
    /// deliberately skipped (counted elsewhere).
    async fn write_one(
        &self,
        conn: &mut SqliteConnection,
        rec: &GroundTruthRecord,
        session: Option<&str>,
    ) -> Result<bool, sqlx::Error> {
        let counters = &self.shared.counters;
        match rec {
            GroundTruthRecord::Event(event) => {
                if !event.journaled() {
                    return Ok(false);
                }
                if event.seq == 0 {
                    counters.rejected.fetch_add(1, Ordering::Relaxed);
                    *lock(&self.shared.last_error) =
                        Some(format!("unsequenced {} event rejected", event.name()));
                    return Ok(false);
                }
                let night_id = night::night_id(event.at_utc);
                equipment_events_repo::insert(&mut *conn, event, session, Some(&night_id)).await?;
                self.link_dithers(conn, event).await?;
            }
            GroundTruthRecord::ConnectionOpened(record) => {
                let night_id = night::night_id(record.connected_at);
                device_connections_repo::open(&mut *conn, record, session, Some(&night_id)).await?;
            }
            GroundTruthRecord::ConnectionClosed {
                connection_id,
                disconnected_at,
                end_class,
            } => {
                device_connections_repo::close(
                    &mut *conn,
                    *connection_id,
                    disconnected_at,
                    *end_class,
                )
                .await?;
            }
            GroundTruthRecord::FocuserMove(record) => {
                focuser_moves_repo::upsert(&mut *conn, record).await?;
            }
            GroundTruthRecord::GuideStep(record) => {
                guide_steps_repo::insert(&mut *conn, record).await?;
            }
            GroundTruthRecord::Dither(record) => {
                dither_repo::upsert(&mut *conn, record).await?;
            }
            GroundTruthRecord::ExposureWindow(row) => {
                exposure_windows_repo::upsert(&mut *conn, row).await?;
            }
            GroundTruthRecord::PointingSample(record) => {
                pointing_samples_repo::upsert(&mut *conn, record).await?;
            }
            GroundTruthRecord::Decision(record) => {
                policy_decisions_repo::upsert(&mut *conn, record).await?;
            }
            GroundTruthRecord::Audit(record) => {
                control_audit_repo::insert(&mut *conn, record).await?;
            }
            GroundTruthRecord::Environment(record) => {
                let night_id = night::night_id(record.observed_at);
                environment_repo::insert_probe(&mut *conn, record, Some(&night_id)).await?;
            }
            GroundTruthRecord::FilterSwitch(record) => {
                filter_switch_repo::upsert(&mut *conn, record).await?;
            }
            GroundTruthRecord::GuiderWindow(record) => {
                guider_state_repo::insert(&mut *conn, record).await?;
            }
            GroundTruthRecord::Telemetry(sample) => {
                telemetry_log_repo::insert(&mut *conn, sample).await?;
            }
            GroundTruthRecord::Profile(update) => {
                if let Err(reason) = profile::validate_patch(update) {
                    counters.rejected.fetch_add(1, Ordering::Relaxed);
                    *lock(&self.shared.last_error) = Some(reason);
                    return Ok(false);
                }
                profile::apply_update(&mut *conn, update).await?;
            }
        }
        Ok(true)
    }

    /// H10: an `exposure_completed` event links the pending dithers to its
    /// frame. The exposure's start is the driver start when known, else the
    /// completion minus the duration, else the completion itself.
    async fn link_dithers(
        &self,
        conn: &mut SqliteConnection,
        event: &EquipmentEvent,
    ) -> Result<(), sqlx::Error> {
        let EventKind::ExposureCompleted {
            exposure,
            completed_at,
        } = &event.event
        else {
            return Ok(());
        };
        let Some(frame_id) = exposure.frame_id.as_deref() else {
            return Ok(());
        };
        let start = exposure.started_at.unwrap_or_else(|| {
            exposure
                .duration_s
                .filter(|d| d.is_finite() && *d >= 0.0)
                .and_then(|d| chrono::Duration::from_std(Duration::from_secs_f64(d)).ok())
                .map(|d| *completed_at - d)
                .unwrap_or(*completed_at)
        });
        let linked = dither_repo::link_next_frame(
            &mut *conn,
            frame_id,
            &start,
            self.config.dither_link_horizon,
        )
        .await?;
        self.shared
            .counters
            .dithers_linked
            .fetch_add(linked, Ordering::Relaxed);
        Ok(())
    }

    /// Mark focusers whose profile update was dropped as unknown, once every
    /// record queued before the drop has been processed.
    async fn apply_invalidations(&self, conn: &mut SqliteConnection) {
        if !self.shared.invalidation_pending.load(Ordering::SeqCst) {
            return;
        }
        let processed = self.shared.counters.processed.load(Ordering::SeqCst);
        let due: Vec<Invalidation> = {
            let mut pending = lock(&self.shared.invalidations);
            let (due, keep): (Vec<_>, Vec<_>) = pending
                .drain(..)
                .partition(|inv| inv.after_processed <= processed);
            *pending = keep;
            if pending.is_empty() {
                self.shared
                    .invalidation_pending
                    .store(false, Ordering::SeqCst);
            }
            due
        };
        for inv in due {
            let record = RelativePosition::unknown(None, "recorder_dropped_update", inv.at);
            let update = profile::patch::relative_position(&inv.identity_key, &record);
            match profile::apply_update(&mut *conn, &update).await {
                Ok(()) => {
                    self.shared
                        .counters
                        .positions_invalidated
                        .fetch_add(1, Ordering::Relaxed);
                }
                Err(err) => self.shared.note_error("invalidate focuser position", &err),
            }
        }
    }
}

fn record_name(rec: &GroundTruthRecord) -> &'static str {
    match rec {
        GroundTruthRecord::Event(_) => "equipment_events",
        GroundTruthRecord::ConnectionOpened(_) | GroundTruthRecord::ConnectionClosed { .. } => {
            "device_connections"
        }
        GroundTruthRecord::FocuserMove(_) => "focuser_moves",
        GroundTruthRecord::GuideStep(_) => "guide_steps",
        GroundTruthRecord::Dither(_) => "dither_events",
        GroundTruthRecord::ExposureWindow(_) => "exposure_device_windows",
        GroundTruthRecord::PointingSample(_) => "pointing_samples",
        GroundTruthRecord::Decision(_) => "policy_decisions",
        GroundTruthRecord::Audit(_) => "control_audit_logs",
        GroundTruthRecord::Environment(_) => "environment_log",
        GroundTruthRecord::FilterSwitch(_) => "filter_switch_events",
        GroundTruthRecord::GuiderWindow(_) => "guider_state_log",
        GroundTruthRecord::Telemetry(_) => "telemetry_log",
        GroundTruthRecord::Profile(_) => "device_profiles",
    }
}

// ---------------------------------------------------------------------------
// Downsampling
// ---------------------------------------------------------------------------

const NEVER: i64 = i64::MIN;

/// The downsampling rule shared by telemetry and environment rows, on the
/// samples' own times (ms). `due` is when the next sample is wanted (`NEVER`
/// before the first). Returns the next due time when the sample at `at` is
/// kept.
///
/// - A sample is kept from one slack ([`DOWNSAMPLE_SLACK_FRACTION`] of the
///   period) before `due`, so a source polled at the target rate with jitter
///   keeps every sample.
/// - The next due time advances by one period, so a faster source averages
///   exactly the target rate.
/// - After a gap the schedule restarts from the sample; a sample from before
///   the previous one (the host clock stepped back) restarts it too.
fn next_due(due: i64, at: i64, period: i64) -> Option<i64> {
    let slack = (period as f64 * DOWNSAMPLE_SLACK_FRACTION) as i64;
    if due == NEVER || at < due.saturating_sub(period).saturating_sub(slack) {
        return Some(at.saturating_add(period));
    }
    if at < due.saturating_sub(slack) {
        return None;
    }
    Some(
        due.saturating_add(period)
            .max(at.saturating_add(period).saturating_sub(slack)),
    )
}

/// Per-kind rate limiter for `telemetry_log` (H19): one sample per period per
/// device kind on average, decided on the sample's own time, lock-free.
#[derive(Debug)]
pub struct Downsampler {
    /// Period in milliseconds; `None` disables the series.
    period_ms: Option<i64>,
    due_ms: [AtomicI64; 11],
}

impl Downsampler {
    /// `hz <= 0` (or not finite) disables the series.
    pub fn from_hz(hz: f64) -> Self {
        let period_ms = (hz.is_finite() && hz > 0.0).then(|| ((1000.0 / hz).round() as i64).max(1));
        Self {
            period_ms,
            due_ms: std::array::from_fn(|_| AtomicI64::new(NEVER)),
        }
    }

    fn slot(&self, kind: DeviceKind) -> &AtomicI64 {
        let index = DeviceKind::ALL.iter().position(|k| *k == kind).unwrap_or(0);
        &self.due_ms[index]
    }

    /// Whether [`Downsampler::accept`] would keep this sample, without
    /// recording it.
    pub fn would_accept(&self, kind: DeviceKind, at: DateTime<Utc>) -> bool {
        self.period_ms.is_some_and(|period| {
            next_due(
                self.slot(kind).load(Ordering::Acquire),
                at.timestamp_millis(),
                period,
            )
            .is_some()
        })
    }

    /// Keep (and remember) a sample, or refuse it as too soon.
    pub fn accept(&self, kind: DeviceKind, at: DateTime<Utc>) -> bool {
        let Some(period) = self.period_ms else {
            return false;
        };
        let at = at.timestamp_millis();
        let slot = self.slot(kind);
        let mut due = slot.load(Ordering::Acquire);
        loop {
            let Some(next) = next_due(due, at, period) else {
                return false;
            };
            match slot.compare_exchange_weak(due, next, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return true,
                Err(current) => due = current,
            }
        }
    }
}

/// Per-key limiter for `environment_log` probe rows: one row per device
/// identity per interval (§3.9), with the telemetry rule.
#[derive(Debug)]
pub struct IntervalThrottle {
    period_ms: i64,
    due_ms: Mutex<HashMap<String, i64>>,
}

impl IntervalThrottle {
    /// A zero interval keeps every row.
    pub fn new(interval: Duration) -> Self {
        Self {
            period_ms: i64::try_from(interval.as_millis()).unwrap_or(i64::MAX),
            due_ms: Mutex::new(HashMap::new()),
        }
    }

    pub fn accept(&self, key: &str, at: DateTime<Utc>) -> bool {
        if self.period_ms <= 0 {
            return true;
        }
        let mut due = lock(&self.due_ms);
        let current = due.get(key).copied().unwrap_or(NEVER);
        match next_due(current, at.timestamp_millis(), self.period_ms) {
            Some(next) => {
                due.insert(key.to_string(), next);
                true
            }
            None => false,
        }
    }
}

/// A telemetry sample of a device snapshot (H19), or `None` when the
/// snapshot does not serialise.
pub fn telemetry_sample<S: Serialize>(
    kind: DeviceKind,
    at: DateTime<Utc>,
    snapshot: &S,
) -> Option<TelemetrySample> {
    Some(TelemetrySample {
        source: kind,
        at,
        payload: serde_json::to_value(snapshot).ok()?,
    })
}

/// An `environment_log` record from a focuser temperature probe (F-f): the
/// probe temperature as `ambient_temp_c`, with the focuser's identity.
pub fn focuser_probe_row(
    observed_at: DateTime<Utc>,
    identity_key: &str,
    celsius: f64,
) -> Option<EnvironmentRecord> {
    celsius.is_finite().then(|| EnvironmentRecord {
        observed_at,
        device_identity_key: Some(identity_key.to_string()),
        ambient_temp_c: Some(celsius),
        ..EnvironmentRecord::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::{GatedCommand, GatedOp, MountCmd, PixelUnit, Settle};
    use crate::equipment::events::ProfileUpdate;
    use crate::equipment::events::{
        AuditRecord, AuditStage, ConnectionEnd, DeviceConnectionRecord, DitherBackend,
        DitherRecord, ExposureWindowRow, FilterSwitchRecord, FocuserMoveRecord, GuideStepRecord,
        GuiderStateRecord, PointingPurpose, PointingSampleRecord, PolicyDecisionRecord,
    };
    use crate::equipment::profile::{PositionCloseState, ProfileStore};
    use crate::equipment::snapshot::{
        BacklashModel, BacklashOwner, ExposureRef, GuideStepSample, MoveMode, PositionSource,
    };
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::Clock;
    use crate::equipment::types::{
        AutonomyMode, ClockQuality, CommandId, CommandSource, DeviceIdentity, DriverId, Endpoint,
        EqFrame, JNowFlavour, MountSyncMode, OpOutcome, PierSide, Priority,
    };
    use serde_json::json;
    use uuid::Uuid;

    const BUDGET: Duration = Duration::from_secs(5);

    /// O11 c backstop: a second `park_failed` for the same command is a
    /// duplicate report and is not journaled twice; other commands and other
    /// kinds are untouched.
    #[test]
    fn a_duplicate_park_failed_for_one_command_is_dropped() {
        let (recorder, _writer) = Recorder::channel(config());
        let clock = ManualClock::new();
        let command = CommandId::new();
        let park_failed = |command: CommandId, seq: u64| {
            let mut event = EquipmentEvent::new(
                EventKind::ParkFailed {
                    class: crate::equipment::types::ErrorClass::Driver,
                    reason: "park sensor fault".into(),
                },
                &clock,
            )
            .with_command(command);
            event.seq = seq;
            GroundTruthRecord::Event(Arc::new(event))
        };
        recorder.record(park_failed(command, 1));
        recorder.record(park_failed(command, 2));
        recorder.record(park_failed(CommandId::new(), 3));
        let mut parked = EquipmentEvent::new(
            EventKind::Parked {
                method: crate::equipment::events::ParkMethod::Driver,
            },
            &clock,
        )
        .with_command(command);
        parked.seq = 4;
        recorder.record(GroundTruthRecord::Event(Arc::new(parked)));
        let stats = recorder.stats();
        assert_eq!(stats.enqueued, 3);
        assert_eq!(stats.rejected, 1);
    }

    fn config() -> RecorderConfig {
        RecorderConfig {
            capacity: 64,
            ..RecorderConfig::default()
        }
    }

    fn event(seq: u64, kind: EventKind, clock: &ManualClock) -> GroundTruthRecord {
        GroundTruthRecord::Event(Arc::new(EquipmentEvent {
            seq,
            ..EquipmentEvent::new(kind, clock)
        }))
    }

    async fn count(pool: &SqlitePool, table: &str) -> i64 {
        let row: (i64,) = sqlx::query_as(&format!("SELECT COUNT(*) FROM {table}"))
            .fetch_one(pool)
            .await
            .unwrap();
        row.0
    }

    fn identity(kind: DeviceKind) -> DeviceIdentity {
        DeviceIdentity::new(
            kind,
            DriverId::Sim,
            Endpoint::InProcess { name: "sim".into() },
            crate::equipment::types::identity_key::sim(kind),
            "Simulator",
        )
    }

    fn dither(at: DateTime<Utc>) -> DitherRecord {
        DitherRecord {
            id: Uuid::new_v4(),
            command_id: Some(CommandId::new()),
            source: CommandSource::System,
            policy_version: None,
            backend: DitherBackend::Phd2,
            amount: 5.0,
            amount_unit: PixelUnit::GuidePx,
            ra_only: false,
            settle: Settle::default(),
            requested_at: at,
            dx: None,
            dy: None,
            settle_done_at: None,
            settle_status: None,
            settle_error: None,
            total_frames: None,
            dropped_frames: None,
            max_distance_px: None,
            next_frame_id: None,
            next_frame_accepted: None,
        }
    }

    async fn insert_frame(pool: &SqlitePool, id: &str) {
        sqlx::query("INSERT INTO frames (id, local_path, captured_at) VALUES (?, '/f', ?)")
            .bind(id)
            .bind("2026-01-01T00:00:00.000Z")
            .execute(pool)
            .await
            .unwrap();
    }

    /// Every `GroundTruthRecord` variant lands in its table, in one flush.
    #[tokio::test]
    async fn every_record_kind_reaches_its_table() {
        let pool = crate::storage::test_pool().await;
        insert_frame(&pool, "frame-1").await;
        let recorder = Recorder::start(pool.clone(), config());
        recorder.set_session(Some("session-a".into()));
        let clock = ManualClock::new();
        let at = clock.now_utc();
        let mount = identity(DeviceKind::Mount);
        let connection = mount.connection_id;
        let cmd = GatedCommand::new(
            GatedOp::Mount(MountCmd::Park),
            CommandSource::Operator { overriding: false },
            AutonomyMode::Manual,
            Priority::Exposure,
        );

        let records = vec![
            GroundTruthRecord::ConnectionOpened(Box::new(DeviceConnectionRecord {
                identity: mount.clone(),
                capabilities: json!({"can_park": true}),
                connected_at: at,
                session_id: None,
                night_id: None,
            })),
            event(1, EventKind::SlewStarted, &clock),
            GroundTruthRecord::FocuserMove(Box::new(FocuserMoveRecord {
                id: Uuid::new_v4(),
                command_id: None,
                connection_id: Uuid::new_v4(),
                mode: MoveMode::Relative,
                commanded_target: 100,
                logical_start: Some(0),
                logical_end: Some(100),
                raw_start: None,
                raw_end: None,
                position_source: PositionSource::Emulated,
                direction: None,
                reversal: None,
                backlash_model: BacklashModel::None,
                backlash_owner: BacklashOwner::Firmware,
                tempcomp_before: None,
                tempcomp_after: None,
                temperature_start_c: None,
                temperature_end_c: None,
                commanded_at: at,
                started_at: None,
                stopped_at: None,
                settled_at: None,
                outcome: OpOutcome::Succeeded,
                source: CommandSource::Operator { overriding: false },
                run_id: None,
            })),
            GroundTruthRecord::GuideStep(Box::new(GuideStepRecord {
                connection_id: Uuid::new_v4(),
                step: GuideStepSample {
                    frame: 1,
                    received_at_utc: at,
                    ..GuideStepSample::default()
                },
            })),
            GroundTruthRecord::Dither(Box::new(dither(at))),
            GroundTruthRecord::ExposureWindow(Box::new(ExposureWindowRow {
                frame_id: "frame-1".into(),
                clock_quality: ClockQuality::Simulator,
                ..ExposureWindowRow::default()
            })),
            GroundTruthRecord::PointingSample(Box::new(PointingSampleRecord {
                id: Uuid::new_v4(),
                plate_solve_event_id: None,
                frame_id: Some("frame-1".into()),
                mount_identity_key: mount.identity_key.clone(),
                mount_frame: EqFrame::J2000,
                jnow_flavour: JNowFlavour::Apparent,
                does_refraction: Some(false),
                mount_ra_native_h: None,
                mount_dec_native_deg: None,
                mount_ra_j2000_h: None,
                mount_dec_j2000_deg: None,
                solved_ra_j2000_h: 1.0,
                solved_dec_j2000_deg: 2.0,
                err_ra_arcsec: None,
                err_dec_arcsec: None,
                err_total_arcsec: None,
                pier_side: PierSide::East,
                ha_h: None,
                lst_h: None,
                alt_deg: None,
                az_deg: None,
                sync_generation: 0,
                sync_mode: MountSyncMode::Mount,
                coordinate_precision_arcsec: None,
                temperature_c: None,
                mid_exposure_utc: None,
                purpose: PointingPurpose::AutoSolve,
            })),
            GroundTruthRecord::Decision(Box::new(PolicyDecisionRecord {
                decision_id: Uuid::new_v4(),
                policy: "m2".into(),
                policy_version: "1".into(),
                mode: AutonomyMode::Advisory,
                context: json!({}),
                measurement_ages_ms: Default::default(),
                safe_arms: vec![],
                chosen_arm: None,
                propensities: None,
                requested_action: None,
                validated_action: None,
                executed_action: None,
                command_id: None,
                reward_components: None,
                attribution_window_start: None,
                attribution_window_end: None,
                interrupted: false,
                override_applied: false,
                parent_frame_ids: vec![],
                created_at: at,
            })),
            GroundTruthRecord::Audit(Box::new(AuditRecord::for_command(
                &cmd,
                AuditStage::Proposed,
                true,
                None,
                at,
            ))),
            GroundTruthRecord::Environment(Box::new(EnvironmentRecord {
                observed_at: at,
                device_identity_key: Some("sim:weather".into()),
                humidity_pct: Some(70.0),
                ..EnvironmentRecord::default()
            })),
            GroundTruthRecord::FilterSwitch(Box::new(FilterSwitchRecord {
                id: Uuid::new_v4(),
                session_id: None,
                from_filter: None,
                to_filter: "L".into(),
                focus_offset: None,
                duration_ms: None,
                success: true,
                command_id: None,
                from_slot: None,
                to_slot: 0,
                requested_at: at,
                wheel_done_at: None,
                verified_at: None,
                settle_ms_actual: None,
                offset_from: None,
                offset_to: None,
                offset_applied_steps: None,
                focuser_move_id: None,
                guiding_paused: false,
                handshake_ok: true,
                outcome: OpOutcome::Succeeded,
                trigger: None,
                policy_version: None,
                arm: None,
                propensity: None,
                background_before: None,
                background_after: None,
            })),
            GroundTruthRecord::GuiderWindow(Box::new(GuiderStateRecord {
                timestamp: at,
                session_id: None,
                connection_id: Uuid::new_v4(),
                frame_id: Some("frame-1".into()),
                window_start: at,
                window_end: at,
                rms: None,
                ra_aggressiveness: None,
                dec_aggressiveness: None,
                guide_star_snr: None,
                dither_active: false,
                algorithm_params: vec![],
                executed_arm_ra: None,
                executed_arm_dec: None,
                policy_version: None,
                freeze_reason: None,
            })),
            GroundTruthRecord::Telemetry(
                telemetry_sample(DeviceKind::Mount, at, &json!({"ra": 1})).unwrap(),
            ),
            GroundTruthRecord::Profile(profile::patch::last_used(&mount, "sim", at)),
            GroundTruthRecord::ConnectionClosed {
                connection_id: connection,
                disconnected_at: at,
                end_class: ConnectionEnd::Operator,
            },
        ];
        let n = records.len() as u64;
        for rec in records {
            recorder.record(rec);
        }
        recorder.flush(BUDGET).await.unwrap();

        for table in [
            "device_connections",
            "equipment_events",
            "focuser_moves",
            "guide_steps",
            "dither_events",
            "exposure_device_windows",
            "pointing_samples",
            "policy_decisions",
            "control_audit_logs",
            "environment_log",
            "filter_switch_events",
            "guider_state_log",
            "telemetry_log",
            "device_profiles",
        ] {
            assert_eq!(count(&pool, table).await, 1, "{table}");
        }
        let stats = recorder.stats();
        assert_eq!(stats.enqueued, n);
        assert_eq!(stats.written, n);
        assert_eq!((stats.failed, stats.dropped, stats.rejected), (0, 0, 0));
        assert_eq!(stats.pending, 0);

        let (session, night): (Option<String>, Option<String>) =
            sqlx::query_as("SELECT session_id, night_id FROM equipment_events")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(session.as_deref(), Some("session-a"));
        assert_eq!(night, Some(night::night_id(at)));
        let closed = device_connections_repo::load(&pool, connection)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(closed.end_class.as_deref(), Some("operator"));
        assert_eq!(closed.session_id.as_deref(), Some("session-a"));
        let stopped = recorder.shutdown(clock.now_utc(), BUDGET).await.unwrap();
        assert!(stopped.closed);
    }

    /// The caller never waits for the database: with the writer not running,
    /// a full queue drops and counts instead of blocking.
    #[tokio::test]
    async fn a_full_queue_drops_and_counts_without_blocking() {
        let pool = crate::storage::test_pool().await;
        let (recorder, writer) = Recorder::channel(RecorderConfig {
            capacity: 4,
            ..RecorderConfig::default()
        });
        let clock = ManualClock::new();
        let started = std::time::Instant::now();
        for seq in 1..=100 {
            recorder.record(event(seq, EventKind::SettleBegin, &clock));
        }
        // Also from a plain worker thread, outside the runtime.
        let from_thread = Arc::clone(&recorder);
        std::thread::spawn(move || {
            let clock = ManualClock::new();
            for seq in 101..=150 {
                from_thread.record(event(seq, EventKind::SettleBegin, &clock));
            }
        })
        .join()
        .unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "record() must never wait for the writer"
        );
        let stats = recorder.stats();
        assert_eq!(stats.enqueued, 4);
        assert_eq!(stats.dropped, 146);
        assert_eq!(stats.pending, 4);

        writer.spawn(pool.clone());
        recorder.flush(BUDGET).await.unwrap();
        assert_eq!(count(&pool, "equipment_events").await, 4);
        let seqs: Vec<(i64,)> = sqlx::query_as("SELECT seq FROM equipment_events ORDER BY seq")
            .fetch_all(&pool)
            .await
            .unwrap();
        assert_eq!(
            seqs,
            vec![(1,), (2,), (3,), (4,)],
            "the first four, in order"
        );
        assert_eq!(recorder.stats().written, 4);
    }

    /// A record that violates a constraint is counted; the rest of its batch
    /// is still written, in order.
    #[tokio::test]
    async fn a_failed_record_does_not_lose_its_batch() {
        let pool = crate::storage::test_pool().await;
        let (recorder, writer) = Recorder::channel(config());
        let clock = ManualClock::new();
        recorder.record(event(1, EventKind::GuidingStarted, &clock));
        // No such frame: the window's foreign key fails.
        recorder.record(GroundTruthRecord::ExposureWindow(Box::new(
            ExposureWindowRow {
                frame_id: "missing".into(),
                ..ExposureWindowRow::default()
            },
        )));
        // No such session: environment_log.session_id fails.
        recorder.record(GroundTruthRecord::Environment(Box::new(
            EnvironmentRecord {
                observed_at: clock.now_utc(),
                session_id: Some("no-session".into()),
                ..EnvironmentRecord::default()
            },
        )));
        recorder.record(event(2, EventKind::GuidingStopped, &clock));
        // Unsequenced events are refused; guide steps are not journaled.
        recorder.record(event(0, EventKind::GuiderPaused, &clock));
        recorder.record(event(
            3,
            EventKind::GuideStep {
                step: Box::default(),
            },
            &clock,
        ));
        writer.spawn(pool.clone());
        recorder.flush(BUDGET).await.unwrap();

        let stats = recorder.stats();
        assert_eq!(stats.processed, 6);
        assert_eq!(stats.failed, 2);
        assert_eq!(stats.rejected, 1);
        assert_eq!(stats.written, 2);
        assert_eq!(stats.batches, 1);
        assert!(stats.last_error.is_some());
        let kinds: Vec<(i64, String)> =
            sqlx::query_as("SELECT seq, kind FROM equipment_events ORDER BY seq")
                .fetch_all(&pool)
                .await
                .unwrap();
        assert_eq!(
            kinds,
            vec![
                (1, "guiding_started".to_string()),
                (2, "guiding_stopped".to_string())
            ]
        );
        assert_eq!(count(&pool, "environment_log").await, 0);
    }

    /// H10: the frame that follows a dither is linked by the recorder.
    #[tokio::test]
    async fn exposure_completed_links_the_pending_dither() {
        let pool = crate::storage::test_pool().await;
        let recorder = Recorder::start(pool.clone(), config());
        let clock = ManualClock::new();
        let first = dither(clock.now_utc());
        recorder.record(GroundTruthRecord::Dither(Box::new(first.clone())));
        clock.advance_ms(15_000);
        let started = clock.now_utc();
        clock.advance_ms(120_000);
        recorder.record(event(
            1,
            EventKind::ExposureCompleted {
                exposure: ExposureRef {
                    started_at: Some(started),
                    duration_s: Some(120.0),
                    frame_id: Some("frame-1".into()),
                    clock_quality: ClockQuality::DriverExposureStart,
                    ..ExposureRef::default()
                },
                completed_at: clock.now_utc(),
            },
            &clock,
        ));
        // A second dither, then a frame with an unknown start time: its start
        // is taken as completion minus duration.
        clock.advance_ms(1_000);
        let second = dither(clock.now_utc());
        recorder.record(GroundTruthRecord::Dither(Box::new(second.clone())));
        clock.advance_ms(65_000);
        recorder.record(event(
            2,
            EventKind::ExposureCompleted {
                exposure: ExposureRef {
                    started_at: None,
                    duration_s: Some(60.0),
                    frame_id: Some("frame-2".into()),
                    ..ExposureRef::default()
                },
                completed_at: clock.now_utc(),
            },
            &clock,
        ));
        // A preview with no frame id links nothing.
        recorder.record(event(
            3,
            EventKind::ExposureCompleted {
                exposure: ExposureRef::default(),
                completed_at: clock.now_utc(),
            },
            &clock,
        ));
        recorder.flush(BUDGET).await.unwrap();

        let first = dither_repo::load(&pool, first.id).await.unwrap().unwrap();
        let second = dither_repo::load(&pool, second.id).await.unwrap().unwrap();
        assert_eq!(first.next_frame_id.as_deref(), Some("frame-1"));
        assert_eq!(second.next_frame_id.as_deref(), Some("frame-2"));
        assert_eq!(recorder.stats().dithers_linked, 2);
    }

    /// H19: 1 Hz per kind on average, decided on sample time.
    #[tokio::test]
    async fn telemetry_is_downsampled_per_kind() {
        let pool = crate::storage::test_pool().await;
        let recorder = Recorder::start(
            pool.clone(),
            RecorderConfig {
                capacity: 1024,
                ..RecorderConfig::default()
            },
        );
        let clock = ManualClock::new();
        // 10 Hz mount snapshots and 2 Hz focuser snapshots for 100 s.
        let mut due_ticks = 0;
        for tick in 0..1000 {
            let at = clock.now_utc();
            let due = recorder.telemetry_due(DeviceKind::Mount, at);
            due_ticks += usize::from(due);
            recorder.record(GroundTruthRecord::Telemetry(
                telemetry_sample(DeviceKind::Mount, at, &json!({"tick": tick})).unwrap(),
            ));
            // telemetry_due agrees with what record() decided.
            assert!(
                !recorder.telemetry_due(DeviceKind::Mount, at),
                "tick {tick}"
            );
            if tick % 5 == 0 {
                recorder.record(GroundTruthRecord::Telemetry(
                    telemetry_sample(DeviceKind::Focuser, at, &json!({"tick": tick})).unwrap(),
                ));
            }
            clock.advance_ms(100);
        }
        recorder.flush(BUDGET).await.unwrap();
        let rows: Vec<(String, String)> =
            sqlx::query_as("SELECT source, payload FROM telemetry_log ORDER BY id")
                .fetch_all(&pool)
                .await
                .unwrap();
        let mounts = rows.iter().filter(|(s, _)| s == "mount").count();
        let focusers = rows.iter().filter(|(s, _)| s == "focuser").count();
        // 100 s at 1 Hz: one row per second, plus the first sample's phase.
        assert!((100..=101).contains(&mounts), "{mounts} mount rows");
        assert_eq!(mounts, due_ticks);
        assert_eq!(focusers, 100);
        assert_eq!(rows[0], ("mount".to_string(), r#"{"tick":0}"#.to_string()));
        let kept = (mounts + focusers) as u64;
        assert_eq!(recorder.stats().downsampled, 1000 + 200 - kept);

        // A 1 Hz source with jitter keeps every sample.
        let sampler = Downsampler::from_hz(1.0);
        let t0 = clock.now_utc();
        for (i, offset_ms) in [0_i64, 980, 2010, 2950, 4000].iter().enumerate() {
            let at = t0 + chrono::Duration::milliseconds(*offset_ms);
            assert!(sampler.accept(DeviceKind::Weather, at), "sample {i}");
        }
        // A host clock step backwards restarts the series.
        assert!(sampler.accept(DeviceKind::Weather, t0));
        // 0 Hz disables the series.
        let off = Downsampler::from_hz(0.0);
        assert!(!off.accept(DeviceKind::Mount, t0));
        assert!(!off.would_accept(DeviceKind::Mount, t0));
    }

    /// One probe row per device identity per `[weather] log_interval_s`.
    #[tokio::test]
    async fn environment_rows_follow_the_log_interval() {
        let pool = crate::storage::test_pool().await;
        let recorder = Recorder::start(
            pool.clone(),
            RecorderConfig {
                environment_interval_s: 60,
                ..config()
            },
        );
        let clock = ManualClock::new();
        for _ in 0..13 {
            // A weather device polled every 10 s and a focuser probe.
            recorder.record(GroundTruthRecord::Environment(Box::new(
                EnvironmentRecord {
                    observed_at: clock.now_utc(),
                    device_identity_key: Some("alpaca:oc".into()),
                    ambient_temp_c: Some(4.0),
                    ..EnvironmentRecord::default()
                },
            )));
            if let Some(row) = focuser_probe_row(clock.now_utc(), "alpaca:focuser", 6.5) {
                recorder.record(GroundTruthRecord::Environment(Box::new(row)));
            }
            clock.advance_ms(10_000);
        }
        assert!(focuser_probe_row(clock.now_utc(), "alpaca:focuser", f64::NAN).is_none());
        recorder.flush(BUDGET).await.unwrap();
        let rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT device_identity_key, COUNT(*) FROM environment_log \
             GROUP BY device_identity_key ORDER BY device_identity_key",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        // 0, 60 and 120 s.
        assert_eq!(
            rows,
            vec![
                ("alpaca:focuser".to_string(), 3),
                ("alpaca:oc".to_string(), 3)
            ]
        );
        assert!(IntervalThrottle::new(Duration::ZERO).accept("k", clock.now_utc()));
    }

    /// §2.15: a dropped focuser profile write can never leave a stale origin
    /// behind. The record that was dropped might have been `in_motion`.
    #[tokio::test]
    async fn a_dropped_focuser_update_invalidates_its_position() {
        let pool = crate::storage::test_pool().await;
        let store = ProfileStore::new(pool.clone());
        let key = "alpaca:relative-focuser";
        let clock = ManualClock::new();
        let connection = Uuid::new_v4();
        let (recorder, writer) = Recorder::channel(RecorderConfig {
            capacity: 2,
            ..RecorderConfig::default()
        });
        // A completed move is queued, then the queue fills and the next
        // update (the start of another move) is dropped.
        recorder.record(GroundTruthRecord::Profile(
            profile::patch::relative_position(
                key,
                &RelativePosition::completed(1200, connection, None, clock.now_utc()),
            ),
        ));
        recorder.record(event(1, EventKind::SettleBegin, &clock));
        clock.advance_ms(1000);
        recorder.record(GroundTruthRecord::Profile(
            profile::patch::relative_position(
                key,
                &RelativePosition::in_motion(connection, None, clock.now_utc()),
            ),
        ));
        assert_eq!(recorder.stats().dropped, 1);

        writer.spawn(pool.clone());
        recorder.flush(BUDGET).await.unwrap();
        let restored = store.restore_relative_position(key).await.unwrap();
        assert_eq!(restored.source, PositionSource::EmulatedOriginUnknown);
        assert_eq!(restored.reason, "recorder_dropped_update");
        assert_eq!(recorder.stats().positions_invalidated, 1);
        let stored = store.load(key).await.unwrap().unwrap();
        assert_eq!(
            stored.relative_position().unwrap().state,
            PositionCloseState::Unknown
        );

        // A later completed move makes it restorable again.
        clock.advance_ms(1000);
        recorder.record(GroundTruthRecord::Profile(
            profile::patch::relative_position(
                key,
                &RelativePosition::completed(1350, connection, None, clock.now_utc()),
            ),
        ));
        recorder.flush(BUDGET).await.unwrap();
        let restored = store.restore_relative_position(key).await.unwrap();
        assert_eq!(restored.position, Some(1350));

        // Non-focuser drops do not touch positions; malformed patches are
        // rejected.
        recorder.record(GroundTruthRecord::Profile(ProfileUpdate {
            identity_key: key.into(),
            kind: DeviceKind::Focuser,
            patch: json!(["not", "an", "object"]),
            updated_at: clock.now_utc(),
        }));
        recorder.flush(BUDGET).await.unwrap();
        assert_eq!(recorder.stats().rejected, 1);
        assert_eq!(
            store.restore_relative_position(key).await.unwrap().position,
            Some(1350)
        );
    }

    /// §2.15 steps 6 and 7: shutdown writes what was queued, closes the
    /// connections still open with `end_class = shutdown`, then refuses new
    /// records.
    #[tokio::test]
    async fn shutdown_flushes_closes_connections_then_refuses() {
        let pool = crate::storage::test_pool().await;
        let recorder = Recorder::start(pool.clone(), config());
        let clock = ManualClock::new();
        let mount = identity(DeviceKind::Mount);
        recorder.record(GroundTruthRecord::ConnectionOpened(Box::new(
            DeviceConnectionRecord {
                identity: mount.clone(),
                capabilities: json!({}),
                connected_at: clock.now_utc(),
                session_id: None,
                night_id: None,
            },
        )));
        for seq in 1..=10 {
            recorder.record(event(seq, EventKind::SettleBegin, &clock));
        }
        clock.advance_ms(3_600_000);
        let stats = recorder
            .shutdown(clock.now_utc(), Duration::from_secs(2))
            .await
            .unwrap();
        assert_eq!(stats.written, 11);
        assert!(stats.closed);
        let row = device_connections_repo::load(&pool, mount.connection_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(row.end_class.as_deref(), Some("shutdown"));
        assert_eq!(
            row.disconnected_at.as_deref(),
            Some("2026-01-01T01:00:00.000Z")
        );

        recorder.record(event(11, EventKind::SettleBegin, &clock));
        assert_eq!(recorder.stats().dropped, 1);
        assert_eq!(count(&pool, "equipment_events").await, 10);
        assert_eq!(recorder.flush(BUDGET).await, Err(FlushError::WriterGone));
    }

    /// Attaching to a store closes the rows a crashed process left open, so
    /// they never claim a device stayed connected.
    #[tokio::test]
    async fn attaching_closes_connections_left_open_by_a_crash() {
        let pool = crate::storage::test_pool().await;
        let clock = ManualClock::new();
        let stale = DeviceConnectionRecord {
            identity: identity(DeviceKind::Focuser),
            capabilities: json!({}),
            connected_at: clock.now_utc(),
            session_id: None,
            night_id: None,
        };
        device_connections_repo::open(&pool, &stale, None, None)
            .await
            .unwrap();
        let recorder = Recorder::start(pool.clone(), config());
        // A connection recorded after attaching stays open.
        let live = identity(DeviceKind::Mount);
        recorder.record(GroundTruthRecord::ConnectionOpened(Box::new(
            DeviceConnectionRecord {
                identity: live.clone(),
                capabilities: json!({}),
                connected_at: clock.now_utc(),
                session_id: None,
                night_id: None,
            },
        )));
        recorder.flush(BUDGET).await.unwrap();
        let row = device_connections_repo::load(&pool, stale.identity.connection_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(row.end_class.as_deref(), Some("lost"));
        assert_eq!(row.disconnected_at, Some(row.connected_at.clone()));
        let row = device_connections_repo::load(&pool, live.connection_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(row.disconnected_at, None);
    }

    /// A worker thread can make its writes durable before moving hardware.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flush_blocking_serves_worker_threads_only() {
        let pool = crate::storage::test_pool().await;
        let recorder = Recorder::start(pool.clone(), config());
        assert_eq!(
            recorder.flush_blocking(BUDGET),
            Err(FlushError::InAsyncContext)
        );
        let worker = Arc::clone(&recorder);
        let result = tokio::task::spawn_blocking(move || {
            // spawn_blocking threads carry a runtime handle; a real device
            // worker is a plain std thread.
            std::thread::spawn(move || {
                let clock = ManualClock::new();
                worker.record(event(1, EventKind::SettleBegin, &clock));
                worker.flush_blocking(BUDGET)
            })
            .join()
            .unwrap()
        })
        .await
        .unwrap();
        assert_eq!(result, Ok(()));
        assert_eq!(count(&pool, "equipment_events").await, 1);

        let (unstarted, _writer) = Recorder::channel(config());
        let result =
            std::thread::spawn(move || unstarted.flush_blocking(Duration::from_millis(50)))
                .join()
                .unwrap();
        assert_eq!(result, Err(FlushError::TimedOut));
    }

    #[test]
    fn config_comes_from_the_equipment_sections() {
        let configs = crate::config::AchernarConfig::default().equipment_configs();
        let config = RecorderConfig::from_configs(&configs);
        assert_eq!(config.telemetry_hz, configs.equipment.telemetry_log_hz);
        assert_eq!(
            config.environment_interval_s,
            configs.weather.log_interval_s
        );
        assert!(config.capacity > 0 && config.max_batch > 0);
        let off = RecorderConfig {
            telemetry_hz: 0.0,
            ..config
        };
        assert_eq!(off.telemetry_hz, 0.0, "0 disables telemetry rows");
    }

    #[test]
    fn the_schedule_keeps_the_target_rate_and_restarts_after_gaps() {
        // First sample, on time, early within the slack, too early.
        assert_eq!(next_due(NEVER, 5_000, 1_000), Some(6_000));
        assert_eq!(next_due(6_000, 6_000, 1_000), Some(7_000));
        assert_eq!(next_due(6_000, 5_900, 1_000), Some(7_000));
        assert_eq!(next_due(6_000, 5_899, 1_000), None);
        // After a long gap the schedule restarts from the sample.
        assert_eq!(next_due(6_000, 60_000, 1_000), Some(60_900));
        // The host clock stepped back by more than a period: restart.
        assert_eq!(next_due(6_000, 3_000, 1_000), Some(4_000));
    }
}
