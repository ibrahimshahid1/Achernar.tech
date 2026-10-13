//! Bounded archival work for physical exposures.
//!
//! Reserve capacity before commanding the camera. A full or disconnected queue
//! then prevents a new exposure instead of dropping bytes after the shutter has
//! fired. Submit owned bytes only after acquisition has left the camera mutex;
//! neither submission nor the acknowledgement belongs in a pixel callback.
//!
//! An acknowledgement means that `CaptureStore` completed its local archive
//! transaction. It is not a qualified storage durability receipt or permission
//! to delete camera originals. Once submitted, a job belongs to the worker and
//! continues even if its caller stops waiting for the acknowledgement.

use sqlx::SqlitePool;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};

use crate::error::{Result, StorageError};

use super::capture_store::{CaptureStore, CommittedCapture, PendingCapture};

struct ArchiveJob {
    capture: PendingCapture,
    acknowledgement: oneshot::Sender<Result<CommittedCapture>>,
}

/// One archival worker shared by acquisition callers.
///
/// The queue bounds waiting jobs and unsubmitted reservations. The worker can
/// also own one in-flight job. Byte limits for native camera files are a
/// separate acquisition constraint; a job count is not a byte budget.
#[derive(Clone)]
pub struct CaptureArchive {
    sender: mpsc::Sender<ArchiveJob>,
    failure: Arc<Mutex<Option<String>>>,
}

impl CaptureArchive {
    /// Start the worker on the current Tokio runtime.
    ///
    /// `capacity` must be positive. The worker owns only the store and pool,
    /// so dropping every sender and reservation lets it finish queued jobs and
    /// exit without retaining application state through a reference cycle.
    pub fn spawn(store: CaptureStore, pool: SqlitePool, capacity: usize) -> Self {
        assert!(capacity > 0, "capture archive capacity must be positive");
        let (sender, mut receiver) = mpsc::channel::<ArchiveJob>(capacity);
        let failure = Arc::new(Mutex::new(None));
        let worker_failure = failure.clone();
        tokio::spawn(async move {
            while let Some(job) = receiver.recv().await {
                let capture_id = job.capture.capture_id.clone();
                let result = store.persist_capture(&pool, job.capture).await;
                if let Err(error) = &result {
                    *worker_failure.lock().unwrap() = Some(error.to_string());
                    // Record the failure even when the request no longer
                    // exists to do so. Preserve the original storage error.
                    store
                        .fail_attempt(&pool, &capture_id, &error.to_string())
                        .await;
                }
                if job.acknowledgement.send(result).is_err() {
                    tracing::debug!(
                        "Capture {capture_id} archive work finished after its caller stopped waiting"
                    );
                }
            }
        });
        Self { sender, failure }
    }

    /// Reserve an archival slot before triggering a physical exposure.
    ///
    /// This never waits for storage. A caller must report the error and pause
    /// new science triggers when the queue is full or the worker is gone.
    pub fn reserve(&self) -> Result<ArchiveReservation> {
        if let Some(error) = self.failure.lock().unwrap().as_ref() {
            return Err(StorageError::Database(format!(
                "Capture archive is paused after a failed write: {error}. Repair storage and restart the daemon before new science exposures."
            )).into());
        }
        let permit = self.sender.clone().try_reserve_owned().map_err(|error| {
            let reason = match error {
                mpsc::error::TrySendError::Full(_) => {
                    "Capture archive queue is full; pause new science exposures"
                }
                mpsc::error::TrySendError::Closed(_) => {
                    "Capture archive worker is unavailable; pause new science exposures"
                }
            };
            StorageError::Database(reason.to_string())
        })?;
        Ok(ArchiveReservation { permit })
    }
}

/// Capacity owned by one acquisition attempt.
///
/// Dropping an unused reservation releases its slot. After submission, dropping
/// the waiting future does not cancel the worker's archival transaction.
pub struct ArchiveReservation {
    permit: mpsc::OwnedPermit<ArchiveJob>,
}

impl ArchiveReservation {
    /// Submit the complete owned exposure bundle and await its local commit.
    ///
    /// The result identifies the capture, frame and every committed asset,
    /// including hashes and byte lengths. An error or missing acknowledgement
    /// must never become an accepted archived scientific sample.
    pub async fn persist(self, capture: PendingCapture) -> Result<CommittedCapture> {
        let capture_id = capture.capture_id.clone();
        let (acknowledgement, receiver) = oneshot::channel();
        // Sending through the reserved permit cannot wait on queue capacity.
        // From this point onward the worker owns the bytes independently of
        // this future and its acknowledgement receiver.
        self.permit.send(ArchiveJob {
            capture,
            acknowledgement,
        });
        receiver.await.map_err(|_| {
            StorageError::Database(format!(
                "Capture {capture_id} archive acknowledgement was lost; local commit status is unknown"
            ))
        })?
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Duration;

    use sha2::{Digest, Sha256};

    use super::*;
    use crate::storage::capture_store::{AssetRole, PendingAsset, TemperatureState};

    fn test_dir() -> PathBuf {
        std::env::temp_dir().join(format!("achernar-archive-{}", uuid::Uuid::new_v4()))
    }

    async fn test_pool() -> SqlitePool {
        // One connection also lets the cancellation test stop a transaction
        // after filesystem staging, without modifying the production worker.
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        for (_, _, sql) in crate::storage::local_store::migrations_for_test() {
            for statement in crate::storage::local_store::split_statements_for_test(sql) {
                if let Err(error) = sqlx::query(&statement).execute(&pool).await {
                    let message = error.to_string();
                    assert!(
                        message.contains("duplicate column name")
                            || message.contains("already exists"),
                        "unexpected migration failure: {error}"
                    );
                }
            }
        }
        pool
    }

    fn sample(frame_id: &str) -> PendingCapture {
        PendingCapture {
            capture_id: format!("capture-{frame_id}"),
            frame_id: frame_id.to_string(),
            session_id: None,
            rig_id: "rig-1".into(),
            camera_name: Some("Archive test camera".into()),
            driver_name: Some("test".into()),
            capture_type: "focus".into(),
            exposure_started_at: 1_785_636_000.0,
            exposure_completed_at: 1_785_636_004.0,
            requested_exposure_s: 4.0,
            requested_gain: Some(800.0),
            applied_exposure_s: 4.0,
            iso: Some(800.0),
            shutter_label: Some("4".into()),
            offset_val: None,
            binning: 1,
            filter_name: None,
            focuser_position: Some(1_000),
            temperature: None,
            temperature_source: None,
            temperature_state: TemperatureState::Missing,
            acquisition: serde_json::json!({"retention_policy": "retain_on_camera"}),
            assets: vec![
                PendingAsset::new(
                    AssetRole::NativeRaw,
                    "image.cr3",
                    "image/x-canon-cr3",
                    b"CAMERA RAW BYTES".to_vec(),
                ),
                PendingAsset::new(
                    AssetRole::CameraJpeg,
                    "image.jpg",
                    "image/jpeg",
                    b"CAMERA JPEG BYTES".to_vec(),
                ),
            ],
        }
    }

    async fn begin_attempt(store: &CaptureStore, pool: &SqlitePool, capture: &PendingCapture) {
        store
            .begin_attempt(
                pool,
                &capture.capture_id,
                None,
                &capture.rig_id,
                &serde_json::json!({}),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn full_queue_rejects_before_submission_and_unused_slot_is_released() {
        let directory = test_dir();
        let pool = test_pool().await;
        let archive = CaptureArchive::spawn(CaptureStore::new(&directory), pool.clone(), 1);
        let reservation = archive.reserve().unwrap();
        let error = archive.reserve().err().expect("queue must be full");
        assert!(error.to_string().contains("queue is full"), "{error}");
        assert!(!directory.exists(), "no archive job was submitted");
        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM frames")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 0);

        drop(reservation);
        assert!(archive.reserve().is_ok(), "unused capacity must return");
    }

    #[tokio::test]
    async fn disconnected_worker_rejects_before_submission() {
        let (sender, receiver) = mpsc::channel(1);
        let archive = CaptureArchive {
            sender,
            failure: Arc::new(Mutex::new(None)),
        };
        drop(receiver);
        let error = archive.reserve().err().expect("worker must be unavailable");
        assert!(
            error.to_string().contains("worker is unavailable"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn acknowledgement_contains_the_committed_native_bundle() {
        let directory = test_dir();
        let pool = test_pool().await;
        let store = CaptureStore::new(&directory);
        let capture = sample("bundle");
        begin_attempt(&store, &pool, &capture).await;
        let expected = capture.assets.clone();
        let archive = CaptureArchive::spawn(store, pool.clone(), 1);
        let committed = archive.reserve().unwrap().persist(capture).await.unwrap();

        assert_eq!(committed.capture_id, "capture-bundle");
        assert_eq!(committed.frame_id, "bundle");
        assert_eq!(committed.status, "committed");
        assert_eq!(committed.assets.len(), 2);
        for (asset, original) in committed.assets.iter().zip(&expected) {
            assert_eq!(asset.role, original.role.as_str());
            assert_eq!(asset.byte_length, original.bytes.len() as i64);
            assert_eq!(
                asset.sha256,
                format!("{:x}", Sha256::digest(&original.bytes))
            );
            assert_eq!(
                std::fs::read(directory.join(&asset.relative_path)).unwrap(),
                original.bytes
            );
        }
        let (capture_id,): (String,) = sqlx::query_as("SELECT capture_id FROM frames WHERE id = ?")
            .bind(&committed.frame_id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(capture_id, committed.capture_id);
        let (count,): (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM frame_assets WHERE frame_id = ?")
                .bind(&committed.frame_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count, 2);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn persistence_failure_reaches_the_caller_and_attempt() {
        let directory = test_dir();
        let pool = test_pool().await;
        let store = CaptureStore::new(&directory);
        let mut capture = sample("failed");
        capture.assets.clear();
        begin_attempt(&store, &pool, &capture).await;
        let archive = CaptureArchive::spawn(store, pool.clone(), 1);

        let error = archive
            .reserve()
            .unwrap()
            .persist(capture)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("no assets"), "{error}");
        let (status, error): (String, String) = sqlx::query_as(
            "SELECT status, error FROM capture_attempts WHERE capture_id = 'capture-failed'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(status, "failed");
        assert!(error.contains("no assets"));
        assert!(!directory.exists());
        assert!(archive
            .reserve()
            .err()
            .unwrap()
            .to_string()
            .contains("paused after a failed write"));
    }

    #[tokio::test]
    async fn lost_acknowledgement_reports_unknown_commit_status() {
        let (sender, mut receiver) = mpsc::channel(1);
        let archive = CaptureArchive {
            sender,
            failure: Arc::new(Mutex::new(None)),
        };
        let reservation = archive.reserve().unwrap();
        let waiting = tokio::spawn(reservation.persist(sample("lost-ack")));
        // Model a worker exiting after receiving owned bytes, before replying.
        let job = receiver.recv().await.unwrap();
        drop(job);
        let error = waiting.await.unwrap().unwrap_err();
        assert!(
            error.to_string().contains("local commit status is unknown"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn submitted_job_commits_after_its_waiting_caller_is_cancelled() {
        let directory = test_dir();
        let pool = test_pool().await;
        let store = CaptureStore::new(&directory);
        let capture = sample("cancelled");
        begin_attempt(&store, &pool, &capture).await;
        let started_at =
            chrono::DateTime::from_timestamp(capture.exposure_started_at as i64, 0).unwrap();
        let staging_manifest = directory
            .join(&capture.rig_id)
            .join(crate::storage::night::night_id(started_at))
            .join("no-session")
            .join(".cancelled.partial")
            .join("manifest.v1.json");

        // Block the worker's database transaction after it stages the files.
        let connection = pool.acquire().await.unwrap();
        let archive = CaptureArchive::spawn(store, pool.clone(), 1);
        let waiting = tokio::spawn(archive.reserve().unwrap().persist(capture));
        tokio::time::timeout(Duration::from_secs(5), async {
            while !staging_manifest.exists() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("worker must own and stage the submitted bundle");
        waiting.abort();
        assert!(waiting.await.unwrap_err().is_cancelled());
        drop(connection);

        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let (count,): (i64,) = sqlx::query_as(
                    "SELECT COUNT(*) FROM frames WHERE id = 'cancelled' AND capture_status = 'committed'",
                )
                .fetch_one(&pool)
                .await
                .unwrap();
                if count == 1 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("caller cancellation must not cancel the archival transaction");
        let (count,): (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM frame_assets WHERE frame_id = 'cancelled'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count, 2);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
