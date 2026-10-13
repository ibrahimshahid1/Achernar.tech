//! capture_store — the canonical destination for a physical exposure.
//!
//! One shutter actuation produces one `frames` row and one or more immutable
//! files recorded in `frame_assets`. SQLite is the catalogue; the filesystem
//! holds the bytes. Parquet, when it happens, is a reproducible export from
//! here — never the primary destination.
//!
//! # Why this is not just `fs::write`
//!
//! `/capture` used to write a FITS and a RAW sidecar directly and log a
//! checksum it then threw away. Nothing in the database knew those files
//! existed, so the dataset builder could not find them, sync had nothing to
//! ship, and a half-finished write left a plausible-looking file with no way
//! to tell it was truncated. An archive you cannot verify is not an archive.
//!
//! # Ordering
//!
//! Filesystem and SQLite cannot commit atomically together, so the ordering is
//! chosen to make the *recoverable* failure the likely one:
//!
//! 1. stage every file under `.<frame-id>.partial`, hashing while writing
//! 2. write `manifest.v1.json` describing what should be there
//! 3. fsync the files and attempt to sync the staging directory
//! 4. open the SQLite transaction and insert frames + frame_assets + outbox
//! 5. rename the staging directory into place
//! 6. commit
//!
//! An interruption before step 5 can leave a `.partial` directory without
//! database rows; an interruption between 5 and 6 can leave a final directory
//! without rows. The manifest records the intended contents. This ordering is
//! a local commit protocol, not a qualified power-loss durability guarantee:
//! checked directory/rename-parent synchronization and orphan reconstruction
//! remain A12 work. Startup reconciliation currently removes partial staging
//! and checks catalogued asset existence/length; it does not reconstruct
//! orphan bundles or verify their content hashes.
//!
//! # Native success is the success that counts
//!
//! For a DSLR, the camera-authored CR3 is the canonical exposure asset. FITS
//! conversion and luma extraction are derivations that can be redone later,
//! so a failure to derive must never discard a valid native asset. Successful
//! local persistence alone cannot authorize deleting camera originals.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::SqlitePool;

use crate::error::{Result, StorageError};

/// Layout version written into every manifest.
const MANIFEST_VERSION: &str = "v1";
const MANIFEST_FILENAME: &str = "manifest.v1.json";

// ═══════════════════════════════════════════════════════════════════
//  Input types
// ═══════════════════════════════════════════════════════════════════

/// Where an asset sits in the bundle, and what it means.
///
/// The subdirectory is part of the role rather than a caller choice, so the
/// on-disk layout stays predictable enough to walk without the database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetRole {
    /// The camera's own RAW (CR2/CR3/RW2/NEF/ARW). Canonical for a DSLR.
    NativeRaw,
    /// The camera's own JPEG, when the body was set to RAW+JPEG or JPEG.
    CameraJpeg,
    /// FITS containing sensor data. It is derived when `derived_from` names a
    /// camera-authored asset; without lineage it is the canonical encoding of
    /// a stream-only astronomy camera exposure.
    SensorFits,
    /// 16-bit luma used by the vision/FQE pipeline.
    AnalysisLuma,
    /// Canonical float32 linear analysis plane; never an integer preview.
    AnalysisFits,
    /// One binary mask plane identified by the analysis descriptor.
    AnalysisMask,
    /// Versioned JSON analysis recipe, geometry and content hashes.
    AnalysisDescriptor,
    /// Small preview for the UI.
    Thumbnail,
    /// Calibrated FITS produced by a later reprocessing pass.
    CalibratedFits,
}

impl AssetRole {
    pub fn as_str(self) -> &'static str {
        match self {
            AssetRole::NativeRaw => "native_raw",
            AssetRole::CameraJpeg => "camera_jpeg",
            AssetRole::SensorFits => "sensor_fits",
            AssetRole::AnalysisLuma => "analysis_luma",
            AssetRole::AnalysisFits => "analysis_fits",
            AssetRole::AnalysisMask => "analysis_mask",
            AssetRole::AnalysisDescriptor => "analysis_descriptor",
            AssetRole::Thumbnail => "thumbnail",
            AssetRole::CalibratedFits => "calibrated_fits",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "native_raw" => AssetRole::NativeRaw,
            "camera_jpeg" => AssetRole::CameraJpeg,
            "sensor_fits" => AssetRole::SensorFits,
            "analysis_luma" => AssetRole::AnalysisLuma,
            "analysis_fits" => AssetRole::AnalysisFits,
            "analysis_mask" => AssetRole::AnalysisMask,
            "analysis_descriptor" => AssetRole::AnalysisDescriptor,
            "thumbnail" => AssetRole::Thumbnail,
            "calibrated_fits" => AssetRole::CalibratedFits,
            _ => return None,
        })
    }

    fn subdir(self) -> &'static str {
        match self {
            AssetRole::NativeRaw | AssetRole::CameraJpeg => "native",
            AssetRole::SensorFits | AssetRole::CalibratedFits => "derived",
            AssetRole::AnalysisLuma
            | AssetRole::AnalysisFits
            | AssetRole::AnalysisMask
            | AssetRole::AnalysisDescriptor => "analysis",
            AssetRole::Thumbnail => "preview",
        }
    }

    /// True when losing this asset loses the exposure. Derived products are
    /// reproducible from a native one; a native one is not reproducible at all.
    fn is_native(self) -> bool {
        matches!(self, AssetRole::NativeRaw | AssetRole::CameraJpeg)
    }
}

/// One file to be written as part of a capture.
#[derive(Debug, Clone)]
pub struct PendingAsset {
    pub role: AssetRole,
    /// Filename within the role's subdirectory, e.g. `image.cr3`.
    pub filename: String,
    pub media_type: String,
    pub bytes: Vec<u8>,
    /// The camera's own name for the file, when it had one.
    pub original_filename: Option<String>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub bit_depth: Option<i64>,
    /// Role this was derived from, recorded as lineage.
    pub derived_from: Option<AssetRole>,
    /// Parent filename within `derived_from`'s role. Required when that role
    /// occurs more than once; role alone cannot identify the source exposure.
    pub derived_from_filename: Option<String>,
    pub provenance: serde_json::Value,
}

impl PendingAsset {
    pub fn new(
        role: AssetRole,
        filename: impl Into<String>,
        media_type: impl Into<String>,
        bytes: Vec<u8>,
    ) -> Self {
        Self {
            role,
            filename: filename.into(),
            media_type: media_type.into(),
            bytes,
            original_filename: None,
            width: None,
            height: None,
            bit_depth: None,
            derived_from: None,
            derived_from_filename: None,
            provenance: serde_json::json!({}),
        }
    }

    pub fn with_dimensions(mut self, w: i64, h: i64, bit_depth: i64) -> Self {
        self.width = Some(w);
        self.height = Some(h);
        self.bit_depth = Some(bit_depth);
        self
    }

    pub fn derived_from(mut self, role: AssetRole) -> Self {
        self.derived_from = Some(role);
        self.derived_from_filename = None;
        self
    }

    /// Name the exact source asset. Parents must precede derivatives in a
    /// bundle so SQLite can enforce the immutable lineage foreign key.
    pub fn derived_from_asset(mut self, role: AssetRole, filename: impl Into<String>) -> Self {
        self.derived_from = Some(role);
        self.derived_from_filename = Some(filename.into());
        self
    }

    pub fn with_provenance(mut self, provenance: serde_json::Value) -> Self {
        self.provenance = provenance;
        self
    }

    pub fn with_original_filename(mut self, name: impl Into<String>) -> Self {
        self.original_filename = Some(name.into());
        self
    }

    fn is_canonical_source(&self) -> bool {
        self.role.is_native() || (self.role == AssetRole::SensorFits && self.derived_from.is_none())
    }
}

/// How a temperature reading was obtained. `22.4` from a probe and `22.4`
/// assumed from a config file are not the same measurement, and a drift model
/// that cannot tell them apart is learning from noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TemperatureState {
    /// Read from a sensor at capture time.
    Measured,
    /// Interpolated or carried forward from an earlier reading.
    Estimated,
    /// No reading available.
    #[default]
    Missing,
}

impl TemperatureState {
    fn as_str(self) -> &'static str {
        match self {
            TemperatureState::Measured => "measured",
            TemperatureState::Estimated => "estimated",
            TemperatureState::Missing => "missing",
        }
    }
}

/// Everything known about one exposure, ready to be committed.
#[derive(Debug, Clone)]
pub struct PendingCapture {
    pub capture_id: String,
    pub frame_id: String,
    pub session_id: Option<String>,
    pub rig_id: String,
    pub camera_name: Option<String>,
    pub driver_name: Option<String>,
    /// light | dark | flat | bias | focus | preview | verification
    pub capture_type: String,

    /// Authoritative exposure bracket from the driver, in Unix seconds.
    pub exposure_started_at: f64,
    pub exposure_completed_at: f64,

    pub requested_exposure_s: f64,
    pub requested_gain: Option<f64>,
    pub applied_exposure_s: f64,
    pub iso: Option<f64>,
    pub shutter_label: Option<String>,
    pub offset_val: Option<i64>,
    pub binning: i64,
    pub filter_name: Option<String>,
    pub focuser_position: Option<i64>,

    pub temperature: Option<f64>,
    pub temperature_source: Option<String>,
    pub temperature_state: TemperatureState,

    /// Camera-specific fields with no column of their own.
    pub acquisition: serde_json::Value,
    pub assets: Vec<PendingAsset>,
}

impl PendingCapture {
    fn archive_status(&self) -> &'static str {
        if self.acquisition["bundle"]["bundle_complete"] == false {
            "native_incomplete"
        } else {
            "committed"
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Output types
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommittedAsset {
    pub id: String,
    pub role: String,
    /// Per-role ordinal in this immutable bundle. Old manifests had one asset
    /// per role and therefore imply variant zero.
    #[serde(default)]
    pub variant: i64,
    pub media_type: String,
    /// Relative to the capture root, so the archive can be moved wholesale.
    pub relative_path: String,
    pub byte_length: i64,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived_from_asset_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CommittedCapture {
    pub capture_id: String,
    pub frame_id: String,
    pub session_id: Option<String>,
    pub status: &'static str,
    pub relative_dir: String,
    pub assets: Vec<CommittedAsset>,
}

/// `manifest.v1.json`, written beside the files so a bundle is self-describing
/// even if the database is lost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureManifest {
    pub manifest_version: String,
    pub capture_id: String,
    pub frame_id: String,
    pub session_id: Option<String>,
    pub rig_id: String,
    pub camera_name: Option<String>,
    pub capture_type: String,
    pub exposure_started_at: String,
    pub exposure_completed_at: String,
    pub requested_exposure_s: f64,
    pub applied_exposure_s: f64,
    pub iso: Option<f64>,
    pub shutter_label: Option<String>,
    pub software_version: String,
    pub acquisition: serde_json::Value,
    pub assets: Vec<CommittedAsset>,
}

// ═══════════════════════════════════════════════════════════════════
//  Store
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct CaptureStore {
    root: PathBuf,
}

impl CaptureStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Record that a capture is about to be attempted.
    ///
    /// Written before the camera is commanded so a hardware failure leaves a
    /// trace. Without it, "the capture silently did nothing" and "no capture
    /// was requested" look identical afterwards.
    pub async fn begin_attempt(
        &self,
        pool: &SqlitePool,
        capture_id: &str,
        session_id: Option<&str>,
        rig_id: &str,
        requested: &serde_json::Value,
    ) -> Result<()> {
        sqlx::query(
            r#"INSERT INTO capture_attempts
                   (capture_id, session_id, rig_id, requested_json, status, started_at)
               VALUES (?, ?, ?, ?, 'started', ?)"#,
        )
        .bind(capture_id)
        .bind(session_id)
        .bind(rig_id)
        .bind(requested.to_string())
        .bind(chrono::Utc::now().to_rfc3339())
        .execute(pool)
        .await
        .map_err(|e| StorageError::Database(format!("record capture attempt: {e}")))?;
        Ok(())
    }

    /// Mark an attempt failed. Best-effort: the capture already failed, and
    /// failing to record that must not mask the original error.
    pub async fn fail_attempt(&self, pool: &SqlitePool, capture_id: &str, error: &str) {
        let result = sqlx::query(
            r#"UPDATE capture_attempts
                  SET status = 'failed', error = ?, finished_at = ?
                WHERE capture_id = ?"#,
        )
        .bind(error)
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(capture_id)
        .execute(pool)
        .await;
        if let Err(e) = result {
            tracing::warn!("Could not record capture failure for {capture_id}: {e}");
        }
    }

    /// Persist one exposure: files first, then the catalogue, then the rename.
    ///
    /// Returns only after SQLite has committed. This is not authorization to
    /// delete camera originals; retention policy and archive durability/recovery
    /// qualification remain independent requirements.
    pub async fn persist_capture(
        &self,
        pool: &SqlitePool,
        capture: PendingCapture,
    ) -> Result<CommittedCapture> {
        if capture.assets.is_empty() {
            return Err(StorageError::Database(
                "Refusing to persist a capture with no assets".to_string(),
            )
            .into());
        }
        if !capture.assets.iter().any(PendingAsset::is_canonical_source) {
            // A bundle of purely derived products has no ground truth behind
            // it; something upstream lost the camera's own output.
            return Err(StorageError::Database(
                "Refusing to persist a capture with no native asset".to_string(),
            )
            .into());
        }
        let mut filenames = std::collections::BTreeSet::new();
        for asset in &capture.assets {
            let relative = (asset.role.subdir(), sanitize_filename(&asset.filename));
            if asset.bytes.is_empty() || !filenames.insert(relative) {
                return Err(StorageError::Database(
                    "Refusing empty assets or duplicate archive filenames".to_string(),
                )
                .into());
            }
        }
        let parents = resolve_asset_parents(&capture.assets)?;
        let archive_status = capture.archive_status();

        let started_at = unix_to_rfc3339(capture.exposure_started_at);
        let completed_at = unix_to_rfc3339(capture.exposure_completed_at);
        let night = night_of(capture.exposure_started_at);

        // <root>/<rig>/<night>/<session>/<frame-id>
        let relative_dir = PathBuf::from(sanitize(&capture.rig_id))
            .join(&night)
            .join(sanitize(
                capture.session_id.as_deref().unwrap_or("no-session"),
            ))
            .join(sanitize(&capture.frame_id));
        let final_dir = self.root.join(&relative_dir);
        let parent = final_dir
            .parent()
            .ok_or_else(|| StorageError::Database("capture path has no parent".into()))?
            .to_path_buf();
        let staging_dir = parent.join(format!(".{}.partial", sanitize(&capture.frame_id)));

        // ─── 1-3. Stage, hash, manifest, flush ───
        // Blocking file I/O; keep it off the async runtime's threads.
        let assets = capture.assets.clone();
        let relative_dir_for_io = relative_dir.clone();
        let staging_for_io = staging_dir.clone();
        let parent_for_io = parent.clone();
        let manifest_seed = ManifestSeed {
            capture_id: capture.capture_id.clone(),
            frame_id: capture.frame_id.clone(),
            session_id: capture.session_id.clone(),
            rig_id: capture.rig_id.clone(),
            camera_name: capture.camera_name.clone(),
            capture_type: capture.capture_type.clone(),
            exposure_started_at: started_at.clone(),
            exposure_completed_at: completed_at.clone(),
            requested_exposure_s: capture.requested_exposure_s,
            applied_exposure_s: capture.applied_exposure_s,
            iso: capture.iso,
            shutter_label: capture.shutter_label.clone(),
            acquisition: capture.acquisition.clone(),
        };

        let staged: Vec<CommittedAsset> = tokio::task::spawn_blocking(move || {
            stage_bundle(
                &parent_for_io,
                &staging_for_io,
                &relative_dir_for_io,
                &assets,
                &parents,
                manifest_seed,
            )
        })
        .await
        .map_err(|e| StorageError::Database(format!("staging task panicked: {e}")))??;

        // ─── 4. Catalogue, inside one transaction ───
        let commit = self
            .write_rows(
                pool,
                &capture,
                &staged,
                &relative_dir,
                &started_at,
                &completed_at,
            )
            .await;

        let tx = match commit {
            Ok(tx) => tx,
            Err(e) => {
                // Nothing was committed and nothing was renamed; drop the
                // staging directory so partials do not accumulate.
                let _ = std::fs::remove_dir_all(&staging_dir);
                return Err(e);
            }
        };

        // ─── 5. Rename into place, then 6. commit ───
        // If the rename fails the transaction is rolled back, so the database
        // never references a directory that is not there.
        if final_dir.exists() {
            // A retry of the same frame id. The bundle is immutable, so the
            // existing one wins and the fresh staging is discarded.
            let _ = std::fs::remove_dir_all(&staging_dir);
        } else if let Err(e) = std::fs::rename(&staging_dir, &final_dir) {
            let _ = tx.rollback().await;
            let _ = std::fs::remove_dir_all(&staging_dir);
            return Err(StorageError::Database(format!(
                "could not move staged capture into place: {e}"
            ))
            .into());
        }

        tx.commit()
            .await
            .map_err(|e| StorageError::Database(format!("commit capture: {e}")))?;

        tracing::info!(
            "Capture {} committed as frame {} ({} asset(s)) at {}",
            capture.capture_id,
            capture.frame_id,
            staged.len(),
            relative_dir.display()
        );

        Ok(CommittedCapture {
            capture_id: capture.capture_id,
            frame_id: capture.frame_id,
            session_id: capture.session_id,
            status: archive_status,
            relative_dir: relative_dir.to_string_lossy().into_owned(),
            assets: staged,
        })
    }

    /// Insert the catalogue rows and leave the transaction open for the caller
    /// to commit after the rename.
    #[allow(clippy::too_many_arguments)]
    async fn write_rows<'a>(
        &self,
        pool: &'a SqlitePool,
        capture: &PendingCapture,
        staged: &[CommittedAsset],
        relative_dir: &Path,
        started_at: &str,
        completed_at: &str,
    ) -> Result<sqlx::Transaction<'a, sqlx::Sqlite>> {
        let mut tx = pool
            .begin()
            .await
            .map_err(|e| StorageError::Database(format!("begin capture transaction: {e}")))?;

        // `frames.local_path` / `checksum_sha256` are legacy aliases of the
        // preferred analysis asset. Dual-written until
        // build_focus_quality_dataset.py joins frame_assets instead.
        let legacy = staged
            .iter()
            .find(|a| a.role == AssetRole::AnalysisLuma.as_str())
            .or_else(|| {
                staged
                    .iter()
                    .find(|a| a.role == AssetRole::SensorFits.as_str())
            })
            .or_else(|| staged.first());
        let legacy_path = legacy.map(|a| a.relative_path.clone()).unwrap_or_default();
        let legacy_sum = legacy.map(|a| a.sha256.clone());
        let primary_dims = capture
            .assets
            .iter()
            .find(|a| a.width.is_some())
            .map(|a| (a.width, a.height))
            .unwrap_or((None, None));

        sqlx::query(
            r#"INSERT INTO frames (
                   id, session_id, local_path, checksum_sha256, width, height,
                   exposure_s, gain, filter_name, temperature, quality_json, captured_at, night_id,
                   capture_id, rig_id, camera_name, capture_type, capture_status,
                   exposure_started_at, exposure_completed_at,
                   requested_exposure_s, requested_gain, iso, shutter_label,
                   offset_val, binning, focuser_position,
                   temperature_source, temperature_state,
                   software_version, driver_name, acquisition_json
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                         ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(&capture.frame_id)
        .bind(&capture.session_id)
        .bind(&legacy_path)
        .bind(&legacy_sum)
        .bind(primary_dims.0)
        .bind(primary_dims.1)
        .bind(capture.applied_exposure_s)
        .bind(capture.iso.or(capture.requested_gain))
        .bind(&capture.filter_name)
        .bind(capture.temperature)
        // Keep the immutable observation and its source bundle in the same
        // transaction; focus must not commit a frame then patch its label.
        .bind(
            capture
                .acquisition
                .get("quality")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}))
                .to_string(),
        )
        .bind(started_at)
        .bind(crate::storage::night::night_id(unix_to_datetime(
            capture.exposure_started_at,
        )))
        .bind(&capture.capture_id)
        .bind(&capture.rig_id)
        .bind(&capture.camera_name)
        .bind(&capture.capture_type)
        .bind(capture.archive_status())
        .bind(started_at)
        .bind(completed_at)
        .bind(capture.requested_exposure_s)
        .bind(capture.requested_gain)
        .bind(capture.iso)
        .bind(&capture.shutter_label)
        .bind(capture.offset_val)
        .bind(capture.binning)
        .bind(capture.focuser_position)
        .bind(&capture.temperature_source)
        .bind(capture.temperature_state.as_str())
        .bind(env!("CARGO_PKG_VERSION"))
        .bind(&capture.driver_name)
        .bind(capture.acquisition.to_string())
        .execute(&mut *tx)
        .await
        .map_err(|e| StorageError::Database(format!("insert frame: {e}")))?;

        for (asset, staged_asset) in capture.assets.iter().zip(staged.iter()) {
            sqlx::query(
                r#"INSERT INTO frame_assets (
                       id, frame_id, role, variant, original_filename, relative_path,
                       media_type, byte_length, checksum_sha256, width, height, bit_depth,
                       derived_from_asset_id, provenance_json, upload_policy, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
            )
            .bind(&staged_asset.id)
            .bind(&capture.frame_id)
            .bind(&staged_asset.role)
            .bind(staged_asset.variant)
            .bind(&asset.original_filename)
            .bind(&staged_asset.relative_path)
            .bind(&staged_asset.media_type)
            .bind(staged_asset.byte_length)
            .bind(&staged_asset.sha256)
            .bind(asset.width)
            .bind(asset.height)
            .bind(asset.bit_depth)
            .bind(&staged_asset.derived_from_asset_id)
            .bind(asset.provenance.to_string())
            .bind(if asset.role.is_native() {
                "metadata_only"
            } else {
                "upload"
            })
            .bind(chrono::Utc::now().to_rfc3339())
            .execute(&mut *tx)
            .await
            .map_err(|e| StorageError::Database(format!("insert frame asset: {e}")))?;
        }

        // Outbox entry so cloud sync has something to pick up without polling
        // the whole table.
        sqlx::query(
            r#"INSERT INTO sync_queue (entity_type, entity_id, operation, payload_json)
               VALUES ('frame', ?, 'create', ?)"#,
        )
        .bind(&capture.frame_id)
        .bind(
            serde_json::json!({
                "capture_id": capture.capture_id,
                "session_id": capture.session_id,
                "relative_dir": relative_dir.to_string_lossy(),
                "assets": staged,
            })
            .to_string(),
        )
        .execute(&mut *tx)
        .await
        .map_err(|e| StorageError::Database(format!("enqueue frame sync: {e}")))?;

        sqlx::query(
            r#"UPDATE capture_attempts
                  SET status = ?, frame_id = ?, finished_at = ?
                WHERE capture_id = ?"#,
        )
        .bind(capture.archive_status())
        .bind(&capture.frame_id)
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(&capture.capture_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| StorageError::Database(format!("close capture attempt: {e}")))?;

        Ok(tx)
    }

    /// Startup reconciliation.
    ///
    /// Crashes leave three kinds of debris, and silently ignoring any of them
    /// turns the archive into something you cannot trust:
    ///
    /// * `.partial` directories — an interrupted stage. Removed.
    /// * committed rows whose files are missing or whose recorded byte length
    ///   no longer matches — marked `corrupt` rather than left looking healthy.
    /// * `started` attempts with no frame — marked `failed`.
    ///
    /// Returns a human-readable summary for the startup log.
    pub async fn reconcile(&self, pool: &SqlitePool) -> Result<ReconcileReport> {
        let mut report = ReconcileReport::default();

        if self.root.exists() {
            let root = self.root.clone();
            let removed = tokio::task::spawn_blocking(move || remove_partials(&root))
                .await
                .map_err(|e| StorageError::Database(format!("reconcile task panicked: {e}")))?;
            report.partials_removed = removed;
        }

        // Check existence and length. Content-hash verification and recovery
        // of uncatalogued final bundles remain A12 qualification work.
        let rows: Vec<(String, String, String, i64)> = sqlx::query_as(
            r#"SELECT fa.frame_id, fa.id, fa.relative_path, fa.byte_length
                 FROM frame_assets fa
                 JOIN frames f ON f.id = fa.frame_id
                WHERE f.capture_status IN ('committed', 'native_incomplete')"#,
        )
        .fetch_all(pool)
        .await
        .map_err(|e| StorageError::Database(format!("list committed assets: {e}")))?;

        let mut bad_frames: Vec<String> = Vec::new();
        for (frame_id, _asset_id, rel, expected_len) in rows {
            let path = self.root.join(&rel);
            match std::fs::metadata(&path) {
                Ok(meta) if meta.len() as i64 == expected_len => {}
                Ok(meta) => {
                    tracing::warn!(
                        "Asset {} is {} bytes, expected {}",
                        rel,
                        meta.len(),
                        expected_len
                    );
                    bad_frames.push(frame_id);
                }
                Err(_) => {
                    tracing::warn!("Asset {} referenced by the database is missing", rel);
                    bad_frames.push(frame_id);
                }
            }
        }
        bad_frames.sort();
        bad_frames.dedup();
        for frame_id in &bad_frames {
            sqlx::query("UPDATE frames SET capture_status = 'corrupt' WHERE id = ?")
                .bind(frame_id)
                .execute(pool)
                .await
                .map_err(|e| StorageError::Database(format!("flag corrupt frame: {e}")))?;
        }
        report.frames_marked_corrupt = bad_frames.len();

        let stranded = sqlx::query(
            r#"UPDATE capture_attempts
                  SET status = 'failed',
                      error = COALESCE(error, 'daemon exited before the capture completed'),
                      finished_at = ?
                WHERE status = 'started'"#,
        )
        .bind(chrono::Utc::now().to_rfc3339())
        .execute(pool)
        .await
        .map_err(|e| StorageError::Database(format!("close stranded attempts: {e}")))?;
        report.attempts_failed = stranded.rows_affected() as usize;

        Ok(report)
    }
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct ReconcileReport {
    pub partials_removed: usize,
    pub frames_marked_corrupt: usize,
    pub attempts_failed: usize,
}

impl ReconcileReport {
    pub fn is_clean(&self) -> bool {
        self.partials_removed == 0 && self.frames_marked_corrupt == 0 && self.attempts_failed == 0
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Filesystem staging
// ═══════════════════════════════════════════════════════════════════

struct ManifestSeed {
    capture_id: String,
    frame_id: String,
    session_id: Option<String>,
    rig_id: String,
    camera_name: Option<String>,
    capture_type: String,
    exposure_started_at: String,
    exposure_completed_at: String,
    requested_exposure_s: f64,
    applied_exposure_s: f64,
    iso: Option<f64>,
    shutter_label: Option<String>,
    acquisition: serde_json::Value,
}

/// Resolve immutable lineage before touching the filesystem. Legacy callers
/// may identify a sole parent by role; repeated roles require its filename.
/// Parents precede derivatives so both the manifest and immediate SQLite
/// foreign-key checks describe the same already-present source.
fn resolve_asset_parents(assets: &[PendingAsset]) -> Result<Vec<Option<usize>>> {
    assets
        .iter()
        .enumerate()
        .map(|(index, asset)| {
            let Some(role) = asset.derived_from else {
                if asset.derived_from_filename.is_some() {
                    return Err(StorageError::Database(
                        "An asset parent filename requires its role".to_string(),
                    )
                    .into());
                }
                return Ok(None);
            };
            let candidates: Vec<usize> = assets
                .iter()
                .enumerate()
                .filter(|(_, parent)| {
                    parent.role == role
                        && asset
                            .derived_from_filename
                            .as_ref()
                            .is_none_or(|name| parent.filename == *name)
                })
                .map(|(parent, _)| parent)
                .collect();
            if candidates.len() != 1 {
                return Err(StorageError::Database(format!(
                    "Asset {} has missing or ambiguous {} parent; identify its exact filename",
                    asset.filename,
                    role.as_str()
                ))
                .into());
            }
            let parent = candidates[0];
            if parent >= index {
                return Err(StorageError::Database(format!(
                    "Parent asset must precede derivative {} in the capture bundle",
                    asset.filename
                ))
                .into());
            }
            Ok(Some(parent))
        })
        .collect()
}

/// Write every asset and its manifest, hash and sync file contents, and attempt
/// staging-directory synchronization. Entries retain the input asset order.
fn stage_bundle(
    parent: &Path,
    staging_dir: &Path,
    relative_dir: &Path,
    assets: &[PendingAsset],
    parents: &[Option<usize>],
    seed: ManifestSeed,
) -> Result<Vec<CommittedAsset>> {
    std::fs::create_dir_all(parent)?;
    // A leftover from a previous crashed attempt at the same id.
    if staging_dir.exists() {
        std::fs::remove_dir_all(staging_dir)?;
    }
    std::fs::create_dir_all(staging_dir)?;

    let mut committed: Vec<CommittedAsset> = Vec::with_capacity(assets.len());
    let mut variants: BTreeMap<&str, i64> = BTreeMap::new();

    for (index, asset) in assets.iter().enumerate() {
        let next_variant = variants.entry(asset.role.as_str()).or_default();
        let variant = *next_variant;
        *next_variant += 1;

        let subdir = staging_dir.join(asset.role.subdir());
        std::fs::create_dir_all(&subdir)?;
        let filename = sanitize_filename(&asset.filename);
        let path = subdir.join(&filename);

        // Hash the bytes we actually write, not the buffer we intended to.
        let mut hasher = Sha256::new();
        {
            let mut file = std::fs::File::create(&path)?;
            file.write_all(&asset.bytes)?;
            hasher.update(&asset.bytes);
            file.sync_all()?;
        }

        committed.push(CommittedAsset {
            id: uuid::Uuid::new_v4().to_string(),
            role: asset.role.as_str().to_string(),
            variant,
            media_type: asset.media_type.clone(),
            relative_path: relative_dir
                .join(asset.role.subdir())
                .join(&filename)
                .to_string_lossy()
                .into_owned(),
            byte_length: asset.bytes.len() as i64,
            sha256: format!("{:x}", hasher.finalize()),
            derived_from_asset_id: parents[index].map(|parent| committed[parent].id.clone()),
        });
    }

    let manifest = CaptureManifest {
        manifest_version: MANIFEST_VERSION.to_string(),
        capture_id: seed.capture_id,
        frame_id: seed.frame_id,
        session_id: seed.session_id,
        rig_id: seed.rig_id,
        camera_name: seed.camera_name,
        capture_type: seed.capture_type,
        exposure_started_at: seed.exposure_started_at,
        exposure_completed_at: seed.exposure_completed_at,
        requested_exposure_s: seed.requested_exposure_s,
        applied_exposure_s: seed.applied_exposure_s,
        iso: seed.iso,
        shutter_label: seed.shutter_label,
        software_version: env!("CARGO_PKG_VERSION").to_string(),
        acquisition: seed.acquisition,
        assets: committed.clone(),
    };
    let manifest_path = staging_dir.join(MANIFEST_FILENAME);
    {
        let mut file = std::fs::File::create(&manifest_path)?;
        file.write_all(serde_json::to_string_pretty(&manifest)?.as_bytes())?;
        file.sync_all()?;
    }

    // Best-effort staging-directory synchronization. This deliberately does
    // not claim a durability receipt: checked subdirectory/rename-parent sync
    // and platform failure qualification remain A12 work.
    if let Ok(dir) = std::fs::File::open(staging_dir) {
        let _ = dir.sync_all();
    }

    Ok(committed)
}

fn remove_partials(root: &Path) -> usize {
    let mut removed = 0;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') && name.ends_with(".partial") {
                tracing::warn!("Removing interrupted capture staging directory {:?}", path);
                if std::fs::remove_dir_all(&path).is_ok() {
                    removed += 1;
                }
            } else {
                stack.push(path);
            }
        }
    }
    removed
}

// ═══════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════

/// Observing night under the same local-noon boundary used by every dataset
/// grouping key. A directory and its catalogue row must never disagree.
fn night_of(unix_seconds: f64) -> String {
    crate::storage::night::night_id(unix_to_datetime(unix_seconds))
}

fn unix_to_datetime(unix_seconds: f64) -> chrono::DateTime<chrono::Utc> {
    let secs = unix_seconds.trunc() as i64;
    let nanos = ((unix_seconds - unix_seconds.trunc()) * 1e9).round() as u32;
    chrono::DateTime::from_timestamp(secs, nanos.min(999_999_999)).unwrap_or_else(chrono::Utc::now)
}

fn unix_to_rfc3339(unix_seconds: f64) -> String {
    unix_to_datetime(unix_seconds).to_rfc3339()
}

/// Keep path components to characters that cannot escape the capture root or
/// confuse a shell. Identifiers here are UUIDs and rig names, so this is a
/// guard against a malformed value rather than an expected transformation.
fn sanitize(component: &str) -> String {
    let cleaned: String = component
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "unknown".to_string()
    } else {
        cleaned
    }
}

fn sanitize_filename(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    // Reject traversal outright rather than trying to repair it.
    if cleaned.is_empty() || cleaned == "." || cleaned == ".." || cleaned.starts_with("..") {
        "asset.bin".to_string()
    } else {
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_pool() -> SqlitePool {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .connect("sqlite::memory:")
            .await
            .unwrap();
        for (_, _, sql) in crate::storage::local_store::migrations_for_test() {
            for statement in crate::storage::local_store::split_statements_for_test(sql) {
                let _ = sqlx::query(&statement).execute(&pool).await;
            }
        }
        sqlx::query("INSERT INTO sessions (id, rig_id) VALUES ('sess-1', 'rig-1')")
            .execute(&pool)
            .await
            .unwrap();
        pool
    }

    fn sample(frame_id: &str) -> PendingCapture {
        PendingCapture {
            capture_id: format!("cap-{frame_id}"),
            frame_id: frame_id.to_string(),
            session_id: Some("sess-1".into()),
            rig_id: "rig-1".into(),
            camera_name: Some("Canon EOS R50".into()),
            driver_name: Some("gphoto2".into()),
            capture_type: "light".into(),
            // 2026-08-02T02:00:00Z — after midnight, so the night must fold
            // back to 2026-08-01.
            exposure_started_at: 1_785_636_000.0,
            exposure_completed_at: 1_785_636_004.0,
            requested_exposure_s: 3.5,
            requested_gain: Some(800.0),
            applied_exposure_s: 4.0,
            iso: Some(800.0),
            shutter_label: Some("4".into()),
            offset_val: None,
            binning: 1,
            filter_name: None,
            focuser_position: Some(12_345),
            temperature: Some(11.5),
            temperature_source: Some("focuser_probe".into()),
            temperature_state: TemperatureState::Measured,
            acquisition: serde_json::json!({"capturetarget": "sdram"}),
            assets: vec![
                PendingAsset::new(
                    AssetRole::NativeRaw,
                    "image.cr3",
                    "image/x-canon-cr3",
                    b"RAWBYTES".to_vec(),
                )
                .with_original_filename("IMG_0042.CR3"),
                PendingAsset::new(
                    AssetRole::SensorFits,
                    "image.fits",
                    "application/fits",
                    b"SIMPLE  =                    T".to_vec(),
                )
                .derived_from(AssetRole::NativeRaw)
                .with_dimensions(6000, 4000, 16),
            ],
        }
    }

    #[tokio::test]
    async fn incomplete_native_bytes_commit_with_ineligible_quality_and_explicit_status() {
        let dir =
            std::env::temp_dir().join(format!("achernar-incomplete-{}", uuid::Uuid::new_v4()));
        let store = CaptureStore::new(&dir);
        let pool = test_pool().await;
        let mut capture = sample("partial-frame");
        capture.assets.truncate(1);
        capture.acquisition["bundle"] = serde_json::json!({
            "bundle_complete": false, "event_complete": true,
            "expected_asset_count": null, "discovered_paths": ["IMG_0042.CR3", "IMG_0042.JPG"],
            "downloaded_count": 1, "missing_count": 1,
        });
        capture.acquisition["quality"] = serde_json::json!({"training_label_eligible": false});
        store
            .begin_attempt(
                &pool,
                &capture.capture_id,
                Some("sess-1"),
                "rig-1",
                &serde_json::json!({}),
            )
            .await
            .unwrap();
        let committed = store.persist_capture(&pool, capture).await.unwrap();
        assert_eq!(committed.status, "native_incomplete");
        assert_eq!(
            std::fs::read(dir.join(&committed.assets[0].relative_path)).unwrap(),
            b"RAWBYTES"
        );
        let (status, quality): (String, String) =
            sqlx::query_as("SELECT capture_status, quality_json FROM frames WHERE id = ?")
                .bind(&committed.frame_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(status, "native_incomplete");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&quality).unwrap()["training_label_eligible"],
            false
        );
        let (attempt_status,): (String,) =
            sqlx::query_as("SELECT status FROM capture_attempts WHERE capture_id = ?")
                .bind(&committed.capture_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(attempt_status, "native_incomplete");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn duplicate_native_filenames_cannot_silently_overwrite_downloads() {
        let dir = std::env::temp_dir().join(format!("achernar-duplicate-{}", uuid::Uuid::new_v4()));
        let store = CaptureStore::new(&dir);
        let pool = test_pool().await;
        let mut capture = sample("duplicate-frame");
        capture.assets.push(capture.assets[0].clone());
        let error = store.persist_capture(&pool, capture).await.unwrap_err();
        assert!(error.to_string().contains("duplicate archive filenames"));
        assert!(!dir.exists());
        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM frames")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn repeated_native_roles_preserve_every_file_and_exact_derivation_parent() {
        let dir = std::env::temp_dir().join(format!("achernar-variants-{}", uuid::Uuid::new_v4()));
        let store = CaptureStore::new(&dir);
        let pool = test_pool().await;
        let mut capture = sample("multiple-native-files");
        capture.assets = vec![
            PendingAsset::new(
                AssetRole::NativeRaw,
                "image-0.cr3",
                "image/x-canon-cr3",
                b"FIRST RAW".to_vec(),
            ),
            PendingAsset::new(
                AssetRole::NativeRaw,
                "image-1.cr3",
                "image/x-canon-cr3",
                b"SECOND RAW".to_vec(),
            ),
            PendingAsset::new(
                AssetRole::CameraJpeg,
                "image-2.jpg",
                "image/jpeg",
                b"LARGE JPEG".to_vec(),
            ),
            PendingAsset::new(
                AssetRole::CameraJpeg,
                "image-3.jpg",
                "image/jpeg",
                b"SMALL JPEG".to_vec(),
            ),
            PendingAsset::new(
                AssetRole::SensorFits,
                "image.fits",
                "application/fits",
                b"FITS FROM FIRST RAW".to_vec(),
            )
            .derived_from_asset(AssetRole::NativeRaw, "image-0.cr3"),
        ];
        let expected = capture.assets.clone();
        let committed = store.persist_capture(&pool, capture).await.unwrap();
        assert_eq!(committed.assets.len(), 5);
        assert_eq!(
            committed
                .assets
                .iter()
                .map(|asset| asset.variant)
                .collect::<Vec<_>>(),
            vec![0, 1, 0, 1, 0],
        );
        let rows: Vec<(String, i64, String, String, Option<String>)> = sqlx::query_as(
            "SELECT role, variant, relative_path, checksum_sha256, derived_from_asset_id FROM frame_assets WHERE frame_id = ? ORDER BY role, variant",
        )
        .bind(&committed.frame_id)
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(rows.len(), 5);
        for (asset, original) in committed.assets.iter().zip(&expected) {
            assert_eq!(
                std::fs::read(dir.join(&asset.relative_path)).unwrap(),
                original.bytes
            );
            assert_eq!(
                asset.sha256,
                format!("{:x}", Sha256::digest(&original.bytes))
            );
            let row = rows
                .iter()
                .find(|row| row.0 == asset.role && row.1 == asset.variant)
                .unwrap();
            assert_eq!(row.2, asset.relative_path);
            assert_eq!(row.3, asset.sha256);
            assert_eq!(row.4, asset.derived_from_asset_id);
        }
        assert_eq!(
            committed.assets[4].derived_from_asset_id.as_deref(),
            Some(committed.assets[0].id.as_str())
        );
        assert_ne!(
            committed.assets[4].derived_from_asset_id.as_deref(),
            Some(committed.assets[1].id.as_str())
        );
        let manifest: CaptureManifest = serde_json::from_slice(
            &std::fs::read(dir.join(&committed.relative_dir).join(MANIFEST_FILENAME)).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest.assets[1].variant, 1);
        assert_eq!(
            manifest.assets[4].derived_from_asset_id,
            committed.assets[4].derived_from_asset_id
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn ambiguous_or_missing_derivation_parent_is_rejected_before_writing() {
        let dir = std::env::temp_dir().join(format!("achernar-lineage-{}", uuid::Uuid::new_v4()));
        let store = CaptureStore::new(&dir);
        let pool = test_pool().await;
        let mut capture = sample("ambiguous-parent");
        capture.assets.insert(
            1,
            PendingAsset::new(
                AssetRole::NativeRaw,
                "another.cr3",
                "image/x-canon-cr3",
                b"OTHER RAW".to_vec(),
            ),
        );
        // The old role-only declaration is safe only for one source per role.
        let error = store
            .persist_capture(&pool, capture.clone())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("ambiguous"), "{error}");
        capture.assets[2] = capture.assets[2]
            .clone()
            .derived_from_asset(AssetRole::NativeRaw, "missing.cr3");
        let error = store.persist_capture(&pool, capture).await.unwrap_err();
        assert!(error.to_string().contains("missing"), "{error}");
        assert!(
            !dir.exists(),
            "invalid lineage must fail before filesystem writes"
        );
        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM frame_assets")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn retention_evidence_survives_attempt_catalogue_and_manifest() {
        use crate::storage::capture_retention::RetentionDecision;

        let dir = std::env::temp_dir().join(format!("achernar-retention-{}", uuid::Uuid::new_v4()));
        let store = CaptureStore::new(&dir);
        let pool = test_pool().await;
        let retention = serde_json::to_value(RetentionDecision::retaining(true)).unwrap();
        let mut capture = sample("retained-frame");
        capture.acquisition["retention"] = retention.clone();
        let requested = serde_json::json!({"retention": retention});
        store
            .begin_attempt(
                &pool,
                &capture.capture_id,
                Some("sess-1"),
                "rig-1",
                &requested,
            )
            .await
            .unwrap();
        let committed = store.persist_capture(&pool, capture).await.unwrap();

        let (attempt_json,): (String,) =
            sqlx::query_as("SELECT requested_json FROM capture_attempts WHERE capture_id = ?")
                .bind(&committed.capture_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        let (acquisition_json,): (String,) =
            sqlx::query_as("SELECT acquisition_json FROM frames WHERE id = ?")
                .bind(&committed.frame_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        let manifest: CaptureManifest = serde_json::from_slice(
            &std::fs::read(dir.join(&committed.relative_dir).join(MANIFEST_FILENAME)).unwrap(),
        )
        .unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&attempt_json).unwrap()["retention"],
            requested["retention"]
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&acquisition_json).unwrap()["retention"],
            requested["retention"]
        );
        assert_eq!(manifest.acquisition["retention"], requested["retention"]);
        assert_eq!(
            manifest.acquisition["retention"]["deletion_eligible"],
            false
        );
        assert_eq!(
            manifest.acquisition["retention"]["camera_copy_state"],
            "unknown"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn persists_bundle_and_catalogue() {
        let dir = std::env::temp_dir().join(format!("achernar-cap-{}", uuid::Uuid::new_v4()));
        let store = CaptureStore::new(&dir);
        let pool = test_pool().await;

        let capture = sample("frame-1");
        store
            .begin_attempt(
                &pool,
                &capture.capture_id,
                Some("sess-1"),
                "rig-1",
                &serde_json::json!({}),
            )
            .await
            .unwrap();
        let committed = store.persist_capture(&pool, capture).await.unwrap();

        assert_eq!(committed.assets.len(), 2);
        // Night folds back across midnight.
        assert!(
            committed.relative_dir.contains("2026-08-01"),
            "expected the pre-midnight night, got {}",
            committed.relative_dir
        );

        for asset in &committed.assets {
            let path = dir.join(&asset.relative_path);
            assert!(path.exists(), "missing {}", asset.relative_path);
            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(bytes.len() as i64, asset.byte_length);
            assert_eq!(format!("{:x}", Sha256::digest(&bytes)), asset.sha256);
        }

        let manifest_path = dir.join(&committed.relative_dir).join(MANIFEST_FILENAME);
        let manifest: CaptureManifest =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        assert_eq!(manifest.applied_exposure_s, 4.0);
        assert_eq!(manifest.requested_exposure_s, 3.5);

        let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM frame_assets WHERE frame_id = ?")
            .bind("frame-1")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count.0, 2);

        // The applied exposure, not the requested one, is what is recorded.
        let row: (f64, f64, String) = sqlx::query_as(
            "SELECT exposure_s, requested_exposure_s, temperature_state FROM frames WHERE id = ?",
        )
        .bind("frame-1")
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.0, 4.0);
        assert_eq!(row.1, 3.5);
        assert_eq!(row.2, "measured");

        let attempt: (String,) =
            sqlx::query_as("SELECT status FROM capture_attempts WHERE capture_id = ?")
                .bind("cap-frame-1")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(attempt.0, "committed");

        let outbox: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM sync_queue WHERE entity_id = ?")
            .bind("frame-1")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(outbox.0, 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn rejects_capture_without_native_asset() {
        let dir = std::env::temp_dir().join(format!("achernar-cap-{}", uuid::Uuid::new_v4()));
        let store = CaptureStore::new(&dir);
        let pool = test_pool().await;

        let mut capture = sample("frame-2");
        capture.assets.retain(|a| !a.role.is_native());
        let err = store.persist_capture(&pool, capture).await.unwrap_err();
        assert!(err.to_string().contains("no native asset"), "{err}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn reconcile_removes_partials_and_flags_missing_files() {
        let dir = std::env::temp_dir().join(format!("achernar-cap-{}", uuid::Uuid::new_v4()));
        let store = CaptureStore::new(&dir);
        let pool = test_pool().await;

        let capture = sample("frame-3");
        store
            .begin_attempt(
                &pool,
                &capture.capture_id,
                Some("sess-1"),
                "rig-1",
                &serde_json::json!({}),
            )
            .await
            .unwrap();
        let committed = store.persist_capture(&pool, capture).await.unwrap();

        // Simulate a crash mid-stage plus a stranded attempt.
        let partial = dir.join("rig-1/2026-08-01/sess-1/.frame-99.partial");
        std::fs::create_dir_all(&partial).unwrap();
        store
            .begin_attempt(&pool, "cap-stranded", None, "rig-1", &serde_json::json!({}))
            .await
            .unwrap();

        // And a file that vanished from under a committed row.
        std::fs::remove_file(dir.join(&committed.assets[0].relative_path)).unwrap();

        let report = store.reconcile(&pool).await.unwrap();
        assert_eq!(report.partials_removed, 1);
        assert_eq!(report.frames_marked_corrupt, 1);
        assert_eq!(report.attempts_failed, 1);
        assert!(!partial.exists());

        let status: (String,) = sqlx::query_as("SELECT capture_status FROM frames WHERE id = ?")
            .bind("frame-3")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(status.0, "corrupt");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn filenames_cannot_escape_the_bundle() {
        assert_eq!(sanitize_filename("../../etc/passwd"), "asset.bin");
        assert_eq!(sanitize_filename("image.cr3"), "image.cr3");
        assert_eq!(sanitize("rig/../.."), "rig______");
    }

    #[test]
    fn night_folds_back_across_midnight() {
        // 2026-08-02T02:00Z belongs to the previous evening's session, so a
        // night that runs past midnight stays in one directory.
        assert_eq!(night_of(1_785_636_000.0), "2026-08-01");
        // 20:00Z the same day starts the next night.
        assert_eq!(night_of(1_785_636_000.0 + 18.0 * 3600.0), "2026-08-02");
    }
}
