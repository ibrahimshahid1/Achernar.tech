//! frames_repo — persists preserved frames for provenance (Sections 5.2, 8.4).
//!
//! The shared vision bus intentionally hides raw pixels from controllers, but
//! the ML training pipeline (FQE, TED, COC) needs the exact frames that produced
//! each label. This store writes every preserved frame as a lossless 16-bit
//! grayscale PNG and registers a row in the `frames` table so that
//! `focus_samples.frame_id` can reference it. Without this, autofocus runs
//! discard their pixels and no FQE dataset can ever be built.

use std::path::{Path, PathBuf};

use image::{ImageBuffer, ImageFormat, Luma};
use sha2::{Digest, Sha256};
use sqlx::SqlitePool;

use crate::error::StorageError;
use crate::vision::quality::LumaFrame;

/// Metadata recorded alongside a persisted frame.
#[derive(Debug, Clone, Default)]
pub struct FrameMeta {
    pub session_id: Option<String>,
    pub filter_name: Option<String>,
    pub temperature: Option<f64>,
    pub exposure_s: Option<f64>,
    pub gain: Option<f64>,
    pub captured_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Free-form JSON describing how the frame was captured and measured
    /// (focuser position, HFR, star count, `max_adu`, …). Stored verbatim in
    /// `frames.quality_json` so the dataset builder can recover labels.
    pub quality_json: serde_json::Value,
}

/// A frame that has been written to disk and registered in the `frames` table.
#[derive(Debug, Clone)]
pub struct PersistedFrame {
    pub id: String,
    pub local_path: String,
}

/// Writes preserved frames to `frames_dir` and registers them in SQLite.
#[derive(Debug, Clone)]
pub struct FrameStore {
    frames_dir: PathBuf,
}

impl FrameStore {
    pub fn new(frames_dir: impl Into<PathBuf>) -> Self {
        Self {
            frames_dir: frames_dir.into(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.frames_dir
    }

    /// Encode `frame` as a lossless 16-bit grayscale PNG and insert a `frames`
    /// row. Returns the generated frame id (a UUID, which is also the file stem).
    pub async fn persist_luma(
        &self,
        pool: &SqlitePool,
        frame: &LumaFrame,
        meta: FrameMeta,
    ) -> crate::error::Result<PersistedFrame> {
        let expected = frame.width.checked_mul(frame.height).unwrap_or(0);
        if frame.width == 0 || frame.height == 0 || expected == 0 || frame.pixels.len() < expected {
            return Err(StorageError::Database(
                "Refusing to persist a frame with invalid dimensions".to_string(),
            )
            .into());
        }

        let id = uuid::Uuid::new_v4().to_string();
        let captured_at = meta.captured_at.unwrap_or_else(chrono::Utc::now);

        // Shard files by capture date so a single directory never grows unbounded.
        let day_dir = self
            .frames_dir
            .join(captured_at.format("%Y-%m-%d").to_string());
        std::fs::create_dir_all(&day_dir)?;
        let path = day_dir.join(format!("{id}.png"));

        let width = frame.width as u32;
        let height = frame.height as u32;
        let pixels: Vec<u16> = frame.pixels[..expected]
            .iter()
            .map(|value| {
                if value.is_finite() {
                    value.round().clamp(0.0, u16::MAX as f32) as u16
                } else {
                    0
                }
            })
            .collect();
        let buffer: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::from_raw(width, height, pixels)
            .ok_or_else(|| {
                StorageError::Database("Frame pixel buffer did not match dimensions".to_string())
            })?;
        buffer
            .save_with_format(&path, ImageFormat::Png)
            .map_err(|err| StorageError::Database(format!("write frame PNG: {err}")))?;

        let local_path = path.to_string_lossy().into_owned();
        let encoded = std::fs::read(&path)?;
        let checksum_sha256 = format!("{:x}", Sha256::digest(&encoded));
        let quality_json =
            serde_json::to_string(&meta.quality_json).unwrap_or_else(|_| "{}".to_string());

        // Derived here rather than asked of the caller: every writer would
        // compute it the same way, and one that forgot would produce a frame
        // that silently drops out of any night-grouped split.
        let night_id = crate::storage::night::night_id(captured_at);

        let insert_result = sqlx::query(
            r#"
            INSERT INTO frames (
                id, session_id, local_path, checksum_sha256, width, height, exposure_s, gain,
                filter_name, temperature, quality_json, captured_at, night_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&id)
        .bind(&meta.session_id)
        .bind(&local_path)
        .bind(&checksum_sha256)
        .bind(width as i64)
        .bind(height as i64)
        .bind(meta.exposure_s)
        .bind(meta.gain)
        .bind(&meta.filter_name)
        .bind(meta.temperature)
        .bind(&quality_json)
        .bind(captured_at.to_rfc3339())
        .bind(&night_id)
        .execute(pool)
        .await;
        if let Err(err) = insert_result {
            // Do not accumulate unindexed frame files when the database write
            // fails; callers can safely retry persistence on the next exposure.
            std::fs::remove_file(&path).ok();
            return Err(StorageError::Database(format!("insert frame row: {err}")).into());
        }

        Ok(PersistedFrame { id, local_path })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn persists_frame_file_and_row() {
        let dir = std::env::temp_dir().join(format!("achernar-frames-{}", uuid::Uuid::new_v4()));
        let store = FrameStore::new(&dir);

        // Build the fixture from the real migrations. This test used to declare
        // its own `frames` table, which is the drift `migrations_for_test`
        // exists to prevent — it went stale the moment migration 007 added
        // `night_id`, and the insert would have failed against a table that no
        // deployed database has looked like since migration 004.
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .connect("sqlite::memory:")
            .await
            .unwrap();
        for (_, _, sql) in crate::storage::local_store::migrations_for_test() {
            for statement in crate::storage::local_store::split_statements_for_test(sql) {
                let _ = sqlx::query(&statement).execute(&pool).await;
            }
        }

        let frame = LumaFrame {
            pixels: vec![1000.0_f32; 8 * 8],
            width: 8,
            height: 8,
            max_adu: 65535.0,
            saturation_adu: Some(65535.0),
        };
        let meta = FrameMeta {
            quality_json: serde_json::json!({"focuser_position": 100, "hfr": 2.5}),
            ..Default::default()
        };

        let persisted = store.persist_luma(&pool, &frame, meta).await.unwrap();
        assert!(std::path::Path::new(&persisted.local_path).exists());

        let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM frames WHERE id = ?")
            .bind(&persisted.id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count.0, 1);
        let checksum: (String,) = sqlx::query_as("SELECT checksum_sha256 FROM frames WHERE id = ?")
            .bind(&persisted.id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(checksum.0.len(), 64);

        // Every frame carries the night it was taken on, so a night-grouped
        // split can see it. A NULL here silently excludes the frame from the
        // split unit the cloud and background models are evaluated on.
        let night: (Option<String>,) = sqlx::query_as("SELECT night_id FROM frames WHERE id = ?")
            .bind(&persisted.id)
            .fetch_one(&pool)
            .await
            .unwrap();
        let night = night.0.expect("night_id is written on every frame");
        assert_eq!(
            night,
            crate::storage::night::night_id(chrono::Utc::now()),
            "night_id must come from the capture timestamp under the local-noon rule"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
