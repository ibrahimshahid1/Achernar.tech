//! pointing_model_repo — Section 8.4; equipment spec §4.3 "Existing tables
//! that get their first writer", M-4 (v6 §8.3.4).
//!
//! `pointing_model_snapshots`: one row per pointing-model refit (WP1B-4),
//! fitted from `pointing_samples`. A snapshot records which mount identity
//! and sync generation it was fitted for inside `model_data`, so a model is
//! never applied across a sync or to another mount.

// Consumed once the pointing model is rewired (WP1B-4).
#![allow(dead_code)]

use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::{Executor, Row, Sqlite};

use crate::storage::columns;

/// One fitted model.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PointingModelSnapshot {
    pub id: String,
    /// `rigs.id`, or `None` when no rig row exists.
    pub rig_id: Option<String>,
    pub sample_count: u32,
    pub rms_error_arcmin: Option<f64>,
    /// The fitted terms plus their provenance: mount identity key, sync
    /// generation, frame and epoch, pier-side handling.
    pub model_data: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

pub async fn insert<'c, E>(executor: E, snapshot: &PointingModelSnapshot) -> Result<(), sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    sqlx::query(
        "INSERT INTO pointing_model_snapshots \
         (id, rig_id, sample_count, rms_error_arcmin, model_data, created_at) \
         VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(&snapshot.id)
    .bind(&snapshot.rig_id)
    .bind(i64::from(snapshot.sample_count))
    .bind(snapshot.rms_error_arcmin.filter(|v| v.is_finite()))
    .bind(columns::json(&snapshot.model_data))
    .bind(columns::time(&snapshot.created_at))
    .execute(executor)
    .await?;
    Ok(())
}

/// The newest model, optionally for one rig.
pub async fn latest<'c, E>(
    executor: E,
    rig_id: Option<&str>,
) -> Result<Option<PointingModelSnapshot>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let row = sqlx::query(
        "SELECT * FROM pointing_model_snapshots WHERE (?1 IS NULL OR rig_id = ?1) \
         ORDER BY created_at DESC, rowid DESC LIMIT 1",
    )
    .bind(rig_id)
    .fetch_optional(executor)
    .await?;
    Ok(row.and_then(|row| {
        let created_at: String = row.get("created_at");
        let model_data: String = row.get("model_data");
        Some(PointingModelSnapshot {
            id: row.get("id"),
            rig_id: row.get("rig_id"),
            sample_count: u32::try_from(row.get::<i64, _>("sample_count")).ok()?,
            rms_error_arcmin: row.get("rms_error_arcmin"),
            model_data: serde_json::from_str(&model_data).ok()?,
            created_at: columns::parse_time(&created_at)?,
        })
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[tokio::test]
    async fn the_latest_model_is_returned_per_rig() {
        let pool = crate::storage::test_pool().await;
        sqlx::query("INSERT INTO rigs (id, name) VALUES ('rig-1', 'Backyard')")
            .execute(&pool)
            .await
            .unwrap();
        let t0 = Utc.with_ymd_and_hms(2026, 10, 26, 8, 0, 0).unwrap();
        let older = PointingModelSnapshot {
            id: "pm-1".into(),
            rig_id: Some("rig-1".into()),
            sample_count: 12,
            rms_error_arcmin: Some(1.4),
            model_data: serde_json::json!({
                "mount_identity_key": "alpaca:esp32-mount-1",
                "sync_generation": 0,
                "terms": {"ih": 0.1}
            }),
            created_at: t0,
        };
        let newer = PointingModelSnapshot {
            id: "pm-2".into(),
            sample_count: 20,
            rms_error_arcmin: Some(0.9),
            created_at: t0 + chrono::Duration::hours(1),
            ..older.clone()
        };
        insert(&pool, &older).await.unwrap();
        insert(&pool, &newer).await.unwrap();
        assert_eq!(
            latest(&pool, Some("rig-1")).await.unwrap(),
            Some(newer.clone())
        );
        assert_eq!(latest(&pool, None).await.unwrap(), Some(newer));
        assert_eq!(latest(&pool, Some("rig-2")).await.unwrap(), None);

        // The rig must exist (the column references rigs).
        let orphan = PointingModelSnapshot {
            id: "pm-3".into(),
            rig_id: Some("rig-2".into()),
            ..older
        };
        assert!(insert(&pool, &orphan).await.is_err());
    }
}
