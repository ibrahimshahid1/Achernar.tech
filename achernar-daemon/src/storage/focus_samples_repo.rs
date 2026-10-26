//! focus_samples_repo — Section 7.4; equipment spec H3, H7, §4.3.
//!
//! `focus_samples`: one autofocus sample. Migration 009 adds the A06
//! freshness and provenance columns: position read back before and after
//! the exposure, the position source, the focuser move, when the focuser
//! settled and the exposure started, the capture identity and clock quality,
//! whether anything moved during the exposure, the filter verification, the
//! temperature source and age, the TempComp state and the backlash model.
//! FQE labels and FOP features read them, and an unknown value stays NULL.
//!
//! The autofocus integration (WP1B-2) writes samples through [`insert`].

// Consumed once autofocus is rewired onto the equipment layer (WP1B-2).
#![allow(dead_code)]

use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::{Executor, Row, Sqlite};
use uuid::Uuid;

use crate::equipment::snapshot::{BacklashModel, PositionSource};
use crate::equipment::types::ClockQuality;
use crate::storage::columns;

/// The 009 provenance of one sample (H3). Every field is optional: a value
/// the layer could not establish stays unknown.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct FocusSampleProvenance {
    pub position_readback_before: Option<i32>,
    pub position_readback_after: Option<i32>,
    pub position_source: Option<PositionSource>,
    pub focuser_move_id: Option<Uuid>,
    pub settled_at: Option<DateTime<Utc>>,
    pub exposure_started_at: Option<DateTime<Utc>>,
    pub capture_id: Option<String>,
    pub clock_quality: Option<ClockQuality>,
    pub moved_during_exposure: Option<bool>,
    pub filter_verified: Option<bool>,
    /// `TemperatureSourceKind` serde name: `weather`, `focuser_probe` or
    /// `camera_sensor`.
    pub temperature_source: Option<String>,
    pub temperature_age_ms: Option<u64>,
    /// `on`, `off` or `unavailable`.
    pub tempcomp_state: Option<String>,
    pub backlash_model: Option<BacklashModel>,
}

/// One sample with its provenance.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct FocusSampleRow {
    pub run_id: String,
    pub position: i32,
    pub hfr: f64,
    pub hfd: Option<f64>,
    pub fwhm: Option<f64>,
    pub star_count: u32,
    pub quality_score: Option<f64>,
    pub temperature: Option<f64>,
    pub captured_at: DateTime<Utc>,
    pub frame_id: Option<String>,
    pub provenance: FocusSampleProvenance,
}

/// Append one sample and return its id. The run must exist.
pub async fn insert<'c, E>(executor: E, sample: &FocusSampleRow) -> Result<i64, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let p = &sample.provenance;
    let result = sqlx::query(
        r#"
        INSERT INTO focus_samples (
            run_id, position, hfr, hfd, fwhm, star_count, quality_score, temperature,
            captured_at, frame_id, position_readback_before, position_readback_after,
            position_source, focuser_move_id, settled_at, exposure_started_at, capture_id,
            clock_quality, moved_during_exposure, filter_verified, temperature_source,
            temperature_age_ms, tempcomp_state, backlash_model
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(&sample.run_id)
    .bind(sample.position)
    .bind(sample.hfr)
    .bind(sample.hfd)
    .bind(sample.fwhm)
    .bind(i64::from(sample.star_count))
    .bind(sample.quality_score)
    .bind(sample.temperature)
    .bind(columns::time(&sample.captured_at))
    .bind(&sample.frame_id)
    .bind(p.position_readback_before)
    .bind(p.position_readback_after)
    .bind(p.position_source.as_ref().and_then(columns::name))
    .bind(p.focuser_move_id.map(|id| id.to_string()))
    .bind(columns::opt_time(p.settled_at.as_ref()))
    .bind(columns::opt_time(p.exposure_started_at.as_ref()))
    .bind(&p.capture_id)
    .bind(p.clock_quality.as_ref().and_then(columns::name))
    .bind(p.moved_during_exposure)
    .bind(p.filter_verified)
    .bind(&p.temperature_source)
    .bind(p.temperature_age_ms.map(columns::int))
    .bind(&p.tempcomp_state)
    .bind(p.backlash_model.as_ref().map(columns::json))
    .execute(executor)
    .await?;
    Ok(result.last_insert_rowid())
}

/// The provenance stored for one sample, or `None` when no such sample.
pub async fn load_provenance<'c, E>(
    executor: E,
    sample_id: i64,
) -> Result<Option<FocusSampleProvenance>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let row = sqlx::query("SELECT * FROM focus_samples WHERE id = ?")
        .bind(sample_id)
        .fetch_optional(executor)
        .await?;
    Ok(row.map(|row| {
        let text = |name: &str| -> Option<String> { row.get(name) };
        FocusSampleProvenance {
            position_readback_before: row.get("position_readback_before"),
            position_readback_after: row.get("position_readback_after"),
            position_source: columns::from_name(text("position_source").as_deref()),
            focuser_move_id: text("focuser_move_id").and_then(|id| Uuid::parse_str(&id).ok()),
            settled_at: columns::parse_opt_time(text("settled_at")),
            exposure_started_at: columns::parse_opt_time(text("exposure_started_at")),
            capture_id: row.get("capture_id"),
            clock_quality: columns::from_name(text("clock_quality").as_deref()),
            moved_during_exposure: row.get("moved_during_exposure"),
            filter_verified: row.get("filter_verified"),
            temperature_source: row.get("temperature_source"),
            temperature_age_ms: row
                .get::<Option<i64>, _>("temperature_age_ms")
                .map(columns::uint),
            tempcomp_state: row.get("tempcomp_state"),
            backlash_model: columns::parse_json(text("backlash_model").as_deref()),
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[tokio::test]
    async fn a_sample_keeps_its_a06_provenance() {
        let pool = crate::storage::test_pool().await;
        sqlx::query(
            "INSERT INTO focus_runs (id, rig_id, start_position, state, started_at) \
             VALUES ('run-1', 'rig', 15000, 'running', '2026-10-26T03:00:00.000Z')",
        )
        .execute(&pool)
        .await
        .unwrap();
        let t0 = Utc.with_ymd_and_hms(2026, 10, 26, 3, 0, 10).unwrap();
        let provenance = FocusSampleProvenance {
            position_readback_before: Some(15_100),
            position_readback_after: Some(15_100),
            position_source: Some(PositionSource::Device),
            focuser_move_id: Some(Uuid::new_v4()),
            settled_at: Some(t0),
            exposure_started_at: Some(t0 + chrono::Duration::milliseconds(250)),
            capture_id: Some("cap-1".into()),
            clock_quality: Some(ClockQuality::HostTriggerBracket),
            moved_during_exposure: Some(false),
            filter_verified: None,
            temperature_source: Some("focuser_probe".into()),
            temperature_age_ms: Some(1800),
            tempcomp_state: Some("off".into()),
            backlash_model: Some(BacklashModel::None),
        };
        let sample = FocusSampleRow {
            run_id: "run-1".into(),
            position: 15_100,
            hfr: 2.4,
            star_count: 31,
            captured_at: t0 + chrono::Duration::seconds(3),
            frame_id: Some("frame-af-1".into()),
            provenance: provenance.clone(),
            ..FocusSampleRow::default()
        };
        let id = insert(&pool, &sample).await.unwrap();
        assert_eq!(load_provenance(&pool, id).await.unwrap(), Some(provenance));
        assert_eq!(load_provenance(&pool, id + 1).await.unwrap(), None);
    }
}
