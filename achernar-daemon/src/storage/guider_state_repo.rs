//! guider_state_repo — Section 8.4; equipment spec §3.5 G-i, H9, §4.3.
//!
//! `guider_state_log`: one row per exposure window (and per policy epoch),
//! written by the guider service (WP1A-7). The 003 columns `ra_rms`,
//! `dec_rms` and `total_rms` hold the arcsecond standard deviations; 009 adds
//! the RMS about zero, the step count, the window bounds and frame, the
//! executed algorithm parameters, the M-1 arms and policy version, and the
//! reason learning was frozen. M-1, M-2, TED and TME read these rows.

// Consumed by the recorder and, once the guider is wired (WP1A-7, WP1B-4),
// by the M-1 context builder.
#![allow(dead_code)]

use serde::Serialize;
use sqlx::{Executor, Row, Sqlite};

use crate::equipment::events::GuiderStateRecord;
use crate::storage::columns;

/// Append one window row and return its id. `session_id` must name an
/// existing session or be `None` (the column references `sessions`).
pub async fn insert<'c, E>(executor: E, record: &GuiderStateRecord) -> Result<i64, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let rms = record.rms.as_ref();
    let result = sqlx::query(
        r#"
        INSERT INTO guider_state_log (
            session_id, ra_rms, dec_rms, total_rms, ra_aggressiveness, dec_aggressiveness,
            guide_star_snr, dither_active, timestamp, connection_id, frame_id, window_start,
            window_end, algorithm_params_json, executed_arm_ra, executed_arm_dec,
            policy_version, freeze_reason, rms0_total_arcsec, steps_n
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(&record.session_id)
    .bind(rms.and_then(|r| r.ra_arcsec))
    .bind(rms.and_then(|r| r.dec_arcsec))
    .bind(rms.and_then(|r| r.total_arcsec))
    .bind(record.ra_aggressiveness)
    .bind(record.dec_aggressiveness)
    .bind(record.guide_star_snr)
    .bind(record.dither_active)
    .bind(columns::time(&record.timestamp))
    .bind(record.connection_id.to_string())
    .bind(&record.frame_id)
    .bind(columns::time(&record.window_start))
    .bind(columns::time(&record.window_end))
    .bind(columns::json(&record.algorithm_params))
    .bind(&record.executed_arm_ra)
    .bind(&record.executed_arm_dec)
    .bind(&record.policy_version)
    .bind(&record.freeze_reason)
    .bind(rms.and_then(|r| r.rms0_total_arcsec))
    .bind(rms.map(|r| i64::from(r.steps_n)))
    .execute(executor)
    .await?;
    Ok(result.last_insert_rowid())
}

/// One stored window row.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StoredGuiderState {
    pub id: i64,
    pub session_id: Option<String>,
    pub ra_rms: Option<f64>,
    pub dec_rms: Option<f64>,
    pub total_rms: Option<f64>,
    pub ra_aggressiveness: Option<f64>,
    pub dec_aggressiveness: Option<f64>,
    pub guide_star_snr: Option<f64>,
    pub dither_active: Option<bool>,
    pub timestamp: String,
    pub connection_id: Option<String>,
    pub frame_id: Option<String>,
    pub window_start: Option<String>,
    pub window_end: Option<String>,
    pub algorithm_params: Option<serde_json::Value>,
    pub executed_arm_ra: Option<String>,
    pub executed_arm_dec: Option<String>,
    pub policy_version: Option<String>,
    pub freeze_reason: Option<String>,
    pub rms0_total_arcsec: Option<f64>,
    pub steps_n: Option<i64>,
}

/// Window rows for one frame.
pub async fn for_frame<'c, E>(
    executor: E,
    frame_id: &str,
) -> Result<Vec<StoredGuiderState>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let rows = sqlx::query("SELECT * FROM guider_state_log WHERE frame_id = ? ORDER BY id")
        .bind(frame_id)
        .fetch_all(executor)
        .await?;
    Ok(rows
        .iter()
        .map(|row| {
            let params: Option<String> = row.get("algorithm_params_json");
            StoredGuiderState {
                id: row.get("id"),
                session_id: row.get("session_id"),
                ra_rms: row.get("ra_rms"),
                dec_rms: row.get("dec_rms"),
                total_rms: row.get("total_rms"),
                ra_aggressiveness: row.get("ra_aggressiveness"),
                dec_aggressiveness: row.get("dec_aggressiveness"),
                guide_star_snr: row.get("guide_star_snr"),
                dither_active: row.get("dither_active"),
                timestamp: row.get("timestamp"),
                connection_id: row.get("connection_id"),
                frame_id: row.get("frame_id"),
                window_start: row.get("window_start"),
                window_end: row.get("window_end"),
                algorithm_params: columns::parse_json(params.as_deref()),
                executed_arm_ra: row.get("executed_arm_ra"),
                executed_arm_dec: row.get("executed_arm_dec"),
                policy_version: row.get("policy_version"),
                freeze_reason: row.get("freeze_reason"),
                rms0_total_arcsec: row.get("rms0_total_arcsec"),
                steps_n: row.get("steps_n"),
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::Axis;
    use crate::equipment::snapshot::{AlgoParam, GuideRms};
    use chrono::{TimeZone, Utc};
    use uuid::Uuid;

    #[tokio::test]
    async fn a_window_row_carries_rms_params_and_freeze_reason() {
        let pool = crate::storage::test_pool().await;
        let t0 = Utc.with_ymd_and_hms(2026, 10, 26, 7, 30, 0).unwrap();
        let rec = GuiderStateRecord {
            timestamp: t0 + chrono::Duration::seconds(300),
            session_id: None,
            connection_id: Uuid::new_v4(),
            frame_id: Some("frame-9".into()),
            window_start: t0,
            window_end: t0 + chrono::Duration::seconds(300),
            rms: Some(GuideRms {
                ra_px: 0.3,
                dec_px: 0.25,
                total_px: 0.39,
                rms0_total_px: 0.41,
                ra_arcsec: Some(0.69),
                dec_arcsec: Some(0.58),
                total_arcsec: Some(0.9),
                rms0_total_arcsec: Some(0.94),
                steps_n: 150,
                ..GuideRms::default()
            }),
            ra_aggressiveness: Some(0.7),
            dec_aggressiveness: None,
            guide_star_snr: Some(42.0),
            dither_active: false,
            algorithm_params: vec![AlgoParam {
                axis: Axis::Ra,
                name: "aggression".into(),
                value: 0.7,
            }],
            executed_arm_ra: None,
            executed_arm_dec: None,
            policy_version: None,
            freeze_reason: Some("cloud".into()),
        };
        insert(&pool, &rec).await.unwrap();
        let rows = for_frame(&pool, "frame-9").await.unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.total_rms, Some(0.9));
        assert_eq!(row.rms0_total_arcsec, Some(0.94));
        assert_eq!(row.steps_n, Some(150));
        assert_eq!(row.dec_aggressiveness, None);
        assert_eq!(row.freeze_reason.as_deref(), Some("cloud"));
        assert_eq!(
            row.window_start.as_deref(),
            Some("2026-10-26T07:30:00.000Z")
        );
        assert_eq!(
            row.algorithm_params,
            Some(serde_json::json!([{"axis": "ra", "name": "aggression", "value": 0.7}]))
        );
    }
}
