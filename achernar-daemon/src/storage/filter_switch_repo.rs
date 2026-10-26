//! filter_switch_repo — Section 8.4; equipment spec §3.4 W-c, H8, §4.3.
//!
//! `filter_switch_events`: one row per filter change, written at the end of
//! the v6 §8.2.1 handshake (`equipment/filter/change.rs`, WP1A-6). Migration
//! 003 created the base columns; 009 added the handshake record: slots,
//! timings, the offset applied and its focuser move, whether guiding was
//! paused, whether the handshake completed, the outcome, and the policy arm
//! and propensity when a policy (FW-2) chose the filter. FW-1, FW-2, FW-3 and
//! FOR read these rows.

// Consumed by the recorder and, once the handshake is wired (WP1A-6), by the
// filter policies.
#![allow(dead_code)]

use serde::Serialize;
use sqlx::{Executor, Row, Sqlite};
use uuid::Uuid;

use crate::equipment::events::FilterSwitchRecord;
use crate::storage::columns;

/// Insert or update one switch. `session_id` must name an existing session
/// or be `None` (the column references `sessions`).
pub async fn upsert<'c, E>(executor: E, record: &FilterSwitchRecord) -> Result<(), sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    sqlx::query(
        r#"
        INSERT INTO filter_switch_events (
            id, session_id, from_filter, to_filter, focus_offset, duration_ms, success,
            created_at, command_id, from_slot, to_slot, requested_at, wheel_done_at,
            verified_at, settle_ms_actual, offset_from, offset_to, offset_applied_steps,
            focuser_move_id, guiding_paused, handshake_ok, outcome, trigger, policy_version,
            arm, propensity, background_before, background_after
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            session_id           = excluded.session_id,
            from_filter          = excluded.from_filter,
            to_filter            = excluded.to_filter,
            focus_offset         = excluded.focus_offset,
            duration_ms          = excluded.duration_ms,
            success              = excluded.success,
            command_id           = excluded.command_id,
            from_slot            = excluded.from_slot,
            to_slot              = excluded.to_slot,
            requested_at         = excluded.requested_at,
            wheel_done_at        = excluded.wheel_done_at,
            verified_at          = excluded.verified_at,
            settle_ms_actual     = excluded.settle_ms_actual,
            offset_from          = excluded.offset_from,
            offset_to            = excluded.offset_to,
            offset_applied_steps = excluded.offset_applied_steps,
            focuser_move_id      = excluded.focuser_move_id,
            guiding_paused       = excluded.guiding_paused,
            handshake_ok         = excluded.handshake_ok,
            outcome              = excluded.outcome,
            trigger              = excluded.trigger,
            policy_version       = excluded.policy_version,
            arm                  = excluded.arm,
            propensity           = excluded.propensity,
            background_before    = excluded.background_before,
            background_after     = excluded.background_after
        "#,
    )
    .bind(record.id.to_string())
    .bind(&record.session_id)
    .bind(&record.from_filter)
    .bind(&record.to_filter)
    .bind(record.focus_offset)
    .bind(record.duration_ms.map(columns::int))
    .bind(record.success)
    .bind(columns::time(&record.requested_at))
    .bind(record.command_id.map(|id| id.to_string()))
    .bind(record.from_slot.map(i64::from))
    .bind(i64::from(record.to_slot))
    .bind(columns::time(&record.requested_at))
    .bind(columns::opt_time(record.wheel_done_at.as_ref()))
    .bind(columns::opt_time(record.verified_at.as_ref()))
    .bind(record.settle_ms_actual.map(columns::int))
    .bind(record.offset_from)
    .bind(record.offset_to)
    .bind(record.offset_applied_steps)
    .bind(record.focuser_move_id.map(|id| id.to_string()))
    .bind(record.guiding_paused)
    .bind(record.handshake_ok)
    .bind(record.outcome.label())
    .bind(&record.trigger)
    .bind(&record.policy_version)
    .bind(&record.arm)
    .bind(record.propensity)
    .bind(record.background_before)
    .bind(record.background_after)
    .execute(executor)
    .await?;
    Ok(())
}

/// One stored switch, including rows written before migration 009 (whose
/// handshake columns are NULL).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StoredFilterSwitch {
    pub id: String,
    pub session_id: Option<String>,
    pub from_filter: Option<String>,
    pub to_filter: String,
    pub focus_offset: Option<i32>,
    pub duration_ms: Option<i64>,
    pub success: bool,
    pub created_at: String,
    pub command_id: Option<String>,
    pub from_slot: Option<i64>,
    pub to_slot: Option<i64>,
    pub requested_at: Option<String>,
    pub wheel_done_at: Option<String>,
    pub verified_at: Option<String>,
    pub settle_ms_actual: Option<i64>,
    pub offset_from: Option<i32>,
    pub offset_to: Option<i32>,
    pub offset_applied_steps: Option<i32>,
    pub focuser_move_id: Option<String>,
    pub guiding_paused: Option<bool>,
    pub handshake_ok: Option<bool>,
    pub outcome: Option<String>,
    pub trigger: Option<String>,
    pub policy_version: Option<String>,
    pub arm: Option<String>,
    pub propensity: Option<f64>,
    pub background_before: Option<f64>,
    pub background_after: Option<f64>,
}

fn from_row(row: &sqlx::sqlite::SqliteRow) -> StoredFilterSwitch {
    StoredFilterSwitch {
        id: row.get("id"),
        session_id: row.get("session_id"),
        from_filter: row.get("from_filter"),
        to_filter: row.get("to_filter"),
        focus_offset: row.get("focus_offset"),
        duration_ms: row.get("duration_ms"),
        success: row.get("success"),
        created_at: row.get("created_at"),
        command_id: row.get("command_id"),
        from_slot: row.get("from_slot"),
        to_slot: row.get("to_slot"),
        requested_at: row.get("requested_at"),
        wheel_done_at: row.get("wheel_done_at"),
        verified_at: row.get("verified_at"),
        settle_ms_actual: row.get("settle_ms_actual"),
        offset_from: row.get("offset_from"),
        offset_to: row.get("offset_to"),
        offset_applied_steps: row.get("offset_applied_steps"),
        focuser_move_id: row.get("focuser_move_id"),
        guiding_paused: row.get("guiding_paused"),
        handshake_ok: row.get("handshake_ok"),
        outcome: row.get("outcome"),
        trigger: row.get("trigger"),
        policy_version: row.get("policy_version"),
        arm: row.get("arm"),
        propensity: row.get("propensity"),
        background_before: row.get("background_before"),
        background_after: row.get("background_after"),
    }
}

pub async fn load<'c, E>(executor: E, id: Uuid) -> Result<Option<StoredFilterSwitch>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let row = sqlx::query("SELECT * FROM filter_switch_events WHERE id = ?")
        .bind(id.to_string())
        .fetch_optional(executor)
        .await?;
    Ok(row.as_ref().map(from_row))
}

/// The most recent switches, newest first: the measured switch and settle
/// durations FOR and FW-3 plan with.
pub async fn recent<'c, E>(executor: E, limit: u32) -> Result<Vec<StoredFilterSwitch>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let rows = sqlx::query("SELECT * FROM filter_switch_events ORDER BY created_at DESC LIMIT ?")
        .bind(i64::from(limit.max(1)))
        .fetch_all(executor)
        .await?;
    Ok(rows.iter().map(from_row).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::types::{CommandId, DeviceError, OpOutcome};
    use chrono::{TimeZone, Utc};

    fn switch() -> FilterSwitchRecord {
        let t0 = Utc.with_ymd_and_hms(2026, 10, 26, 7, 0, 0).unwrap();
        FilterSwitchRecord {
            id: Uuid::new_v4(),
            session_id: None,
            from_filter: Some("L".into()),
            to_filter: "Ha".into(),
            focus_offset: Some(120),
            duration_ms: Some(6400),
            success: true,
            command_id: Some(CommandId::new()),
            from_slot: Some(0),
            to_slot: 4,
            requested_at: t0,
            wheel_done_at: Some(t0 + chrono::Duration::milliseconds(3200)),
            verified_at: Some(t0 + chrono::Duration::milliseconds(3300)),
            settle_ms_actual: Some(500),
            offset_from: Some(0),
            offset_to: Some(120),
            offset_applied_steps: Some(120),
            focuser_move_id: Some(Uuid::new_v4()),
            guiding_paused: true,
            handshake_ok: true,
            outcome: OpOutcome::Succeeded,
            trigger: Some("policy".into()),
            policy_version: Some("fw2-0.1.0".into()),
            arm: Some("Ha".into()),
            propensity: Some(0.25),
            background_before: Some(140.0),
            background_after: None,
        }
    }

    #[tokio::test]
    async fn a_handshake_record_round_trips() {
        let pool = crate::storage::test_pool().await;
        let rec = switch();
        upsert(&pool, &rec).await.unwrap();
        let stored = load(&pool, rec.id).await.unwrap().unwrap();
        assert_eq!(stored.to_filter, "Ha");
        assert_eq!((stored.from_slot, stored.to_slot), (Some(0), Some(4)));
        assert_eq!(stored.offset_applied_steps, Some(120));
        assert_eq!(stored.guiding_paused, Some(true));
        assert_eq!(stored.handshake_ok, Some(true));
        assert_eq!(stored.outcome.as_deref(), Some("succeeded"));
        assert_eq!(stored.propensity, Some(0.25));
        assert_eq!(stored.background_after, None);
        assert_eq!(
            stored.wheel_done_at.as_deref(),
            Some("2026-10-26T07:00:03.200Z")
        );
        assert_eq!(stored.created_at, "2026-10-26T07:00:00.000Z");
    }

    #[tokio::test]
    async fn a_timed_out_handshake_is_not_a_success() {
        let pool = crate::storage::test_pool().await;
        let mut rec = switch();
        rec.success = false;
        rec.handshake_ok = false;
        rec.verified_at = None;
        rec.outcome = OpOutcome::Unknown(DeviceError::outcome_unknown("wheel timeout"));
        upsert(&pool, &rec).await.unwrap();
        let stored = recent(&pool, 5).await.unwrap();
        assert_eq!(stored.len(), 1);
        assert!(!stored[0].success);
        assert_eq!(stored[0].handshake_ok, Some(false));
        assert_eq!(stored[0].outcome.as_deref(), Some("unknown"));
        assert_eq!(stored[0].verified_at, None);
    }

    #[tokio::test]
    async fn the_session_must_exist() {
        let pool = crate::storage::test_pool().await;
        let mut rec = switch();
        rec.session_id = Some("no-such-session".into());
        assert!(upsert(&pool, &rec).await.is_err());
        sqlx::query("INSERT INTO sessions (id) VALUES ('s1')")
            .execute(&pool)
            .await
            .unwrap();
        rec.session_id = Some("s1".into());
        upsert(&pool, &rec).await.unwrap();
    }
}
