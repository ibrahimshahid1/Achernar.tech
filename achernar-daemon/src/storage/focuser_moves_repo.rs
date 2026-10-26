//! `focuser_moves`: one row per focuser move, with read-back (equipment spec
//! §3.3 F-a to F-i; migration 009). Both the raw device position and the
//! logical position are kept, with the position source, backlash model and
//! owner, the TempComp state before and after, and the probe temperature.
//! FOP features #7-11 and #23-31 and the FQE labels read this table.

// Consumed by the recorder and, once the spine wires the focuser (WP1B-1),
// by the FOP feature builder (WP1B-2).
#![allow(dead_code)]

use serde::Serialize;
use sqlx::{Executor, Row, Sqlite};
use uuid::Uuid;

use crate::equipment::events::FocuserMoveRecord;
use crate::storage::columns;

/// Insert or update a move. The focuser operation may record the move when
/// it is commanded and again when it settles; the later record wins.
pub async fn upsert<'c, E>(executor: E, record: &FocuserMoveRecord) -> Result<(), sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    sqlx::query(
        r#"
        INSERT INTO focuser_moves (
            id, command_id, connection_id, mode, commanded_target, logical_start,
            logical_end, raw_start, raw_end, position_source, direction, reversal,
            backlash_model, backlash_owner, tempcomp_before, tempcomp_after,
            temperature_start_c, temperature_end_c, commanded_at, started_at, stopped_at,
            settled_at, outcome, source, run_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            command_id          = excluded.command_id,
            connection_id       = excluded.connection_id,
            mode                = excluded.mode,
            commanded_target    = excluded.commanded_target,
            logical_start       = excluded.logical_start,
            logical_end         = excluded.logical_end,
            raw_start           = excluded.raw_start,
            raw_end             = excluded.raw_end,
            position_source     = excluded.position_source,
            direction           = excluded.direction,
            reversal            = excluded.reversal,
            backlash_model      = excluded.backlash_model,
            backlash_owner      = excluded.backlash_owner,
            tempcomp_before     = excluded.tempcomp_before,
            tempcomp_after      = excluded.tempcomp_after,
            temperature_start_c = excluded.temperature_start_c,
            temperature_end_c   = excluded.temperature_end_c,
            commanded_at        = excluded.commanded_at,
            started_at          = excluded.started_at,
            stopped_at          = excluded.stopped_at,
            settled_at          = excluded.settled_at,
            outcome             = excluded.outcome,
            source              = excluded.source,
            run_id              = excluded.run_id
        "#,
    )
    .bind(record.id.to_string())
    .bind(record.command_id.map(|id| id.to_string()))
    .bind(record.connection_id.to_string())
    .bind(columns::name(&record.mode))
    .bind(record.commanded_target)
    .bind(record.logical_start)
    .bind(record.logical_end)
    .bind(record.raw_start)
    .bind(record.raw_end)
    .bind(columns::name(&record.position_source))
    .bind(record.direction.as_ref().and_then(columns::name))
    .bind(record.reversal)
    .bind(columns::json(&record.backlash_model))
    .bind(columns::name(&record.backlash_owner))
    .bind(record.tempcomp_before)
    .bind(record.tempcomp_after)
    .bind(record.temperature_start_c)
    .bind(record.temperature_end_c)
    .bind(columns::time(&record.commanded_at))
    .bind(columns::opt_time(record.started_at.as_ref()))
    .bind(columns::opt_time(record.stopped_at.as_ref()))
    .bind(columns::opt_time(record.settled_at.as_ref()))
    .bind(record.outcome.label())
    .bind(record.source.label())
    .bind(&record.run_id)
    .execute(executor)
    .await?;
    Ok(())
}

/// One stored move. Enum columns hold their serde names; `backlash_model`
/// is the model's JSON (`{"model": "overshoot", "in_steps": .., ..}`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StoredFocuserMove {
    pub id: String,
    pub command_id: Option<String>,
    pub connection_id: String,
    pub mode: String,
    pub commanded_target: i32,
    pub logical_start: Option<i32>,
    pub logical_end: Option<i32>,
    pub raw_start: Option<i32>,
    pub raw_end: Option<i32>,
    pub position_source: String,
    pub direction: Option<String>,
    pub reversal: Option<bool>,
    pub backlash_model: Option<serde_json::Value>,
    pub backlash_owner: Option<String>,
    pub tempcomp_before: Option<bool>,
    pub tempcomp_after: Option<bool>,
    pub temperature_start_c: Option<f64>,
    pub temperature_end_c: Option<f64>,
    pub commanded_at: String,
    pub started_at: Option<String>,
    pub stopped_at: Option<String>,
    pub settled_at: Option<String>,
    pub outcome: Option<String>,
    pub source: Option<String>,
    pub run_id: Option<String>,
}

fn from_row(row: &sqlx::sqlite::SqliteRow) -> StoredFocuserMove {
    let backlash: Option<String> = row.get("backlash_model");
    StoredFocuserMove {
        id: row.get("id"),
        command_id: row.get("command_id"),
        connection_id: row.get("connection_id"),
        mode: row.get("mode"),
        commanded_target: row.get("commanded_target"),
        logical_start: row.get("logical_start"),
        logical_end: row.get("logical_end"),
        raw_start: row.get("raw_start"),
        raw_end: row.get("raw_end"),
        position_source: row.get("position_source"),
        direction: row.get("direction"),
        reversal: row.get("reversal"),
        backlash_model: columns::parse_json(backlash.as_deref()),
        backlash_owner: row.get("backlash_owner"),
        tempcomp_before: row.get("tempcomp_before"),
        tempcomp_after: row.get("tempcomp_after"),
        temperature_start_c: row.get("temperature_start_c"),
        temperature_end_c: row.get("temperature_end_c"),
        commanded_at: row.get("commanded_at"),
        started_at: row.get("started_at"),
        stopped_at: row.get("stopped_at"),
        settled_at: row.get("settled_at"),
        outcome: row.get("outcome"),
        source: row.get("source"),
        run_id: row.get("run_id"),
    }
}

pub async fn load<'c, E>(executor: E, id: Uuid) -> Result<Option<StoredFocuserMove>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let row = sqlx::query("SELECT * FROM focuser_moves WHERE id = ?")
        .bind(id.to_string())
        .fetch_optional(executor)
        .await?;
    Ok(row.as_ref().map(from_row))
}

/// Every move of one autofocus run, in command order.
pub async fn for_run<'c, E>(
    executor: E,
    run_id: &str,
) -> Result<Vec<StoredFocuserMove>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let rows = sqlx::query("SELECT * FROM focuser_moves WHERE run_id = ? ORDER BY commanded_at")
        .bind(run_id)
        .fetch_all(executor)
        .await?;
    Ok(rows.iter().map(from_row).collect())
}

/// The most recent moves of one device identity across connections, newest
/// first: the read-back history FOP features #23-31 need.
pub async fn history_for_identity<'c, E>(
    executor: E,
    identity_key: &str,
    limit: u32,
) -> Result<Vec<StoredFocuserMove>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let rows = sqlx::query(
        r#"
        SELECT m.* FROM focuser_moves m
        JOIN device_connections c ON c.id = m.connection_id
        WHERE c.identity_key = ?
        ORDER BY m.commanded_at DESC
        LIMIT ?
        "#,
    )
    .bind(identity_key)
    .bind(i64::from(limit.max(1)))
    .fetch_all(executor)
    .await?;
    Ok(rows.iter().map(from_row).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::snapshot::PositionSource;
    use crate::equipment::snapshot::{BacklashModel, BacklashOwner, FocusDirection, MoveMode};
    use crate::equipment::types::{CommandId, CommandSource, DeviceError, OpOutcome};
    use chrono::{TimeZone, Utc};

    #[tokio::test]
    async fn a_move_is_recorded_at_command_and_completed_at_settle() {
        let pool = crate::storage::test_pool().await;
        let t0 = Utc.with_ymd_and_hms(2026, 10, 26, 4, 0, 0).unwrap();
        let mut rec = FocuserMoveRecord {
            id: Uuid::new_v4(),
            command_id: Some(CommandId::new()),
            connection_id: Uuid::new_v4(),
            mode: MoveMode::Absolute,
            commanded_target: 15_200,
            logical_start: Some(15_000),
            logical_end: None,
            raw_start: Some(15_040),
            raw_end: None,
            position_source: PositionSource::Device,
            direction: Some(FocusDirection::Out),
            reversal: Some(true),
            backlash_model: BacklashModel::Overshoot {
                in_steps: 0,
                out_steps: 40,
            },
            backlash_owner: BacklashOwner::Achernar,
            tempcomp_before: Some(true),
            tempcomp_after: None,
            temperature_start_c: Some(6.5),
            temperature_end_c: None,
            commanded_at: t0,
            started_at: None,
            stopped_at: None,
            settled_at: None,
            outcome: OpOutcome::Unknown(DeviceError::outcome_unknown("in progress")),
            source: CommandSource::Operator { overriding: false },
            run_id: Some("run-1".into()),
        };
        upsert(&pool, &rec).await.unwrap();
        rec.logical_end = Some(15_200);
        rec.raw_end = Some(15_240);
        rec.tempcomp_after = Some(true);
        rec.temperature_end_c = Some(6.4);
        rec.started_at = Some(t0 + chrono::Duration::milliseconds(120));
        rec.stopped_at = Some(t0 + chrono::Duration::seconds(3));
        rec.settled_at = Some(t0 + chrono::Duration::seconds(4));
        rec.outcome = OpOutcome::Succeeded;
        upsert(&pool, &rec).await.unwrap();

        let stored = load(&pool, rec.id).await.unwrap().unwrap();
        assert_eq!(stored.mode, "abs");
        assert_eq!(stored.position_source, "device");
        assert_eq!(stored.direction.as_deref(), Some("out"));
        assert_eq!(stored.reversal, Some(true));
        assert_eq!(
            stored.backlash_model,
            Some(serde_json::json!({"model": "overshoot", "in_steps": 0, "out_steps": 40}))
        );
        assert_eq!(stored.backlash_owner.as_deref(), Some("achernar"));
        assert_eq!(
            (stored.raw_start, stored.raw_end),
            (Some(15_040), Some(15_240))
        );
        assert_eq!(
            (stored.logical_start, stored.logical_end),
            (Some(15_000), Some(15_200))
        );
        assert_eq!(stored.outcome.as_deref(), Some("succeeded"));
        assert_eq!(stored.source.as_deref(), Some("operator"));
        assert_eq!(
            stored.started_at.as_deref(),
            Some("2026-10-26T04:00:00.120Z")
        );
        assert_eq!(for_run(&pool, "run-1").await.unwrap().len(), 1);

        // Unknown values stay NULL.
        let raw: (Option<i64>, Option<f64>) = sqlx::query_as(
            "SELECT tempcomp_before IS NULL, temperature_start_c FROM focuser_moves",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(raw, (Some(0), Some(6.5)));
    }
}
