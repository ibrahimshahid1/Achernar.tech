//! `dither_events`: every dither and its settle outcome (equipment spec §3.5
//! G-f, H10; migration 009), linked to the frame that followed it. M-2's
//! reward needs that link: which exposure came after the dither, and whether
//! it was accepted.
//!
//! The guider service writes the row when the dither is requested and again
//! when it settles (an upsert). The recorder links `next_frame_id` when the
//! next exposure completes ([`link_next_frame`]). Acceptance is judged later
//! by whoever grades the frame ([`set_next_frame_accepted`]).

// Consumed by the recorder and, once the spine wires the guider (WP1B-1), by
// the M-2 context builder (WP1B-4).
#![allow(dead_code)]

use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::{Executor, Row, Sqlite};
use uuid::Uuid;

use crate::equipment::events::DitherRecord;
use crate::storage::columns;

/// Insert or update a dither. A later update never erases a link the
/// recorder already made: `next_frame_id` and `next_frame_accepted` keep
/// their stored values unless the update carries new ones.
pub async fn upsert<'c, E>(executor: E, record: &DitherRecord) -> Result<(), sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    sqlx::query(
        r#"
        INSERT INTO dither_events (
            id, command_id, source, policy_version, backend, amount, amount_unit, ra_only,
            settle_pixels, settle_time_s, settle_timeout_s, requested_at, dx, dy,
            settle_done_at, settle_status, settle_error, total_frames, dropped_frames,
            max_distance_px, next_frame_id, next_frame_accepted
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            command_id          = excluded.command_id,
            source              = excluded.source,
            policy_version      = excluded.policy_version,
            backend             = excluded.backend,
            amount              = excluded.amount,
            amount_unit         = excluded.amount_unit,
            ra_only             = excluded.ra_only,
            settle_pixels       = excluded.settle_pixels,
            settle_time_s       = excluded.settle_time_s,
            settle_timeout_s    = excluded.settle_timeout_s,
            requested_at        = excluded.requested_at,
            dx                  = excluded.dx,
            dy                  = excluded.dy,
            settle_done_at      = excluded.settle_done_at,
            settle_status       = excluded.settle_status,
            settle_error        = excluded.settle_error,
            total_frames        = excluded.total_frames,
            dropped_frames      = excluded.dropped_frames,
            max_distance_px     = excluded.max_distance_px,
            next_frame_id       = COALESCE(excluded.next_frame_id, dither_events.next_frame_id),
            next_frame_accepted = COALESCE(excluded.next_frame_accepted,
                                           dither_events.next_frame_accepted)
        "#,
    )
    .bind(record.id.to_string())
    .bind(record.command_id.map(|id| id.to_string()))
    .bind(record.source.label())
    .bind(&record.policy_version)
    .bind(columns::name(&record.backend))
    .bind(record.amount)
    .bind(columns::name(&record.amount_unit))
    .bind(record.ra_only)
    .bind(record.settle.pixels)
    .bind(record.settle.time_s)
    .bind(record.settle.timeout_s)
    .bind(columns::time(&record.requested_at))
    .bind(record.dx)
    .bind(record.dy)
    .bind(columns::opt_time(record.settle_done_at.as_ref()))
    .bind(record.settle_status)
    .bind(&record.settle_error)
    .bind(record.total_frames.map(i64::from))
    .bind(record.dropped_frames.map(i64::from))
    .bind(record.max_distance_px)
    .bind(&record.next_frame_id)
    .bind(record.next_frame_accepted)
    .execute(executor)
    .await?;
    Ok(())
}

/// Link the frame that followed each pending dither (H10).
///
/// A dither is pending while `next_frame_id` is NULL. The completed exposure
/// is its next frame when the exposure started at or after the dither was
/// requested, and no more than `horizon` after it, so a dither left over from
/// an earlier sequence is not credited with an unrelated frame. Several
/// dithers before one frame all link to it. Returns the number linked.
pub async fn link_next_frame<'c, E>(
    executor: E,
    frame_id: &str,
    exposure_start: &DateTime<Utc>,
    horizon: chrono::Duration,
) -> Result<u64, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let earliest = *exposure_start - horizon;
    let result = sqlx::query(
        "UPDATE dither_events SET next_frame_id = ? \
         WHERE next_frame_id IS NULL AND requested_at <= ? AND requested_at >= ?",
    )
    .bind(frame_id)
    .bind(columns::time(exposure_start))
    .bind(columns::time(&earliest))
    .execute(executor)
    .await?;
    Ok(result.rows_affected())
}

/// Record whether the frame that followed a dither was accepted. Returns the
/// number of dithers updated.
pub async fn set_next_frame_accepted<'c, E>(
    executor: E,
    frame_id: &str,
    accepted: bool,
) -> Result<u64, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let result =
        sqlx::query("UPDATE dither_events SET next_frame_accepted = ? WHERE next_frame_id = ?")
            .bind(accepted)
            .bind(frame_id)
            .execute(executor)
            .await?;
    Ok(result.rows_affected())
}

/// One stored dither. `source` is the command-source label
/// (`operator`, `policy:m2@0.3.0`, ...).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StoredDither {
    pub id: String,
    pub command_id: Option<String>,
    pub source: Option<String>,
    pub policy_version: Option<String>,
    pub backend: String,
    pub amount: f64,
    pub amount_unit: String,
    pub ra_only: bool,
    pub settle_pixels: Option<f64>,
    pub settle_time_s: Option<f64>,
    pub settle_timeout_s: Option<f64>,
    pub requested_at: String,
    pub dx: Option<f64>,
    pub dy: Option<f64>,
    pub settle_done_at: Option<String>,
    pub settle_status: Option<i32>,
    pub settle_error: Option<String>,
    pub total_frames: Option<i64>,
    pub dropped_frames: Option<i64>,
    pub max_distance_px: Option<f64>,
    pub next_frame_id: Option<String>,
    pub next_frame_accepted: Option<bool>,
}

fn from_row(row: &sqlx::sqlite::SqliteRow) -> StoredDither {
    StoredDither {
        id: row.get("id"),
        command_id: row.get("command_id"),
        source: row.get("source"),
        policy_version: row.get("policy_version"),
        backend: row.get("backend"),
        amount: row.get("amount"),
        amount_unit: row.get("amount_unit"),
        ra_only: row.get("ra_only"),
        settle_pixels: row.get("settle_pixels"),
        settle_time_s: row.get("settle_time_s"),
        settle_timeout_s: row.get("settle_timeout_s"),
        requested_at: row.get("requested_at"),
        dx: row.get("dx"),
        dy: row.get("dy"),
        settle_done_at: row.get("settle_done_at"),
        settle_status: row.get("settle_status"),
        settle_error: row.get("settle_error"),
        total_frames: row.get("total_frames"),
        dropped_frames: row.get("dropped_frames"),
        max_distance_px: row.get("max_distance_px"),
        next_frame_id: row.get("next_frame_id"),
        next_frame_accepted: row.get("next_frame_accepted"),
    }
}

pub async fn load<'c, E>(executor: E, id: Uuid) -> Result<Option<StoredDither>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let row = sqlx::query("SELECT * FROM dither_events WHERE id = ?")
        .bind(id.to_string())
        .fetch_optional(executor)
        .await?;
    Ok(row.as_ref().map(from_row))
}

/// Dithers requested in `[start, end]`, oldest first.
pub async fn between<'c, E>(
    executor: E,
    start: &DateTime<Utc>,
    end: &DateTime<Utc>,
) -> Result<Vec<StoredDither>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let rows = sqlx::query(
        "SELECT * FROM dither_events WHERE requested_at >= ? AND requested_at <= ? \
         ORDER BY requested_at",
    )
    .bind(columns::time(start))
    .bind(columns::time(end))
    .fetch_all(executor)
    .await?;
    Ok(rows.iter().map(from_row).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::{PixelUnit, Settle};
    use crate::equipment::events::DitherBackend;
    use crate::equipment::types::{CommandId, CommandSource};
    use chrono::TimeZone;

    fn dither(requested_at: DateTime<Utc>) -> DitherRecord {
        DitherRecord {
            id: Uuid::new_v4(),
            command_id: Some(CommandId::new()),
            source: CommandSource::Policy {
                model: "m2".into(),
                version: "0.3.0".into(),
            },
            policy_version: Some("0.3.0".into()),
            backend: DitherBackend::Phd2,
            amount: 5.0,
            amount_unit: PixelUnit::GuidePx,
            ra_only: false,
            settle: Settle::default(),
            requested_at,
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

    #[tokio::test]
    async fn a_settle_update_keeps_the_next_frame_link() {
        let pool = crate::storage::test_pool().await;
        let t0 = Utc.with_ymd_and_hms(2026, 10, 26, 3, 0, 0).unwrap();
        let mut rec = dither(t0);
        upsert(&pool, &rec).await.unwrap();
        let stored = load(&pool, rec.id).await.unwrap().unwrap();
        assert_eq!(stored.source.as_deref(), Some("policy:m2@0.3.0"));
        assert_eq!(stored.backend, "phd2");
        assert_eq!(stored.amount_unit, "guide_px");
        assert_eq!(stored.settle_pixels, Some(1.5));
        assert_eq!(stored.settle_done_at, None);

        let start = t0 + chrono::Duration::seconds(30);
        let linked = link_next_frame(&pool, "frame-1", &start, chrono::Duration::hours(2))
            .await
            .unwrap();
        assert_eq!(linked, 1);
        // The settle outcome arrives after the link; the link survives.
        rec.dx = Some(3.1);
        rec.dy = Some(-2.4);
        rec.settle_done_at = Some(t0 + chrono::Duration::seconds(12));
        rec.settle_status = Some(0);
        rec.total_frames = Some(6);
        rec.dropped_frames = Some(0);
        upsert(&pool, &rec).await.unwrap();
        assert_eq!(
            set_next_frame_accepted(&pool, "frame-1", true)
                .await
                .unwrap(),
            1
        );
        let stored = load(&pool, rec.id).await.unwrap().unwrap();
        assert_eq!(stored.next_frame_id.as_deref(), Some("frame-1"));
        assert_eq!(stored.next_frame_accepted, Some(true));
        assert_eq!(stored.dx, Some(3.1));
        assert_eq!(
            stored.settle_done_at.as_deref(),
            Some("2026-10-26T03:00:12.000Z")
        );
        assert_eq!(between(&pool, &t0, &start).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn only_pending_dithers_inside_the_horizon_are_linked() {
        let pool = crate::storage::test_pool().await;
        let t0 = Utc.with_ymd_and_hms(2026, 10, 26, 3, 0, 0).unwrap();
        let old = dither(t0 - chrono::Duration::hours(5));
        let recent = dither(t0);
        let later = dither(t0 + chrono::Duration::minutes(10));
        for rec in [&old, &recent, &later] {
            upsert(&pool, rec).await.unwrap();
        }
        let start = t0 + chrono::Duration::minutes(1);
        assert_eq!(
            link_next_frame(&pool, "f1", &start, chrono::Duration::hours(2))
                .await
                .unwrap(),
            1
        );
        let next = t0 + chrono::Duration::minutes(11);
        assert_eq!(
            link_next_frame(&pool, "f2", &next, chrono::Duration::hours(2))
                .await
                .unwrap(),
            1
        );
        let get = |id| {
            let pool = pool.clone();
            async move { load(&pool, id).await.unwrap().unwrap().next_frame_id }
        };
        assert_eq!(get(old.id).await, None);
        assert_eq!(get(recent.id).await.as_deref(), Some("f1"));
        assert_eq!(get(later.id).await.as_deref(), Some("f2"));
    }
}
