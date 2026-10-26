//! `telemetry_log`: the downsampled device time series (equipment spec H19;
//! migration 001). One row per accepted snapshot, `source` = the device kind,
//! `event_type = 'snapshot'`, the snapshot JSON as `payload`. TME/AstroM3Edge
//! and session replay read it. The 1 Hz downsampling happens in the
//! recorder, before a sample is queued.

// Consumed by the recorder and, once the spine wires the hub (WP1B-1), by
// TME exports and replay.
#![allow(dead_code)]

use chrono::{DateTime, Utc};
use sqlx::{Executor, Row, Sqlite};

use crate::equipment::events::TelemetrySample;
use crate::equipment::types::{kind_name, DeviceKind};
use crate::storage::columns;

/// `telemetry_log.event_type` of every equipment snapshot row.
pub const SNAPSHOT_EVENT: &str = "snapshot";

pub async fn insert<'c, E>(executor: E, sample: &TelemetrySample) -> Result<i64, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let result = sqlx::query(
        "INSERT INTO telemetry_log (timestamp, source, event_type, payload, level) \
         VALUES (?, ?, ?, ?, 'info')",
    )
    .bind(columns::time(&sample.at))
    .bind(kind_name(sample.source))
    .bind(SNAPSHOT_EVENT)
    .bind(columns::json(&sample.payload))
    .execute(executor)
    .await?;
    Ok(result.last_insert_rowid())
}

/// Equipment snapshot rows in `[start, end]`, oldest first, optionally for
/// one device kind.
pub async fn snapshots_between<'c, E>(
    executor: E,
    source: Option<DeviceKind>,
    start: &DateTime<Utc>,
    end: &DateTime<Utc>,
) -> Result<Vec<TelemetrySample>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let rows = sqlx::query(
        r#"
        SELECT timestamp, source, payload FROM telemetry_log
        WHERE event_type = 'snapshot' AND timestamp >= ?1 AND timestamp <= ?2
          AND (?3 IS NULL OR source = ?3)
        ORDER BY timestamp, id
        "#,
    )
    .bind(columns::time(start))
    .bind(columns::time(end))
    .bind(source.map(kind_name))
    .fetch_all(executor)
    .await?;
    Ok(rows
        .iter()
        .filter_map(|row| {
            let at: String = row.get("timestamp");
            let source: String = row.get("source");
            let payload: String = row.get("payload");
            Some(TelemetrySample {
                source: columns::from_name(Some(&source))?,
                at: columns::parse_time(&at)?,
                payload: serde_json::from_str(&payload).ok()?,
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[tokio::test]
    async fn snapshot_rows_round_trip_by_kind() {
        let pool = crate::storage::test_pool().await;
        let t0 = Utc.with_ymd_and_hms(2026, 10, 26, 6, 0, 0).unwrap();
        for (i, kind) in [DeviceKind::Mount, DeviceKind::Focuser, DeviceKind::Mount]
            .into_iter()
            .enumerate()
        {
            let sample = TelemetrySample {
                source: kind,
                at: t0 + chrono::Duration::seconds(i as i64),
                payload: serde_json::json!({"i": i}),
            };
            insert(&pool, &sample).await.unwrap();
        }
        let end = t0 + chrono::Duration::seconds(10);
        let mounts = snapshots_between(&pool, Some(DeviceKind::Mount), &t0, &end)
            .await
            .unwrap();
        assert_eq!(mounts.len(), 2);
        assert_eq!(mounts[1].payload, serde_json::json!({"i": 2}));
        assert_eq!(mounts[1].at, t0 + chrono::Duration::seconds(2));
        assert_eq!(
            snapshots_between(&pool, None, &t0, &end)
                .await
                .unwrap()
                .len(),
            3
        );
        let raw: (String, String) =
            sqlx::query_as("SELECT source, event_type FROM telemetry_log ORDER BY id LIMIT 1")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(raw, ("mount".to_string(), "snapshot".to_string()));
    }
}
