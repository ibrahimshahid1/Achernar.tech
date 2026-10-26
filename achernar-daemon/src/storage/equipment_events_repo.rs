//! `equipment_events`: the equipment event journal (equipment spec §4.2,
//! §4.3; migration 009), and the `equipment_overheads` view built on it (H18).
//!
//! The recorder writes events in batches. `seq` is assigned by the hub, which
//! continues from [`max_seq`] when a store is attached. `guide_step` events
//! are never journaled here; they are stored in `guide_steps`. Rows store the
//! catalogue name in `kind` and the variant's fields in `payload_json`, so
//! [`StoredEvent::event`] rebuilds the typed [`EquipmentEvent`].

// Consumed by the recorder and, once the spine wires the hub (WP1B-1), by the
// `GET equipment/events` route and the hub's sequence restore.
#![allow(dead_code)]

use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::{Executor, Row, Sqlite};

use crate::equipment::events::EquipmentEvent;
use crate::equipment::types::kind_name;
use crate::storage::columns;

/// The journal tail served by `GET equipment/events` is capped at this many
/// rows (spec §2.12).
pub const MAX_TAIL: u32 = 500;

/// Append one event. Returns `false` when an event with the same `event_id`
/// is already stored, so recording an event twice is harmless. A different
/// event with a sequence number already in use is an error: it means two
/// writers disagree about `seq`, which must not be hidden.
///
/// `seq` must have been assigned by the hub (non-zero).
pub async fn insert<'c, E>(
    executor: E,
    event: &EquipmentEvent,
    session_id: Option<&str>,
    night_id: Option<&str>,
) -> Result<bool, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let (kind, payload) = kind_and_payload(event);
    let result = sqlx::query(
        r#"
        INSERT INTO equipment_events (
            seq, event_id, at_utc, mono_ns, device_kind, connection_id, command_id,
            kind, payload_json, device_time, session_id, night_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO NOTHING
        "#,
    )
    .bind(columns::int(event.seq))
    .bind(event.event_id.to_string())
    .bind(columns::time(&event.at_utc))
    .bind(columns::int(event.mono_ns))
    .bind(event.device_kind.map(kind_name))
    .bind(event.connection_id.map(|id| id.to_string()))
    .bind(event.command_id.map(|id| id.to_string()))
    .bind(kind)
    .bind(payload)
    .bind(columns::opt_time(event.device_time.as_ref()))
    .bind(session_id)
    .bind(night_id)
    .execute(executor)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// The catalogue name and the payload JSON text. Unit variants (no fields)
/// store `{}`, the column default.
fn kind_and_payload(event: &EquipmentEvent) -> (String, String) {
    let payload = serde_json::to_value(&event.event)
        .ok()
        .and_then(|mut value| value.get_mut("payload").map(std::mem::take))
        .unwrap_or_else(|| serde_json::json!({}));
    (event.name().to_string(), payload.to_string())
}

/// The highest stored `seq`, or `None` for an empty journal. The hub
/// continues numbering from here when a store is attached.
pub async fn max_seq<'c, E>(executor: E) -> Result<Option<u64>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let row: (Option<i64>,) = sqlx::query_as("SELECT MAX(seq) FROM equipment_events")
        .fetch_one(executor)
        .await?;
    Ok(row.0.map(columns::uint))
}

/// One journal row.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StoredEvent {
    pub seq: u64,
    pub event_id: String,
    pub at_utc: String,
    pub mono_ns: u64,
    pub device_kind: Option<String>,
    pub connection_id: Option<String>,
    pub command_id: Option<String>,
    pub kind: String,
    pub payload: serde_json::Value,
    pub device_time: Option<String>,
    pub session_id: Option<String>,
    pub night_id: Option<String>,
}

impl StoredEvent {
    fn from_row(row: &sqlx::sqlite::SqliteRow) -> Self {
        let payload: String = row.get("payload_json");
        Self {
            seq: columns::uint(row.get("seq")),
            event_id: row.get("event_id"),
            at_utc: row.get("at_utc"),
            mono_ns: columns::uint(row.get("mono_ns")),
            device_kind: row.get("device_kind"),
            connection_id: row.get("connection_id"),
            command_id: row.get("command_id"),
            kind: row.get("kind"),
            payload: serde_json::from_str(&payload).unwrap_or(serde_json::Value::Null),
            device_time: row.get("device_time"),
            session_id: row.get("session_id"),
            night_id: row.get("night_id"),
        }
    }

    pub fn at(&self) -> Option<DateTime<Utc>> {
        columns::parse_time(&self.at_utc)
    }

    /// The typed event, or `None` when the row's kind or payload no longer
    /// matches the catalogue (for example a row written by a newer build).
    /// Times come back at millisecond precision, the stored precision.
    pub fn event(&self) -> Option<EquipmentEvent> {
        let mut envelope = serde_json::json!({
            "seq": self.seq,
            "event_id": self.event_id,
            "at_utc": self.at_utc,
            "mono_ns": self.mono_ns,
            "device_kind": self.device_kind,
            "connection_id": self.connection_id,
            "command_id": self.command_id,
            "device_time": self.device_time,
            "kind": self.kind,
        });
        let unit = self.payload.as_object().is_some_and(|map| map.is_empty());
        if !unit {
            envelope["payload"] = self.payload.clone();
            return serde_json::from_value(envelope).ok();
        }
        // A unit variant has no payload; a struct variant with only optional
        // fields may still have stored `{}`. Try both.
        serde_json::from_value(envelope.clone()).ok().or_else(|| {
            envelope["payload"] = serde_json::json!({});
            serde_json::from_value(envelope).ok()
        })
    }
}

const COLUMNS: &str = "seq, event_id, at_utc, mono_ns, device_kind, connection_id, command_id, \
                       kind, payload_json, device_time, session_id, night_id";

/// Events with `seq > since_seq`, oldest first, optionally only the given
/// catalogue kinds, at most `limit` rows (capped at [`MAX_TAIL`]).
pub async fn tail<'c, E>(
    executor: E,
    since_seq: u64,
    kinds: &[&str],
    limit: u32,
) -> Result<Vec<StoredEvent>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let limit = limit.clamp(1, MAX_TAIL);
    let kinds_json = (!kinds.is_empty()).then(|| columns::json(&kinds));
    let sql = format!(
        "SELECT {COLUMNS} FROM equipment_events \
         WHERE seq > ? AND (?2 IS NULL OR kind IN (SELECT value FROM json_each(?2))) \
         ORDER BY seq LIMIT ?3"
    );
    let rows = sqlx::query(&sql)
        .bind(columns::int(since_seq))
        .bind(kinds_json)
        .bind(i64::from(limit))
        .fetch_all(executor)
        .await?;
    Ok(rows.iter().map(StoredEvent::from_row).collect())
}

/// Every journaled event of one command, in order (gate stages, operation
/// records, device events).
pub async fn for_command<'c, E>(
    executor: E,
    command_id: &str,
) -> Result<Vec<StoredEvent>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let sql = format!("SELECT {COLUMNS} FROM equipment_events WHERE command_id = ? ORDER BY seq");
    let rows = sqlx::query(&sql)
        .bind(command_id)
        .fetch_all(executor)
        .await?;
    Ok(rows.iter().map(StoredEvent::from_row).collect())
}

/// One row of the `equipment_overheads` view (H18): a finished operation and
/// how long it took. Read by FOR, FW-3 and planning.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Overhead {
    pub command_id: String,
    /// The `OpKind` serde name, e.g. `filter_handshake`.
    pub operation: Option<String>,
    /// `succeeded`, `failed`, `aborted`, `timed_out` or `unknown`.
    pub outcome: Option<String>,
    pub device_kind: Option<String>,
    pub started_at: Option<String>,
    pub ended_at: String,
    pub duration_s: Option<f64>,
    pub session_id: Option<String>,
    pub night_id: Option<String>,
}

/// Overheads, newest first, optionally for one operation kind and from a
/// given time on.
pub async fn overheads<'c, E>(
    executor: E,
    operation: Option<&str>,
    since: Option<&DateTime<Utc>>,
    limit: u32,
) -> Result<Vec<Overhead>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let rows = sqlx::query(
        r#"
        SELECT command_id, operation, outcome, device_kind, started_at, ended_at,
               duration_s, session_id, night_id
        FROM equipment_overheads
        WHERE (?1 IS NULL OR operation = ?1) AND (?2 IS NULL OR ended_at >= ?2)
        ORDER BY ended_at DESC
        LIMIT ?3
        "#,
    )
    .bind(operation)
    .bind(columns::opt_time(since))
    .bind(i64::from(limit.max(1)))
    .fetch_all(executor)
    .await?;
    Ok(rows
        .iter()
        .map(|row| Overhead {
            command_id: row.get("command_id"),
            operation: row.get("operation"),
            outcome: row.get("outcome"),
            device_kind: row.get("device_kind"),
            started_at: row.get("started_at"),
            ended_at: row.get("ended_at"),
            duration_s: row.get("duration_s"),
            session_id: row.get("session_id"),
            night_id: row.get("night_id"),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::events::EventKind;
    use crate::equipment::ops::OpKind;
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::Clock;
    use crate::equipment::types::{
        CommandId, CommandSource, DeviceKind, EqCoord, EqFrame, OpOutcome, PierSide,
    };
    use uuid::Uuid;

    fn sequenced(seq: u64, event: EquipmentEvent) -> EquipmentEvent {
        EquipmentEvent { seq, ..event }
    }

    #[tokio::test]
    async fn events_round_trip_through_the_journal() {
        let pool = crate::storage::test_pool().await;
        let clock = ManualClock::new();
        let connection = Uuid::new_v4();
        let command = CommandId::new();
        let events = vec![
            sequenced(
                1,
                EquipmentEvent::new(EventKind::SlewStarted, &clock)
                    .for_device(DeviceKind::Mount, connection)
                    .with_command(command),
            ),
            sequenced(
                2,
                EquipmentEvent::new(
                    EventKind::SlewSettled {
                        position: Some(EqCoord::new(5.5, 20.0, EqFrame::JNowApparent)),
                        error_arcmin: Some(0.3),
                    },
                    &clock,
                )
                .for_device(DeviceKind::Mount, connection)
                .with_command(command)
                .with_device_time(clock.now_utc()),
            ),
            sequenced(
                3,
                EquipmentEvent::new(
                    EventKind::PierSideChanged {
                        from: PierSide::East,
                        to: PierSide::Unknown,
                    },
                    &clock,
                ),
            ),
            sequenced(
                4,
                EquipmentEvent::new(EventKind::DeviceReconnected { reason: None }, &clock),
            ),
        ];
        for event in &events {
            assert!(insert(&pool, event, Some("s1"), Some("2025-12-31"))
                .await
                .unwrap());
        }
        // The same event again is a no-op, not an error.
        assert!(!insert(&pool, &events[0], None, None).await.unwrap());
        assert_eq!(max_seq(&pool).await.unwrap(), Some(4));

        let stored = tail(&pool, 0, &[], 100).await.unwrap();
        assert_eq!(stored.len(), 4);
        assert_eq!(stored[0].kind, "slew_started");
        assert_eq!(stored[0].payload, serde_json::json!({}));
        assert_eq!(stored[0].session_id.as_deref(), Some("s1"));
        assert_eq!(stored[1].device_kind.as_deref(), Some("mount"));
        assert_eq!(stored[1].at_utc, "2026-01-01T00:00:00.000Z");
        for (row, original) in stored.iter().zip(&events) {
            assert_eq!(row.event().as_ref(), Some(original), "{}", row.kind);
        }

        let only = tail(&pool, 1, &["pier_side_changed"], 100).await.unwrap();
        assert_eq!(only.len(), 1);
        assert_eq!(only[0].seq, 3);
        assert_eq!(
            only[0].payload,
            serde_json::json!({"from": "east", "to": null})
        );
        let by_command = for_command(&pool, &command.to_string()).await.unwrap();
        assert_eq!(
            by_command.iter().map(|e| e.seq).collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[tokio::test]
    async fn a_conflicting_sequence_number_is_an_error() {
        let pool = crate::storage::test_pool().await;
        let clock = ManualClock::new();
        let a = sequenced(7, EquipmentEvent::new(EventKind::GuidingStarted, &clock));
        let b = sequenced(7, EquipmentEvent::new(EventKind::GuidingStopped, &clock));
        insert(&pool, &a, None, None).await.unwrap();
        assert!(insert(&pool, &b, None, None).await.is_err());
    }

    #[tokio::test]
    async fn the_tail_is_capped_and_empty_journals_have_no_max() {
        let pool = crate::storage::test_pool().await;
        assert_eq!(max_seq(&pool).await.unwrap(), None);
        let clock = ManualClock::new();
        for seq in 1..=(u64::from(MAX_TAIL) + 5) {
            let event = sequenced(seq, EquipmentEvent::new(EventKind::SettleBegin, &clock));
            insert(&pool, &event, None, None).await.unwrap();
        }
        assert_eq!(
            tail(&pool, 0, &[], 10_000).await.unwrap().len(),
            MAX_TAIL as usize
        );
    }

    #[tokio::test]
    async fn overheads_come_from_operation_ended_events() {
        let pool = crate::storage::test_pool().await;
        let clock = ManualClock::new();
        let command = CommandId::new();
        let commanded = sequenced(
            1,
            EquipmentEvent::new(
                EventKind::OperationCommanded {
                    op: OpKind::Dither,
                    source: CommandSource::System,
                },
                &clock,
            )
            .with_command(command),
        );
        clock.advance_ms(12_500);
        let ended = sequenced(
            2,
            EquipmentEvent::new(
                EventKind::OperationEnded {
                    op: OpKind::Dither,
                    outcome: OpOutcome::Succeeded,
                    duration_ms: 12_500,
                },
                &clock,
            )
            .for_device(DeviceKind::Guider, Uuid::new_v4())
            .with_command(command),
        );
        insert(&pool, &commanded, None, Some("n1")).await.unwrap();
        insert(&pool, &ended, None, Some("n1")).await.unwrap();
        let rows = overheads(&pool, Some("dither"), None, 10).await.unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.outcome.as_deref(), Some("succeeded"));
        assert_eq!(row.device_kind.as_deref(), Some("guider"));
        assert_eq!(row.started_at.as_deref(), Some("2026-01-01T00:00:00.000Z"));
        assert_eq!(row.duration_s, Some(12.5));
        assert_eq!(row.night_id.as_deref(), Some("n1"));
        assert!(overheads(&pool, Some("slew"), None, 10)
            .await
            .unwrap()
            .is_empty());
    }
}
