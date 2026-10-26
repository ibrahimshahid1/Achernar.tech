//! `device_connections`: one row per device connection (equipment spec §2.6
//! step 10, §3.1 "Records", §4.3; migration 009).
//!
//! This is the provenance table for v6 §3.2 release records and §6.5
//! analytics: which device identity, driver, version and qualification state
//! was connected when a frame was taken. A row opens at connect and closes
//! exactly once, with an `end_class`.

// Consumed by the recorder and, once the spine wires the hub (WP1B-1), by
// connect/disconnect, shutdown and the provenance queries.
#![allow(dead_code)]

use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::{Executor, Row, Sqlite};
use uuid::Uuid;

use crate::equipment::events::{ConnectionEnd, DeviceConnectionRecord};
use crate::equipment::types::{kind_name, DeviceKind};
use crate::storage::columns;

/// Insert the row for a new connection. Re-recording the same connection
/// refreshes the identity claims (name, driver info, capabilities) but never
/// moves `connected_at` or reopens a closed row.
pub async fn open<'c, E>(
    executor: E,
    record: &DeviceConnectionRecord,
    session_id: Option<&str>,
    night_id: Option<&str>,
) -> Result<(), sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let identity = &record.identity;
    sqlx::query(
        r#"
        INSERT INTO device_connections (
            id, kind, driver, identity_key, unique_id, name, description, driver_info,
            driver_version, interface_version, endpoint, capabilities_json, qualification,
            connected_at, session_id, night_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            identity_key      = excluded.identity_key,
            unique_id         = excluded.unique_id,
            name              = excluded.name,
            description       = excluded.description,
            driver_info       = excluded.driver_info,
            driver_version    = excluded.driver_version,
            interface_version = excluded.interface_version,
            endpoint          = excluded.endpoint,
            capabilities_json = excluded.capabilities_json,
            qualification     = excluded.qualification
        "#,
    )
    .bind(identity.connection_id.to_string())
    .bind(kind_name(identity.kind))
    .bind(identity.driver.as_str())
    .bind(&identity.identity_key)
    .bind(&identity.unique_id)
    .bind(&identity.name)
    .bind(&identity.description)
    .bind(&identity.driver_info)
    .bind(&identity.driver_version)
    .bind(identity.interface_version)
    .bind(identity.endpoint.describe())
    .bind(record.capabilities.to_string())
    .bind(identity.qualification.as_str())
    .bind(columns::time(&record.connected_at))
    .bind(record.session_id.as_deref().or(session_id))
    .bind(record.night_id.as_deref().or(night_id))
    .execute(executor)
    .await?;
    Ok(())
}

/// Close one connection. Only the first close counts; returns `false` when
/// the row was already closed or does not exist.
pub async fn close<'c, E>(
    executor: E,
    connection_id: Uuid,
    disconnected_at: &DateTime<Utc>,
    end_class: ConnectionEnd,
) -> Result<bool, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let result = sqlx::query(
        "UPDATE device_connections SET disconnected_at = ?, end_class = ? \
         WHERE id = ? AND disconnected_at IS NULL",
    )
    .bind(columns::time(disconnected_at))
    .bind(columns::name(&end_class))
    .bind(connection_id.to_string())
    .execute(executor)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// Close every open row at `at`, e.g. with `end_class = shutdown` during
/// daemon shutdown (§2.15 step 6). Returns the number closed.
pub async fn close_all_open<'c, E>(
    executor: E,
    at: &DateTime<Utc>,
    end_class: ConnectionEnd,
) -> Result<u64, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let result = sqlx::query(
        "UPDATE device_connections SET disconnected_at = ?, end_class = ? \
         WHERE disconnected_at IS NULL",
    )
    .bind(columns::time(at))
    .bind(columns::name(&end_class))
    .execute(executor)
    .await?;
    Ok(result.rows_affected())
}

/// Close rows left open by a previous process that ended without a clean
/// shutdown. The real end time is unknown, so the row ends at the last event
/// journaled for that connection, or at its connect time when there is none;
/// it is never stretched to "now". Run once when a store is attached, before
/// any new connection is recorded.
pub async fn close_stale<'c, E>(executor: E, end_class: ConnectionEnd) -> Result<u64, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let result = sqlx::query(
        r#"
        UPDATE device_connections
        SET disconnected_at = COALESCE(
                (SELECT MAX(e.at_utc) FROM equipment_events e
                  WHERE e.connection_id = device_connections.id),
                connected_at),
            end_class = ?
        WHERE disconnected_at IS NULL
        "#,
    )
    .bind(columns::name(&end_class))
    .execute(executor)
    .await?;
    Ok(result.rows_affected())
}

/// One stored connection.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StoredConnection {
    pub id: String,
    pub kind: String,
    pub driver: String,
    pub identity_key: String,
    pub unique_id: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub driver_info: Option<String>,
    pub driver_version: Option<String>,
    pub interface_version: Option<i32>,
    pub endpoint: Option<String>,
    pub capabilities: serde_json::Value,
    pub qualification: String,
    pub connected_at: String,
    pub disconnected_at: Option<String>,
    pub end_class: Option<String>,
    pub session_id: Option<String>,
    pub night_id: Option<String>,
}

impl StoredConnection {
    fn from_row(row: &sqlx::sqlite::SqliteRow) -> Self {
        let capabilities: String = row.get("capabilities_json");
        Self {
            id: row.get("id"),
            kind: row.get("kind"),
            driver: row.get("driver"),
            identity_key: row.get("identity_key"),
            unique_id: row.get("unique_id"),
            name: row.get("name"),
            description: row.get("description"),
            driver_info: row.get("driver_info"),
            driver_version: row.get("driver_version"),
            interface_version: row.get("interface_version"),
            endpoint: row.get("endpoint"),
            capabilities: serde_json::from_str(&capabilities).unwrap_or(serde_json::Value::Null),
            qualification: row.get("qualification"),
            connected_at: row.get("connected_at"),
            disconnected_at: row.get("disconnected_at"),
            end_class: row.get("end_class"),
            session_id: row.get("session_id"),
            night_id: row.get("night_id"),
        }
    }
}

const COLUMNS: &str = "id, kind, driver, identity_key, unique_id, name, description, driver_info, \
                       driver_version, interface_version, endpoint, capabilities_json, \
                       qualification, connected_at, disconnected_at, end_class, session_id, \
                       night_id";

pub async fn load<'c, E>(
    executor: E,
    connection_id: Uuid,
) -> Result<Option<StoredConnection>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let sql = format!("SELECT {COLUMNS} FROM device_connections WHERE id = ?");
    let row = sqlx::query(&sql)
        .bind(connection_id.to_string())
        .fetch_optional(executor)
        .await?;
    Ok(row.as_ref().map(StoredConnection::from_row))
}

/// The connection of `kind` that was open at `at`: the device a frame or a
/// sample taken at that time came from (v6 §3.2, §6.5).
pub async fn active_at<'c, E>(
    executor: E,
    kind: DeviceKind,
    at: &DateTime<Utc>,
) -> Result<Option<StoredConnection>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let sql = format!(
        "SELECT {COLUMNS} FROM device_connections \
         WHERE kind = ?1 AND connected_at <= ?2 \
           AND (disconnected_at IS NULL OR disconnected_at > ?2) \
         ORDER BY connected_at DESC LIMIT 1"
    );
    let row = sqlx::query(&sql)
        .bind(kind_name(kind))
        .bind(columns::time(at))
        .fetch_optional(executor)
        .await?;
    Ok(row.as_ref().map(StoredConnection::from_row))
}

/// Connections of one identity, newest first.
pub async fn for_identity<'c, E>(
    executor: E,
    identity_key: &str,
    limit: u32,
) -> Result<Vec<StoredConnection>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let sql = format!(
        "SELECT {COLUMNS} FROM device_connections WHERE identity_key = ? \
         ORDER BY connected_at DESC LIMIT ?"
    );
    let rows = sqlx::query(&sql)
        .bind(identity_key)
        .bind(i64::from(limit.max(1)))
        .fetch_all(executor)
        .await?;
    Ok(rows.iter().map(StoredConnection::from_row).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::events::{EquipmentEvent, EventKind};
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::Clock;
    use crate::equipment::types::{DeviceIdentity, DriverId, Endpoint, Qualification};
    use chrono::TimeZone;

    fn record(at: DateTime<Utc>) -> DeviceConnectionRecord {
        let mut identity = DeviceIdentity::new(
            DeviceKind::Mount,
            DriverId::Alpaca,
            Endpoint::AlpacaHttp {
                base: "http://10.0.0.5:11111".into(),
                device_type: "telescope".into(),
                number: 0,
            },
            "alpaca:esp32-mount-1",
            "ESP32 Mount",
        );
        identity.unique_id = Some("esp32-mount-1".into());
        identity.driver_version = Some("1.4.0".into());
        identity.interface_version = Some(4);
        DeviceConnectionRecord {
            identity,
            capabilities: serde_json::json!({"alignment_mode": "polar", "can_park": true}),
            connected_at: at,
            session_id: None,
            night_id: None,
        }
    }

    #[tokio::test]
    async fn a_connection_opens_and_closes_once() {
        let pool = crate::storage::test_pool().await;
        let t0 = Utc.with_ymd_and_hms(2026, 10, 26, 2, 0, 0).unwrap();
        let rec = record(t0);
        let id = rec.identity.connection_id;
        open(&pool, &rec, Some("s1"), Some("2026-10-25"))
            .await
            .unwrap();

        let row = load(&pool, id).await.unwrap().unwrap();
        assert_eq!(row.kind, "mount");
        assert_eq!(row.driver, "alpaca");
        assert_eq!(row.identity_key, "alpaca:esp32-mount-1");
        assert_eq!(row.unique_id.as_deref(), Some("esp32-mount-1"));
        assert_eq!(row.interface_version, Some(4));
        assert_eq!(
            row.endpoint.as_deref(),
            Some("http://10.0.0.5:11111/api/v1/telescope/0")
        );
        assert_eq!(row.qualification, "implemented_unqualified");
        assert_eq!(row.capabilities["can_park"], serde_json::json!(true));
        assert_eq!(row.connected_at, "2026-10-26T02:00:00.000Z");
        assert_eq!(row.session_id.as_deref(), Some("s1"));
        assert_eq!(row.disconnected_at, None);

        // Re-recording refreshes claims but keeps the connect time.
        let mut again = record(t0 + chrono::Duration::minutes(5));
        again.identity.connection_id = id;
        again.identity.qualification = Qualification::Qualified {
            record_id: "rig-acceptance-1".into(),
        };
        open(&pool, &again, None, None).await.unwrap();
        let row = load(&pool, id).await.unwrap().unwrap();
        assert_eq!(row.qualification, "qualified");
        assert_eq!(row.connected_at, "2026-10-26T02:00:00.000Z");

        let t1 = t0 + chrono::Duration::hours(3);
        assert!(close(&pool, id, &t1, ConnectionEnd::Operator)
            .await
            .unwrap());
        assert!(!close(
            &pool,
            id,
            &(t1 + chrono::Duration::hours(1)),
            ConnectionEnd::Lost
        )
        .await
        .unwrap());
        let row = load(&pool, id).await.unwrap().unwrap();
        assert_eq!(row.end_class.as_deref(), Some("operator"));
        assert_eq!(
            row.disconnected_at.as_deref(),
            Some("2026-10-26T05:00:00.000Z")
        );

        // Provenance: which mount was connected at a given time.
        let mid = t0 + chrono::Duration::hours(1);
        assert_eq!(
            active_at(&pool, DeviceKind::Mount, &mid)
                .await
                .unwrap()
                .map(|c| c.id),
            Some(id.to_string())
        );
        assert!(active_at(&pool, DeviceKind::Mount, &t1)
            .await
            .unwrap()
            .is_none());
        assert!(active_at(&pool, DeviceKind::Focuser, &mid)
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            for_identity(&pool, "alpaca:esp32-mount-1", 5)
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn stale_rows_end_at_their_last_event_and_shutdown_closes_the_rest() {
        let pool = crate::storage::test_pool().await;
        let clock = ManualClock::new();
        let stale = record(clock.now_utc());
        let quiet = record(clock.now_utc());
        open(&pool, &stale, None, None).await.unwrap();
        open(&pool, &quiet, None, None).await.unwrap();
        clock.advance_ms(90_000);
        let event = EquipmentEvent {
            seq: 1,
            ..EquipmentEvent::new(EventKind::SlewStarted, &clock)
                .for_device(DeviceKind::Mount, stale.identity.connection_id)
        };
        crate::storage::equipment_events_repo::insert(&pool, &event, None, None)
            .await
            .unwrap();

        assert_eq!(close_stale(&pool, ConnectionEnd::Lost).await.unwrap(), 2);
        let row = load(&pool, stale.identity.connection_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            row.disconnected_at.as_deref(),
            Some("2026-01-01T00:01:30.000Z")
        );
        assert_eq!(row.end_class.as_deref(), Some("lost"));
        let row = load(&pool, quiet.identity.connection_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            row.disconnected_at.as_deref(),
            Some("2026-01-01T00:00:00.000Z")
        );

        let live = record(clock.now_utc());
        open(&pool, &live, None, None).await.unwrap();
        let at = clock.now_utc();
        assert_eq!(
            close_all_open(&pool, &at, ConnectionEnd::Shutdown)
                .await
                .unwrap(),
            1
        );
        let row = load(&pool, live.identity.connection_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(row.end_class.as_deref(), Some("shutdown"));
    }
}
