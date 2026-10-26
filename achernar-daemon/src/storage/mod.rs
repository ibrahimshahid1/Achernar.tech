//! Storage layer — Sections 5.2, 7.4, 8.4.
//! SQLite repository modules for all persistent data.
//!
//! The equipment-layer repositories (equipment spec §4.3, migration 009) are
//! written by `equipment::recorder` through one batched writer task, and read
//! by the hub, the API and the models. They share the column encodings in
//! [`columns`].

pub mod analysis_assets;
pub mod capture_archive;
pub mod capture_retention;
pub mod capture_store;
pub mod control_audit_repo;
pub mod device_connections_repo;
pub mod device_profiles_repo;
pub mod dither_repo;
pub mod environment_repo;
pub mod equipment_events_repo;
pub mod exposure_windows_repo;
pub mod filter_switch_repo;
pub mod focus_actions_repo;
pub mod focus_drift_repo;
pub mod focus_model_registry_repo;
pub mod focus_predictions_repo;
pub mod focus_runs_repo;
pub mod focus_samples_repo;
pub mod focuser_moves_repo;
pub mod frames_repo;
pub mod guide_steps_repo;
pub mod guider_state_repo;
pub mod local_store;
pub mod night;
pub mod plate_solve_repo;
pub mod pointing_model_repo;
pub mod pointing_samples_repo;
pub mod policy_decisions_repo;
pub mod sessions_repo;
pub mod sync_client;
pub mod telemetry_log_repo;

/// Column encodings shared by the equipment-layer repositories, following the
/// migration 009 conventions: times are RFC 3339 UTC text with milliseconds,
/// enums are stored by their serde name, structured values are JSON text, and
/// a missing value is NULL, never a synthesised 0.
#[allow(dead_code)]
pub(crate) mod columns {
    use chrono::{DateTime, SecondsFormat, Utc};
    use serde::de::DeserializeOwned;
    use serde::Serialize;

    /// `2026-01-01T00:00:00.000Z`. One format everywhere, so text columns
    /// compare in time order.
    pub fn time(at: &DateTime<Utc>) -> String {
        at.to_rfc3339_opts(SecondsFormat::Millis, true)
    }

    pub fn opt_time(at: Option<&DateTime<Utc>>) -> Option<String> {
        at.map(time)
    }

    /// Parse a stored time. Accepts any RFC 3339 form, and SQLite's
    /// `datetime('now')` default (`YYYY-MM-DD HH:MM:SS`, UTC).
    pub fn parse_time(text: &str) -> Option<DateTime<Utc>> {
        if let Ok(at) = DateTime::parse_from_rfc3339(text) {
            return Some(at.with_timezone(&Utc));
        }
        chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S")
            .ok()
            .map(|naive| naive.and_utc())
    }

    pub fn parse_opt_time(text: Option<String>) -> Option<DateTime<Utc>> {
        text.as_deref().and_then(parse_time)
    }

    /// The serde name of an enum value, e.g. `jnow_apparent`. `None` when
    /// the value serialises to anything but a string (for example
    /// `PierSide::Unknown`, which is JSON `null`).
    pub fn name<T: Serialize>(value: &T) -> Option<String> {
        match serde_json::to_value(value) {
            Ok(serde_json::Value::String(name)) => Some(name),
            _ => None,
        }
    }

    /// Parse an enum from its stored serde name.
    pub fn from_name<T: DeserializeOwned>(name: Option<&str>) -> Option<T> {
        let name = name?;
        serde_json::from_value(serde_json::Value::String(name.to_string())).ok()
    }

    /// JSON text of any value.
    pub fn json<T: Serialize>(value: &T) -> String {
        serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
    }

    /// Parse stored JSON text; `None` for NULL or malformed text.
    pub fn parse_json<T: DeserializeOwned>(text: Option<&str>) -> Option<T> {
        serde_json::from_str(text?).ok()
    }

    /// SQLite integers are signed; saturate rather than wrap.
    pub fn int(value: u64) -> i64 {
        i64::try_from(value).unwrap_or(i64::MAX)
    }

    pub fn uint(value: i64) -> u64 {
        u64::try_from(value).unwrap_or(0)
    }
}

/// An in-memory database with every migration applied, for repository and
/// recorder tests. One connection that never expires: each new SQLite
/// `:memory:` connection would otherwise be a different, empty database.
/// Foreign keys are on, as in `LocalStore::open`.
#[cfg(test)]
pub(crate) async fn test_pool() -> sqlx::SqlitePool {
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
    use std::str::FromStr;

    let options = SqliteConnectOptions::from_str("sqlite::memory:")
        .unwrap()
        .foreign_keys(true);
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .min_connections(1)
        .idle_timeout(None)
        .max_lifetime(None)
        .connect_with(options)
        .await
        .unwrap();
    for (version, _, sql) in local_store::migrations_for_test() {
        for statement in local_store::split_statements_for_test(sql) {
            if let Err(err) = sqlx::query(&statement).execute(&pool).await {
                let msg = err.to_string().to_ascii_lowercase();
                assert!(
                    msg.contains("duplicate column name") || msg.contains("already exists"),
                    "migration {version} failed: {err}"
                );
            }
        }
    }
    pool
}
