//! Local SQLite store for daemon offline-first operation.
//!
//! The daemon owns the durable edge state: focus runs/samples/actions,
//! frame metadata, equipment profiles, sync queue entries, and control logs.
//! Cloud sync can replay these tables later without requiring the rig to stay
//! online during acquisition.

use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use sqlx::SqlitePool;
use std::path::Path;

#[derive(Clone)]
pub struct LocalStore {
    pool: SqlitePool,
}

impl LocalStore {
    pub async fn open(database_path: &str) -> crate::error::Result<Self> {
        if database_path != ":memory:" {
            if let Some(parent) = Path::new(database_path).parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
        }

        let options = SqliteConnectOptions::new()
            .filename(database_path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .foreign_keys(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await
            .map_err(storage_error("open local SQLite store"))?;

        sqlx::query("PRAGMA journal_mode = WAL")
            .execute(&pool)
            .await
            .map_err(storage_error("enable SQLite WAL"))?;
        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(&pool)
            .await
            .map_err(storage_error("enable SQLite foreign keys"))?;

        let store = Self { pool };
        store.create_schema().await?;
        Ok(store)
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    async fn create_schema(&self) -> crate::error::Result<()> {
        for statement in SCHEMA {
            sqlx::query(statement)
                .execute(&self.pool)
                .await
                .map_err(storage_error("create local SQLite schema"))?;
        }
        Ok(())
    }
}

fn storage_error(context: &'static str) -> impl Fn(sqlx::Error) -> crate::error::AchernarError {
    move |err| crate::error::StorageError::Database(format!("{context}: {err}")).into()
}

const SCHEMA: &[&str] = &[
    r#"
    CREATE TABLE IF NOT EXISTS focus_runs (
        id TEXT PRIMARY KEY,
        rig_id TEXT NOT NULL,
        filter_name TEXT,
        start_position INTEGER NOT NULL,
        best_position INTEGER,
        best_hfr REAL,
        state TEXT NOT NULL,
        temperature REAL,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS focus_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        position INTEGER NOT NULL,
        hfr REAL NOT NULL,
        fwhm REAL,
        star_count INTEGER NOT NULL DEFAULT 0,
        quality_score REAL,
        temperature REAL,
        captured_at TEXT NOT NULL,
        frame_id TEXT,
        FOREIGN KEY (run_id) REFERENCES focus_runs(id) ON DELETE CASCADE
    )
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS focus_actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        action TEXT NOT NULL,
        from_position INTEGER,
        to_position INTEGER,
        steps INTEGER,
        reason TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (run_id) REFERENCES focus_runs(id) ON DELETE SET NULL
    )
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS focus_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        model_name TEXT,
        model_version TEXT,
        predicted_offset INTEGER,
        confidence REAL,
        payload_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (run_id) REFERENCES focus_runs(id) ON DELETE SET NULL
    )
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS frames (
        id TEXT PRIMARY KEY,
        session_id TEXT,
        local_path TEXT NOT NULL,
        object_key TEXT,
        checksum_sha256 TEXT,
        width INTEGER,
        height INTEGER,
        exposure_s REAL,
        gain REAL,
        filter_name TEXT,
        temperature REAL,
        wcs_json TEXT NOT NULL DEFAULT '{}',
        quality_json TEXT NOT NULL DEFAULT '{}',
        captured_at TEXT NOT NULL,
        synced_at TEXT
    )
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS equipment_profiles (
        id TEXT PRIMARY KEY,
        equipment_type TEXT NOT NULL,
        name TEXT NOT NULL,
        driver TEXT NOT NULL,
        options_json TEXT NOT NULL DEFAULT '{}',
        last_seen_at TEXT,
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS sync_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_type TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        operation TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        synced_at TEXT
    )
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS control_audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        actor TEXT NOT NULL DEFAULT 'local',
        action TEXT NOT NULL,
        target TEXT,
        payload_json TEXT NOT NULL DEFAULT '{}',
        accepted INTEGER NOT NULL DEFAULT 1,
        reason TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    "#,
    "CREATE INDEX IF NOT EXISTS focus_samples_run_idx ON focus_samples(run_id)",
    "CREATE INDEX IF NOT EXISTS frames_session_idx ON frames(session_id, captured_at)",
    "CREATE INDEX IF NOT EXISTS sync_queue_pending_idx ON sync_queue(synced_at, created_at)",
    "CREATE INDEX IF NOT EXISTS control_audit_logs_created_idx ON control_audit_logs(created_at)",
];
