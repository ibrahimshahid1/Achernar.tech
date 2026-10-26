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

    /// Apply every migration that has not run yet.
    ///
    /// `migrations/*.sql` is the single source of truth for the schema. It did
    /// not used to be: the files sat unexecuted while a `const SCHEMA` in this
    /// module created a *different* set of tables, so the two definitions of
    /// `frames` disagreed and reading the migrations told you nothing about
    /// the database you actually had. The embedded copy is gone.
    ///
    /// The runner is deliberately small rather than pulling in `sqlx::migrate!`
    /// — it needs one behaviour that generic runners do not have, described at
    /// `is_benign` below.
    async fn create_schema(&self) -> crate::error::Result<()> {
        sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS schema_migrations (
                   version     INTEGER PRIMARY KEY,
                   name        TEXT NOT NULL,
                   applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
               )"#,
        )
        .execute(&self.pool)
        .await
        .map_err(storage_error("create schema_migrations"))?;

        for (version, name, sql) in MIGRATIONS {
            let already: Option<(i64,)> =
                sqlx::query_as("SELECT version FROM schema_migrations WHERE version = ?")
                    .bind(*version)
                    .fetch_optional(&self.pool)
                    .await
                    .map_err(storage_error("read schema_migrations"))?;
            if already.is_some() {
                continue;
            }

            tracing::info!("Applying migration {:03}_{}", version, name);
            for statement in split_statements(sql) {
                match sqlx::query(&statement).execute(&self.pool).await {
                    Ok(_) => {}
                    Err(err) if is_benign(&err) => {
                        // Databases created before this runner existed already
                        // carry some of these columns, added at startup by the
                        // old `ensure_column` calls. `ALTER TABLE ADD COLUMN`
                        // is not idempotent the way `CREATE TABLE IF NOT
                        // EXISTS` is, so a first run of the ledger against an
                        // existing store would otherwise abort on a column
                        // that is already correct.
                        tracing::debug!(
                            "migration {:03}: skipping already-applied statement ({err})",
                            version
                        );
                    }
                    Err(err) => {
                        return Err(crate::error::StorageError::Database(format!(
                            "migration {version:03}_{name} failed on statement `{}`: {err}",
                            statement.chars().take(120).collect::<String>()
                        ))
                        .into())
                    }
                }
            }

            sqlx::query("INSERT INTO schema_migrations (version, name) VALUES (?, ?)")
                .bind(*version)
                .bind(*name)
                .execute(&self.pool)
                .await
                .map_err(storage_error("record schema migration"))?;
        }
        Ok(())
    }
}

/// Migrations, embedded at compile time so a deployed binary carries its own
/// schema and cannot be run against a stale checkout of `migrations/`.
const MIGRATIONS: &[(i64, &str, &str)] = &[
    (
        1,
        "initial_schema",
        include_str!("../../migrations/001_initial_schema.sql"),
    ),
    (
        2,
        "focus_schema",
        include_str!("../../migrations/002_focus_schema.sql"),
    ),
    (
        3,
        "filter_mount_schema",
        include_str!("../../migrations/003_filter_mount_schema.sql"),
    ),
    (
        4,
        "frame_assets",
        include_str!("../../migrations/004_frame_assets.sql"),
    ),
    (
        5,
        "focuser_scale",
        include_str!("../../migrations/005_focuser_scale.sql"),
    ),
    (
        6,
        "environment_log",
        include_str!("../../migrations/006_environment_log.sql"),
    ),
    (
        7,
        "grouping_keys",
        include_str!("../../migrations/007_grouping_keys.sql"),
    ),
    (
        8,
        "plate_solve_provenance",
        include_str!("../../migrations/008_plate_solve_provenance.sql"),
    ),
    (
        9,
        "equipment_layer",
        include_str!("../../migrations/009_equipment_layer.sql"),
    ),
];

/// True for errors that mean "this statement's effect is already present".
///
/// Narrow on purpose: only the two SQLite messages that indicate an additive
/// change already exists. Anything else is a real failure and aborts startup,
/// because a half-applied schema is worse than refusing to start.
fn is_benign(err: &sqlx::Error) -> bool {
    let msg = err.to_string().to_ascii_lowercase();
    msg.contains("duplicate column name") || msg.contains("already exists")
}

/// Split a migration file into statements on `;` at end of line.
///
/// Sufficient for this schema — it contains no triggers, no `BEGIN…END`
/// blocks, and no semicolons inside string literals. Line comments are
/// stripped so a `--` containing a semicolon cannot split a statement.
fn split_statements(sql: &str) -> Vec<String> {
    let mut statements = Vec::new();
    let mut current = String::new();
    for line in sql.lines() {
        let code = match line.find("--") {
            Some(idx) => &line[..idx],
            None => line,
        };
        if code.trim().is_empty() {
            continue;
        }
        current.push_str(code);
        current.push('\n');
        if code.trim_end().ends_with(';') {
            let trimmed = current.trim().trim_end_matches(';').trim().to_string();
            if !trimmed.is_empty() {
                statements.push(trimmed);
            }
            current.clear();
        }
    }
    let tail = current.trim().trim_end_matches(';').trim().to_string();
    if !tail.is_empty() {
        statements.push(tail);
    }
    statements
}

/// Migration list for tests that need a schema in an in-memory database.
/// Exposed so other modules' tests build their fixture from the *same*
/// migrations the daemon runs, rather than a hand-copied CREATE TABLE that
/// silently drifts (which is how the schema split in the first place).
#[cfg(test)]
pub(crate) fn migrations_for_test() -> &'static [(i64, &'static str, &'static str)] {
    MIGRATIONS
}

#[cfg(test)]
pub(crate) fn split_statements_for_test(sql: &str) -> Vec<String> {
    split_statements(sql)
}

#[cfg(test)]
mod tests {
    use super::MIGRATIONS;

    /// Every `migrations/*.sql` file must appear in `MIGRATIONS`.
    ///
    /// `006_environment_log.sql` sat on disk for a week without being listed
    /// here, so the daemon never created `environment_log` or
    /// `disturbance_log`. It went unnoticed because `tools/import_session.py`
    /// applies that one file itself, so the importer's own tests passed against
    /// tables the daemon would never have. Adding a migration is two edits in
    /// two directories and nothing connected them; now something does.
    #[test]
    fn every_migration_file_is_registered() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations");
        let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
            .expect("migrations directory")
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                (path.extension()? == "sql").then(|| path.file_name()?.to_str().map(str::to_string))?
            })
            .collect();
        on_disk.sort();

        let mut registered: Vec<String> = MIGRATIONS
            .iter()
            .map(|(version, name, _)| format!("{version:03}_{name}.sql"))
            .collect();
        registered.sort();

        assert_eq!(
            on_disk, registered,
            "migrations/ and the MIGRATIONS ledger disagree. A file present on \
             disk but absent here never runs; the reverse will not compile."
        );
    }

    /// Versions are the ledger's primary key and its ordering.
    #[test]
    fn migration_versions_are_unique_and_ascending() {
        let versions: Vec<i64> = MIGRATIONS.iter().map(|(version, _, _)| *version).collect();
        let mut sorted = versions.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(versions, sorted, "migration versions must be unique and ascending");
    }
}

/// Migration 009 (equipment layer, spec §4.3) on a fresh database and on one
/// already at 008, and the `equipment_overheads` view against the event
/// serialisation the recorder will write.
#[cfg(test)]
mod equipment_migration_tests {
    use super::{split_statements, LocalStore, MIGRATIONS};
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
    use sqlx::SqlitePool;

    const NEW_TABLES: [&str; 9] = [
        "device_connections",
        "equipment_events",
        "exposure_device_windows",
        "focuser_moves",
        "guide_steps",
        "dither_events",
        "pointing_samples",
        "policy_decisions",
        "device_profiles",
    ];

    fn temp_db(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("achernar-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join("achernar.db");
        (dir, db)
    }

    async fn names(pool: &SqlitePool, sql: &str) -> Vec<String> {
        sqlx::query_as::<_, (String,)>(sql)
            .fetch_all(pool)
            .await
            .unwrap()
            .into_iter()
            .map(|(name,)| name)
            .collect()
    }

    async fn columns(pool: &SqlitePool, table: &str) -> Vec<String> {
        names(
            pool,
            &format!("SELECT name FROM pragma_table_info('{table}')"),
        )
        .await
    }

    async fn assert_equipment_schema(pool: &SqlitePool) {
        let tables = names(pool, "SELECT name FROM sqlite_master WHERE type = 'table'").await;
        for table in NEW_TABLES {
            assert!(tables.iter().any(|t| t == table), "missing table {table}");
        }
        let views = names(pool, "SELECT name FROM sqlite_master WHERE type = 'view'").await;
        assert!(views.iter().any(|v| v == "equipment_overheads"));
        for (table, column) in [
            ("focus_samples", "position_readback_before"),
            ("focus_samples", "backlash_model"),
            ("focus_actions", "verification_frame_id"),
            ("filter_switch_events", "handshake_ok"),
            ("filter_switch_events", "background_after"),
            ("guider_state_log", "rms0_total_arcsec"),
            ("environment_log", "sensor_ages_json"),
            ("environment_log", "device_identity_key"),
            ("focuser_profiles", "identity_key"),
            ("control_audit_logs", "command_id"),
            ("control_audit_logs", "stage"),
        ] {
            assert!(
                columns(pool, table).await.iter().any(|c| c == column),
                "missing {table}.{column}"
            );
        }
        let versions: Vec<(i64,)> =
            sqlx::query_as("SELECT version FROM schema_migrations ORDER BY version")
                .fetch_all(pool)
                .await
                .unwrap();
        let versions: Vec<i64> = versions.into_iter().map(|(v,)| v).collect();
        assert_eq!(versions, (1..=9).collect::<Vec<i64>>());
    }

    #[tokio::test]
    async fn migration_009_applies_on_a_fresh_database() {
        let (dir, db) = temp_db("009-fresh");
        let store = LocalStore::open(db.to_str().unwrap()).await.unwrap();
        let pool = store.pool();
        assert_equipment_schema(pool).await;

        // The overheads view reads the payload exactly as events.rs writes it.
        use crate::equipment::events::{EquipmentEvent, EventKind};
        use crate::equipment::ops::OpKind;
        use crate::equipment::testing::manual_clock::ManualClock;
        use crate::equipment::types::{CommandId, CommandSource, OpOutcome};
        let clock = ManualClock::new();
        let command = CommandId::new();
        let commanded = EquipmentEvent::new(
            EventKind::OperationCommanded {
                op: OpKind::FilterHandshake,
                source: CommandSource::Operator { overriding: false },
            },
            &clock,
        )
        .with_command(command);
        clock.advance_ms(4250);
        let ended = EquipmentEvent::new(
            EventKind::OperationEnded {
                op: OpKind::FilterHandshake,
                outcome: OpOutcome::Succeeded,
                duration_ms: 4250,
            },
            &clock,
        )
        .with_command(command);
        for (seq, event) in [(1_i64, &commanded), (2, &ended)] {
            let value = serde_json::to_value(event).unwrap();
            sqlx::query(
                "INSERT INTO equipment_events \
                 (seq, event_id, at_utc, mono_ns, device_kind, command_id, kind, payload_json) \
                 VALUES (?, ?, ?, ?, 'filter_wheel', ?, ?, ?)",
            )
            .bind(seq)
            .bind(event.event_id.to_string())
            .bind(
                event
                    .at_utc
                    .to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            )
            .bind(event.mono_ns as i64)
            .bind(command.to_string())
            .bind(value["kind"].as_str().unwrap())
            .bind(value["payload"].to_string())
            .execute(pool)
            .await
            .unwrap();
        }
        let row: (String, String, String, String, f64) = sqlx::query_as(
            "SELECT command_id, operation, outcome, started_at, duration_s \
             FROM equipment_overheads",
        )
        .fetch_one(pool)
        .await
        .unwrap();
        assert_eq!(row.0, command.to_string());
        assert_eq!(row.1, "filter_handshake");
        assert_eq!(row.2, "succeeded");
        assert_eq!(row.3, "2026-01-01T00:00:00.000Z");
        assert!((row.4 - 4.25).abs() < 1e-9);

        // Reopening is a no-op.
        drop(store);
        let store = LocalStore::open(db.to_str().unwrap()).await.unwrap();
        assert_equipment_schema(store.pool()).await;
        drop(store);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn migration_009_applies_on_a_database_already_at_008() {
        let (dir, db) = temp_db("009-from-008");
        {
            let options = SqliteConnectOptions::new()
                .filename(&db)
                .create_if_missing(true)
                .foreign_keys(true);
            let pool = SqlitePoolOptions::new()
                .max_connections(1)
                .connect_with(options)
                .await
                .unwrap();
            sqlx::query(
                "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, \
                 name TEXT NOT NULL, applied_at TEXT NOT NULL DEFAULT (datetime('now')))",
            )
            .execute(&pool)
            .await
            .unwrap();
            for (version, name, sql) in MIGRATIONS.iter().filter(|(v, _, _)| *v <= 8) {
                for statement in split_statements(sql) {
                    sqlx::query(&statement).execute(&pool).await.unwrap();
                }
                sqlx::query("INSERT INTO schema_migrations (version, name) VALUES (?, ?)")
                    .bind(*version)
                    .bind(*name)
                    .execute(&pool)
                    .await
                    .unwrap();
            }
            for statement in [
                "INSERT INTO focus_runs (id, rig_id, start_position, state, started_at) \
                 VALUES ('r1', 'rig', 100, 'converged', '2026-10-01T00:00:00Z')",
                "INSERT INTO focus_samples (run_id, position, hfr, captured_at) \
                 VALUES ('r1', 100, 2.5, '2026-10-01T00:00:01Z')",
                "INSERT INTO environment_log (observed_at, ambient_temp_c) \
                 VALUES ('2026-10-01T00:00:00Z', 4.5)",
                "INSERT INTO control_audit_logs (action) VALUES ('legacy')",
            ] {
                sqlx::query(statement).execute(&pool).await.unwrap();
            }
            pool.close().await;
        }

        let store = LocalStore::open(db.to_str().unwrap()).await.unwrap();
        let pool = store.pool();
        assert_equipment_schema(pool).await;
        let sample: (i64, Option<i64>, Option<String>) = sqlx::query_as(
            "SELECT position, position_readback_before, position_source FROM focus_samples",
        )
        .fetch_one(pool)
        .await
        .unwrap();
        assert_eq!(sample, (100, None, None), "existing rows keep their values");
        let env: (f64, Option<f64>) =
            sqlx::query_as("SELECT ambient_temp_c, pressure_hpa FROM environment_log")
                .fetch_one(pool)
                .await
                .unwrap();
        assert_eq!(env, (4.5, None));
        let audit: (String, Option<String>) =
            sqlx::query_as("SELECT action, command_id FROM control_audit_logs")
                .fetch_one(pool)
                .await
                .unwrap();
        assert_eq!(audit, ("legacy".to_string(), None));
        drop(store);
        let _ = std::fs::remove_dir_all(dir);
    }
}

fn storage_error(context: &'static str) -> impl Fn(sqlx::Error) -> crate::error::AchernarError {
    move |err| crate::error::StorageError::Database(format!("{context}: {err}")).into()
}
