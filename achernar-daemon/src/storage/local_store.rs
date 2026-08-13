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

fn storage_error(context: &'static str) -> impl Fn(sqlx::Error) -> crate::error::AchernarError {
    move |err| crate::error::StorageError::Database(format!("{context}: {err}")).into()
}
