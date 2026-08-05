-- Migration 001 — baseline schema.
--
-- AUTHORITY NOTE. Until now this directory was decorative: nothing ever
-- executed these files. The schema that actually existed at runtime was a
-- `const SCHEMA: &[&str]` inside `src/storage/local_store.rs`, and the two had
-- drifted — most visibly `frames`, which was defined here with columns
-- (frame_type, file_path, hfr, accepted, …) that the running database has
-- never had, while the live table has columns (local_path, wcs_json,
-- quality_json, …) that were absent here. Reading this directory to learn the
-- schema gave you the wrong answer.
--
-- This file is now the baseline and reflects exactly what the daemon creates,
-- so an existing database and a fresh one converge. `local_store.rs` executes
-- these files in order and records them in `schema_migrations`.
--
-- Every statement is idempotent (IF NOT EXISTS), so applying the baseline over
-- a database created by the previous embedded schema is a no-op.

-- ─── Observing sessions ───
-- A capture belongs to a session; a session belongs to a rig and a night.
-- That grouping is what makes leakage-safe train/test splits possible later,
-- so it belongs in the baseline rather than being reconstructed at export.
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    name            TEXT,
    rig_id          TEXT,
    target_name     TEXT,
    target_ra       REAL,
    target_dec      REAL,
    location_lat    REAL,
    location_lon    REAL,
    location_alt    REAL,
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at        TEXT,
    notes           TEXT,
    status          TEXT NOT NULL DEFAULT 'active'  -- active, completed, aborted
);

-- ─── Autofocus ───
CREATE TABLE IF NOT EXISTS focus_runs (
    id              TEXT PRIMARY KEY,
    rig_id          TEXT NOT NULL,
    filter_name     TEXT,
    start_position  INTEGER NOT NULL,
    best_position   INTEGER,
    best_hfr        REAL,
    state           TEXT NOT NULL,
    temperature     REAL,
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS focus_samples (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    position        INTEGER NOT NULL,
    hfr             REAL NOT NULL,
    hfd             REAL,
    fwhm            REAL,
    star_count      INTEGER NOT NULL DEFAULT 0,
    quality_score   REAL,
    temperature     REAL,
    captured_at     TEXT NOT NULL,
    frame_id        TEXT,
    FOREIGN KEY (run_id) REFERENCES focus_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS focus_actions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT,
    action          TEXT NOT NULL,
    from_position   INTEGER,
    to_position     INTEGER,
    steps           INTEGER,
    reason          TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (run_id) REFERENCES focus_runs(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS focus_predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT,
    model_name        TEXT,
    model_version     TEXT,
    predicted_offset  INTEGER,
    confidence        REAL,
    payload_json      TEXT NOT NULL DEFAULT '{}',
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (run_id) REFERENCES focus_runs(id) ON DELETE SET NULL
);

-- ─── Frames ───
-- Migration 004 widens this into a per-exposure capture record and moves the
-- file bytes out into `frame_assets`. `local_path` / `checksum_sha256` stay
-- as aliases of the analysis_luma asset because the FQE dataset builder still
-- reads them.
CREATE TABLE IF NOT EXISTS frames (
    id              TEXT PRIMARY KEY,
    session_id      TEXT,
    local_path      TEXT NOT NULL,
    object_key      TEXT,
    checksum_sha256 TEXT,
    width           INTEGER,
    height          INTEGER,
    exposure_s      REAL,
    gain            REAL,
    filter_name     TEXT,
    temperature     REAL,
    wcs_json        TEXT NOT NULL DEFAULT '{}',
    quality_json    TEXT NOT NULL DEFAULT '{}',
    captured_at     TEXT NOT NULL,
    synced_at       TEXT
);

-- ─── Equipment, sync, audit ───
CREATE TABLE IF NOT EXISTS equipment_profiles (
    id              TEXT PRIMARY KEY,
    equipment_type  TEXT NOT NULL,
    name            TEXT NOT NULL,
    driver          TEXT NOT NULL,
    options_json    TEXT NOT NULL DEFAULT '{}',
    last_seen_at    TEXT,
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sync_queue (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type     TEXT NOT NULL,
    entity_id       TEXT NOT NULL,
    operation       TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    synced_at       TEXT
);

CREATE TABLE IF NOT EXISTS control_audit_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    actor           TEXT NOT NULL DEFAULT 'local',
    action          TEXT NOT NULL,
    target          TEXT,
    payload_json    TEXT NOT NULL DEFAULT '{}',
    accepted        INTEGER NOT NULL DEFAULT 1,
    reason          TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS telemetry_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL DEFAULT (datetime('now')),
    source          TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload         TEXT NOT NULL,  -- JSON
    level           TEXT DEFAULT 'info'
);

CREATE INDEX IF NOT EXISTS focus_samples_run_idx          ON focus_samples(run_id);
CREATE INDEX IF NOT EXISTS frames_session_idx             ON frames(session_id, captured_at);
CREATE INDEX IF NOT EXISTS sync_queue_pending_idx         ON sync_queue(synced_at, created_at);
CREATE INDEX IF NOT EXISTS control_audit_logs_created_idx ON control_audit_logs(created_at);
CREATE INDEX IF NOT EXISTS telemetry_timestamp_idx        ON telemetry_log(timestamp);
