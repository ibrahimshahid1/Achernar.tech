-- Migration 002 — rig/focuser configuration and focus model tracking.
--
-- This file previously redefined focus_runs, focus_samples, focus_predictions
-- and focus_actions with a *different* shape from the ones the daemon actually
-- creates (TEXT ids vs AUTOINCREMENT integers, `ended_at` vs `completed_at`,
-- `action_type` vs `action`). Because nothing executed these files the
-- conflict was invisible; the moment they ran, `CREATE TABLE IF NOT EXISTS`
-- would silently keep whichever definition landed first, so the schema you got
-- depended on file ordering. Those four tables now live in 001 only, defined
-- as the code actually uses them.
--
-- What remains here are the tables 001 does not define.

CREATE TABLE IF NOT EXISTS rigs (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    telescope       TEXT,
    camera          TEXT,
    focuser         TEXT,
    filter_wheel    TEXT,
    focal_length_mm REAL,
    focal_ratio     REAL,
    pixel_size_um   REAL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS focuser_profiles (
    id               TEXT PRIMARY KEY,
    rig_id           TEXT NOT NULL REFERENCES rigs(id),
    focuser_name     TEXT NOT NULL,
    min_position     INTEGER NOT NULL DEFAULT 0,
    max_position     INTEGER NOT NULL DEFAULT 50000,
    step_size        INTEGER DEFAULT 50,
    backlash_steps   INTEGER DEFAULT 0,
    backlash_dir     TEXT DEFAULT 'outward',
    temp_coefficient REAL,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS focuser_filter_offsets (
    id              TEXT PRIMARY KEY,
    profile_id      TEXT NOT NULL REFERENCES focuser_profiles(id),
    filter_name     TEXT NOT NULL,
    offset_steps    INTEGER NOT NULL DEFAULT 0,
    UNIQUE(profile_id, filter_name)
);

-- Focus-drift forecaster (FDF) decisions, one row per evaluation.
CREATE TABLE IF NOT EXISTS focus_drift_events (
    id                TEXT PRIMARY KEY,
    session_id        TEXT REFERENCES sessions(id),
    predicted_drift   REAL NOT NULL,
    temperature       REAL,
    confidence        REAL,
    refocus_triggered INTEGER DEFAULT 0,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS focus_model_registry (
    id              TEXT PRIMARY KEY,
    model_name      TEXT NOT NULL,
    version         TEXT NOT NULL,
    task            TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    is_active       INTEGER NOT NULL DEFAULT 1,
    metadata_json   TEXT,  -- Full metadata.json contents
    deployed_at     TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(model_name, version)
);

CREATE INDEX IF NOT EXISTS idx_focus_drift_session ON focus_drift_events(session_id);
