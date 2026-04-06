-- Migration 002: Focus Schema
-- Section 7.4: All focus-related tables.

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
    id              TEXT PRIMARY KEY,
    rig_id          TEXT NOT NULL REFERENCES rigs(id),
    focuser_name    TEXT NOT NULL,
    min_position    INTEGER NOT NULL DEFAULT 0,
    max_position    INTEGER NOT NULL DEFAULT 50000,
    step_size       INTEGER DEFAULT 50,
    backlash_steps  INTEGER DEFAULT 0,
    backlash_dir    TEXT DEFAULT 'outward',
    temp_coefficient REAL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS focuser_filter_offsets (
    id              TEXT PRIMARY KEY,
    profile_id      TEXT NOT NULL REFERENCES focuser_profiles(id),
    filter_name     TEXT NOT NULL,
    offset_steps    INTEGER NOT NULL DEFAULT 0,
    UNIQUE(profile_id, filter_name)
);

CREATE TABLE IF NOT EXISTS focus_runs (
    id              TEXT PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(id),
    rig_id          TEXT REFERENCES rigs(id),
    filter_name     TEXT,
    method          TEXT NOT NULL DEFAULT 'hybrid',  -- classical, ml, hybrid
    start_position  INTEGER NOT NULL,
    end_position    INTEGER,
    best_position   INTEGER,
    best_hfr        REAL,
    temperature     REAL,
    state           TEXT NOT NULL DEFAULT 'initializing',
    total_steps     INTEGER DEFAULT 0,
    total_travel    INTEGER DEFAULT 0,
    duration_ms     INTEGER,
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at        TEXT,
    error_message   TEXT
);

CREATE TABLE IF NOT EXISTS focus_samples (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES focus_runs(id),
    position        INTEGER NOT NULL,
    hfr             REAL NOT NULL,
    fwhm            REAL,
    star_count      INTEGER,
    quality_score   REAL,
    temperature     REAL,
    sample_index    INTEGER NOT NULL,
    captured_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS focus_predictions (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES focus_runs(id),
    model_name      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    predicted_offset REAL NOT NULL,
    uncertainty     REAL,
    confidence      REAL,
    features_json   TEXT,  -- JSON blob of input features
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS focus_actions (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES focus_runs(id),
    action_type     TEXT NOT NULL,  -- move, measure, evaluate
    proposed_step   INTEGER,
    actual_step     INTEGER,
    source          TEXT NOT NULL,  -- classical, ml, hybrid, fallback
    confidence      REAL,
    safety_verdict  TEXT NOT NULL,  -- approved, modified, rejected
    executed        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS focus_drift_events (
    id              TEXT PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(id),
    predicted_drift REAL NOT NULL,
    temperature     REAL,
    confidence      REAL,
    refocus_triggered INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
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

CREATE INDEX IF NOT EXISTS idx_focus_runs_session ON focus_runs(session_id);
CREATE INDEX IF NOT EXISTS idx_focus_samples_run ON focus_samples(run_id);
CREATE INDEX IF NOT EXISTS idx_focus_predictions_run ON focus_predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_focus_actions_run ON focus_actions(run_id);
CREATE INDEX IF NOT EXISTS idx_focus_drift_session ON focus_drift_events(session_id);
