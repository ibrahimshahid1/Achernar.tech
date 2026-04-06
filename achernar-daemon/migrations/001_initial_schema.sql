-- Migration 001: Initial Schema
-- Sections 5.2: Sessions, frames, and telemetry logging.
-- Run at daemon startup.

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

CREATE TABLE IF NOT EXISTS frames (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(id),
    frame_type      TEXT NOT NULL DEFAULT 'light',  -- light, dark, flat, bias, dark_flat
    filter_name     TEXT,
    exposure_s      REAL NOT NULL,
    gain            INTEGER,
    offset_val      INTEGER,
    bin_x           INTEGER DEFAULT 1,
    bin_y           INTEGER DEFAULT 1,
    temperature     REAL,
    hfr             REAL,
    fwhm            REAL,
    star_count      INTEGER,
    background_adu  REAL,
    file_path       TEXT,
    width           INTEGER,
    height          INTEGER,
    captured_at     TEXT NOT NULL DEFAULT (datetime('now')),
    quality_score   REAL,
    cloud_prob      REAL,
    accepted        INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS telemetry_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL DEFAULT (datetime('now')),
    source          TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload         TEXT NOT NULL,  -- JSON
    level           TEXT DEFAULT 'info'
);

CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id);
CREATE INDEX IF NOT EXISTS idx_frames_captured ON frames(captured_at);
CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_telemetry_source ON telemetry_log(source);
