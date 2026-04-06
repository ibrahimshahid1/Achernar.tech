-- Migration 003: Filter & Mount Schema
-- Section 8.4: Filter switch events, guider state, plate solves, pointing model.

CREATE TABLE IF NOT EXISTS filter_switch_events (
    id              TEXT PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(id),
    from_filter     TEXT,
    to_filter       TEXT NOT NULL,
    focus_offset    INTEGER,
    duration_ms     INTEGER,
    success         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS guider_state_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT REFERENCES sessions(id),
    ra_rms          REAL,
    dec_rms         REAL,
    total_rms       REAL,
    ra_aggressiveness REAL,
    dec_aggressiveness REAL,
    ra_correction   REAL,
    dec_correction  REAL,
    guide_star_snr  REAL,
    dither_active   INTEGER DEFAULT 0,
    timestamp       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS plate_solve_events (
    id              TEXT PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(id),
    solved_ra       REAL,
    solved_dec      REAL,
    rotation_deg    REAL,
    pixel_scale     REAL,
    pointing_error_arcmin REAL,
    solve_time_ms   INTEGER,
    solver          TEXT,
    success         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pointing_model_snapshots (
    id              TEXT PRIMARY KEY,
    rig_id          TEXT REFERENCES rigs(id),
    sample_count    INTEGER NOT NULL,
    rms_error_arcmin REAL,
    model_data      TEXT NOT NULL,  -- JSON serialized thin-plate spline coefficients
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_filter_switch_session ON filter_switch_events(session_id);
CREATE INDEX IF NOT EXISTS idx_guider_state_session ON guider_state_log(session_id);
CREATE INDEX IF NOT EXISTS idx_guider_state_time ON guider_state_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_plate_solve_session ON plate_solve_events(session_id);
