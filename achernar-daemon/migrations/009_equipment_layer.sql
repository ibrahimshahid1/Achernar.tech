-- ─────────────────────────────────────────────────────────────────────────────
-- 009_equipment_layer.sql
--
-- Ground truth for the native equipment layer (equipment spec §4.3). Every
-- Achernar v6 model that needs device state ties it to event times; these
-- tables are where those times, states and outcomes are kept.
--
-- Additive only: new tables, and `ALTER TABLE … ADD COLUMN` on existing ones.
-- No earlier migration is edited. The repositories (WP1A-9) write the rows;
-- the record structs in `src/equipment/events.rs` name every column.
--
-- Conventions:
--   *_at, *_utc     RFC 3339 UTC text with milliseconds
--   mono_ns         nanoseconds since daemon start (monotonic, per process)
--   booleans        INTEGER 0/1; NULL means unknown, never false
--   *_json          JSON text
-- A missing value is NULL, never a synthesised 0.
-- ─────────────────────────────────────────────────────────────────────────────

-- ─── One row per device connection ───
CREATE TABLE IF NOT EXISTS device_connections (
    id                TEXT PRIMARY KEY,          -- connection_id
    kind              TEXT NOT NULL,             -- camera | mount | focuser | ...
    driver            TEXT NOT NULL,             -- alpaca | indi | phd2 | direct | sim | ols_*
    identity_key      TEXT NOT NULL,
    unique_id         TEXT,
    name              TEXT,
    description       TEXT,
    driver_info       TEXT,
    driver_version    TEXT,
    interface_version INTEGER,
    endpoint          TEXT,
    capabilities_json TEXT NOT NULL DEFAULT '{}',
    qualification     TEXT NOT NULL DEFAULT 'implemented_unqualified',
    connected_at      TEXT NOT NULL,
    disconnected_at   TEXT,
    end_class         TEXT,                      -- operator | lost | shutdown | replaced | failed
    session_id        TEXT,
    night_id          TEXT
);

CREATE INDEX IF NOT EXISTS device_connections_identity_idx
    ON device_connections(identity_key, connected_at);

-- ─── The equipment event journal ───
-- `seq` is assigned by the hub and continues from MAX(seq) when a store is
-- attached. `guide_step` events are not journaled here; they go to
-- guide_steps.
CREATE TABLE IF NOT EXISTS equipment_events (
    seq           INTEGER PRIMARY KEY,
    event_id      TEXT NOT NULL UNIQUE,
    at_utc        TEXT NOT NULL,
    mono_ns       INTEGER NOT NULL,
    device_kind   TEXT,
    connection_id TEXT,
    command_id    TEXT,
    kind          TEXT NOT NULL,
    payload_json  TEXT NOT NULL DEFAULT '{}',
    device_time   TEXT,
    session_id    TEXT,
    night_id      TEXT
);

CREATE INDEX IF NOT EXISTS equipment_events_time_idx    ON equipment_events(at_utc);
CREATE INDEX IF NOT EXISTS equipment_events_kind_idx    ON equipment_events(kind, at_utc);
CREATE INDEX IF NOT EXISTS equipment_events_command_idx ON equipment_events(command_id);

-- ─── Device state over each committed exposure (H1) ───
CREATE TABLE IF NOT EXISTS exposure_device_windows (
    frame_id                 TEXT PRIMARY KEY REFERENCES frames(id) ON DELETE CASCADE,
    exposure_started_at      TEXT,
    exposure_completed_at    TEXT,
    clock_quality            TEXT NOT NULL DEFAULT 'unknown',
    finalised                INTEGER NOT NULL DEFAULT 0,
    mount_start_json         TEXT,
    mount_mid_json           TEXT,
    mount_end_json           TEXT,
    mount_frame              TEXT,               -- j2000 | jnow_mean | jnow_apparent | unknown
    pier_side                TEXT,               -- east | west | NULL
    tracking                 INTEGER,            -- 1 only if tracking throughout
    slewing_any              INTEGER,
    pulse_guiding_any        INTEGER,
    focuser_pos_start        INTEGER,
    focuser_pos_end          INTEGER,
    focuser_moved            INTEGER,
    focuser_temperature_c    REAL,
    filter_slot              INTEGER,
    filter_name              TEXT,
    filter_verified          INTEGER,
    rotator_angle_deg        REAL,
    guiding_state            TEXT,
    guide_rms_ra_px          REAL,
    guide_rms_dec_px         REAL,
    guide_rms_total_arcsec   REAL,
    guide_rms0_total_arcsec  REAL,               -- RMS about zero
    guide_peak_ra_px         REAL,
    guide_peak_dec_px        REAL,
    guide_steps_n            INTEGER,
    dither_in_window         INTEGER,
    settle_in_window         INTEGER,
    safety_state             TEXT,               -- none | safe | unsafe | unknown
    weather_json             TEXT,
    cover_json               TEXT,
    resume_block_active      INTEGER NOT NULL DEFAULT 0,
    quiescent                INTEGER,
    quiescence_reason        TEXT,
    events_json              TEXT NOT NULL DEFAULT '[]',   -- equipment_events.seq list
    created_at               TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ─── One row per focuser move, with read-back ───
CREATE TABLE IF NOT EXISTS focuser_moves (
    id                  TEXT PRIMARY KEY,
    command_id          TEXT,
    connection_id       TEXT NOT NULL,
    mode                TEXT NOT NULL,           -- abs | rel
    commanded_target    INTEGER NOT NULL,
    logical_start       INTEGER,
    logical_end         INTEGER,
    raw_start           INTEGER,
    raw_end             INTEGER,
    position_source     TEXT NOT NULL,           -- device | emulated | emulated_restored | ...
    direction           TEXT,                    -- in | out
    reversal            INTEGER,
    backlash_model      TEXT,                    -- JSON: {"model": "none" | "overshoot" | "absolute", ...}
    backlash_owner      TEXT,                    -- achernar | firmware
    tempcomp_before     INTEGER,
    tempcomp_after      INTEGER,
    temperature_start_c REAL,
    temperature_end_c   REAL,
    commanded_at        TEXT NOT NULL,
    started_at          TEXT,
    stopped_at          TEXT,
    settled_at          TEXT,
    outcome             TEXT,                    -- succeeded | failed | aborted | timed_out | unknown
    source              TEXT,
    run_id              TEXT
);

CREATE INDEX IF NOT EXISTS focuser_moves_time_idx    ON focuser_moves(commanded_at);
CREATE INDEX IF NOT EXISTS focuser_moves_command_idx ON focuser_moves(command_id);
CREATE INDEX IF NOT EXISTS focuser_moves_run_idx     ON focuser_moves(run_id);

-- ─── Raw PHD2 guide steps, unmodified (PHD2 signs and times) ───
CREATE TABLE IF NOT EXISTS guide_steps (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    connection_id      TEXT NOT NULL,
    frame              INTEGER NOT NULL,
    phd2_time_s        REAL NOT NULL,
    phd2_timestamp     REAL,
    received_at_utc    TEXT NOT NULL,
    mount              TEXT,
    dx                 REAL NOT NULL,
    dy                 REAL NOT NULL,
    ra_raw             REAL,
    dec_raw            REAL,
    ra_guide           REAL,
    dec_guide          REAL,
    ra_duration_ms     INTEGER,
    ra_direction       TEXT,
    dec_duration_ms    INTEGER,
    dec_direction      TEXT,
    star_mass          REAL,
    snr                REAL,
    hfd                REAL,
    avg_dist           REAL,
    ra_limited         INTEGER,
    dec_limited        INTEGER,
    error_code         INTEGER,
    phase              TEXT NOT NULL,            -- guiding | dithering | settling | paused | recovering
    pixel_scale_arcsec REAL
);

CREATE INDEX IF NOT EXISTS guide_steps_time_idx       ON guide_steps(received_at_utc);
CREATE INDEX IF NOT EXISTS guide_steps_connection_idx ON guide_steps(connection_id, frame);

-- ─── Dithers and their settle outcome (H10) ───
CREATE TABLE IF NOT EXISTS dither_events (
    id                  TEXT PRIMARY KEY,
    command_id          TEXT,
    source              TEXT,
    policy_version      TEXT,
    backend             TEXT NOT NULL,           -- phd2 | direct
    amount              REAL NOT NULL,
    amount_unit         TEXT NOT NULL,           -- guide_px | main_px
    ra_only             INTEGER NOT NULL DEFAULT 0,
    settle_pixels       REAL,
    settle_time_s       REAL,
    settle_timeout_s    REAL,
    requested_at        TEXT NOT NULL,
    dx                  REAL,
    dy                  REAL,
    settle_done_at      TEXT,
    settle_status       INTEGER,
    settle_error        TEXT,
    total_frames        INTEGER,
    dropped_frames      INTEGER,
    max_distance_px     REAL,
    next_frame_id       TEXT,
    next_frame_accepted INTEGER
);

CREATE INDEX IF NOT EXISTS dither_events_time_idx ON dither_events(requested_at);

-- ─── Mount pointing residuals per solve (H12; M-4, M-3) ───
CREATE TABLE IF NOT EXISTS pointing_samples (
    id                          TEXT PRIMARY KEY,
    plate_solve_event_id        TEXT,
    frame_id                    TEXT,
    mount_identity_key          TEXT NOT NULL,
    mount_frame                 TEXT NOT NULL,
    jnow_flavour                TEXT,            -- mean | apparent
    does_refraction             INTEGER,
    mount_ra_native_h           REAL,
    mount_dec_native_deg        REAL,
    mount_ra_j2000_h            REAL,
    mount_dec_j2000_deg         REAL,
    solved_ra_j2000_h           REAL NOT NULL,
    solved_dec_j2000_deg        REAL NOT NULL,
    err_ra_arcsec               REAL,            -- multiplied by cos(dec)
    err_dec_arcsec              REAL,
    err_total_arcsec            REAL,
    pier_side                   TEXT,
    ha_h                        REAL,
    lst_h                       REAL,
    alt_deg                     REAL,
    az_deg                      REAL,
    sync_generation             INTEGER NOT NULL DEFAULT 0,
    sync_mode                   TEXT,            -- mount | offset
    coordinate_precision_arcsec REAL,
    temperature_c               REAL,
    mid_exposure_utc            TEXT,
    purpose                     TEXT NOT NULL,   -- center | auto_solve | manual
    created_at                  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS pointing_samples_frame_idx ON pointing_samples(frame_id);
CREATE INDEX IF NOT EXISTS pointing_samples_mount_idx ON pointing_samples(mount_identity_key, created_at);

-- ─── Policy decision records (H23; DC §7 "Decision") ───
CREATE TABLE IF NOT EXISTS policy_decisions (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id              TEXT NOT NULL UNIQUE,
    policy                   TEXT NOT NULL,
    policy_version           TEXT NOT NULL,
    mode                     TEXT NOT NULL,      -- manual | advisory | proposal | bounded
    context_json             TEXT NOT NULL DEFAULT '{}',
    measurement_ages_json    TEXT NOT NULL DEFAULT '{}',
    safe_arms_json           TEXT NOT NULL DEFAULT '[]',
    chosen_arm               TEXT,
    propensities_json        TEXT,               -- NULL when the choice was deterministic
    requested_action_json    TEXT,
    validated_action_json    TEXT,
    executed_action_json     TEXT,
    command_id               TEXT,
    reward_components_json   TEXT,
    attribution_window_start TEXT,
    attribution_window_end   TEXT,
    interrupted              INTEGER NOT NULL DEFAULT 0,
    override                 INTEGER NOT NULL DEFAULT 0,
    parent_frame_ids_json    TEXT NOT NULL DEFAULT '[]',
    created_at               TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS policy_decisions_policy_idx  ON policy_decisions(policy, created_at);
CREATE INDEX IF NOT EXISTS policy_decisions_command_idx ON policy_decisions(command_id);

-- ─── Identity-keyed device profiles (D17) ───
-- Backlash and owner, settle times, travel bounds, filter names/offsets/
-- throughput/bandwidth, the relative-focuser position and its close state,
-- rotator sync offset, site-sync choices, switch roles, temperature role,
-- optics, the last-used option and qualification records.
CREATE TABLE IF NOT EXISTS device_profiles (
    identity_key TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    profile_json TEXT NOT NULL DEFAULT '{}',
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ─── Extensions of existing tables ───

-- Focus samples: A06 freshness and provenance (H3, H7).
ALTER TABLE focus_samples ADD COLUMN position_readback_before INTEGER;
ALTER TABLE focus_samples ADD COLUMN position_readback_after INTEGER;
ALTER TABLE focus_samples ADD COLUMN position_source TEXT;
ALTER TABLE focus_samples ADD COLUMN focuser_move_id TEXT;
ALTER TABLE focus_samples ADD COLUMN settled_at TEXT;
ALTER TABLE focus_samples ADD COLUMN exposure_started_at TEXT;
ALTER TABLE focus_samples ADD COLUMN capture_id TEXT;
ALTER TABLE focus_samples ADD COLUMN clock_quality TEXT;
ALTER TABLE focus_samples ADD COLUMN moved_during_exposure INTEGER;
ALTER TABLE focus_samples ADD COLUMN filter_verified INTEGER;
ALTER TABLE focus_samples ADD COLUMN temperature_source TEXT;
ALTER TABLE focus_samples ADD COLUMN temperature_age_ms INTEGER;
ALTER TABLE focus_samples ADD COLUMN tempcomp_state TEXT;
ALTER TABLE focus_samples ADD COLUMN backlash_model TEXT;

-- Focus actions: the verification frame after the final move (H4).
ALTER TABLE focus_actions ADD COLUMN verification_frame_id TEXT;
ALTER TABLE focus_actions ADD COLUMN verification_hfr REAL;
ALTER TABLE focus_actions ADD COLUMN verified INTEGER;
ALTER TABLE focus_actions ADD COLUMN focuser_move_id TEXT;

-- Filter switches: the v6 §8.2.1 handshake record (H8).
ALTER TABLE filter_switch_events ADD COLUMN command_id TEXT;
ALTER TABLE filter_switch_events ADD COLUMN from_slot INTEGER;
ALTER TABLE filter_switch_events ADD COLUMN to_slot INTEGER;
ALTER TABLE filter_switch_events ADD COLUMN requested_at TEXT;
ALTER TABLE filter_switch_events ADD COLUMN wheel_done_at TEXT;
ALTER TABLE filter_switch_events ADD COLUMN verified_at TEXT;
ALTER TABLE filter_switch_events ADD COLUMN settle_ms_actual INTEGER;
ALTER TABLE filter_switch_events ADD COLUMN offset_from INTEGER;
ALTER TABLE filter_switch_events ADD COLUMN offset_to INTEGER;
ALTER TABLE filter_switch_events ADD COLUMN offset_applied_steps INTEGER;
ALTER TABLE filter_switch_events ADD COLUMN focuser_move_id TEXT;
ALTER TABLE filter_switch_events ADD COLUMN guiding_paused INTEGER;
ALTER TABLE filter_switch_events ADD COLUMN handshake_ok INTEGER;
ALTER TABLE filter_switch_events ADD COLUMN outcome TEXT;
ALTER TABLE filter_switch_events ADD COLUMN trigger TEXT;
ALTER TABLE filter_switch_events ADD COLUMN policy_version TEXT;
ALTER TABLE filter_switch_events ADD COLUMN arm TEXT;
ALTER TABLE filter_switch_events ADD COLUMN propensity REAL;
ALTER TABLE filter_switch_events ADD COLUMN background_before REAL;
ALTER TABLE filter_switch_events ADD COLUMN background_after REAL;

-- Guider state: one row per exposure window and policy epoch (H9).
ALTER TABLE guider_state_log ADD COLUMN connection_id TEXT;
ALTER TABLE guider_state_log ADD COLUMN frame_id TEXT;
ALTER TABLE guider_state_log ADD COLUMN window_start TEXT;
ALTER TABLE guider_state_log ADD COLUMN window_end TEXT;
ALTER TABLE guider_state_log ADD COLUMN algorithm_params_json TEXT;
ALTER TABLE guider_state_log ADD COLUMN executed_arm_ra TEXT;
ALTER TABLE guider_state_log ADD COLUMN executed_arm_dec TEXT;
ALTER TABLE guider_state_log ADD COLUMN policy_version TEXT;
ALTER TABLE guider_state_log ADD COLUMN freeze_reason TEXT;
ALTER TABLE guider_state_log ADD COLUMN rms0_total_arcsec REAL;
ALTER TABLE guider_state_log ADD COLUMN steps_n INTEGER;

-- Environment: the remaining ObservingConditions fields and their ages,
-- written with source = 'probe' and the device identity.
ALTER TABLE environment_log ADD COLUMN pressure_hpa REAL;
ALTER TABLE environment_log ADD COLUMN cloud_cover_pct REAL;
ALTER TABLE environment_log ADD COLUMN sky_temp_c REAL;
ALTER TABLE environment_log ADD COLUMN sky_quality REAL;
ALTER TABLE environment_log ADD COLUMN sky_brightness_lux REAL;
ALTER TABLE environment_log ADD COLUMN star_fwhm_arcsec REAL;
ALTER TABLE environment_log ADD COLUMN rain_rate_mm_h REAL;
ALTER TABLE environment_log ADD COLUMN wind_speed_ms REAL;
ALTER TABLE environment_log ADD COLUMN wind_gust_ms REAL;
ALTER TABLE environment_log ADD COLUMN wind_dir_deg REAL;
ALTER TABLE environment_log ADD COLUMN sensor_ages_json TEXT;
ALTER TABLE environment_log ADD COLUMN device_identity_key TEXT;

-- Focuser profiles: keyed by device identity. Rows keyed by today's display
-- names are re-keyed on the first connect of a matching identity.
ALTER TABLE focuser_profiles ADD COLUMN identity_key TEXT;
CREATE INDEX IF NOT EXISTS focuser_profiles_identity_idx ON focuser_profiles(identity_key);

-- Control audit: one row per gate stage (H21). The full record is in
-- payload_json; these two columns make it joinable.
ALTER TABLE control_audit_logs ADD COLUMN command_id TEXT;
ALTER TABLE control_audit_logs ADD COLUMN stage TEXT;    -- proposed | validated | rejected | executed | outcome
CREATE INDEX IF NOT EXISTS control_audit_logs_command_idx ON control_audit_logs(command_id);

-- ─── Operation overheads (H18; FOR, FW-3, planning) ───
-- One row per finished operation, from its `operation_ended` event: filter
-- switches, autofocus runs, slew plus settle, flips, dithers, center loops.
CREATE VIEW IF NOT EXISTS equipment_overheads AS
SELECT
    e.command_id                                          AS command_id,
    json_extract(e.payload_json, '$.op')                  AS operation,
    json_extract(e.payload_json, '$.outcome.status')      AS outcome,
    e.device_kind                                         AS device_kind,
    (SELECT MIN(s.at_utc)
       FROM equipment_events s
      WHERE s.command_id = e.command_id)                  AS started_at,
    e.at_utc                                              AS ended_at,
    json_extract(e.payload_json, '$.duration_ms') / 1000.0 AS duration_s,
    e.session_id                                          AS session_id,
    e.night_id                                            AS night_id
FROM equipment_events e
WHERE e.kind = 'operation_ended'
  AND e.command_id IS NOT NULL;
