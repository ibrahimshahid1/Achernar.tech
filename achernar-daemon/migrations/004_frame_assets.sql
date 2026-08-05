-- Migration 004 — one exposure = one `frames` row + N immutable `frame_assets`.
--
-- `frames` was previously "a PNG file we saved", which broke as soon as one
-- exposure produced more than one file. A mirrorless body set to RAW+JPEG
-- writes two files per shutter actuation; the daemon then derives a FITS and
-- an analysis luma image from them. Those are four artifacts of ONE physical
-- exposure, and modelling them as four unrelated frames destroys exactly the
-- grouping a leakage-safe train/test split depends on.
--
-- So: `frames` becomes the capture/exposure record (what the shutter did),
-- and `frame_assets` holds the files (what landed on disk). SQLite is the
-- catalogue; the filesystem holds bytes; Parquet is a reproducible export
-- from here, never the primary destination.
--
-- Compatibility: `frames.local_path` and `frames.checksum_sha256` are retained
-- and dual-written as aliases of the preferred analysis_luma asset, because
-- `ml/data_pipeline/build_focus_quality_dataset.py` still reads them. Once
-- that builder joins `frame_assets`, both columns can be dropped.

-- ─── Capture/exposure columns on frames ───
-- Nullable throughout so existing rows (written by persist_luma, which has no
-- shutter behind it) remain valid.

ALTER TABLE frames ADD COLUMN capture_id           TEXT;
ALTER TABLE frames ADD COLUMN rig_id               TEXT;
ALTER TABLE frames ADD COLUMN camera_name          TEXT;
ALTER TABLE frames ADD COLUMN capture_type         TEXT NOT NULL DEFAULT 'light';
ALTER TABLE frames ADD COLUMN capture_status       TEXT NOT NULL DEFAULT 'committed';

-- Authoritative exposure bracket, taken by the driver around the trigger —
-- not a wall-clock read after the download finished.
ALTER TABLE frames ADD COLUMN exposure_started_at  TEXT;
ALTER TABLE frames ADD COLUMN exposure_completed_at TEXT;

-- Requested vs applied, kept separately and on purpose. A DSLR quantises to
-- its shutter ladder, so a 3.5 s request becomes a 4 s exposure; recording
-- only one number means either the intent or the physics is lost.
ALTER TABLE frames ADD COLUMN requested_exposure_s REAL;
ALTER TABLE frames ADD COLUMN requested_gain       REAL;
ALTER TABLE frames ADD COLUMN iso                  REAL;
ALTER TABLE frames ADD COLUMN shutter_label        TEXT;
ALTER TABLE frames ADD COLUMN offset_val           INTEGER;
ALTER TABLE frames ADD COLUMN binning              INTEGER DEFAULT 1;
ALTER TABLE frames ADD COLUMN focuser_position     INTEGER;

-- Temperature provenance. "22.4" from a sensor and "22.4" assumed from a
-- config file are not the same measurement, and a drift model trained without
-- that distinction learns from noise.
ALTER TABLE frames ADD COLUMN temperature_source   TEXT;
ALTER TABLE frames ADD COLUMN temperature_state    TEXT NOT NULL DEFAULT 'missing';

-- Reproducibility: which build made this, and how it was configured.
ALTER TABLE frames ADD COLUMN software_version     TEXT;
ALTER TABLE frames ADD COLUMN driver_name          TEXT;
-- Immutable snapshot of camera-specific fields that have no column of their
-- own (per-body maker settings, live-view state, capture target, …).
ALTER TABLE frames ADD COLUMN acquisition_json     TEXT NOT NULL DEFAULT '{}';

CREATE INDEX IF NOT EXISTS frames_capture_idx ON frames(capture_id);
CREATE INDEX IF NOT EXISTS frames_rig_night_idx ON frames(rig_id, captured_at);

-- ─── Assets ───
CREATE TABLE IF NOT EXISTS frame_assets (
    id                    TEXT PRIMARY KEY,
    frame_id              TEXT NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    -- native_raw | camera_jpeg | sensor_fits | analysis_luma | thumbnail |
    -- calibrated_fits
    role                  TEXT NOT NULL,
    -- Distinguishes repeats of the same role: a reprocessed FITS is variant 1,
    -- not an overwrite of variant 0. Assets are immutable.
    variant               INTEGER NOT NULL DEFAULT 0,
    original_filename     TEXT,
    -- Relative to the configured capture root, so the whole archive can be
    -- moved or mounted elsewhere without rewriting the database.
    relative_path         TEXT NOT NULL,
    object_key            TEXT,
    media_type            TEXT NOT NULL,
    byte_length           INTEGER NOT NULL,
    checksum_sha256       TEXT NOT NULL,
    width                 INTEGER,
    height                INTEGER,
    bit_depth             INTEGER,
    -- Lineage: the FITS derives from the RAW, the luma from the FITS.
    derived_from_asset_id TEXT REFERENCES frame_assets(id),
    provenance_json       TEXT NOT NULL DEFAULT '{}',
    upload_policy         TEXT NOT NULL DEFAULT 'metadata_only',
    created_at            TEXT NOT NULL,
    synced_at             TEXT,
    UNIQUE(frame_id, role, variant)
);

CREATE INDEX IF NOT EXISTS frame_assets_frame_idx ON frame_assets(frame_id);
CREATE INDEX IF NOT EXISTS frame_assets_role_idx  ON frame_assets(role);
CREATE INDEX IF NOT EXISTS frame_assets_sync_idx  ON frame_assets(synced_at);

-- ─── Capture attempts ───
-- Written before the camera is commanded, so a hardware failure leaves a trace
-- instead of vanishing. Without this, "the mount was slewing and the capture
-- silently did nothing" is indistinguishable from "no capture was requested".
CREATE TABLE IF NOT EXISTS capture_attempts (
    capture_id      TEXT PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(id),
    rig_id          TEXT,
    requested_json  TEXT NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'started', -- started|committed|failed
    error           TEXT,
    frame_id        TEXT REFERENCES frames(id),
    started_at      TEXT NOT NULL,
    finished_at     TEXT
);

CREATE INDEX IF NOT EXISTS capture_attempts_status_idx ON capture_attempts(status, started_at);

-- ─── Measurements ───
-- Append-only. HFR/FWHM/FQE/cloud results are opinions produced by a specific
-- pipeline version; overwriting them inside `frames` makes it impossible to
-- re-evaluate a frame later without destroying what the previous model said.
CREATE TABLE IF NOT EXISTS frame_measurements (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id              TEXT NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    measurement_type      TEXT NOT NULL,     -- hfr | fwhm | star_count | fqe | cloud | trailing
    value                 REAL,
    payload_json          TEXT NOT NULL DEFAULT '{}',
    pipeline_version      TEXT,
    model_version         TEXT,
    preprocessing_version TEXT,
    valid                 INTEGER NOT NULL DEFAULT 1,
    rejection_reason      TEXT,
    created_at            TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS frame_measurements_frame_idx
    ON frame_measurements(frame_id, measurement_type);
