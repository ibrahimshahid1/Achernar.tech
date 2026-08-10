-- ─────────────────────────────────────────────────────────────────────────────
-- 006_environment_log.sql
--
-- Somewhere for observations that no instrument in this system can make.
--
-- `hardware/sensors.rs` is still `todo!()`, and even once it is not, three of
-- the quantities below are judgements rather than measurements: sky condition,
-- transparency, and seeing impression come from a person looking up. The Focus
-- Drift Forecaster needs ambient temperature and the Cloud/Obstruction
-- Classifier needs the sky-condition label; neither is recoverable after the
-- night has ended, so both need a home that does not depend on a driver
-- existing.
--
-- Every column except the timestamp is nullable on purpose. A night where
-- humidity was not recorded must be representable as "not recorded" rather than
-- as a plausible number, for the same reason `frames.temperature_state` exists.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS environment_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT REFERENCES sessions(id),
    -- Local date of *sunset*, so one observing night is one id across midnight.
    night_id            TEXT,
    rig_id              TEXT,
    observed_at         TEXT NOT NULL,

    ambient_temp_c      REAL,
    humidity_pct        REAL,
    dew_point_c         REAL,
    wind_beaufort       INTEGER,

    -- RB-5 five-point scale: 0 clear, 1 thin, 2 moderate, 3 heavy, 4 overcast.
    -- Collected at five levels because the spec's Section 7.6.4 head is binary
    -- while cloud_classifier.py and coc_runner.rs both use five classes. Five
    -- collapses to two; two can never be expanded to five, and the sky is not
    -- re-observable.
    sky_class           INTEGER CHECK (sky_class IS NULL OR sky_class BETWEEN 0 AND 4),

    -- Naked-eye limiting magnitude, if estimated.
    transparency_nelm   REAL,
    -- Subjective 1-5. Useful only because it is recorded consistently.
    seeing_impression   INTEGER,

    -- Where the row came from: 'manual_log' | 'probe' | 'forecast'.
    -- A forecast temperature and a measured one are not the same observation.
    source              TEXT NOT NULL DEFAULT 'manual_log',
    notes               TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS environment_log_time_idx  ON environment_log(observed_at);
CREATE INDEX IF NOT EXISTS environment_log_night_idx ON environment_log(night_id, observed_at);

-- ─── Disturbances ───
-- One unlogged bump in the middle of a focus sweep turns a clean V-curve into a
-- bimodal one, and the fitter will converge on the wrong minimum with no
-- outward sign that anything went wrong. This table is how a run gets dropped
-- on purpose rather than silently poisoning a model.
CREATE TABLE IF NOT EXISTS disturbance_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT REFERENCES sessions(id),
    night_id        TEXT,
    occurred_at     TEXT NOT NULL,
    -- bump | refocus | cloud | headlights | dew | battery | wind_gust | other
    kind            TEXT NOT NULL,
    affected_run_id TEXT,
    description     TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS disturbance_log_time_idx  ON disturbance_log(occurred_at);
CREATE INDEX IF NOT EXISTS disturbance_log_run_idx   ON disturbance_log(affected_run_id);
