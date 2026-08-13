-- ─────────────────────────────────────────────────────────────────────────────
-- 007_grouping_keys.sql
--
-- The keys that make a focus run joinable to the night it happened on.
--
-- Two separate problems, one cause.
--
-- 1. Migration 006 gave the observer somewhere to write ambient temperature,
--    humidity and a sky-condition class. The Focus Drift Forecaster needs the
--    first of those and the Focus Offset Predictor needs both it and the third.
--    But `focus_runs` carried no night_id and no session_id, and autofocus
--    frames were written with a NULL session_id, so there was no column
--    anywhere to join `environment_log` to a focus run on. The data was being
--    collected into a table nothing could reach.
--
-- 2. The grouping hierarchy has five levels — star, frame, run, night, rig —
--    and names night as the split unit for the cloud, background and bandit
--    models. `focus_runs.id` gave us the run level and `rigs.id` the rig level.
--    The night level existed only in prose. A split that cannot group by night
--    lets two runs an hour apart, sharing seeing, transparency and sky
--    brightness, land on opposite sides of a train/test boundary and report a
--    generalisation score that is really a memorisation score.
--
-- night_id is the local date of *sunset*, so one observing night is one id
-- across midnight. This matches `night_id_for()` in tools/import_session.py
-- exactly, and it has to: the importer and the daemon write rows that get
-- grouped together, and two different definitions of "night" would silently
-- split one session in half.
--
-- Both columns are nullable and are NOT backfilled. Deriving a sunset-local
-- date from an existing UTC timestamp needs the site's UTC offset, which no
-- row records, and a plausible-looking wrong night_id is worse than an absent
-- one — it would group runs from adjacent nights and there would be no way to
-- tell afterwards. Pre-existing rows predate the focuser and carry no
-- trainable focus data, so nothing is lost by leaving them NULL.
-- ─────────────────────────────────────────────────────────────────────────────

-- ─── Focus runs ───
-- Written by api::focus::persist_focus_run from the run's start time.
ALTER TABLE focus_runs ADD COLUMN night_id TEXT;

-- The session this run belonged to, when the run happened inside one. NULL for
-- a bare `POST /focus/run` with no session open, which is the common case for
-- engineering sweeps. No FOREIGN KEY: SQLite cannot add one via ALTER TABLE,
-- and the daemon must be able to record a run whether or not a session row
-- exists.
ALTER TABLE focus_runs ADD COLUMN session_id TEXT;

CREATE INDEX IF NOT EXISTS focus_runs_night_idx   ON focus_runs(night_id, started_at);
CREATE INDEX IF NOT EXISTS focus_runs_session_idx ON focus_runs(session_id);

-- ─── Frames ───
-- frames already has session_id, but a session is not a night: a session ends
-- when the daemon stops and several can share one night, while a single long
-- session cannot span two. The night-level split needs the night itself.
--
-- This is the column the cloud classifier's split will group on — its rows
-- arrive through frame_measurements, keyed by frame_id — so it matters for the
-- campaign that starts on cloudy nights, not only for focus work.
ALTER TABLE frames ADD COLUMN night_id TEXT;

CREATE INDEX IF NOT EXISTS frames_night_idx ON frames(night_id, captured_at);
