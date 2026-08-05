-- 005: measured focuser scale.
--
-- steps_per_micron is a mechanical property of the focuser and drawtube. It is
-- not printed on anything, few operators know it for their own equipment, and
-- requiring it by hand is what kept the focus-drift refocus threshold unset.
--
-- It is measurable: the slope of an autofocus V-curve wing, against the focal
-- ratio and the sensor pixel pitch, gives it directly
-- (focus::focus_drift::steps_per_micron_from_vcurve). This column is where that
-- measurement lands so it survives a restart.
--
-- Kept on focuser_profiles rather than rigs because the scale belongs to the
-- focuser: swap the focuser and the number changes, swap the camera and it does
-- not. `rigs` already holds focal_ratio and pixel_size_um, which are the two
-- inputs the measurement needs and which belong to the optics and the sensor
-- respectively.

ALTER TABLE focuser_profiles ADD COLUMN steps_per_micron REAL;

-- When it was measured. A scale taken through a focal reducer and then read
-- back after the reducer came off is worse than no scale, because it looks
-- authoritative. The timestamp makes that staleness visible rather than
-- leaving the number to age silently.
ALTER TABLE focuser_profiles ADD COLUMN scale_measured_at TEXT;

-- The upsert in device_profiles_repo::record_focuser_scale needs a unique key
-- to conflict on. focuser_profiles has no such constraint today, so two runs
-- would otherwise accumulate duplicate rows for the same focuser and
-- load_focuser_scale would return whichever the planner happened to reach
-- first.
CREATE UNIQUE INDEX IF NOT EXISTS idx_focuser_profiles_rig_name
    ON focuser_profiles(rig_id, focuser_name);
