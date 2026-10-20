-- ─────────────────────────────────────────────────────────────────────────────
-- 008_plate_solve_provenance.sql
--
-- `plate_solve_events` (migration 003) had no writer until ASTAP was wired in,
-- and no way to say which exposure a solution belongs to. A solved position
-- that cannot be traced to its frame cannot be paired with the mount position
-- at that exposure, which is the only thing the pointing model (M-4) wants it
-- for.
--
-- Units for the 003 columns, fixed here before the first row is written:
--   solved_ra, solved_dec  degrees, ICRS (J2000) — the star catalogue's frame,
--                          not the JNow most mount drivers report
--   rotation_deg           position angle of the image +Y axis, degrees east
--                          of north, in the FITS pixel frame of the solved file
--   pixel_scale            arcsec per pixel
--   pointing_error_arcmin  left NULL: it needs the mount coordinate at exposure
--                          time and an epoch conversion, neither recorded yet
--
-- Every attempt is a row, solved or not. A pointing model trained only on the
-- frames that solved never learns where the sky was too thin to solve.
-- ─────────────────────────────────────────────────────────────────────────────

-- What was solved: the exposure, and the exact archived bytes.
ALTER TABLE plate_solve_events ADD COLUMN frame_id TEXT REFERENCES frames(id);
ALTER TABLE plate_solve_events ADD COLUMN asset_id TEXT REFERENCES frame_assets(id);
ALTER TABLE plate_solve_events ADD COLUMN asset_sha256 TEXT;

-- Why it ran: manual (an API request) | startup | post_slew (the
-- deterministic trigger rules of v6 §8.3.3).
ALTER TABLE plate_solve_events ADD COLUMN trigger TEXT;

-- What the solver was told. A hint is a search starting point, never evidence:
-- hinted | blind.
ALTER TABLE plate_solve_events ADD COLUMN mode TEXT;
ALTER TABLE plate_solve_events ADD COLUMN hint_ra_deg REAL;
ALTER TABLE plate_solve_events ADD COLUMN hint_dec_deg REAL;
ALTER TABLE plate_solve_events ADD COLUMN search_radius_deg REAL;
-- arcsec/px, and where it came from: request | rig_optics. NULL when the
-- solver searched for the scale itself.
ALTER TABLE plate_solve_events ADD COLUMN scale_hint_arcsec REAL;
ALTER TABLE plate_solve_events ADD COLUMN scale_hint_source TEXT;

-- What it found, beyond the 003 columns.
ALTER TABLE plate_solve_events ADD COLUMN field_width_deg REAL;
ALTER TABLE plate_solve_events ADD COLUMN field_height_deg REAL;
-- det(CD) > 0 in the solved file's pixel frame.
ALTER TABLE plate_solve_events ADD COLUMN mirrored INTEGER;
-- The full TAN solution: crpix, crval, cd, naxis.
ALTER TABLE plate_solve_events ADD COLUMN wcs_json TEXT;

-- Why it failed (stable codes from plate_solver::SolveFailureKind), the
-- solver's own words, and the command line that produced the row.
ALTER TABLE plate_solve_events ADD COLUMN failure_kind TEXT;
ALTER TABLE plate_solve_events ADD COLUMN solver_message TEXT;
ALTER TABLE plate_solve_events ADD COLUMN solver_command TEXT;

CREATE INDEX IF NOT EXISTS idx_plate_solve_frame ON plate_solve_events(frame_id);
