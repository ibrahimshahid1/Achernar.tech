//! plate_solve_repo — Section 8.4.
//!
//! Which archived file a plate solve reads, and the record of what it found.
//! Column units are declared in migration 008.
//!
//! Two readers serve the equipment layer (WP1B-4):
//!
//! - [`latest_successful_solve`] is the plate-solve state of the shared
//!   perception bus (v6 §8.1): `last_solve_at` and `pointing_error_arcmin`
//!   come from `plate_solve_events`, never from a value carried forward
//!   between observations, and the solved WCS (with the pier side and
//!   rotator angle at that solve, and the mount and mid-exposure of its
//!   pointing sample, which tie it to a mount connection) is what lets TED
//!   attribute a trail to a sky axis (H15).
//! - [`set_pointing_error`] fills `plate_solve_events.pointing_error_arcmin`
//!   for the first time, from the H12 pointing sample of that solve.

use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::{Row, SqlitePool};

use crate::equipment::types::PierSide;
use crate::processing::plate_solver::TanWcs;
use crate::storage::columns;

/// An archived exposure and the file a solver should read from it.
#[derive(Debug, Clone)]
pub struct SolvableFrame {
    pub frame_id: String,
    pub session_id: Option<String>,
    /// `frames.binning`. `None` when the row never recorded it, which leaves
    /// the rig-optics scale unknown rather than "probably 1x1".
    pub binning: Option<i64>,
    pub asset: Option<SolvableAsset>,
    /// light | dark | flat | bias | focus | preview | verification | center
    pub capture_type: Option<String>,
    pub camera_name: Option<String>,
    /// The driver's exposure start and end (`frames.exposure_started_at`,
    /// `exposure_completed_at`): the window a pointing sample is read from.
    pub exposure_started_at: Option<DateTime<Utc>>,
    pub exposure_completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct SolvableAsset {
    pub id: String,
    pub role: String,
    /// Relative to the capture root.
    pub relative_path: String,
    pub sha256: String,
}

/// `Ok(None)` when no such frame exists.
///
/// The sensor FITS is preferred: it is the canonical integer encoding every
/// solver reads, with `BAYERPAT` intact. The float analysis plane keeps NaN
/// for masked pixels, which no solver documents tolerating. A calibrated FITS,
/// once reprocessing produces one, is the fallback.
pub async fn load_solvable_frame(
    pool: &SqlitePool,
    frame_id: &str,
) -> Result<Option<SolvableFrame>, sqlx::Error> {
    type FrameRow = (
        String,
        Option<String>,
        Option<i64>,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
    );
    let frame: Option<FrameRow> = sqlx::query_as(
        "SELECT id, session_id, binning, capture_type, camera_name, exposure_started_at, \
         exposure_completed_at FROM frames WHERE id = ?",
    )
    .bind(frame_id)
    .fetch_optional(pool)
    .await?;
    let Some((frame_id, session_id, binning, capture_type, camera_name, started, completed)) =
        frame
    else {
        return Ok(None);
    };

    let asset: Option<(String, String, String, String)> = sqlx::query_as(
        r#"
        SELECT id, role, relative_path, checksum_sha256
        FROM frame_assets
        WHERE frame_id = ? AND role IN ('sensor_fits', 'calibrated_fits')
        ORDER BY CASE role WHEN 'sensor_fits' THEN 0 ELSE 1 END, variant
        LIMIT 1
        "#,
    )
    .bind(&frame_id)
    .fetch_optional(pool)
    .await?;

    Ok(Some(SolvableFrame {
        frame_id,
        session_id,
        binning,
        asset: asset.map(|(id, role, relative_path, sha256)| SolvableAsset {
            id,
            role,
            relative_path,
            sha256,
        }),
        capture_type,
        camera_name,
        exposure_started_at: columns::parse_opt_time(started),
        exposure_completed_at: columns::parse_opt_time(completed),
    }))
}

/// One row of `plate_solve_events`, solved or not.
#[derive(Debug, Clone, Default)]
pub struct PlateSolveRecord {
    pub session_id: Option<String>,
    pub frame_id: String,
    pub asset_id: String,
    pub asset_sha256: String,
    pub solver: String,
    /// `manual` | `startup` | `post_slew` | `post_flip` | `recovery` |
    /// `center`.
    pub trigger: String,
    /// `hinted` | `blind`.
    pub mode: String,
    pub hint_ra_deg: Option<f64>,
    pub hint_dec_deg: Option<f64>,
    pub search_radius_deg: Option<f64>,
    pub scale_hint_arcsec: Option<f64>,
    /// `request` | `rig_optics`.
    pub scale_hint_source: Option<String>,
    pub success: bool,
    pub ra_deg: Option<f64>,
    pub dec_deg: Option<f64>,
    pub position_angle_deg: Option<f64>,
    pub pixel_scale_arcsec: Option<f64>,
    pub field_width_deg: Option<f64>,
    pub field_height_deg: Option<f64>,
    pub mirrored: Option<bool>,
    pub wcs_json: Option<String>,
    pub failure_kind: Option<String>,
    pub solver_message: Option<String>,
    pub solver_command: String,
    pub solve_time_ms: i64,
}

/// Append the record and return its id. `created_at` is written with
/// milliseconds, so the perception bus can say how old a solve is.
pub async fn record_solve(
    pool: &SqlitePool,
    record: &PlateSolveRecord,
) -> Result<String, sqlx::Error> {
    let id = uuid::Uuid::new_v4().to_string();
    sqlx::query(
        r#"
        INSERT INTO plate_solve_events (
            id, session_id, frame_id, asset_id, asset_sha256, solver, trigger, mode,
            hint_ra_deg, hint_dec_deg, search_radius_deg,
            scale_hint_arcsec, scale_hint_source,
            success, solved_ra, solved_dec, rotation_deg, pixel_scale,
            field_width_deg, field_height_deg, mirrored, wcs_json,
            failure_kind, solver_message, solver_command, solve_time_ms, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(&id)
    .bind(&record.session_id)
    .bind(&record.frame_id)
    .bind(&record.asset_id)
    .bind(&record.asset_sha256)
    .bind(&record.solver)
    .bind(&record.trigger)
    .bind(&record.mode)
    .bind(record.hint_ra_deg)
    .bind(record.hint_dec_deg)
    .bind(record.search_radius_deg)
    .bind(record.scale_hint_arcsec)
    .bind(&record.scale_hint_source)
    .bind(record.success)
    .bind(record.ra_deg)
    .bind(record.dec_deg)
    .bind(record.position_angle_deg)
    .bind(record.pixel_scale_arcsec)
    .bind(record.field_width_deg)
    .bind(record.field_height_deg)
    .bind(record.mirrored)
    .bind(&record.wcs_json)
    .bind(&record.failure_kind)
    .bind(&record.solver_message)
    .bind(&record.solver_command)
    .bind(record.solve_time_ms)
    .bind(columns::time(&Utc::now()))
    .execute(pool)
    .await?;
    Ok(id)
}

/// Fill `pointing_error_arcmin` of one solve from its pointing sample (H12).
/// Returns whether the row exists. A non-finite error is never written.
pub async fn set_pointing_error(
    pool: &SqlitePool,
    event_id: &str,
    pointing_error_arcmin: f64,
) -> Result<bool, sqlx::Error> {
    if !pointing_error_arcmin.is_finite() {
        return Ok(false);
    }
    let result =
        sqlx::query("UPDATE plate_solve_events SET pointing_error_arcmin = ? WHERE id = ?")
            .bind(pointing_error_arcmin)
            .bind(event_id)
            .execute(pool)
            .await?;
    Ok(result.rows_affected() > 0)
}

/// The exposure that produced a solvable frame: the settings the center loop
/// reuses when it is not told otherwise (M-n step 2).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SolvedExposure {
    pub exposure_s: f64,
    pub gain: Option<f64>,
    pub binning: Option<i64>,
}

/// The newest successful solve, with what TED needs to attribute a trail to
/// a sky axis (H15) and what the perception bus publishes (v6 §8.1).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LatestSolve {
    pub event_id: String,
    pub frame_id: Option<String>,
    /// When the solve was recorded.
    pub solved_at: Option<DateTime<Utc>>,
    /// The solved exposure's start, from the driver's clock.
    pub exposure_started_at: Option<DateTime<Utc>>,
    pub trigger: Option<String>,
    pub ra_deg: Option<f64>,
    pub dec_deg: Option<f64>,
    pub position_angle_deg: Option<f64>,
    pub pixel_scale_arcsec: Option<f64>,
    pub mirrored: Option<bool>,
    /// The full TAN solution, when it was recorded.
    pub wcs: Option<TanWcs>,
    /// Filled by the H12 pointing sample of this solve.
    pub pointing_error_arcmin: Option<f64>,
    pub camera_name: Option<String>,
    /// The mount's pier side at that exposure (its pointing sample); `None`
    /// when no sample exists or the side was unknown.
    pub pier_side: Option<PierSide>,
    /// The mount of that exposure's pointing sample.
    pub mount_identity_key: Option<String>,
    /// That pointing sample's mid-exposure, which places the solve in a
    /// mount connection's segment (`pointing_model::connection_segment_start`).
    pub sample_mid_exposure_utc: Option<DateTime<Utc>>,
    /// The rotator angle at that exposure (its device window), when one was
    /// recorded.
    pub rotator_angle_deg: Option<f64>,
    pub exposure: Option<SolvedExposure>,
}

/// The newest successful solve, `Ok(None)` when nothing has solved yet.
/// Ordered by insertion (`rowid`): `created_at` mixes SQLite's default text
/// form with RFC 3339 across daemon versions.
pub async fn latest_successful_solve(
    pool: &SqlitePool,
) -> Result<Option<LatestSolve>, sqlx::Error> {
    let row = sqlx::query(
        r#"
        SELECT e.id, e.frame_id, e.created_at, e.trigger, e.solved_ra, e.solved_dec,
               e.rotation_deg, e.pixel_scale, e.mirrored, e.wcs_json, e.pointing_error_arcmin,
               f.camera_name, f.exposure_started_at, f.requested_exposure_s, f.exposure_s,
               f.requested_gain, f.binning,
               (SELECT p.pier_side FROM pointing_samples p
                 WHERE p.plate_solve_event_id = e.id ORDER BY p.rowid DESC LIMIT 1) AS pier_side,
               (SELECT p.mount_identity_key FROM pointing_samples p
                 WHERE p.plate_solve_event_id = e.id ORDER BY p.rowid DESC LIMIT 1)
                 AS mount_identity_key,
               (SELECT p.mid_exposure_utc FROM pointing_samples p
                 WHERE p.plate_solve_event_id = e.id ORDER BY p.rowid DESC LIMIT 1)
                 AS sample_mid_exposure_utc,
               (SELECT w.rotator_angle_deg FROM exposure_device_windows w
                 WHERE w.frame_id = e.frame_id) AS rotator_angle_deg
        FROM plate_solve_events e
        LEFT JOIN frames f ON f.id = e.frame_id
        WHERE e.success = 1
        ORDER BY e.rowid DESC
        LIMIT 1
        "#,
    )
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|row| {
        let text = |name: &str| -> Option<String> { row.get(name) };
        let exposure_s = row
            .get::<Option<f64>, _>("requested_exposure_s")
            .or_else(|| row.get::<Option<f64>, _>("exposure_s"))
            .filter(|s| s.is_finite() && *s > 0.0);
        LatestSolve {
            event_id: row.get("id"),
            frame_id: row.get("frame_id"),
            solved_at: columns::parse_opt_time(text("created_at")),
            exposure_started_at: columns::parse_opt_time(text("exposure_started_at")),
            trigger: row.get("trigger"),
            ra_deg: row.get("solved_ra"),
            dec_deg: row.get("solved_dec"),
            position_angle_deg: row.get("rotation_deg"),
            pixel_scale_arcsec: row.get("pixel_scale"),
            mirrored: row.get("mirrored"),
            wcs: columns::parse_json(text("wcs_json").as_deref()),
            pointing_error_arcmin: row.get("pointing_error_arcmin"),
            camera_name: row.get("camera_name"),
            pier_side: columns::from_name::<PierSide>(text("pier_side").as_deref())
                .filter(|side| *side != PierSide::Unknown),
            mount_identity_key: text("mount_identity_key"),
            sample_mid_exposure_utc: columns::parse_opt_time(text("sample_mid_exposure_utc")),
            rotator_angle_deg: row.get("rotator_angle_deg"),
            exposure: exposure_s.map(|exposure_s| SolvedExposure {
                exposure_s,
                gain: row.get("requested_gain"),
                binning: row.get("binning"),
            }),
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_pool() -> SqlitePool {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        for (_, _, sql) in crate::storage::local_store::migrations_for_test() {
            for statement in crate::storage::local_store::split_statements_for_test(sql) {
                if let Err(error) = sqlx::query(&statement).execute(&pool).await {
                    let message = error.to_string();
                    assert!(
                        message.contains("duplicate column name")
                            || message.contains("already exists"),
                        "unexpected migration failure: {error}"
                    );
                }
            }
        }
        pool
    }

    async fn insert_frame(pool: &SqlitePool, id: &str, binning: Option<i64>) {
        sqlx::query(
            "INSERT INTO frames (id, session_id, local_path, captured_at, binning) \
             VALUES (?, NULL, '', '2026-10-15T02:00:00Z', ?)",
        )
        .bind(id)
        .bind(binning)
        .execute(pool)
        .await
        .unwrap();
    }

    async fn insert_asset(pool: &SqlitePool, frame_id: &str, role: &str, variant: i64) -> String {
        let id = format!("{frame_id}-{role}-{variant}");
        sqlx::query(
            "INSERT INTO frame_assets (id, frame_id, role, variant, relative_path, media_type, \
             byte_length, checksum_sha256, created_at) \
             VALUES (?, ?, ?, ?, ?, 'application/fits', 2880, ?, '2026-10-15T02:00:00Z')",
        )
        .bind(&id)
        .bind(frame_id)
        .bind(role)
        .bind(variant)
        .bind(format!("{frame_id}/{role}-{variant}.fits"))
        .bind(format!("sha-{role}-{variant}"))
        .execute(pool)
        .await
        .unwrap();
        id
    }

    #[tokio::test]
    async fn the_original_sensor_fits_is_what_gets_solved() {
        let pool = test_pool().await;
        insert_frame(&pool, "f1", Some(2)).await;
        insert_asset(&pool, "f1", "analysis_fits", 0).await;
        insert_asset(&pool, "f1", "calibrated_fits", 0).await;
        insert_asset(&pool, "f1", "sensor_fits", 1).await;
        let original = insert_asset(&pool, "f1", "sensor_fits", 0).await;

        let frame = load_solvable_frame(&pool, "f1").await.unwrap().unwrap();
        assert_eq!(frame.binning, Some(2));
        let asset = frame.asset.unwrap();
        assert_eq!(asset.id, original);
        assert_eq!(asset.sha256, "sha-sensor_fits-0");
    }

    #[tokio::test]
    async fn a_frame_without_a_solvable_file_says_so() {
        let pool = test_pool().await;
        insert_frame(&pool, "luma-only", None).await;
        insert_asset(&pool, "luma-only", "analysis_luma", 0).await;
        let frame = load_solvable_frame(&pool, "luma-only")
            .await
            .unwrap()
            .unwrap();
        assert!(frame.asset.is_none());
        assert_eq!(frame.binning, None);
        assert!(load_solvable_frame(&pool, "missing")
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn the_exposure_window_and_camera_come_with_the_frame() {
        let pool = test_pool().await;
        sqlx::query(
            "INSERT INTO frames (id, local_path, captured_at, capture_type, camera_name, \
             exposure_started_at, exposure_completed_at) VALUES ('w', '', \
             '2026-10-26T03:00:00Z', 'center', 'R50', '2026-10-26T03:00:00.250Z', \
             '2026-10-26T03:00:05.250Z')",
        )
        .execute(&pool)
        .await
        .unwrap();
        let frame = load_solvable_frame(&pool, "w").await.unwrap().unwrap();
        assert_eq!(frame.capture_type.as_deref(), Some("center"));
        assert_eq!(frame.camera_name.as_deref(), Some("R50"));
        let (start, end) = (
            frame.exposure_started_at.unwrap(),
            frame.exposure_completed_at.unwrap(),
        );
        assert_eq!((end - start).num_milliseconds(), 5000);
    }

    #[tokio::test]
    async fn failed_and_successful_attempts_are_both_kept() {
        let pool = test_pool().await;
        insert_frame(&pool, "f1", Some(1)).await;
        let asset = insert_asset(&pool, "f1", "sensor_fits", 0).await;
        let base = PlateSolveRecord {
            frame_id: "f1".into(),
            asset_id: asset,
            asset_sha256: "sha".into(),
            solver: "astap".into(),
            mode: "blind".into(),
            solver_command: "astap_cli -f frame.fits".into(),
            ..PlateSolveRecord::default()
        };
        record_solve(
            &pool,
            &PlateSolveRecord {
                failure_kind: Some("not_enough_stars".into()),
                solve_time_ms: 900,
                ..base.clone()
            },
        )
        .await
        .unwrap();
        record_solve(
            &pool,
            &PlateSolveRecord {
                success: true,
                ra_deg: Some(83.822),
                dec_deg: Some(-5.391),
                pixel_scale_arcsec: Some(1.2),
                mirrored: Some(false),
                solve_time_ms: 2100,
                ..base
            },
        )
        .await
        .unwrap();

        let rows: Vec<(i64, Option<String>, Option<f64>, Option<i64>)> = sqlx::query_as(
            "SELECT success, failure_kind, solved_ra, mirrored FROM plate_solve_events \
             WHERE frame_id = 'f1' ORDER BY solve_time_ms",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(
            rows,
            vec![
                (0, Some("not_enough_stars".into()), None, None),
                (1, None, Some(83.822), Some(0)),
            ]
        );
    }

    #[tokio::test]
    async fn the_latest_solve_comes_from_the_table_with_its_own_timestamp_and_geometry() {
        let pool = test_pool().await;
        assert!(latest_successful_solve(&pool).await.unwrap().is_none());
        insert_frame(&pool, "f1", Some(2)).await;
        sqlx::query(
            "UPDATE frames SET camera_name = 'SV105', requested_exposure_s = 4.0, \
             requested_gain = 120 WHERE id = 'f1'",
        )
        .execute(&pool)
        .await
        .unwrap();
        let asset = insert_asset(&pool, "f1", "sensor_fits", 0).await;
        let wcs = TanWcs {
            crpix: [2000.5, 1500.5],
            crval: [83.8, -5.4],
            cd: [[-3.0e-4, 0.0], [0.0, 3.0e-4]],
            naxis: [4000, 3000],
        };
        let solved = PlateSolveRecord {
            frame_id: "f1".into(),
            asset_id: asset.clone(),
            trigger: "post_slew".into(),
            success: true,
            ra_deg: Some(83.8),
            dec_deg: Some(-5.4),
            position_angle_deg: Some(0.0),
            wcs_json: Some(serde_json::to_string(&wcs).unwrap()),
            ..PlateSolveRecord::default()
        };
        let first = record_solve(&pool, &solved).await.unwrap();
        // A later failure does not hide the last success.
        record_solve(
            &pool,
            &PlateSolveRecord {
                frame_id: "f1".into(),
                asset_id: asset,
                failure_kind: Some("not_enough_stars".into()),
                ..PlateSolveRecord::default()
            },
        )
        .await
        .unwrap();
        assert!(set_pointing_error(&pool, &first, 1.25).await.unwrap());
        assert!(!set_pointing_error(&pool, &first, f64::NAN).await.unwrap());
        assert!(!set_pointing_error(&pool, "no-such-solve", 1.0)
            .await
            .unwrap());
        sqlx::query(
            "INSERT INTO pointing_samples (id, plate_solve_event_id, frame_id, \
             mount_identity_key, mount_frame, solved_ra_j2000_h, solved_dec_j2000_deg, \
             pier_side, purpose, mid_exposure_utc) VALUES ('s1', ?, 'f1', 'sim:mount', \
             'jnow_apparent', 5.58, -5.4, 'west', 'manual', '2026-10-26T04:01:00Z')",
        )
        .bind(&first)
        .execute(&pool)
        .await
        .unwrap();

        let latest = latest_successful_solve(&pool).await.unwrap().unwrap();
        assert_eq!(latest.event_id, first);
        assert_eq!(latest.trigger.as_deref(), Some("post_slew"));
        assert_eq!(latest.pointing_error_arcmin, Some(1.25));
        assert_eq!(latest.wcs, Some(wcs));
        assert_eq!(latest.camera_name.as_deref(), Some("SV105"));
        assert_eq!(latest.pier_side, Some(PierSide::West));
        assert_eq!(latest.mount_identity_key.as_deref(), Some("sim:mount"));
        assert_eq!(
            latest.sample_mid_exposure_utc,
            Some(
                DateTime::parse_from_rfc3339("2026-10-26T04:01:00Z")
                    .unwrap()
                    .with_timezone(&Utc)
            )
        );
        assert_eq!(latest.rotator_angle_deg, None);
        assert!(latest.solved_at.is_some());
        assert_eq!(
            latest.exposure,
            Some(SolvedExposure {
                exposure_s: 4.0,
                gain: Some(120.0),
                binning: Some(2),
            })
        );
    }
}
