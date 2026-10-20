//! plate_solve_repo — Section 8.4.
//!
//! Which archived file a plate solve reads, and the record of what it found.
//! Column units are declared in migration 008.

use sqlx::SqlitePool;

/// An archived exposure and the file a solver should read from it.
#[derive(Debug, Clone)]
pub struct SolvableFrame {
    pub frame_id: String,
    pub session_id: Option<String>,
    /// `frames.binning`. `None` when the row never recorded it, which leaves
    /// the rig-optics scale unknown rather than "probably 1x1".
    pub binning: Option<i64>,
    pub asset: Option<SolvableAsset>,
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
    let frame: Option<(String, Option<String>, Option<i64>)> =
        sqlx::query_as("SELECT id, session_id, binning FROM frames WHERE id = ?")
            .bind(frame_id)
            .fetch_optional(pool)
            .await?;
    let Some((frame_id, session_id, binning)) = frame else {
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
    /// `manual` | `startup` | `post_slew`.
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

/// Append the record and return its id.
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
            failure_kind, solver_message, solver_command, solve_time_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    .execute(pool)
    .await?;
    Ok(id)
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
}
