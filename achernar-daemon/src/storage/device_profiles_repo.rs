//! device_profiles_repo — Sections 7.4, 7.6.3.
//!
//! Reads the per-rig equipment description from the `rigs` row (migration 002)
//! and the per-focuser mechanical scale from `focuser_profiles`.
//!
//! `rigs.focal_ratio` and `rigs.pixel_size_um` are the reason the focus-drift
//! threshold does not have to be typed into `achernar.toml`: they are exactly
//! the optical facts the critical-focus-zone calculation needs, and they have
//! been in the schema since long before the forecaster was wired up. Asking the
//! operator to state them a second time in a config file is the friction that
//! guarantees the feature stays switched off.

use crate::config::RigOptics;
use sqlx::SqlitePool;

/// Load the optical constants for `rig_id`.
///
/// A missing row is `Ok(None)`, not an error: plenty of installations never
/// create one, and the caller's response is the same either way — fall through
/// to config, and if that is empty too, run the filter without a threshold.
pub async fn load_rig_optics(
    pool: &SqlitePool,
    rig_id: &str,
) -> Result<Option<RigOptics>, sqlx::Error> {
    let row: Option<(Option<f64>, Option<f64>, Option<f64>)> =
        sqlx::query_as("SELECT focal_ratio, focal_length_mm, pixel_size_um FROM rigs WHERE id = ?")
            .bind(rig_id)
            .fetch_optional(pool)
            .await?;

    Ok(row.map(|(focal_ratio, focal_length_mm, pixel_size_um)| RigOptics {
        // Guard against zeros and negatives left by an earlier import. A focal
        // ratio of 0 is not a focal ratio; treating it as absent is better than
        // propagating a division by zero into a refocus threshold.
        focal_ratio: focal_ratio.filter(|v| v.is_finite() && *v > 0.0),
        focal_length_mm: focal_length_mm.filter(|v| v.is_finite() && *v > 0.0),
        pixel_size_um: pixel_size_um.filter(|v| v.is_finite() && *v > 0.0),
    }))
}

/// Record a focuser scale measured from an autofocus V-curve.
///
/// Upserts onto `focuser_profiles`, keyed by (rig, focuser name), so the
/// measurement survives a restart and never has to be entered by hand.
///
/// `scale_measured_at` is stored alongside it because a scale measured through
/// a focal reducer and then read back after the reducer came off is worse than
/// no scale at all. The timestamp at least makes the staleness visible.
pub async fn record_focuser_scale(
    pool: &SqlitePool,
    rig_id: &str,
    focuser_name: &str,
    steps_per_micron: f64,
) -> Result<(), sqlx::Error> {
    if !steps_per_micron.is_finite() || steps_per_micron <= 0.0 {
        return Ok(());
    }
    sqlx::query(
        r#"
        INSERT INTO focuser_profiles (id, rig_id, focuser_name, steps_per_micron, scale_measured_at)
        VALUES (?, ?, ?, ?, datetime('now'))
        ON CONFLICT(rig_id, focuser_name) DO UPDATE SET
            steps_per_micron  = excluded.steps_per_micron,
            scale_measured_at = excluded.scale_measured_at,
            updated_at        = datetime('now')
        "#,
    )
    .bind(uuid::Uuid::new_v4().to_string())
    .bind(rig_id)
    .bind(focuser_name)
    .bind(steps_per_micron)
    .execute(pool)
    .await?;
    Ok(())
}

/// The most recently measured focuser scale for this rig + focuser, if any.
pub async fn load_focuser_scale(
    pool: &SqlitePool,
    rig_id: &str,
    focuser_name: &str,
) -> Result<Option<f64>, sqlx::Error> {
    let row: Option<(Option<f64>,)> = sqlx::query_as(
        "SELECT steps_per_micron FROM focuser_profiles WHERE rig_id = ? AND focuser_name = ?",
    )
    .bind(rig_id)
    .bind(focuser_name)
    .fetch_optional(pool)
    .await?;
    Ok(row.and_then(|(v,)| v).filter(|v| v.is_finite() && *v > 0.0))
}
