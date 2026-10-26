//! `pointing_samples`: one mount pointing residual per plate solve (equipment
//! spec H12, M-4 and M-3; migration 009). Each sample carries the mount's
//! native frame and JNow flavour, DoesRefraction, pier side, hour angle,
//! sync generation and sync mode, so residual series are modelled per mount
//! identity and segmented at every sync (v6 §8.3.4).

// Consumed by the recorder and, once plate solving is wired (WP1B-4), by the
// pointing model and the center loop.
#![allow(dead_code)]

use sqlx::{Executor, Row, Sqlite};
use uuid::Uuid;

use crate::equipment::events::PointingSampleRecord;
use crate::storage::columns;

/// Insert a sample (replacing one with the same id).
pub async fn upsert<'c, E>(executor: E, record: &PointingSampleRecord) -> Result<(), sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    sqlx::query(
        r#"
        INSERT OR REPLACE INTO pointing_samples (
            id, plate_solve_event_id, frame_id, mount_identity_key, mount_frame, jnow_flavour,
            does_refraction, mount_ra_native_h, mount_dec_native_deg, mount_ra_j2000_h,
            mount_dec_j2000_deg, solved_ra_j2000_h, solved_dec_j2000_deg, err_ra_arcsec,
            err_dec_arcsec, err_total_arcsec, pier_side, ha_h, lst_h, alt_deg, az_deg,
            sync_generation, sync_mode, coordinate_precision_arcsec, temperature_c,
            mid_exposure_utc, purpose
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(record.id.to_string())
    .bind(&record.plate_solve_event_id)
    .bind(&record.frame_id)
    .bind(&record.mount_identity_key)
    .bind(record.mount_frame.as_str())
    .bind(columns::name(&record.jnow_flavour))
    .bind(record.does_refraction)
    .bind(record.mount_ra_native_h)
    .bind(record.mount_dec_native_deg)
    .bind(record.mount_ra_j2000_h)
    .bind(record.mount_dec_j2000_deg)
    .bind(record.solved_ra_j2000_h)
    .bind(record.solved_dec_j2000_deg)
    .bind(record.err_ra_arcsec)
    .bind(record.err_dec_arcsec)
    .bind(record.err_total_arcsec)
    .bind(record.pier_side.as_str())
    .bind(record.ha_h)
    .bind(record.lst_h)
    .bind(record.alt_deg)
    .bind(record.az_deg)
    .bind(columns::int(record.sync_generation))
    .bind(columns::name(&record.sync_mode))
    .bind(record.coordinate_precision_arcsec)
    .bind(record.temperature_c)
    .bind(columns::opt_time(record.mid_exposure_utc.as_ref()))
    .bind(columns::name(&record.purpose))
    .execute(executor)
    .await?;
    Ok(())
}

fn from_row(row: &sqlx::sqlite::SqliteRow) -> Option<PointingSampleRecord> {
    let text = |name: &str| -> Option<String> { row.get(name) };
    let id: String = row.get("id");
    Some(PointingSampleRecord {
        id: Uuid::parse_str(&id).ok()?,
        plate_solve_event_id: row.get("plate_solve_event_id"),
        frame_id: row.get("frame_id"),
        mount_identity_key: row.get("mount_identity_key"),
        mount_frame: columns::from_name(text("mount_frame").as_deref()).unwrap_or_default(),
        jnow_flavour: columns::from_name(text("jnow_flavour").as_deref()).unwrap_or_default(),
        does_refraction: row.get("does_refraction"),
        mount_ra_native_h: row.get("mount_ra_native_h"),
        mount_dec_native_deg: row.get("mount_dec_native_deg"),
        mount_ra_j2000_h: row.get("mount_ra_j2000_h"),
        mount_dec_j2000_deg: row.get("mount_dec_j2000_deg"),
        solved_ra_j2000_h: row.get("solved_ra_j2000_h"),
        solved_dec_j2000_deg: row.get("solved_dec_j2000_deg"),
        err_ra_arcsec: row.get("err_ra_arcsec"),
        err_dec_arcsec: row.get("err_dec_arcsec"),
        err_total_arcsec: row.get("err_total_arcsec"),
        pier_side: columns::from_name(text("pier_side").as_deref()).unwrap_or_default(),
        ha_h: row.get("ha_h"),
        lst_h: row.get("lst_h"),
        alt_deg: row.get("alt_deg"),
        az_deg: row.get("az_deg"),
        sync_generation: columns::uint(row.get("sync_generation")),
        sync_mode: columns::from_name(text("sync_mode").as_deref()).unwrap_or_default(),
        coordinate_precision_arcsec: row.get("coordinate_precision_arcsec"),
        temperature_c: row.get("temperature_c"),
        mid_exposure_utc: columns::parse_opt_time(text("mid_exposure_utc")),
        purpose: columns::from_name(text("purpose").as_deref())?,
    })
}

/// Samples of one mount identity, newest first, optionally for one sync
/// generation only (M-4 fits never mix generations).
pub async fn for_mount<'c, E>(
    executor: E,
    mount_identity_key: &str,
    sync_generation: Option<u64>,
    limit: u32,
) -> Result<Vec<PointingSampleRecord>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let rows = sqlx::query(
        r#"
        SELECT * FROM pointing_samples
        WHERE mount_identity_key = ?1 AND (?2 IS NULL OR sync_generation = ?2)
        ORDER BY created_at DESC, rowid DESC
        LIMIT ?3
        "#,
    )
    .bind(mount_identity_key)
    .bind(sync_generation.map(columns::int))
    .bind(i64::from(limit.max(1)))
    .fetch_all(executor)
    .await?;
    Ok(rows.iter().filter_map(from_row).collect())
}

/// The sample taken from one frame, if any.
pub async fn for_frame<'c, E>(
    executor: E,
    frame_id: &str,
) -> Result<Vec<PointingSampleRecord>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let rows = sqlx::query("SELECT * FROM pointing_samples WHERE frame_id = ? ORDER BY rowid")
        .bind(frame_id)
        .fetch_all(executor)
        .await?;
    Ok(rows.iter().filter_map(from_row).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::events::PointingPurpose;
    use crate::equipment::types::{EqFrame, JNowFlavour, MountSyncMode, PierSide};
    use chrono::{TimeZone, Utc};

    fn sample(generation: u64) -> PointingSampleRecord {
        PointingSampleRecord {
            id: Uuid::new_v4(),
            plate_solve_event_id: Some("ps-1".into()),
            frame_id: Some("frame-1".into()),
            mount_identity_key: "alpaca:esp32-mount-1".into(),
            mount_frame: EqFrame::JNowApparent,
            jnow_flavour: JNowFlavour::Apparent,
            does_refraction: None,
            mount_ra_native_h: Some(5.512),
            mount_dec_native_deg: Some(22.01),
            mount_ra_j2000_h: Some(5.497),
            mount_dec_j2000_deg: Some(21.99),
            solved_ra_j2000_h: 5.5,
            solved_dec_j2000_deg: 22.0,
            err_ra_arcsec: Some(-37.4),
            err_dec_arcsec: Some(36.0),
            err_total_arcsec: Some(51.9),
            pier_side: PierSide::Unknown,
            ha_h: Some(-1.2),
            lst_h: Some(4.3),
            alt_deg: Some(55.0),
            az_deg: Some(120.0),
            sync_generation: generation,
            sync_mode: MountSyncMode::Offset,
            coordinate_precision_arcsec: None,
            temperature_c: Some(7.0),
            mid_exposure_utc: Some(Utc.with_ymd_and_hms(2026, 10, 26, 4, 1, 0).unwrap()),
            purpose: PointingPurpose::Center,
        }
    }

    #[tokio::test]
    async fn samples_keep_frame_flavour_pier_side_and_generation() {
        let pool = crate::storage::test_pool().await;
        let a = sample(0);
        let b = sample(1);
        upsert(&pool, &a).await.unwrap();
        upsert(&pool, &b).await.unwrap();
        let all = for_mount(&pool, "alpaca:esp32-mount-1", None, 10)
            .await
            .unwrap();
        assert_eq!(all.len(), 2);
        let gen1 = for_mount(&pool, "alpaca:esp32-mount-1", Some(1), 10)
            .await
            .unwrap();
        assert_eq!(gen1, vec![b]);
        assert_eq!(for_frame(&pool, "frame-1").await.unwrap().len(), 2);
        let raw: (String, String, Option<String>, Option<i64>, String) = sqlx::query_as(
            "SELECT mount_frame, jnow_flavour, pier_side, does_refraction, sync_mode \
             FROM pointing_samples WHERE id = ?",
        )
        .bind(a.id.to_string())
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(
            raw,
            (
                "jnow_apparent".to_string(),
                "apparent".to_string(),
                None,
                None,
                "offset".to_string()
            )
        );
    }
}
