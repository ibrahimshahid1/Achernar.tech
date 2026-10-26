//! `exposure_device_windows`: device state over each committed exposure
//! (equipment spec H1, D7, §4.3; migration 009). Built by the capture
//! integration (WP1B-3) from a `DeviceWindow` after the exposure, so capture
//! adds no device I/O. Every v6 model reads ground truth from here: mount
//! start, mid and end state, tracking and slewing, focuser motion, the
//! verified filter, guiding statistics, safety, weather and cover state.
//!
//! The row references `frames(id)`, so it is written after the frame commit.
//! A window is rewritten in place when it is finalised (D7). Missing values
//! are NULL; they are never written as 0.

// Consumed by the recorder and, once capture is wired (WP1B-3), by the
// frame-provenance readers.
#![allow(dead_code)]

use sqlx::{Executor, Row, Sqlite};

use crate::equipment::events::ExposureWindowRow;
use crate::storage::columns;

/// Insert or replace the window for one frame.
pub async fn upsert<'c, E>(executor: E, row: &ExposureWindowRow) -> Result<(), sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let opt_json = |value: &Option<serde_json::Value>| value.as_ref().map(columns::json);
    sqlx::query(
        r#"
        INSERT INTO exposure_device_windows (
            frame_id, exposure_started_at, exposure_completed_at, clock_quality, finalised,
            mount_start_json, mount_mid_json, mount_end_json, mount_frame, pier_side,
            tracking, slewing_any, pulse_guiding_any, focuser_pos_start, focuser_pos_end,
            focuser_moved, focuser_temperature_c, filter_slot, filter_name, filter_verified,
            rotator_angle_deg, guiding_state, guide_rms_ra_px, guide_rms_dec_px,
            guide_rms_total_arcsec, guide_rms0_total_arcsec, guide_peak_ra_px,
            guide_peak_dec_px, guide_steps_n, dither_in_window, settle_in_window,
            safety_state, weather_json, cover_json, resume_block_active, quiescent,
            quiescence_reason, events_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(frame_id) DO UPDATE SET
            exposure_started_at     = excluded.exposure_started_at,
            exposure_completed_at   = excluded.exposure_completed_at,
            clock_quality           = excluded.clock_quality,
            finalised               = excluded.finalised,
            mount_start_json        = excluded.mount_start_json,
            mount_mid_json          = excluded.mount_mid_json,
            mount_end_json          = excluded.mount_end_json,
            mount_frame             = excluded.mount_frame,
            pier_side               = excluded.pier_side,
            tracking                = excluded.tracking,
            slewing_any             = excluded.slewing_any,
            pulse_guiding_any       = excluded.pulse_guiding_any,
            focuser_pos_start       = excluded.focuser_pos_start,
            focuser_pos_end         = excluded.focuser_pos_end,
            focuser_moved           = excluded.focuser_moved,
            focuser_temperature_c   = excluded.focuser_temperature_c,
            filter_slot             = excluded.filter_slot,
            filter_name             = excluded.filter_name,
            filter_verified         = excluded.filter_verified,
            rotator_angle_deg       = excluded.rotator_angle_deg,
            guiding_state           = excluded.guiding_state,
            guide_rms_ra_px         = excluded.guide_rms_ra_px,
            guide_rms_dec_px        = excluded.guide_rms_dec_px,
            guide_rms_total_arcsec  = excluded.guide_rms_total_arcsec,
            guide_rms0_total_arcsec = excluded.guide_rms0_total_arcsec,
            guide_peak_ra_px        = excluded.guide_peak_ra_px,
            guide_peak_dec_px       = excluded.guide_peak_dec_px,
            guide_steps_n           = excluded.guide_steps_n,
            dither_in_window        = excluded.dither_in_window,
            settle_in_window        = excluded.settle_in_window,
            safety_state            = excluded.safety_state,
            weather_json            = excluded.weather_json,
            cover_json              = excluded.cover_json,
            resume_block_active     = excluded.resume_block_active,
            quiescent               = excluded.quiescent,
            quiescence_reason       = excluded.quiescence_reason,
            events_json             = excluded.events_json
        "#,
    )
    .bind(&row.frame_id)
    .bind(columns::opt_time(row.exposure_started_at.as_ref()))
    .bind(columns::opt_time(row.exposure_completed_at.as_ref()))
    .bind(columns::name(&row.clock_quality))
    .bind(row.finalised)
    .bind(opt_json(&row.mount_start_json))
    .bind(opt_json(&row.mount_mid_json))
    .bind(opt_json(&row.mount_end_json))
    .bind(row.mount_frame.map(|frame| frame.as_str()))
    .bind(row.pier_side.and_then(|side| side.as_str()))
    .bind(row.tracking)
    .bind(row.slewing_any)
    .bind(row.pulse_guiding_any)
    .bind(row.focuser_pos_start)
    .bind(row.focuser_pos_end)
    .bind(row.focuser_moved)
    .bind(row.focuser_temperature_c)
    .bind(row.filter_slot.map(i64::from))
    .bind(&row.filter_name)
    .bind(row.filter_verified)
    .bind(row.rotator_angle_deg)
    .bind(row.guiding_state.as_ref().and_then(columns::name))
    .bind(row.guide_rms_ra_px)
    .bind(row.guide_rms_dec_px)
    .bind(row.guide_rms_total_arcsec)
    .bind(row.guide_rms0_total_arcsec)
    .bind(row.guide_peak_ra_px)
    .bind(row.guide_peak_dec_px)
    .bind(row.guide_steps_n.map(i64::from))
    .bind(row.dither_in_window)
    .bind(row.settle_in_window)
    .bind(row.safety_state.as_ref().and_then(columns::name))
    .bind(opt_json(&row.weather_json))
    .bind(opt_json(&row.cover_json))
    .bind(row.resume_block_active)
    .bind(row.quiescent)
    .bind(&row.quiescence_reason)
    .bind(columns::json(&row.events))
    .execute(executor)
    .await?;
    Ok(())
}

/// The window of one frame, rebuilt as the record that wrote it.
pub async fn load<'c, E>(
    executor: E,
    frame_id: &str,
) -> Result<Option<ExposureWindowRow>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let row = sqlx::query("SELECT * FROM exposure_device_windows WHERE frame_id = ?")
        .bind(frame_id)
        .fetch_optional(executor)
        .await?;
    Ok(row.map(|row| {
        let text = |name: &str| -> Option<String> { row.get(name) };
        let json = |name: &str| columns::parse_json::<serde_json::Value>(text(name).as_deref());
        let events: String = row.get("events_json");
        ExposureWindowRow {
            frame_id: row.get("frame_id"),
            exposure_started_at: columns::parse_opt_time(text("exposure_started_at")),
            exposure_completed_at: columns::parse_opt_time(text("exposure_completed_at")),
            clock_quality: columns::from_name(text("clock_quality").as_deref()).unwrap_or_default(),
            finalised: row.get("finalised"),
            mount_start_json: json("mount_start_json"),
            mount_mid_json: json("mount_mid_json"),
            mount_end_json: json("mount_end_json"),
            mount_frame: columns::from_name(text("mount_frame").as_deref()),
            pier_side: columns::from_name(text("pier_side").as_deref()),
            tracking: row.get("tracking"),
            slewing_any: row.get("slewing_any"),
            pulse_guiding_any: row.get("pulse_guiding_any"),
            focuser_pos_start: row.get("focuser_pos_start"),
            focuser_pos_end: row.get("focuser_pos_end"),
            focuser_moved: row.get("focuser_moved"),
            focuser_temperature_c: row.get("focuser_temperature_c"),
            filter_slot: row
                .get::<Option<i64>, _>("filter_slot")
                .and_then(|slot| u32::try_from(slot).ok()),
            filter_name: row.get("filter_name"),
            filter_verified: row.get("filter_verified"),
            rotator_angle_deg: row.get("rotator_angle_deg"),
            guiding_state: columns::from_name(text("guiding_state").as_deref()),
            guide_rms_ra_px: row.get("guide_rms_ra_px"),
            guide_rms_dec_px: row.get("guide_rms_dec_px"),
            guide_rms_total_arcsec: row.get("guide_rms_total_arcsec"),
            guide_rms0_total_arcsec: row.get("guide_rms0_total_arcsec"),
            guide_peak_ra_px: row.get("guide_peak_ra_px"),
            guide_peak_dec_px: row.get("guide_peak_dec_px"),
            guide_steps_n: row
                .get::<Option<i64>, _>("guide_steps_n")
                .and_then(|n| u32::try_from(n).ok()),
            dither_in_window: row.get("dither_in_window"),
            settle_in_window: row.get("settle_in_window"),
            safety_state: columns::from_name(text("safety_state").as_deref()),
            weather_json: json("weather_json"),
            cover_json: json("cover_json"),
            resume_block_active: row.get("resume_block_active"),
            quiescent: row.get("quiescent"),
            quiescence_reason: row.get("quiescence_reason"),
            events: serde_json::from_str(&events).unwrap_or_default(),
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::snapshot::{GuiderAppState, SafetyGate};
    use crate::equipment::types::{ClockQuality, EqFrame, PierSide};
    use chrono::{TimeZone, Utc};

    pub(crate) async fn insert_frame(pool: &sqlx::SqlitePool, id: &str) {
        sqlx::query("INSERT INTO frames (id, local_path, captured_at) VALUES (?, '/tmp/f', ?)")
            .bind(id)
            .bind("2026-10-26T04:00:00.000Z")
            .execute(pool)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn a_window_round_trips_and_is_finalised_in_place() {
        let pool = crate::storage::test_pool().await;
        insert_frame(&pool, "frame-7").await;
        let t0 = Utc.with_ymd_and_hms(2026, 10, 26, 4, 0, 0).unwrap();
        let mut row = ExposureWindowRow {
            frame_id: "frame-7".into(),
            exposure_started_at: Some(t0),
            exposure_completed_at: Some(t0 + chrono::Duration::seconds(120)),
            clock_quality: ClockQuality::HostTriggerBracket,
            finalised: false,
            mount_mid_json: Some(serde_json::json!({"ra_h": 5.5, "dec_deg": 22.0})),
            mount_frame: Some(EqFrame::JNowApparent),
            pier_side: Some(PierSide::West),
            tracking: Some(true),
            slewing_any: Some(false),
            focuser_pos_start: Some(15_200),
            focuser_pos_end: Some(15_200),
            focuser_moved: Some(false),
            filter_verified: Some(false),
            guiding_state: Some(GuiderAppState::Guiding),
            guide_rms_ra_px: Some(0.31),
            guide_rms0_total_arcsec: Some(0.92),
            guide_steps_n: Some(40),
            dither_in_window: Some(false),
            safety_state: Some(SafetyGate::None),
            resume_block_active: false,
            quiescent: Some(true),
            events: vec![3, 4, 9],
            ..ExposureWindowRow::default()
        };
        upsert(&pool, &row).await.unwrap();
        assert_eq!(load(&pool, "frame-7").await.unwrap().as_ref(), Some(&row));

        row.finalised = true;
        row.mount_end_json = Some(serde_json::json!({"ra_h": 5.51}));
        upsert(&pool, &row).await.unwrap();
        assert_eq!(load(&pool, "frame-7").await.unwrap(), Some(row));

        let raw: (Option<i64>, String, Option<String>) = sqlx::query_as(
            "SELECT rotator_angle_deg IS NULL, pier_side, guiding_state \
             FROM exposure_device_windows",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(
            raw,
            (Some(1), "west".to_string(), Some("guiding".to_string()))
        );
    }

    #[tokio::test]
    async fn a_window_needs_its_frame() {
        let pool = crate::storage::test_pool().await;
        let row = ExposureWindowRow {
            frame_id: "no-such-frame".into(),
            ..ExposureWindowRow::default()
        };
        assert!(upsert(&pool, &row).await.is_err());
    }
}
