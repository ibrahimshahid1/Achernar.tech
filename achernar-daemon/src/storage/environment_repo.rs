//! `environment_log` probe rows (equipment spec §3.9, F-f, §4.3; migrations
//! 006 and 009). Weather devices (ObservingConditions) and focuser probes
//! write rows with `source = 'probe'`, the device identity and the age of
//! each sensor. Operator logs (`manual_log`) and imports (`forecast`) share
//! the table and are never written here. FDF, FOP #12 and #37 and COC
//! corroboration read these rows; a value the device did not report stays
//! NULL.

// Consumed by the recorder and, once weather and focuser probes are wired
// (WP1A-8, WP1A-5), by FDF and the FOP feature builder.
#![allow(dead_code)]

use chrono::{DateTime, Utc};
use sqlx::{Executor, Row, Sqlite};

use crate::equipment::events::EnvironmentRecord;
use crate::storage::columns;

/// The `source` value of every row this module writes.
pub const PROBE_SOURCE: &str = "probe";

/// Append one probe row and return its id. `night_id` fills in the record's
/// own night when it has none.
pub async fn insert_probe<'c, E>(
    executor: E,
    record: &EnvironmentRecord,
    night_id: Option<&str>,
) -> Result<i64, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let result = sqlx::query(
        r#"
        INSERT INTO environment_log (
            session_id, night_id, rig_id, observed_at, ambient_temp_c, humidity_pct,
            dew_point_c, pressure_hpa, cloud_cover_pct, sky_temp_c, sky_quality,
            sky_brightness_lux, star_fwhm_arcsec, rain_rate_mm_h, wind_speed_ms,
            wind_gust_ms, wind_dir_deg, sensor_ages_json, device_identity_key, source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(&record.session_id)
    .bind(record.night_id.as_deref().or(night_id))
    .bind(&record.rig_id)
    .bind(columns::time(&record.observed_at))
    .bind(finite(record.ambient_temp_c))
    .bind(finite(record.humidity_pct))
    .bind(finite(record.dew_point_c))
    .bind(finite(record.pressure_hpa))
    .bind(finite(record.cloud_cover_pct))
    .bind(finite(record.sky_temp_c))
    .bind(finite(record.sky_quality))
    .bind(finite(record.sky_brightness_lux))
    .bind(finite(record.star_fwhm_arcsec))
    .bind(finite(record.rain_rate_mm_h))
    .bind(finite(record.wind_speed_ms))
    .bind(finite(record.wind_gust_ms))
    .bind(finite(record.wind_dir_deg))
    .bind(columns::json(&record.sensor_ages_s))
    .bind(&record.device_identity_key)
    .bind(PROBE_SOURCE)
    .execute(executor)
    .await?;
    Ok(result.last_insert_rowid())
}

/// NaN and infinities are unknown, never numbers (§3.9).
fn finite(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite())
}

/// Probe rows observed in `[start, end]`, oldest first, optionally for one
/// device identity.
pub async fn probe_rows_between<'c, E>(
    executor: E,
    device_identity_key: Option<&str>,
    start: &DateTime<Utc>,
    end: &DateTime<Utc>,
) -> Result<Vec<EnvironmentRecord>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let rows = sqlx::query(
        r#"
        SELECT * FROM environment_log
        WHERE source = 'probe' AND observed_at >= ?1 AND observed_at <= ?2
          AND (?3 IS NULL OR device_identity_key = ?3)
        ORDER BY observed_at, id
        "#,
    )
    .bind(columns::time(start))
    .bind(columns::time(end))
    .bind(device_identity_key)
    .fetch_all(executor)
    .await?;
    Ok(rows
        .iter()
        .filter_map(|row| {
            let observed: String = row.get("observed_at");
            let ages: Option<String> = row.get("sensor_ages_json");
            Some(EnvironmentRecord {
                observed_at: columns::parse_time(&observed)?,
                session_id: row.get("session_id"),
                night_id: row.get("night_id"),
                rig_id: row.get("rig_id"),
                device_identity_key: row.get("device_identity_key"),
                ambient_temp_c: row.get("ambient_temp_c"),
                humidity_pct: row.get("humidity_pct"),
                dew_point_c: row.get("dew_point_c"),
                pressure_hpa: row.get("pressure_hpa"),
                cloud_cover_pct: row.get("cloud_cover_pct"),
                sky_temp_c: row.get("sky_temp_c"),
                sky_quality: row.get("sky_quality"),
                sky_brightness_lux: row.get("sky_brightness_lux"),
                star_fwhm_arcsec: row.get("star_fwhm_arcsec"),
                rain_rate_mm_h: row.get("rain_rate_mm_h"),
                wind_speed_ms: row.get("wind_speed_ms"),
                wind_gust_ms: row.get("wind_gust_ms"),
                wind_dir_deg: row.get("wind_dir_deg"),
                sensor_ages_s: columns::parse_json(ages.as_deref()).unwrap_or_default(),
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[tokio::test]
    async fn probe_rows_keep_only_known_values_and_their_ages() {
        let pool = crate::storage::test_pool().await;
        let at = Utc.with_ymd_and_hms(2026, 10, 26, 2, 30, 0).unwrap();
        let rec = EnvironmentRecord {
            observed_at: at,
            device_identity_key: Some("alpaca:oc-1".into()),
            ambient_temp_c: Some(5.25),
            humidity_pct: Some(81.0),
            sky_temp_c: Some(-18.5),
            wind_speed_ms: Some(f64::NAN),
            sensor_ages_s: [("Temperature".to_string(), 4.0)].into_iter().collect(),
            ..EnvironmentRecord::default()
        };
        insert_probe(&pool, &rec, Some("2026-10-25")).await.unwrap();
        // An operator row in the same table is not a probe row.
        sqlx::query(
            "INSERT INTO environment_log (observed_at, ambient_temp_c, source) \
             VALUES (?, 9.0, 'manual_log')",
        )
        .bind(columns::time(&at))
        .execute(&pool)
        .await
        .unwrap();

        let rows = probe_rows_between(&pool, Some("alpaca:oc-1"), &at, &at)
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.ambient_temp_c, Some(5.25));
        assert_eq!(row.wind_speed_ms, None, "NaN is unknown");
        assert_eq!(row.pressure_hpa, None);
        assert_eq!(row.night_id.as_deref(), Some("2026-10-25"));
        assert_eq!(row.sensor_ages_s["Temperature"], 4.0);
        let source: (String,) =
            sqlx::query_as("SELECT source FROM environment_log WHERE device_identity_key = ?")
                .bind("alpaca:oc-1")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(source.0, "probe");
    }
}
