//! `guide_steps`: every PHD2 GuideStep, stored unmodified (equipment spec
//! §3.5 G-h, H9; migration 009). PHD2's signs, its `Time` and its event
//! `Timestamp` are kept as they arrived, beside the daemon receive time and
//! the phase tag. Steps are tagged, never dropped, and are joined to
//! exposures afterwards through windows (M-1, M-2, TED, TME).

// Consumed by the recorder and, once the spine wires the guider (WP1B-1), by
// window statistics and the M-1 context builder.
#![allow(dead_code)]

use chrono::{DateTime, Utc};
use sqlx::{Executor, Row, Sqlite};
use uuid::Uuid;

use crate::equipment::events::GuideStepRecord;
use crate::equipment::snapshot::{GuidePhase, GuideStepSample};
use crate::storage::columns;

/// Append one step and return its row id.
pub async fn insert<'c, E>(executor: E, record: &GuideStepRecord) -> Result<i64, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let s = &record.step;
    let result = sqlx::query(
        r#"
        INSERT INTO guide_steps (
            connection_id, frame, phd2_time_s, phd2_timestamp, received_at_utc, mount,
            dx, dy, ra_raw, dec_raw, ra_guide, dec_guide, ra_duration_ms, ra_direction,
            dec_duration_ms, dec_direction, star_mass, snr, hfd, avg_dist, ra_limited,
            dec_limited, error_code, phase, pixel_scale_arcsec
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(record.connection_id.to_string())
    .bind(columns::int(s.frame))
    .bind(s.phd2_time_s)
    .bind(s.phd2_timestamp)
    .bind(columns::time(&s.received_at_utc))
    .bind(&s.mount)
    .bind(s.dx)
    .bind(s.dy)
    .bind(s.ra_raw)
    .bind(s.dec_raw)
    .bind(s.ra_guide)
    .bind(s.dec_guide)
    .bind(s.ra_duration_ms)
    .bind(&s.ra_direction)
    .bind(s.dec_duration_ms)
    .bind(&s.dec_direction)
    .bind(s.star_mass)
    .bind(s.snr)
    .bind(s.hfd)
    .bind(s.avg_dist)
    .bind(s.ra_limited)
    .bind(s.dec_limited)
    .bind(s.error_code)
    .bind(columns::name(&s.phase))
    .bind(s.pixel_scale_arcsec)
    .execute(executor)
    .await?;
    Ok(result.last_insert_rowid())
}

/// One stored step.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredGuideStep {
    pub id: i64,
    pub connection_id: Option<Uuid>,
    pub step: GuideStepSample,
}

/// Steps received in `[start, end]` (daemon receive time), oldest first,
/// optionally for one guider connection.
pub async fn between<'c, E>(
    executor: E,
    connection_id: Option<Uuid>,
    start: &DateTime<Utc>,
    end: &DateTime<Utc>,
) -> Result<Vec<StoredGuideStep>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let rows = sqlx::query(
        r#"
        SELECT * FROM guide_steps
        WHERE received_at_utc >= ?1 AND received_at_utc <= ?2
          AND (?3 IS NULL OR connection_id = ?3)
        ORDER BY received_at_utc, id
        "#,
    )
    .bind(columns::time(start))
    .bind(columns::time(end))
    .bind(connection_id.map(|id| id.to_string()))
    .fetch_all(executor)
    .await?;
    Ok(rows
        .iter()
        .map(|row| {
            let received: String = row.get("received_at_utc");
            let phase: Option<String> = row.get("phase");
            let connection: Option<String> = row.get("connection_id");
            StoredGuideStep {
                id: row.get("id"),
                connection_id: connection.and_then(|id| Uuid::parse_str(&id).ok()),
                step: GuideStepSample {
                    frame: columns::uint(row.get("frame")),
                    phd2_time_s: row.get("phd2_time_s"),
                    phd2_timestamp: row.get("phd2_timestamp"),
                    received_at_utc: columns::parse_time(&received).unwrap_or_default(),
                    mount: row.get("mount"),
                    dx: row.get("dx"),
                    dy: row.get("dy"),
                    ra_raw: row.get("ra_raw"),
                    dec_raw: row.get("dec_raw"),
                    ra_guide: row.get("ra_guide"),
                    dec_guide: row.get("dec_guide"),
                    ra_duration_ms: row.get("ra_duration_ms"),
                    ra_direction: row.get("ra_direction"),
                    dec_duration_ms: row.get("dec_duration_ms"),
                    dec_direction: row.get("dec_direction"),
                    star_mass: row.get("star_mass"),
                    snr: row.get("snr"),
                    hfd: row.get("hfd"),
                    avg_dist: row.get("avg_dist"),
                    ra_limited: row.get("ra_limited"),
                    dec_limited: row.get("dec_limited"),
                    error_code: row.get("error_code"),
                    phase: columns::from_name::<GuidePhase>(phase.as_deref()).unwrap_or_default(),
                    pixel_scale_arcsec: row.get("pixel_scale_arcsec"),
                },
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[tokio::test]
    async fn steps_are_stored_unmodified() {
        let pool = crate::storage::test_pool().await;
        let at = Utc.with_ymd_and_hms(2026, 10, 26, 3, 0, 0).unwrap();
        let connection = Uuid::new_v4();
        let step = GuideStepSample {
            frame: 1234,
            phd2_time_s: 812.25,
            phd2_timestamp: Some(1_793_000_000.125),
            received_at_utc: at,
            mount: Some("Mount".into()),
            // Negative signs are PHD2's and must survive untouched.
            dx: -0.4375,
            dy: 0.125,
            ra_raw: Some(-0.33),
            dec_raw: Some(0.21),
            ra_guide: Some(-0.2),
            dec_guide: None,
            ra_duration_ms: Some(140),
            ra_direction: Some("West".into()),
            dec_duration_ms: None,
            dec_direction: None,
            star_mass: Some(5400.0),
            snr: Some(38.5),
            hfd: Some(2.9),
            avg_dist: Some(0.31),
            ra_limited: Some(false),
            dec_limited: None,
            error_code: None,
            phase: GuidePhase::Settling,
            pixel_scale_arcsec: Some(2.3),
        };
        let id = insert(
            &pool,
            &GuideStepRecord {
                connection_id: connection,
                step: step.clone(),
            },
        )
        .await
        .unwrap();
        assert!(id > 0);

        let stored = between(&pool, Some(connection), &at, &at).await.unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].connection_id, Some(connection));
        assert_eq!(stored[0].step, step);
        assert!(between(&pool, Some(Uuid::new_v4()), &at, &at)
            .await
            .unwrap()
            .is_empty());
        let raw: (String, Option<i64>) =
            sqlx::query_as("SELECT phase, dec_limited FROM guide_steps")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(raw, ("settling".to_string(), None));
    }
}
