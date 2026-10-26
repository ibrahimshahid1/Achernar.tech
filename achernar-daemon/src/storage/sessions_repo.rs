//! sessions_repo — Section 5.2; equipment spec §4.3 "Existing tables that get
//! their first writer".
//!
//! `sessions`: an observing session, and the site it was observed from. The
//! `location_*` columns take the configured `[site]`, or the mount's site when
//! `[site] sync = "from_mount"` adopts it (§3.2 M-b). An unknown site stays
//! NULL: no location is ever synthesised, so geometry consumers report "site
//! unknown" rather than computing from a default. Longitude is east-positive,
//! elevation in metres, as in [`Site`].

// Consumed once sessions are opened by the spine (WP1B-1) and the recorder
// is given a session id.
#![allow(dead_code)]

use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::{Executor, Row, Sqlite};

use crate::equipment::types::Site;
use crate::storage::columns;

/// `sessions.status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    Active,
    Completed,
    Aborted,
}

impl SessionStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            SessionStatus::Active => "active",
            SessionStatus::Completed => "completed",
            SessionStatus::Aborted => "aborted",
        }
    }
}

/// A session to open.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NewSession {
    pub id: String,
    pub name: Option<String>,
    pub rig_id: Option<String>,
    pub target_name: Option<String>,
    pub target_ra: Option<f64>,
    pub target_dec: Option<f64>,
    pub site: Option<Site>,
    pub started_at: Option<DateTime<Utc>>,
    pub notes: Option<String>,
}

/// Open a session (status `active`).
pub async fn open<'c, E>(executor: E, session: &NewSession) -> Result<(), sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let site = valid(session.site.as_ref());
    sqlx::query(
        r#"
        INSERT INTO sessions (
            id, name, rig_id, target_name, target_ra, target_dec, location_lat,
            location_lon, location_alt, started_at, notes, status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, datetime('now')), ?, 'active')
        "#,
    )
    .bind(&session.id)
    .bind(&session.name)
    .bind(&session.rig_id)
    .bind(&session.target_name)
    .bind(session.target_ra)
    .bind(session.target_dec)
    .bind(site.map(|s| s.lat_deg))
    .bind(site.map(|s| s.lon_deg))
    .bind(site.map(|s| s.elev_m))
    .bind(columns::opt_time(session.started_at.as_ref()))
    .bind(&session.notes)
    .execute(executor)
    .await?;
    Ok(())
}

/// A site that fails validation is unknown, never stored.
fn valid(site: Option<&Site>) -> Option<&Site> {
    site.filter(|site| site.validate().is_ok())
}

/// Set (or clear, with `None`) the session's site. Returns `false` when no
/// such session exists.
pub async fn set_location<'c, E>(
    executor: E,
    session_id: &str,
    site: Option<&Site>,
) -> Result<bool, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let site = valid(site);
    let result = sqlx::query(
        "UPDATE sessions SET location_lat = ?, location_lon = ?, location_alt = ? WHERE id = ?",
    )
    .bind(site.map(|s| s.lat_deg))
    .bind(site.map(|s| s.lon_deg))
    .bind(site.map(|s| s.elev_m))
    .bind(session_id)
    .execute(executor)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// End an active session. Returns `false` when it was not active.
pub async fn close<'c, E>(
    executor: E,
    session_id: &str,
    status: SessionStatus,
    ended_at: &DateTime<Utc>,
) -> Result<bool, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let result = sqlx::query(
        "UPDATE sessions SET status = ?, ended_at = ? WHERE id = ? AND status = 'active'",
    )
    .bind(status.as_str())
    .bind(columns::time(ended_at))
    .bind(session_id)
    .execute(executor)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// One stored session. `location` is `None` unless all three columns are set.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StoredSession {
    pub id: String,
    pub name: Option<String>,
    pub rig_id: Option<String>,
    pub target_name: Option<String>,
    pub target_ra: Option<f64>,
    pub target_dec: Option<f64>,
    pub location_lat: Option<f64>,
    pub location_lon: Option<f64>,
    pub location_alt: Option<f64>,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub notes: Option<String>,
    pub status: String,
}

fn from_row(row: &sqlx::sqlite::SqliteRow) -> StoredSession {
    StoredSession {
        id: row.get("id"),
        name: row.get("name"),
        rig_id: row.get("rig_id"),
        target_name: row.get("target_name"),
        target_ra: row.get("target_ra"),
        target_dec: row.get("target_dec"),
        location_lat: row.get("location_lat"),
        location_lon: row.get("location_lon"),
        location_alt: row.get("location_alt"),
        started_at: row.get("started_at"),
        ended_at: row.get("ended_at"),
        notes: row.get("notes"),
        status: row.get("status"),
    }
}

pub async fn load<'c, E>(
    executor: E,
    session_id: &str,
) -> Result<Option<StoredSession>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let row = sqlx::query("SELECT * FROM sessions WHERE id = ?")
        .bind(session_id)
        .fetch_optional(executor)
        .await?;
    Ok(row.as_ref().map(from_row))
}

/// The most recently started active session, if any.
pub async fn active<'c, E>(executor: E) -> Result<Option<StoredSession>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let row = sqlx::query(
        "SELECT * FROM sessions WHERE status = 'active' ORDER BY started_at DESC, rowid DESC \
         LIMIT 1",
    )
    .fetch_optional(executor)
    .await?;
    Ok(row.as_ref().map(from_row))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::types::SiteSource;
    use chrono::TimeZone;

    #[tokio::test]
    async fn a_session_records_its_site_and_closes_once() {
        let pool = crate::storage::test_pool().await;
        let t0 = Utc.with_ymd_and_hms(2026, 10, 26, 0, 30, 0).unwrap();
        open(
            &pool,
            &NewSession {
                id: "s1".into(),
                name: Some("M42".into()),
                started_at: Some(t0),
                ..NewSession::default()
            },
        )
        .await
        .unwrap();
        let stored = load(&pool, "s1").await.unwrap().unwrap();
        assert_eq!(stored.location_lat, None, "no site configured: unknown");
        assert_eq!(stored.status, "active");
        assert_eq!(stored.started_at, "2026-10-26T00:30:00.000Z");

        let site = Site {
            lat_deg: 35.9,
            lon_deg: -79.05,
            elev_m: 150.0,
            source: SiteSource::Mount,
        };
        assert!(set_location(&pool, "s1", Some(&site)).await.unwrap());
        let stored = active(&pool).await.unwrap().unwrap();
        assert_eq!(
            (
                stored.location_lat,
                stored.location_lon,
                stored.location_alt
            ),
            (Some(35.9), Some(-79.05), Some(150.0))
        );
        // An invalid site is unknown, never stored.
        let bad = Site {
            lat_deg: 95.0,
            ..site
        };
        assert!(set_location(&pool, "s1", Some(&bad)).await.unwrap());
        assert_eq!(load(&pool, "s1").await.unwrap().unwrap().location_lat, None);
        assert!(!set_location(&pool, "nope", Some(&site)).await.unwrap());

        let t1 = t0 + chrono::Duration::hours(6);
        assert!(close(&pool, "s1", SessionStatus::Completed, &t1)
            .await
            .unwrap());
        assert!(!close(&pool, "s1", SessionStatus::Aborted, &t1)
            .await
            .unwrap());
        let stored = load(&pool, "s1").await.unwrap().unwrap();
        assert_eq!(stored.status, "completed");
        assert_eq!(stored.ended_at.as_deref(), Some("2026-10-26T06:30:00.000Z"));
        assert!(active(&pool).await.unwrap().is_none());
    }
}
