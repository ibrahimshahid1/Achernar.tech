//! focus_drift_repo — Section 7.6.3.
//!
//! Persists two different things with two different lifetimes:
//!
//! * **Decisions** — one `focus_drift_events` row per prediction that mattered,
//!   an append-only audit of what the forecaster advised and on what evidence.
//! * **Filter state** — a single JSON snapshot of the Kalman state, overwritten
//!   as it learns, so a daemon restart mid-session does not throw away a night
//!   of temperature-focus correlation.
//!
//! The snapshot deliberately does not live in `models/`. What one telescope
//! learned last night is not a shipped artefact: it is per-rig, it changes
//! continuously, and committing it would mean every clone starts with another
//! observatory's thermal model.

use crate::focus::focus_drift::{DriftPrediction, DriftState};
use sqlx::SqlitePool;
use std::path::Path;

/// Record a forecaster evaluation.
///
/// `temperature` is `None` when no reading was available. It is stored as NULL
/// rather than as a plausible-looking number, because a drift model fed
/// invented temperatures learns the invention. See
/// `capture_store::TemperatureState`.
pub async fn record_event(
    pool: &SqlitePool,
    session_id: Option<&str>,
    prediction: &DriftPrediction,
    temperature: Option<f64>,
) -> Result<String, sqlx::Error> {
    let id = uuid::Uuid::new_v4().to_string();
    sqlx::query(
        r#"
        INSERT INTO focus_drift_events (
            id, session_id, predicted_drift, temperature, confidence, refocus_triggered
        ) VALUES (?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(&id)
    .bind(session_id)
    .bind(prediction.predicted_drift_steps)
    .bind(temperature)
    // `exceed_probability`, not a made-up score: the column records the
    // confidence the decision was actually taken on. NULL when no refocus
    // threshold was known — the drift was still predicted, but nothing was
    // decided, and a 0.0 there would read as "decided, and confident it is
    // fine".
    .bind(prediction.exceed_probability)
    .bind(prediction.refocus_recommended.map(i32::from))
    .execute(pool)
    .await?;
    Ok(id)
}

/// Most recent events first, for the operator-facing history.
pub async fn recent_events(
    pool: &SqlitePool,
    limit: i64,
) -> Result<Vec<DriftEventRow>, sqlx::Error> {
    sqlx::query_as::<_, DriftEventRow>(
        r#"
        SELECT id, session_id, predicted_drift, temperature, confidence,
               refocus_triggered, created_at
        FROM focus_drift_events
        ORDER BY created_at DESC, rowid DESC
        LIMIT ?
        "#,
    )
    .bind(limit.clamp(1, 1000))
    .fetch_all(pool)
    .await
}

#[derive(Debug, Clone, sqlx::FromRow, serde::Serialize)]
pub struct DriftEventRow {
    pub id: String,
    pub session_id: Option<String>,
    pub predicted_drift: f64,
    pub temperature: Option<f64>,
    pub confidence: Option<f64>,
    pub refocus_triggered: i64,
    pub created_at: String,
}

/// Write the filter state, atomically.
///
/// Staged to a sibling `.tmp` and renamed, because the alternative — truncating
/// the real file and writing into it — leaves a half-written snapshot if the
/// process dies mid-write, and `from_state` would reject it on the next start.
/// Losing the snapshot costs a night of relearning; the rename makes that
/// impossible rather than merely unlikely.
pub fn save_state(path: &Path, state: &DriftState) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let json = serde_json::to_string_pretty(state)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json.as_bytes())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Read a filter state snapshot.
pub fn load_state(path: &Path) -> Result<DriftState, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&text).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state() -> DriftState {
        DriftState {
            focus_position: 12_345.0,
            drift_rate: -0.002,
            temperature_coefficient: 18.5,
            covariance: vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 400.0],
            last_temperature: Some(4.5),
            last_update: Some(chrono::Utc::now()),
            temperature_rate_c_per_s: -1.0e-4,
            update_count: 7,
        }
    }

    #[test]
    fn state_round_trips_through_disk() {
        let dir = std::env::temp_dir().join(format!("achernar-drift-{}", uuid::Uuid::new_v4()));
        let path = dir.join("focus_drift_state.json");

        let original = sample_state();
        save_state(&path, &original).expect("save");
        let restored = load_state(&path).expect("load");

        assert_eq!(restored.update_count, original.update_count);
        assert_eq!(restored.covariance.len(), 9);
        assert!((restored.temperature_coefficient - original.temperature_coefficient).abs() < 1e-9);
        assert_eq!(restored.last_temperature, original.last_temperature);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_leaves_no_temp_file_behind() {
        let dir = std::env::temp_dir().join(format!("achernar-drift-{}", uuid::Uuid::new_v4()));
        let path = dir.join("focus_drift_state.json");
        save_state(&path, &sample_state()).expect("save");

        assert!(path.exists());
        assert!(
            !path.with_extension("json.tmp").exists(),
            "staging file must be renamed, not left alongside the real one"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
