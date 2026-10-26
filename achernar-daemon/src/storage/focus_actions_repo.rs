//! focus_actions_repo — Section 7.4; equipment spec H4, §4.3.
//!
//! `focus_actions`: the moves an autofocus run commanded. Migration 009 adds
//! the verification frame taken after the final move (H4): its id and HFR,
//! whether the run was verified, and the focuser move that reached the
//! position. FDF updates only from a verified, non-simulator run (H5); FOP
//! labels need the verification too.
//!
//! The autofocus integration (WP1B-2) writes actions through [`insert`] and
//! records the verification with [`record_verification`].

// Consumed once autofocus is rewired onto the equipment layer (WP1B-2).
#![allow(dead_code)]

use serde::Serialize;
use sqlx::{Executor, Row, Sqlite};
use uuid::Uuid;

/// One commanded action.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct FocusActionRow {
    pub run_id: Option<String>,
    /// e.g. `move_to_best`.
    pub action: String,
    pub from_position: Option<i32>,
    pub to_position: Option<i32>,
    pub steps: Option<i32>,
    pub reason: Option<String>,
    pub focuser_move_id: Option<Uuid>,
}

/// The H4 verification of an action.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct FocusVerification {
    pub verification_frame_id: Option<String>,
    pub verification_hfr: Option<f64>,
    pub verified: bool,
}

/// Append one action and return its id.
pub async fn insert<'c, E>(executor: E, action: &FocusActionRow) -> Result<i64, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let result = sqlx::query(
        "INSERT INTO focus_actions \
         (run_id, action, from_position, to_position, steps, reason, focuser_move_id) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&action.run_id)
    .bind(&action.action)
    .bind(action.from_position)
    .bind(action.to_position)
    .bind(action.steps)
    .bind(&action.reason)
    .bind(action.focuser_move_id.map(|id| id.to_string()))
    .execute(executor)
    .await?;
    Ok(result.last_insert_rowid())
}

/// Record the verification frame of an action. Returns `false` when no such
/// action exists.
pub async fn record_verification<'c, E>(
    executor: E,
    action_id: i64,
    verification: &FocusVerification,
) -> Result<bool, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let result = sqlx::query(
        "UPDATE focus_actions SET verification_frame_id = ?, verification_hfr = ?, verified = ? \
         WHERE id = ?",
    )
    .bind(&verification.verification_frame_id)
    .bind(verification.verification_hfr.filter(|v| v.is_finite()))
    .bind(verification.verified)
    .bind(action_id)
    .execute(executor)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// The verification of one action: `None` when there is no such action, and
/// `Some(None)` when it has not been verified yet.
pub async fn load_verification<'c, E>(
    executor: E,
    action_id: i64,
) -> Result<Option<Option<FocusVerification>>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let row = sqlx::query(
        "SELECT verification_frame_id, verification_hfr, verified FROM focus_actions WHERE id = ?",
    )
    .bind(action_id)
    .fetch_optional(executor)
    .await?;
    Ok(row.map(|row| {
        let verified: Option<bool> = row.get("verified");
        verified.map(|verified| FocusVerification {
            verification_frame_id: row.get("verification_frame_id"),
            verification_hfr: row.get("verification_hfr"),
            verified,
        })
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn an_action_is_verified_after_the_final_move() {
        let pool = crate::storage::test_pool().await;
        let id = insert(
            &pool,
            &FocusActionRow {
                run_id: None,
                action: "move_to_best".into(),
                from_position: Some(15_400),
                to_position: Some(15_210),
                steps: Some(-190),
                reason: Some("vcurve".into()),
                focuser_move_id: Some(Uuid::new_v4()),
            },
        )
        .await
        .unwrap();
        assert_eq!(load_verification(&pool, id).await.unwrap(), Some(None));
        let verification = FocusVerification {
            verification_frame_id: Some("frame-v1".into()),
            verification_hfr: Some(2.1),
            verified: true,
        };
        assert!(record_verification(&pool, id, &verification).await.unwrap());
        assert_eq!(
            load_verification(&pool, id).await.unwrap(),
            Some(Some(verification))
        );
        assert!(
            !record_verification(&pool, id + 1, &FocusVerification::default())
                .await
                .unwrap()
        );
        assert_eq!(load_verification(&pool, id + 1).await.unwrap(), None);
    }
}
