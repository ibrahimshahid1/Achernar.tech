//! `policy_decisions`: the H23 decision record (equipment spec §4.2 H23,
//! §4.3; DC §7 "Decision"; migration 009). Every policy proposal (FW-2, M-1,
//! M-2, the learned M-3, the exposure bandit, FOR) writes one row: the
//! pre-decision context with measurement ages, the safe arms, the policy and
//! its version, propensities only when the choice was randomised, the
//! requested, validated and executed action, the reward components, and
//! whether it was interrupted or overridden.
//!
//! A decision is written when it is proposed and updated in place, by
//! `decision_id`, as its execution and reward become known.

// Consumed by the recorder and, once the policies are wired (WP1A-6,
// WP1B-4), by their training exports.
#![allow(dead_code)]

use sqlx::{Executor, Row, Sqlite};
use uuid::Uuid;

use crate::equipment::events::PolicyDecisionRecord;
use crate::equipment::types::CommandId;
use crate::storage::columns;

/// Insert a decision, or update it with the same `decision_id`. The original
/// `created_at` is kept.
pub async fn upsert<'c, E>(executor: E, record: &PolicyDecisionRecord) -> Result<(), sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let opt_json = |value: &Option<serde_json::Value>| value.as_ref().map(columns::json);
    sqlx::query(
        r#"
        INSERT INTO policy_decisions (
            decision_id, policy, policy_version, mode, context_json, measurement_ages_json,
            safe_arms_json, chosen_arm, propensities_json, requested_action_json,
            validated_action_json, executed_action_json, command_id, reward_components_json,
            attribution_window_start, attribution_window_end, interrupted, override,
            parent_frame_ids_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(decision_id) DO UPDATE SET
            policy                   = excluded.policy,
            policy_version           = excluded.policy_version,
            mode                     = excluded.mode,
            context_json             = excluded.context_json,
            measurement_ages_json    = excluded.measurement_ages_json,
            safe_arms_json           = excluded.safe_arms_json,
            chosen_arm               = excluded.chosen_arm,
            propensities_json        = excluded.propensities_json,
            requested_action_json    = excluded.requested_action_json,
            validated_action_json    = excluded.validated_action_json,
            executed_action_json     = excluded.executed_action_json,
            command_id               = excluded.command_id,
            reward_components_json   = excluded.reward_components_json,
            attribution_window_start = excluded.attribution_window_start,
            attribution_window_end   = excluded.attribution_window_end,
            interrupted              = excluded.interrupted,
            override                 = excluded.override,
            parent_frame_ids_json    = excluded.parent_frame_ids_json
        "#,
    )
    .bind(record.decision_id.to_string())
    .bind(&record.policy)
    .bind(&record.policy_version)
    .bind(columns::name(&record.mode))
    .bind(columns::json(&record.context))
    .bind(columns::json(&record.measurement_ages_ms))
    .bind(columns::json(&record.safe_arms))
    .bind(&record.chosen_arm)
    .bind(opt_json(&record.propensities))
    .bind(opt_json(&record.requested_action))
    .bind(opt_json(&record.validated_action))
    .bind(opt_json(&record.executed_action))
    .bind(record.command_id.map(|id| id.to_string()))
    .bind(opt_json(&record.reward_components))
    .bind(columns::opt_time(record.attribution_window_start.as_ref()))
    .bind(columns::opt_time(record.attribution_window_end.as_ref()))
    .bind(record.interrupted)
    .bind(record.override_applied)
    .bind(columns::json(&record.parent_frame_ids))
    .bind(columns::time(&record.created_at))
    .execute(executor)
    .await?;
    Ok(())
}

fn from_row(row: &sqlx::sqlite::SqliteRow) -> Option<PolicyDecisionRecord> {
    let text = |name: &str| -> Option<String> { row.get(name) };
    let json = |name: &str| columns::parse_json::<serde_json::Value>(text(name).as_deref());
    let decision_id: String = row.get("decision_id");
    let created_at: String = row.get("created_at");
    Some(PolicyDecisionRecord {
        decision_id: Uuid::parse_str(&decision_id).ok()?,
        policy: row.get("policy"),
        policy_version: row.get("policy_version"),
        mode: columns::from_name(text("mode").as_deref())?,
        context: json("context_json").unwrap_or(serde_json::Value::Null),
        measurement_ages_ms: columns::parse_json(text("measurement_ages_json").as_deref())
            .unwrap_or_default(),
        safe_arms: columns::parse_json(text("safe_arms_json").as_deref()).unwrap_or_default(),
        chosen_arm: row.get("chosen_arm"),
        propensities: json("propensities_json"),
        requested_action: json("requested_action_json"),
        validated_action: json("validated_action_json"),
        executed_action: json("executed_action_json"),
        command_id: text("command_id")
            .and_then(|id| Uuid::parse_str(&id).ok())
            .map(CommandId),
        reward_components: json("reward_components_json"),
        attribution_window_start: columns::parse_opt_time(text("attribution_window_start")),
        attribution_window_end: columns::parse_opt_time(text("attribution_window_end")),
        interrupted: row.get("interrupted"),
        override_applied: row.get("override"),
        parent_frame_ids: columns::parse_json(text("parent_frame_ids_json").as_deref())
            .unwrap_or_default(),
        created_at: columns::parse_time(&created_at)?,
    })
}

pub async fn load<'c, E>(
    executor: E,
    decision_id: Uuid,
) -> Result<Option<PolicyDecisionRecord>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let row = sqlx::query("SELECT * FROM policy_decisions WHERE decision_id = ?")
        .bind(decision_id.to_string())
        .fetch_optional(executor)
        .await?;
    Ok(row.as_ref().and_then(from_row))
}

/// Decisions of one policy, newest first.
pub async fn for_policy<'c, E>(
    executor: E,
    policy: &str,
    limit: u32,
) -> Result<Vec<PolicyDecisionRecord>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let rows = sqlx::query(
        "SELECT * FROM policy_decisions WHERE policy = ? ORDER BY created_at DESC, id DESC \
         LIMIT ?",
    )
    .bind(policy)
    .bind(i64::from(limit.max(1)))
    .fetch_all(executor)
    .await?;
    Ok(rows.iter().filter_map(from_row).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::types::AutonomyMode;
    use chrono::{TimeZone, Utc};
    use serde_json::json;

    #[tokio::test]
    async fn a_decision_is_proposed_then_completed_in_place() {
        let pool = crate::storage::test_pool().await;
        let t0 = Utc.with_ymd_and_hms(2026, 10, 26, 5, 0, 0).unwrap();
        let mut rec = PolicyDecisionRecord {
            decision_id: Uuid::new_v4(),
            policy: "fw2".into(),
            policy_version: "0.1.0".into(),
            mode: AutonomyMode::Proposal,
            context: json!({"filter": "L", "background_e": 140.0}),
            measurement_ages_ms: [("background".to_string(), 1200)].into_iter().collect(),
            safe_arms: vec!["continue".into(), "Ha".into()],
            chosen_arm: Some("continue".into()),
            // Deterministic choice: no propensities.
            propensities: None,
            requested_action: Some(json!({"arm": "continue"})),
            validated_action: None,
            executed_action: None,
            command_id: None,
            reward_components: None,
            attribution_window_start: None,
            attribution_window_end: None,
            interrupted: false,
            override_applied: false,
            parent_frame_ids: vec!["frame-1".into()],
            created_at: t0,
        };
        upsert(&pool, &rec).await.unwrap();
        assert_eq!(
            load(&pool, rec.decision_id).await.unwrap().as_ref(),
            Some(&rec)
        );
        let raw: (Option<String>,) =
            sqlx::query_as("SELECT propensities_json FROM policy_decisions")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(raw.0, None, "deterministic decisions store no propensities");

        rec.validated_action = Some(json!({"arm": "continue"}));
        rec.executed_action = Some(json!({"arm": "continue"}));
        rec.command_id = Some(CommandId::new());
        rec.reward_components = Some(json!({"snr_gain": 0.12}));
        rec.attribution_window_start = Some(t0);
        rec.attribution_window_end = Some(t0 + chrono::Duration::minutes(10));
        let mut later = rec.clone();
        later.created_at = t0 + chrono::Duration::minutes(10);
        upsert(&pool, &later).await.unwrap();
        // created_at stays the proposal time.
        assert_eq!(load(&pool, rec.decision_id).await.unwrap(), Some(rec));
        assert_eq!(for_policy(&pool, "fw2", 10).await.unwrap().len(), 1);
    }
}
