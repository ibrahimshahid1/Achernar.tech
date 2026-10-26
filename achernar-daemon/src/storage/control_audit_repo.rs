//! `control_audit_logs`: one row per gate stage of every gated command
//! (equipment spec §2.9 step 4, H21; v6 §7.2; migrations 001 and 009).
//! Stages run proposed, then validated or rejected, then executed, then
//! outcome. Columns: `actor` is the source label, `action` the operation's
//! serde name, `target` the claimed domains, `payload_json` the whole
//! [`AuditRecord`], plus `command_id` and `stage` so the stages of one
//! command join.

// Consumed by the recorder and, once the gate is wired (WP1A-2, WP1B-1), by
// decision-record exports.
#![allow(dead_code)]

use serde::Serialize;
use sqlx::{Executor, Row, Sqlite};

use crate::equipment::commands::Domain;
use crate::equipment::events::AuditRecord;
use crate::equipment::types::kind_name;
use crate::storage::columns;

/// Append one stage row and return its id.
pub async fn insert<'c, E>(executor: E, record: &AuditRecord) -> Result<i64, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let result = sqlx::query(
        r#"
        INSERT INTO control_audit_logs (
            actor, action, target, payload_json, accepted, reason, created_at, command_id, stage
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(record.source.label())
    .bind(columns::name(&record.op))
    .bind(target(&record.domains))
    .bind(columns::json(record))
    .bind(record.accepted)
    .bind(&record.reason)
    .bind(columns::time(&record.at))
    .bind(record.command_id.to_string())
    .bind(columns::name(&record.stage))
    .execute(executor)
    .await?;
    Ok(result.last_insert_rowid())
}

/// `mount,guider,camera_boundary`: the domains a command claimed, readable
/// without parsing JSON. NULL when it claimed none (abort, e-stop).
fn target(domains: &[Domain]) -> Option<String> {
    let names: Vec<&str> = domains
        .iter()
        .map(|domain| match domain {
            Domain::Device(kind) => kind_name(*kind),
            Domain::CameraBoundary => "camera_boundary",
        })
        .collect();
    (!names.is_empty()).then(|| names.join(","))
}

/// One stored stage row.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StoredAudit {
    pub id: i64,
    pub actor: String,
    pub action: String,
    pub target: Option<String>,
    pub accepted: bool,
    pub reason: Option<String>,
    pub created_at: String,
    pub command_id: Option<String>,
    pub stage: Option<String>,
    /// The full record, when the row was written by the gate.
    pub record: Option<AuditRecord>,
}

/// Every stage of one command, in write order.
pub async fn for_command<'c, E>(
    executor: E,
    command_id: &str,
) -> Result<Vec<StoredAudit>, sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    let rows = sqlx::query("SELECT * FROM control_audit_logs WHERE command_id = ? ORDER BY id")
        .bind(command_id)
        .fetch_all(executor)
        .await?;
    Ok(rows
        .iter()
        .map(|row| {
            let payload: String = row.get("payload_json");
            StoredAudit {
                id: row.get("id"),
                actor: row.get("actor"),
                action: row.get("action"),
                target: row.get("target"),
                accepted: row.get("accepted"),
                reason: row.get("reason"),
                created_at: row.get("created_at"),
                command_id: row.get("command_id"),
                stage: row.get("stage"),
                record: serde_json::from_str(&payload).ok(),
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::{GatedCommand, GatedOp, MountCmd};
    use crate::equipment::events::AuditStage;
    use crate::equipment::types::{AutonomyMode, CommandSource, DeviceKind, OpOutcome, Priority};
    use chrono::{TimeZone, Utc};

    #[tokio::test]
    async fn every_gate_stage_is_one_joinable_row() {
        let pool = crate::storage::test_pool().await;
        let at = Utc.with_ymd_and_hms(2026, 10, 26, 6, 0, 0).unwrap();
        let cmd = GatedCommand::new(
            GatedOp::Mount(MountCmd::Park),
            CommandSource::Policy {
                model: "m3".into(),
                version: "0.2.0".into(),
            },
            AutonomyMode::Proposal,
            Priority::MountRecovery,
        );
        let mut stages = vec![
            AuditRecord::for_command(&cmd, AuditStage::Proposed, true, None, at),
            AuditRecord::for_command(&cmd, AuditStage::Validated, true, None, at),
            AuditRecord::for_command(&cmd, AuditStage::Executed, true, None, at),
        ];
        let mut outcome = AuditRecord::for_command(&cmd, AuditStage::Outcome, true, None, at);
        outcome.outcome = Some(OpOutcome::Succeeded);
        outcome.annotations.push("limits_unchecked".into());
        stages.push(outcome);
        for stage in &stages {
            insert(&pool, stage).await.unwrap();
        }
        let rows = for_command(&pool, &cmd.command_id.to_string())
            .await
            .unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(
            rows.iter()
                .map(|r| r.stage.as_deref().unwrap())
                .collect::<Vec<_>>(),
            vec!["proposed", "validated", "executed", "outcome"]
        );
        assert_eq!(rows[0].actor, "policy:m3@0.2.0");
        assert_eq!(rows[0].action, "park");
        assert_eq!(rows[0].created_at, "2026-10-26T06:00:00.000Z");
        assert_eq!(rows[3].record.as_ref(), Some(&stages[3]));
        let expected_target = target(&cmd.domains);
        assert_eq!(rows[0].target, expected_target);
        assert_eq!(
            target(&[Domain::Device(DeviceKind::Mount), Domain::CameraBoundary]).as_deref(),
            Some("mount,camera_boundary")
        );
        assert_eq!(target(&[]), None);
    }

    #[tokio::test]
    async fn a_rejection_is_recorded_with_its_reason() {
        let pool = crate::storage::test_pool().await;
        let cmd = GatedCommand::new(
            GatedOp::Mount(MountCmd::Unpark),
            CommandSource::Operator { overriding: false },
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        let rec = AuditRecord::for_command(
            &cmd,
            AuditStage::Rejected,
            false,
            Some("ownership_conflict".into()),
            Utc::now(),
        );
        insert(&pool, &rec).await.unwrap();
        let rows = for_command(&pool, &cmd.command_id.to_string())
            .await
            .unwrap();
        assert!(!rows[0].accepted);
        assert_eq!(rows[0].reason.as_deref(), Some("ownership_conflict"));
        assert_eq!(rows[0].actor, "operator");
    }
}
