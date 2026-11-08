//! `GET safety` (spec §2.12, §3.10). Owner: WP1B-1.
//!
//! The combined safety state of every connected monitor (`none` without
//! one), each monitor's reading with its age, and the active safety resume
//! blocks. A monitor that is not Ready counts as not safe for new automatic
//! actions (§2.8); the on-unsafe policy runs only on a confirmed Unsafe
//! reading or on a lost monitor.

use axum::{extract::State, Json};
use serde_json::{json, Value};

use crate::equipment::handle::SafetyMonitorHandle;
use crate::equipment::snapshot::{ResumeBlockKind, SafetySummary};
use crate::equipment::traits::EquipmentReader;
use crate::state::AppState;

/// The combined safety state: the safety supervisor's (WP1A-8) when one
/// runs, else the hub's derived one (`none` without a monitor). Watch reads
/// only; no driver is touched.
pub fn summary(state: &AppState) -> SafetySummary {
    state
        .equipment_ctx
        .safety_supervisor()
        .map(|supervisor| supervisor.summary())
        .unwrap_or_else(|| state.equipment.snapshot().safety.clone())
}

/// The `safety {state, changed_at}` block of `/status` and
/// `TELEMETRY_UPDATE` (additive, §2.13), so a client can show the combined
/// safety state without polling `GET safety`.
pub fn status_block(state: &AppState) -> Value {
    let summary = summary(state);
    json!({
        "state": summary.state,
        "changed_at": summary.changed_at,
    })
}

pub fn status_value(state: &AppState) -> Value {
    let summary = summary(state);
    let now = state.equipment.clock().now_mono_ns();
    let monitor = state
        .equipment
        .handle::<SafetyMonitorHandle>()
        .filter(|h| !h.is_retired())
        .map(|handle| {
            crate::equipment::status::safety_monitor_block(
                &handle.snapshot(),
                &handle.identity(),
                &handle.health(),
                now,
            )
        });
    let blocks: Vec<_> = state
        .equipment
        .resume_blocks()
        .borrow()
        .iter()
        .filter(|b| {
            matches!(
                b.kind,
                ResumeBlockKind::SafetyUnsafe | ResumeBlockKind::SafetyUnknown
            )
        })
        .cloned()
        .collect();
    let policy_running = state
        .equipment_ctx
        .safety_supervisor()
        .is_some_and(|s| s.policy_running());
    json!({
        "state": summary.state,
        "sources": summary.sources,
        "changed_at": summary.changed_at,
        "monitor": monitor,
        "resume_blocks": blocks,
        "policy_running": policy_running,
        "on_unsafe": format!(
            "{:?}",
            state
                .equipment_ctx
                .configs
                .safety
                .effective_on_unsafe(&state.equipment_ctx.configs.mount.safety)
        )
        .to_ascii_lowercase(),
    })
}

pub async fn status(State(state): State<AppState>) -> Json<Value> {
    Json(status_value(&state))
}
