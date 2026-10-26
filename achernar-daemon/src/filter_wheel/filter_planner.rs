//! Filter planner — Section 8.2.4 (FW-3).
//! Stub for ILP (Integer Linear Programming) plan consumption. A validated
//! plan can be locked, so that no recommendation source overrides it (v6
//! §8.2.5); the filter safety check refuses such overrides while the lock
//! holds. Plan steps still run through the gate as filter handshakes.

use chrono::{DateTime, Utc};

use crate::equipment::filter::check::{LockedPlan, PlanLock};

pub struct FilterPlanner;

impl FilterPlanner {
    /// Consume an ILP-generated filter plan and scheduleexecutions.
    pub fn execute_plan(&self, plan: &serde_json::Value) -> crate::error::Result<()> {
        let slots = plan
            .get("slots")
            .or_else(|| plan.get("actions"))
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                crate::error::ConfigError::ParseError(
                    "Filter plan must contain a slots or actions array".to_string(),
                )
            })?;

        for (idx, slot) in slots.iter().enumerate() {
            let filter = slot
                .get("filter")
                .or_else(|| slot.get("filter_name"))
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    crate::error::ConfigError::ParseError(format!(
                        "Filter plan slot {idx} is missing filter"
                    ))
                })?;
            let duration = slot
                .get("duration_s")
                .or_else(|| slot.get("integration_seconds"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            if filter.trim().is_empty() || duration < 0.0 {
                return Err(crate::error::ConfigError::ParseError(format!(
                    "Filter plan slot {idx} has invalid filter or duration"
                ))
                .into());
            }
        }

        Ok(())
    }

    /// Validate a plan and lock it. The plan id is `plan_id` or `id`.
    pub fn activate(
        &self,
        plan: &serde_json::Value,
        lock: &PlanLock,
        now: DateTime<Utc>,
    ) -> crate::error::Result<LockedPlan> {
        self.execute_plan(plan)?;
        let plan_id = plan
            .get("plan_id")
            .or_else(|| plan.get("id"))
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                crate::error::ConfigError::ParseError(
                    "Filter plan must carry a plan_id to be locked".to_string(),
                )
            })?;
        let locked = LockedPlan {
            plan_id: plan_id.to_string(),
            locked_at: now,
            reason: Some("filter plan active".to_string()),
        };
        lock.lock(locked.clone());
        Ok(locked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn only_a_valid_plan_with_an_id_is_locked() {
        let lock = PlanLock::new();
        let planner = FilterPlanner;
        let bad = json!({"plan_id": "p1", "slots": [{"filter": "", "duration_s": 60}]});
        assert!(planner.activate(&bad, &lock, Utc::now()).is_err());
        assert!(!lock.is_locked());
        let anonymous = json!({"slots": [{"filter": "Ha", "duration_s": 600}]});
        assert!(planner.activate(&anonymous, &lock, Utc::now()).is_err());
        let plan = json!({"plan_id": "night-1", "slots": [{"filter": "Ha", "duration_s": 600}]});
        let locked = planner.activate(&plan, &lock, Utc::now()).unwrap();
        assert_eq!(locked.plan_id, "night-1");
        assert_eq!(lock.current().unwrap().plan_id, "night-1");
    }
}
