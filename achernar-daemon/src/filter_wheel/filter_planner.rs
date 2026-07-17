//! Filter planner — Section 8.2.
//! Stub for ILP (Integer Linear Programming) plan consumption.

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
}
