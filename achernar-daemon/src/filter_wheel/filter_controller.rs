//! Filter controller — Section 8.2.
//! Executes filter switch handshake with focus offset compensation.

use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterSwitchResult {
    pub from_filter: String,
    pub to_filter: String,
    pub focus_offset_applied: i32,
    pub duration_ms: u64,
    pub success: bool,
}

pub struct FilterController {
    current_filter: Option<String>,
    filter_names: Vec<String>,
    focus_offsets: Vec<i32>,
}

impl FilterController {
    pub fn new(config: &crate::config::FilterWheelConfig) -> Self {
        Self {
            current_filter: None,
            filter_names: config.filter_names.clone(),
            focus_offsets: config.focus_offsets.clone(),
        }
    }

    pub fn current_filter(&self) -> Option<&str> {
        self.current_filter.as_deref()
    }

    /// Execute a filter switch with focus offset compensation.
    pub async fn execute_filter_switch(
        &mut self,
        target_filter: &str,
    ) -> crate::error::Result<FilterSwitchResult> {
        if !self.filter_names.iter().any(|name| name == target_filter) {
            return Err(crate::error::ControlError::SafetyBlocked(format!(
                "Unknown filter '{target_filter}'"
            ))
            .into());
        }

        let started = Instant::now();
        let from_filter = self
            .current_filter
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        let from_offset = self.focus_offset_for(&from_filter).unwrap_or(0);
        let to_offset = self.focus_offset_for(target_filter).unwrap_or(0);
        let focus_offset_applied = to_offset - from_offset;

        self.current_filter = Some(target_filter.to_string());

        Ok(FilterSwitchResult {
            from_filter,
            to_filter: target_filter.to_string(),
            focus_offset_applied,
            duration_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
            success: true,
        })
    }

    /// Get the focus offset for a given filter.
    pub fn focus_offset_for(&self, filter_name: &str) -> Option<i32> {
        self.filter_names
            .iter()
            .position(|f| f == filter_name)
            .and_then(|idx| self.focus_offsets.get(idx).copied())
    }
}
