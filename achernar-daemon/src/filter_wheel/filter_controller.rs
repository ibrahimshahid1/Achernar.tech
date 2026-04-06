//! Filter controller — Section 8.2.
//! Executes filter switch handshake with focus offset compensation.

use serde::{Deserialize, Serialize};

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
        _target_filter: &str,
    ) -> crate::error::Result<FilterSwitchResult> {
        todo!("Implement filter switch handshake — Section 8.2")
    }

    /// Get the focus offset for a given filter.
    pub fn focus_offset_for(&self, filter_name: &str) -> Option<i32> {
        self.filter_names
            .iter()
            .position(|f| f == filter_name)
            .and_then(|idx| self.focus_offsets.get(idx).copied())
    }
}
