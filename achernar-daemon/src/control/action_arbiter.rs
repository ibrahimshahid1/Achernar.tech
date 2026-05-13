//! Action arbiter — Section 7.3.2.
//! ActionRequest, ControlDomain, priority queue for arbitrating hardware commands.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Domain of control that an action targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ControlDomain {
    Focuser,
    Mount,
    Camera,
    FilterWheel,
    Guider,
}

/// Priority level for action requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionPriority {
    /// Safety-critical actions (e.g., emergency park)
    Critical = 0,
    /// Time-sensitive actions (e.g., abort current exposure)
    High = 1,
    /// Normal autonomous actions (e.g., autofocus move)
    Normal = 2,
    /// Background/deferred actions (e.g., drift correction)
    Low = 3,
}

impl Ord for ActionPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        (*self as u8).cmp(&(*other as u8)).reverse()
    }
}

impl PartialOrd for ActionPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A request to perform a hardware action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRequest {
    pub id: uuid::Uuid,
    pub domain: ControlDomain,
    pub priority: ActionPriority,
    pub source: ActionSource,
    pub description: String,
    pub payload: serde_json::Value,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Who originated the action request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionSource {
    User,
    FocusAgent,
    MountController,
    FilterController,
    VisionLoop,
    SafetyGuard,
}

impl Eq for ActionRequest {}
impl PartialEq for ActionRequest {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Ord for ActionRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl PartialOrd for ActionRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The action arbiter manages a priority queue of pending actions
/// and ensures only one action per ControlDomain executes at a time.
pub struct ActionArbiter {
    queue: BinaryHeap<ActionRequest>,
    active: std::collections::HashMap<ControlDomain, uuid::Uuid>,
}

impl ActionArbiter {
    pub fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
            active: std::collections::HashMap::new(),
        }
    }

    /// Submit an action request to the arbiter.
    pub fn submit(&mut self, request: ActionRequest) {
        tracing::debug!(
            "Action submitted: {} [{:?}]",
            request.description,
            request.priority
        );
        self.queue.push(request);
    }

    /// Get the next action to execute, if the domain is free.
    pub fn next_action(&mut self) -> Option<ActionRequest> {
        while let Some(action) = self.queue.pop() {
            if !self.active.contains_key(&action.domain) {
                self.active.insert(action.domain, action.id);
                return Some(action);
            }
            // Re-queue if domain is busy
            self.queue.push(action);
            return None;
        }
        None
    }

    /// Mark an action as completed, freeing the domain.
    pub fn complete(&mut self, domain: ControlDomain) {
        self.active.remove(&domain);
    }

    /// Cancel all pending actions for a domain.
    pub fn cancel_domain(&mut self, domain: ControlDomain) {
        let remaining: Vec<_> = self.queue.drain().filter(|a| a.domain != domain).collect();
        for item in remaining {
            self.queue.push(item);
        }
        self.active.remove(&domain);
    }
}
