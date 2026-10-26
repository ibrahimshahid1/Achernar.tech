//! Action arbiter — Section 7.3.2.
//! ActionRequest, ControlDomain, priority queue for arbitrating hardware commands.

use crate::control::ownership::DomainClaim;
use crate::equipment::types::{CommandId, CommandSource};
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
        let action = self.queue.pop()?;
        if let std::collections::hash_map::Entry::Vacant(slot) = self.active.entry(action.domain) {
            slot.insert(action.id);
            return Some(action);
        }
        // Re-queue if domain is busy
        self.queue.push(action);
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

// ---------------------------------------------------------------------------
// Equipment-layer arbitration (equipment spec §2.9 step 2, D9). Used by
// `equipment::gate::CommandGate` and the device worker; the queue above is
// unchanged.
// ---------------------------------------------------------------------------

/// Preemption rank of the safety tier. An operator override cancels any
/// holder strictly below it.
pub const SAFETY_RANK: u8 = 6;

/// True when a request may take a domain (or a running operation) from a
/// holder of `holder_rank`.
///
/// - An operator (direct or relay) never preempts silently: only with
///   `overriding = true`, and only a holder below safety.
/// - Any other source preempts only with a strictly higher rank, so an
///   automatic command (rank 4 or less) never preempts an operator-held
///   domain (rank 5).
pub fn outranks(request_rank: u8, request_source: &CommandSource, holder_rank: u8) -> bool {
    if request_source.is_operator() {
        request_source.overriding() && holder_rank < SAFETY_RANK
    } else {
        request_rank > holder_rank
    }
}

/// Outcome of arbitrating one domain.
#[derive(Debug, Clone, PartialEq)]
pub enum Arbitration {
    /// The domain is free, or already held by this command.
    Grant,
    /// The holder is cancelled; its outcome is `Aborted { by }`.
    Preempt {
        holder: CommandId,
        override_applied: bool,
    },
    /// HTTP 409 `ownership_conflict`.
    Conflict { holder: CommandId },
}

/// Arbitrate one domain between its holder and a request.
pub fn arbitrate(
    holder: Option<&DomainClaim>,
    request_id: CommandId,
    request_rank: u8,
    request_source: &CommandSource,
) -> Arbitration {
    let Some(holder) = holder else {
        return Arbitration::Grant;
    };
    if holder.command_id == request_id {
        return Arbitration::Grant;
    }
    if outranks(request_rank, request_source, holder.rank) {
        Arbitration::Preempt {
            holder: holder.command_id,
            override_applied: request_source.is_operator(),
        }
    } else {
        Arbitration::Conflict {
            holder: holder.command_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::types::Priority;

    fn holder(rank: u8) -> DomainClaim {
        DomainClaim {
            command_id: CommandId::new(),
            rank,
            priority: Priority::Exposure,
            source: CommandSource::System,
            claimed_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn operators_need_an_override_and_never_beat_safety() {
        let plain = CommandSource::Operator { overriding: false };
        let over = CommandSource::Relay { overriding: true };
        assert!(!outranks(5, &plain, 1), "no silent operator preemption");
        assert!(
            outranks(5, &over, 5),
            "override beats another operator hold"
        );
        assert!(outranks(5, &over, 4));
        assert!(!outranks(5, &over, SAFETY_RANK));
        let policy = CommandSource::Policy {
            model: "fw2".into(),
            version: "1".into(),
        };
        assert!(
            !outranks(4, &policy, 5),
            "automatic never beats an operator"
        );
        assert!(outranks(3, &policy, 2));
        assert!(!outranks(2, &policy, 2), "equal rank does not preempt");
        assert!(outranks(6, &CommandSource::Safety, 5));
        let held = holder(3);
        let id = CommandId::new();
        assert_eq!(arbitrate(None, id, 1, &policy), Arbitration::Grant);
        assert_eq!(
            arbitrate(Some(&held), held.command_id, 1, &policy),
            Arbitration::Grant
        );
        assert_eq!(
            arbitrate(Some(&held), id, 5, &over),
            Arbitration::Preempt {
                holder: held.command_id,
                override_applied: true
            }
        );
        assert_eq!(
            arbitrate(Some(&held), id, 6, &CommandSource::Safety),
            Arbitration::Preempt {
                holder: held.command_id,
                override_applied: false
            }
        );
        assert_eq!(
            arbitrate(Some(&held), id, 5, &plain),
            Arbitration::Conflict {
                holder: held.command_id
            }
        );
    }
}
