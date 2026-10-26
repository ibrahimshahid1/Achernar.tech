//! Device ownership tracking — Section 7.3.
//! Tracks which subsystem currently holds a device lock.

use crate::control::action_arbiter::ControlDomain;
use crate::equipment::commands::Domain;
use crate::equipment::types::{CommandId, CommandSource, Priority};
use std::collections::{BTreeMap, HashMap, VecDeque};

/// Who currently owns a device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceOwner {
    User,
    FocusAgent,
    MountController,
    FilterController,
    VisionLoop,
    Idle,
}

/// Tracks device ownership for safe concurrent access.
pub struct OwnershipTracker {
    owners: HashMap<ControlDomain, DeviceOwner>,
}

impl OwnershipTracker {
    pub fn new() -> Self {
        Self {
            owners: HashMap::new(),
        }
    }

    /// Attempt to acquire ownership of a device.
    pub fn acquire(&mut self, domain: ControlDomain, owner: DeviceOwner) -> Result<(), String> {
        let current = self.owners.get(&domain).unwrap_or(&DeviceOwner::Idle);
        if *current != DeviceOwner::Idle && *current != owner {
            return Err(format!("Device {:?} locked by {:?}", domain, current));
        }
        self.owners.insert(domain, owner);
        Ok(())
    }

    /// Release ownership of a device.
    pub fn release(&mut self, domain: ControlDomain) {
        self.owners.insert(domain, DeviceOwner::Idle);
    }

    /// Check who owns a device.
    pub fn owner_of(&self, domain: ControlDomain) -> &DeviceOwner {
        self.owners.get(&domain).unwrap_or(&DeviceOwner::Idle)
    }
}

// ---------------------------------------------------------------------------
// Equipment-layer ownership (equipment spec §2.9 step 1). Used by
// `equipment::gate::CommandGate`; the tracker above is unchanged.
// ---------------------------------------------------------------------------

/// Preemption records kept for `preempted_by` lookups.
const PREEMPTION_HISTORY: usize = 256;

/// One ownership claim held by an authorised command.
#[derive(Debug, Clone, PartialEq)]
pub struct DomainClaim {
    pub command_id: CommandId,
    /// Preemption rank of the holder (§2.9 arbiter table).
    pub rank: u8,
    pub priority: Priority,
    pub source: CommandSource,
    pub claimed_at: chrono::DateTime<chrono::Utc>,
}

/// Who holds each ownership domain.
///
/// Claims are released by command id, so a late release from a preempted
/// command never frees a domain that its preemptor now holds.
#[derive(Debug, Default)]
pub struct DomainLedger {
    claims: BTreeMap<Domain, DomainClaim>,
    preempted: VecDeque<(CommandId, CommandId)>,
}

impl DomainLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn holder(&self, domain: Domain) -> Option<&DomainClaim> {
        self.claims.get(&domain)
    }

    /// Take `domain` for `claim`, returning the claim it replaced.
    pub fn claim(&mut self, domain: Domain, claim: DomainClaim) -> Option<DomainClaim> {
        self.claims.insert(domain, claim)
    }

    /// Release every domain still held by `command_id`.
    pub fn release(&mut self, command_id: CommandId) -> Vec<Domain> {
        let released: Vec<Domain> = self
            .claims
            .iter()
            .filter(|(_, claim)| claim.command_id == command_id)
            .map(|(domain, _)| *domain)
            .collect();
        for domain in &released {
            self.claims.remove(domain);
        }
        released
    }

    /// True when `command_id` holds every one of `domains`.
    pub fn holds_all(&self, command_id: CommandId, domains: &[Domain]) -> bool {
        domains.iter().all(|domain| {
            self.claims
                .get(domain)
                .is_some_and(|claim| claim.command_id == command_id)
        })
    }

    /// Remember that `holder` lost its domains to `by`.
    pub fn note_preempted(&mut self, holder: CommandId, by: CommandId) {
        self.preempted.push_back((holder, by));
        while self.preempted.len() > PREEMPTION_HISTORY {
            self.preempted.pop_front();
        }
    }

    /// The command that preempted `command_id`, if any.
    pub fn preempted_by(&self, command_id: CommandId) -> Option<CommandId> {
        self.preempted
            .iter()
            .rev()
            .find(|(holder, _)| *holder == command_id)
            .map(|(_, by)| *by)
    }

    pub fn claims(&self) -> impl Iterator<Item = (&Domain, &DomainClaim)> {
        self.claims.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::types::DeviceKind;

    fn claim(rank: u8) -> DomainClaim {
        DomainClaim {
            command_id: CommandId::new(),
            rank,
            priority: Priority::Exposure,
            source: CommandSource::System,
            claimed_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn ledger_releases_by_command_id_only() {
        let mount = Domain::Device(DeviceKind::Mount);
        let guider = Domain::Device(DeviceKind::Guider);
        let mut ledger = DomainLedger::new();
        let old = claim(2);
        let new = claim(6);
        ledger.claim(mount, old.clone());
        ledger.claim(guider, old.clone());
        assert!(ledger.holds_all(old.command_id, &[mount, guider]));
        // A preemptor takes the mount; the old holder's late release must not
        // free it.
        assert_eq!(ledger.claim(mount, new.clone()), Some(old.clone()));
        ledger.note_preempted(old.command_id, new.command_id);
        assert_eq!(ledger.release(old.command_id), vec![guider]);
        assert_eq!(ledger.holder(mount), Some(&new));
        assert_eq!(ledger.preempted_by(old.command_id), Some(new.command_id));
        assert_eq!(ledger.preempted_by(new.command_id), None);
        assert_eq!(ledger.release(new.command_id), vec![mount]);
        assert!(ledger.is_empty());
    }
}
