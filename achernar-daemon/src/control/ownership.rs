//! Device ownership tracking — Section 7.3.
//! Tracks which subsystem currently holds a device lock.

use crate::control::action_arbiter::ControlDomain;
use std::collections::HashMap;

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
