//! Filter controller — Section 8.2.1.
//!
//! A thin front over the equipment-layer handshake (`equipment::filter::change`,
//! spec §3.4 W-c). The old controller recorded a switch without moving any
//! hardware and computed offsets from the `[filter_wheel]` TOML arrays. Now:
//!
//! - a switch runs the full handshake behind the gate (the caller passes the
//!   gate's permit);
//! - the current filter is the wheel's verified slot, never a remembered
//!   name;
//! - offsets come from the reconciled identity profile (or the device),
//!   never the TOML seeds (D17).

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::equipment::commands::FilterTarget;
use crate::equipment::filter::change::{FilterChanger, HandshakeReport, HandshakeRequest, Peers};
use crate::equipment::filter::read_wheel;
use crate::equipment::handle::Permit;
use crate::equipment::types::OpOutcome;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterSwitchResult {
    pub from_filter: Option<String>,
    pub to_filter: Option<String>,
    /// The relative focuser move applied; `None` when none could be.
    pub focus_offset_applied: Option<i32>,
    pub duration_ms: u64,
    /// The handshake completed and raised no resume block.
    pub success: bool,
    pub outcome: OpOutcome,
    pub resume_blocked: bool,
    pub focus_verification_required: bool,
}

impl FilterSwitchResult {
    pub fn from_report(report: &HandshakeReport) -> Self {
        let record = report.record.as_ref();
        Self {
            from_filter: record.and_then(|r| r.from_filter.clone()),
            to_filter: record.map(|r| r.to_filter.clone()),
            focus_offset_applied: report.offset_applied_steps,
            duration_ms: report.duration_ms,
            success: report.handshake_ok,
            outcome: report.outcome.clone(),
            resume_blocked: report.resume_block.is_some(),
            focus_verification_required: !report.focus_verification.is_empty(),
        }
    }
}

pub struct FilterController {
    changer: Arc<FilterChanger>,
}

impl FilterController {
    pub fn new(changer: Arc<FilterChanger>) -> Self {
        Self { changer }
    }

    pub fn changer(&self) -> &Arc<FilterChanger> {
        &self.changer
    }

    /// The name of the wheel's verified current slot, if it has one.
    pub fn current_filter(&self) -> Option<String> {
        let snap = self.changer.wheel().snapshot();
        if !snap.verified {
            return None;
        }
        let slot = snap.verified_slot?;
        read_wheel(&self.changer.model())
            .name_of(slot)
            .map(str::to_string)
    }

    /// Run the handshake to `target_filter`. `permit` is the gate's permit
    /// for `GatedOp::FilterHandshake`.
    pub async fn execute_filter_switch(
        &self,
        target_filter: &str,
        peers: Peers,
        permit: Permit,
    ) -> FilterSwitchResult {
        let request = HandshakeRequest::new(FilterTarget::Name(target_filter.to_string()));
        let report = self.changer.change(peers, request, permit).await;
        FilterSwitchResult::from_report(&report)
    }

    /// The reconciled focus offset of a filter (profile, else device);
    /// never a TOML seed.
    pub fn focus_offset_for(&self, filter_name: &str) -> Option<i32> {
        let wheel = read_wheel(&self.changer.model());
        let slot = wheel
            .resolve(&FilterTarget::Name(filter_name.to_string()))
            .ok()?;
        wheel.slot(slot).and_then(|s| s.slot.offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::clock::SystemClock;
    use crate::equipment::commands::{FilterWheelCmd, GatedCommand, GatedOp};
    use crate::equipment::filter::change::{HandshakeServices, NoChildChecks, TokioTimer};
    use crate::equipment::filter::reconcile::reconcile;
    use crate::equipment::filter::test_support::CollectingSink;
    use crate::equipment::filter::{shared_wheel, FilterEventSink, FilterSettings};
    use crate::equipment::snapshot::{FilterWheelSnapshot, SlotInfo};
    use crate::equipment::testing::fake_handles::{
        fake_filter_wheel, FakeCameraActivity, FakeResponse, RecordingSink,
    };
    use crate::equipment::traits::{CameraActivity, Clock, RecorderSink};
    use crate::equipment::types::{AutonomyMode, CommandSource, Observed, Priority, Source};

    fn at(slot: u32) -> FilterWheelSnapshot {
        FilterWheelSnapshot {
            position: Observed::fresh(slot, Source::Property, chrono::Utc::now(), 0),
            verified_slot: Some(slot),
            verified: true,
            ..FilterWheelSnapshot::default()
        }
    }

    #[tokio::test]
    async fn the_controller_runs_the_handshake_and_reads_reconciled_offsets() {
        let (handle, fake) = fake_filter_wheel(at(0));
        fake.set_responder(|cmd, snap| {
            let FilterWheelCmd::SetPosition(slot) = cmd;
            *snap = FilterWheelSnapshot {
                connection_id: snap.connection_id,
                ..at(*slot)
            };
            FakeResponse::succeed()
        });
        let slots: Vec<SlotInfo> = ["L", "Ha"]
            .iter()
            .enumerate()
            .map(|(i, name)| SlotInfo {
                index0: i as u32,
                device_name: Some((*name).into()),
                device_focus_offset: Some(10 * i as i32),
            })
            .collect();
        let services = HandshakeServices {
            clock: Arc::new(SystemClock) as Arc<dyn Clock>,
            timer: Arc::new(TokioTimer),
            events: CollectingSink::new() as Arc<dyn FilterEventSink>,
            recorder: RecordingSink::new() as Arc<dyn RecorderSink>,
            child_checks: Arc::new(NoChildChecks),
        };
        let settings = FilterSettings {
            settle_ms: 0,
            ..FilterSettings::default()
        };
        let changer = Arc::new(FilterChanger::new(
            handle,
            shared_wheel(reconcile(&slots, None)),
            settings,
            services,
        ));
        let controller = FilterController::new(changer);
        assert_eq!(controller.current_filter().as_deref(), Some("L"));
        assert_eq!(controller.focus_offset_for("ha"), Some(10));
        assert_eq!(controller.focus_offset_for("SII"), None);

        let cmd = GatedCommand::new(
            GatedOp::FilterHandshake {
                target: FilterTarget::Name("Ha".into()),
            },
            CommandSource::Operator { overriding: false },
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        let camera = FakeCameraActivity::new();
        let peers = Peers {
            camera: Some(camera as Arc<dyn CameraActivity>),
            ..Peers::default()
        };
        let result = controller
            .execute_filter_switch("Ha", peers, Permit::new(&cmd, None))
            .await;
        assert!(result.success, "{result:?}");
        assert_eq!(result.from_filter.as_deref(), Some("L"));
        assert_eq!(result.to_filter.as_deref(), Some("Ha"));
        assert_eq!(fake.commands(), vec![FilterWheelCmd::SetPosition(1)]);
        // No focuser: the 10-step offset is reported, not applied.
        assert_eq!(result.focus_offset_applied, None);
        assert!(result.focus_verification_required);
        assert_eq!(controller.current_filter().as_deref(), Some("Ha"));
    }
}
