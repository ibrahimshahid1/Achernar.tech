//! Frame-aware sync (§3.2 M-g). Owner: WP1A-4.
//!
//! `sync_mode = "mount"`: requires CanSync and tracking on; the target is
//! converted to the native frame and sent as SyncToCoordinates. Convergence
//! within 5″ within 5 s is verified; otherwise the outcome is `Unknown`.
//! Every sync increments `sync_generation`, which segments the M-4 residual
//! series, and emits `sync_applied{before, after, generation}`.
//!
//! `sync_mode = "offset"` (NINA's NoSync) never syncs the mount. Achernar
//! keeps the solved pointing offset and applies it to later slews in that
//! sky region, so the mount's own pointing model (and the M-4 data) is left
//! untouched. The event is `sync_applied{after: null}` with the generation
//! unchanged.

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};

use crate::equipment::events::EventKind;
use crate::equipment::ops::{OpKind, OpProgress, Operation};
use crate::equipment::traits::Clock;
use crate::equipment::types::{DeviceError, DeviceResult, EqCoord, MountSyncMode, OpOutcome};

use super::{
    duration_ns, fresh_position, is_parked, parked_error, position, CommandMeta, DynMount,
    MountShared, ACTIVE_POLL, SETTER_DEADLINE, VERIFY_TIMEOUT,
};

/// Convergence tolerance (M-g).
pub const SYNC_TOLERANCE_ARCSEC: f64 = 5.0;
/// An offset applies to targets within this radius of where it was solved.
pub const OFFSET_REGION_RADIUS_DEG: f64 = 15.0;

/// One solved pointing offset, native frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OffsetEntry {
    /// Where on the sky it was solved (true position).
    pub at: EqCoord,
    /// Mount reading minus truth, hours (wrapped to ±12).
    pub d_ra_h: f64,
    pub d_dec_deg: f64,
    pub recorded_at: DateTime<Utc>,
}

/// The `sync_mode = "offset"` store, one entry per sky region.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PointingOffsets {
    entries: Vec<OffsetEntry>,
}

fn signed_ra_delta(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(24.0);
    if d > 12.0 {
        d - 24.0
    } else {
        d
    }
}

impl PointingOffsets {
    /// Record that the mount reads `mount_position` where the sky truth is
    /// `true_position`. Replaces older entries in the same region.
    pub fn record(&mut self, mount_position: EqCoord, true_position: EqCoord, at: DateTime<Utc>) {
        self.entries.retain(|entry| {
            entry
                .at
                .separation_arcmin(&true_position)
                .is_none_or(|sep| sep / 60.0 > OFFSET_REGION_RADIUS_DEG)
        });
        self.entries.push(OffsetEntry {
            at: true_position,
            d_ra_h: signed_ra_delta(mount_position.ra_h, true_position.ra_h),
            d_dec_deg: mount_position.dec_deg - true_position.dec_deg,
            recorded_at: at,
        });
    }

    /// The nearest entry within the region radius of `target`.
    pub fn offset_for(&self, target: &EqCoord) -> Option<&OffsetEntry> {
        self.entries
            .iter()
            .filter_map(|entry| {
                entry
                    .at
                    .separation_arcmin(target)
                    .filter(|sep| sep / 60.0 <= OFFSET_REGION_RADIUS_DEG)
                    .map(|sep| (sep, entry))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, entry)| entry)
    }

    /// The mount coordinate to command so the true pointing is `target`.
    pub fn apply(&self, target: EqCoord) -> EqCoord {
        match self.offset_for(&target) {
            Some(entry) => EqCoord {
                ra_h: EqCoord::wrap_ra_h(target.ra_h + entry.d_ra_h),
                dec_deg: (target.dec_deg + entry.d_dec_deg).clamp(-90.0, 90.0),
                frame: target.frame,
            },
            None => target,
        }
    }

    pub fn entries(&self) -> &[OffsetEntry] {
        &self.entries
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// SyncToCoordinates, or an offset record in `offset` mode.
pub struct SyncOp {
    shared: Arc<MountShared>,
    meta: CommandMeta,
    target: EqCoord,
    native: Option<EqCoord>,
    before: Option<EqCoord>,
    sent_mono_ns: u64,
    done: bool,
}

impl SyncOp {
    pub fn new(shared: Arc<MountShared>, meta: CommandMeta, target: EqCoord) -> Self {
        Self {
            shared,
            meta,
            target,
            native: None,
            before: None,
            sent_mono_ns: 0,
            done: false,
        }
    }

    fn applied(&self, after: Option<EqCoord>, generation: u64) {
        self.shared.emit(
            EventKind::SyncApplied {
                before: self.before,
                after,
                generation,
            },
            Some(self.meta.command_id),
        );
    }
}

impl Operation<DynMount> for SyncOp {
    fn kind(&self) -> OpKind {
        OpKind::Sync
    }

    fn deadline(&self) -> Duration {
        SETTER_DEADLINE
    }

    fn start(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> DeviceResult<()> {
        let state = drv.state()?;
        if is_parked(&state) {
            return Err(parked_error("syncing"));
        }
        let native = self.shared.to_native(self.target)?;
        let frame = self.shared.capabilities().native_frame;
        self.before = position(&state, frame);
        self.native = Some(native);
        match self.shared.settings().sync_mode {
            MountSyncMode::Offset => {
                let Some(before) = self.before else {
                    return Err(DeviceError::invalid_operation(
                        "the mount position is unknown, so no pointing offset can be recorded",
                    ));
                };
                let now = clock.now_utc();
                self.shared
                    .with_offsets(|offsets| offsets.record(before, native, now));
                self.applied(None, self.shared.sync_generation());
                self.done = true;
                Ok(())
            }
            MountSyncMode::Mount => {
                if !self.shared.capabilities().can_sync {
                    return Err(DeviceError::unsupported(
                        "the mount does not report CanSync",
                    ));
                }
                if state.tracking.known() == Some(&false) {
                    return Err(DeviceError::invalid_operation(
                        "sync needs tracking on (M-g); turn tracking on first",
                    ));
                }
                drv.sync(native)?;
                self.sent_mono_ns = clock.now_mono_ns();
                Ok(())
            }
        }
    }

    fn step(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> OpProgress {
        if self.done {
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        let frame = self.shared.capabilities().native_frame;
        let after = drv.state().ok().and_then(|s| fresh_position(&s, frame));
        let converged = after.zip(self.native).is_some_and(|(at, aim)| {
            at.separation_arcmin(&EqCoord {
                frame: at.frame,
                ..aim
            })
            .is_some_and(|sep| sep * 60.0 <= SYNC_TOLERANCE_ARCSEC)
        });
        if converged {
            let generation = self.shared.bump_sync_generation();
            self.applied(after, generation);
            self.done = true;
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        if clock.now_mono_ns().saturating_sub(self.sent_mono_ns) >= duration_ns(VERIFY_TIMEOUT) {
            // The mount's model may have changed anyway: segment the series.
            let generation = self.shared.bump_sync_generation();
            self.applied(after, generation);
            self.done = true;
            return OpProgress::Done(OpOutcome::Unknown(DeviceError::outcome_unknown(format!(
                "the mount did not converge within {SYNC_TOLERANCE_ARCSEC}\" of the sync target in {} s",
                VERIFY_TIMEOUT.as_secs()
            ))));
        }
        OpProgress::Continue {
            next_poll: ACTIVE_POLL,
        }
    }

    fn cancel(&mut self, _drv: &mut DynMount) -> DeviceResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::super::MountSettings;
    use super::*;
    use crate::equipment::commands::MountCmd;
    use crate::equipment::sim::mount::{SimFaults, SimMountProfile};
    use crate::equipment::traits::MountDriver;
    use crate::equipment::types::{EqFrame, ErrorClass};

    fn current(rig: &mut Rig) -> EqCoord {
        let snapshot = rig.tick(Duration::from_millis(10), None);
        EqCoord::new(
            snapshot.state.ra_h.value.unwrap(),
            snapshot.state.dec_deg.value.unwrap(),
            EqFrame::Unknown,
        )
    }

    #[test]
    fn a_sync_converges_and_bumps_the_generation() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let here = current(&mut rig);
        let solved = EqCoord::new(here.ra_h + 0.01, here.dec_deg - 0.2, EqFrame::Unknown);
        rig.runtime.take_events();
        assert_eq!(
            rig.run(MountCmd::Sync(solved), operator()).unwrap(),
            OpOutcome::Succeeded
        );
        let snapshot = rig.tick(Duration::from_secs(1), None);
        assert_eq!(snapshot.sync_generation, 1);
        let events = rig.runtime.take_events();
        let applied = events.iter().find(|e| e.name() == "sync_applied").unwrap();
        assert!(matches!(
            applied.event,
            EventKind::SyncApplied { before: Some(b), after: Some(a), generation: 1 }
                if (b.dec_deg - here.dec_deg).abs() < 1e-6 && (a.dec_deg - solved.dec_deg).abs() < 1e-6
        ));
    }

    #[test]
    fn a_sync_that_does_not_converge_is_unknown_and_still_segments() {
        let mut rig = rig(SimMountProfile::esp32_like());
        rig.sim.control().set_faults(SimFaults {
            sync_ignored: true,
            ..SimFaults::default()
        });
        let here = current(&mut rig);
        let solved = EqCoord::new(here.ra_h, here.dec_deg + 1.0, EqFrame::Unknown);
        let outcome = rig.run(MountCmd::Sync(solved), operator()).unwrap();
        assert!(matches!(outcome, OpOutcome::Unknown(_)), "{outcome:?}");
        assert_eq!(rig.tick(Duration::from_secs(1), None).sync_generation, 1);
    }

    #[test]
    fn sync_needs_tracking_and_can_sync() {
        let mut rig = rig(SimMountProfile::esp32_like());
        rig.sim.set_tracking(false).unwrap();
        let here = current(&mut rig);
        let outcome = rig.run(MountCmd::Sync(here), operator()).unwrap();
        assert!(matches!(outcome, OpOutcome::Failed(e) if e.class == ErrorClass::InvalidOperation));
        let mut profile = SimMountProfile::esp32_like();
        profile.can_sync = false;
        let mut rig = super::super::test_support::rig(profile);
        let here = current(&mut rig);
        let outcome = rig.run(MountCmd::Sync(here), operator()).unwrap();
        assert!(matches!(outcome, OpOutcome::Failed(e) if e.class == ErrorClass::Unsupported));
    }

    #[test]
    fn offset_mode_records_an_offset_and_never_syncs_the_mount() {
        let settings = MountSettings {
            sync_mode: MountSyncMode::Offset,
            ..MountSettings::default()
        };
        let mut rig = rig_with(SimMountProfile::esp32_like(), settings, Some(site()));
        let here = current(&mut rig);
        let solved = EqCoord::new(here.ra_h, here.dec_deg - 0.25, EqFrame::Unknown);
        rig.runtime.take_events();
        assert_eq!(
            rig.run(MountCmd::Sync(solved), operator()).unwrap(),
            OpOutcome::Succeeded
        );
        assert!(!rig.sim.control().calls().iter().any(|c| c == "sync"));
        let snapshot = rig.tick(Duration::from_secs(1), None);
        assert_eq!(snapshot.sync_generation, 0, "the mount model is untouched");
        let shared = rig.runtime.shared();
        let entry = shared.with_offsets(|o| o.entries()[0]);
        assert!((entry.d_dec_deg - 0.25).abs() < 1e-6);
        let events = rig.runtime.take_events();
        assert!(events.iter().any(|e| matches!(
            e.event,
            EventKind::SyncApplied {
                after: None,
                generation: 0,
                ..
            }
        )));
    }

    #[test]
    fn offsets_apply_within_their_region_only() {
        let mut offsets = PointingOffsets::default();
        let truth = EqCoord::new(23.9, 10.0, EqFrame::JNowApparent);
        let mount = EqCoord::new(0.1, 10.5, EqFrame::JNowApparent);
        offsets.record(mount, truth, Utc::now());
        assert!((offsets.entries()[0].d_ra_h - 0.2).abs() < 1e-9, "RA wraps");
        let near = offsets.apply(EqCoord::new(0.0, 12.0, EqFrame::JNowApparent));
        assert!((near.ra_h - 0.2).abs() < 1e-9 && (near.dec_deg - 12.5).abs() < 1e-9);
        let far = EqCoord::new(12.0, -40.0, EqFrame::JNowApparent);
        assert_eq!(offsets.apply(far), far);
        // A new solve in the same region replaces the old one.
        offsets.record(truth, truth, Utc::now());
        assert_eq!(offsets.entries().len(), 1);
        assert_eq!(offsets.entries()[0].d_dec_deg, 0.0);
        offsets.clear();
        assert!(offsets.entries().is_empty());
    }
}
