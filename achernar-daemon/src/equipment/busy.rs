//! Critical-transaction flags (spec §2.9 "BusySet", C9, H20). Owner: WP1A-2.
//!
//! The compute-aware runtime never switches model selection while any flag
//! is set (v6 §6.7.6, §6.7.9). A flag is set by either source:
//!
//! - an explicit claim ([`BusyPublisher::claim`]), held by a guard that
//!   clears it on drop: the camera hook (`exposing`), the focus run
//!   (`autofocus`), the center loop (`centering`), the flip workflow
//!   (`flipping`);
//! - the device snapshots ([`derive_busy`]): slewing, parking, settling,
//!   dithering, guider settling, filter changes, cover and calibrator moves.
//!
//! The published [`BusySet`] is the union, sent only when it changes.

use std::sync::{Arc, Mutex, MutexGuard, Weak};

use tokio::sync::watch;

use super::ops::{OpKind, OpPhase};
use super::ring::DeviceSnapshot;
use super::snapshot::{BusySet, CalibratorState, CoverState, EquipmentSnapshot};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// One [`BusySet`] flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BusyFlag {
    Exposing,
    Autofocus,
    Slewing,
    Settling,
    Parking,
    Flipping,
    Centering,
    Dithering,
    GuiderSettling,
    FilterChanging,
    CoverMoving,
    CalibratorChanging,
}

impl BusyFlag {
    pub const ALL: [BusyFlag; 12] = [
        BusyFlag::Exposing,
        BusyFlag::Autofocus,
        BusyFlag::Slewing,
        BusyFlag::Settling,
        BusyFlag::Parking,
        BusyFlag::Flipping,
        BusyFlag::Centering,
        BusyFlag::Dithering,
        BusyFlag::GuiderSettling,
        BusyFlag::FilterChanging,
        BusyFlag::CoverMoving,
        BusyFlag::CalibratorChanging,
    ];

    fn index(self) -> usize {
        self as usize
    }

    pub fn get(self, busy: &BusySet) -> bool {
        match self {
            BusyFlag::Exposing => busy.exposing,
            BusyFlag::Autofocus => busy.autofocus,
            BusyFlag::Slewing => busy.slewing,
            BusyFlag::Settling => busy.settling,
            BusyFlag::Parking => busy.parking,
            BusyFlag::Flipping => busy.flipping,
            BusyFlag::Centering => busy.centering,
            BusyFlag::Dithering => busy.dithering,
            BusyFlag::GuiderSettling => busy.guider_settling,
            BusyFlag::FilterChanging => busy.filter_changing,
            BusyFlag::CoverMoving => busy.cover_moving,
            BusyFlag::CalibratorChanging => busy.calibrator_changing,
        }
    }

    pub fn set(self, busy: &mut BusySet, on: bool) {
        let slot = match self {
            BusyFlag::Exposing => &mut busy.exposing,
            BusyFlag::Autofocus => &mut busy.autofocus,
            BusyFlag::Slewing => &mut busy.slewing,
            BusyFlag::Settling => &mut busy.settling,
            BusyFlag::Parking => &mut busy.parking,
            BusyFlag::Flipping => &mut busy.flipping,
            BusyFlag::Centering => &mut busy.centering,
            BusyFlag::Dithering => &mut busy.dithering,
            BusyFlag::GuiderSettling => &mut busy.guider_settling,
            BusyFlag::FilterChanging => &mut busy.filter_changing,
            BusyFlag::CoverMoving => &mut busy.cover_moving,
            BusyFlag::CalibratorChanging => &mut busy.calibrator_changing,
        };
        *slot = on;
    }
}

/// The union of two busy sets.
pub fn union(a: BusySet, b: BusySet) -> BusySet {
    let mut out = a;
    for flag in BusyFlag::ALL {
        if flag.get(&b) {
            flag.set(&mut out, true);
        }
    }
    out
}

#[derive(Debug, Default)]
struct BusyState {
    claims: [u32; 12],
    derived: BusySet,
}

impl BusyState {
    fn merged(&self) -> BusySet {
        let mut out = self.derived;
        for flag in BusyFlag::ALL {
            if self.claims[flag.index()] > 0 {
                flag.set(&mut out, true);
            }
        }
        out
    }
}

/// Publishes the [`BusySet`] watch (H20).
#[derive(Debug)]
pub struct BusyPublisher {
    tx: watch::Sender<BusySet>,
    state: Mutex<BusyState>,
}

impl Default for BusyPublisher {
    fn default() -> Self {
        Self::new()
    }
}

impl BusyPublisher {
    pub fn new() -> Self {
        let (tx, _) = watch::channel(BusySet::default());
        Self {
            tx,
            state: Mutex::new(BusyState::default()),
        }
    }

    pub fn subscribe(&self) -> watch::Receiver<BusySet> {
        self.tx.subscribe()
    }

    pub fn current(&self) -> BusySet {
        *self.tx.borrow()
    }

    fn publish(&self, state: &BusyState) -> bool {
        let merged = state.merged();
        self.tx.send_if_modified(|current| {
            if *current == merged {
                false
            } else {
                *current = merged;
                true
            }
        })
    }

    /// Hold `flag` until the guard drops. Claims nest: the flag clears when
    /// the last guard for it goes.
    pub fn claim(self: &Arc<Self>, flag: BusyFlag) -> BusyGuard {
        let mut state = lock(&self.state);
        state.claims[flag.index()] = state.claims[flag.index()].saturating_add(1);
        self.publish(&state);
        BusyGuard {
            publisher: Arc::downgrade(self),
            flag,
        }
    }

    fn release(&self, flag: BusyFlag) {
        let mut state = lock(&self.state);
        state.claims[flag.index()] = state.claims[flag.index()].saturating_sub(1);
        self.publish(&state);
    }

    /// Replace the snapshot-derived flags. Returns true when the published
    /// set changed.
    pub fn set_derived(&self, derived: BusySet) -> bool {
        let mut state = lock(&self.state);
        state.derived = derived;
        self.publish(&state)
    }
}

/// Clears its flag on drop (on success, error, cancel and panic alike).
#[derive(Debug)]
pub struct BusyGuard {
    publisher: Weak<BusyPublisher>,
    flag: BusyFlag,
}

impl BusyGuard {
    pub fn flag(&self) -> BusyFlag {
        self.flag
    }
}

impl Drop for BusyGuard {
    fn drop(&mut self) {
        if let Some(publisher) = self.publisher.upgrade() {
            publisher.release(self.flag);
        }
    }
}

fn op_is<S: DeviceSnapshot>(snap: &S, kinds: &[OpKind]) -> bool {
    snap.active_op().is_some_and(|op| kinds.contains(&op.kind))
}

fn settling<S: DeviceSnapshot>(snap: &S) -> bool {
    snap.active_op()
        .is_some_and(|op| op.phase == OpPhase::Settling)
}

/// The flags implied by the device snapshots in an aggregate.
pub fn derive_busy(snap: &EquipmentSnapshot) -> BusySet {
    let mut busy = BusySet {
        exposing: snap.camera.exposing.known().copied().unwrap_or(false),
        ..BusySet::default()
    };
    if let Some(mount) = &snap.mount {
        busy.slewing = mount.state.slewing.known().copied().unwrap_or(false)
            || mount.jog_active
            || op_is(
                mount.as_ref(),
                &[
                    OpKind::Slew,
                    OpKind::SlewAltAz,
                    OpKind::Home,
                    OpKind::MoveAxis,
                ],
            );
        busy.parking = op_is(mount.as_ref(), &[OpKind::Park, OpKind::Unpark]);
        busy.flipping = op_is(mount.as_ref(), &[OpKind::Flip]);
        busy.centering = op_is(mount.as_ref(), &[OpKind::Center]);
        busy.settling |= settling(mount.as_ref());
    }
    if let Some(focuser) = &snap.focuser {
        busy.settling |= focuser.settling || settling(focuser.as_ref());
        busy.autofocus = op_is(focuser.as_ref(), &[OpKind::Autofocus]);
    }
    if let Some(wheel) = &snap.filter_wheel {
        busy.filter_changing = wheel.moving
            || op_is(
                wheel.as_ref(),
                &[OpKind::FilterMove, OpKind::FilterHandshake],
            );
        busy.settling |= settling(wheel.as_ref());
    }
    if let Some(guider) = &snap.guider {
        busy.dithering = op_is(guider.as_ref(), &[OpKind::Dither]);
        busy.guider_settling = guider.settling;
    }
    if let Some(rotator) = &snap.rotator {
        busy.settling |= settling(rotator.as_ref());
    }
    if let Some(cover) = &snap.cover_calibrator {
        busy.cover_moving = cover.state.cover_moving.known().copied().unwrap_or(false)
            || cover.state.cover_state.known() == Some(&CoverState::Moving)
            || op_is(cover.as_ref(), &[OpKind::CoverOpen, OpKind::CoverClose]);
        busy.calibrator_changing = cover
            .state
            .calibrator_changing
            .known()
            .copied()
            .unwrap_or(false)
            || cover.state.calibrator_state.known() == Some(&CalibratorState::NotReady)
            || op_is(
                cover.as_ref(),
                &[OpKind::CalibratorOn, OpKind::CalibratorOff],
            );
    }
    busy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::ops::ActiveOp;
    use crate::equipment::snapshot::{FilterWheelSnapshot, GuiderSnapshot, MountSnapshot};
    use crate::equipment::types::{CommandId, Observed, Source};
    use chrono::Utc;

    fn op(kind: OpKind, phase: OpPhase) -> ActiveOp {
        ActiveOp {
            command_id: CommandId::new(),
            kind,
            phase,
            started_at: Utc::now(),
            started_mono_ns: 0,
            deadline_mono_ns: 0,
        }
    }

    #[test]
    fn claims_nest_and_clear_on_drop() {
        let busy = Arc::new(BusyPublisher::new());
        let mut rx = busy.subscribe();
        let a = busy.claim(BusyFlag::Exposing);
        let b = busy.claim(BusyFlag::Exposing);
        assert!(rx.borrow_and_update().exposing);
        drop(a);
        assert!(busy.current().exposing, "still held by the second guard");
        drop(b);
        assert!(!busy.current().exposing);
        assert!(rx.has_changed().unwrap());
        let _af = busy.claim(BusyFlag::Autofocus);
        assert!(busy.current().any());
    }

    #[test]
    fn derived_flags_merge_with_claims_and_publish_only_changes() {
        let busy = Arc::new(BusyPublisher::new());
        let mut rx = busy.subscribe();
        let derived = BusySet {
            slewing: true,
            ..BusySet::default()
        };
        assert!(busy.set_derived(derived));
        assert!(!busy.set_derived(derived), "unchanged sets are not re-sent");
        let _guard = busy.claim(BusyFlag::Centering);
        let now = *rx.borrow_and_update();
        assert!(now.slewing && now.centering);
        assert!(busy.set_derived(BusySet::default()));
        assert!(!busy.current().slewing && busy.current().centering);
    }

    #[test]
    fn snapshots_imply_the_motion_flags() {
        let at = Utc::now();
        let mut snap = EquipmentSnapshot::default();
        let mut mount = MountSnapshot::default();
        mount.state.slewing = Observed::fresh(true, Source::DeviceState, at, 1);
        snap.mount = Some(Arc::new(mount));
        let wheel = FilterWheelSnapshot {
            active_op: Some(op(OpKind::FilterMove, OpPhase::Settling)),
            ..FilterWheelSnapshot::default()
        };
        snap.filter_wheel = Some(Arc::new(wheel));
        let guider = GuiderSnapshot {
            settling: true,
            active_op: Some(op(OpKind::Dither, OpPhase::InProgress)),
            ..GuiderSnapshot::default()
        };
        snap.guider = Some(Arc::new(guider));
        snap.camera.exposing = Observed::fresh(true, Source::Derived, at, 1);
        let busy = derive_busy(&snap);
        assert!(busy.slewing && busy.filter_changing && busy.settling);
        assert!(busy.dithering && busy.guider_settling && busy.exposing);
        assert!(!busy.parking && !busy.autofocus);
        assert!(!derive_busy(&EquipmentSnapshot::default()).any());
        let merged = union(
            BusySet {
                parking: true,
                ..BusySet::default()
            },
            busy,
        );
        assert!(merged.parking && merged.slewing);
    }
}
