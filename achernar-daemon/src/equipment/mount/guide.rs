//! Pulse guiding and guide rates (§3.2 M-i, D20). Owner: WP1A-4.
//!
//! - A pulse requires CanPulseGuide, is refused while parked, and is clamped
//!   to `[mount] max_pulse_ms`. The refusal while PHD2 is `Guiding` or
//!   `Calibrating` is the gate check in `limits.rs`, which sees the guider
//!   snapshot (D20).
//! - Completion is IsPulseGuiding false. A short pulse whose first read is
//!   already false counts as success. A mount that cannot report
//!   IsPulseGuiding completes after the duration plus one poll.
//! - `guide_dec_reverse_on_pier_west` swaps North and South while the mount
//!   points west of the pier (ASCOM pulse-guide FAQ).
//! - Guide rates are read whenever the mount implements them, not only when
//!   CanSetGuideRates (NINA leaves them NaN otherwise). They are written only
//!   when CanSetGuideRates. `[mount] guide_rate` is a fallback with source
//!   `Profile` (applied by the snapshot builder).

use std::sync::Arc;
use std::time::Duration;

use crate::equipment::events::EventKind;
use crate::equipment::ops::{OpKind, OpProgress, Operation};
use crate::equipment::traits::Clock;
use crate::equipment::types::{
    DeviceError, DeviceResult, GuideDirection, ObsState, OpOutcome, PierSide,
};

use super::slew::abort_slew;
use super::{
    duration_ns, is_parked, parked_error, CommandMeta, DynMount, MountShared, ACTIVE_POLL,
    PULSE_POLL, SETTER_DEADLINE, SIDEREAL_RATE_ARCSEC_S, VERIFY_TIMEOUT,
};

/// Guide rates above this many times sidereal are refused as a typo.
pub const MAX_GUIDE_RATE_SIDEREAL: f64 = 10.0;

/// The pulse length actually sent.
pub fn clamp_pulse_ms(ms: u32, max_pulse_ms: u32) -> u32 {
    ms.min(max_pulse_ms.max(1))
}

/// North and South swap while pointing west, when configured.
pub fn effective_direction(
    dir: GuideDirection,
    side: Option<PierSide>,
    reverse_on_west: bool,
) -> GuideDirection {
    if !reverse_on_west || side != Some(PierSide::West) {
        return dir;
    }
    match dir {
        GuideDirection::North => GuideDirection::South,
        GuideDirection::South => GuideDirection::North,
        other => other,
    }
}

/// One Achernar-issued pulse (polled at 10 Hz, §2.7).
pub struct PulseGuideOp {
    shared: Arc<MountShared>,
    meta: CommandMeta,
    requested: GuideDirection,
    requested_ms: u32,
    sent_ms: u32,
    sent_mono_ns: u64,
}

impl PulseGuideOp {
    pub fn new(shared: Arc<MountShared>, meta: CommandMeta, dir: GuideDirection, ms: u32) -> Self {
        Self {
            shared,
            meta,
            requested: dir,
            requested_ms: ms,
            sent_ms: 0,
            sent_mono_ns: 0,
        }
    }
}

impl Operation<DynMount> for PulseGuideOp {
    fn kind(&self) -> OpKind {
        OpKind::PulseGuide
    }

    fn deadline(&self) -> Duration {
        Duration::from_millis(u64::from(clamp_pulse_ms(
            self.requested_ms,
            self.shared.settings().max_pulse_ms,
        ))) + Duration::from_secs(10)
    }

    fn start(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> DeviceResult<()> {
        if !self.shared.capabilities().can_pulse_guide {
            return Err(DeviceError::unsupported(
                "the mount does not report CanPulseGuide",
            ));
        }
        if self.requested_ms == 0 {
            return Err(DeviceError::invalid_value(
                "a pulse needs a duration above 0 ms",
            ));
        }
        let state = drv.state()?;
        if is_parked(&state) {
            return Err(parked_error("pulse guiding"));
        }
        let settings = self.shared.settings();
        let ms = clamp_pulse_ms(self.requested_ms, settings.max_pulse_ms);
        let dir = effective_direction(
            self.requested,
            state.side_of_pier.known().copied(),
            settings.guide_dec_reverse_on_pier_west,
        );
        drv.pulse_guide(dir, ms)?;
        self.sent_ms = ms;
        self.sent_mono_ns = clock.now_mono_ns();
        self.shared.emit(
            EventKind::PulseGuide {
                dir,
                ms,
                source: self.meta.source.clone(),
            },
            Some(self.meta.command_id),
        );
        Ok(())
    }

    fn step(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> OpProgress {
        let reading = drv.state().ok().map(|s| s.is_pulse_guiding);
        match reading.as_ref().and_then(|r| r.fresh_value()) {
            Some(false) => return OpProgress::Done(OpOutcome::Succeeded),
            Some(true) => {
                return OpProgress::Continue {
                    next_poll: PULSE_POLL,
                }
            }
            None => {}
        }
        // No completion signal: the duration plus one poll.
        let unsupported = reading.is_some_and(|r| r.state == ObsState::Unsupported);
        let due = self.sent_mono_ns
            + duration_ns(Duration::from_millis(u64::from(self.sent_ms)) + PULSE_POLL);
        if unsupported && clock.now_mono_ns() >= due {
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        OpProgress::Continue {
            next_poll: PULSE_POLL,
        }
    }

    fn cancel(&mut self, drv: &mut DynMount) -> DeviceResult<()> {
        abort_slew(drv)
    }
}

/// Write both guide rates (arcsec/s; the driver converts to its units).
pub struct SetGuideRatesOp {
    shared: Arc<MountShared>,
    _meta: CommandMeta,
    ra_arcsec_s: f64,
    dec_arcsec_s: f64,
    sent_mono_ns: u64,
}

impl SetGuideRatesOp {
    pub fn new(
        shared: Arc<MountShared>,
        meta: CommandMeta,
        ra_arcsec_s: f64,
        dec_arcsec_s: f64,
    ) -> DeviceResult<Self> {
        let max = MAX_GUIDE_RATE_SIDEREAL * SIDEREAL_RATE_ARCSEC_S;
        for rate in [ra_arcsec_s, dec_arcsec_s] {
            if !rate.is_finite() || rate <= 0.0 || rate > max {
                return Err(DeviceError::invalid_value(format!(
                    "guide rate {rate} arcsec/s is outside (0, {max:.1}]"
                )));
            }
        }
        Ok(Self {
            shared,
            _meta: meta,
            ra_arcsec_s,
            dec_arcsec_s,
            sent_mono_ns: 0,
        })
    }
}

fn close(read: Option<&f64>, want: f64) -> bool {
    read.is_some_and(|r| (r - want).abs() <= (want * 0.01).max(0.05))
}

impl Operation<DynMount> for SetGuideRatesOp {
    fn kind(&self) -> OpKind {
        OpKind::SetGuideRates
    }

    fn deadline(&self) -> Duration {
        SETTER_DEADLINE
    }

    fn start(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> DeviceResult<()> {
        if !self.shared.capabilities().can_set_guide_rates {
            return Err(DeviceError::unsupported(
                "the mount does not report CanSetGuideRates",
            ));
        }
        drv.set_guide_rates(self.ra_arcsec_s / 3600.0, self.dec_arcsec_s / 3600.0)?;
        self.sent_mono_ns = clock.now_mono_ns();
        Ok(())
    }

    fn step(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> OpProgress {
        let Ok(state) = drv.state() else {
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL,
            };
        };
        let (ra, dec) = (
            &state.guide_rate_ra_arcsec_s,
            &state.guide_rate_dec_arcsec_s,
        );
        if ra.state == ObsState::Unsupported && dec.state == ObsState::Unsupported {
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        let fresh_since = |o: &crate::equipment::types::Observed<f64>| {
            o.mono_ns.is_some_and(|at| at >= self.sent_mono_ns)
        };
        if fresh_since(ra)
            && fresh_since(dec)
            && close(ra.fresh_value(), self.ra_arcsec_s)
            && close(dec.fresh_value(), self.dec_arcsec_s)
        {
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        if clock.now_mono_ns().saturating_sub(self.sent_mono_ns) >= duration_ns(VERIFY_TIMEOUT) {
            return OpProgress::Done(OpOutcome::Failed(DeviceError::driver(format!(
                "guide rates read {:?}/{:?} arcsec/s after writing {}/{}",
                ra.value, dec.value, self.ra_arcsec_s, self.dec_arcsec_s
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
    use crate::equipment::types::{ErrorClass, Source};

    #[test]
    fn a_pulse_completes_when_is_pulse_guiding_reads_false() {
        let mut rig = rig(SimMountProfile::esp32_like());
        rig.runtime.take_events();
        let started = rig.clock.now_mono_ns();
        let outcome = rig
            .run(
                MountCmd::PulseGuide {
                    dir: GuideDirection::North,
                    ms: 300,
                },
                operator(),
            )
            .unwrap();
        assert_eq!(outcome, OpOutcome::Succeeded);
        let elapsed = rig.clock.now_mono_ns() - started;
        assert!((300_000_000..=500_000_000).contains(&elapsed), "{elapsed}");
        let events = rig.runtime.take_events();
        assert!(events.iter().any(|e| matches!(
            &e.event,
            EventKind::PulseGuide {
                dir: GuideDirection::North,
                ms: 300,
                ..
            }
        )));
    }

    #[test]
    fn an_instant_false_short_pulse_counts_as_success() {
        let mut rig = rig(SimMountProfile::esp32_like());
        rig.sim.control().set_faults(SimFaults {
            instant_pulse: true,
            ..SimFaults::default()
        });
        let started = rig.clock.now_mono_ns();
        let outcome = rig
            .run(
                MountCmd::PulseGuide {
                    dir: GuideDirection::East,
                    ms: 20,
                },
                operator(),
            )
            .unwrap();
        assert_eq!(outcome, OpOutcome::Succeeded);
        assert!(
            rig.clock.now_mono_ns() - started <= 300_000_000,
            "first poll decides"
        );
    }

    #[test]
    fn pulses_without_is_pulse_guiding_wait_the_duration_plus_a_poll() {
        let mut profile = SimMountProfile::esp32_like();
        profile.reports_pulse_guiding = false;
        let mut rig = rig(profile);
        let started = rig.clock.now_mono_ns();
        let outcome = rig
            .run(
                MountCmd::PulseGuide {
                    dir: GuideDirection::West,
                    ms: 500,
                },
                operator(),
            )
            .unwrap();
        assert_eq!(outcome, OpOutcome::Succeeded);
        assert!(rig.clock.now_mono_ns() - started >= 600_000_000);
    }

    #[test]
    fn pulses_are_clamped_reversed_on_west_and_refused_when_parked() {
        let settings = MountSettings {
            max_pulse_ms: 1000,
            guide_dec_reverse_on_pier_west: true,
            ..MountSettings::default()
        };
        let mut profile = SimMountProfile::esp32_like();
        profile.pier_side = crate::equipment::sim::mount::SimPierSide::Static(PierSide::West);
        let mut rig = rig_with(profile, settings, Some(site()));
        rig.runtime.take_events();
        let outcome = rig
            .run(
                MountCmd::PulseGuide {
                    dir: GuideDirection::North,
                    ms: 4000,
                },
                operator(),
            )
            .unwrap();
        assert_eq!(outcome, OpOutcome::Succeeded);
        assert!(rig
            .sim
            .control()
            .calls()
            .iter()
            .any(|c| c == "pulse_guide:South:1000"));
        assert_eq!(
            effective_direction(GuideDirection::East, Some(PierSide::West), true),
            GuideDirection::East
        );
        assert_eq!(
            effective_direction(GuideDirection::North, Some(PierSide::East), true),
            GuideDirection::North
        );
        assert_eq!(clamp_pulse_ms(7000, 5000), 5000);

        let mut profile = SimMountProfile::esp32_like();
        profile.start_parked = true;
        let mut rig = super::super::test_support::rig(profile);
        let outcome = rig
            .run(
                MountCmd::PulseGuide {
                    dir: GuideDirection::North,
                    ms: 100,
                },
                operator(),
            )
            .unwrap();
        assert!(
            matches!(outcome, OpOutcome::Failed(e) if e.class == ErrorClass::InvalidWhileParked)
        );
    }

    #[test]
    fn guide_rates_are_written_only_when_settable_and_verified() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let outcome = rig
            .run(
                MountCmd::SetGuideRates {
                    ra_arcsec_s: 10.0,
                    dec_arcsec_s: 5.0,
                },
                operator(),
            )
            .unwrap();
        assert_eq!(outcome, OpOutcome::Succeeded);
        let snapshot = rig.tick(Duration::from_secs(1), None);
        assert!((snapshot.state.guide_rate_ra_arcsec_s.value.unwrap() - 10.0).abs() < 1e-9);
        assert!((snapshot.state.guide_rate_dec_arcsec_s.value.unwrap() - 5.0).abs() < 1e-9);
        // The GEM profile has CanSetGuideRates = false but still reports them.
        let mut rig = super::super::test_support::rig(SimMountProfile::german_equatorial());
        let snapshot = rig.tick(Duration::from_secs(1), None);
        assert_eq!(
            snapshot.state.guide_rate_ra_arcsec_s.source,
            Source::Property
        );
        assert!(snapshot.state.guide_rate_ra_arcsec_s.is_fresh());
        let outcome = rig
            .run(
                MountCmd::SetGuideRates {
                    ra_arcsec_s: 10.0,
                    dec_arcsec_s: 5.0,
                },
                operator(),
            )
            .unwrap();
        assert!(matches!(outcome, OpOutcome::Failed(e) if e.class == ErrorClass::Unsupported));
        assert!(SetGuideRatesOp::new(rig.runtime.shared(), operator(), -1.0, 5.0).is_err());
    }

    #[test]
    fn the_profile_guide_rate_fills_in_only_when_the_mount_reports_none() {
        let mut profile = SimMountProfile::esp32_like();
        profile.guide_rate_deg_s = None;
        let mut rig = rig(profile);
        let snapshot = rig.tick(Duration::from_secs(1), None);
        let rate = &snapshot.state.guide_rate_ra_arcsec_s;
        assert_eq!(rate.source, Source::Profile);
        assert!((rate.value.unwrap() - 0.5 * SIDEREAL_RATE_ARCSEC_S).abs() < 1e-9);
    }
}
