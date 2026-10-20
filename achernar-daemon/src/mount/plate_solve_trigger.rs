//! Plate-solve trigger — Section 8.3 (M-3).
//!
//! Decides which archived frames get solved without anyone asking. v6 §8.3.3
//! keeps deterministic startup, post-slew, post-flip and recovery rules
//! authoritative until a learned trigger is validated. Two of those have
//! events in the daemon today and are implemented here; meridian flips and
//! fault recovery have no implementation yet to hook the other two to.
//!
//! * **Startup** — the first light frame archived after the daemon starts.
//!   No exposure-time test: nothing about a session start says the telescope
//!   moved, so a camera with a wrong clock still qualifies.
//! * **Post-slew** — the first light frame whose exposure *began* after the
//!   mount reported arriving at a commanded target. A frame exposed while the
//!   mount was still moving is never used: its stars are trails.
//!
//! A rule stays pending until a solve under it succeeds. A failure — cloud, a
//! closed roof, a lens cap — leaves it pending, and `retry_after` keeps a long
//! cloudy spell from turning every frame into a minutes-long blind solve.
//!
//! Everything here is pure state; the runtime glue that feeds it mount probes
//! and archived frames, and runs the solves, is `super::auto_solve`.

use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::hardware::mount::EqCoord;

/// Why a solve ran. The serialized names are stored in
/// `plate_solve_events.trigger`, alongside `manual` for API requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SolveReason {
    Startup,
    PostSlew,
}

impl SolveReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Startup => "startup",
            Self::PostSlew => "post_slew",
        }
    }
}

/// A frame the archive just committed.
#[derive(Debug, Clone, Copy)]
pub struct ArchivedFrame<'a> {
    pub frame_id: &'a str,
    /// light | dark | flat | bias | focus | preview | verification
    pub capture_type: &'a str,
    /// From the driver's clock at the trigger, not the time the file landed.
    pub exposure_started_at: DateTime<Utc>,
}

/// A solve the trigger wants run.
#[derive(Debug, Clone, PartialEq)]
pub struct SolveJob {
    pub reason: SolveReason,
    pub frame_id: String,
    /// The commanded slew target. `None` for startup: the caller hints with
    /// the mount's reported position if it has one, otherwise solves blind.
    pub target: Option<EqCoord>,
    generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PendingRule {
    pub reason: SolveReason,
    pub target: Option<EqCoord>,
    /// A slew was commanded and the mount has not yet been seen to arrive. No
    /// frame qualifies until it has.
    pub awaiting_arrival: bool,
    /// Frames must begin exposing at or after this: the arrival time after a
    /// slew. `None` for startup, which has no motion to rule out — any light
    /// frame archived after the daemon starts will do, whatever its clock says.
    pub not_before: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TriggerStatus {
    pub enabled: bool,
    pub pending: Option<PendingRule>,
    pub in_flight: bool,
    pub retry_after_s: u64,
}

pub struct PlateSolveTrigger {
    enabled: bool,
    pending: Option<PendingRule>,
    /// Bumped by every slew, so a late settle report or a finished solve for
    /// an older target cannot act on a newer one.
    generation: u64,
    in_flight: bool,
    last_attempt: Option<DateTime<Utc>>,
    retry_after: Duration,
}

impl PlateSolveTrigger {
    /// A trigger with the startup rule pending.
    pub fn new(retry_after: Duration) -> Self {
        Self {
            enabled: true,
            pending: Some(PendingRule {
                reason: SolveReason::Startup,
                target: None,
                awaiting_arrival: false,
                not_before: None,
            }),
            generation: 0,
            in_flight: false,
            last_attempt: None,
            retry_after,
        }
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::new(Duration::ZERO)
        }
    }

    /// Changes with every commanded slew.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn status(&self) -> TriggerStatus {
        TriggerStatus {
            enabled: self.enabled,
            pending: self.enabled.then_some(self.pending).flatten(),
            in_flight: self.in_flight,
            retry_after_s: self.retry_after.as_secs(),
        }
    }

    /// A slew was commanded. Whatever was pending is moot — the telescope is
    /// going somewhere else. Returns the generation the settle report must
    /// quote.
    pub fn slew_commanded(&mut self, target: EqCoord) -> u64 {
        self.generation += 1;
        self.pending = Some(PendingRule {
            reason: SolveReason::PostSlew,
            target: Some(target),
            awaiting_arrival: true,
            not_before: None,
        });
        self.last_attempt = None;
        self.generation
    }

    /// The mount reported arriving for slew `generation` at `at`. Ignored if a
    /// later slew has superseded it.
    pub fn slew_settled(&mut self, generation: u64, at: DateTime<Utc>) -> bool {
        match self.pending.as_mut() {
            Some(rule) if generation == self.generation && rule.reason == SolveReason::PostSlew => {
                rule.awaiting_arrival = false;
                rule.not_before = Some(at);
                true
            }
            _ => false,
        }
    }

    /// The mount never reported arriving. Without a moment after which frames
    /// are known to be still, there is no frame to solve: drop the rule.
    pub fn slew_abandoned(&mut self, generation: u64) {
        if generation == self.generation
            && matches!(
                self.pending,
                Some(PendingRule {
                    reason: SolveReason::PostSlew,
                    ..
                })
            )
        {
            self.pending = None;
        }
    }

    /// A frame was archived at `now`. Returns a job if it should be solved.
    pub fn frame_archived(&mut self, frame: ArchivedFrame, now: DateTime<Utc>) -> Option<SolveJob> {
        if !self.enabled || self.in_flight {
            return None;
        }
        let rule = self.pending?;
        // Darks, flats and bias frames have no stars; autofocus frames are
        // defocused on purpose. Only sky frames meant to be sharp qualify.
        if !matches!(frame.capture_type, "light" | "verification") {
            return None;
        }
        if rule.awaiting_arrival
            || rule
                .not_before
                .is_some_and(|not_before| frame.exposure_started_at < not_before)
        {
            return None;
        }
        if let Some(last) = self.last_attempt {
            let retry_after = chrono::Duration::from_std(self.retry_after).ok()?;
            if now < last + retry_after {
                return None;
            }
        }
        self.in_flight = true;
        self.last_attempt = Some(now);
        Some(SolveJob {
            reason: rule.reason,
            frame_id: frame.frame_id.to_string(),
            target: rule.target,
            generation: self.generation,
        })
    }

    /// A job finished. Success satisfies the rule it served, unless a slew
    /// has replaced that rule in the meantime.
    pub fn job_finished(&mut self, job: &SolveJob, solved: bool) {
        self.in_flight = false;
        if solved && job.generation == self.generation {
            self.pending = None;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Slew settling
// ═══════════════════════════════════════════════════════════════════

/// Two probes within this of each other: the mount is not moving. Probes are
/// seconds apart and a slewing mount covers degrees per second; a tracking
/// mount holds RA and Dec constant.
const STATIONARY_DEG: f64 = 10.0 / 3600.0;

/// A reported position within this of the commanded target counts as there.
/// Generous, because drivers differ in what they report after a GoTo.
const ARRIVED_DEG: f64 = 0.05;

/// Decides from the passive mount probes when a commanded slew has finished.
///
/// The mount interface has no "slewing" flag, and GoTo returns as soon as the
/// command is accepted. So arrival is inferred: two consecutive probes taken
/// after the command, agreeing with each other, and either at the target or
/// after the mount was seen moving. The second clause catches drivers whose
/// reported position never quite matches the commanded numbers; the first
/// catches a slew so short that it finished between two probes.
pub struct SlewSettle {
    target: EqCoord,
    commanded_at: DateTime<Utc>,
    first: Option<EqCoord>,
    previous: Option<EqCoord>,
    seen_moving: bool,
}

impl SlewSettle {
    pub fn new(target: EqCoord, commanded_at: DateTime<Utc>) -> Self {
        Self {
            target,
            commanded_at,
            first: None,
            previous: None,
            seen_moving: false,
        }
    }

    /// Feed one probe; returns the arrival time once the mount has settled.
    pub fn observe(
        &mut self,
        probed_at: DateTime<Utc>,
        position: EqCoord,
    ) -> Option<DateTime<Utc>> {
        // A probe from before the command describes where the mount was.
        if probed_at <= self.commanded_at {
            return None;
        }
        let first = *self.first.get_or_insert(position);
        if separation_deg(first, position) > STATIONARY_DEG {
            self.seen_moving = true;
        }
        let previous = self.previous.replace(position)?;
        let still = separation_deg(previous, position) <= STATIONARY_DEG;
        let arrived = separation_deg(position, self.target) <= ARRIVED_DEG
            && separation_deg(previous, self.target) <= ARRIVED_DEG;
        (still && (arrived || self.seen_moving)).then_some(probed_at)
    }
}

/// Great-circle distance between two mount coordinates (RA hours, Dec degrees).
fn separation_deg(a: EqCoord, b: EqCoord) -> f64 {
    let (ra1, dec1) = ((a.ra * 15.0).to_radians(), a.dec.to_radians());
    let (ra2, dec2) = ((b.ra * 15.0).to_radians(), b.dec.to_radians());
    let dra = ra2 - ra1;
    let numerator = (dec2.cos() * dra.sin())
        .hypot(dec1.cos() * dec2.sin() - dec1.sin() * dec2.cos() * dra.cos());
    let denominator = dec1.sin() * dec2.sin() + dec1.cos() * dec2.cos() * dra.cos();
    numerator.atan2(denominator).to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(seconds: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_792_029_600 + seconds, 0).unwrap()
    }

    fn light(id: &str, started: i64) -> ArchivedFrame<'_> {
        ArchivedFrame {
            frame_id: id,
            capture_type: "light",
            exposure_started_at: t(started),
        }
    }

    const M67: EqCoord = EqCoord {
        ra: 8.855,
        dec: 11.8,
    };
    const M42: EqCoord = EqCoord {
        ra: 5.588,
        dec: -5.39,
    };

    #[test]
    fn the_first_light_frame_after_startup_is_solved_once_it_succeeds() {
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let dark = ArchivedFrame {
            capture_type: "dark",
            ..light("d", 5)
        };
        assert_eq!(
            trigger.frame_archived(dark, t(10)),
            None,
            "no stars in a dark"
        );

        let job = trigger.frame_archived(light("a", 20), t(30)).unwrap();
        assert_eq!(
            (job.reason, job.frame_id.as_str(), job.target),
            (SolveReason::Startup, "a", None)
        );
        assert_eq!(
            trigger.frame_archived(light("b", 31), t(40)),
            None,
            "one solve at a time"
        );

        trigger.job_finished(&job, true);
        assert_eq!(
            trigger.frame_archived(light("c", 50), t(60)),
            None,
            "rule satisfied"
        );
        assert!(trigger.status().pending.is_none());
    }

    #[test]
    fn a_failed_solve_retries_only_after_the_cooldown() {
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let job = trigger.frame_archived(light("a", 10), t(20)).unwrap();
        trigger.job_finished(&job, false);

        assert_eq!(
            trigger.frame_archived(light("b", 30), t(60)),
            None,
            "still cooling down"
        );
        let retry = trigger.frame_archived(light("c", 130), t(140)).unwrap();
        assert_eq!(retry.frame_id, "c");
    }

    #[test]
    fn after_a_slew_only_frames_begun_after_arrival_qualify() {
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let generation = trigger.slew_commanded(M67);

        // The mount has not been seen to arrive: nothing qualifies, however late.
        assert_eq!(trigger.frame_archived(light("moving", 50), t(60)), None);

        assert!(trigger.slew_settled(generation, t(90)));
        // Began exposing before arrival: trailed stars.
        assert_eq!(trigger.frame_archived(light("early", 85), t(100)), None);
        let job = trigger.frame_archived(light("late", 95), t(110)).unwrap();
        assert_eq!((job.reason, job.target), (SolveReason::PostSlew, Some(M67)));
    }

    #[test]
    fn a_wrong_camera_clock_passes_startup_but_never_a_slew() {
        // The OLS simulator stamps frames from its playback clock: 1970.
        let epoch = ArchivedFrame {
            exposure_started_at: DateTime::from_timestamp(0, 0).unwrap(),
            ..light("sim", 0)
        };
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let job = trigger.frame_archived(epoch, t(10)).unwrap();
        assert_eq!(job.reason, SolveReason::Startup);
        trigger.job_finished(&job, true);

        // After a slew, a frame that cannot be shown to follow the arrival is
        // indistinguishable from one exposed while the mount was moving.
        let generation = trigger.slew_commanded(M67);
        trigger.slew_settled(generation, t(30));
        assert_eq!(trigger.frame_archived(epoch, t(40)), None);
    }

    #[test]
    fn a_newer_slew_supersedes_everything_about_the_older_one() {
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let first = trigger.slew_commanded(M67);
        trigger.slew_settled(first, t(30));
        let job = trigger.frame_archived(light("m67", 40), t(50)).unwrap();

        let second = trigger.slew_commanded(M42);
        // A late arrival report and a late success for M67 change nothing.
        assert!(!trigger.slew_settled(first, t(60)));
        trigger.job_finished(&job, true);
        let pending = trigger.status().pending.unwrap();
        assert_eq!(
            (pending.target, pending.awaiting_arrival),
            (Some(M42), true)
        );

        trigger.slew_settled(second, t(120));
        let job = trigger.frame_archived(light("m42", 125), t(130)).unwrap();
        assert_eq!(job.target, Some(M42));
    }

    #[test]
    fn an_abandoned_slew_leaves_nothing_to_solve() {
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let generation = trigger.slew_commanded(M67);
        trigger.slew_abandoned(generation);
        assert!(trigger.status().pending.is_none());
        assert_eq!(trigger.frame_archived(light("a", 900), t(910)), None);
    }

    #[test]
    fn a_disabled_trigger_never_asks() {
        let mut trigger = PlateSolveTrigger::disabled();
        assert_eq!(trigger.frame_archived(light("a", 3600), Utc::now()), None);
        let status = trigger.status();
        assert!(!status.enabled && status.pending.is_none());
    }

    #[test]
    fn settling_needs_two_still_probes_at_the_target_after_the_command() {
        let mut settle = SlewSettle::new(M67, t(0));
        // A probe taken before the command says where the mount used to be.
        assert_eq!(settle.observe(t(-1), M67), None);
        assert_eq!(settle.observe(t(2), M42), None);
        assert_eq!(settle.observe(t(4), EqCoord { ra: 7.2, dec: 3.0 }), None);
        assert_eq!(settle.observe(t(6), M67), None, "one probe is not still");
        assert_eq!(settle.observe(t(8), M67), Some(t(8)));
    }

    #[test]
    fn a_driver_that_reports_slightly_different_numbers_still_settles_after_moving() {
        // Reported position ends 0.3 deg from the commanded numbers — outside
        // the arrival tolerance — but the mount was seen moving and stopped.
        let mut settle = SlewSettle::new(M67, t(0));
        let reported = EqCoord {
            ra: 8.855,
            dec: 12.1,
        };
        assert_eq!(settle.observe(t(2), M42), None);
        assert_eq!(settle.observe(t(4), reported), None);
        assert_eq!(settle.observe(t(6), reported), Some(t(6)));
    }

    #[test]
    fn a_mount_that_never_moves_away_from_elsewhere_does_not_settle() {
        // Still, but neither at the target nor ever seen moving: the GoTo has
        // not started yet, or was ignored. Not a settled slew.
        let mut settle = SlewSettle::new(M67, t(0));
        for second in [2, 4, 6, 8] {
            assert_eq!(settle.observe(t(second), M42), None);
        }
    }

    #[test]
    fn separation_handles_the_ra_wrap() {
        let a = EqCoord {
            ra: 23.99,
            dec: 0.0,
        };
        let b = EqCoord { ra: 0.01, dec: 0.0 };
        assert!((separation_deg(a, b) - 0.3).abs() < 1e-9);
    }
}
