//! Plate-solve trigger — Section 8.3 (M-3, v6 §8.3.3; equipment spec H11).
//!
//! Decides which archived frames get solved without anyone asking. v6
//! §8.3.3 keeps deterministic rules authoritative until a learned trigger is
//! validated. Every rule is driven by an equipment event from the hub's bus;
//! nothing here infers motion from position probes.
//!
//! * **Startup** — the first light frame archived after the daemon starts.
//!   No exposure-time test: nothing about a session start says the telescope
//!   moved, so a camera with a wrong clock still qualifies.
//! * **PostSlew** — armed by `slew_commanded`, and fired by that slew's
//!   `slew_settled` (§3.2 M-c step 9): the first light frame whose exposure
//!   *began* after the settle qualifies. A frame exposed while the mount was
//!   moving is never used. A slew that ends without settling (aborted,
//!   failed, or a mount that cannot report Slewing) drops the rule.
//! * **PostFlip** — `pier_side_changed` (during a slew it re-labels that
//!   slew's rule) or `meridian_flip{done}`. Polar and Alt/Az mounts never
//!   change pier side, so the rule never fires for them, which is correct.
//! * **Recovery** — `device_reconnected` of the mount or the guider,
//!   `star_lost`, `lock_position_lost`, the safety state returning to safe,
//!   and the vision loop's cloud-cleared event.
//! * **Center** — the center loop (M-n) solves its own frames under this
//!   reason; its `centered` event satisfies whatever rule was pending.
//!
//! A rule stays pending until a solve under it succeeds. A failure — cloud, a
//! closed roof, a lens cap — leaves it pending, and `retry_after` keeps a long
//! cloudy spell from turning every frame into a minutes-long blind solve.
//!
//! A PostSlew hint is the slew's J2000 target, else its native-frame target.
//! An Alt/Az slew gives none: its RA/Dec is informative only (RA 0 h, Dec 0°
//! when the site or LST is unknown), so the solve falls back to the mount's
//! own report after the settle, or runs blind. A failed hinted solve drops
//! the hint, so a wrong one is never retried.
//!
//! Every decision about an eligible light frame while a rule is pending is an
//! H23 `policy_decisions` record ([`decision_record`]): the rule, the M-3
//! readings with their ages, the safe arms (`solve`, `skip`) and the chosen
//! one. The rules are deterministic, so no propensities are recorded. These
//! rows are the behaviour log a learned trigger is later trained and
//! evaluated against.
//!
//! Everything here is pure state; the runtime glue that subscribes it to the
//! bus, feeds it archived frames and runs the solves is `super::auto_solve`.

use std::collections::BTreeMap;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::json;
use uuid::Uuid;

use crate::equipment::events::{EquipmentEvent, EventKind, FlipPhase, PolicyDecisionRecord};
use crate::equipment::snapshot::SafetyGate;
use crate::equipment::types::{
    AutonomyMode, CommandId, DeviceKind, EqCoord, EqFrame, OpOutcome, PierSide,
};

/// `policy_decisions.policy`.
pub const POLICY: &str = "m3_plate_solve_trigger";
/// `policy_decisions.policy_version`.
pub const POLICY_VERSION: &str = "deterministic-rules-2";

/// Why a solve ran. The serialized names are stored in
/// `plate_solve_events.trigger`, alongside `manual` for API requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SolveReason {
    Startup,
    PostSlew,
    PostFlip,
    Recovery,
    Center,
}

impl SolveReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Startup => "startup",
            Self::PostSlew => "post_slew",
            Self::PostFlip => "post_flip",
            Self::Recovery => "recovery",
            Self::Center => "center",
        }
    }
}

/// A frame the archive just committed.
#[derive(Debug, Clone, Copy)]
pub struct ArchivedFrame<'a> {
    pub frame_id: &'a str,
    /// light | dark | flat | bias | focus | preview | verification | center
    pub capture_type: &'a str,
    /// From the driver's clock at the trigger, not the time the file landed.
    pub exposure_started_at: DateTime<Utc>,
}

/// A solve the trigger wants run.
#[derive(Debug, Clone, PartialEq)]
pub struct SolveJob {
    pub reason: SolveReason,
    pub frame_id: String,
    /// Where to look: the slew target (J2000 when it was commanded so, else
    /// the native target). `None` for an Alt/Az slew, after a failed hinted
    /// attempt, and for the other rules: the caller hints with the mount's
    /// reported position if it has one, otherwise solves blind.
    pub hint: Option<EqCoord>,
    /// What raised the rule, e.g. `star_lost`.
    pub cause: Option<String>,
    /// The H23 decision this job executes.
    pub decision_id: Uuid,
    generation: u64,
}

/// A rule waiting for a frame. Serialised with `hint` and, for clients of
/// the earlier `GET platesolve/status`, the same position as the
/// deprecated `target {ra, dec}` (RA hours, Dec degrees).
#[derive(Debug, Clone, PartialEq)]
pub struct PendingRule {
    pub reason: SolveReason,
    pub hint: Option<EqCoord>,
    /// A slew was commanded and has not yet settled. No frame qualifies until
    /// it has.
    pub awaiting_arrival: bool,
    /// Frames must begin exposing at or after this: the settle time after a
    /// slew, or the time of the event that raised the rule. `None` for
    /// startup, which has no motion to rule out — any light frame archived
    /// after the daemon starts will do, whatever its clock says.
    pub not_before: Option<DateTime<Utc>>,
    /// The event that raised the rule.
    pub cause: Option<String>,
    /// The slew command this rule follows.
    pub command_id: Option<CommandId>,
    pub raised_at: Option<DateTime<Utc>>,
}

impl Serialize for PendingRule {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let target = self.hint.map(|hint| crate::hardware::mount::EqCoord {
            ra: hint.ra_h,
            dec: hint.dec_deg,
        });
        let mut rule = serializer.serialize_struct("PendingRule", 8)?;
        rule.serialize_field("reason", &self.reason)?;
        rule.serialize_field("hint", &self.hint)?;
        rule.serialize_field("target", &target)?;
        rule.serialize_field("awaiting_arrival", &self.awaiting_arrival)?;
        rule.serialize_field("not_before", &self.not_before)?;
        rule.serialize_field("cause", &self.cause)?;
        rule.serialize_field("command_id", &self.command_id)?;
        rule.serialize_field("raised_at", &self.raised_at)?;
        rule.end()
    }
}

/// A rule that ended without a solve, for the status route.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DroppedRule {
    pub reason: SolveReason,
    pub why: String,
    pub at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TriggerStatus {
    pub enabled: bool,
    pub pending: Option<PendingRule>,
    pub in_flight: bool,
    pub retry_after_s: u64,
    /// The last bus event the trigger has seen (`equipment_events.seq`).
    pub last_event_seq: u64,
    pub last_solved_at: Option<DateTime<Utc>>,
    pub last_dropped: Option<DroppedRule>,
}

/// What the trigger decided about one archived frame.
#[derive(Debug, Clone, PartialEq)]
pub enum FrameDecision {
    /// Solve it.
    Solve(SolveJob),
    /// A rule is pending and the frame is a candidate, but not this one.
    Skip {
        rule: PendingRule,
        why: &'static str,
        decision_id: Uuid,
    },
    /// Nothing to decide: disabled, no rule pending, a solve in flight, or a
    /// frame type that is never solved.
    Ignore,
}

/// The M-3 readings at a decision (v6 §8.3.3), each with its age when it
/// comes from a device. Missing readings stay missing.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct TriggerReadings {
    pub ha_h: Option<f64>,
    pub alt_deg: Option<f64>,
    pub airmass: Option<f64>,
    pub pier_side: Option<PierSide>,
    pub temperature_c: Option<f64>,
    pub guide_rms_arcsec: Option<f64>,
    pub star_count: Option<u32>,
    pub cloud_probability: Option<f64>,
    #[serde(skip)]
    pub measurement_ages_ms: BTreeMap<String, u64>,
}

/// One kind of rule change, for logging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleChange {
    Armed(SolveReason),
    Raised(SolveReason),
    Relabelled(SolveReason),
    Dropped(SolveReason),
    Satisfied,
}

pub struct PlateSolveTrigger {
    enabled: bool,
    pending: Option<PendingRule>,
    /// Bumped by every new rule, so a late settle report or a finished solve
    /// for an older rule cannot act on a newer one.
    generation: u64,
    in_flight: bool,
    last_attempt: Option<DateTime<Utc>>,
    retry_after: Duration,
    last_event_seq: u64,
    last_safety: Option<SafetyGate>,
    last_solved_at: Option<DateTime<Utc>>,
    last_solved_position: Option<EqCoord>,
    last_solved_temperature_c: Option<f64>,
    last_dropped: Option<DroppedRule>,
    /// The bus this trigger is subscribed to (see `auto_solve::attach`).
    bus: Option<std::sync::Weak<crate::equipment::hub::EquipmentHub>>,
}

impl PlateSolveTrigger {
    /// A trigger with the startup rule pending.
    pub fn new(retry_after: Duration) -> Self {
        Self {
            enabled: true,
            pending: Some(PendingRule {
                reason: SolveReason::Startup,
                hint: None,
                awaiting_arrival: false,
                not_before: None,
                cause: Some("daemon_start".into()),
                command_id: None,
                raised_at: None,
            }),
            generation: 0,
            in_flight: false,
            last_attempt: None,
            retry_after,
            last_event_seq: 0,
            last_safety: None,
            last_solved_at: None,
            last_solved_position: None,
            last_solved_temperature_c: None,
            last_dropped: None,
            bus: None,
        }
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::new(Duration::ZERO)
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn status(&self) -> TriggerStatus {
        TriggerStatus {
            enabled: self.enabled,
            pending: self.enabled.then(|| self.pending.clone()).flatten(),
            in_flight: self.in_flight,
            retry_after_s: self.retry_after.as_secs(),
            last_event_seq: self.last_event_seq,
            last_solved_at: self.last_solved_at,
            last_dropped: self.last_dropped.clone(),
        }
    }

    /// Whether this trigger is fed by `hub`'s bus.
    pub fn attached_to(&self, hub: &std::sync::Arc<crate::equipment::hub::EquipmentHub>) -> bool {
        self.bus
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .is_some_and(|attached| std::sync::Arc::ptr_eq(&attached, hub))
    }

    /// Record that `hub`'s bus now feeds this trigger, from `next_seq` on.
    pub fn mark_attached(
        &mut self,
        hub: &std::sync::Arc<crate::equipment::hub::EquipmentHub>,
        next_seq: u64,
    ) {
        self.bus = Some(std::sync::Arc::downgrade(hub));
        self.last_event_seq = self.last_event_seq.max(next_seq.saturating_sub(1));
    }

    fn set(&mut self, rule: PendingRule) {
        self.generation += 1;
        self.pending = Some(rule);
        self.last_attempt = None;
    }

    fn drop_rule(&mut self, why: &str, at: DateTime<Utc>) -> Option<RuleChange> {
        let rule = self.pending.take()?;
        self.last_dropped = Some(DroppedRule {
            reason: rule.reason,
            why: why.to_string(),
            at,
        });
        Some(RuleChange::Dropped(rule.reason))
    }

    /// Raise a rule unless a solve is already pending: a pending rule (or a
    /// slew still on its way) will produce its own solve.
    fn raise(&mut self, reason: SolveReason, cause: &str, at: DateTime<Utc>) -> Option<RuleChange> {
        if self.pending.is_some() {
            return None;
        }
        self.set(PendingRule {
            reason,
            hint: None,
            awaiting_arrival: false,
            not_before: Some(at),
            cause: Some(cause.to_string()),
            command_id: None,
            raised_at: Some(at),
        });
        Some(RuleChange::Raised(reason))
    }

    /// The vision loop saw the sky clear (Recovery).
    pub fn cloud_cleared(&mut self, at: DateTime<Utc>) -> Option<RuleChange> {
        if !self.enabled {
            return None;
        }
        self.raise(SolveReason::Recovery, "cloud_cleared", at)
    }

    /// Feed one bus event. Events are applied once, in sequence order.
    pub fn observe(&mut self, event: &EquipmentEvent) -> Option<RuleChange> {
        if event.seq != 0 {
            if event.seq <= self.last_event_seq {
                return None;
            }
            self.last_event_seq = event.seq;
        }
        if !self.enabled {
            return None;
        }
        let at = event.at_utc;
        let slew_rule = |rule: &PendingRule| {
            matches!(rule.reason, SolveReason::PostSlew | SolveReason::PostFlip)
                && rule.awaiting_arrival
        };
        let same_command = |rule: &PendingRule| {
            rule.command_id.is_none()
                || event.command_id.is_none()
                || rule.command_id == event.command_id
        };
        match &event.event {
            EventKind::SlewCommanded {
                target,
                native_target,
                altaz,
            } => {
                let hint = if altaz.is_some() {
                    // Informative only, and (0 h, 0°) without a site or LST.
                    None
                } else if target.frame == EqFrame::J2000 {
                    Some(*target)
                } else {
                    native_target.or_else(|| (target.frame != EqFrame::Unknown).then_some(*target))
                };
                self.set(PendingRule {
                    reason: SolveReason::PostSlew,
                    hint,
                    awaiting_arrival: true,
                    not_before: None,
                    cause: Some("slew_commanded".into()),
                    command_id: event.command_id,
                    raised_at: Some(at),
                });
                Some(RuleChange::Armed(SolveReason::PostSlew))
            }
            EventKind::SlewSettled { .. } => match self.pending.as_mut() {
                Some(rule) if slew_rule(rule) && same_command(rule) => {
                    rule.awaiting_arrival = false;
                    rule.not_before = Some(at);
                    Some(RuleChange::Armed(rule.reason))
                }
                _ => None,
            },
            EventKind::SlewEnded { outcome, .. } => match self.pending.as_mut() {
                Some(rule) if slew_rule(rule) && same_command(rule) => {
                    if matches!(outcome, OpOutcome::Succeeded) {
                        // Settled is always emitted before a successful end;
                        // this only covers an end that arrived alone.
                        rule.awaiting_arrival = false;
                        rule.not_before = Some(at);
                        Some(RuleChange::Armed(rule.reason))
                    } else {
                        let why = format!("slew ended {} before settling", outcome.label());
                        self.drop_rule(&why, at)
                    }
                }
                _ => None,
            },
            EventKind::PierSideChanged { from, to }
                if *from != PierSide::Unknown && *to != PierSide::Unknown && from != to =>
            {
                match self.pending.as_mut() {
                    Some(rule) if slew_rule(rule) => {
                        rule.reason = SolveReason::PostFlip;
                        rule.cause = Some("pier_side_changed".into());
                        Some(RuleChange::Relabelled(SolveReason::PostFlip))
                    }
                    _ => {
                        self.set(PendingRule {
                            reason: SolveReason::PostFlip,
                            hint: None,
                            awaiting_arrival: false,
                            not_before: Some(at),
                            cause: Some("pier_side_changed".into()),
                            command_id: event.command_id,
                            raised_at: Some(at),
                        });
                        Some(RuleChange::Raised(SolveReason::PostFlip))
                    }
                }
            }
            EventKind::MeridianFlip {
                phase: FlipPhase::Done,
            } => {
                let hint = self.pending.as_ref().and_then(|rule| rule.hint);
                self.set(PendingRule {
                    reason: SolveReason::PostFlip,
                    hint,
                    awaiting_arrival: false,
                    not_before: Some(at),
                    cause: Some("meridian_flip_done".into()),
                    command_id: event.command_id,
                    raised_at: Some(at),
                });
                Some(RuleChange::Raised(SolveReason::PostFlip))
            }
            EventKind::DeviceReconnected { .. }
                if matches!(
                    event.device_kind,
                    Some(DeviceKind::Mount | DeviceKind::Guider)
                ) =>
            {
                self.raise(SolveReason::Recovery, "device_reconnected", at)
            }
            EventKind::DeviceLost { .. } if event.device_kind == Some(DeviceKind::Mount) => {
                match &self.pending {
                    Some(rule) if slew_rule(rule) => self.drop_rule("the mount was lost", at),
                    _ => None,
                }
            }
            EventKind::StarLost { .. } => self.raise(SolveReason::Recovery, "star_lost", at),
            EventKind::LockPositionLost => {
                self.raise(SolveReason::Recovery, "lock_position_lost", at)
            }
            EventKind::SafetyChanged { state, .. } => {
                let previous = self.last_safety.replace(*state);
                let resumed = *state == SafetyGate::Safe
                    && matches!(previous, Some(SafetyGate::Unsafe | SafetyGate::Unknown));
                if resumed {
                    self.raise(SolveReason::Recovery, "safety_resumed", at)
                } else {
                    None
                }
            }
            EventKind::Centered { .. } => match &self.pending {
                Some(rule) if !rule.awaiting_arrival && rule.reason != SolveReason::Startup => {
                    self.pending = None;
                    self.generation += 1;
                    Some(RuleChange::Satisfied)
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// A frame was archived at `now`. Says whether it should be solved.
    pub fn frame_archived(&mut self, frame: ArchivedFrame, now: DateTime<Utc>) -> FrameDecision {
        if !self.enabled || self.in_flight {
            return FrameDecision::Ignore;
        }
        let Some(rule) = self.pending.clone() else {
            return FrameDecision::Ignore;
        };
        // Darks, flats and bias frames have no stars; autofocus frames are
        // defocused on purpose; center frames are solved by the center loop.
        // Only sky frames meant to be sharp qualify.
        if !matches!(frame.capture_type, "light" | "verification") {
            return FrameDecision::Ignore;
        }
        let skip = |why| FrameDecision::Skip {
            rule: rule.clone(),
            why,
            decision_id: Uuid::new_v4(),
        };
        if rule.awaiting_arrival {
            return skip("awaiting_slew_settled");
        }
        if rule
            .not_before
            .is_some_and(|not_before| frame.exposure_started_at < not_before)
        {
            return skip("exposure_began_before_rule");
        }
        if let Some(last) = self.last_attempt {
            let Ok(retry_after) = chrono::Duration::from_std(self.retry_after) else {
                return skip("retry_cooldown");
            };
            if now < last + retry_after {
                return skip("retry_cooldown");
            }
        }
        self.in_flight = true;
        self.last_attempt = Some(now);
        FrameDecision::Solve(SolveJob {
            reason: rule.reason,
            frame_id: frame.frame_id.to_string(),
            hint: rule.hint,
            cause: rule.cause.clone(),
            decision_id: Uuid::new_v4(),
            generation: self.generation,
        })
    }

    /// A job finished. Success satisfies the rule it served, unless a newer
    /// rule has replaced it in the meantime.
    pub fn job_finished(
        &mut self,
        job: &SolveJob,
        solved: Option<EqCoord>,
        temperature_c: Option<f64>,
        at: DateTime<Utc>,
    ) {
        self.in_flight = false;
        if let Some(position) = solved {
            self.last_solved_at = Some(at);
            self.last_solved_position = Some(position);
            self.last_solved_temperature_c = temperature_c.or(self.last_solved_temperature_c);
            if job.generation == self.generation {
                self.pending = None;
            }
        } else if job.hint.is_some() && job.generation == self.generation {
            // The hint may be what failed: the next attempt uses the mount's
            // report or solves blind.
            if let Some(rule) = self.pending.as_mut() {
                rule.hint = None;
            }
        }
    }

    /// The H23 record of one decision (see the module docs); `None` for
    /// [`FrameDecision::Ignore`].
    pub fn decision_record(
        &self,
        frame: &ArchivedFrame,
        decision: &FrameDecision,
        readings: &TriggerReadings,
        now: DateTime<Utc>,
    ) -> Option<PolicyDecisionRecord> {
        let (decision_id, rule, chosen, why, hint) = match decision {
            FrameDecision::Solve(job) => (
                job.decision_id,
                self.pending.as_ref(),
                "solve",
                None,
                job.hint,
            ),
            FrameDecision::Skip {
                rule,
                why,
                decision_id,
            } => (*decision_id, Some(rule), "skip", Some(*why), rule.hint),
            FrameDecision::Ignore => return None,
        };
        let minutes_since_last_solve = self
            .last_solved_at
            .map(|at| (now - at).num_milliseconds() as f64 / 60_000.0);
        let temperature_delta_c = readings
            .temperature_c
            .zip(self.last_solved_temperature_c)
            .map(|(now, then)| now - then);
        let slew_distance_deg = hint
            .zip(self.last_solved_position)
            .and_then(|(hint, last)| {
                let hint = EqCoord {
                    frame: last.frame,
                    ..hint
                };
                hint.separation_arcmin(&last).map(|arcmin| arcmin / 60.0)
            });
        let safe_arms = match decision {
            FrameDecision::Solve(_) => vec!["solve".to_string(), "skip".to_string()],
            _ => vec!["skip".to_string()],
        };
        Some(PolicyDecisionRecord {
            decision_id,
            policy: POLICY.to_string(),
            policy_version: POLICY_VERSION.to_string(),
            mode: AutonomyMode::Bounded,
            context: json!({
                "frame_id": frame.frame_id,
                "capture_type": frame.capture_type,
                "exposure_started_at": frame.exposure_started_at,
                "rule": rule,
                "skip_reason": why,
                "minutes_since_last_solve": minutes_since_last_solve,
                "temperature_delta_c": temperature_delta_c,
                "slew_distance_deg": slew_distance_deg,
                "readings": readings,
                "retry_after_s": self.retry_after.as_secs(),
            }),
            measurement_ages_ms: readings.measurement_ages_ms.clone(),
            safe_arms,
            chosen_arm: Some(chosen.to_string()),
            propensities: None,
            requested_action: Some(json!({"arm": chosen, "frame_id": frame.frame_id})),
            validated_action: None,
            executed_action: None,
            command_id: None,
            reward_components: None,
            attribution_window_start: None,
            attribution_window_end: None,
            interrupted: false,
            override_applied: false,
            parent_frame_ids: vec![frame.frame_id.to_string()],
            created_at: now,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::events::EquipmentEvent;
    use crate::equipment::types::{DeviceError, FailureClass};

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
        ra_h: 8.855,
        dec_deg: 11.8,
        frame: EqFrame::J2000,
    };
    const M42: EqCoord = EqCoord {
        ra_h: 5.588,
        dec_deg: -5.39,
        frame: EqFrame::J2000,
    };

    /// A sequenced mount event at `seconds`.
    struct Bus {
        seq: u64,
    }

    impl Bus {
        fn new() -> Self {
            Self { seq: 0 }
        }

        fn event(
            &mut self,
            kind: Option<DeviceKind>,
            seconds: i64,
            command: Option<CommandId>,
            event: EventKind,
        ) -> EquipmentEvent {
            self.seq += 1;
            EquipmentEvent {
                seq: self.seq,
                event_id: Uuid::new_v4(),
                at_utc: t(seconds),
                mono_ns: 0,
                device_kind: kind,
                connection_id: None,
                command_id: command,
                device_time: None,
                event,
            }
        }

        fn mount(
            &mut self,
            seconds: i64,
            command: Option<CommandId>,
            event: EventKind,
        ) -> EquipmentEvent {
            self.event(Some(DeviceKind::Mount), seconds, command, event)
        }
    }

    fn commanded(target: EqCoord) -> EventKind {
        EventKind::SlewCommanded {
            target,
            native_target: Some(EqCoord::new(
                target.ra_h + 0.02,
                target.dec_deg,
                EqFrame::JNowApparent,
            )),
            altaz: None,
        }
    }

    fn settled() -> EventKind {
        EventKind::SlewSettled {
            position: None,
            error_arcmin: Some(0.2),
        }
    }

    fn solve(decision: FrameDecision) -> SolveJob {
        match decision {
            FrameDecision::Solve(job) => job,
            other => panic!("expected a solve, got {other:?}"),
        }
    }

    fn skip_reason(decision: FrameDecision) -> &'static str {
        match decision {
            FrameDecision::Skip { why, .. } => why,
            other => panic!("expected a skip, got {other:?}"),
        }
    }

    #[test]
    fn the_first_light_frame_after_startup_is_solved_once_it_succeeds() {
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let dark = ArchivedFrame {
            capture_type: "dark",
            ..light("d", 5)
        };
        assert_eq!(trigger.frame_archived(dark, t(10)), FrameDecision::Ignore);

        let job = solve(trigger.frame_archived(light("a", 20), t(30)));
        assert_eq!(
            (job.reason, job.frame_id.as_str(), job.hint),
            (SolveReason::Startup, "a", None)
        );
        assert_eq!(
            trigger.frame_archived(light("b", 31), t(40)),
            FrameDecision::Ignore,
            "one solve at a time"
        );

        trigger.job_finished(&job, Some(M42), None, t(45));
        assert_eq!(
            trigger.frame_archived(light("c", 50), t(60)),
            FrameDecision::Ignore,
            "rule satisfied"
        );
        assert!(trigger.status().pending.is_none());
        assert_eq!(trigger.status().last_solved_at, Some(t(45)));
    }

    #[test]
    fn a_failed_solve_retries_only_after_the_cooldown() {
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let job = solve(trigger.frame_archived(light("a", 10), t(20)));
        trigger.job_finished(&job, None, None, t(25));

        assert_eq!(
            skip_reason(trigger.frame_archived(light("b", 30), t(60))),
            "retry_cooldown"
        );
        let retry = solve(trigger.frame_archived(light("c", 130), t(140)));
        assert_eq!(retry.frame_id, "c");
    }

    #[test]
    fn post_slew_fires_from_slew_settled_and_only_for_frames_begun_after_it() {
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let mut bus = Bus::new();
        let slew = Some(CommandId::new());
        assert_eq!(
            trigger.observe(&bus.mount(40, slew, commanded(M67))),
            Some(RuleChange::Armed(SolveReason::PostSlew))
        );
        trigger.observe(&bus.mount(41, slew, EventKind::SlewStarted));

        // Not settled: nothing qualifies, however late the frame.
        assert_eq!(
            skip_reason(trigger.frame_archived(light("moving", 50), t(60))),
            "awaiting_slew_settled"
        );
        // Arrival alone is not a settle.
        trigger.observe(&bus.mount(88, slew, EventKind::SlewArrived { position: None }));
        assert!(trigger.status().pending.unwrap().awaiting_arrival);

        trigger.observe(&bus.mount(90, slew, settled()));
        let pending = trigger.status().pending.unwrap();
        assert_eq!(
            (pending.awaiting_arrival, pending.not_before),
            (false, Some(t(90)))
        );
        assert_eq!(
            skip_reason(trigger.frame_archived(light("early", 85), t(100))),
            "exposure_began_before_rule",
            "began exposing before the settle: trailed stars"
        );
        let job = solve(trigger.frame_archived(light("late", 95), t(110)));
        assert_eq!((job.reason, job.hint), (SolveReason::PostSlew, Some(M67)));
        // An event replayed after a lag is not applied twice.
        let replay = EquipmentEvent {
            seq: 2,
            ..bus.mount(41, slew, EventKind::SlewStarted)
        };
        assert_eq!(trigger.observe(&replay), None);
    }

    /// `GET platesolve/status`: a pending rule carries `hint` and, for the
    /// clients of the earlier contract, the same position as `target {ra,
    /// dec}`; no position is null in both.
    #[test]
    fn a_pending_rule_keeps_the_earlier_target_key_beside_its_hint() {
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let mut bus = Bus::new();
        trigger.observe(&bus.mount(40, Some(CommandId::new()), commanded(M67)));
        let status = serde_json::to_value(trigger.status()).unwrap();
        let pending = &status["pending"];
        assert_eq!(pending["reason"], json!("post_slew"), "{status}");
        assert_eq!(pending["hint"]["ra_h"], json!(M67.ra_h), "{status}");
        assert_eq!(pending["hint"]["dec_deg"], json!(M67.dec_deg), "{status}");
        assert_eq!(
            pending["target"],
            json!({"ra": M67.ra_h, "dec": M67.dec_deg}),
            "{status}"
        );
        assert_eq!(pending["awaiting_arrival"], json!(true), "{status}");
        for key in ["not_before", "cause", "command_id", "raised_at"] {
            assert!(pending.get(key).is_some(), "{key}: {status}");
        }
        // Startup has no position.
        let startup = serde_json::to_value(
            PlateSolveTrigger::new(Duration::from_secs(120))
                .status()
                .pending
                .unwrap(),
        )
        .unwrap();
        assert!(startup["hint"].is_null() && startup["target"].is_null());
    }

    #[test]
    fn an_alt_az_slew_gives_no_hint_and_a_failed_hint_is_dropped() {
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let mut bus = Bus::new();
        let slew = Some(CommandId::new());
        // Site or LST unknown: the event's RA/Dec is the default (0 h, 0°).
        let altaz = EventKind::SlewCommanded {
            target: EqCoord::default(),
            native_target: None,
            altaz: Some(crate::equipment::types::AltAz {
                alt_deg: 45.0,
                az_deg: 200.0,
            }),
        };
        trigger.observe(&bus.mount(10, slew, altaz));
        assert_eq!(trigger.status().pending.unwrap().hint, None);
        trigger.observe(&bus.mount(20, slew, settled()));
        let job = solve(trigger.frame_archived(light("altaz", 25), t(30)));
        assert_eq!((job.reason, job.hint), (SolveReason::PostSlew, None));
        trigger.job_finished(&job, None, None, t(35));

        // A hinted post-slew solve that fails is retried without the hint.
        let second = Some(CommandId::new());
        trigger.observe(&bus.mount(200, second, commanded(M67)));
        trigger.observe(&bus.mount(210, second, settled()));
        let job = solve(trigger.frame_archived(light("m67", 215), t(220)));
        assert_eq!(job.hint, Some(M67));
        trigger.job_finished(&job, None, None, t(225));
        let pending = trigger.status().pending.unwrap();
        assert_eq!(
            (pending.reason, pending.hint),
            (SolveReason::PostSlew, None)
        );
        let retry = solve(trigger.frame_archived(light("m67b", 400), t(400)));
        assert_eq!(retry.hint, None);

        // An equatorial target in an unknown frame with no native target is
        // no hint either.
        let unknown = EventKind::SlewCommanded {
            target: EqCoord::new(8.0, 20.0, EqFrame::Unknown),
            native_target: None,
            altaz: None,
        };
        trigger.observe(&bus.mount(500, None, unknown));
        assert_eq!(trigger.status().pending.unwrap().hint, None);
    }

    #[test]
    fn a_native_frame_slew_hints_with_the_native_target() {
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let mut bus = Bus::new();
        let native = EqCoord::new(8.9, 11.8, EqFrame::Unknown);
        trigger.observe(&bus.mount(1, None, commanded(native)));
        let hint = trigger.status().pending.unwrap().hint.unwrap();
        assert_eq!(hint.frame, EqFrame::JNowApparent);
    }

    #[test]
    fn a_settle_for_another_command_is_ignored_and_a_newer_slew_supersedes() {
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let mut bus = Bus::new();
        let (first, second) = (Some(CommandId::new()), Some(CommandId::new()));
        trigger.observe(&bus.mount(10, first, commanded(M67)));
        trigger.observe(&bus.mount(20, first, settled()));
        let job = solve(trigger.frame_archived(light("m67", 30), t(40)));

        trigger.observe(&bus.mount(50, second, commanded(M42)));
        // A late settle of the first slew, and a late success for M67, do
        // not arm or satisfy the M42 rule.
        assert_eq!(trigger.observe(&bus.mount(60, first, settled())), None);
        trigger.job_finished(&job, Some(M67), None, t(61));
        let pending = trigger.status().pending.unwrap();
        assert_eq!((pending.hint, pending.awaiting_arrival), (Some(M42), true));

        trigger.observe(&bus.mount(120, second, settled()));
        let job = solve(trigger.frame_archived(light("m42", 125), t(130)));
        assert_eq!(job.hint, Some(M42));
    }

    #[test]
    fn a_slew_that_ends_without_settling_leaves_nothing_to_solve() {
        for outcome in [
            OpOutcome::Aborted { by: None },
            OpOutcome::TimedOut,
            OpOutcome::Unknown(DeviceError::outcome_unknown("no Slewing member")),
        ] {
            let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
            let mut bus = Bus::new();
            trigger.observe(&bus.mount(10, None, commanded(M67)));
            assert!(matches!(
                trigger.observe(&bus.mount(
                    20,
                    None,
                    EventKind::SlewEnded {
                        outcome: outcome.clone(),
                        duration_ms: 1
                    }
                )),
                Some(RuleChange::Dropped(SolveReason::PostSlew))
            ));
            assert!(trigger.status().pending.is_none());
            assert!(trigger.status().last_dropped.is_some());
            assert_eq!(
                trigger.frame_archived(light("a", 900), t(910)),
                FrameDecision::Ignore
            );
        }
        // Losing the mount mid-slew drops it too.
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let mut bus = Bus::new();
        trigger.observe(&bus.mount(10, None, commanded(M67)));
        trigger.observe(&bus.mount(
            20,
            None,
            EventKind::DeviceLost {
                class: FailureClass::Timeout,
                reason: None,
            },
        ));
        assert!(trigger.status().pending.is_none());
    }

    #[test]
    fn a_pier_side_change_marks_the_slew_post_flip_and_a_flip_raises_post_flip() {
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let mut bus = Bus::new();
        trigger.observe(&bus.mount(10, None, commanded(M67)));
        trigger.observe(&bus.mount(
            15,
            None,
            EventKind::PierSideChanged {
                from: PierSide::East,
                to: PierSide::West,
            },
        ));
        let pending = trigger.status().pending.unwrap();
        assert_eq!(
            (pending.reason, pending.awaiting_arrival),
            (SolveReason::PostFlip, true)
        );
        trigger.observe(&bus.mount(30, None, settled()));
        let job = solve(trigger.frame_archived(light("after", 31), t(40)));
        assert_eq!(job.reason, SolveReason::PostFlip);
        trigger.job_finished(&job, Some(M67), None, t(41));

        // An unknown side is not a flip.
        assert_eq!(
            trigger.observe(&bus.mount(
                50,
                None,
                EventKind::PierSideChanged {
                    from: PierSide::Unknown,
                    to: PierSide::West,
                },
            )),
            None
        );
        trigger.observe(&bus.mount(
            60,
            None,
            EventKind::MeridianFlip {
                phase: FlipPhase::Done,
            },
        ));
        let pending = trigger.status().pending.unwrap();
        assert_eq!(
            (pending.reason, pending.not_before),
            (SolveReason::PostFlip, Some(t(60)))
        );
    }

    #[test]
    fn recovery_events_raise_a_solve_when_none_is_pending() {
        let mut bus = Bus::new();
        let cases = [
            (
                bus.event(
                    Some(DeviceKind::Mount),
                    10,
                    None,
                    EventKind::DeviceReconnected { reason: None },
                ),
                "device_reconnected",
            ),
            (
                bus.event(
                    Some(DeviceKind::Guider),
                    10,
                    None,
                    EventKind::StarLost {
                        frame: None,
                        star_mass: None,
                        snr: None,
                        avg_dist: None,
                        error_code: None,
                        status: None,
                    },
                ),
                "star_lost",
            ),
            (
                bus.event(
                    Some(DeviceKind::Guider),
                    10,
                    None,
                    EventKind::LockPositionLost,
                ),
                "lock_position_lost",
            ),
        ];
        for (event, cause) in cases {
            let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
            let job = solve(trigger.frame_archived(light("start", 1), t(2)));
            trigger.job_finished(&job, Some(M42), None, t(3));
            assert_eq!(
                trigger.observe(&event),
                Some(RuleChange::Raised(SolveReason::Recovery)),
                "{cause}"
            );
            let pending = trigger.status().pending.unwrap();
            assert_eq!(pending.cause.as_deref(), Some(cause));
            let job = solve(trigger.frame_archived(light("next", 11), t(20)));
            assert_eq!(job.reason, SolveReason::Recovery);
        }
        // A focuser reconnect does not move the pointing.
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let job = solve(trigger.frame_archived(light("start", 1), t(2)));
        trigger.job_finished(&job, Some(M42), None, t(3));
        let focuser = bus.event(
            Some(DeviceKind::Focuser),
            10,
            None,
            EventKind::DeviceReconnected { reason: None },
        );
        assert_eq!(trigger.observe(&focuser), None);
        // The cloud-cleared signal from the vision loop.
        assert_eq!(
            trigger.cloud_cleared(t(30)),
            Some(RuleChange::Raised(SolveReason::Recovery))
        );
    }

    #[test]
    fn safety_returning_to_safe_is_a_recovery_and_a_steady_safe_is_not() {
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let job = solve(trigger.frame_archived(light("start", 1), t(2)));
        trigger.job_finished(&job, Some(M42), None, t(3));
        let mut bus = Bus::new();
        let safety = |bus: &mut Bus, state, s| {
            bus.event(
                Some(DeviceKind::SafetyMonitor),
                s,
                None,
                EventKind::SafetyChanged {
                    state,
                    sources: Vec::new(),
                },
            )
        };
        assert_eq!(
            trigger.observe(&safety(&mut bus, SafetyGate::Safe, 5)),
            None
        );
        assert_eq!(
            trigger.observe(&safety(&mut bus, SafetyGate::Unsafe, 6)),
            None
        );
        assert_eq!(
            trigger.observe(&safety(&mut bus, SafetyGate::Safe, 7)),
            Some(RuleChange::Raised(SolveReason::Recovery))
        );
    }

    #[test]
    fn a_centered_target_satisfies_the_pending_rule() {
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let job = solve(trigger.frame_archived(light("start", 1), t(2)));
        trigger.job_finished(&job, Some(M42), None, t(3));
        let mut bus = Bus::new();
        let center = Some(CommandId::new());
        trigger.observe(&bus.mount(10, center, commanded(M67)));
        trigger.observe(&bus.mount(20, center, settled()));
        assert_eq!(
            trigger.observe(&bus.mount(
                40,
                None,
                EventKind::Centered {
                    error_arcmin: 0.4,
                    attempts: 2
                }
            )),
            Some(RuleChange::Satisfied)
        );
        assert!(trigger.status().pending.is_none());
        // Center frames are the center loop's to solve.
        trigger.observe(&bus.mount(50, None, commanded(M42)));
        trigger.observe(&bus.mount(60, None, settled()));
        let center_frame = ArchivedFrame {
            capture_type: "center",
            ..light("c", 61)
        };
        assert_eq!(
            trigger.frame_archived(center_frame, t(62)),
            FrameDecision::Ignore
        );
    }

    #[test]
    fn a_wrong_camera_clock_passes_startup_but_never_a_slew() {
        // The OLS simulator stamps frames from its playback clock: 1970.
        let epoch = ArchivedFrame {
            exposure_started_at: DateTime::from_timestamp(0, 0).unwrap(),
            ..light("sim", 0)
        };
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let job = solve(trigger.frame_archived(epoch, t(10)));
        assert_eq!(job.reason, SolveReason::Startup);
        trigger.job_finished(&job, Some(M42), None, t(11));

        let mut bus = Bus::new();
        trigger.observe(&bus.mount(20, None, commanded(M67)));
        trigger.observe(&bus.mount(30, None, settled()));
        assert_eq!(
            skip_reason(trigger.frame_archived(epoch, t(40))),
            "exposure_began_before_rule"
        );
    }

    #[test]
    fn a_disabled_trigger_never_asks() {
        let mut trigger = PlateSolveTrigger::disabled();
        assert_eq!(
            trigger.frame_archived(light("a", 3600), Utc::now()),
            FrameDecision::Ignore
        );
        let mut bus = Bus::new();
        assert_eq!(trigger.observe(&bus.mount(1, None, commanded(M67))), None);
        assert_eq!(trigger.cloud_cleared(t(2)), None);
        let status = trigger.status();
        assert!(!status.enabled && status.pending.is_none());
    }

    #[test]
    fn every_decision_about_a_candidate_frame_is_a_deterministic_h23_record() {
        let mut trigger = PlateSolveTrigger::new(Duration::from_secs(120));
        let job = solve(trigger.frame_archived(light("start", 1), t(2)));
        trigger.job_finished(&job, Some(M42), Some(4.0), t(3));
        let mut bus = Bus::new();
        trigger.observe(&bus.mount(10, None, commanded(M67)));
        let readings = TriggerReadings {
            ha_h: Some(-1.5),
            alt_deg: Some(55.0),
            temperature_c: Some(2.5),
            measurement_ages_ms: [("mount".to_string(), 120)].into_iter().collect(),
            ..TriggerReadings::default()
        };

        let frame = light("moving", 11);
        let decision = trigger.frame_archived(frame, t(12));
        let record = trigger
            .decision_record(&frame, &decision, &readings, t(12))
            .unwrap();
        assert_eq!(record.policy, POLICY);
        assert_eq!(record.chosen_arm.as_deref(), Some("skip"));
        assert_eq!(record.safe_arms, vec!["skip".to_string()]);
        assert_eq!(
            record.context["skip_reason"],
            json!("awaiting_slew_settled")
        );
        assert_eq!(record.propensities, None);
        assert_eq!(record.measurement_ages_ms["mount"], 120);

        trigger.observe(&bus.mount(20, None, settled()));
        let frame = light("ok", 21);
        let decision = trigger.frame_archived(frame, t(22));
        let record = trigger
            .decision_record(&frame, &decision, &readings, t(22))
            .unwrap();
        assert_eq!(record.chosen_arm.as_deref(), Some("solve"));
        assert_eq!(record.context["rule"]["reason"], json!("post_slew"));
        assert_eq!(record.context["temperature_delta_c"], json!(-1.5));
        let distance = record.context["slew_distance_deg"].as_f64().unwrap();
        assert!(distance > 40.0, "{distance}");
        assert!(record.context["minutes_since_last_solve"].as_f64().unwrap() > 0.0);
        assert_eq!(
            match decision {
                FrameDecision::Solve(job) => job.decision_id,
                _ => unreachable!(),
            },
            record.decision_id
        );
        assert_eq!(
            trigger.decision_record(&frame, &FrameDecision::Ignore, &readings, t(22)),
            None
        );
    }
}
