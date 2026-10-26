//! The safety hub (spec §3.10, §2.8 fail-safe rule). Owner: WP1A-8.
//!
//! **Combination (AND).** With N monitors the state is `Safe` only if every
//! monitor is `Ready` and its latest reading is fresh, no older than
//! [`HubSettings::stale_after`], and safe. A confirmed unsafe reading (fresh,
//! from a connected monitor) makes the state `Unsafe`; otherwise any monitor
//! that is not Ready (§2.8: any other state counts as not safe), has no fresh
//! reading, or reads stale makes it `Unknown`. With no monitor configured the
//! state is `None`: nothing is triggered, and learning that needs weather
//! corroboration stays frozen (v6 §8.1, §8.3.1).
//!
//! **Gating is fail-safe.** `Unknown` and `Unsafe` both refuse new automatic
//! actions ([`SafetyGateCheck`]); a read error, a disconnect and a stale
//! value are never "safe". While a `safety_*` resume block is active the
//! check keeps refusing them even after a safe reading, so resuming after
//! unsafe stays the operator's decision (Tier 1, §3.10).
//!
//! **Policy edges.** The hub reports a [`PolicyTrigger`] only for a
//! *confirmed* unsafe reading (a fresh `Unsafe`, or `Warning` when
//! `warning_is_unsafe`) or for a monitor that goes `Lost` (NINA's
//! "disconnected after having been connected"). A `Degraded` monitor, a
//! failed poll or a stale value only makes the state `Unknown`; it never
//! triggers the policy, so a Wi-Fi blip never parks the mount (§9.3 item 1).
//! The hub reports raw edges; the supervisor latches them so the policy runs
//! once per unsafe episode.

use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{is_safing_op, level_is_unsafe};
use crate::config::SafetyConfig;
use crate::equipment::commands::{GatedCommand, SafetyRejection};
use crate::equipment::events::{EquipmentEvent, EventKind};
use crate::equipment::handle::SafetyMonitorHandle;
use crate::equipment::snapshot::{
    EquipmentSnapshot, ResumeBlockKind, SafetyGate, SafetyLevel, SafetySource, SafetySummary,
};
use crate::equipment::traits::{Clock, SafetyCheck};
use crate::equipment::types::{LinkState, ObsState, Observed, Source};

/// A reading older than this is stale for gating: three missed 2 s polls.
pub const DEFAULT_STALE_AFTER: Duration = Duration::from_secs(6);

/// The gate check's name (`safety_rejected{check}`).
pub const SAFETY_CHECK_NAME: &str = "safety_monitor";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HubSettings {
    pub stale_after: Duration,
    /// `[safety] warning_is_unsafe`.
    pub warning_is_unsafe: bool,
}

impl Default for HubSettings {
    fn default() -> Self {
        Self {
            stale_after: DEFAULT_STALE_AFTER,
            warning_is_unsafe: true,
        }
    }
}

impl HubSettings {
    pub fn from_config(config: &SafetyConfig) -> Self {
        Self {
            warning_is_unsafe: config.warning_is_unsafe,
            ..Self::default()
        }
    }
}

/// What the hub knows about one monitor at one evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct MonitorInput {
    pub connection_id: Uuid,
    pub name: String,
    pub link: LinkState,
    pub level: Observed<SafetyLevel>,
    pub detail: Option<String>,
}

impl MonitorInput {
    /// Read a monitor's handle without touching the device. A retired
    /// handle is `Disconnected`; a snapshot published for another connection
    /// counts as no reading.
    pub fn from_handle(handle: &SafetyMonitorHandle) -> Self {
        let identity = handle.identity();
        let snapshot = handle.snapshot();
        let link = if handle.is_retired() {
            LinkState::Disconnected
        } else {
            handle.health().state
        };
        let (level, detail) = if snapshot.connection_id == identity.connection_id {
            (snapshot.level.clone(), snapshot.detail.clone())
        } else {
            (Observed::not_set(Source::Derived), None)
        };
        Self {
            connection_id: identity.connection_id,
            name: identity.name.clone(),
            link,
            level,
            detail,
        }
    }
}

/// One monitor's contribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MonitorVerdict {
    Safe,
    /// A confirmed unsafe reading.
    Unsafe,
    /// Not Ready, no fresh reading, or a stale one.
    Unknown,
}

/// The reading if it is usable now: fresh, timestamped and young enough.
pub fn current_level(
    settings: &HubSettings,
    input: &MonitorInput,
    now_mono_ns: u64,
) -> Option<SafetyLevel> {
    if input.level.state != ObsState::Fresh {
        return None;
    }
    let age_ms = input.level.age_ms(now_mono_ns)?;
    let limit_ms = u64::try_from(settings.stale_after.as_millis()).unwrap_or(u64::MAX);
    if age_ms > limit_ms {
        return None;
    }
    input.level.value
}

pub fn verdict(settings: &HubSettings, input: &MonitorInput, now_mono_ns: u64) -> MonitorVerdict {
    if !input.link.is_connected() {
        return MonitorVerdict::Unknown;
    }
    match current_level(settings, input, now_mono_ns) {
        // A fresh unsafe reading is confirmed even from a degraded link.
        Some(level) if level_is_unsafe(level, settings.warning_is_unsafe) => MonitorVerdict::Unsafe,
        // Safe needs a Ready monitor (§2.8).
        Some(_) if input.link == LinkState::Ready => MonitorVerdict::Safe,
        _ => MonitorVerdict::Unknown,
    }
}

/// Why the on-unsafe policy should run (§3.10).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "trigger", rename_all = "snake_case")]
pub enum PolicyTrigger {
    /// A fresh unsafe reading (or a warning under `warning_is_unsafe`).
    ConfirmedUnsafe {
        connection_id: Uuid,
        name: String,
        level: SafetyLevel,
        detail: Option<String>,
    },
    /// A configured monitor went `Lost` after having been connected.
    MonitorLost { connection_id: Uuid, name: String },
}

impl PolicyTrigger {
    pub fn connection_id(&self) -> Uuid {
        match self {
            PolicyTrigger::ConfirmedUnsafe { connection_id, .. }
            | PolicyTrigger::MonitorLost { connection_id, .. } => *connection_id,
        }
    }

    pub fn reason(&self) -> String {
        match self {
            PolicyTrigger::ConfirmedUnsafe {
                name,
                level,
                detail,
                ..
            } => {
                let level = match level {
                    SafetyLevel::Warning => "a warning (counted as unsafe)",
                    _ => "unsafe",
                };
                match detail {
                    Some(detail) => format!("safety monitor {name} reports {level}: {detail}"),
                    None => format!("safety monitor {name} reports {level}"),
                }
            }
            PolicyTrigger::MonitorLost { name, .. } => {
                format!("safety monitor {name} was lost after having been connected")
            }
        }
    }
}

/// The result of one evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct Evaluation {
    pub summary: SafetySummary,
    pub previous: SafetyGate,
    pub verdicts: Vec<(Uuid, MonitorVerdict)>,
    pub triggers: Vec<PolicyTrigger>,
    /// `safety_changed`, when the state or any source changed.
    pub event: Option<EquipmentEvent>,
    /// `Unknown` only because monitors have not delivered a first reading.
    pub startup: bool,
}

impl Evaluation {
    pub fn state_changed(&self) -> bool {
        self.summary.state != self.previous
    }
}

#[derive(Debug, Clone, Copy)]
struct Memo {
    confirmed_unsafe: bool,
    link: LinkState,
    /// A usable reading has arrived at least once.
    ever_read: bool,
}

/// Combines monitors and detects edges. Pure: no I/O, time from the clock.
#[derive(Debug, Clone)]
pub struct SafetyMonitorHub {
    settings: HubSettings,
    memos: HashMap<Uuid, Memo>,
    summary: SafetySummary,
}

impl SafetyMonitorHub {
    pub fn new(settings: HubSettings) -> Self {
        Self {
            settings,
            memos: HashMap::new(),
            summary: SafetySummary::default(),
        }
    }

    pub fn settings(&self) -> &HubSettings {
        &self.settings
    }

    pub fn summary(&self) -> &SafetySummary {
        &self.summary
    }

    pub fn evaluate(&mut self, inputs: &[MonitorInput], clock: &dyn Clock) -> Evaluation {
        let now_mono = clock.now_mono_ns();
        let mut triggers = Vec::new();
        let mut sources = Vec::with_capacity(inputs.len());
        let mut verdicts = Vec::with_capacity(inputs.len());
        let mut startup = true;
        for input in inputs {
            let verdict = verdict(&self.settings, input, now_mono);
            let memo = self.memos.entry(input.connection_id).or_insert(Memo {
                confirmed_unsafe: false,
                link: LinkState::Connecting,
                ever_read: false,
            });
            let confirmed = verdict == MonitorVerdict::Unsafe;
            if confirmed && !memo.confirmed_unsafe {
                triggers.push(PolicyTrigger::ConfirmedUnsafe {
                    connection_id: input.connection_id,
                    name: input.name.clone(),
                    level: input.level.value.unwrap_or(SafetyLevel::Unsafe),
                    detail: input.detail.clone(),
                });
            }
            if input.link == LinkState::Lost && memo.link != LinkState::Lost {
                triggers.push(PolicyTrigger::MonitorLost {
                    connection_id: input.connection_id,
                    name: input.name.clone(),
                });
            }
            if verdict != MonitorVerdict::Unknown {
                memo.ever_read = true;
            } else if memo.ever_read
                || matches!(input.link, LinkState::Lost | LinkState::Disconnected)
            {
                startup = false;
            }
            memo.confirmed_unsafe = confirmed;
            memo.link = input.link;
            sources.push(SafetySource {
                connection_id: input.connection_id,
                name: input.name.clone(),
                link: input.link,
                level: current_level(&self.settings, input, now_mono),
            });
            verdicts.push((input.connection_id, verdict));
        }
        self.memos
            .retain(|id, _| inputs.iter().any(|input| input.connection_id == *id));

        let state = if inputs.is_empty() {
            SafetyGate::None
        } else if verdicts.iter().any(|(_, v)| *v == MonitorVerdict::Unsafe) {
            SafetyGate::Unsafe
        } else if verdicts.iter().any(|(_, v)| *v == MonitorVerdict::Unknown) {
            SafetyGate::Unknown
        } else {
            SafetyGate::Safe
        };
        let previous = self.summary.state;
        let state_changed = state != previous;
        let sources_changed = sources != self.summary.sources;
        let summary = SafetySummary {
            state,
            sources,
            changed_at: if state_changed {
                Some(clock.now_utc())
            } else {
                self.summary.changed_at
            },
        };
        let event = (state_changed || sources_changed).then(|| {
            EquipmentEvent::new(
                EventKind::SafetyChanged {
                    state,
                    sources: summary.sources.clone(),
                },
                clock,
            )
        });
        self.summary = summary.clone();
        Evaluation {
            summary,
            previous,
            verdicts,
            triggers,
            event,
            startup: state == SafetyGate::Unknown && startup,
        }
    }
}

fn op_name(cmd: &GatedCommand) -> String {
    match serde_json::to_value(cmd.kind()) {
        Ok(serde_json::Value::String(name)) => name,
        _ => format!("{:?}", cmd.kind()),
    }
}

/// The gate plug-in for §2.8 and §3.10 "Gating": while safety is not `Safe`
/// or `None`, new automatic actions are refused (autofocus triggers,
/// dithers, flips, automatic captures and slews). Operator commands are not
/// refused here (the mount's own limits check handles them), and neither is
/// anything that stops or secures the rig ([`super::is_safing_op`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct SafetyGateCheck;

impl SafetyCheck for SafetyGateCheck {
    fn name(&self) -> &'static str {
        SAFETY_CHECK_NAME
    }

    fn applies_to(&self, cmd: &GatedCommand) -> bool {
        cmd.source.is_automatic() && !is_safing_op(&cmd.op)
    }

    fn check(&self, cmd: &GatedCommand, snap: &EquipmentSnapshot) -> Result<(), SafetyRejection> {
        let state = snap.safety.state;
        let safety_block = snap.resume_blocks.iter().any(|block| {
            matches!(
                block.kind,
                ResumeBlockKind::SafetyUnsafe | ResumeBlockKind::SafetyUnknown
            )
        });
        if state.allows_automatic() && !safety_block {
            return Ok(());
        }
        let why = match state {
            SafetyGate::Unsafe => "a safety monitor reports unsafe",
            SafetyGate::Unknown => {
                "a safety monitor is not ready or its reading is stale (fail-safe)"
            }
            _ => "a safety resume block is active; resuming after unsafe is manual",
        };
        Err(SafetyRejection {
            check: SAFETY_CHECK_NAME.into(),
            reason: format!("{why}; automatic {} is refused", op_name(cmd)),
            details: Some(serde_json::json!({
                "state": state,
                "sources": snap.safety.sources,
            })),
        })
    }

    /// An automatic command accepted with no monitor configured is audited
    /// `no_safety_monitor`.
    fn annotations(&self, _cmd: &GatedCommand, snap: &EquipmentSnapshot) -> Vec<String> {
        if snap.safety.state == SafetyGate::None {
            vec!["no_safety_monitor".into()]
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::{GatedOp, GuiderCmd, MountCmd, PixelUnit, Settle};
    use crate::equipment::snapshot::SafetySnapshot;
    use crate::equipment::testing::fake_handles::fake_safety_monitor;
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::{AutonomyMode, CommandSource, Priority};

    fn input(clock: &ManualClock, link: LinkState, level: Option<SafetyLevel>) -> MonitorInput {
        MonitorInput {
            connection_id: Uuid::new_v4(),
            name: "cloudwatcher".into(),
            link,
            level: match level {
                Some(level) => Observed::fresh_at(level, Source::Property, clock),
                None => Observed::error(Source::Property),
            },
            detail: None,
        }
    }

    fn reading(clock: &ManualClock, mut input: MonitorInput, level: SafetyLevel) -> MonitorInput {
        input.level = Observed::fresh_at(level, Source::Property, clock);
        input.link = LinkState::Ready;
        input
    }

    #[test]
    fn no_monitor_is_none_and_triggers_nothing() {
        let clock = ManualClock::new();
        let mut hub = SafetyMonitorHub::new(HubSettings::default());
        let eval = hub.evaluate(&[], &clock);
        assert_eq!(eval.summary.state, SafetyGate::None);
        assert!(eval.triggers.is_empty());
        assert!(eval.event.is_none(), "None -> None is not a change");
        assert!(eval.summary.state.allows_automatic());
    }

    #[test]
    fn a_fresh_safe_reading_is_safe_and_staleness_is_unknown() {
        let clock = ManualClock::new();
        let mut hub = SafetyMonitorHub::new(HubSettings::default());
        let m = input(&clock, LinkState::Ready, Some(SafetyLevel::Safe));
        let eval = hub.evaluate(std::slice::from_ref(&m), &clock);
        assert_eq!(eval.summary.state, SafetyGate::Safe);
        assert!(eval.state_changed());
        assert!(matches!(
            eval.event.as_ref().unwrap().event,
            EventKind::SafetyChanged {
                state: SafetyGate::Safe,
                ..
            }
        ));
        assert_eq!(eval.summary.sources[0].level, Some(SafetyLevel::Safe));
        // Same reading again: no event.
        assert!(hub
            .evaluate(std::slice::from_ref(&m), &clock)
            .event
            .is_none());

        // No new reading for longer than stale_after: unknown, no trigger.
        clock.advance(DEFAULT_STALE_AFTER + Duration::from_millis(1));
        let eval = hub.evaluate(std::slice::from_ref(&m), &clock);
        assert_eq!(eval.summary.state, SafetyGate::Unknown);
        assert!(eval.triggers.is_empty());
        assert!(!eval.startup);
        assert_eq!(eval.summary.sources[0].level, None);
        assert!(!eval.summary.state.allows_automatic());
    }

    #[test]
    fn one_failed_poll_or_a_degraded_link_never_triggers_the_policy() {
        let clock = ManualClock::new();
        let mut hub = SafetyMonitorHub::new(HubSettings::default());
        let safe = input(&clock, LinkState::Ready, Some(SafetyLevel::Safe));
        hub.evaluate(std::slice::from_ref(&safe), &clock);

        // The worker publishes an Error level and marks the link degraded.
        let failed = MonitorInput {
            link: LinkState::Degraded,
            level: Observed::error(Source::Property),
            ..safe.clone()
        };
        let eval = hub.evaluate(std::slice::from_ref(&failed), &clock);
        assert_eq!(eval.summary.state, SafetyGate::Unknown);
        assert!(eval.triggers.is_empty());

        // Even a stale "unsafe" value from before the failure does not count.
        let stale_unsafe = MonitorInput {
            link: LinkState::Degraded,
            level: Observed::fresh_at(SafetyLevel::Unsafe, Source::Property, &clock).into_stale(),
            ..safe.clone()
        };
        let eval = hub.evaluate(std::slice::from_ref(&stale_unsafe), &clock);
        assert_eq!(eval.summary.state, SafetyGate::Unknown);
        assert!(eval.triggers.is_empty());

        // A degraded monitor is not safe even with a fresh safe reading
        // (§2.8), but a fresh unsafe reading from it is still confirmed.
        let degraded_safe = MonitorInput {
            link: LinkState::Degraded,
            ..safe.clone()
        };
        let eval = hub.evaluate(std::slice::from_ref(&degraded_safe), &clock);
        assert_eq!(eval.summary.state, SafetyGate::Unknown);
        let degraded_unsafe = MonitorInput {
            link: LinkState::Degraded,
            level: Observed::fresh_at(SafetyLevel::Unsafe, Source::Property, &clock),
            ..safe.clone()
        };
        let eval = hub.evaluate(std::slice::from_ref(&degraded_unsafe), &clock);
        assert_eq!(eval.summary.state, SafetyGate::Unsafe);
        assert_eq!(eval.triggers.len(), 1);

        // Connecting and disconnected links are not safe either.
        for link in [LinkState::Connecting, LinkState::Disconnected] {
            let m = MonitorInput {
                link,
                ..safe.clone()
            };
            let eval = hub.evaluate(std::slice::from_ref(&m), &clock);
            assert_eq!(eval.summary.state, SafetyGate::Unknown, "{link:?}");
            assert!(eval.triggers.is_empty());
        }
    }

    #[test]
    fn a_confirmed_unsafe_reading_triggers_once_per_edge() {
        let clock = ManualClock::new();
        let mut hub = SafetyMonitorHub::new(HubSettings::default());
        let m = input(&clock, LinkState::Ready, Some(SafetyLevel::Safe));
        hub.evaluate(std::slice::from_ref(&m), &clock);

        let unsafe_reading = reading(&clock, m.clone(), SafetyLevel::Unsafe);
        let eval = hub.evaluate(std::slice::from_ref(&unsafe_reading), &clock);
        assert_eq!(eval.summary.state, SafetyGate::Unsafe);
        assert_eq!(
            eval.triggers,
            vec![PolicyTrigger::ConfirmedUnsafe {
                connection_id: m.connection_id,
                name: "cloudwatcher".into(),
                level: SafetyLevel::Unsafe,
                detail: None,
            }]
        );
        assert!(eval.triggers[0].reason().contains("reports unsafe"));
        clock.advance_ms(2000);
        let again = reading(&clock, m.clone(), SafetyLevel::Unsafe);
        assert!(hub.evaluate(&[again], &clock).triggers.is_empty());
    }

    #[test]
    fn a_lost_monitor_triggers_and_warnings_follow_the_setting() {
        let clock = ManualClock::new();
        let mut hub = SafetyMonitorHub::new(HubSettings::default());
        let m = input(&clock, LinkState::Ready, Some(SafetyLevel::Safe));
        hub.evaluate(std::slice::from_ref(&m), &clock);
        let lost = MonitorInput {
            link: LinkState::Lost,
            level: Observed::error(Source::Property),
            ..m.clone()
        };
        let eval = hub.evaluate(std::slice::from_ref(&lost), &clock);
        assert_eq!(eval.summary.state, SafetyGate::Unknown);
        assert!(matches!(
            eval.triggers.as_slice(),
            [PolicyTrigger::MonitorLost { .. }]
        ));
        assert!(hub
            .evaluate(std::slice::from_ref(&lost), &clock)
            .triggers
            .is_empty());

        let warning = input(&clock, LinkState::Ready, Some(SafetyLevel::Warning));
        let mut strict = SafetyMonitorHub::new(HubSettings::default());
        let eval = strict.evaluate(std::slice::from_ref(&warning), &clock);
        assert_eq!(eval.summary.state, SafetyGate::Unsafe);
        assert!(eval.triggers[0].reason().contains("warning"));
        let mut lenient = SafetyMonitorHub::new(HubSettings {
            warning_is_unsafe: false,
            ..HubSettings::default()
        });
        let eval = lenient.evaluate(&[warning], &clock);
        assert_eq!(eval.summary.state, SafetyGate::Safe);
        assert!(eval.triggers.is_empty());
    }

    #[test]
    fn the_hub_is_an_and_of_every_monitor() {
        let clock = ManualClock::new();
        let mut hub = SafetyMonitorHub::new(HubSettings::default());
        let a = input(&clock, LinkState::Ready, Some(SafetyLevel::Safe));
        let b = MonitorInput {
            connection_id: Uuid::new_v4(),
            name: "rain sensor".into(),
            ..a.clone()
        };
        assert_eq!(
            hub.evaluate(&[a.clone(), b.clone()], &clock).summary.state,
            SafetyGate::Safe
        );
        let b_unknown = MonitorInput {
            link: LinkState::Degraded,
            level: Observed::error(Source::Property),
            ..b.clone()
        };
        let eval = hub.evaluate(&[a.clone(), b_unknown.clone()], &clock);
        assert_eq!(eval.summary.state, SafetyGate::Unknown);
        let b_unsafe = reading(&clock, b.clone(), SafetyLevel::Unsafe);
        let eval = hub.evaluate(&[a.clone(), b_unsafe], &clock);
        assert_eq!(eval.summary.state, SafetyGate::Unsafe);
        assert_eq!(eval.triggers.len(), 1);
        assert_eq!(eval.triggers[0].connection_id(), b.connection_id);
        assert_eq!(eval.summary.sources.len(), 2);
        // Unsafe outranks unknown.
        let a_unknown = MonitorInput {
            link: LinkState::Degraded,
            level: Observed::error(Source::Property),
            ..a.clone()
        };
        let b_unsafe = reading(&clock, b, SafetyLevel::Unsafe);
        assert_eq!(
            hub.evaluate(&[a_unknown, b_unsafe], &clock).summary.state,
            SafetyGate::Unsafe
        );
        // Removing every monitor returns to none.
        assert_eq!(hub.evaluate(&[], &clock).summary.state, SafetyGate::None);
    }

    #[test]
    fn a_monitor_without_a_first_reading_is_startup_unknown() {
        let clock = ManualClock::new();
        let mut hub = SafetyMonitorHub::new(HubSettings::default());
        let m = MonitorInput {
            level: Observed::not_set(Source::Derived),
            ..input(&clock, LinkState::Connecting, Some(SafetyLevel::Safe))
        };
        let eval = hub.evaluate(std::slice::from_ref(&m), &clock);
        assert_eq!(eval.summary.state, SafetyGate::Unknown);
        assert!(eval.startup);
        let ok = reading(&clock, m.clone(), SafetyLevel::Safe);
        hub.evaluate(&[ok], &clock);
        let failed = MonitorInput {
            link: LinkState::Degraded,
            ..m
        };
        assert!(!hub.evaluate(&[failed], &clock).startup);
    }

    #[test]
    fn inputs_come_from_handles_without_touching_devices() {
        let clock = ManualClock::new();
        let (handle, fake) = fake_safety_monitor(SafetySnapshot::default());
        fake.update(|snap| {
            snap.level = Observed::fresh_at(SafetyLevel::Unsafe, Source::Property, &clock)
        });
        let m = MonitorInput::from_handle(&handle);
        assert_eq!(m.link, LinkState::Ready);
        assert_eq!(m.level.value, Some(SafetyLevel::Unsafe));
        assert_eq!(m.name, "Fake safety_monitor");
        // A snapshot from another connection is no reading.
        fake.update(|snap| snap.connection_id = Uuid::new_v4());
        assert_eq!(MonitorInput::from_handle(&handle).level.value, None);
        fake.retire();
        assert_eq!(
            MonitorInput::from_handle(&handle).link,
            LinkState::Disconnected
        );
    }

    #[test]
    fn the_gate_check_refuses_automatic_actions_unless_safe_or_none() {
        let check = SafetyGateCheck;
        let dither = GatedCommand::new(
            GatedOp::Guider(GuiderCmd::Dither {
                amount: 5.0,
                unit: PixelUnit::GuidePx,
                ra_only: false,
                settle: Settle::default(),
            }),
            CommandSource::Trigger {
                rule: "dither_every_n".into(),
            },
            AutonomyMode::Bounded,
            Priority::Exposure,
        );
        let operator_slew = GatedCommand::new(
            GatedOp::Mount(MountCmd::Unpark),
            CommandSource::Operator { overriding: false },
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        let automatic_park = GatedCommand::new(
            GatedOp::Mount(MountCmd::Park),
            CommandSource::System,
            AutonomyMode::Bounded,
            Priority::MountRecovery,
        );
        assert_eq!(check.name(), SAFETY_CHECK_NAME);
        assert!(check.applies_to(&dither));
        assert!(!check.applies_to(&operator_slew), "operator commands pass");
        assert!(!check.applies_to(&automatic_park), "safing ops pass");

        let mut snap = EquipmentSnapshot::default();
        assert!(check.check(&dither, &snap).is_ok(), "none allows");
        assert_eq!(check.annotations(&dither, &snap), vec!["no_safety_monitor"]);
        for state in [SafetyGate::Unknown, SafetyGate::Unsafe] {
            snap.safety.state = state;
            let rejection = check.check(&dither, &snap).unwrap_err();
            assert_eq!(rejection.check, SAFETY_CHECK_NAME);
            assert!(
                rejection.reason.contains("automatic dither"),
                "{}",
                rejection.reason
            );
        }
        snap.safety.state = SafetyGate::Safe;
        assert!(check.check(&dither, &snap).is_ok());
        assert!(check.annotations(&dither, &snap).is_empty());

        // Safe again, but the safety block from the unsafe episode is still
        // active: automatic actions wait for the operator's resume (Tier 1).
        snap.resume_blocks
            .push(crate::equipment::snapshot::ResumeBlock {
                id: Uuid::new_v4(),
                kind: ResumeBlockKind::SafetyUnsafe,
                raised_at: clock_now(),
                raised_mono_ns: 0,
                device_kind: None,
                connection_id: None,
                command_id: None,
                reason: "rain".into(),
            });
        let rejection = check.check(&dither, &snap).unwrap_err();
        assert!(rejection.reason.contains("resume"), "{}", rejection.reason);
        snap.resume_blocks[0].kind = ResumeBlockKind::Estop;
        assert!(
            check.check(&dither, &snap).is_ok(),
            "other blocks are the gate's own resume-block rule, not this check"
        );
    }

    fn clock_now() -> chrono::DateTime<chrono::Utc> {
        ManualClock::new().now_utc()
    }
}
