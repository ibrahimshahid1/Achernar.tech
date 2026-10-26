//! Guiding through PHD2 (spec §3.5; WP1A-7). The direct guider and M-1
//! actuation are WP2-7 (`direct`, `dither_select`, `algo`).
//!
//! - [`phd2_proto`]: PHD2 event and RPC types, line framing.
//! - [`phd2_client`]: the TCP client, per-method timeouts, the out-of-band
//!   `stop_capture` abort path.
//! - [`service`]: `connect` and the `GuiderService` task behind
//!   `GuiderHandle` (G-a to G-j read side), plus `GuiderWindows` (H9).
//! - [`rms`]: Welford RMS, RMS about zero, peaks (G-i).
//! - [`steps`]: raw step records, phase tags, the step history (G-h).
//!
//! This file holds the pieces other packages call without a connection:
//! the gate's [`GuiderCheck`] (§2.9), the §2.13 `guiding` telemetry block
//! ([`guiding_telemetry`]), the G-l camera-role check
//! ([`camera_role_conflict`]), the G-m autofocus plan
//! ([`autofocus_guiding_plan`]), the busy flags and the D20 helper
//! [`was_guiding`].

pub mod algo;
pub mod direct;
pub mod dither_select;
pub mod phd2_client;
pub mod phd2_proto;
pub mod rms;
pub mod service;
pub mod steps;

use serde_json::{json, Value};

use crate::config::{GuiderConfig, GuidingDuringAutofocus};
use crate::equipment::commands::{GatedCommand, GatedOp, GuiderCmd, SafetyRejection, Settle};
use crate::equipment::events::EventKind;
use crate::equipment::ops::OpKind;
use crate::equipment::snapshot::{EquipmentSnapshot, GuiderAppState, GuiderSnapshot};
use crate::equipment::traits::SafetyCheck;
use crate::equipment::types::{CommandSource, DeviceKind};

/// Name of the guider safety check (`safety_rejected{check}`).
pub const GUIDER_CHECK: &str = "guider";

/// The guider's gate check (§2.9 step 3): dither and `set_algo_param` run
/// only between exposures; dither only while PHD2 is Guiding; a policy may
/// set an algorithm parameter only when `[guider] allow_policy_actuation`.
///
/// "Between exposures" uses the camera hook in the snapshot. An exposure
/// that is integrating refuses every source. An unknown camera state
/// refuses automatic sources and lets an operator through with the
/// annotation `camera_state_unknown`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GuiderCheck {
    pub allow_policy_actuation: bool,
}

impl GuiderCheck {
    pub fn from_config(config: &GuiderConfig) -> Self {
        Self {
            allow_policy_actuation: config.allow_policy_actuation,
        }
    }

    fn exposing(snap: &EquipmentSnapshot) -> Option<bool> {
        snap.camera.exposing.fresh_value().copied()
    }
}

impl SafetyCheck for GuiderCheck {
    fn name(&self) -> &'static str {
        GUIDER_CHECK
    }

    fn applies_to(&self, cmd: &GatedCommand) -> bool {
        matches!(
            cmd.op,
            GatedOp::Guider(GuiderCmd::Dither { .. } | GuiderCmd::SetAlgoParam { .. })
        )
    }

    fn check(&self, cmd: &GatedCommand, snap: &EquipmentSnapshot) -> Result<(), SafetyRejection> {
        let reject = |reason: &str| Err(SafetyRejection::new(GUIDER_CHECK, reason));
        match Self::exposing(snap) {
            Some(true) => return reject("camera_exposing"),
            Some(false) => {}
            None if cmd.source.is_automatic() => return reject("camera_state_unknown"),
            None => {}
        }
        match &cmd.op {
            GatedOp::Guider(GuiderCmd::Dither { .. }) => {
                let Some(guider) = snap.guider.as_deref() else {
                    return reject("guider_not_connected");
                };
                if snap
                    .link(DeviceKind::Guider)
                    .is_some_and(|link| !link.is_connected())
                {
                    return reject("guider_not_ready");
                }
                match guider.app_state.known() {
                    Some(GuiderAppState::Guiding) => Ok(()),
                    _ => reject("not_guiding"),
                }
            }
            GatedOp::Guider(GuiderCmd::SetAlgoParam { .. }) => {
                if matches!(cmd.source, CommandSource::Policy { .. })
                    && !self.allow_policy_actuation
                {
                    return reject("policy_actuation_disabled");
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn annotations(&self, cmd: &GatedCommand, snap: &EquipmentSnapshot) -> Vec<String> {
        if self.applies_to(cmd) && Self::exposing(snap).is_none() {
            vec!["camera_state_unknown".into()]
        } else {
            Vec::new()
        }
    }
}

/// D20: the guider counts as guiding for pre-motion coordination
/// (`coord.rs`) in these states; `was_guiding` is recorded from this before
/// `GuiderCmd::Stop`.
pub fn was_guiding(snap: &GuiderSnapshot) -> bool {
    matches!(
        snap.app_state.known(),
        Some(GuiderAppState::Guiding | GuiderAppState::Calibrating | GuiderAppState::LostLock)
    )
}

/// The `BusySet` flags the guider sets (H20): `(dithering, guider_settling)`.
pub fn busy_flags(snap: &GuiderSnapshot) -> (bool, bool) {
    let dithering = snap
        .active_op
        .as_ref()
        .is_some_and(|op| op.kind == OpKind::Dither);
    (dithering, snap.settling)
}

/// The §2.13 `guiding` block (today hard-coded null in `telemetry.rs`). Every
/// field is nullable; `rms_total` is arcsec, under the key the frontend
/// reads. `rms_total_px` is an additive field for rigs whose pixel scale is
/// unknown.
pub fn guiding_telemetry(snap: &GuiderSnapshot) -> Value {
    let rms = snap.rms.as_ref();
    let step = snap.last_step.as_ref();
    json!({
        "state": snap.app_state.value.and_then(|s| serde_json::to_value(s).ok()),
        "rms_ra_arcsec": rms.and_then(|r| r.ra_arcsec),
        "rms_dec_arcsec": rms.and_then(|r| r.dec_arcsec),
        "rms_total": rms.and_then(|r| r.total_arcsec),
        "rms_total_px": rms.map(|r| r.total_px),
        "dx": step.map(|s| s.dx),
        "dy": step.map(|s| s.dy),
        "snr": step.and_then(|s| s.snr),
        "star_mass": step.and_then(|s| s.star_mass),
        "hfd": step.and_then(|s| s.hfd),
        "settling": snap.settling,
        "pixel_scale": snap.pixel_scale_arcsec,
        "calibrated": snap.calibrated,
        "last_dither": snap.last_dither.as_ref().map(|d| json!({
            "at": d.at,
            "dx": d.dx,
            "dy": d.dy,
            "settle_status": d.settle_status,
        })),
        "observed_at": snap.app_state.observed_at,
    })
}

fn normalise(name: &str) -> String {
    name.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// G-l: an OLS camera whose name matches PHD2's guide camera
/// (`get_current_equipment`) conflicts with PHD2's ownership. Names match
/// when, after trimming, collapsing whitespace and ignoring case, they are
/// equal or one contains the other (at least 4 characters). The spine
/// refuses the connection unless forced and records the role.
pub fn camera_role_conflict(guider_camera: Option<&str>, camera_name: &str) -> Option<EventKind> {
    let guider = normalise(guider_camera?);
    let camera = normalise(camera_name);
    if guider.is_empty() || camera.is_empty() {
        return None;
    }
    let (short, long) = if guider.len() <= camera.len() {
        (&guider, &camera)
    } else {
        (&camera, &guider)
    };
    let matches = guider == camera || (short.chars().count() >= 4 && long.contains(short.as_str()));
    matches.then(|| EventKind::CameraRoleConflict {
        camera_name: camera_name.to_string(),
        guider_camera: guider_camera.unwrap_or_default().to_string(),
    })
}

/// G-m: what to do with guiding around an autofocus run
/// (`[guider] guiding_during_autofocus`, after NINA's
/// `AutoFocusDisableGuiding`). Issued by the focus integration (WP1B-2)
/// through the guider handle.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AutofocusGuidingPlan {
    pub before: Option<GuiderCmd>,
    pub after: Vec<GuiderCmd>,
}

/// `keep`: nothing. `pause`: pause before, then resume and settle after.
/// `stop`: stop before, then restart and settle after. Nothing is planned
/// when PHD2 is not guiding. The settle after the run is a `guide` request,
/// which PHD2 answers with a settle when it is already guiding.
pub fn autofocus_guiding_plan(
    mode: GuidingDuringAutofocus,
    state: Option<GuiderAppState>,
    settle: Settle,
) -> AutofocusGuidingPlan {
    if state != Some(GuiderAppState::Guiding) {
        return AutofocusGuidingPlan::default();
    }
    let restart = GuiderCmd::Start {
        settle,
        recalibrate: false,
    };
    match mode {
        GuidingDuringAutofocus::Keep => AutofocusGuidingPlan::default(),
        GuidingDuringAutofocus::Pause => AutofocusGuidingPlan {
            before: Some(GuiderCmd::Pause { full: false }),
            after: vec![GuiderCmd::Resume, restart],
        },
        GuidingDuringAutofocus::Stop => AutofocusGuidingPlan {
            before: Some(GuiderCmd::Stop),
            after: vec![restart],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::{Axis, PixelUnit};
    use crate::equipment::ops::{ActiveOp, OpPhase};
    use crate::equipment::snapshot::{CameraSummary, GuideRms, GuideStepSample, LastDither};
    use crate::equipment::types::{
        AutonomyMode, CommandId, LinkState, ObsState, Observed, Priority, Source,
    };
    use chrono::Utc;
    use std::sync::Arc;

    fn dither(source: CommandSource) -> GatedCommand {
        GatedCommand::new(
            GatedOp::Guider(GuiderCmd::Dither {
                amount: 3.0,
                unit: PixelUnit::GuidePx,
                ra_only: false,
                settle: Settle::default(),
            }),
            source,
            AutonomyMode::Manual,
            Priority::Exposure,
        )
    }

    fn snapshot(state: GuiderAppState, exposing: Option<bool>) -> EquipmentSnapshot {
        EquipmentSnapshot {
            guider: Some(Arc::new(GuiderSnapshot {
                app_state: Observed::fresh(state, Source::Phd2Event, Utc::now(), 1),
                ..GuiderSnapshot::default()
            })),
            camera: CameraSummary {
                exposing: match exposing {
                    Some(value) => Observed::fresh(value, Source::Derived, Utc::now(), 1),
                    None => Observed::not_set(Source::Derived),
                },
                ..CameraSummary::default()
            },
            ..EquipmentSnapshot::default()
        }
    }

    fn policy() -> CommandSource {
        CommandSource::Policy {
            model: "m2".into(),
            version: "0.1".into(),
        }
    }

    #[test]
    fn dither_runs_only_between_exposures_and_while_guiding() {
        let check = GuiderCheck::default();
        let operator = dither(CommandSource::Operator { overriding: false });
        assert!(check.applies_to(&operator));
        assert!(check
            .check(&operator, &snapshot(GuiderAppState::Guiding, Some(false)))
            .is_ok());
        let err = check
            .check(&operator, &snapshot(GuiderAppState::Guiding, Some(true)))
            .unwrap_err();
        assert_eq!(
            (err.check.as_str(), err.reason.as_str()),
            ("guider", "camera_exposing")
        );
        let err = check
            .check(&operator, &snapshot(GuiderAppState::Looping, Some(false)))
            .unwrap_err();
        assert_eq!(err.reason, "not_guiding");
        let err = check
            .check(
                &operator,
                &snapshot(GuiderAppState::Calibrating, Some(false)),
            )
            .unwrap_err();
        assert_eq!(err.reason, "not_guiding");
        let err = check
            .check(&operator, &EquipmentSnapshot::default())
            .unwrap_err();
        assert_eq!(err.reason, "guider_not_connected");
        // A lost guider link refuses even with a stale "Guiding" state.
        let mut lost = snapshot(GuiderAppState::Guiding, Some(false));
        lost.links.insert(DeviceKind::Guider, LinkState::Lost);
        assert_eq!(
            check.check(&operator, &lost).unwrap_err().reason,
            "guider_not_ready"
        );
    }

    #[test]
    fn an_unknown_camera_state_refuses_automatic_sources_only() {
        let check = GuiderCheck::default();
        let snap = snapshot(GuiderAppState::Guiding, None);
        let automatic = dither(policy());
        assert_eq!(
            check.check(&automatic, &snap).unwrap_err().reason,
            "camera_state_unknown"
        );
        let operator = dither(CommandSource::Operator { overriding: false });
        assert!(check.check(&operator, &snap).is_ok());
        assert_eq!(
            check.annotations(&operator, &snap),
            vec!["camera_state_unknown"]
        );
        // A stale camera reading is not fresh evidence either.
        let mut stale = snapshot(GuiderAppState::Guiding, Some(false));
        stale.camera.exposing.state = ObsState::Stale;
        assert!(check.check(&automatic, &stale).is_err());
    }

    #[test]
    fn policy_algorithm_writes_need_the_actuation_switch() {
        let op = GatedOp::Guider(GuiderCmd::SetAlgoParam {
            axis: Axis::Ra,
            name: "aggression".into(),
            value: 0.6,
        });
        let cmd = GatedCommand::new(
            op,
            policy(),
            AutonomyMode::Bounded,
            Priority::Recommendation,
        );
        let snap = snapshot(GuiderAppState::Guiding, Some(false));
        assert_eq!(
            GuiderCheck::default()
                .check(&cmd, &snap)
                .unwrap_err()
                .reason,
            "policy_actuation_disabled"
        );
        let allowed = GuiderCheck::from_config(&GuiderConfig {
            allow_policy_actuation: true,
            ..GuiderConfig::default()
        });
        assert!(allowed.check(&cmd, &snap).is_ok());
        let exposing = snapshot(GuiderAppState::Guiding, Some(true));
        assert!(allowed.check(&cmd, &exposing).is_err());
        // Other guider commands are not this check's business.
        let stop = GatedCommand::new(
            GatedOp::Guider(GuiderCmd::Stop),
            CommandSource::System,
            AutonomyMode::Manual,
            Priority::MountRecovery,
        );
        assert!(!GuiderCheck::default().applies_to(&stop));
    }

    #[test]
    fn telemetry_is_null_until_reported_and_carries_arcsec_rms() {
        let empty = guiding_telemetry(&GuiderSnapshot::default());
        assert_eq!(empty["state"], Value::Null);
        assert_eq!(empty["rms_total"], Value::Null);
        assert_eq!(empty["dx"], Value::Null);
        assert_eq!(empty["last_dither"], Value::Null);
        let at = Utc::now();
        let snap = GuiderSnapshot {
            app_state: Observed::fresh(GuiderAppState::LostLock, Source::Phd2Event, at, 1),
            rms: Some(GuideRms {
                total_px: 0.4,
                total_arcsec: Some(0.6),
                ra_arcsec: Some(0.4),
                dec_arcsec: Some(0.45),
                ..GuideRms::default()
            }),
            last_step: Some(GuideStepSample {
                dx: -0.2,
                dy: 0.1,
                snr: Some(30.0),
                ..GuideStepSample::default()
            }),
            last_dither: Some(LastDither {
                at,
                dx: Some(1.0),
                dy: Some(-1.0),
                settle_status: Some(0),
            }),
            pixel_scale_arcsec: Some(1.5),
            calibrated: Some(true),
            ..GuiderSnapshot::default()
        };
        let t = guiding_telemetry(&snap);
        assert_eq!(t["state"], json!("lost_lock"));
        assert_eq!(t["rms_total"], json!(0.6));
        assert_eq!(t["rms_total_px"], json!(0.4));
        assert_eq!(t["dx"], json!(-0.2));
        assert_eq!(t["snr"], json!(30.0));
        assert_eq!(t["hfd"], Value::Null);
        assert_eq!(t["last_dither"]["settle_status"], json!(0));
        assert_eq!(t["pixel_scale"], json!(1.5));
    }

    #[test]
    fn busy_flags_and_was_guiding_follow_the_snapshot() {
        let mut snap = GuiderSnapshot::default();
        assert_eq!(busy_flags(&snap), (false, false));
        assert!(!was_guiding(&snap));
        snap.settling = true;
        snap.active_op = Some(ActiveOp {
            command_id: CommandId::new(),
            kind: OpKind::Dither,
            phase: OpPhase::Settling,
            started_at: Utc::now(),
            started_mono_ns: 0,
            deadline_mono_ns: 1,
        });
        assert_eq!(busy_flags(&snap), (true, true));
        for (state, expected) in [
            (GuiderAppState::Guiding, true),
            (GuiderAppState::Calibrating, true),
            (GuiderAppState::LostLock, true),
            (GuiderAppState::Paused, false),
            (GuiderAppState::Looping, false),
        ] {
            snap.app_state = Observed::fresh(state, Source::Phd2Event, Utc::now(), 1);
            assert_eq!(was_guiding(&snap), expected, "{state:?}");
        }
    }

    #[test]
    fn the_guide_camera_role_check_matches_names_conservatively() {
        let conflict = camera_role_conflict(Some("SVBONY SV105"), "sv105").unwrap();
        assert_eq!(conflict.name(), "camera_role_conflict");
        assert!(camera_role_conflict(Some("ZWO ASI120MM Mini"), "zwo  asi120mm mini").is_some());
        assert!(camera_role_conflict(Some("ZWO ASI120MM"), "ZWO ASI294MC").is_none());
        assert!(camera_role_conflict(Some("Cam"), "Camera Simulator").is_none());
        assert!(camera_role_conflict(None, "SV105").is_none());
        assert!(camera_role_conflict(Some(""), "SV105").is_none());
    }

    #[test]
    fn the_autofocus_plan_follows_the_configured_mode() {
        let settle = Settle::default();
        let guiding = Some(GuiderAppState::Guiding);
        assert_eq!(
            autofocus_guiding_plan(GuidingDuringAutofocus::Keep, guiding, settle),
            AutofocusGuidingPlan::default()
        );
        let pause = autofocus_guiding_plan(GuidingDuringAutofocus::Pause, guiding, settle);
        assert_eq!(pause.before, Some(GuiderCmd::Pause { full: false }));
        assert_eq!(pause.after.len(), 2);
        assert_eq!(pause.after[0], GuiderCmd::Resume);
        let stop = autofocus_guiding_plan(GuidingDuringAutofocus::Stop, guiding, settle);
        assert_eq!(stop.before, Some(GuiderCmd::Stop));
        assert!(matches!(
            stop.after[..],
            [GuiderCmd::Start {
                recalibrate: false,
                ..
            }]
        ));
        assert_eq!(
            autofocus_guiding_plan(
                GuidingDuringAutofocus::Stop,
                Some(GuiderAppState::Stopped),
                settle
            ),
            AutofocusGuidingPlan::default(),
            "nothing to restore when PHD2 was not guiding"
        );
    }
}
