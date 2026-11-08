//! Dither policy — Section 8.3 (M-2, v6 §8.3.2; equipment spec §7.2.16).
//!
//! The contract is what PHD2 accepts: a **magnitude**, a **settle**
//! (`{pixels, time, timeout}`) and an **explicit unit** (§2.14). PHD2 picks
//! the direction itself and reports the realised offset in
//! `GuidingDithered{dx, dy}`, which `dither_events` records (H10). The
//! previous `dx`/`dy` output could not be executed by PHD2 at all.
//!
//! The policy sizes the dither in **main-camera pixels**, the unit the
//! stack cares about, and converts to guide-camera pixels with the rig's
//! image scales (main from the rig profile, guide from PHD2
//! `get_pixel_scale`). Without both scales no PHD2 command is proposed: PHD2
//! dithers in guide pixels only, and a guessed scale would dither by an
//! invented amount. Settle pixels are always guide pixels, PHD2's unit.
//!
//! Every decision is an H23 `policy_decisions` record: the inputs and their
//! ages, the safe arms (`dither`, `skip`), the chosen arm and the requested,
//! validated (converted) action. The rule is deterministic, so no
//! propensities are recorded. A dither runs only between exposures and only
//! while PHD2 is guiding; the gate's guider check enforces both again.

// The sequencer that calls this between exposures is Tier 2 (WP2-8); until
// then only tests call it.
#![allow(dead_code)]

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use crate::equipment::commands::{GatedCommand, GatedOp, GuiderCmd, PixelUnit, Settle};
use crate::equipment::events::PolicyDecisionRecord;
use crate::equipment::types::{AutonomyMode, CommandSource, Priority};

/// `policy_decisions.policy`.
pub const POLICY: &str = "m2_dither_policy";
/// `policy_decisions.policy_version`.
pub const POLICY_VERSION: &str = "heuristic-2";

/// Smallest and largest dither, main-camera pixels.
pub const MIN_AMOUNT_MAIN_PX: f64 = 1.0;
pub const MAX_AMOUNT_MAIN_PX: f64 = 15.0;

/// One dither: magnitude, settle and an explicit unit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DitherParams {
    pub amount: f64,
    pub unit: PixelUnit,
    pub ra_only: bool,
    /// PHD2 settle; `pixels` is in guide-camera pixels.
    pub settle: Settle,
}

impl DitherParams {
    /// The same dither in guide-camera pixels. `Err` names the missing or
    /// invalid scale.
    pub fn to_guide_px(
        self,
        main_scale_arcsec_px: Option<f64>,
        guide_scale_arcsec_px: Option<f64>,
    ) -> Result<DitherParams, &'static str> {
        match self.unit {
            PixelUnit::GuidePx => Ok(self),
            PixelUnit::MainPx => {
                let main = main_scale_arcsec_px
                    .filter(|s| s.is_finite() && *s > 0.0)
                    .ok_or("main_image_scale_unknown")?;
                let guide = guide_scale_arcsec_px
                    .filter(|s| s.is_finite() && *s > 0.0)
                    .ok_or("guide_image_scale_unknown")?;
                Ok(DitherParams {
                    amount: self.amount * main / guide,
                    unit: PixelUnit::GuidePx,
                    ..self
                })
            }
        }
    }

    /// The PHD2 command; only guide pixels are accepted.
    pub fn guider_cmd(&self) -> Result<GuiderCmd, &'static str> {
        if self.unit != PixelUnit::GuidePx {
            return Err("phd2_dithers_in_guide_px");
        }
        if !(self.amount.is_finite() && self.amount > 0.0) {
            return Err("dither_amount_invalid");
        }
        Ok(GuiderCmd::Dither {
            amount: self.amount,
            unit: PixelUnit::GuidePx,
            ra_only: self.ra_only,
            settle: self.settle,
        })
    }
}

/// What M-2 reads (v6 §8.3.2). Missing values stay missing.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct DitherInputs {
    /// Guide RMS over the last exposure, arcsec (R24).
    pub guide_rms_arcsec: Option<f64>,
    /// Image FWHM on the main camera, arcsec (vision).
    pub seeing_fwhm_arcsec: Option<f64>,
    /// Main-camera image scale, arcsec/px (rig profile).
    pub main_scale_arcsec_px: Option<f64>,
    /// Guide-camera image scale, arcsec/px (PHD2 `get_pixel_scale`).
    pub guide_scale_arcsec_px: Option<f64>,
    pub filter: Option<String>,
    /// Fraction of recent frames rejected, and of hot pixels (vision).
    pub recent_rejection_fraction: Option<f64>,
    pub hot_pixel_fraction: Option<f64>,
    /// No exposure is integrating.
    pub between_exposures: bool,
    /// PHD2 reports Guiding.
    pub guiding: bool,
    #[serde(skip)]
    pub measurement_ages_ms: BTreeMap<String, u64>,
    #[serde(skip)]
    pub parent_frame_ids: Vec<String>,
}

/// One M-2 decision.
#[derive(Debug, Clone, PartialEq)]
pub struct DitherDecision {
    pub decision_id: Uuid,
    /// The dither the policy wants, in main-camera pixels.
    pub requested: DitherParams,
    /// The same dither in guide pixels, when the scales allow it.
    pub validated: Option<DitherParams>,
    /// The gated PHD2 command, when the dither may run.
    pub proposal: Option<GatedCommand>,
    /// Why no command was proposed; empty when one was.
    pub refused: Vec<String>,
    pub record: PolicyDecisionRecord,
}

/// The deterministic M-2 rule.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DitherPolicy {
    pub ra_only: bool,
    /// `[guider]` settle defaults (time and fallback pixels).
    pub settle_defaults: Settle,
}

impl DitherPolicy {
    /// Size the dither: about three main pixels plus a share of the seeing
    /// and the guide error, so consecutive frames land on different pixels
    /// even when guiding wanders; then a settle tolerance scaled to the
    /// guide error and a timeout that grows with the amount.
    pub fn predict(&self, inputs: &DitherInputs) -> DitherParams {
        let main = inputs
            .main_scale_arcsec_px
            .filter(|s| s.is_finite() && *s > 0.0);
        let in_main = |arcsec: Option<f64>| {
            arcsec
                .filter(|a| a.is_finite() && *a >= 0.0)
                .zip(main)
                .map(|(a, scale)| (a / scale).min(20.0))
                .unwrap_or(0.0)
        };
        let amount = (3.0
            + 0.6 * in_main(inputs.seeing_fwhm_arcsec)
            + 0.8 * in_main(inputs.guide_rms_arcsec))
        .clamp(MIN_AMOUNT_MAIN_PX, MAX_AMOUNT_MAIN_PX);
        let rms = inputs
            .guide_rms_arcsec
            .filter(|r| r.is_finite() && *r >= 0.0);
        let settle_pixels = rms
            .zip(
                inputs
                    .guide_scale_arcsec_px
                    .filter(|s| s.is_finite() && *s > 0.0),
            )
            .map(|(rms, guide)| (2.0 * rms / guide).clamp(0.5, 3.0))
            .unwrap_or(self.settle_defaults.pixels);
        let timeout = (10.0 + amount * 1.5 + 4.0 * rms.unwrap_or(0.0))
            .clamp(10.0, 60.0)
            .max(self.settle_defaults.time_s + 5.0);
        DitherParams {
            amount,
            unit: PixelUnit::MainPx,
            ra_only: self.ra_only,
            settle: Settle {
                pixels: settle_pixels,
                time_s: self.settle_defaults.time_s,
                timeout_s: timeout,
            },
        }
    }

    /// Decide, propose and record (see the module docs).
    pub fn decide(
        &self,
        inputs: &DitherInputs,
        mode: AutonomyMode,
        now: DateTime<Utc>,
    ) -> DitherDecision {
        let decision_id = Uuid::new_v4();
        let requested = self.predict(inputs);
        let mut refused = Vec::new();
        if !inputs.between_exposures {
            refused.push("exposure_integrating".to_string());
        }
        if !inputs.guiding {
            refused.push("guider_not_guiding".to_string());
        }
        let validated = match requested
            .to_guide_px(inputs.main_scale_arcsec_px, inputs.guide_scale_arcsec_px)
        {
            Ok(params) => Some(params),
            Err(reason) => {
                refused.push(reason.to_string());
                None
            }
        };
        let command = validated.and_then(|params| match params.guider_cmd() {
            Ok(cmd) => Some(cmd),
            Err(reason) => {
                refused.push(reason.to_string());
                None
            }
        });
        let proposal = match command {
            Some(cmd) if refused.is_empty() => Some(
                GatedCommand::new(
                    GatedOp::Guider(cmd),
                    CommandSource::Policy {
                        model: POLICY.to_string(),
                        version: POLICY_VERSION.to_string(),
                    },
                    mode,
                    Priority::Recommendation,
                )
                .with_params(json!({
                    "op": "dither",
                    "policy": POLICY,
                    "policy_version": POLICY_VERSION,
                    "decision_id": decision_id,
                    "requested": requested,
                    "validated": validated,
                })),
            ),
            _ => None,
        };
        let chosen = if proposal.is_some() { "dither" } else { "skip" };
        let safe_arms = if refused.is_empty() {
            vec!["dither".to_string(), "skip".to_string()]
        } else {
            vec!["skip".to_string()]
        };
        let record = PolicyDecisionRecord {
            decision_id,
            policy: POLICY.to_string(),
            policy_version: POLICY_VERSION.to_string(),
            mode,
            context: json!({
                "inputs": inputs,
                "refused": refused,
            }),
            measurement_ages_ms: inputs.measurement_ages_ms.clone(),
            safe_arms,
            chosen_arm: Some(chosen.to_string()),
            propensities: None,
            requested_action: Some(json!({"arm": chosen, "dither": requested})),
            validated_action: validated.map(|params| json!({"dither": params})),
            executed_action: None,
            command_id: proposal.as_ref().map(|cmd| cmd.command_id),
            reward_components: None,
            attribution_window_start: None,
            attribution_window_end: None,
            interrupted: false,
            override_applied: false,
            parent_frame_ids: inputs.parent_frame_ids.clone(),
            created_at: now,
        };
        DitherDecision {
            decision_id,
            requested,
            validated,
            proposal,
            refused,
            record,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> DitherInputs {
        DitherInputs {
            guide_rms_arcsec: Some(1.2),
            seeing_fwhm_arcsec: Some(3.0),
            main_scale_arcsec_px: Some(1.5),
            guide_scale_arcsec_px: Some(4.0),
            between_exposures: true,
            guiding: true,
            measurement_ages_ms: [("guider".to_string(), 800)].into_iter().collect(),
            parent_frame_ids: vec!["frame-41".into()],
            ..DitherInputs::default()
        }
    }

    #[test]
    fn the_contract_is_a_magnitude_a_settle_and_an_explicit_unit() {
        let params = DitherPolicy::default().predict(&inputs());
        assert_eq!(params.unit, PixelUnit::MainPx);
        // 3 + 0.6 * (3.0 / 1.5) + 0.8 * (1.2 / 1.5) = 4.84 main pixels.
        assert!((params.amount - 4.84).abs() < 1e-9, "{}", params.amount);
        // Settle tolerance in guide pixels: 2 * 1.2 / 4.0.
        assert!((params.settle.pixels - 0.6).abs() < 1e-9);
        assert!(params.settle.timeout_s >= params.settle.time_s);
        let json = serde_json::to_value(params).unwrap();
        assert_eq!(json["unit"], json!("main_px"));
        assert!(json.get("dx_pixels").is_none());
    }

    #[test]
    fn main_pixels_become_guide_pixels_only_with_both_scales() {
        let params = DitherPolicy::default().predict(&inputs());
        let guide = params.to_guide_px(Some(1.5), Some(4.0)).unwrap();
        assert_eq!(guide.unit, PixelUnit::GuidePx);
        assert!((guide.amount - 4.84 * 1.5 / 4.0).abs() < 1e-9);
        assert_eq!(
            params.to_guide_px(None, Some(4.0)),
            Err("main_image_scale_unknown")
        );
        assert_eq!(
            params.to_guide_px(Some(1.5), None),
            Err("guide_image_scale_unknown")
        );
        assert_eq!(params.guider_cmd(), Err("phd2_dithers_in_guide_px"));
        assert!(matches!(
            guide.guider_cmd().unwrap(),
            GuiderCmd::Dither {
                unit: PixelUnit::GuidePx,
                ..
            }
        ));
    }

    #[test]
    fn a_dither_is_proposed_between_exposures_while_guiding_and_recorded() {
        let decision =
            DitherPolicy::default().decide(&inputs(), AutonomyMode::Proposal, Utc::now());
        assert!(decision.refused.is_empty(), "{:?}", decision.refused);
        let proposal = decision.proposal.as_ref().unwrap();
        assert_eq!(proposal.priority, Priority::Recommendation);
        assert!(matches!(proposal.source, CommandSource::Policy { .. }));
        let record = &decision.record;
        assert_eq!(record.policy, POLICY);
        assert_eq!(record.chosen_arm.as_deref(), Some("dither"));
        assert_eq!(record.command_id, Some(proposal.command_id));
        assert_eq!(record.propensities, None);
        assert_eq!(
            record.requested_action.as_ref().unwrap()["dither"]["unit"],
            json!("main_px")
        );
        assert_eq!(
            record.validated_action.as_ref().unwrap()["dither"]["unit"],
            json!("guide_px")
        );
        assert_eq!(record.measurement_ages_ms["guider"], 800);
        assert_eq!(record.parent_frame_ids, vec!["frame-41".to_string()]);
    }

    #[test]
    fn no_dither_while_exposing_unguided_or_without_scales() {
        let policy = DitherPolicy::default();
        let exposing = DitherInputs {
            between_exposures: false,
            ..inputs()
        };
        let decision = policy.decide(&exposing, AutonomyMode::Proposal, Utc::now());
        assert!(decision.proposal.is_none());
        assert_eq!(decision.refused, vec!["exposure_integrating".to_string()]);
        assert_eq!(decision.record.chosen_arm.as_deref(), Some("skip"));

        let unguided = DitherInputs {
            guiding: false,
            ..inputs()
        };
        assert!(policy
            .decide(&unguided, AutonomyMode::Proposal, Utc::now())
            .refused
            .contains(&"guider_not_guiding".to_string()));

        let no_scale = DitherInputs {
            guide_scale_arcsec_px: None,
            ..inputs()
        };
        let decision = policy.decide(&no_scale, AutonomyMode::Proposal, Utc::now());
        assert!(decision.proposal.is_none() && decision.validated.is_none());
        assert_eq!(
            decision.refused,
            vec!["guide_image_scale_unknown".to_string()]
        );
        // The amount is still sized without the guide scale; the settle falls
        // back to the configured tolerance rather than an invented one.
        assert_eq!(decision.requested.settle.pixels, Settle::default().pixels);
    }
}
