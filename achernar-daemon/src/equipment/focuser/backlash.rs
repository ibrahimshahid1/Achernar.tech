//! Backlash models (spec §3.3 F-e), implemented independently from this
//! specification.
//!
//! A focuser's gear train has slop: after a reversal the motor turns a few
//! steps before the draw tube follows. Three models are offered, plus a
//! declaration that the firmware deals with it:
//!
//! - `none`: the move goes straight to the target.
//! - `overshoot{in, out}`: when travelling in a direction whose amount is
//!   non-zero, go past the target by that amount, pause, then come back. The
//!   last approach is then always from the same side, so the slop is always
//!   taken up the same way. Skipped (and reported) when the overshoot point
//!   would leave the travel bounds.
//! - `absolute{in, out}`: compensate only when the direction of travel
//!   reverses. The extra steps accumulate in a virtual offset, and the
//!   logical position is `raw - offset`, so the compensation never shows in
//!   the position the operator, the models or the profile see. The offset is
//!   kept within `[-in, +out]`, so asymmetric amounts cannot drift, and the
//!   raw target of **every** move is checked against the travel bounds: when
//!   the compensated target would leave them the compensation is skipped,
//!   and when even the current offset would, the offset is dropped (the move
//!   goes to the plain logical target and the session reports a
//!   `focuser_position_reset`).
//! - `backlash_owner = firmware`: the device compensates internally and
//!   Achernar applies none, whatever the model says.
//!
//! The planner works on absolute positions. For a relative focuser the
//! positions are in the session's step-counter space (the logical position,
//! or 0 when the origin is unknown); the motion operation turns consecutive
//! targets into relative steps.
//!
//! The `[focus.backlash]` table (`enabled`, `compensation_steps`,
//! `always_approach_from`) gives the global default; an identity profile
//! overrides it per device (D17).

use serde::{Deserialize, Serialize};

use crate::config::BacklashConfig;
use crate::equipment::snapshot::{BacklashModel, BacklashOwner, FocusDirection};

/// What one planned leg is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegPurpose {
    /// The point past the target an overshoot goes to before returning.
    Overshoot,
    /// The (raw) target itself.
    Target,
}

/// One planned leg: an absolute raw (or counter) position to reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Leg {
    pub raw_target: i32,
    pub purpose: LegPurpose,
}

/// Compensation state carried from one move to the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BacklashState {
    /// `raw - logical`. Only the `absolute` model makes it non-zero.
    pub offset: i32,
    /// Direction of the last completed move; `None` when unknown (first
    /// move of a session, or after an interrupted move).
    pub last_direction: Option<FocusDirection>,
}

/// Why compensation was not applied to a move that would have had it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkipReason {
    /// The overshoot point, or the compensated raw target, is outside the
    /// travel bounds.
    OutsideTravelBounds,
}

/// The legs of one logical move, and the state once it completes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BacklashPlan {
    /// The model actually applied (`none` when the firmware owns backlash).
    pub model: BacklashModel,
    pub owner: BacklashOwner,
    pub logical_start: i32,
    pub logical_target: i32,
    /// Raw position at the start (`logical_start + offset`).
    pub raw_start: i32,
    /// Raw position the move ends at.
    pub raw_target: i32,
    /// Legs in order; empty when the focuser is already at the target.
    pub legs: Vec<Leg>,
    /// Direction of the logical move; `None` for a zero move.
    pub direction: Option<FocusDirection>,
    /// The direction differs from the last completed move's; `None` when
    /// either is unknown.
    pub reversal: Option<bool>,
    pub skipped: Option<SkipReason>,
    /// `absolute` only: the offset this move removes because the raw target
    /// would otherwise leave the travel bounds. The logical position then
    /// re-baselines by this many steps.
    pub offset_dropped: Option<i32>,
    /// The compensation state once every leg has completed.
    pub state_after: BacklashState,
}

impl BacklashPlan {
    /// True when the plan adds steps beyond the plain move.
    pub fn compensates(&self) -> bool {
        // A plain move keeps the starting offset (`raw_start - logical_start`).
        let plain_raw_target = i64::from(self.logical_target) + i64::from(self.raw_start)
            - i64::from(self.logical_start);
        self.legs.len() > 1 || i64::from(self.raw_target) != plain_raw_target
    }
}

/// The model in force: a firmware-owned focuser gets none (F-e).
pub fn effective_model(model: BacklashModel, owner: BacklashOwner) -> BacklashModel {
    match owner {
        BacklashOwner::Firmware => BacklashModel::None,
        BacklashOwner::Achernar => model,
    }
}

/// The global default from `[focus.backlash]`.
///
/// `always_approach_from` names the side the final approach comes from:
/// `"inward"` overshoots on outward moves (so the last leg travels in), and
/// anything else, the default `"outward"`, overshoots on inward moves.
/// Disabled, or zero steps, means `none`.
pub fn model_from_config(config: &BacklashConfig) -> BacklashModel {
    let steps = config.compensation_steps.max(0);
    if !config.enabled || steps == 0 {
        return BacklashModel::None;
    }
    let approach_inward = config
        .always_approach_from
        .as_deref()
        .map(|side| side.trim().eq_ignore_ascii_case("inward"))
        .unwrap_or(false);
    if approach_inward {
        BacklashModel::Overshoot {
            in_steps: 0,
            out_steps: steps,
        }
    } else {
        BacklashModel::Overshoot {
            in_steps: steps,
            out_steps: 0,
        }
    }
}

/// Direction from `from` to `to`; `None` when equal.
pub fn direction(from: i32, to: i32) -> Option<FocusDirection> {
    match to.cmp(&from) {
        std::cmp::Ordering::Greater => Some(FocusDirection::Out),
        std::cmp::Ordering::Less => Some(FocusDirection::In),
        std::cmp::Ordering::Equal => None,
    }
}

fn within(bounds: Option<(i32, i32)>, value: i64) -> bool {
    match bounds {
        Some((lo, hi)) => value >= i64::from(lo) && value <= i64::from(hi),
        None => value >= i64::from(i32::MIN) && value <= i64::from(i32::MAX),
    }
}

/// Plan one logical move.
///
/// `bounds` are the raw travel bounds (`[0, MaxStep]` intersected with the
/// `FocusSafetyGuard` and profile bounds) or `None` when they are unknown (a
/// relative focuser whose origin is unknown). Every leg's raw target lies
/// within them whenever the logical target does.
pub fn plan(
    model: BacklashModel,
    owner: BacklashOwner,
    state: BacklashState,
    logical_start: i32,
    logical_target: i32,
    bounds: Option<(i32, i32)>,
) -> BacklashPlan {
    let model = effective_model(model, owner);
    let dir = direction(logical_start, logical_target);
    let reversal = match (state.last_direction, dir) {
        (Some(last), Some(now)) => Some(last != now),
        _ => None,
    };
    // A model change or an ownership change must not leave a stale offset
    // behind: only the absolute model keeps one.
    let offset = match model {
        BacklashModel::Absolute { .. } => state.offset,
        _ => 0,
    };
    let raw_start = logical_start.saturating_add(state.offset);
    let mut plan = BacklashPlan {
        model,
        owner,
        logical_start,
        logical_target,
        raw_start,
        raw_target: logical_target.saturating_add(offset),
        legs: Vec::new(),
        direction: dir,
        reversal,
        skipped: None,
        offset_dropped: None,
        state_after: BacklashState {
            offset,
            last_direction: dir.or(state.last_direction),
        },
    };
    let Some(dir) = dir else {
        // Already at the logical target. A stale offset from another model
        // still needs one leg to remove it.
        if plan.raw_target != raw_start {
            plan.legs.push(Leg {
                raw_target: plan.raw_target,
                purpose: LegPurpose::Target,
            });
        }
        return plan;
    };

    match model {
        BacklashModel::None => {
            plan.legs.push(Leg {
                raw_target: plan.raw_target,
                purpose: LegPurpose::Target,
            });
        }
        BacklashModel::Overshoot {
            in_steps,
            out_steps,
        } => {
            let amount = match dir {
                FocusDirection::Out => out_steps.max(0),
                FocusDirection::In => in_steps.max(0),
            };
            let target = i64::from(plan.raw_target);
            let past = match dir {
                FocusDirection::Out => target + i64::from(amount),
                FocusDirection::In => target - i64::from(amount),
            };
            if amount > 0 && within(bounds, past) {
                plan.legs.push(Leg {
                    raw_target: past as i32,
                    purpose: LegPurpose::Overshoot,
                });
            } else if amount > 0 {
                plan.skipped = Some(SkipReason::OutsideTravelBounds);
            }
            plan.legs.push(Leg {
                raw_target: plan.raw_target,
                purpose: LegPurpose::Target,
            });
        }
        BacklashModel::Absolute {
            in_steps,
            out_steps,
        } => {
            let (in_steps, out_steps) = (i64::from(in_steps.max(0)), i64::from(out_steps.max(0)));
            // The offset stays within one dead band, so unequal in and out
            // amounts cannot make it (and the persisted state) drift.
            let limit = |value: i64| value.clamp(-in_steps, out_steps);
            let delta = match (reversal, dir) {
                (Some(true), FocusDirection::Out) => out_steps,
                (Some(true), FocusDirection::In) => -in_steps,
                _ => 0,
            };
            let kept = limit(i64::from(offset));
            let compensated = limit(kept + delta);
            let raw_with = |offset: i64| i64::from(logical_target) + offset;
            // Every move's raw target is checked, not just a reversal's: the
            // offset alone can push a target near either end out of bounds.
            let chosen = if within(bounds, raw_with(compensated)) {
                compensated
            } else if within(bounds, raw_with(kept)) {
                plan.skipped = Some(SkipReason::OutsideTravelBounds);
                kept
            } else {
                // The logical target is inside the bounds, so dropping the
                // offset always yields a valid raw target.
                plan.skipped = Some(SkipReason::OutsideTravelBounds);
                if offset != 0 {
                    plan.offset_dropped = Some(offset);
                }
                0
            };
            // `chosen` lies in [-in, +out] (or is zero) and `raw_with(chosen)`
            // passed `within`, so both fit in an i32.
            plan.state_after.offset = chosen as i32;
            plan.raw_target = raw_with(chosen) as i32;
            plan.legs.push(Leg {
                raw_target: plan.raw_target,
                purpose: LegPurpose::Target,
            });
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_STATE: BacklashState = BacklashState {
        offset: 0,
        last_direction: None,
    };

    fn targets(plan: &BacklashPlan) -> Vec<i32> {
        plan.legs.iter().map(|leg| leg.raw_target).collect()
    }

    #[test]
    fn none_model_goes_straight_to_the_target() {
        let p = plan(
            BacklashModel::None,
            BacklashOwner::Achernar,
            NO_STATE,
            1000,
            1200,
            Some((0, 5000)),
        );
        assert_eq!(targets(&p), vec![1200]);
        assert_eq!(p.direction, Some(FocusDirection::Out));
        assert_eq!(p.reversal, None, "first move: reversal unknown");
        assert_eq!(p.state_after.last_direction, Some(FocusDirection::Out));
        assert!(!p.compensates());
    }

    #[test]
    fn overshoot_goes_past_then_returns_only_in_its_direction() {
        let model = BacklashModel::Overshoot {
            in_steps: 0,
            out_steps: 40,
        };
        let out = plan(
            model,
            BacklashOwner::Achernar,
            NO_STATE,
            1000,
            1200,
            Some((0, 5000)),
        );
        assert_eq!(targets(&out), vec![1240, 1200]);
        assert_eq!(out.legs[0].purpose, LegPurpose::Overshoot);
        assert_eq!(out.legs[1].purpose, LegPurpose::Target);
        assert!(out.compensates());
        // Moving in needs no overshoot: the last approach is already inward.
        let inward = plan(
            model,
            BacklashOwner::Achernar,
            out.state_after,
            1200,
            1100,
            Some((0, 5000)),
        );
        assert_eq!(targets(&inward), vec![1100]);
        assert_eq!(inward.reversal, Some(true));
        assert_eq!(inward.state_after.offset, 0);
    }

    #[test]
    fn overshoot_is_skipped_when_it_would_leave_the_travel_bounds() {
        let model = BacklashModel::Overshoot {
            in_steps: 50,
            out_steps: 50,
        };
        let p = plan(
            model,
            BacklashOwner::Achernar,
            NO_STATE,
            1000,
            4980,
            Some((0, 5000)),
        );
        assert_eq!(targets(&p), vec![4980]);
        assert_eq!(p.skipped, Some(SkipReason::OutsideTravelBounds));
        let p = plan(
            model,
            BacklashOwner::Achernar,
            NO_STATE,
            1000,
            20,
            Some((0, 5000)),
        );
        assert_eq!(targets(&p), vec![20]);
        assert_eq!(p.skipped, Some(SkipReason::OutsideTravelBounds));
    }

    #[test]
    fn absolute_compensates_only_on_reversal_and_hides_the_offset() {
        let model = BacklashModel::Absolute {
            in_steps: 20,
            out_steps: 30,
        };
        let bounds = Some((0, 5000));
        // First move: direction unknown before, no compensation.
        let a = plan(model, BacklashOwner::Achernar, NO_STATE, 1000, 1200, bounds);
        assert_eq!(targets(&a), vec![1200]);
        assert_eq!(a.state_after.offset, 0);
        // Same direction again: none.
        let b = plan(
            model,
            BacklashOwner::Achernar,
            a.state_after,
            1200,
            1300,
            bounds,
        );
        assert_eq!(targets(&b), vec![1300]);
        assert_eq!(b.reversal, Some(false));
        // Reversal inward: 20 extra steps in, logical stays 1250.
        let c = plan(
            model,
            BacklashOwner::Achernar,
            b.state_after,
            1300,
            1250,
            bounds,
        );
        assert_eq!(c.reversal, Some(true));
        assert_eq!(targets(&c), vec![1230]);
        assert_eq!(c.state_after.offset, -20);
        assert_eq!(c.raw_target - c.state_after.offset, 1250);
        // Reversal outward: 30 extra steps out from the offset raw start.
        let d = plan(
            model,
            BacklashOwner::Achernar,
            c.state_after,
            1250,
            1400,
            bounds,
        );
        assert_eq!(d.raw_start, 1230);
        assert_eq!(targets(&d), vec![1410]);
        assert_eq!(d.state_after.offset, 10);
        assert_eq!(d.raw_target - d.state_after.offset, 1400);
    }

    #[test]
    fn absolute_skips_compensation_outside_bounds() {
        let model = BacklashModel::Absolute {
            in_steps: 20,
            out_steps: 20,
        };
        let state = BacklashState {
            offset: 0,
            last_direction: Some(FocusDirection::Out),
        };
        let p = plan(
            model,
            BacklashOwner::Achernar,
            state,
            100,
            10,
            Some((0, 5000)),
        );
        assert_eq!(targets(&p), vec![10]);
        assert_eq!(p.skipped, Some(SkipReason::OutsideTravelBounds));
        assert_eq!(p.state_after.offset, 0);
        assert_eq!(p.state_after.last_direction, Some(FocusDirection::In));
    }

    #[test]
    fn absolute_checks_every_raw_target_against_the_bounds() {
        let model = BacklashModel::Absolute {
            in_steps: 20,
            out_steps: 20,
        };
        let bounds = Some((0, 50_000));
        let after_in = BacklashState {
            offset: -20,
            last_direction: Some(FocusDirection::In),
        };
        // No reversal, but the offset alone would send Position -10: the
        // offset is dropped and the move goes to the plain target.
        let p = plan(model, BacklashOwner::Achernar, after_in, 500, 10, bounds);
        assert_eq!(targets(&p), vec![10]);
        assert_eq!(p.skipped, Some(SkipReason::OutsideTravelBounds));
        assert_eq!(p.offset_dropped, Some(-20));
        assert_eq!(
            p.state_after,
            BacklashState {
                offset: 0,
                last_direction: Some(FocusDirection::In)
            }
        );
        // A reversal to 0 whose compensation would pass 0: skipped, raw 0.
        let after_out = BacklashState {
            offset: 0,
            last_direction: Some(FocusDirection::Out),
        };
        let p = plan(model, BacklashOwner::Achernar, after_out, 500, 0, bounds);
        assert_eq!((targets(&p), p.reversal), (vec![0], Some(true)));
        assert_eq!(p.skipped, Some(SkipReason::OutsideTravelBounds));
        assert_eq!(p.offset_dropped, None, "nothing to drop");
        // The same at MaxStep, from either side of the dead band.
        let p = plan(
            model,
            BacklashOwner::Achernar,
            BacklashState {
                offset: 0,
                last_direction: Some(FocusDirection::In),
            },
            1_000,
            50_000,
            bounds,
        );
        assert_eq!(targets(&p), vec![50_000]);
        assert_eq!(p.skipped, Some(SkipReason::OutsideTravelBounds));
        let p = plan(
            model,
            BacklashOwner::Achernar,
            after_in,
            1_000,
            50_000,
            bounds,
        );
        assert_eq!(targets(&p), vec![50_000], "the reversal cancels the offset");
        assert_eq!((p.skipped, p.state_after.offset), (None, 0));
        let held_out = BacklashState {
            offset: 20,
            last_direction: Some(FocusDirection::Out),
        };
        let p = plan(
            model,
            BacklashOwner::Achernar,
            held_out,
            1_000,
            50_000,
            bounds,
        );
        assert_eq!(targets(&p), vec![50_000]);
        assert_eq!(p.offset_dropped, Some(20));
        // The FocusSafetyGuard minimum binds the raw target as well.
        let p = plan(
            model,
            BacklashOwner::Achernar,
            after_in,
            5_000,
            1_000,
            Some((1_000, 50_000)),
        );
        assert_eq!(targets(&p), vec![1_000]);
        assert_eq!(p.state_after.offset, 0);
    }

    #[test]
    fn asymmetric_absolute_amounts_cannot_drift() {
        let model = BacklashModel::Absolute {
            in_steps: 20,
            out_steps: 30,
        };
        let mut state = NO_STATE;
        let mut at = 10_000;
        let mut offsets = Vec::new();
        for i in 0..20 {
            let target = if i % 2 == 0 { at + 500 } else { at - 500 };
            let p = plan(
                model,
                BacklashOwner::Achernar,
                state,
                at,
                target,
                Some((0, 50_000)),
            );
            assert_eq!(p.raw_target - p.state_after.offset, target);
            state = p.state_after;
            at = target;
            offsets.push(state.offset);
        }
        assert!(
            offsets.iter().all(|o| (-20..=30).contains(o)),
            "{offsets:?}"
        );
        // It settles into one dead band instead of growing by 10 a cycle.
        assert_eq!(offsets[..6], [0, -20, 10, -10, 20, 0]);
        assert!(
            offsets[6..].chunks(2).all(|pair| pair == [30, 10]),
            "{offsets:?}"
        );
    }

    #[test]
    fn firmware_owned_backlash_gets_no_compensation() {
        let model = BacklashModel::Overshoot {
            in_steps: 0,
            out_steps: 40,
        };
        let p = plan(
            model,
            BacklashOwner::Firmware,
            NO_STATE,
            1000,
            1200,
            Some((0, 5000)),
        );
        assert_eq!(p.model, BacklashModel::None);
        assert_eq!(targets(&p), vec![1200]);
        assert_eq!(
            effective_model(model, BacklashOwner::Firmware),
            BacklashModel::None
        );
    }

    #[test]
    fn zero_moves_have_no_legs_and_keep_the_last_direction() {
        let state = BacklashState {
            offset: 0,
            last_direction: Some(FocusDirection::In),
        };
        let p = plan(
            BacklashModel::Absolute {
                in_steps: 5,
                out_steps: 5,
            },
            BacklashOwner::Achernar,
            state,
            700,
            700,
            None,
        );
        assert!(p.legs.is_empty());
        assert_eq!(p.direction, None);
        assert_eq!(p.state_after.last_direction, Some(FocusDirection::In));
    }

    #[test]
    fn switching_away_from_absolute_removes_the_offset() {
        let state = BacklashState {
            offset: -20,
            last_direction: Some(FocusDirection::In),
        };
        let p = plan(
            BacklashModel::None,
            BacklashOwner::Achernar,
            state,
            1000,
            1000,
            Some((0, 5000)),
        );
        // Raw 980 must become raw 1000 so logical == raw again.
        assert_eq!(p.raw_start, 980);
        assert_eq!(targets(&p), vec![1000]);
        assert_eq!(p.state_after.offset, 0);
    }

    #[test]
    fn config_defaults_map_to_an_overshoot_toward_the_approach_side() {
        let disabled = BacklashConfig::default();
        assert_eq!(model_from_config(&disabled), BacklashModel::None);
        let outward = BacklashConfig {
            enabled: true,
            compensation_steps: 35,
            always_approach_from: None,
        };
        assert_eq!(
            model_from_config(&outward),
            BacklashModel::Overshoot {
                in_steps: 35,
                out_steps: 0
            }
        );
        let inward = BacklashConfig {
            enabled: true,
            compensation_steps: 35,
            always_approach_from: Some("Inward".into()),
        };
        assert_eq!(
            model_from_config(&inward),
            BacklashModel::Overshoot {
                in_steps: 0,
                out_steps: 35
            }
        );
        let zero = BacklashConfig {
            enabled: true,
            compensation_steps: 0,
            always_approach_from: None,
        };
        assert_eq!(model_from_config(&zero), BacklashModel::None);
    }
}
