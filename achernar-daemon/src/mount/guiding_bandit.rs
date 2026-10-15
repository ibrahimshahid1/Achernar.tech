//! Guiding bandit — Section 8.3 (M-1).
//! LinUCB aggressiveness tuner for RA and Dec guiding.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidingBanditState {
    pub ra_aggressiveness: f64,
    pub dec_aggressiveness: f64,
    pub ra_rms: f64,
    pub dec_rms: f64,
}

pub struct GuidingBandit {
    ra_alpha: f64,
    dec_alpha: f64,
    arms: Vec<f64>,
    ra_state: Vec<AxisArmState>,
    dec_state: Vec<AxisArmState>,
}

impl GuidingBandit {
    pub fn new(alpha: f64) -> Self {
        let arms = vec![0.10, 0.30, 0.50, 0.70, 1.00, 1.30];
        Self {
            ra_alpha: alpha,
            dec_alpha: alpha,
            ra_state: arms.iter().map(|_| AxisArmState::new(0)).collect(),
            dec_state: arms.iter().map(|_| AxisArmState::new(0)).collect(),
            arms,
        }
    }

    pub fn select_aggressiveness(&self, context: &[f64]) -> GuidingBanditState {
        let ra_aggressiveness =
            select_axis_arm(&self.arms, &self.ra_state, context, self.ra_alpha).unwrap_or(0.70);
        let dec_aggressiveness =
            select_axis_arm(&self.arms, &self.dec_state, context, self.dec_alpha).unwrap_or(0.70);

        GuidingBanditState {
            ra_aggressiveness,
            dec_aggressiveness,
            ra_rms: context.get(4).copied().unwrap_or(0.0),
            dec_rms: context.get(5).copied().unwrap_or(0.0),
        }
    }

    /// Credit `reward` to the arm that was actually executed on `axis`.
    ///
    /// LinUCB must update the executed arm, never a recomputed greedy arm:
    /// exploration deliberately runs non-greedy arms. Returns `false` and
    /// leaves all state unchanged when the axis or executed aggressiveness is
    /// not one of this bandit's arms, or the reward or context is not finite.
    pub fn update(
        &mut self,
        axis: &str,
        executed_aggressiveness: f64,
        reward: f64,
        context: &[f64],
    ) -> bool {
        if context.is_empty()
            || !reward.is_finite()
            || context.iter().any(|value| !value.is_finite())
        {
            return false;
        }
        let is_ra = axis.eq_ignore_ascii_case("ra");
        if !is_ra && !axis.eq_ignore_ascii_case("dec") {
            return false;
        }
        let Some(arm_idx) = self
            .arms
            .iter()
            .position(|arm| (arm - executed_aggressiveness).abs() <= 1e-9)
        else {
            return false;
        };

        self.ensure_dimension(context.len());
        let target = if is_ra {
            &mut self.ra_state
        } else {
            &mut self.dec_state
        };
        target[arm_idx].update(context, reward.clamp(-5.0, 2.0));
        true
    }

    fn ensure_dimension(&mut self, dimension: usize) {
        for state in self.ra_state.iter_mut().chain(self.dec_state.iter_mut()) {
            if state.dimension() != dimension {
                *state = AxisArmState::new(dimension);
            }
        }
    }
}

#[derive(Debug, Clone)]
struct AxisArmState {
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
}

impl AxisArmState {
    fn new(dimension: usize) -> Self {
        let mut a = vec![vec![0.0; dimension]; dimension];
        for (idx, row) in a.iter_mut().enumerate() {
            row[idx] = 1.0;
        }
        Self {
            a,
            b: vec![0.0; dimension],
        }
    }

    fn dimension(&self) -> usize {
        self.b.len()
    }

    fn update(&mut self, context: &[f64], reward: f64) {
        for row in 0..context.len() {
            self.b[row] += reward * context[row];
            for col in 0..context.len() {
                self.a[row][col] += context[row] * context[col];
            }
        }
    }
}

fn select_axis_arm(
    arms: &[f64],
    states: &[AxisArmState],
    context: &[f64],
    alpha: f64,
) -> Option<f64> {
    if context.is_empty() {
        return None;
    }

    states
        .iter()
        .enumerate()
        .filter(|(_, state)| state.dimension() == context.len())
        .map(|(idx, state)| (idx, axis_score(state, context, alpha).2))
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .and_then(|(idx, _)| arms.get(idx).copied())
}

fn axis_score(state: &AxisArmState, context: &[f64], alpha: f64) -> (f64, f64, f64) {
    if state.dimension() != context.len() {
        return (0.0, 1.0, f64::MAX);
    }
    let theta = solve_linear(&state.a, &state.b).unwrap_or_else(|| vec![0.0; context.len()]);
    let inv_context =
        solve_linear(&state.a, context).unwrap_or_else(|| context.iter().map(|_| 0.0).collect());
    let expected = dot(&theta, context);
    let uncertainty = dot(context, &inv_context).max(0.0).sqrt();
    (expected, uncertainty, expected + alpha * uncertainty)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn solve_linear(matrix: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();

    for col in 0..n {
        let pivot = (col..n)
            .max_by(|a_idx, b_idx| a[*a_idx][col].abs().total_cmp(&a[*b_idx][col].abs()))?;
        if a[pivot][col].abs() <= f64::EPSILON {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let divisor = a[col][col];
        for c in col..n {
            a[col][c] /= divisor;
        }
        b[col] /= divisor;
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for c in col..n {
                a[row][c] -= factor * a[col][c];
            }
            b[row] -= factor * b[col];
        }
    }

    Some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONTEXT: [f64; 6] = [1.0, 0.5, 0.2, 0.1, 0.8, 0.6];

    fn updated_arms(states: &[AxisArmState]) -> Vec<usize> {
        states
            .iter()
            .enumerate()
            .filter(|(_, state)| state.b.iter().any(|value| *value != 0.0))
            .map(|(idx, _)| idx)
            .collect()
    }

    #[test]
    fn update_credits_the_executed_arm_not_the_greedy_arm() {
        let mut bandit = GuidingBandit::new(1.0);
        // All arms tie before learning, so a greedy recomputation would pick
        // the last arm (1.30). The executed arm here is the first (0.10).
        assert!(bandit.update("ra", 0.10, 1.0, &CONTEXT));
        assert_eq!(updated_arms(&bandit.ra_state), vec![0]);
        assert!(updated_arms(&bandit.dec_state).is_empty());
    }

    #[test]
    fn exploratory_selection_is_credited_to_that_arm() {
        let mut bandit = GuidingBandit::new(1_000.0);
        // Teach the 0.70 arm a strong positive reward so it is greedy-best.
        for _ in 0..20 {
            assert!(bandit.update("dec", 0.70, 2.0, &CONTEXT));
        }
        let selected = bandit.select_aggressiveness(&CONTEXT).dec_aggressiveness;
        assert_ne!(selected, 0.70, "large alpha should explore an untried arm");

        let before = bandit.dec_state[3].b.clone();
        assert!(bandit.update("dec", selected, -1.0, &CONTEXT));
        let explored = bandit.arms.iter().position(|arm| *arm == selected).unwrap();
        assert!(bandit.dec_state[explored]
            .b
            .iter()
            .any(|value| *value != 0.0));
        assert_eq!(
            bandit.dec_state[3].b, before,
            "greedy arm must not absorb the reward"
        );
    }

    #[test]
    fn unknown_axis_or_arm_or_bad_reward_leaves_state_unchanged() {
        let mut bandit = GuidingBandit::new(1.0);
        assert!(!bandit.update("alt", 0.70, 1.0, &CONTEXT));
        assert!(!bandit.update("ra", 0.42, 1.0, &CONTEXT));
        assert!(!bandit.update("ra", 0.70, f64::NAN, &CONTEXT));
        let mut poisoned = CONTEXT;
        poisoned[4] = f64::NAN;
        assert!(!bandit.update("ra", 0.70, 1.0, &poisoned));
        assert!(!bandit.update("ra", 0.70, 1.0, &[]));
        // No dimension reset happened either: states remain uninitialised.
        assert!(bandit.ra_state.iter().all(|state| state.dimension() == 0));
        assert!(bandit.dec_state.iter().all(|state| state.dimension() == 0));
    }
}
