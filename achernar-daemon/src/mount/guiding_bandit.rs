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

    pub fn update(&mut self, reward: f64, axis: &str, context: &[f64]) {
        if context.is_empty() {
            return;
        }

        self.ensure_dimension(context.len());
        let target = if axis.eq_ignore_ascii_case("ra") {
            &mut self.ra_state
        } else if axis.eq_ignore_ascii_case("dec") {
            &mut self.dec_state
        } else {
            return;
        };
        let arm_idx = target
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                axis_score(a, context, 0.0)
                    .2
                    .total_cmp(&axis_score(b, context, 0.0).2)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        target[arm_idx].update(context, reward.clamp(-5.0, 2.0));
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
