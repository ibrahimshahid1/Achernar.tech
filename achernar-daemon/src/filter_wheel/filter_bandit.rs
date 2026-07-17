//! Filter bandit — Section 8.2 (FW-2).
//! LinUCB contextual bandit for filter arm selection.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterBanditArm {
    pub filter_name: String,
    pub estimated_reward: f64,
    pub uncertainty: f64,
    pub ucb_score: f64,
    pub times_pulled: u32,
}

pub struct FilterBandit {
    alpha: f64,
    arms: Vec<FilterBanditArm>,
    states: Vec<LinUcbState>,
}

#[derive(Debug, Clone)]
struct LinUcbState {
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
}

impl FilterBandit {
    pub fn new(filter_names: &[String], alpha: f64) -> Self {
        let arms = filter_names
            .iter()
            .map(|name| FilterBanditArm {
                filter_name: name.clone(),
                estimated_reward: 0.0,
                uncertainty: 1.0,
                ucb_score: f64::MAX,
                times_pulled: 0,
            })
            .collect();
        let states = filter_names.iter().map(|_| LinUcbState::new(0)).collect();
        Self {
            alpha,
            arms,
            states,
        }
    }

    /// Select the best filter arm using LinUCB.
    pub fn select_arm(&self, context: &[f64]) -> &FilterBanditArm {
        let Some(first) = self.arms.first() else {
            panic!("FilterBandit requires at least one arm");
        };
        if context.is_empty() {
            return self
                .arms
                .iter()
                .max_by(|a, b| a.ucb_score.total_cmp(&b.ucb_score))
                .unwrap_or(first);
        }

        let best_idx = self
            .states
            .iter()
            .enumerate()
            .filter(|(_, state)| state.dimension() == context.len())
            .map(|(idx, state)| (idx, linucb_score(state, context, self.alpha).2))
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .unwrap_or_else(|| {
                self.arms
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, arm)| arm.times_pulled)
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            });

        &self.arms[best_idx]
    }

    /// Update arm with observed reward.
    pub fn update(&mut self, filter_name: &str, reward: f64, context: &[f64]) {
        if context.is_empty() {
            return;
        }
        let Some(idx) = self
            .arms
            .iter()
            .position(|arm| arm.filter_name == filter_name)
        else {
            return;
        };

        self.ensure_dimension(context.len());
        self.states[idx].update(context, reward.clamp(-1.0, 1.0));
        let (estimated_reward, uncertainty, ucb_score) =
            linucb_score(&self.states[idx], context, self.alpha);
        let arm = &mut self.arms[idx];
        arm.estimated_reward = estimated_reward;
        arm.uncertainty = uncertainty;
        arm.ucb_score = ucb_score;
        arm.times_pulled = arm.times_pulled.saturating_add(1);
    }

    fn ensure_dimension(&mut self, dimension: usize) {
        for state in &mut self.states {
            if state.dimension() != dimension {
                *state = LinUcbState::new(dimension);
            }
        }
    }
}

impl LinUcbState {
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

fn linucb_score(state: &LinUcbState, context: &[f64], alpha: f64) -> (f64, f64, f64) {
    if state.dimension() != context.len() {
        return (0.0, 1.0, f64::MAX);
    }

    let theta = solve_linear(&state.a, &state.b).unwrap_or_else(|| vec![0.0; context.len()]);
    let inv_context =
        solve_linear(&state.a, context).unwrap_or_else(|| context.iter().map(|_| 0.0).collect());
    let estimated_reward = dot(&theta, context);
    let uncertainty = dot(context, &inv_context).max(0.0).sqrt();
    let ucb_score = estimated_reward + alpha * uncertainty;
    (estimated_reward, uncertainty, ucb_score)
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
