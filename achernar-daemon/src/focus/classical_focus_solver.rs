//! Classical focus solver — Section 7.3.4.
//! HFD/HFR computation + parabola/V-curve fitting.

use serde::{Deserialize, Serialize};

/// Result of a classical curve fit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveFitResult {
    pub optimal_position: f64,
    pub predicted_hfr: f64,
    pub r_squared: f64,
    pub coefficients: Vec<f64>,
    pub method: CurveFitMethod,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CurveFitMethod {
    Parabola,
    VCurve,
    Hyperbola,
}

/// Classical focus metric solver using HFD/HFR.
pub struct ClassicalFocusSolver;

impl ClassicalFocusSolver {
    /// Compute HFR (Half-Flux Radius) from a star image crop.
    pub fn compute_hfr(_image_data: &[f32], _width: usize, _height: usize) -> f64 {
        todo!("Implement HFR computation")
    }

    /// Compute HFD (Half-Flux Diameter) from a star image crop.
    pub fn compute_hfd(_image_data: &[f32], _width: usize, _height: usize) -> f64 {
        todo!("Implement HFD computation")
    }

    /// Fit a parabola to position-HFR samples and find the minimum.
    pub fn fit_parabola(_positions: &[i32], _hfr_values: &[f64]) -> Option<CurveFitResult> {
        todo!("Implement parabola fit — least squares on y = a*x^2 + b*x + c")
    }

    /// Fit a V-curve (two linear segments) to position-HFR samples.
    pub fn fit_v_curve(_positions: &[i32], _hfr_values: &[f64]) -> Option<CurveFitResult> {
        todo!("Implement V-curve fit")
    }

    /// Fit a hyperbola to position-HFR samples.
    pub fn fit_hyperbola(_positions: &[i32], _hfr_values: &[f64]) -> Option<CurveFitResult> {
        todo!("Implement hyperbola fit")
    }
}
