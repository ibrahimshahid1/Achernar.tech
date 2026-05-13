//! Focus subsystem — Sections 7.3.4–7.3.5.
//! Autofocus state machine, classical solver, policy, inference, features, drift, and backlash.

pub mod backlash;
pub mod classical_focus_solver;
pub mod focus_agent;
pub mod focus_drift;
pub mod focus_features;
pub mod focus_inference;
pub mod focus_policy;
