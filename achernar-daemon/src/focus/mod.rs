//! Focus subsystem — Sections 7.3.4–7.3.5.
//! Autofocus state machine, classical solver, policy, drift, and backlash.
//!
//! Two modules that used to live here are gone.
//!
//! `focus_inference` declared an ONNX runner for the Focus Quality Estimator
//! and `todo!()`d its implementation, while the FQE has run real ONNX inference
//! from `vision/fqe_runner.rs` for some time — a second, unimplemented
//! declaration of a solved problem, sitting in the module a reader would look
//! in first.
//!
//! `focus_features` declared a 15-field struct calling itself the 38-feature
//! vector of Section 7.6.2, with a `raw: Vec<f64>` escape hatch and a
//! `todo!()` tensor builder. It named features the daemon does not produce.
//! The authoritative description of that vector — names, order, normalization
//! and the database column behind each one — is `ml/contracts/fop_features.py`,
//! generated into `fop_features_v1.json`, which is what the runtime side should
//! read when it comes to build the vector online.

pub mod backlash;
pub mod classical_focus_solver;
pub mod focus_agent;
pub mod focus_drift;
pub mod focus_policy;
