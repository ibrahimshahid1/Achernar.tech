//! Focus subsystem — Sections 7.3.4–7.3.5.
//! Autofocus state machine, classical solver, policy, drift, and the FOP
//! feature vector.
//!
//! Three modules that used to live here are gone.
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
//! generated into `fop_features_v1.json`. `fop_features` builds it online from
//! the equipment layer (equipment spec H7) and is checked against that JSON.
//!
//! `backlash` was an unwired "always approach from one side" compensator. It
//! is superseded by `equipment/focuser/backlash.rs`, which applies the
//! `none`, `overshoot` and `absolute` models on the focuser's worker for every
//! move, autofocus included (equipment spec F-e).

pub mod classical_focus_solver;
pub mod focus_agent;
pub mod focus_drift;
pub mod focus_policy;
pub mod fop_features;
