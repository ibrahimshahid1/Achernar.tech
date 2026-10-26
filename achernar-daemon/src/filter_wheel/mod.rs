//! Filter wheel subsystem — Section 8.2.
//!
//! Rewired onto the equipment layer (equipment spec §3.4, §7.2.11-7.2.14;
//! WP1A-6):
//!
//! - `filter_controller`: a thin front over the equipment handshake
//!   (`equipment::filter::change`). It no longer pretends to switch and never
//!   reads the TOML seed offsets.
//! - `filter_snr_scorer` (FW-1): consistent electron units, no hard-coded
//!   detector, sky or target constants; `uncalibrated` without RB-C inputs.
//! - `filter_bandit` (FW-2): an explicit continue arm, freeze inputs and H23
//!   decision records.
//! - `filter_safety`: the same rules as the gate plug-in
//!   (`equipment::filter::check`).
//! - `filter_planner` (FW-3): validates a plan and locks it.

// The spine (WP1B-1) wires this subsystem into `AppState` and the routes;
// until then it is exercised by tests only.
#![allow(dead_code)]

pub mod filter_bandit;
pub mod filter_controller;
pub mod filter_planner;
pub mod filter_safety;
pub mod filter_snr_scorer;
