//! Mount subsystem — Section 8.3.
//!
//! The mount-side models and their hooks into the equipment layer (WP1B-4):
//! the plate-solve trigger (M-3, H11), the center loop (M-n), the pointing
//! model (M-4, H12), the dither policy contract (M-2) and the guiding
//! bandit's context builder (M-1). Mount limits live in
//! `equipment/mount/limits.rs`; device control lives in `equipment/mount`.

pub mod auto_solve;
pub mod center;
pub mod dither_policy;
pub mod guiding_bandit;
pub mod plate_solve_trigger;
pub mod pointing_model;
