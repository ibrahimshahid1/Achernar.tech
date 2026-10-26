//! ASCOM Alpaca client and device drivers, hand-written over `ureq` (O7).
//!
//! Module tree created by WP0; owner: WP1A-1.

pub mod client;
pub mod covercalibrator;
pub mod devicestate;
pub mod discovery;
pub mod dome;
pub mod errors;
pub mod filterwheel;
pub mod focuser;
pub mod form;
pub mod observingconditions;
pub mod rotator;
pub mod safetymonitor;
pub mod switch;
pub mod telescope;
