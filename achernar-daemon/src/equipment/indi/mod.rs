//! Native INDI client and drivers (Tier 2). Never ports libindi, KStars or Ekos
//! code (§6).
//!
//! Module tree created by WP0; owner: WP2-1.

pub mod cap_light;
pub mod client;
pub mod dome;
pub mod filterwheel;
pub mod focuser;
#[cfg(test)]
pub mod mock;
pub mod props;
pub mod rotator;
pub mod telescope;
pub mod weather;
pub mod xml;
