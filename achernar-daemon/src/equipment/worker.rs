//! Placeholder created by WP0. Owner: WP1A-2 (§2.5, D3).
//!
//! `DeviceWorker`: one std thread per connected device, named
//! `eq-{kind}-{connection id prefix}`; drives `Operation` objects; adaptive
//! cadence; panic containment under `catch_unwind`.
//!
//! This file holds only this comment until its owner implements it against the
//! frozen WP0 contracts in `types`, `traits`, `events`, `snapshot`, `commands`,
//! `ops`, `clock` and `handle` (spec §2.4, §5.1 rule 8).
