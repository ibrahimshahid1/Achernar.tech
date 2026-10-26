//! Placeholder created by WP0. Owner: WP1A-1 (§3.1 C-3).
//!
//! Alpaca response checks in order: HTTP status (plain-text body), content
//! type, `ErrorNumber` and `ErrorMessage`, then `Value`; mapped with
//! `DeviceError::from_http_status` and `from_alpaca_error`.
//!
//! This file holds only this comment until its owner implements it against the
//! frozen WP0 contracts in `types`, `traits`, `events`, `snapshot`, `commands`,
//! `ops`, `clock` and `handle` (spec §2.4, §5.1 rule 8).
