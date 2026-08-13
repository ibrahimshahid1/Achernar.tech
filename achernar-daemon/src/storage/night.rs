//! `night_id` — the observing night a row belongs to.
//!
//! An observing night crosses midnight, so the calendar date of a timestamp is
//! the wrong grouping key: a sweep at 23:50 and the next one at 00:10 are the
//! same night, the same seeing, the same tube temperature, and the same
//! transparency, but they land on different dates. Grouping on the date would
//! let them fall on opposite sides of a train/test split and report a
//! generalisation score that is really a memorisation score.
//!
//! The convention is **the local date of sunset**: anything before local noon
//! belongs to the previous day. Noon is used rather than an actual sunset
//! calculation because it needs no site coordinates, never moves with the
//! season, and no one observes through it.
//!
//! This must stay identical to `night_id_for()` in `tools/import_session.py`.
//! The importer writes rows the daemon later groups with its own, and two
//! definitions of "night" would cut one session in half with nothing to
//! indicate it had happened.

use chrono::{DateTime, Datelike, Duration, Local, TimeZone, Timelike, Utc};

/// The night id for `when`, as `YYYY-MM-DD` in the host's local timezone.
///
/// Local time, not UTC: the whole point is to fall on the observer's side of
/// midnight, and in the Americas a UTC date rolls over in the middle of the
/// observing session. The daemon runs at the site, so the host timezone is the
/// site timezone.
pub fn night_id(when: DateTime<Utc>) -> String {
    night_id_local(when.with_timezone(&Local))
}

/// The same rule against an already-localized timestamp. Split out so tests can
/// pin a timezone instead of depending on the machine running them.
pub fn night_id_local<Tz: TimeZone>(local: DateTime<Tz>) -> String {
    let sunset_day = if local.hour() < 12 {
        // Before local noon — this belongs to the night that began yesterday.
        local.clone() - Duration::days(1)
    } else {
        local.clone()
    };
    format!(
        "{:04}-{:02}-{:02}",
        sunset_day.year(),
        sunset_day.month(),
        sunset_day.day()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::FixedOffset;

    /// Piedmont NC in observing season: UTC-5.
    fn eastern() -> FixedOffset {
        FixedOffset::west_opt(5 * 3600).unwrap()
    }

    fn at(y: i32, m: u32, d: u32, h: u32, min: u32) -> DateTime<FixedOffset> {
        eastern()
            .with_ymd_and_hms(y, m, d, h, min, 0)
            .single()
            .expect("valid local time")
    }

    #[test]
    fn evening_belongs_to_its_own_date() {
        assert_eq!(night_id_local(at(2026, 11, 14, 21, 30)), "2026-11-14");
    }

    #[test]
    fn after_midnight_belongs_to_the_previous_date() {
        // The case the whole module exists for: 00:10 is the same night as the
        // 23:50 sweep that preceded it by twenty minutes.
        assert_eq!(night_id_local(at(2026, 11, 15, 0, 10)), "2026-11-14");
        assert_eq!(night_id_local(at(2026, 11, 15, 5, 45)), "2026-11-14");
    }

    #[test]
    fn a_session_spanning_midnight_is_one_night() {
        let before = night_id_local(at(2026, 11, 14, 23, 50));
        let after = night_id_local(at(2026, 11, 15, 0, 10));
        assert_eq!(before, after);
    }

    #[test]
    fn noon_is_the_boundary() {
        assert_eq!(night_id_local(at(2026, 11, 15, 11, 59)), "2026-11-14");
        assert_eq!(night_id_local(at(2026, 11, 15, 12, 0)), "2026-11-15");
    }

    #[test]
    fn month_and_year_boundaries_roll_back_correctly() {
        assert_eq!(night_id_local(at(2026, 12, 1, 2, 0)), "2026-11-30");
        assert_eq!(night_id_local(at(2027, 1, 1, 3, 0)), "2026-12-31");
    }

    /// A UTC timestamp during a US observing session is already on the next UTC
    /// date, which is exactly the trap `night_id` exists to avoid.
    #[test]
    fn utc_conversion_does_not_leak_the_utc_date() {
        // 2026-11-15T02:00Z is 2026-11-14 21:00 Eastern — night of the 14th.
        let utc = Utc.with_ymd_and_hms(2026, 11, 15, 2, 0, 0).single().unwrap();
        let eastern_view = utc.with_timezone(&eastern());
        assert_eq!(night_id_local(eastern_view), "2026-11-14");
    }
}
