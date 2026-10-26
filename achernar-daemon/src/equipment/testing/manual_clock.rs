//! A clock that moves only when a test advances it (D22). Every timing rule
//! (deadlines, settle waits, cadence, staleness, window assembly) is tested
//! deterministically with it, without sleeps, so Windows CI is not exposed to
//! wall-clock races.

use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use chrono::{DateTime, TimeZone, Utc};

use crate::equipment::clock::duration_ns;
use crate::equipment::traits::Clock;

#[derive(Debug, Clone, Copy)]
struct ManualState {
    utc: DateTime<Utc>,
    mono_ns: u64,
}

/// Manual UTC and monotonic time. `advance` moves both in lockstep;
/// `set_utc` steps only the wall clock (a host clock correction).
#[derive(Debug)]
pub struct ManualClock {
    state: Mutex<ManualState>,
    changed: Condvar,
}

impl Default for ManualClock {
    fn default() -> Self {
        Self::new()
    }
}

impl ManualClock {
    /// 2026-01-01T00:00:00Z, monotonic 0.
    pub fn new() -> Self {
        Self::starting_at(Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap())
    }

    pub fn starting_at(utc: DateTime<Utc>) -> Self {
        Self {
            state: Mutex::new(ManualState { utc, mono_ns: 0 }),
            changed: Condvar::new(),
        }
    }

    fn state(&self) -> MutexGuard<'_, ManualState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn advance(&self, by: Duration) {
        let mut state = self.state();
        state.mono_ns = state.mono_ns.saturating_add(duration_ns(by));
        if let Ok(delta) = chrono::Duration::from_std(by) {
            state.utc += delta;
        }
        drop(state);
        self.changed.notify_all();
    }

    pub fn advance_ms(&self, ms: u64) {
        self.advance(Duration::from_millis(ms));
    }

    /// Step the wall clock only; monotonic time is unchanged.
    pub fn set_utc(&self, utc: DateTime<Utc>) {
        self.state().utc = utc;
        self.changed.notify_all();
    }

    /// Block the calling (test) thread until monotonic time reaches
    /// `mono_ns`, or until `real_timeout` of real time passes. Returns true
    /// when the time was reached.
    pub fn wait_until(&self, mono_ns: u64, real_timeout: Duration) -> bool {
        let deadline = Instant::now() + real_timeout;
        let mut state = self.state();
        while state.mono_ns < mono_ns {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let (guard, _) = self
                .changed
                .wait_timeout(state, deadline - now)
                .unwrap_or_else(|e| e.into_inner());
            state = guard;
        }
        true
    }
}

impl Clock for ManualClock {
    fn now_utc(&self) -> DateTime<Utc> {
        self.state().utc
    }

    fn now_mono_ns(&self) -> u64 {
        self.state().mono_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn manual_clock_round_trips_utc_and_monotonic_time() {
        let clock = ManualClock::new();
        let start = clock.now_utc();
        assert_eq!(clock.now_mono_ns(), 0);
        clock.advance_ms(1500);
        assert_eq!(clock.now_mono_ns(), 1_500_000_000);
        assert_eq!(
            clock.now_utc() - start,
            chrono::Duration::milliseconds(1500)
        );

        // A wall-clock step does not move monotonic time.
        let stepped = start + chrono::Duration::hours(1);
        clock.set_utc(stepped);
        assert_eq!(clock.now_utc(), stepped);
        assert_eq!(clock.now_mono_ns(), 1_500_000_000);
        clock.advance(Duration::from_secs(2));
        assert_eq!(clock.now_utc(), stepped + chrono::Duration::seconds(2));
        assert_eq!(clock.now_mono_ns(), 3_500_000_000);
    }

    #[test]
    fn waiters_wake_when_another_thread_advances() {
        let clock = Arc::new(ManualClock::new());
        assert!(!clock.wait_until(1, Duration::from_millis(5)));
        let advancer = Arc::clone(&clock);
        let thread = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            advancer.advance(Duration::from_millis(250));
        });
        assert!(clock.wait_until(250_000_000, Duration::from_secs(5)));
        thread.join().unwrap();
        let as_dyn: &dyn Clock = clock.as_ref();
        assert_eq!(as_dyn.now_mono_ns(), 250_000_000);
    }
}
