//! Production time source (D22). Frozen by WP0.
//!
//! Workers, gate timeouts, settle waits and window assembly take a
//! [`Clock`] trait object; production code uses [`SystemClock`], tests use
//! `testing::manual_clock::ManualClock`. Monotonic time is nanoseconds since
//! daemon start, shared by every component so ring buffers, events and
//! observations can be compared directly.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};

use super::traits::Clock;

static DAEMON_START: OnceLock<Instant> = OnceLock::new();

/// The process-wide monotonic origin. The first call fixes it.
pub fn daemon_start() -> Instant {
    *DAEMON_START.get_or_init(Instant::now)
}

/// Wall clock for UTC, `Instant` since [`daemon_start`] for monotonic time.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl SystemClock {
    pub fn shared() -> Arc<dyn Clock> {
        Arc::new(SystemClock)
    }
}

impl Clock for SystemClock {
    fn now_utc(&self) -> DateTime<Utc> {
        Utc::now()
    }

    fn now_mono_ns(&self) -> u64 {
        u64::try_from(daemon_start().elapsed().as_nanos()).unwrap_or(u64::MAX)
    }
}

/// Monotonic nanoseconds for a duration, saturating.
pub fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Time left until a monotonic deadline, or zero once it has passed.
pub fn remaining(clock: &dyn Clock, deadline_mono_ns: u64) -> Duration {
    Duration::from_nanos(deadline_mono_ns.saturating_sub(clock.now_mono_ns()))
}

/// True once the monotonic deadline has passed.
pub fn expired(clock: &dyn Clock, deadline_mono_ns: u64) -> bool {
    clock.now_mono_ns() >= deadline_mono_ns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_clock_is_monotonic_from_daemon_start() {
        let clock = SystemClock;
        let a = clock.now_mono_ns();
        let b = clock.now_mono_ns();
        assert!(b >= a);
        let deadline = a + duration_ns(Duration::from_secs(3600));
        assert!(!expired(&clock, deadline));
        assert!(remaining(&clock, deadline) > Duration::from_secs(3500));
        assert!(expired(&clock, a));
        assert_eq!(remaining(&clock, a), Duration::ZERO);
    }
}
