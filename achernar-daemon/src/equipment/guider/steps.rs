//! Raw guide steps, phase tags and exposure windows (spec §3.5 G-h, G-i; H9;
//! WP1A-7).
//!
//! - [`sample_from_event`] stores a PHD2 `GuideStep` **unmodified**: PHD2's
//!   signs, `Time`, the event `Timestamp`, the daemon receive time and every
//!   optional field as sent. Nothing missing is filled in.
//! - [`PhaseTracker`] tags each step `guiding`, `dithering`, `settling`,
//!   `paused` or `recovering`. Steps are tagged, never dropped. Dither and
//!   settle tags are bounded on the injected clock, so a dither PHD2 never
//!   settles cannot tag every later step.
//! - [`ClockOffset`] estimates `received - Timestamp` as an EWMA (§2.14) and
//!   reports whether it is stable enough to place steps on the daemon clock.
//! - [`StepHistory`] keeps recent steps plus the markers (dither, settle,
//!   star lost, parameter change, state changes) that decide whether an
//!   exposure's guiding can be attributed (M-1 freeze reasons), and builds
//!   the per-exposure [`GuideWindowStats`] and `guider_state_log` record.

use std::collections::VecDeque;
use std::time::Duration;

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use uuid::Uuid;

use super::phd2_proto::GuideStepEvent;
use super::rms::{window_rms, PhaseFilter};
use crate::equipment::clock::duration_ns;
use crate::equipment::events::{GuideStepRecord, GuiderStateRecord};
use crate::equipment::snapshot::{
    AlgoParam, GuidePhase, GuideStepSample, GuideWindowStats, GuiderAppState,
};

/// Build the stored sample. `None` when PHD2 omitted a required field
/// (`Frame`, `Time`, `dx`, `dy`): such a step is counted as malformed, never
/// completed with invented values.
pub fn sample_from_event(
    event: &GuideStepEvent,
    timestamp: Option<f64>,
    received_at: DateTime<Utc>,
    phase: GuidePhase,
    pixel_scale_arcsec: Option<f64>,
) -> Option<GuideStepSample> {
    let to_i32 = |v: Option<i64>| v.and_then(|x| i32::try_from(x).ok());
    Some(GuideStepSample {
        frame: event.frame?,
        phd2_time_s: event.time?,
        phd2_timestamp: timestamp,
        received_at_utc: received_at,
        mount: event.mount.clone(),
        dx: event.dx?,
        dy: event.dy?,
        ra_raw: event.ra_distance_raw,
        dec_raw: event.dec_distance_raw,
        ra_guide: event.ra_distance_guide,
        dec_guide: event.dec_distance_guide,
        ra_duration_ms: to_i32(event.ra_duration_ms),
        ra_direction: event.ra_direction.clone(),
        dec_duration_ms: to_i32(event.dec_duration_ms),
        dec_direction: event.dec_direction.clone(),
        star_mass: event.star_mass,
        snr: event.snr,
        hfd: event.hfd,
        avg_dist: event.avg_dist,
        ra_limited: event.ra_limited,
        dec_limited: event.dec_limited,
        error_code: to_i32(event.error_code),
        phase,
        pixel_scale_arcsec,
    })
}

/// The `guide_steps` row for a sample.
pub fn step_record(connection_id: Uuid, step: GuideStepSample) -> GuideStepRecord {
    GuideStepRecord {
        connection_id,
        step,
    }
}

// ---------------------------------------------------------------------------
// Phase tags
// ---------------------------------------------------------------------------

/// Decides each step's phase from the events around it (G-h).
///
/// - `dithering`: from a dither (Achernar's request or PHD2's
///   `GuidingDithered`) until a `Settling` event reports the star within the
///   settle tolerance.
/// - `settling`: from then, or from `SettleBegin` after a guide start, until
///   `SettleDone`.
/// - `paused`: between `Paused` and `Resumed`, or while PHD2 reports Paused.
/// - `recovering`: the first step after `StarLost` or `LockPositionLost`,
///   and any step while PHD2 reports LostLock.
/// - `guiding`: everything else.
///
/// The `dithering` and `settling` tags are bounded on the injected clock
/// (all times are `now_mono_ns`), because the event that ends them may never
/// come:
/// - PHD2 runs a settle only for event-server `guide` and `dither`. A dither
///   from its Manual Guide dialog or legacy socket server sends
///   `GuidingDithered` with no `SettleBegin`, `Settling` or `SettleDone`. A
///   dither Achernar did not request therefore lapses after a short grace
///   unless a settle event follows.
/// - An Achernar dither that got no answer may never have happened.
/// - A lost `SettleDone` would otherwise tag every later step.
///
/// Each sign of a settle in progress (Achernar's request, `GuidingDithered`
/// for it, `SettleBegin`, `Settling`) renews the tag for the settle timeout
/// plus margin: Achernar's own settle's, or the default for another
/// client's. A lapsed tag returns to `guiding`.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseTracker {
    dithering: bool,
    settling: bool,
    paused: bool,
    recovering: bool,
    settle_pixels: Option<f64>,
    /// Settle timeout plus margin of Achernar's own settle, in ns.
    own_settle_ns: Option<u64>,
    /// Settle timeout plus margin for a settle Achernar did not request.
    default_settle_ns: u64,
    /// Achernar sent `dither`; its `GuidingDithered` has not arrived.
    dither_requested: bool,
    /// When the `dithering` / `settling` tag lapses (clock mono ns).
    expires_mono_ns: Option<u64>,
}

impl Default for PhaseTracker {
    fn default() -> Self {
        Self::new(Self::DEFAULT_SETTLE_BOUND)
    }
}

impl PhaseTracker {
    /// The `[guider]` default settle timeout (40 s) plus the 10 s margin.
    pub const DEFAULT_SETTLE_BOUND: Duration = Duration::from_secs(50);
    /// How long a `GuidingDithered` Achernar did not request stays tagged
    /// without a settle event, at the least ([`Self::unsolicited_grace`]).
    pub const UNSOLICITED_DITHER_GRACE: Duration = Duration::from_secs(5);

    /// `default_settle_bound`: the settle timeout plus margin applied to a
    /// settle Achernar did not request.
    pub fn new(default_settle_bound: Duration) -> Self {
        Self {
            dithering: false,
            settling: false,
            paused: false,
            recovering: false,
            settle_pixels: None,
            own_settle_ns: None,
            default_settle_ns: duration_ns(default_settle_bound),
            dither_requested: false,
            expires_mono_ns: None,
        }
    }

    /// The grace for a dither Achernar did not request: at least
    /// [`Self::UNSOLICITED_DITHER_GRACE`], and two guide exposures, so a
    /// settle event that comes with the next frame is not missed.
    pub fn unsolicited_grace(exposure_ms: Option<u64>) -> Duration {
        let two_frames = Duration::from_millis(exposure_ms.unwrap_or(0).saturating_mul(2));
        Self::UNSOLICITED_DITHER_GRACE.max(two_frames)
    }

    fn renew(&mut self, now_mono_ns: u64) {
        let bound = self.own_settle_ns.unwrap_or(self.default_settle_ns);
        self.expires_mono_ns = Some(now_mono_ns.saturating_add(bound));
    }

    fn lapsed(&self, now_mono_ns: u64) -> bool {
        self.expires_mono_ns.is_some_and(|t| now_mono_ns >= t)
    }

    /// Achernar sent `dither` with this settle tolerance; `settle_bound`
    /// is its settle timeout plus margin.
    pub fn on_dither_sent(&mut self, settle_pixels: f64, settle_bound: Duration, now_mono_ns: u64) {
        self.dithering = true;
        self.settling = false;
        self.settle_pixels = Some(settle_pixels);
        self.own_settle_ns = Some(duration_ns(settle_bound));
        self.dither_requested = true;
        self.renew(now_mono_ns);
    }

    /// Achernar sent `guide`; its settle starts with `SettleBegin`.
    pub fn on_guide_sent(&mut self, settle_pixels: f64, settle_bound: Duration) {
        self.settle_pixels = Some(settle_pixels);
        self.own_settle_ns = Some(duration_ns(settle_bound));
    }

    /// `GuidingDithered`. For Achernar's own dither PHD2 runs the settle;
    /// for any other the tag lapses after `grace` unless a settle event
    /// follows.
    pub fn on_dithered(&mut self, now_mono_ns: u64, grace: Duration) {
        self.dithering = true;
        self.settling = false;
        if std::mem::take(&mut self.dither_requested) {
            self.renew(now_mono_ns);
        } else {
            self.expires_mono_ns = Some(now_mono_ns.saturating_add(duration_ns(grace)));
        }
    }

    pub fn on_settle_begin(&mut self, now_mono_ns: u64) {
        if !self.dithering {
            self.settling = true;
        }
        self.renew(now_mono_ns);
    }

    /// `Settling`. `default_pixels` is used when the settle was not
    /// Achernar's own.
    pub fn on_settling(&mut self, distance_px: Option<f64>, default_pixels: f64, now_mono_ns: u64) {
        let tolerance = self.settle_pixels.unwrap_or(default_pixels);
        if self.dithering {
            if distance_px.is_some_and(|d| d <= tolerance) {
                self.dithering = false;
                self.settling = true;
            }
        } else {
            self.settling = true;
        }
        self.renew(now_mono_ns);
    }

    /// `SettleDone`, a refused dither, or a settle given up on.
    pub fn on_settle_done(&mut self) {
        self.dithering = false;
        self.settling = false;
        self.settle_pixels = None;
        self.own_settle_ns = None;
        self.dither_requested = false;
        self.expires_mono_ns = None;
    }

    pub fn on_star_lost(&mut self) {
        self.recovering = true;
    }

    pub fn on_paused(&mut self) {
        self.paused = true;
    }

    pub fn on_resumed(&mut self) {
        self.paused = false;
    }

    /// Guiding stopped: nothing carries over to the next run.
    pub fn on_guiding_stopped(&mut self) {
        *self = Self {
            default_settle_ns: self.default_settle_ns,
            ..Self::new(Duration::ZERO)
        };
    }

    /// Drop a `dithering` or `settling` tag whose time ran out. Returns
    /// `true` when a tag was dropped.
    pub fn expire(&mut self, now_mono_ns: u64) -> bool {
        if !self.lapsed(now_mono_ns) {
            return false;
        }
        let tagged = self.dithering || self.settling;
        self.on_settle_done();
        tagged
    }

    /// A dither or settle is in progress at `now_mono_ns`.
    pub fn is_settling(&self, now_mono_ns: u64) -> bool {
        (self.settling || self.dithering) && !self.lapsed(now_mono_ns)
    }

    /// The tag for the step that arrived at `now_mono_ns`, with PHD2's
    /// current state.
    pub fn phase_for_step(&mut self, state: GuiderAppState, now_mono_ns: u64) -> GuidePhase {
        self.expire(now_mono_ns);
        if self.paused || state == GuiderAppState::Paused {
            GuidePhase::Paused
        } else if self.dithering {
            GuidePhase::Dithering
        } else if self.settling {
            GuidePhase::Settling
        } else if self.recovering || state == GuiderAppState::LostLock {
            self.recovering = false;
            GuidePhase::Recovering
        } else {
            GuidePhase::Guiding
        }
    }
}

// ---------------------------------------------------------------------------
// PHD2 clock offset
// ---------------------------------------------------------------------------

/// EWMA of `received - Timestamp` (§2.14) and of its absolute deviation.
/// Steps are placed on the daemon clock as `Timestamp + offset` only while
/// the deviation stays within [`ClockOffset::MAX_JITTER_S`]; otherwise
/// windows fall back to the daemon receive time (§8 risk 4).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClockOffset {
    offset_s: Option<f64>,
    jitter_s: f64,
    samples: u64,
}

impl ClockOffset {
    pub const ALPHA: f64 = 0.1;
    pub const MAX_JITTER_S: f64 = 0.5;
    pub const MIN_SAMPLES: u64 = 3;

    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, timestamp: f64, received_at: DateTime<Utc>) {
        if !timestamp.is_finite() {
            return;
        }
        let received = received_at.timestamp_micros() as f64 / 1e6;
        let sample = received - timestamp;
        self.samples += 1;
        match self.offset_s {
            None => self.offset_s = Some(sample),
            Some(offset) => {
                let deviation = (sample - offset).abs();
                self.jitter_s += Self::ALPHA * (deviation - self.jitter_s);
                self.offset_s = Some(offset + Self::ALPHA * (sample - offset));
            }
        }
    }

    /// The EWMA offset in seconds (published as `clock_offset_s`).
    pub fn offset_s(&self) -> Option<f64> {
        self.offset_s
    }

    pub fn jitter_s(&self) -> f64 {
        self.jitter_s
    }

    pub fn is_stable(&self) -> bool {
        self.samples >= Self::MIN_SAMPLES && self.jitter_s <= Self::MAX_JITTER_S
    }

    /// When a step happened, on the daemon clock.
    pub fn step_time(&self, timestamp: Option<f64>, received_at: DateTime<Utc>) -> DateTime<Utc> {
        match (timestamp, self.offset_s) {
            (Some(ts), Some(offset)) if self.is_stable() && ts.is_finite() => {
                let micros = ((ts + offset) * 1e6).round();
                DateTime::<Utc>::from_timestamp_micros(micros as i64).unwrap_or(received_at)
            }
            _ => received_at,
        }
    }
}

// ---------------------------------------------------------------------------
// History and windows
// ---------------------------------------------------------------------------

/// Something that affects whether a window's guiding can be attributed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Marker {
    Dither,
    SettleBegin,
    SettleDone,
    StarLost,
    LockPositionLost,
    ParamChanged,
    State(GuiderAppState),
}

/// Recent guide steps and markers, bounded by age and count.
#[derive(Debug, Clone)]
pub struct StepHistory {
    retention: ChronoDuration,
    max_steps: usize,
    steps: VecDeque<(DateTime<Utc>, GuideStepSample)>,
    markers: VecDeque<(DateTime<Utc>, Marker)>,
}

impl Default for StepHistory {
    fn default() -> Self {
        Self::new(ChronoDuration::minutes(30), 20_000)
    }
}

/// Inputs for the `guider_state_log` row that are not in the history.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StateRecordContext {
    pub connection_id: Uuid,
    pub algo_params: Vec<AlgoParam>,
    pub aggressiveness_param_ra: Option<String>,
    pub aggressiveness_param_dec: Option<String>,
}

impl StepHistory {
    pub fn new(retention: ChronoDuration, max_steps: usize) -> Self {
        Self {
            retention,
            max_steps: max_steps.max(1),
            steps: VecDeque::new(),
            markers: VecDeque::new(),
        }
    }

    /// Add a step placed at `at` on the daemon clock.
    pub fn push_step(&mut self, at: DateTime<Utc>, step: GuideStepSample) {
        self.steps.push_back((at, step));
        while self.steps.len() > self.max_steps {
            self.steps.pop_front();
        }
        self.prune(at);
    }

    pub fn push_marker(&mut self, at: DateTime<Utc>, marker: Marker) {
        self.markers.push_back((at, marker));
        self.prune(at);
    }

    fn prune(&mut self, now: DateTime<Utc>) {
        let cutoff = now - self.retention;
        while self.steps.front().is_some_and(|(t, _)| *t < cutoff) {
            self.steps.pop_front();
        }
        if self.markers.front().is_none_or(|(t, _)| *t >= cutoff) {
            return;
        }
        // Keep the newest state marker before the cutoff: it is the state at
        // the start of any window that is still answerable.
        let keep = self
            .markers
            .iter()
            .rposition(|(t, m)| *t < cutoff && matches!(m, Marker::State(_)));
        let mut index = 0;
        self.markers.retain(|(t, _)| {
            let kept = *t >= cutoff || Some(index) == keep;
            index += 1;
            kept
        });
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Steps placed inside `[start, end]`.
    pub fn steps_between(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<GuideStepSample> {
        self.steps
            .iter()
            .filter(|(t, _)| *t >= start && *t <= end)
            .map(|(_, s)| s.clone())
            .collect()
    }

    fn markers_between(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> impl Iterator<Item = Marker> + '_ {
        self.markers
            .iter()
            .filter(move |(t, _)| *t >= start && *t <= end)
            .map(|(_, m)| *m)
    }

    /// PHD2's state at `at`: the last state marker at or before it.
    pub fn state_at(&self, at: DateTime<Utc>) -> Option<GuiderAppState> {
        self.markers
            .iter()
            .rev()
            .filter(|(t, _)| *t <= at)
            .find_map(|(_, m)| match m {
                Marker::State(s) => Some(*s),
                _ => None,
            })
    }

    /// G-i statistics for `[start, end]` (dither and settle steps excluded by
    /// default).
    pub fn window_stats(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        filter: &PhaseFilter,
    ) -> GuideWindowStats {
        let steps = self.steps_between(start, end);
        let rms = window_rms(&steps, filter);
        let markers: Vec<Marker> = self.markers_between(start, end).collect();
        let dither_in_window = markers.contains(&Marker::Dither)
            || steps.iter().any(|s| s.phase == GuidePhase::Dithering);
        let settle_in_window = markers
            .iter()
            .any(|m| matches!(m, Marker::SettleBegin | Marker::SettleDone))
            || steps.iter().any(|s| s.phase == GuidePhase::Settling);
        let pulses = steps
            .iter()
            .map(|s| {
                u32::from(s.ra_duration_ms.is_some_and(|d| d > 0))
                    + u32::from(s.dec_duration_ms.is_some_and(|d| d > 0))
            })
            .sum();
        GuideWindowStats {
            state: self.state_at(end),
            rms,
            dither_in_window,
            settle_in_window,
            pulses_n: pulses,
        }
    }

    /// Why M-1 cannot attribute this window's guiding, or `None` when it can.
    pub fn freeze_reason(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        stats: &GuideWindowStats,
    ) -> Option<String> {
        let markers: Vec<Marker> = self.markers_between(start, end).collect();
        let reason = if stats.rms.is_none() {
            "no_guide_steps"
        } else if markers
            .iter()
            .any(|m| matches!(m, Marker::StarLost | Marker::LockPositionLost))
        {
            "star_lost"
        } else if markers.contains(&Marker::ParamChanged) {
            "guide_param_changed"
        } else if stats.dither_in_window {
            "dither_in_window"
        } else if stats.settle_in_window {
            "settle_in_window"
        } else if stats.state != Some(GuiderAppState::Guiding)
            || markers
                .iter()
                .any(|m| matches!(m, Marker::State(s) if *s != GuiderAppState::Guiding))
        {
            "not_guiding"
        } else {
            return None;
        };
        Some(reason.to_string())
    }

    /// One `guider_state_log` row for an exposure window (H9).
    pub fn state_record(
        &self,
        ctx: &StateRecordContext,
        frame_id: Option<String>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        now: DateTime<Utc>,
    ) -> GuiderStateRecord {
        let filter = PhaseFilter::default();
        let stats = self.window_stats(start, end, &filter);
        let freeze_reason = self.freeze_reason(start, end, &stats);
        let counted: Vec<f64> = self
            .steps_between(start, end)
            .iter()
            .filter(|s| filter.includes(s.phase))
            .filter_map(|s| s.snr)
            .collect();
        let snr = (!counted.is_empty()).then(|| counted.iter().sum::<f64>() / counted.len() as f64);
        let param = |name: &Option<String>, axis| {
            name.as_ref().and_then(|name| {
                ctx.algo_params
                    .iter()
                    .find(|p| p.axis == axis && &p.name == name)
                    .map(|p| p.value)
            })
        };
        GuiderStateRecord {
            timestamp: now,
            session_id: None,
            connection_id: ctx.connection_id,
            frame_id,
            window_start: start,
            window_end: end,
            rms: stats.rms,
            ra_aggressiveness: param(
                &ctx.aggressiveness_param_ra,
                crate::equipment::commands::Axis::Ra,
            ),
            dec_aggressiveness: param(
                &ctx.aggressiveness_param_dec,
                crate::equipment::commands::Axis::Dec,
            ),
            guide_star_snr: snr,
            dither_active: stats.dither_in_window,
            algorithm_params: ctx.algo_params.clone(),
            executed_arm_ra: None,
            executed_arm_dec: None,
            policy_version: None,
            freeze_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::Axis;
    use chrono::TimeZone;

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 10, 26, 22, 0, 0).unwrap()
    }

    fn sample(frame: u64, ra: f64, dec: f64, phase: GuidePhase) -> GuideStepSample {
        GuideStepSample {
            frame,
            ra_raw: Some(ra),
            dec_raw: Some(dec),
            phase,
            pixel_scale_arcsec: Some(2.0),
            snr: Some(20.0 + frame as f64),
            ra_duration_ms: Some(100),
            ..GuideStepSample::default()
        }
    }

    #[test]
    fn samples_are_stored_unmodified_and_incomplete_steps_are_refused() {
        let event = GuideStepEvent {
            frame: Some(17),
            time: Some(33.5),
            mount: Some("Mount".into()),
            dx: Some(-0.5),
            dy: Some(0.25),
            ra_distance_raw: Some(-0.4),
            dec_distance_raw: Some(0.3),
            ra_duration_ms: Some(240),
            ra_direction: Some("West".into()),
            error_code: Some(0),
            ..GuideStepEvent::default()
        };
        let at = t0();
        let s =
            sample_from_event(&event, Some(1.7e9), at, GuidePhase::Settling, Some(1.48)).unwrap();
        assert_eq!(s.frame, 17);
        assert_eq!(s.phd2_time_s, 33.5);
        assert_eq!(s.phd2_timestamp, Some(1.7e9));
        assert_eq!(s.received_at_utc, at);
        assert_eq!((s.dx, s.dy), (-0.5, 0.25));
        assert_eq!(s.ra_raw, Some(-0.4));
        assert_eq!(s.ra_duration_ms, Some(240));
        assert_eq!(s.dec_duration_ms, None);
        assert_eq!(s.ra_limited, None);
        assert_eq!(s.error_code, Some(0));
        assert_eq!(s.phase, GuidePhase::Settling);
        assert_eq!(s.pixel_scale_arcsec, Some(1.48));
        let rec = step_record(Uuid::nil(), s.clone());
        assert_eq!(rec.step, s);

        let missing = GuideStepEvent { dy: None, ..event };
        assert!(sample_from_event(&missing, None, at, GuidePhase::Guiding, None).is_none());
    }

    const S: u64 = 1_000_000_000;

    #[test]
    fn phases_follow_dither_settle_pause_and_star_loss() {
        let mut p = PhaseTracker::default();
        let guiding = GuiderAppState::Guiding;
        let bound = Duration::from_secs(50);
        assert_eq!(p.phase_for_step(guiding, 0), GuidePhase::Guiding);
        p.on_dither_sent(1.5, bound, 0);
        p.on_dithered(S, PhaseTracker::UNSOLICITED_DITHER_GRACE);
        assert_eq!(p.phase_for_step(guiding, 2 * S), GuidePhase::Dithering);
        p.on_settling(Some(3.0), 1.5, 3 * S);
        assert_eq!(p.phase_for_step(guiding, 4 * S), GuidePhase::Dithering);
        p.on_settling(Some(1.2), 1.5, 5 * S);
        assert_eq!(p.phase_for_step(guiding, 6 * S), GuidePhase::Settling);
        assert!(p.is_settling(6 * S));
        p.on_settle_done();
        assert!(!p.is_settling(7 * S));
        assert_eq!(p.phase_for_step(guiding, 7 * S), GuidePhase::Guiding);
        // A guide start settles without a dither.
        p.on_guide_sent(1.0, bound);
        p.on_settle_begin(8 * S);
        assert_eq!(p.phase_for_step(guiding, 9 * S), GuidePhase::Settling);
        p.on_settle_done();
        p.on_star_lost();
        assert_eq!(p.phase_for_step(guiding, 10 * S), GuidePhase::Recovering);
        assert_eq!(p.phase_for_step(guiding, 11 * S), GuidePhase::Guiding);
        assert_eq!(
            p.phase_for_step(GuiderAppState::LostLock, 12 * S),
            GuidePhase::Recovering
        );
        p.on_paused();
        assert_eq!(p.phase_for_step(guiding, 13 * S), GuidePhase::Paused);
        p.on_resumed();
        assert_eq!(p.phase_for_step(guiding, 14 * S), GuidePhase::Guiding);
        assert_eq!(
            p.phase_for_step(GuiderAppState::Paused, 15 * S),
            GuidePhase::Paused
        );
        p.on_dithered(16 * S, PhaseTracker::UNSOLICITED_DITHER_GRACE);
        p.on_guiding_stopped();
        assert_eq!(p, PhaseTracker::default());
        // The configured default settle bound survives a stop.
        let mut q = PhaseTracker::new(Duration::from_secs(70));
        q.on_guiding_stopped();
        assert_eq!(q, PhaseTracker::new(Duration::from_secs(70)));
    }

    /// PHD2's Manual Guide dialog and legacy socket server dither without a
    /// settle: `GuidingDithered` and nothing else. The tag must not stick.
    #[test]
    fn a_dither_nobody_settles_returns_to_guiding_after_the_grace() {
        let guiding = GuiderAppState::Guiding;
        let grace = PhaseTracker::unsolicited_grace(Some(2000));
        assert_eq!(grace, Duration::from_secs(5));
        assert_eq!(
            PhaseTracker::unsolicited_grace(Some(4000)),
            Duration::from_secs(8),
            "two guide frames"
        );
        assert_eq!(PhaseTracker::unsolicited_grace(None), grace);
        let mut p = PhaseTracker::default();
        p.on_dithered(10 * S, grace);
        assert_eq!(p.phase_for_step(guiding, 12 * S), GuidePhase::Dithering);
        assert!(p.is_settling(14 * S));
        assert!(!p.is_settling(15 * S), "lapsed, before any step");
        assert_eq!(p.phase_for_step(guiding, 15 * S), GuidePhase::Guiding);
        for i in 0..1000 {
            assert_eq!(p.phase_for_step(guiding, (16 + i) * S), GuidePhase::Guiding);
        }
        assert!(!p.expire(2000 * S), "nothing left to drop");

        // A settle event within the grace makes it a settle PHD2 runs: the
        // tag then lasts for the default settle bound after the last one.
        let mut p = PhaseTracker::default();
        p.on_dithered(0, grace);
        p.on_settle_begin(3 * S);
        p.on_settling(Some(4.0), 1.5, 20 * S);
        assert_eq!(p.phase_for_step(guiding, 30 * S), GuidePhase::Dithering);
        assert_eq!(p.phase_for_step(guiding, 69 * S), GuidePhase::Dithering);
        // No SettleDone ever came: dropped 50 s after the last Settling.
        assert!(p.is_settling(69 * S));
        assert!(p.expire(70 * S));
        assert_eq!(p.phase_for_step(guiding, 71 * S), GuidePhase::Guiding);
    }

    #[test]
    fn an_unanswered_dither_and_a_lost_settle_done_lapse_on_the_clock() {
        let guiding = GuiderAppState::Guiding;
        // Achernar's dither got no answer, so PHD2 may never have dithered.
        let mut p = PhaseTracker::default();
        p.on_dither_sent(1.5, Duration::from_secs(30), 0);
        assert_eq!(p.phase_for_step(guiding, 29 * S), GuidePhase::Dithering);
        assert_eq!(p.phase_for_step(guiding, 30 * S), GuidePhase::Guiding);

        // Achernar's own dither keeps its settle bound through
        // GuidingDithered (not the short grace).
        let mut p = PhaseTracker::default();
        p.on_dither_sent(1.5, Duration::from_secs(30), 0);
        p.on_dithered(S, PhaseTracker::UNSOLICITED_DITHER_GRACE);
        assert_eq!(p.phase_for_step(guiding, 20 * S), GuidePhase::Dithering);
        assert_eq!(p.phase_for_step(guiding, 31 * S), GuidePhase::Guiding);

        // A guide start whose SettleDone was lost.
        let mut p = PhaseTracker::default();
        p.on_guide_sent(1.5, Duration::from_secs(20));
        p.on_settle_begin(0);
        p.on_settling(Some(0.5), 1.5, 5 * S);
        assert_eq!(p.phase_for_step(guiding, 24 * S), GuidePhase::Settling);
        assert_eq!(p.phase_for_step(guiding, 25 * S), GuidePhase::Guiding);
        // The next run starts clean.
        p.on_settle_begin(30 * S);
        assert_eq!(p.phase_for_step(guiding, 31 * S), GuidePhase::Settling);
        assert!(!p.is_settling(30 * S + 50 * S), "default bound again");
    }

    #[test]
    fn the_clock_offset_is_an_ewma_with_a_jitter_guard() {
        let mut c = ClockOffset::new();
        let base = 1_800_000_000.0;
        let received =
            |s: f64| DateTime::<Utc>::from_timestamp_micros(((base + s) * 1e6) as i64).unwrap();
        // PHD2's clock runs 2 s behind ours, with 50 ms of latency.
        for i in 0..10 {
            let ts = base + i as f64 - 2.0;
            c.update(ts, received(i as f64 + 0.05));
        }
        let offset = c.offset_s().unwrap();
        assert!((offset - 2.05).abs() < 1e-6, "{offset}");
        assert!(c.is_stable());
        let placed = c.step_time(Some(base + 20.0 - 2.0), received(99.0));
        assert!((placed - received(20.05)).num_milliseconds().abs() <= 1);
        // Large jitter: fall back to the receive time.
        for i in 0..20 {
            let lag = if i % 2 == 0 { 0.0 } else { 3.0 };
            c.update(base + i as f64, received(i as f64 + lag));
        }
        assert!(!c.is_stable());
        assert_eq!(c.step_time(Some(base), received(7.0)), received(7.0));
        assert_eq!(
            ClockOffset::new().step_time(Some(base), received(1.0)),
            received(1.0)
        );
    }

    #[test]
    fn windows_report_rms_flags_state_and_freeze_reasons() {
        let mut h = StepHistory::default();
        let at = |s: i64| t0() + ChronoDuration::seconds(s);
        h.push_marker(at(0), Marker::State(GuiderAppState::Guiding));
        for i in 1..=10 {
            h.push_step(
                at(i),
                sample(i as u64, 0.1 * i as f64, -0.05, GuidePhase::Guiding),
            );
        }
        // A clean window.
        let filter = PhaseFilter::default();
        let stats = h.window_stats(at(1), at(5), &filter);
        assert_eq!(stats.rms.unwrap().steps_n, 5);
        assert_eq!(stats.state, Some(GuiderAppState::Guiding));
        assert!(!stats.dither_in_window && !stats.settle_in_window);
        assert_eq!(stats.pulses_n, 5);
        assert_eq!(h.freeze_reason(at(1), at(5), &stats), None);

        // A dither inside the window.
        h.push_marker(at(11), Marker::Dither);
        h.push_step(at(12), sample(11, 4.0, 4.0, GuidePhase::Dithering));
        h.push_step(at(13), sample(12, 1.0, 1.0, GuidePhase::Settling));
        h.push_marker(at(14), Marker::SettleDone);
        h.push_step(at(15), sample(13, 0.1, 0.1, GuidePhase::Guiding));
        let stats = h.window_stats(at(10), at(15), &filter);
        assert!(stats.dither_in_window && stats.settle_in_window);
        assert_eq!(
            stats.rms.unwrap().steps_n,
            2,
            "dither and settle steps excluded"
        );
        assert_eq!(
            h.freeze_reason(at(10), at(15), &stats).as_deref(),
            Some("dither_in_window")
        );
        // Star lost outranks the dither.
        h.push_marker(at(16), Marker::StarLost);
        h.push_step(at(17), sample(14, 0.2, 0.2, GuidePhase::Recovering));
        let stats = h.window_stats(at(10), at(17), &filter);
        assert_eq!(
            h.freeze_reason(at(10), at(17), &stats).as_deref(),
            Some("star_lost")
        );
        // No steps at all.
        let empty = h.window_stats(at(100), at(200), &filter);
        assert!(empty.rms.is_none());
        assert_eq!(
            h.freeze_reason(at(100), at(200), &empty).as_deref(),
            Some("no_guide_steps")
        );
        // A parameter change.
        h.push_marker(at(18), Marker::ParamChanged);
        h.push_step(at(19), sample(15, 0.1, 0.1, GuidePhase::Guiding));
        let stats = h.window_stats(at(18), at(19), &filter);
        assert_eq!(
            h.freeze_reason(at(18), at(19), &stats).as_deref(),
            Some("guide_param_changed")
        );
        // Guiding stopped mid-window.
        h.push_marker(at(20), Marker::State(GuiderAppState::Stopped));
        let stats = h.window_stats(at(19), at(21), &filter);
        assert_eq!(stats.state, Some(GuiderAppState::Stopped));
        assert_eq!(
            h.freeze_reason(at(19), at(21), &stats).as_deref(),
            Some("not_guiding")
        );
    }

    #[test]
    fn state_records_carry_rms_snr_and_executed_parameters() {
        let mut h = StepHistory::default();
        let at = |s: i64| t0() + ChronoDuration::seconds(s);
        h.push_marker(at(0), Marker::State(GuiderAppState::Guiding));
        for i in 1..=4 {
            h.push_step(at(i), sample(i as u64, 0.2, -0.2, GuidePhase::Guiding));
        }
        let ctx = StateRecordContext {
            connection_id: Uuid::new_v4(),
            algo_params: vec![
                AlgoParam {
                    axis: Axis::Ra,
                    name: "aggression".into(),
                    value: 0.7,
                },
                AlgoParam {
                    axis: Axis::Dec,
                    name: "aggressiveness".into(),
                    value: 0.5,
                },
            ],
            aggressiveness_param_ra: Some("aggression".into()),
            aggressiveness_param_dec: Some("aggressiveness".into()),
        };
        let rec = h.state_record(&ctx, Some("frame-1".into()), at(1), at(4), at(5));
        assert_eq!(rec.connection_id, ctx.connection_id);
        assert_eq!(rec.frame_id.as_deref(), Some("frame-1"));
        assert_eq!(rec.ra_aggressiveness, Some(0.7));
        assert_eq!(rec.dec_aggressiveness, Some(0.5));
        assert_eq!(rec.guide_star_snr, Some(22.5));
        assert_eq!(rec.rms.unwrap().steps_n, 4);
        assert!(!rec.dither_active);
        assert_eq!(rec.freeze_reason, None);
        assert_eq!(rec.algorithm_params.len(), 2);
    }

    #[test]
    fn history_is_bounded_by_age_and_count_but_keeps_the_last_state() {
        let mut h = StepHistory::new(ChronoDuration::seconds(60), 5);
        let at = |s: i64| t0() + ChronoDuration::seconds(s);
        h.push_marker(at(0), Marker::State(GuiderAppState::Guiding));
        for i in 0..10 {
            h.push_step(at(i), sample(i as u64, 0.0, 0.0, GuidePhase::Guiding));
        }
        assert_eq!(h.len(), 5);
        h.push_step(at(500), sample(99, 0.0, 0.0, GuidePhase::Guiding));
        assert_eq!(h.len(), 1);
        assert_eq!(h.state_at(at(500)), Some(GuiderAppState::Guiding));
    }
}
