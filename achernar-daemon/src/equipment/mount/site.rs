//! Site and UTC on connect (§3.2 M-b), and the explicit site and UTC
//! writes. Owner: WP1A-4.
//!
//! **Site.** The mount's site is compared with `[site]` within
//! `tolerance_deg` and `elevation_tolerance_m`:
//! - `verify_only` (the default) emits `site_mismatch` and changes nothing;
//! - `to_mount` writes the mount, only while it is not slewing, then reads
//!   back and verifies;
//! - `from_mount` adopts the mount's site for the session (provenance
//!   `mount`);
//! - `none` does nothing.
//!
//! **Time.** UTCDate is read; when |Δ| >= `time_tolerance_s` a
//! `clock_mismatch{delta_s}` is emitted. UTC is written only when
//! `[mount] time_sync = true`, only at connect, before any motion. A
//! NotImplemented write warns and continues (NINA `CheckMountTime`).
//!
//! Connect writes nothing by default (D16).
//!
//! `MountCmd::SetSite(site)` writes the mount and verifies, refused while
//! slewing. A site whose `source` is `Mount` means "adopt the mount's own
//! site for this session" and sends nothing: that is how the spine maps
//! `POST mount/site-sync {direction: "from_mount"}`; `to_mount` maps to
//! `SetSite([site])`.

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};

use crate::config::SiteSyncMode;
use crate::equipment::events::EventKind;
use crate::equipment::ops::{OpKind, OpProgress, Operation};
use crate::equipment::snapshot::MountState;
use crate::equipment::traits::{Clock, MountDriver};
use crate::equipment::types::{
    DeviceError, DeviceResult, ErrorClass, ObsState, OpOutcome, Site, SiteSource,
};

use super::{
    clock_delta, duration_ns, plan, CommandMeta, DynMount, MountShared, ACTIVE_POLL,
    SETTER_DEADLINE, VERIFY_TIMEOUT,
};

/// Members re-read after a site or UTC write.
const SITE_MEMBERS: &[&str] = &["sitelatitude", "sitelongitude", "siteelevation", "utcdate"];

/// What connect-time M-b found and did.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SiteTimeReport {
    /// The session site: `[site]`, or the mount's under `from_mount`.
    pub session_site: Option<Site>,
    pub mount_site: Option<Site>,
    pub site_mismatch: bool,
    pub site_written: bool,
    /// Mount UTC minus host UTC, seconds, after any write.
    pub clock_delta_s: Option<f64>,
    pub clock_mismatch: bool,
    pub utc_written: bool,
    pub warnings: Vec<String>,
}

fn slewing(state: &MountState) -> bool {
    state.slewing.known() == Some(&true)
}

/// A mount site whose elevation is unknown (NaN: the mount does not
/// implement SiteElevation) takes `fallback`'s elevation, else 0 m. Lat and
/// lon are the mount's.
pub fn with_known_elevation(site: Site, fallback: Option<&Site>) -> Site {
    if site.elev_m.is_finite() {
        return site;
    }
    Site {
        elev_m: fallback
            .map(|f| f.elev_m)
            .filter(|e| e.is_finite())
            .unwrap_or(0.0),
        ..site
    }
}

/// Adopt the mount's site for the session (`from_mount`), with provenance
/// `mount`; a warning is returned when its elevation had to be filled in.
fn adopt_mount_site(mount: Site, fallback: Option<&Site>) -> (Site, Option<String>) {
    let adopted = Site {
        source: SiteSource::Mount,
        ..with_known_elevation(mount, fallback)
    };
    let warning = (!mount.elev_m.is_finite()).then(|| {
        format!(
            "the mount does not report SiteElevation; using {} m",
            adopted.elev_m
        )
    });
    (adopted, warning)
}

fn reread(drv: &mut dyn MountDriver) -> Option<MountState> {
    drv.poll(&plan(0, false, SITE_MEMBERS)).ok()?;
    drv.state().ok()
}

/// M-b at connect. Never moves hardware; writes only under `to_mount` or
/// `time_sync`.
pub fn verify_site_and_time(
    drv: &mut dyn MountDriver,
    shared: &MountShared,
    configured: Option<Site>,
) -> SiteTimeReport {
    let settings = shared.settings().clone();
    let mut report = SiteTimeReport {
        session_site: configured,
        ..SiteTimeReport::default()
    };
    let Some(state) = reread(drv).or_else(|| drv.state().ok()) else {
        report
            .warnings
            .push("the mount state could not be read at connect".into());
        return report;
    };
    let mount_site = state.site.known().copied();
    report.mount_site = mount_site;

    // ---- Site ----
    let differs = |a: &Site, b: &Site| {
        a.differs_from(
            b,
            settings.site_tolerance_deg,
            settings.elevation_tolerance_m,
        )
    };
    match (settings.site_sync, configured, mount_site) {
        (SiteSyncMode::None, _, _) => {}
        (SiteSyncMode::FromMount, _, Some(mount)) => {
            let (adopted, warning) = adopt_mount_site(mount, configured.as_ref());
            report.session_site = Some(adopted);
            report.warnings.extend(warning);
        }
        (SiteSyncMode::FromMount, _, None) => report
            .warnings
            .push("[site] sync = from_mount, but the mount reports no site; keeping [site]".into()),
        (SiteSyncMode::VerifyOnly, Some(config), Some(mount)) => {
            if differs(&config, &mount) {
                report.site_mismatch = true;
                shared.emit(
                    EventKind::SiteMismatch {
                        configured: Some(config),
                        mount: Some(mount),
                    },
                    None,
                );
            }
        }
        (SiteSyncMode::ToMount, Some(config), mount) => {
            let needs_write = mount.is_none_or(|m| differs(&config, &m));
            if needs_write && slewing(&state) {
                report
                    .warnings
                    .push("[site] sync = to_mount skipped: the mount is slewing".into());
            } else if needs_write {
                match drv.set_site(&config) {
                    Ok(()) => {
                        report.site_written = true;
                        let after = reread(drv).and_then(|s| s.site.fresh_value().copied());
                        match after {
                            Some(after) if differs(&config, &after) => {
                                report.site_mismatch = true;
                                shared.emit(
                                    EventKind::SiteMismatch {
                                        configured: Some(config),
                                        mount: Some(after),
                                    },
                                    None,
                                );
                                report
                                    .warnings
                                    .push("the mount did not keep the site written to it".into());
                            }
                            Some(after) => report.mount_site = Some(after),
                            None => report
                                .warnings
                                .push("the site was written but cannot be read back".into()),
                        }
                    }
                    Err(err) => report
                        .warnings
                        .push(format!("writing the site failed: {err}")),
                }
            }
        }
        (SiteSyncMode::VerifyOnly | SiteSyncMode::ToMount, None, _) => {}
        (SiteSyncMode::VerifyOnly, Some(_), None) => {}
    }
    shared.set_session_site(report.session_site);

    // ---- Time ----
    let mut delta = clock_delta(&state);
    if let Some(d) = delta {
        if d.abs() >= settings.time_tolerance_s {
            report.clock_mismatch = true;
            shared.emit(EventKind::ClockMismatch { delta_s: d }, None);
            if settings.time_sync && !slewing(&state) {
                match drv.set_utc(shared.clock().now_utc()) {
                    Ok(()) => {
                        report.utc_written = true;
                        delta = reread(drv).as_ref().and_then(clock_delta).or(delta);
                    }
                    Err(err) if err.class == ErrorClass::Unsupported || err.is_unknown_member() => {
                        report
                            .warnings
                            .push("the mount does not implement writing UTCDate; its clock was left as is".into());
                    }
                    Err(err) => report.warnings.push(format!("writing UTC failed: {err}")),
                }
            }
        }
    }
    report.clock_delta_s = delta;
    shared.set_clock_delta_s(delta);
    for warning in &report.warnings {
        tracing::warn!("mount connect: {warning}");
    }
    report
}

/// Write the site and verify, or adopt the mount's site.
pub struct SetSiteOp {
    shared: Arc<MountShared>,
    meta: CommandMeta,
    site: Site,
    sent_mono_ns: u64,
    done: bool,
}

impl SetSiteOp {
    pub fn new(shared: Arc<MountShared>, meta: CommandMeta, site: Site) -> DeviceResult<Self> {
        site.validate().map_err(DeviceError::invalid_value)?;
        Ok(Self {
            shared,
            meta,
            site,
            sent_mono_ns: 0,
            done: false,
        })
    }
}

impl Operation<DynMount> for SetSiteOp {
    fn kind(&self) -> OpKind {
        OpKind::SetSite
    }

    fn deadline(&self) -> Duration {
        SETTER_DEADLINE
    }

    fn start(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> DeviceResult<()> {
        let state = drv.state()?;
        if self.site.source == SiteSource::Mount {
            let Some(mount) = state.site.known().copied() else {
                return Err(DeviceError::unsupported(
                    "the mount reports no site to adopt",
                ));
            };
            let (adopted, warning) = adopt_mount_site(mount, self.shared.session_site().as_ref());
            if let Some(warning) = warning {
                tracing::warn!("mount site adoption: {warning}");
            }
            self.shared.set_session_site(Some(adopted));
            self.done = true;
            return Ok(());
        }
        if slewing(&state) {
            return Err(DeviceError::invalid_operation(
                "the site is not written while the mount is slewing",
            ));
        }
        drv.set_site(&self.site)?;
        self.sent_mono_ns = clock.now_mono_ns();
        Ok(())
    }

    fn step(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> OpProgress {
        if self.done {
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        let Ok(state) = drv.state() else {
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL,
            };
        };
        if state.site.state == ObsState::Unsupported {
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        let settings = self.shared.settings();
        let fresh_since = state.site.mono_ns.is_some_and(|at| at >= self.sent_mono_ns);
        if let (true, Some(read)) = (fresh_since, state.site.fresh_value()) {
            if read.differs_from(
                &self.site,
                settings.site_tolerance_deg,
                settings.elevation_tolerance_m,
            ) {
                self.shared.emit(
                    EventKind::SiteMismatch {
                        configured: Some(self.site),
                        mount: Some(*read),
                    },
                    Some(self.meta.command_id),
                );
                return OpProgress::Done(OpOutcome::Failed(DeviceError::driver(
                    "the mount reports a different site after the write",
                )));
            }
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        if clock.now_mono_ns().saturating_sub(self.sent_mono_ns) >= duration_ns(VERIFY_TIMEOUT) {
            return OpProgress::Done(OpOutcome::Unknown(DeviceError::outcome_unknown(
                "the site was written but could not be read back",
            )));
        }
        OpProgress::Continue {
            next_poll: ACTIVE_POLL,
        }
    }

    fn cancel(&mut self, _drv: &mut DynMount) -> DeviceResult<()> {
        Ok(())
    }
}

/// Write UTCDate and verify the mount clock follows.
pub struct SetUtcOp {
    shared: Arc<MountShared>,
    _meta: CommandMeta,
    utc: DateTime<Utc>,
    sent_utc: Option<DateTime<Utc>>,
    sent_mono_ns: u64,
}

impl SetUtcOp {
    pub fn new(shared: Arc<MountShared>, meta: CommandMeta, utc: DateTime<Utc>) -> Self {
        Self {
            shared,
            _meta: meta,
            utc,
            sent_utc: None,
            sent_mono_ns: 0,
        }
    }
}

impl Operation<DynMount> for SetUtcOp {
    fn kind(&self) -> OpKind {
        OpKind::SetUtc
    }

    fn deadline(&self) -> Duration {
        SETTER_DEADLINE
    }

    fn start(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> DeviceResult<()> {
        if slewing(&drv.state()?) {
            return Err(DeviceError::invalid_operation(
                "the mount clock is not written while the mount is slewing",
            ));
        }
        drv.set_utc(self.utc)?;
        self.sent_utc = Some(clock.now_utc());
        self.sent_mono_ns = clock.now_mono_ns();
        Ok(())
    }

    fn step(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> OpProgress {
        let Ok(state) = drv.state() else {
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL,
            };
        };
        if state.utc.state == ObsState::Unsupported {
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        let fresh_since = state.utc.mono_ns.is_some_and(|at| at >= self.sent_mono_ns);
        if let (true, Some(read), Some(host_at), Some(sent)) = (
            fresh_since,
            state.utc.fresh_value(),
            state.utc.observed_at,
            self.sent_utc,
        ) {
            let expected = self.utc + (host_at - sent);
            let error_s = (*read - expected).num_milliseconds().abs() as f64 / 1000.0;
            self.shared.set_clock_delta_s(clock_delta(&state));
            return if error_s < self.shared.settings().time_tolerance_s.max(1.0) {
                OpProgress::Done(OpOutcome::Succeeded)
            } else {
                OpProgress::Done(OpOutcome::Failed(DeviceError::driver(format!(
                    "the mount clock is {error_s:.1} s from the time written"
                ))))
            };
        }
        if clock.now_mono_ns().saturating_sub(self.sent_mono_ns) >= duration_ns(VERIFY_TIMEOUT) {
            return OpProgress::Done(OpOutcome::Unknown(DeviceError::outcome_unknown(
                "UTCDate was written but could not be read back",
            )));
        }
        OpProgress::Continue {
            next_poll: ACTIVE_POLL,
        }
    }

    fn cancel(&mut self, _drv: &mut DynMount) -> DeviceResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::super::{MountCmd, MountSettings};
    use super::*;
    use crate::equipment::sim::mount::{SimMount, SimMountControl, SimMountProfile};
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::{Device, IdentityConverter};
    use crate::equipment::types::{EqCoord, EqFrame};

    fn far_site() -> Site {
        Site {
            lat_deg: 41.0,
            ..site()
        }
    }

    fn settings(sync: SiteSyncMode) -> MountSettings {
        MountSettings {
            site_sync: sync,
            ..MountSettings::default()
        }
    }

    struct Connected {
        control: SimMountControl,
        shared: Arc<MountShared>,
        report: SiteTimeReport,
    }

    impl Connected {
        fn names(&self) -> Vec<&'static str> {
            self.shared.events().names()
        }

        fn wrote(&self, call: &str) -> bool {
            self.control.calls().iter().any(|c| c == call)
        }
    }

    /// Connect a simulator and run M-b exactly once.
    fn connect_with(
        profile: SimMountProfile,
        settings: MountSettings,
        configured: Option<Site>,
        before: impl FnOnce(&SimMountControl),
    ) -> Connected {
        let clock = Arc::new(ManualClock::new());
        let mut sim = SimMount::new(profile, clock.clone());
        sim.connect(Duration::from_secs(1)).unwrap();
        before(&sim.control());
        let shared = MountShared::new(
            settings,
            Arc::new(IdentityConverter),
            clock,
            sim.identity().connection_id,
        );
        shared.set_capabilities(sim.capabilities().unwrap());
        let report = verify_site_and_time(&mut sim, &shared, configured);
        Connected {
            control: sim.control(),
            shared,
            report,
        }
    }

    fn connect(
        profile: SimMountProfile,
        settings: MountSettings,
        configured: Option<Site>,
    ) -> Connected {
        connect_with(profile, settings, configured, |_| {})
    }

    #[test]
    fn verify_only_reports_a_mismatch_and_writes_nothing() {
        let c = connect(
            SimMountProfile::esp32_like(),
            settings(SiteSyncMode::VerifyOnly),
            Some(far_site()),
        );
        assert!(c.report.site_mismatch && !c.report.site_written);
        assert_eq!(c.report.session_site, Some(far_site()));
        assert!(c.names().contains(&"site_mismatch"));
        assert!(!c.wrote("set_site") && !c.wrote("set_utc"));
        assert_eq!(c.control.site().unwrap().lat_deg, 40.0);
    }

    #[test]
    fn a_matching_site_is_quiet() {
        let c = connect(
            SimMountProfile::esp32_like(),
            settings(SiteSyncMode::VerifyOnly),
            Some(site()),
        );
        assert!(!c.report.site_mismatch);
        assert!(c.names().is_empty(), "{:?}", c.names());
    }

    #[test]
    fn to_mount_writes_and_verifies() {
        let c = connect(
            SimMountProfile::esp32_like(),
            settings(SiteSyncMode::ToMount),
            Some(far_site()),
        );
        assert!(
            c.report.site_written && !c.report.site_mismatch,
            "{:?}",
            c.report
        );
        assert_eq!(c.control.site().unwrap().lat_deg, 41.0);
        assert_eq!(c.report.mount_site.unwrap().lat_deg, 41.0);
    }

    #[test]
    fn to_mount_is_skipped_while_slewing() {
        let c = connect_with(
            SimMountProfile::esp32_like(),
            settings(SiteSyncMode::ToMount),
            Some(far_site()),
            |control| control.external_goto(EqCoord::new(3.0, -10.0, EqFrame::JNowApparent)),
        );
        assert!(!c.report.site_written);
        assert!(c.report.warnings.iter().any(|w| w.contains("slewing")));
        assert!(!c.wrote("set_site"));
    }

    #[test]
    fn from_mount_adopts_the_mount_site_for_the_session() {
        let c = connect(
            SimMountProfile::esp32_like(),
            settings(SiteSyncMode::FromMount),
            Some(far_site()),
        );
        let session = c.report.session_site.unwrap();
        assert_eq!((session.lat_deg, session.source), (40.0, SiteSource::Mount));
        assert_eq!(c.shared.session_site(), Some(session));
        assert!(!c.report.site_written);
    }

    #[test]
    fn a_clock_mismatch_is_reported_and_written_only_with_time_sync() {
        let mut profile = SimMountProfile::esp32_like();
        profile.clock_offset_s = 42.0;
        let c = connect(profile.clone(), MountSettings::default(), Some(site()));
        assert!(c.report.clock_mismatch && !c.report.utc_written);
        assert!((c.report.clock_delta_s.unwrap() - 42.0).abs() < 0.01);
        assert!(c.names().contains(&"clock_mismatch"));
        assert!(
            !c.wrote("set_utc"),
            "connect writes nothing by default (D16)"
        );

        let synced = MountSettings {
            time_sync: true,
            ..MountSettings::default()
        };
        let c = connect(profile, synced, Some(site()));
        assert!(c.report.utc_written);
        assert!(c.report.clock_delta_s.unwrap().abs() < 0.01);
        assert!(c.control.clock_offset_s().abs() < 0.01);
    }

    #[test]
    fn a_small_clock_delta_is_left_alone() {
        let mut profile = SimMountProfile::esp32_like();
        profile.clock_offset_s = 2.0;
        let synced = MountSettings {
            time_sync: true,
            ..MountSettings::default()
        };
        let c = connect(profile, synced, Some(site()));
        assert!(!c.report.clock_mismatch && !c.report.utc_written);
    }

    #[test]
    fn site_commands_write_verify_or_adopt() {
        let mut rig = rig(SimMountProfile::esp32_like());
        assert_eq!(
            rig.run(MountCmd::SetSite(far_site()), operator()).unwrap(),
            OpOutcome::Succeeded
        );
        assert_eq!(rig.sim.control().site().unwrap().lat_deg, 41.0);
        let adopt = Site {
            source: SiteSource::Mount,
            ..site()
        };
        let calls = rig.sim.control().calls().len();
        assert_eq!(
            rig.run(MountCmd::SetSite(adopt), operator()).unwrap(),
            OpOutcome::Succeeded
        );
        assert_eq!(
            rig.sim.control().calls().len(),
            calls,
            "adopting sends nothing"
        );
        assert_eq!(rig.runtime.shared().session_site().unwrap().lat_deg, 41.0);
        assert!(SetSiteOp::new(
            rig.runtime.shared(),
            operator(),
            Site {
                lat_deg: 99.0,
                ..site()
            }
        )
        .is_err());
        let now = rig.clock.now_utc();
        assert_eq!(
            rig.run(MountCmd::SetUtc(now), operator()).unwrap(),
            OpOutcome::Succeeded
        );
    }
}
