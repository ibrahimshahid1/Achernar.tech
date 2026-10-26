import { useState } from "react";
import { achernarApi, DRIVE_RATES, type MountDescriptor } from "../../api/achernar";
import {
    ageText,
    alignmentLabel,
    dms,
    frameLabel,
    hms,
    mountActionGates,
    mountNativeFrame,
    num,
    parseDec,
    parseRa,
    qualificationLabel,
    qualificationTone,
    UNKNOWN,
    yesNo,
    type ActionGate,
    type RunCommand,
    type TargetFrame,
} from "../../lib/equipmentView";
import type { DeviceHealthEntry, DriveRate, MountBlock } from "../../types/achernar";
import "../equipmentPanels.css";

/** The guiding status panel lives with the guider; kept here under its old name. */
export { default as GuidingPanel } from "../guider";

function Row({ label, value, tone }: { label: string; value: string; tone?: string }) {
    return <div className="eq-row"><dt>{label}</dt><dd className={tone ? `tone-${tone}` : undefined}>{value}</dd></div>;
}

/** Mount status panel: pointing in both frames, geometry, state and site (§2.13). */
export function MountStatus({ mount, descriptor }: { mount: MountBlock | null | undefined; descriptor?: MountDescriptor | null }) {
    if (!mount) return <p className="eq-empty">No mount status reported.</p>;
    const site = mount.site;
    return <dl className="eq-grid" aria-label="Mount status">
        <Row label={`RA (${frameLabel(mount.frame)})`} value={hms(mount.ra)} />
        <Row label={`Dec (${frameLabel(mount.frame)})`} value={dms(mount.dec)} />
        <Row label="RA (J2000)" value={hms(mount.ra_j2000)} />
        <Row label="Dec (J2000)" value={dms(mount.dec_j2000)} />
        <Row label="Altitude" value={num(mount.alt, 2, "°")} />
        <Row label="Azimuth" value={num(mount.az, 2, "°")} />
        <Row label="LST" value={hms(mount.lst_h)} />
        <Row label="Hour angle" value={num(mount.ha_h, 3, " h")} />
        <Row label="Pier side" value={mount.pier_side ?? UNKNOWN} />
        <Row label="Alignment" value={alignmentLabel(mount.alignment_mode)} />
        <Row label="Meridian flip" value={mount.can_flip == null ? UNKNOWN : mount.can_flip ? "capable" : "not capable"} />
        <Row label="Slewing" value={yesNo(mount.slewing)} tone={mount.slewing ? "warn" : undefined} />
        <Row label="Tracking" value={mount.tracking == null ? UNKNOWN : mount.tracking ? `on (${mount.tracking_rate ?? "rate unknown"})` : "off"} />
        <Row label="Parked" value={yesNo(mount.parked)} />
        <Row label="At home" value={yesNo(mount.at_home)} />
        <Row label="Pulse guiding" value={yesNo(mount.is_pulse_guiding)} />
        <Row label="Guide rates" value={mount.guide_rate_ra_arcsec_s == null && mount.guide_rate_dec_arcsec_s == null ? UNKNOWN
            : `RA ${num(mount.guide_rate_ra_arcsec_s, 2)} / Dec ${num(mount.guide_rate_dec_arcsec_s, 2)} "/s`} />
        <Row label="Site" value={site ? `${num(site.lat_deg, 4, "°")}, ${num(site.lon_deg, 4, "°")} E, ${num(site.elev_m, 0, " m")}${site.source ? ` (${site.source})` : ""}` : "site unknown"} />
        <Row label="Clock offset" value={num(mount.clock_delta_s, 1, " s")} />
        <Row label="Reading age" value={ageText(mount.age_ms)} />
        {descriptor && <Row label="Driver" value={[descriptor.identity.driver_info, descriptor.identity.driver_version].filter(Boolean).join(" ") || descriptor.identity.driver} />}
        {descriptor && <Row label="Interface" value={descriptor.capabilities.interface_version == null ? UNKNOWN : `v${descriptor.capabilities.interface_version}${descriptor.capabilities.device_state ? ", DeviceState" : ""}`} />}
    </dl>;
}

function GatedButton({ gate, onClick, children, className = "btn" }: { gate: ActionGate & { disabled?: boolean }; onClick: () => void; children: string; className?: string }) {
    return <button type="button" className={className} disabled={!gate.enabled || gate.disabled} title={gate.reason} onClick={onClick}>{children}</button>;
}

/**
 * Mount controls (§2.12): abort, tracking, park, unpark, home, GoTo, sync,
 * center and site sync. GoTo is gated on CanSlewAsync and abort on nothing
 * (O10, D19); every command still passes the daemon's gate. Coordinates
 * are J2000 (catalogue and plate-solve frame) and the daemon converts them;
 * while the native frame is unknown the operator may instead send them as
 * mount-native, with a warning that nothing converts them.
 *
 * `abortable` comes from the console's last known status, not the live
 * poll, so a failed or late poll never takes the abort away (D19).
 */
export function MountControls({ mount, health, descriptor, online, abortable, busy, run }: {
    mount: MountBlock | null | undefined;
    health?: DeviceHealthEntry | null;
    descriptor?: MountDescriptor | null;
    online: boolean;
    abortable: boolean;
    busy: boolean;
    run: RunCommand;
}) {
    const [ra, setRa] = useState("");
    const [dec, setDec] = useState("");
    const [rate, setRate] = useState<DriveRate>("sidereal");
    const [targetFrame, setTargetFrame] = useState<TargetFrame>("j2000");
    const connected = online && mount?.connected === true;
    const gates = mountActionGates(mount, descriptor, targetFrame);
    const native = mountNativeFrame(mount, descriptor);
    const raValue = parseRa(ra);
    const decValue = parseDec(dec);
    const target = raValue != null && decValue != null ? { ra: raValue, dec: decValue, frame: targetFrame } : null;
    const locked = busy || !connected;
    const rates = descriptor?.capabilities.tracking_rates?.length ? descriptor.capabilities.tracking_rates : DRIVE_RATES;
    const disabled = (gate: ActionGate) => ({ ...gate, disabled: locked });
    const frameName = targetFrame === "j2000" ? "J2000" : "mount frame";
    const absentText = !online
        ? abortable ? "Live mount status is unavailable; abort is still sent to the daemon." : "Live mount status is unavailable."
        : health?.state === "lost" ? "The mount is lost; abort still reaches its abort path." : "No mount connected.";
    return <div className="eq-panel">
        <div className="eq-toolbar">
            <button type="button" className="btn eq-abort" disabled={!abortable} title="AbortSlew out of band; never gated (D19)"
                onClick={() => void run("mount.abort", () => achernarApi.abortMount())}>Abort mount motion</button>
            {descriptor?.qualification && <span className={`qual-badge tone-${qualificationTone(descriptor.qualification)}`}>{qualificationLabel(descriptor.qualification)}</span>}
            {descriptor?.identity.setup_url && <a className="eq-link" href={descriptor.identity.setup_url} target="_blank" rel="noreferrer">Driver setup page</a>}
        </div>
        {!connected && <p className="eq-empty">{absentText}</p>}
        {connected && <>
            <div className="eq-toolbar">
                <GatedButton gate={disabled(gates.tracking)} onClick={() => void run("mount.set_tracking", () => achernarApi.setTracking(true))}>Start tracking</GatedButton>
                <GatedButton gate={disabled(gates.tracking)} onClick={() => void run("mount.set_tracking", () => achernarApi.setTracking(false))}>Stop tracking</GatedButton>
                <label className="eq-inline">Rate <select className="cam-select" value={rate} disabled={locked} onChange={(event) => setRate(event.target.value as DriveRate)}>
                    {rates.map((entry) => <option key={entry} value={entry}>{entry}</option>)}
                </select></label>
                <GatedButton gate={disabled(gates.tracking)} onClick={() => void run("mount.set_tracking_rate", () => achernarApi.setMountTrackingRate(rate))}>Set rate</GatedButton>
            </div>
            <div className="eq-toolbar">
                <GatedButton gate={disabled(gates.park)} onClick={() => void run("mount.park", () => achernarApi.parkMount())}>Park</GatedButton>
                <GatedButton gate={disabled(gates.unpark)} onClick={() => void run("mount.unpark", () => achernarApi.unparkMount())}>Unpark</GatedButton>
                <GatedButton gate={disabled(gates.home)} onClick={() => void run("mount.find_home", () => achernarApi.findMountHome())}>Find home</GatedButton>
            </div>
            <div className="eq-toolbar">
                <label className="eq-inline">Target frame <select className="cam-select" value={targetFrame} disabled={locked}
                    onChange={(event) => setTargetFrame(event.target.value === "mount" ? "mount" : "j2000")}>
                    <option value="j2000">J2000 (converted)</option>
                    <option value="mount">Mount-native (unconverted)</option>
                </select></label>
                <label className="eq-inline">RA {frameName} <input className="cam-num" value={ra} placeholder="05:35:17.3 or 5.588" disabled={locked} onChange={(event) => setRa(event.target.value)} /></label>
                <label className="eq-inline">Dec {frameName} <input className="cam-num" value={dec} placeholder="-05:23:28 or -5.391" disabled={locked} onChange={(event) => setDec(event.target.value)} /></label>
                <GatedButton gate={disabled({ ...gates.goto, enabled: gates.goto.enabled && target != null })} onClick={() => target && void run("mount.slew", () => achernarApi.slewMount(target))}>Slew</GatedButton>
                <GatedButton gate={disabled({ ...gates.sync, enabled: gates.sync.enabled && target != null })} onClick={() => target && void run("mount.sync", () => achernarApi.syncMount(target))}>Sync</GatedButton>
                <GatedButton gate={disabled({ ...gates.center, enabled: gates.center.enabled && target != null })} onClick={() => target && void run("mount.center", () => achernarApi.centerMount(target))}>Center</GatedButton>
            </div>
            {(ra.trim() || dec.trim()) && !target && <p className="eq-hint tone-warn">Enter RA as hours (0 to 24) and Dec as degrees (-90 to +90), decimal or h:m:s / d:m:s.</p>}
            {!gates.frame.enabled && <p className="eq-hint tone-warn" role="status">{gates.frame.reason}</p>}
            {targetFrame === "mount" && <p className="eq-hint tone-warn" role="status">Mount-native: the coordinates are sent as typed, with frame &quot;mount&quot;, and nothing converts them.{native && native !== "unknown"
                ? ` They must already be in the mount's own frame, ${frameLabel(native)}.`
                : " They must already be in the mount's own frame, which the daemon does not know: a J2000 catalogue position sent to a JNow mount is off by the precession since J2000."}</p>}
            {[gates.goto, gates.park, gates.unpark, gates.home].filter((gate) => !gate.enabled && gate.reason && gate.reason !== gates.frame.reason)
                .filter((gate, index, list) => list.findIndex((other) => other.reason === gate.reason) === index).slice(0, 2)
                .map((gate) => <p key={gate.reason} className="eq-hint">{gate.reason}</p>)}
            <div className="eq-toolbar">
                <GatedButton gate={disabled(gates.siteSync)} onClick={() => void run("mount.site_sync", () => achernarApi.syncMountSite("to_mount"))}>Write [site] to mount</GatedButton>
                <GatedButton gate={disabled(gates.siteSync)} onClick={() => void run("mount.site_sync", () => achernarApi.syncMountSite("from_mount"))}>Adopt mount site</GatedButton>
            </div>
            <p className="eq-hint">{targetFrame === "mount"
                ? "Slew, sync and center send mount-native coordinates, unconverted."
                : native && native !== "unknown"
                    ? `Slew, sync and center send J2000; the daemon converts to the mount's native frame, ${frameLabel(native)}.`
                    : "Slew, sync and center send J2000; the daemon converts to the mount's native frame once it is known."} Jog and pulse guiding are not offered here: they are direct-API only and need a dead-man (D12).</p>
        </>}
    </div>;
}

/** Plate solve result display. */
export function PlatesolveStatus() {
    return <div id="platesolve-status">Plate Solve Status — stub</div>;
}
