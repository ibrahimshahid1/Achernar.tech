import { useEffect, useState } from "react";
import type { DiscoverableEquipment, DiscoveryReport, DriverCapability } from "../api/achernar";
import type { DeviceBlockCommon, DeviceHealthEntry } from "../types/achernar";
import { healthDetail, linkTone, qualificationLabel, qualificationTone } from "../lib/equipmentView";
import {
    connectionHeading,
    deviceLinkState,
    discoverySummary,
    endpointError,
    hardwareChoices,
    hardwareConnectPayload,
    INDI_DEFAULT_PORT,
    manualConnectPayload,
    PHD2_BASE_PORT,
    transportFor,
    type DeviceKindSpec,
    type TransportFields,
} from "./hardwareDeviceSelection";
import "./hardwareDeviceSelector.css";

type Props = {
    spec: DeviceKindSpec;
    device?: DeviceBlockCommon | null;
    health?: DeviceHealthEntry | null;
    /** This kind's `/drivers` entries; empty while unknown. */
    drivers: DriverCapability[];
    online: boolean;
    report: DiscoveryReport | null;
    discoveryError?: string;
    scanning: boolean;
    busy: boolean;
    discoveryAllowed: boolean;
    discoveryBlockedReason?: string;
    discover: (type: DiscoverableEquipment, endpoint?: string) => Promise<void>;
    resetDiscovery: (type: DiscoverableEquipment) => void;
    action: (name: string, payload?: Record<string, unknown>) => Promise<void>;
};

export default function HardwareDeviceSelector(props: Props) {
    const { spec, device, health, online, report, busy, scanning } = props;
    const type = spec.kind;
    const label = spec.label;
    const [selectedKey, setSelectedKey] = useState("");
    const [endpoint, setEndpoint] = useState("");
    const choices = hardwareChoices(type, report);
    const selected = choices.find((choice) => choice.key === selectedKey);
    const connected = online && device?.connected === true;
    const state = deviceLinkState(device, health);
    const headingTone = connected ? linkTone(state ?? "ready") : state === "lost" ? "bad" : "muted";
    const locked = busy || !online;
    const endpointProblem = endpointError(endpoint);
    const alpacaDiscovery = spec.discovery === "alpaca";
    const discovers = spec.discovery !== "none";
    const detail = healthDetail(health);
    useEffect(() => {
        // Enumeration order may change between scans. Require a new explicit
        // choice even when another device inherited the same driver/index key.
        setSelectedKey("");
    }, [report, type]);
    const discover = () => props.discover(type, alpacaDiscovery && endpoint.trim() ? endpoint.trim() : undefined);
    const onOpen = () => {
        // A connected device is already known from authoritative telemetry.
        // Merely inspecting its selector must never rescan the active camera.
        if (discovers && !connected && !report && !props.discoveryError && !endpointProblem && !locked && props.discoveryAllowed) void discover();
    };
    return <div className={`hardware-device-selector link-${state ?? "none"}`} data-kind={type}>
        <div className="hardware-device-heading">
            <span>{label}</span>
            <strong className={`tone-${headingTone}`}>{connectionHeading(online, connected, state)}</strong>
            {connected && device?.qualification && <span className={`qual-badge tone-${qualificationTone(device.qualification)}`}
                title="D21: only a rig acceptance record makes a device qualified">{qualificationLabel(device.qualification)}</span>}
        </div>
        {discovers ? <select className="cam-select hardware-device-choice" aria-label={`${label} device`} value={connected ? "__connected__" : selectedKey}
            disabled={!online || (busy && !scanning) || connected} onFocus={onOpen} onChange={(event) => setSelectedKey(event.target.value)}>
            {connected ? <option value="__connected__">{device?.name || label} — connected</option> : <option value="">{scanning ? "Discovering devices…" : choices.length ? "Choose a device" : report ? "No selectable devices" : "Choose / discover devices"}</option>}
            {choices.map((choice) => <option key={choice.key} value={choice.key}>{choice.name} · {choice.driverLabel} · #{choice.index}</option>)}
        </select> : <span className="hardware-device-choice hardware-device-static">{connected ? `${device?.name || label} — connected` : "Addressed by host and port"}</span>}
        <div className="hardware-device-actions">
            {discovers && <button type="button" className="btn" disabled={locked || !props.discoveryAllowed || !!endpointProblem} onClick={() => void discover()} aria-label={`Refresh ${label} devices`}>{scanning ? "Discovering…" : "Refresh"}</button>}
            {connected || state === "lost" || state === "degraded"
                ? <button type="button" className="btn" disabled={locked} onClick={() => void props.action("equipment.disconnect", { equipment_type: type })}>Disconnect</button>
                : discovers && <button type="button" className="btn" disabled={locked || !selected} onClick={() => selected && void props.action("equipment.connect", hardwareConnectPayload(type, selected))}>Connect</button>}
        </div>
        {alpacaDiscovery && !connected && <label className="hardware-device-endpoint">Alpaca server URL <input className="cam-num" type="url" value={endpoint} disabled={locked || !props.discoveryAllowed}
            placeholder="http://host:port" onChange={(event) => {
                setEndpoint(event.target.value);
                setSelectedKey("");
                // A choice from the previous endpoint must not remain
                // connectable beside newly edited server text.
                props.resetDiscovery(type);
            }} /> <span>Enter your server, then Refresh.</span></label>}
        <small className="hardware-device-detail" role="status">{!online ? "Hardware state is unavailable." : connected ? `${device?.name || label} is connected; disconnect before choosing another device.`
            : props.discoveryError || endpointProblem || discoverySummary(type, report)}</small>
        {online && detail && <small className={`hardware-device-detail tone-${linkTone(state)}`}>{detail}</small>}
        {online && !props.discoveryAllowed && props.discoveryBlockedReason && <small className="hardware-device-detail">{props.discoveryBlockedReason}</small>}
        {type !== "camera" && !connected && online && <ManualConnect key={type} kind={type} label={label} drivers={props.drivers} locked={locked}
            openByDefault={!discovers} action={props.action} />}
        {report && <details className="hardware-device-discovery"><summary>Discovery details · {new Date(report.scanned_at).toLocaleTimeString()}</summary>
            {(type === "camera" ? report.probes : report.device_probes?.filter((probe) => probe.equipment_type === type) ?? []).map((probe, index) => <p key={`${probe.driver_id}-${index}`}>{probe.driver_label || probe.driver_id || label}: {probe.status}{probe.detail ? ` — ${probe.detail}` : ""}</p>)}
            {report.notes.map((note, index) => <p key={`note-${index}`}>{note}</p>)}
        </details>}
    </div>;
}

/**
 * Connect by address: the driver's `/drivers` entry decides the transport
 * (Alpaca URL and device number, INDI host:port and device, PHD2 host:port,
 * or the simulator), and the option is built from those fields only.
 */
export function ManualConnect({ kind, label, drivers, locked, openByDefault, action }: {
    kind: DiscoverableEquipment;
    label: string;
    drivers: DriverCapability[];
    locked: boolean;
    openByDefault: boolean;
    action: (name: string, payload?: Record<string, unknown>) => Promise<void>;
}) {
    const [driverId, setDriverId] = useState(() => drivers.find((entry) => entry.enabled)?.id ?? "");
    const [fields, setFields] = useState<TransportFields>({});
    useEffect(() => {
        // The driver list can arrive after the first render.
        if (!driverId) {
            const first = drivers.find((entry) => entry.enabled);
            if (first) setDriverId(first.id);
        }
    }, [drivers, driverId]);
    const driver = drivers.find((entry) => entry.id === driverId);
    const transport = driver ? transportFor(driver) : null;
    const built = driver ? manualConnectPayload(kind, driver, fields) : { error: drivers.length ? "Choose a driver." : "The daemon has not reported its drivers yet." };
    const set = (key: keyof TransportFields) => (event: { target: { value: string } }) => setFields((previous) => ({ ...previous, [key]: event.target.value }));
    return <details className="hardware-manual-connect" open={openByDefault}>
        <summary>Connect {label.toLowerCase()} by address</summary>
        <div className="hardware-manual-row">
            <label>Driver <select className="cam-select" aria-label={`${label} driver`} value={driverId} disabled={locked}
                onChange={(event) => { setDriverId(event.target.value); setFields({}); }}>
                <option value="">Choose a driver</option>
                {drivers.map((entry) => <option key={entry.id} value={entry.id} disabled={!entry.enabled}>
                    {entry.label}{entry.enabled ? "" : ` (${entry.qualification === "planned" ? "planned" : "unavailable"})`}
                </option>)}
            </select></label>
            {driver?.qualification && <span className={`qual-badge tone-${qualificationTone(driver.qualification)}`}>{qualificationLabel(driver.qualification)}</span>}
        </div>
        {transport === "alpaca" && <div className="hardware-manual-row">
            <label>Server URL <input className="cam-num wide" type="url" value={fields.url ?? ""} disabled={locked} placeholder="http://host:port" onChange={set("url")} /></label>
            <label>Device number <input className="cam-num" type="number" min="0" step="1" value={fields.deviceNumber ?? ""} disabled={locked} placeholder="0" onChange={set("deviceNumber")} /></label>
        </div>}
        {transport === "indi" && <div className="hardware-manual-row">
            <label>indiserver host <input className="cam-num" value={fields.host ?? ""} disabled={locked} placeholder="localhost" onChange={set("host")} /></label>
            <label>Port <input className="cam-num" type="number" min="1" max="65535" value={fields.port ?? ""} disabled={locked} placeholder={String(INDI_DEFAULT_PORT)} onChange={set("port")} /></label>
            <label>INDI device <input className="cam-num" value={fields.device ?? ""} disabled={locked} placeholder="optional" onChange={set("device")} /></label>
        </div>}
        {transport === "phd2" && <div className="hardware-manual-row">
            <label>PHD2 host <input className="cam-num" value={fields.host ?? ""} disabled={locked} placeholder="localhost" onChange={set("host")} /></label>
            <label>Port <input className="cam-num" type="number" min="1" max="65535" value={fields.port ?? ""} disabled={locked} placeholder={String(PHD2_BASE_PORT)} onChange={set("port")} /></label>
            <span className="hardware-manual-hint">PHD2 instance n listens on {PHD2_BASE_PORT} + n − 1.</span>
        </div>}
        {(transport === "sim" || transport === "option") && (driver?.option_hint
            ? <div className="hardware-manual-row"><label>Option <input className="cam-num wide" value={fields.option ?? ""} disabled={locked} placeholder={driver.option_hint} onChange={set("option")} /></label></div>
            : transport === "sim" && <p className="hardware-manual-hint">In-process simulator; no address needed.</p>)}
        {driver?.reason && <p className="hardware-manual-hint">{driver.reason}</p>}
        <div className="hardware-manual-row">
            <button type="button" className="btn" disabled={locked || !built.payload} onClick={() => built.payload && void action("equipment.connect", built.payload)}>Connect</button>
            {driver && built.error && <span className="hardware-manual-hint tone-warn">{built.error}</span>}
        </div>
    </details>;
}
