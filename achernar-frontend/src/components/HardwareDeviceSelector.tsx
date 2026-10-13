import { useEffect, useState } from "react";
import type { DiscoverableEquipment, DiscoveryReport } from "../api/achernar";
import { discoverySummary, endpointError, hardwareChoices, hardwareConnectPayload } from "./hardwareDeviceSelection";
import "./hardwareDeviceSelector.css";

type Props = {
    type: DiscoverableEquipment;
    label: string;
    device?: { connected?: boolean; name?: string } | null;
    online: boolean;
    reason?: string | null;
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
    const { type, label, device, online, report, busy, scanning } = props;
    const [selectedKey, setSelectedKey] = useState("");
    const [endpoint, setEndpoint] = useState("");
    const choices = hardwareChoices(type, report);
    const selected = choices.find((choice) => choice.key === selectedKey);
    const connected = online && device?.connected === true;
    const locked = busy || !online;
    const endpointProblem = endpointError(endpoint);
    const auxiliary = type === "mount" || type === "focuser";
    const connectionSupported = type !== "filter_wheel";
    useEffect(() => {
        // Enumeration order may change between scans. Require a new explicit
        // choice even when another device inherited the same driver/index key.
        setSelectedKey("");
    }, [report, type]);
    const discover = () => props.discover(type, auxiliary && endpoint.trim() ? endpoint.trim() : undefined);
    const onOpen = () => {
        // A connected device is already known from authoritative telemetry.
        // Merely inspecting its selector must never rescan the active camera.
        if (!connected && !report && !props.discoveryError && !endpointProblem && !locked && props.discoveryAllowed) void discover();
    };
    return <div className="hardware-device-selector">
        <div className="hardware-device-heading"><span>{label}</span><strong>{!online ? "Unknown" : connected ? "Connected" : "Not connected"}</strong></div>
        <select className="cam-select hardware-device-choice" aria-label={`${label} device`} value={connected ? "__connected__" : selectedKey}
            disabled={!online || (busy && !scanning) || connected} onFocus={onOpen} onChange={(event) => setSelectedKey(event.target.value)}>
            {connected ? <option value="__connected__">{device?.name || label} — connected</option> : <option value="">{scanning ? "Discovering devices…" : choices.length ? "Choose a device" : report ? "No selectable devices" : "Choose / discover devices"}</option>}
            {choices.map((choice) => <option key={choice.key} value={choice.key}>{choice.name} · {choice.driverLabel} · #{choice.index}</option>)}
        </select>
        <div className="hardware-device-actions">
            <button type="button" className="btn" disabled={locked || !props.discoveryAllowed || !!endpointProblem} onClick={() => void discover()} aria-label={`Refresh ${label} devices`}>{scanning ? "Discovering…" : "Refresh"}</button>
            {connectionSupported && (connected
                ? <button type="button" className="btn" disabled={locked} onClick={() => void props.action("equipment.disconnect", { equipment_type: type })}>Disconnect</button>
                : <button type="button" className="btn" disabled={locked || !selected} onClick={() => selected && void props.action("equipment.connect", hardwareConnectPayload(type, selected))}>Connect</button>)}
        </div>
        {auxiliary && !connected && <label className="hardware-device-endpoint">Alpaca server URL <input className="cam-num" type="url" value={endpoint} disabled={locked || !props.discoveryAllowed}
            placeholder="http://host:port" onChange={(event) => {
                setEndpoint(event.target.value);
                setSelectedKey("");
                // A choice from the previous endpoint must not remain
                // connectable beside newly edited server text.
                props.resetDiscovery(type);
            }} /> <span>Enter your server, then Refresh.</span></label>}
        <small className="hardware-device-detail" role="status">{!online ? "Hardware state is unavailable." : connected ? `${device?.name || label} is connected; disconnect before choosing another device.`
            : props.discoveryError || endpointProblem || discoverySummary(type, report)}</small>
        {!connected && online && props.reason && <small className="hardware-device-detail">{props.reason}</small>}
        {online && !props.discoveryAllowed && props.discoveryBlockedReason && <small className="hardware-device-detail">{props.discoveryBlockedReason}</small>}
        {report && <details className="hardware-device-discovery"><summary>Discovery details · {new Date(report.scanned_at).toLocaleTimeString()}</summary>
            {(type === "camera" ? report.probes : report.device_probes?.filter((probe) => probe.equipment_type === type) ?? []).map((probe, index) => <p key={`${probe.driver_id}-${index}`}>{probe.driver_label || probe.driver_id || label}: {probe.status}{probe.detail ? ` — ${probe.detail}` : ""}</p>)}
            {report.notes.map((note, index) => <p key={`note-${index}`}>{note}</p>)}
        </details>}
    </div>;
}
