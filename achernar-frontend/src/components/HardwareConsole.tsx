import { useEffect, useRef, useState } from "react";
import { achernarApi, usesDjangoRelay, type DiscoverableEquipment, type DiscoveryReport, type DriverCapabilityReport } from "../api/achernar";
import { useDaemonStatus } from "../hooks/useDaemonStatus";
import { cardsFromDaemon } from "../hooks/useEquipment";
import type { EquipmentState } from "../types/achernar";
import CameraControls from "./CameraControls";
import EquipmentCard from "./EquipmentCard";
import MjpegLiveView from "./MjpegLiveView";
import CloudValidationPanel from "./CloudValidationPanel";
import HardwareDeviceSelector from "./HardwareDeviceSelector";
import { hardwareConnectionKey } from "./hardwareDeviceSelection";

const EQUIPMENT = [
    { id: "camera", label: "Camera", field: "camera" },
    { id: "mount", label: "Mount", field: "mount" },
    { id: "focuser", label: "Focuser", field: "focuser" },
    { id: "filter", label: "Filter wheel", field: "filter_wheel" },
] as const;
type Tab = typeof EQUIPMENT[number]["id"];
type ActionResult = { at: string; text: string };

export default function HardwareConsole() {
    const status = useDaemonStatus();
    const [tab, setTab] = useState<Tab>("camera");
    const [results, setResults] = useState<ActionResult[]>([]);
    const [busy, setBusy] = useState(false);
    const [scanning, setScanning] = useState<DiscoverableEquipment | null>(null);
    const [discoveries, setDiscoveries] = useState<Partial<Record<DiscoverableEquipment, DiscoveryReport>>>({});
    const [discoveryErrors, setDiscoveryErrors] = useState<Partial<Record<DiscoverableEquipment, string>>>({});
    const discoveryGeneration = useRef(0);
    const discoveryPending = useRef(false);
    const onlineRef = useRef(status.online);
    onlineRef.current = status.online;
    const data = status.data;
    const connectionKey = hardwareConnectionKey(data, status.online);
    const connectionKeyRef = useRef(connectionKey);
    connectionKeyRef.current = connectionKey;
    const source = usesDjangoRelay() ? "Django relay" : "Direct daemon";
    const equipment = data ? cardsFromDaemon(data, source, "Connected hardware") : [];
    const cameraActive = data?.camera?.streaming === true || data?.camera?.exposing === true;
    useEffect(() => {
        // Advanced controls can connect/disconnect independently of a scan.
        // Retire both cached options and late results from that connection.
        discoveryGeneration.current += 1;
        discoveryPending.current = false;
        setScanning(null);
        setDiscoveries({});
        setDiscoveryErrors({});
    }, [connectionKey]);
    const resetDiscovery = (type: DiscoverableEquipment) => {
        setDiscoveries((previous) => ({ ...previous, [type]: undefined }));
        setDiscoveryErrors((previous) => ({ ...previous, [type]: undefined }));
    };
    const discover = async (type: DiscoverableEquipment, endpoint?: string) => {
        if (!status.online || busy || discoveryPending.current || cameraActive || (type === "camera" && data?.camera?.connected === true)) return;
        discoveryPending.current = true;
        const generation = ++discoveryGeneration.current;
        const requestedConnectionKey = connectionKeyRef.current;
        setScanning(type);
        setDiscoveryErrors((previous) => ({ ...previous, [type]: undefined }));
        try {
            const report = await achernarApi.discoverDevices({ equipment_type: type,
                ...(endpoint ? { driver: "alpaca", option: endpoint } : {}) });
            if (generation !== discoveryGeneration.current || !onlineRef.current || requestedConnectionKey !== connectionKeyRef.current) return;
            if (!Array.isArray(report.cameras) || !Array.isArray(report.probes) || !Array.isArray(report.notes) || typeof report.scanned_at !== "string") {
                throw new Error("The daemon returned an invalid device discovery report.");
            }
            setDiscoveries((previous) => ({ ...previous, [type]: report }));
        } catch (error) {
            if (generation !== discoveryGeneration.current || !onlineRef.current || requestedConnectionKey !== connectionKeyRef.current) return;
            setDiscoveries((previous) => ({ ...previous, [type]: undefined }));
            setDiscoveryErrors((previous) => ({ ...previous, [type]: error instanceof Error ? error.message : "Device discovery failed." }));
        } finally {
            if (generation === discoveryGeneration.current) {
                discoveryPending.current = false;
                setScanning(null);
            }
        }
    };

    const action = async (name: string, payload: Record<string, unknown> = {}) => {
        if (name === "equipment.connect" || name === "equipment.disconnect") {
            // An auxiliary replacement can have the same name and a brief
            // disconnected state that status polling never observes.
            discoveryGeneration.current += 1;
            discoveryPending.current = false;
            setScanning(null);
            setDiscoveries({});
            setDiscoveryErrors({});
        }
        setBusy(true);
        try {
            const result = await achernarApi.command(name, payload);
            const text = result.pending ? `${name}: still running on the daemon.`
                : result.ok === false ? `${name}: ${result.error ?? "failed"}`
                    : `${name}: ${JSON.stringify(result.payload ?? { status: "completed" })}`;
            setResults((previous) => [{ at: new Date().toLocaleTimeString(), text }, ...previous].slice(0, 20));
        } catch (error) {
            setResults((previous) => [{ at: new Date().toLocaleTimeString(), text: `${name}: ${error instanceof Error ? error.message : "failed"}` }, ...previous].slice(0, 20));
        } finally {
            setBusy(false);
            void status.refetch();
        }
    };

    return (
        <main className="hardware-console">
            <section className="panel hardware-overview">
                <div className="section-row">
                    <h1>Hardware test console</h1>
                    <button type="button" className="btn" onClick={() => void status.refetch()} disabled={status.isFetching}>Refresh status</button>
                </div>
                <p role="status">{status.online ? `${equipment.length} connected ${equipment.length === 1 ? "device" : "devices"}` : status.isLoading ? "Checking daemon..." : "Daemon unavailable; hardware state unknown."}</p>
                {status.error && <p className="error-details">{status.error instanceof Error ? status.error.message : "Status request failed."}</p>}
                <div className="hardware-status-list" aria-label="Live connection status">
                    {EQUIPMENT.map(({ id, label, field }) => {
                        const reason = data?.device_health?.[field]?.reason;
                        const discoveryAllowed = !cameraActive && !(field === "camera" && data?.camera?.connected === true);
                        const blockedReason = cameraActive ? "Stop preview or capture before refreshing device discovery."
                                : "The active camera is shown from live status. Disconnect it before rescanning the camera bus.";
                        return <HardwareDeviceSelector key={id} type={field} label={label} device={data?.[field]} online={status.online}
                            reason={reason} report={discoveries[field] ?? null} discoveryError={discoveryErrors[field]} scanning={scanning === field}
                            busy={busy || scanning !== null} discoveryAllowed={discoveryAllowed} discoveryBlockedReason={blockedReason}
                            discover={discover} resetDiscovery={resetDiscovery} action={action} />;
                    })}
                </div>
                <div className="equipment-grid hardware-equipment-grid">
                    {equipment.map((card) => <EquipmentCard key={card.id} model={card} />)}
                </div>
            </section>

            <div className="equipment-tabbar" role="tablist" aria-label="Hardware tests">
                {EQUIPMENT.map(({ id, label }) => (
                    <button key={id} id={`hardware-tab-${id}`} type="button" role="tab" aria-selected={tab === id} aria-controls={`hardware-panel-${id}`} className={`tab-button ${tab === id ? "active" : ""}`} onClick={() => setTab(id)}>{label}</button>
                ))}
            </div>
            <section id={`hardware-panel-${tab}`} role="tabpanel" aria-labelledby={`hardware-tab-${tab}`} className="panel hardware-test-panel">
                {tab === "camera" && <div className="hardware-camera-test">
                    <MjpegLiveView cameraConnected={data?.camera?.connected === true} cameraConnectionId={data?.camera?.connection_id ?? null} />
                    <CameraControls />
                    <CloudValidationPanel cameraConnectionId={status.online ? data?.camera?.connection_id ?? null : null} streaming={status.online && data?.camera?.streaming === true} currentFrame={data?.camera?.latest_frame} />
                </div>}
                {tab === "mount" && <DeviceTest key="mount" type="mount" device={data?.mount} online={status.online} busy={busy} action={action} />}
                {tab === "focuser" && <DeviceTest key="focuser" type="focuser" device={data?.focuser} online={status.online} busy={busy} action={action} />}
                {tab === "filter" && <>
                    <h2>Filter wheel</h2>
                    <p>{data?.filter_wheel?.connected === true ? `${data.filter_wheel.name ?? "Filter wheel"}: slot ${data.filter_wheel.slot ?? "unknown"}, filter ${data.filter_wheel.filter ?? "unknown"}.` : "No filter wheel connected."}</p>
                    <p>Filter wheel connection and movement commands are unavailable in this daemon build.</p>
                </>}
            </section>
            {results.length > 0 && <section className="panel hardware-action-results" aria-label="Command results">
                <h2>Command results</h2>
                {results.map((result, index) => <p key={`${result.at}-${index}`}><time>{result.at}</time> {result.text}</p>)}
            </section>}
        </main>
    );
}

function DeviceTest({ type, device, online, busy, action }: {
    type: "mount" | "focuser";
    device: EquipmentState["mount"] | EquipmentState["focuser"];
    online: boolean;
    busy: boolean;
    action: (name: string, payload?: Record<string, unknown>) => Promise<void>;
}) {
    const [report, setReport] = useState<DriverCapabilityReport | null>(null);
    const [driver, setDriver] = useState("");
    const [option, setOption] = useState("");
    const [ra, setRa] = useState("");
    const [dec, setDec] = useState("");
    const [position, setPosition] = useState("");
    const [error, setError] = useState("");
    useEffect(() => {
        let cancelled = false;
        void achernarApi.getDriverCapabilities().then((next) => { if (!cancelled) setReport(next); }).catch((failure) => { if (!cancelled) setError(failure instanceof Error ? failure.message : "Driver list unavailable."); });
        return () => { cancelled = true; };
    }, [type]);
    const drivers = report?.[type] ?? [];
    const selected = drivers.find((entry) => entry.id === driver);
    const connected = device?.connected === true;
    const locked = busy || !online;
    const coordsValid = ra.trim() !== "" && dec.trim() !== "" && Number.isFinite(Number(ra)) && Number.isFinite(Number(dec)) && Number(ra) >= 0 && Number(ra) < 24 && Number(dec) >= -90 && Number(dec) <= 90;
    const positionValid = position.trim() !== "" && Number.isSafeInteger(Number(position)) && Number(position) >= 0;
    return <>
        <h2>{type === "mount" ? "Mount" : "Focuser"} controls</h2>
        <p>{connected ? device?.name ?? "Connected" : "Not connected."}</p>
        {error && <p className="error-details">{error}</p>}
        {!connected ? <div className="button-row">
            <label>Driver <select className="cam-select" aria-label={`${type} driver`} value={driver} disabled={locked} onChange={(event) => setDriver(event.target.value)}><option value="">Choose a driver</option>{drivers.map((entry) => <option key={entry.id} value={entry.id} disabled={!entry.enabled}>{entry.label}{entry.enabled ? "" : " (unavailable)"}</option>)}</select></label>
            <label>Connection option <input className="cam-num" value={option} placeholder={selected?.option_hint ?? "Driver connection option"} disabled={locked} onChange={(event) => setOption(event.target.value)} /></label>
            <button type="button" className="btn" disabled={locked || !selected?.enabled} onClick={() => void action("equipment.connect", { equipment_type: type, driver, ...(option.trim() ? { option: option.trim() } : {}) })}>Connect</button>
        </div> : <>
            <div className="button-row"><button type="button" className="btn" disabled={locked} onClick={() => void action("equipment.disconnect", { equipment_type: type })}>Disconnect</button></div>
            {type === "mount" ? <>
                <div className="button-row">
                    <button type="button" className="btn" disabled={locked} onClick={() => void action("mount.set_tracking", { enabled: true })}>Start tracking</button>
                    <button type="button" className="btn" disabled={locked} onClick={() => void action("mount.set_tracking", { enabled: false })}>Stop tracking</button>
                    <button type="button" className="btn" disabled={!online} onClick={() => void action("mount.abort")}>Abort motion</button>
                </div>
                <div className="button-row">
                    <label>RA (hours) <input className="cam-num" type="number" min="0" max="24" value={ra} disabled={locked} onChange={(event) => setRa(event.target.value)} /></label>
                    <label>Dec (degrees) <input className="cam-num" type="number" min="-90" max="90" value={dec} disabled={locked} onChange={(event) => setDec(event.target.value)} /></label>
                    <button type="button" className="btn" disabled={locked || !coordsValid} onClick={() => void action("mount.slew", { ra: Number(ra), dec: Number(dec) })}>Slew</button>
                    <button type="button" className="btn" disabled={locked || !coordsValid} onClick={() => void action("mount.sync", { ra: Number(ra), dec: Number(dec) })}>Sync coordinates</button>
                </div>
            </> : <>
                <div className="button-row">{[-100, -10, 10, 100].map((steps) => <button key={steps} type="button" className="btn" disabled={locked} onClick={() => void action("focuser.move_relative", { steps })}>{steps > 0 ? "+" : ""}{steps} steps</button>)}<button type="button" className="btn" disabled={!online} onClick={() => void action("focuser.abort")}>Abort motion</button></div>
                <div className="button-row"><label>Absolute position <input className="cam-num" type="number" min="0" step="1" value={position} disabled={locked} onChange={(event) => setPosition(event.target.value)} /></label><button type="button" className="btn" disabled={locked || !positionValid || !(device && "absolute" in device && device.absolute)} onClick={() => void action("focuser.move_absolute", { position: Number(position) })}>Move to position</button></div>
            </>}
        </>}
    </>;
}
