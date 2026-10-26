import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { achernarApi, usesDjangoRelay, type DiscoverableEquipment, type DiscoveryReport, type EstopReport } from "../api/achernar";
import { useDaemonStatus } from "../hooks/useDaemonStatus";
import { cardsFromDaemon } from "../hooks/useEquipment";
import { getHardwareTransport } from "../lib/runtime";
import { drainJournal, newestFirst } from "../lib/equipmentJournal";
import type { RunCommand } from "../lib/equipmentView";
import type { EquipmentEvent, ResumeBlock } from "../types/achernar";
import CameraControls from "./CameraControls";
import EquipmentCard from "./EquipmentCard";
import EquipmentEventLog from "./EquipmentEventLog";
import MjpegLiveView from "./MjpegLiveView";
import CloudValidationPanel from "./CloudValidationPanel";
import HardwareDeviceSelector from "./HardwareDeviceSelector";
import { FilterWheelControl } from "./filter";
import FocuserControls from "./focuser";
import GuiderPanel from "./guider";
import { MountControls, MountStatus } from "./mount";
import { BusyFlags, EmergencyStopButton, ResumeBlockBanner, SafetyPanel } from "./safety";
import WeatherPanel from "./weather";
import { DEVICE_KINDS, deviceBlock, driversFor, hardwareConnectionKey, plannedKinds } from "./hardwareDeviceSelection";
import "./equipmentPanels.css";

const TABS = [
    { id: "camera", label: "Camera" },
    { id: "mount", label: "Mount" },
    { id: "focuser", label: "Focuser" },
    { id: "filter", label: "Filter wheel" },
    { id: "guider", label: "Guider" },
    { id: "environment", label: "Weather & safety" },
    { id: "events", label: "Events" },
] as const;
type Tab = typeof TABS[number]["id"];
type ActionResult = { at: string; text: string };

const EVENT_POLL_MS = 2000;
const EVENT_LIMIT = 200;
/** The daemon's per-request cap on `equipment/events`. */
const EVENT_PAGE = 500;
const EVENT_CATCH_UP_PAGES = 40;

function errorText(error: unknown, fallback: string): string {
    return error instanceof Error ? error.message : fallback;
}

export default function HardwareConsole() {
    const status = useDaemonStatus();
    const transport = getHardwareTransport();
    const [tab, setTab] = useState<Tab>("camera");
    const [results, setResults] = useState<ActionResult[]>([]);
    // A count, not a flag: an abort may finish while the command it
    // interrupted is still in flight.
    const [inflight, setInflight] = useState(0);
    const busy = inflight > 0;
    const [scanning, setScanning] = useState<DiscoverableEquipment | null>(null);
    const [discoveries, setDiscoveries] = useState<Partial<Record<DiscoverableEquipment, DiscoveryReport>>>({});
    const [discoveryErrors, setDiscoveryErrors] = useState<Partial<Record<DiscoverableEquipment, string>>>({});
    const [estopPending, setEstopPending] = useState(false);
    const [estopAbortExposure, setEstopAbortExposure] = useState(false);
    const [estopReport, setEstopReport] = useState<EstopReport | null>(null);
    const [estopError, setEstopError] = useState<string | null>(null);
    const [events, setEvents] = useState<EquipmentEvent[]>([]);
    const [eventsError, setEventsError] = useState<string | null>(null);
    const [showGuideSteps, setShowGuideSteps] = useState(false);
    const eventSeq = useRef(0);
    const discoveryGeneration = useRef(0);
    const discoveryPending = useRef(false);
    const onlineRef = useRef(status.online);
    onlineRef.current = status.online;
    const data = status.data;
    const online = status.online;
    const connectionKey = hardwareConnectionKey(data, online);
    const connectionKeyRef = useRef(connectionKey);
    connectionKeyRef.current = connectionKey;
    const source = usesDjangoRelay() ? "Django relay" : "Direct daemon";
    const equipment = data ? cardsFromDaemon(data, source, "Connected hardware") : [];
    const cameraActive = data?.camera?.streaming === true || data?.camera?.exposing === true;
    const mountConnectionId = data?.mount?.connected === true ? data.mount.connection_id ?? data.mount.name ?? "mount" : null;

    const drivers = useQuery({
        queryKey: ["daemon", "drivers", transport, data?.daemon_id ?? null],
        queryFn: () => achernarApi.getDriverCapabilities(),
        enabled: online,
        staleTime: 60_000,
        retry: false,
    });
    // Capabilities (the O10 gates) per mount connection; re-read on reconnect.
    const mountDescriptor = useQuery({
        queryKey: ["daemon", "mount-descriptor", transport, mountConnectionId],
        queryFn: () => achernarApi.getMountDescriptor(),
        enabled: online && mountConnectionId != null,
        staleTime: Infinity,
        retry: 1,
    });
    const safety = useQuery({
        queryKey: ["daemon", "safety", transport],
        queryFn: () => achernarApi.getSafety(),
        enabled: online && tab === "environment",
        refetchInterval: 2000,
        retry: false,
    });

    useEffect(() => {
        // Advanced controls can connect/disconnect independently of a scan.
        // Retire both cached options and late results from that connection.
        discoveryGeneration.current += 1;
        discoveryPending.current = false;
        setScanning(null);
        setDiscoveries({});
        setDiscoveryErrors({});
    }, [connectionKey]);

    // The journal tail, polled only while the Events tab is open. The same
    // action works on both transports: the relay passes `since_seq` as the
    // payload, the direct route as a query parameter.
    useEffect(() => {
        if (tab !== "events" || !online) return;
        let cancelled = false;
        let polling = false;
        const poll = async () => {
            if (polling) return;
            polling = true;
            try {
                const drained = await drainJournal((since_seq, limit) => achernarApi.getEquipmentEvents({ since_seq, limit }),
                    eventSeq.current, EVENT_PAGE, EVENT_CATCH_UP_PAGES);
                if (cancelled) return;
                eventSeq.current = drained.nextSeq;
                setEventsError(null);
                if (drained.reset) setEvents([]);
                else if (drained.events.length) setEvents((previous) => newestFirst(previous, drained.events, EVENT_LIMIT));
            } catch (error) {
                if (!cancelled) setEventsError(errorText(error, "The equipment journal is unavailable."));
            } finally {
                polling = false;
            }
        };
        void poll();
        const timer = window.setInterval(() => void poll(), EVENT_POLL_MS);
        return () => {
            cancelled = true;
            window.clearInterval(timer);
        };
    }, [tab, online]);

    const log = useCallback((text: string) => {
        setResults((previous) => [{ at: new Date().toLocaleTimeString(), text }, ...previous].slice(0, 20));
    }, []);

    const resetDiscovery = (type: DiscoverableEquipment) => {
        setDiscoveries((previous) => ({ ...previous, [type]: undefined }));
        setDiscoveryErrors((previous) => ({ ...previous, [type]: undefined }));
    };
    const discover = async (type: DiscoverableEquipment, endpoint?: string) => {
        if (!online || busy || discoveryPending.current || cameraActive || (type === "camera" && data?.camera?.connected === true)) return;
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
            setDiscoveryErrors((previous) => ({ ...previous, [type]: errorText(error, "Device discovery failed.") }));
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
        setInflight((count) => count + 1);
        try {
            const result = await achernarApi.command(name, payload);
            log(result.pending ? `${name}: still running on the daemon.`
                : result.ok === false ? `${name}: ${result.error ?? "failed"}`
                    : `${name}: ${JSON.stringify(result.payload ?? { status: "completed" })}`);
        } catch (error) {
            log(`${name}: ${errorText(error, "failed")}`);
        } finally {
            setInflight((count) => Math.max(0, count - 1));
            void status.refetch();
        }
    };

    /** Typed API calls from the panels, with client-side argument checks. */
    const run: RunCommand = async (label, call) => {
        setInflight((count) => count + 1);
        try {
            const result = await call();
            log(`${label}: ${JSON.stringify(result ?? { status: "completed" })}`);
        } catch (error) {
            log(`${label}: ${errorText(error, "failed")}`);
        } finally {
            setInflight((count) => Math.max(0, count - 1));
            void status.refetch();
        }
    };

    // Never gated on other commands: an e-stop does not wait for anything.
    const estop = async () => {
        setEstopPending(true);
        setEstopError(null);
        try {
            const result = await achernarApi.emergencyStop({ abort_exposure: estopAbortExposure });
            if (result.pending) {
                setEstopReport(null);
                setEstopError("E-stop sent; the relay has not returned per-device results yet. Watch the device states.");
            } else if (result.ok === false) {
                setEstopReport(null);
                setEstopError(result.error ?? "The e-stop request failed.");
            } else {
                setEstopReport((result.payload ?? null) as EstopReport | null);
            }
            log(`equipment.estop: ${result.pending ? "pending" : result.ok === false ? result.error ?? "failed" : (result.payload as EstopReport | undefined)?.status ?? "sent"}`);
        } catch (error) {
            setEstopReport(null);
            setEstopError(errorText(error, "The e-stop request failed."));
            log(`equipment.estop: ${errorText(error, "failed")}`);
        } finally {
            setEstopPending(false);
            void status.refetch();
        }
    };

    const clearBlock = (block: ResumeBlock) => void run("equipment.clear_resume_block", () => achernarApi.clearResumeBlock(block.id));
    const health = (key: string) => data?.device_health?.[key] ?? null;
    const planned = plannedKinds(drivers.data);
    const shownEvents = showGuideSteps ? events : events.filter((event) => event.kind !== "guide_step");

    return (
        <main className="hardware-console">
            <section className="panel eq-safety-strip" aria-label="Emergency stop and resume blocks">
                <EmergencyStopButton online={online} pending={estopPending} abortExposure={estopAbortExposure}
                    onAbortExposureChange={setEstopAbortExposure} onStop={() => void estop()} report={estopReport} error={estopError} />
                <ResumeBlockBanner blocks={data?.resume_blocks} online={online} busy={busy} onClear={clearBlock} />
                <BusyFlags busy={data?.busy} />
            </section>

            <section className="panel hardware-overview">
                <div className="section-row">
                    <h1>Hardware test console</h1>
                    <button type="button" className="btn" onClick={() => void status.refetch()} disabled={status.isFetching}>Refresh status</button>
                </div>
                <p role="status">{online ? `${equipment.length} connected ${equipment.length === 1 ? "device" : "devices"}` : status.isLoading ? "Checking daemon..." : "Daemon unavailable; hardware state unknown."}</p>
                {status.error && <p className="error-details">{errorText(status.error, "Status request failed.")}</p>}
                {drivers.error && <p className="error-details">Driver list unavailable: {errorText(drivers.error, "request failed")}</p>}
                <div className="hardware-status-list" aria-label="Live connection status">
                    {DEVICE_KINDS.map((spec) => {
                        const discoveryAllowed = !cameraActive && !(spec.kind === "camera" && data?.camera?.connected === true);
                        const blockedReason = cameraActive ? "Stop preview or capture before refreshing device discovery."
                            : "The active camera is shown from live status. Disconnect it before rescanning the camera bus.";
                        return <HardwareDeviceSelector key={spec.kind} spec={spec} device={deviceBlock(data, spec)} health={health(spec.healthKey)}
                            drivers={driversFor(drivers.data, spec.kind)} online={online}
                            report={discoveries[spec.kind] ?? null} discoveryError={discoveryErrors[spec.kind]} scanning={scanning === spec.kind}
                            busy={busy || scanning !== null} discoveryAllowed={discoveryAllowed} discoveryBlockedReason={blockedReason}
                            discover={discover} resetDiscovery={resetDiscovery} action={action} />;
                    })}
                </div>
                {planned.length > 0 && <p className="eq-planned">Planned device classes: {planned.map((entry) => entry.label).join(", ")}. {planned[0].reason}</p>}
                <div className="equipment-grid hardware-equipment-grid">
                    {equipment.map((card) => <EquipmentCard key={card.id} model={card} />)}
                </div>
            </section>

            <div className="equipment-tabbar" role="tablist" aria-label="Hardware tests">
                {TABS.map(({ id, label }) => (
                    <button key={id} id={`hardware-tab-${id}`} type="button" role="tab" aria-selected={tab === id} aria-controls={`hardware-panel-${id}`} className={`tab-button ${tab === id ? "active" : ""}`} onClick={() => setTab(id)}>{label}</button>
                ))}
            </div>
            <section id={`hardware-panel-${tab}`} role="tabpanel" aria-labelledby={`hardware-tab-${tab}`} className="panel hardware-test-panel">
                {tab === "camera" && <div className="hardware-camera-test">
                    <MjpegLiveView cameraConnected={data?.camera?.connected === true} cameraConnectionId={data?.camera?.connection_id ?? null} />
                    <CameraControls />
                    <CloudValidationPanel cameraConnectionId={online ? data?.camera?.connection_id ?? null : null} streaming={online && data?.camera?.streaming === true} currentFrame={data?.camera?.latest_frame} />
                </div>}
                {tab === "mount" && <>
                    <h2>Mount</h2>
                    <MountControls mount={data?.mount} health={health("mount")} descriptor={mountDescriptor.data} online={online} busy={busy} run={run} />
                    {mountDescriptor.error && <p className="eq-hint tone-warn">Capabilities not read: {errorText(mountDescriptor.error, "request failed")}. The daemon still checks every command.</p>}
                    {data?.mount?.connected === true && <MountStatus mount={data.mount} descriptor={mountDescriptor.data} />}
                </>}
                {tab === "focuser" && <>
                    <h2>Focuser</h2>
                    <FocuserControls focuser={data?.focuser} health={health("focuser")} online={online} busy={busy} run={run} />
                </>}
                {tab === "filter" && <>
                    <h2>Filter wheel</h2>
                    <FilterWheelControl wheel={data?.filter_wheel} online={online} busy={busy} run={run} />
                </>}
                {tab === "guider" && <>
                    <h2>Guider</h2>
                    <GuiderPanel guiding={data?.guiding ?? data?.guider} health={health("guider")} online={online} busy={busy} run={run} />
                </>}
                {tab === "environment" && <>
                    <h2>Safety</h2>
                    <SafetyPanel summary={safety.data} monitor={data?.safety_monitor} online={online} />
                    {safety.error && <p className="eq-hint tone-warn">Combined safety state unavailable: {errorText(safety.error, "request failed")}</p>}
                    <h2>Weather</h2>
                    <WeatherPanel weather={data?.weather} online={online} />
                </>}
                {tab === "events" && <>
                    <h2>Equipment events</h2>
                    <EquipmentEventLog events={shownEvents} error={eventsError} showGuideSteps={showGuideSteps} onShowGuideSteps={setShowGuideSteps} />
                </>}
            </section>
            {results.length > 0 && <section className="panel hardware-action-results" aria-label="Command results">
                <h2>Command results</h2>
                {results.map((result, index) => <p key={`${result.at}-${index}`}><time>{result.at}</time> {result.text}</p>)}
            </section>}
        </main>
    );
}
