import { useCallback, useEffect, useState } from "react";
import { cloudApi } from "../api/cloud";
import { daemonApi, type CamParam, type CamFormat, type DriverCapability, type DiscoveryReport } from "../api/daemon";
import { getHardwareTransport } from "../lib/runtime";

/**
 * Manual camera control panel.
 *
 * Philosophy: the operator always has direct, NINA/SharpCap-style control over
 * every supported camera parameter. AI/autonomy can layer on later, but this
 * panel stays manual while the hardware/control spine is being proven.
 *
 * Transport-aware: browser mode uses Django relay, desktop mode talks to the
 * local daemon directly.
 */

const DRIVERS: DriverCapability[] = [
    { id: "sim", label: "Simulator", compiled: true, enabled: true },
    { id: "alpaca", label: "Alpaca", compiled: true, enabled: true },
    { id: "svb", label: "SVBony", compiled: true, enabled: true },
    { id: "asi", label: "ZWO ASI", compiled: true, enabled: true },
    { id: "toup", label: "ToupTek", compiled: true, enabled: true },
    { id: "uvc", label: "UVC / Webcam", compiled: true, enabled: true },
    { id: "gphoto2", label: "DSLR (gphoto2)", compiled: true, enabled: true },
    { id: "indi", label: "INDI", compiled: true, enabled: true },
    { id: "indigo", label: "INDIGO", compiled: true, enabled: true },
    { id: "wdir", label: "Watch Folder", compiled: true, enabled: true },
];

function humanize(name: string): string {
    return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function fmtVal(v: number): string {
    return Number.isInteger(v) ? String(v) : String(Number(v.toFixed(2)));
}

function capabilityReason(driver: DriverCapability): string {
    return driver.reason ?? (driver.enabled ? "Available in this daemon build." : "Unavailable in this daemon build.");
}

export default function CameraControls() {
    const hardwareTransport = getHardwareTransport();
    const cameraApi = hardwareTransport === "django" ? cloudApi : daemonApi;
    const [params, setParams] = useState<CamParam[]>([]);
    const [formats, setFormats] = useState<CamFormat[]>([]);
    const [selectedFormat, setSelectedFormat] = useState(0);
    const [connected, setConnected] = useState(false);
    const [driver, setDriver] = useState("sim");
    const [driverCapabilities, setDriverCapabilities] = useState<DriverCapability[] | null>(null);
    const [discovery, setDiscovery] = useState<DiscoveryReport | null>(null);
    const [streaming, setStreaming] = useState(false);
    const [message, setMessage] = useState("No camera connected.");
    const [busy, setBusy] = useState(false);
    const [exposure, setExposure] = useState(1);
    const [lastCapture, setLastCapture] = useState<string | null>(null);
    const [captureWarnings, setCaptureWarnings] = useState<string[]>([]);
    /**
     * Which camera within the selected driver's enumeration. Set by picking a
     * device from a scan; 0 when the operator chose a driver by hand, which is
     * the right default for the single-camera case.
     */
    const [selectedCameraIndex, setSelectedCameraIndex] = useState(0);

    const refresh = useCallback(async () => {
        try {
            const [p, f] = await Promise.all([
                cameraApi.getCameraParameters(),
                cameraApi.getCameraFormats(),
            ]);
            setParams(p);
            setFormats(f);
            setConnected(true);

            // Default to a live-view format when the body has one. On a DSLR a
            // `still` format is one mechanical shutter actuation per frame, so
            // landing on index 0 and pressing Start can burn thousands of
            // actuations just framing a target.
            const liveview = f.find((fmt) => fmt.mode === "liveview");
            setSelectedFormat(liveview ? liveview.index : (f[0]?.index ?? 0));

            setMessage(`${p.length} controls available.`);
        } catch {
            setConnected(false);
            setParams([]);
            setFormats([]);
            setMessage("No camera connected.");
        }
    }, [cameraApi]);

    useEffect(() => {
        refresh();
    }, [refresh]);

    useEffect(() => {
        let cancelled = false;

        cameraApi
            .getDriverCapabilities()
            .then((report) => {
                if (!cancelled) {
                    setDriverCapabilities(report.camera);
                }
            })
            .catch(() => {
                if (!cancelled) {
                    setDriverCapabilities(null);
                }
            });

        return () => {
            cancelled = true;
        };
    }, [cameraApi]);

    const driverOptions = driverCapabilities ?? DRIVERS;
    const selectedDriver = driverOptions.find((d) => d.id === driver);
    const selectedDriverEnabled = selectedDriver?.enabled ?? false;
    const unavailableDrivers = driverOptions.filter((d) => !d.enabled);

    useEffect(() => {
        if (!driverCapabilities) {
            return;
        }
        const current = driverCapabilities.find((d) => d.id === driver);
        if (current?.enabled) {
            return;
        }
        const firstEnabled = driverCapabilities.find((d) => d.enabled);
        if (firstEnabled) {
            setDriver(firstEnabled.id);
            setMessage(`${current?.label ?? driver} is unavailable in this build; selected ${firstEnabled.label}.`);
        }
    }, [driver, driverCapabilities]);

    const connect = async () => {
        if (!selectedDriverEnabled) {
            setMessage(selectedDriver?.reason ?? `${selectedDriver?.label ?? driver} is unavailable in this build.`);
            return;
        }
        setBusy(true);
        setMessage(`Connecting ${driver}...`);
        try {
            // Pass the index the scan reported for this driver. `/discover` has
            // always returned it and the UI has always dropped it, so with two
            // cameras attached both connects opened camera 0 — which is exactly
            // the guiding + imaging setup, failing silently.
            const res = await cameraApi.connectCamera(driver, undefined, selectedCameraIndex);
            setMessage(`Connected: ${res.camera ?? driver}`);
            await refresh();
        } catch (e) {
            setMessage(e instanceof Error ? e.message : "Connect failed.");
        } finally {
            setBusy(false);
        }
    };

    /**
     * Scan local USB/SDK buses for attached cameras.
     *
     * Only offered on the direct-daemon transport: discovery probes hardware on
     * the machine the daemon runs on, which the Django relay does not expose.
     * Finding a device selects its driver but does NOT connect — the operator
     * still presses Connect (manual-first).
     */
    const scanForDevices = async () => {
        setBusy(true);
        setMessage("Scanning for devices...");
        setDiscovery(null);
        try {
            const report = await daemonApi.discoverDevices();
            setDiscovery(report);
            if (report.cameras.length > 0) {
                // Preselect the first found device as the likely choice — its
                // driver AND its index, so a two-camera rig does not silently
                // default both to enumeration slot 0.
                setDriver(report.cameras[0].driver_id);
                setSelectedCameraIndex(report.cameras[0].index);
                const plural = report.cameras.length === 1 ? "camera" : "cameras";
                setMessage(`Found ${report.cameras.length} ${plural}. Select one and press Connect.`);
            } else {
                setMessage(report.notes[report.notes.length - 1] ?? "No cameras found.");
            }
        } catch (e) {
            setMessage(e instanceof Error ? e.message : "Scan failed.");
        } finally {
            setBusy(false);
        }
    };

    const setParam = async (name: string, value: number) => {
        // Optimistic update, then reconcile with the value the hardware applied.
        setParams((prev) => prev.map((p) => (p.name === name ? { ...p, current: value } : p)));
        try {
            const res = await cameraApi.setCameraParameter(name, value);
            setParams((prev) => prev.map((p) => (p.name === name ? { ...p, current: res.value } : p)));
        } catch {
            setMessage(`Failed to set ${name}.`);
        }
    };

    const startStream = async () => {
        setBusy(true);
        try {
            const res = await cameraApi.startCameraStream(selectedFormat);
            setStreaming(true);
            setMessage(`Streaming ${res.width}x${res.height} ${res.format}`);
        } catch (e) {
            setMessage(e instanceof Error ? e.message : "Start failed.");
        } finally {
            setBusy(false);
        }
    };

    const stopStream = async () => {
        setBusy(true);
        try {
            await cameraApi.stopCameraStream();
            setStreaming(false);
            setMessage("Stream stopped.");
        } catch (e) {
            setMessage(e instanceof Error ? e.message : "Stop failed.");
        } finally {
            setBusy(false);
        }
    };

    const capture = async () => {
        setBusy(true);
        setMessage("Capturing...");
        try {
            // Capture reads from the active stream, so ensure it's running.
            if (!streaming) {
                await cameraApi.startCameraStream(selectedFormat);
                setStreaming(true);
            }
            // Send gain only if this body actually has a numeric `gain` option.
            // A DSLR exposes ISO instead, and the daemon reads `<= 0` as "leave
            // the operator's setting alone" — so sending 0 does not mean "no
            // opinion", it overrides whatever ISO they dialled in on the body.
            const gainParam = params.find((p) => p.name === "gain");
            const gain = gainParam ? Math.round(gainParam.current) : undefined;
            const bin = formats[selectedFormat]?.bin ?? 1;
            const res = await cameraApi.capture({ exposure, gain, binning: bin });
            setLastCapture(res.path ?? res.id);

            // Surface whatever the daemon flagged. These are real conditions —
            // an exposure the body could not reach, files left on the card, a
            // FITS that could not be derived — and a warning nobody sees is
            // the same as no warning.
            setCaptureWarnings(res.metadata?.warnings ?? []);

            const applied = res.metadata?.exposure;
            const shutter = res.metadata?.shutter;
            const appliedNote =
                typeof applied === "number" && Math.abs(applied - exposure) > 1e-6
                    ? ` at ${shutter ?? `${applied}s`} (asked ${exposure}s)`
                    : "";
            setMessage(`Saved ${res.width}×${res.height}${appliedNote} → ${res.path ?? res.id}`);
        } catch (e) {
            setMessage(e instanceof Error ? e.message : "Capture failed.");
        } finally {
            setBusy(false);
        }
    };

    const locked = busy;

    return (
        <div className="cam-panel">
            <div className="cam-head">
                <div className="cam-title">Camera Control</div>
                <div className="cam-transport">{hardwareTransport === "django" ? "Django relay" : "Direct daemon"}</div>
            </div>

            {!connected ? (
                <div className="cam-connect">
                    <select className="cam-select grow" value={driver} onChange={(e) => setDriver(e.target.value)}>
                        {driverOptions.map((d) => (
                            <option key={d.id} value={d.id} disabled={!d.enabled} title={d.reason}>
                                {d.label}
                                {d.enabled ? "" : " - unavailable"}
                            </option>
                        ))}
                    </select>
                    <button type="button" className="cam-btn primary" onClick={connect} disabled={busy || !selectedDriverEnabled}>
                        Connect
                    </button>
                    {hardwareTransport !== "django" && (
                        <button
                            type="button"
                            className="cam-btn"
                            onClick={scanForDevices}
                            disabled={busy}
                            title="Probe local USB buses for attached cameras. Network devices (Alpaca/INDI) are not scanned."
                        >
                            {busy ? "Scanning..." : "Scan for devices"}
                        </button>
                    )}
                    {discovery && discovery.cameras.length > 0 && (
                        <div className="cam-discovery" aria-label="Discovered cameras">
                            {discovery.cameras.map((cam) => (
                                <button
                                    key={`${cam.driver_id}-${cam.index}`}
                                    type="button"
                                    className={`cam-discovered${
                                        driver === cam.driver_id && selectedCameraIndex === cam.index
                                            ? " selected"
                                            : ""
                                    }`}
                                    onClick={() => {
                                        setDriver(cam.driver_id);
                                        setSelectedCameraIndex(cam.index);
                                    }}
                                    title={`Select ${cam.name} (${cam.driver_label} #${cam.index}), then press Connect`}
                                >
                                    <span className="cam-driver-name">{cam.name}</span>
                                    <span className="cam-driver-detail">
                                        {cam.driver_label} · #{cam.index}
                                    </span>
                                </button>
                            ))}
                        </div>
                    )}
                    {discovery && (
                        <div className="cam-driver-reasons" aria-label="Scan details">
                            {/* Probes that errored or timed out are worth surfacing:
                                "no ASI attached" and "the ASI probe broke" look the
                                same in an empty list but need different fixes. */}
                            {discovery.probes
                                .filter((p) => p.status === "failed" || p.status === "timeout")
                                .map((p) => (
                                    <div key={p.driver_id} className="cam-driver-reason">
                                        <span className="cam-driver-name">{p.driver_label}</span>
                                        <span className="cam-driver-detail">
                                            {p.status === "timeout" ? "Timed out" : "Probe failed"}
                                            {p.detail ? ` - ${p.detail}` : ""}
                                        </span>
                                    </div>
                                ))}
                            {discovery.notes.map((note) => (
                                <div key={note} className="cam-note">
                                    {note}
                                </div>
                            ))}
                        </div>
                    )}
                    {unavailableDrivers.length > 0 && (
                        <div className="cam-driver-reasons" aria-label="Unavailable camera drivers">
                            {unavailableDrivers.map((d) => (
                                <div key={d.id} className="cam-driver-reason" title={capabilityReason(d)}>
                                    <span className="cam-driver-name">{d.label}</span>
                                    <span className="cam-driver-detail">{capabilityReason(d)}</span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            ) : (
                <>
                    <div className="cam-row">
                        <select
                            className="cam-select grow"
                            value={selectedFormat}
                            onChange={(e) => setSelectedFormat(Number(e.target.value))}
                            disabled={locked || streaming}
                        >
                            {formats.map((f) => (
                                <option key={f.index} value={f.index}>
                                    {f.format} · {f.width}×{f.height} · bin{f.bin} · {f.framerate.toFixed(0)}fps
                                    {f.mode === "still" ? " · still (1 shutter/frame)" : " · live view"}
                                </option>
                            ))}
                        </select>
                        {streaming ? (
                            <button type="button" className="cam-btn" onClick={stopStream} disabled={busy}>
                                Stop
                            </button>
                        ) : (
                            <button type="button" className="cam-btn primary" onClick={startStream} disabled={locked}>
                                Start
                            </button>
                        )}
                        <button type="button" className="cam-btn ghost" onClick={refresh} disabled={busy}>
                            Refresh
                        </button>
                    </div>

                    <div className="cam-row cam-capture">
                        <label className="cam-field">
                            <span>Exposure (s)</span>
                            <input
                                className="cam-num"
                                type="number"
                                min={0}
                                step={0.1}
                                value={exposure}
                                disabled={locked}
                                onChange={(e) => setExposure(Number(e.target.value))}
                            />
                        </label>
                        <button type="button" className="cam-btn primary" onClick={capture} disabled={locked}>
                            Capture → FITS
                        </button>
                    </div>

                    {lastCapture && (
                        <div className="cam-saved" title={lastCapture}>
                            Saved: <code>{lastCapture}</code>
                        </div>
                    )}

                    {captureWarnings.length > 0 && (
                        <ul className="cam-warnings" aria-label="Capture warnings">
                            {captureWarnings.map((w, i) => (
                                <li key={`${i}-${w.slice(0, 24)}`}>{w}</li>
                            ))}
                        </ul>
                    )}

                    {formats[selectedFormat]?.mode === "still" && (
                        <div className="cam-warnings" role="note">
                            This format takes one shutter actuation per frame. Streaming it
                            fires the mechanical shutter continuously — use a live-view
                            format for framing and focusing if the body offers one.
                        </div>
                    )}

                    {params.length > 0 && (
                        <div className="cam-params">
                            {params.map((p) => (
                                <ParamRow
                                    key={p.name}
                                    param={p}
                                    disabled={locked || p.read_only}
                                    onChange={(v) => setParam(p.name, v)}
                                />
                            ))}
                        </div>
                    )}
                </>
            )}

            <div className="cam-status">{message} Control path: {hardwareTransport === "django" ? "Django relay" : "direct daemon"}.</div>
        </div>
    );
}

function ParamRow({
    param,
    disabled,
    onChange,
}: {
    param: CamParam;
    disabled: boolean;
    onChange: (value: number) => void;
}) {
    if (param.kind === "bool") {
        return (
            <label className="cam-param cam-param-bool">
                <span className="cam-param-label">{humanize(param.name)}</span>
                <input
                    className="cam-switch"
                    type="checkbox"
                    checked={param.current > 0.5}
                    disabled={disabled}
                    onChange={(e) => onChange(e.target.checked ? 1 : 0)}
                />
            </label>
        );
    }

    // A selection option's value is an index into `choices`; the label is the
    // only part with physical meaning. Rendering index 17 on a slider gives the
    // operator a control they cannot read, which is not manual control — so
    // these get a <select> showing "1/125", not a number line from 0 to 51.
    if (param.kind === "selection" && param.choices.length > 0) {
        const index = Math.round(param.current);
        return (
            <div className="cam-param cam-param-selection">
                <div className="cam-param-top">
                    <span className="cam-param-label">
                        {humanize(param.name)}
                        {param.read_only ? " · read-only" : ""}
                    </span>
                    <span className="cam-param-val">
                        {param.choices[index] ?? `#${index}`}
                    </span>
                </div>
                <select
                    className="cam-select"
                    value={index}
                    disabled={disabled}
                    onChange={(e) => onChange(Number(e.target.value))}
                >
                    {param.choices.map((label, i) => (
                        <option key={`${label}-${i}`} value={i}>
                            {label}
                        </option>
                    ))}
                </select>
            </div>
        );
    }

    const step = param.step > 0 ? param.step : 1;

    return (
        <div className="cam-param">
            <div className="cam-param-top">
                <span className="cam-param-label">
                    {humanize(param.name)}
                    {param.read_only ? " · read-only" : ""}
                </span>
                <span className="cam-param-val">{fmtVal(param.current)}</span>
            </div>
            {!param.read_only && (
                <div className="cam-param-ctl">
                    <input
                        className="cam-slider"
                        type="range"
                        min={param.min}
                        max={param.max}
                        step={step}
                        value={param.current}
                        disabled={disabled}
                        onChange={(e) => onChange(Number(e.target.value))}
                    />
                    <input
                        className="cam-num"
                        type="number"
                        min={param.min}
                        max={param.max}
                        step={step}
                        value={param.current}
                        disabled={disabled}
                        onChange={(e) => onChange(Number(e.target.value))}
                    />
                </div>
            )}
        </div>
    );
}
