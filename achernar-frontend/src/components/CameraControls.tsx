import { useCallback, useEffect, useState } from "react";
import { daemonApi, type CamParam, type CamFormat, type DriverCapability } from "../api/daemon";

/**
 * Manual camera control panel.
 *
 * Philosophy: the operator always has direct, NINA/SharpCap-style control over
 * every supported camera parameter. AI/autonomy is an opt-in layer on top — in
 * "Autonomous" mode the same controls are shown read-only while a model drives
 * them, and the operator can reclaim manual control at any time.
 *
 * Talks to the local daemon directly (CORS-permissive on :3030).
 */

type Mode = "manual" | "auto";

const DRIVERS: DriverCapability[] = [
    { id: "svb", label: "SVBony", compiled: true, enabled: true },
    { id: "asi", label: "ZWO ASI", compiled: true, enabled: true },
    { id: "toup", label: "ToupTek", compiled: true, enabled: true },
    { id: "uvc", label: "UVC / Webcam", compiled: true, enabled: true },
    { id: "gphoto2", label: "DSLR (gphoto2)", compiled: true, enabled: true },
    { id: "indi", label: "INDI", compiled: true, enabled: true },
    { id: "indigo", label: "INDIGO", compiled: true, enabled: true },
    { id: "alpaca", label: "Alpaca", compiled: true, enabled: true },
    { id: "wdir", label: "Watch Folder", compiled: true, enabled: true },
    { id: "sim", label: "Simulator", compiled: true, enabled: true },
];

function humanize(name: string): string {
    return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function fmtVal(v: number): string {
    return Number.isInteger(v) ? String(v) : String(Number(v.toFixed(2)));
}

export default function CameraControls() {
    const [mode, setMode] = useState<Mode>("manual");
    const [params, setParams] = useState<CamParam[]>([]);
    const [formats, setFormats] = useState<CamFormat[]>([]);
    const [selectedFormat, setSelectedFormat] = useState(0);
    const [connected, setConnected] = useState(false);
    const [driver, setDriver] = useState("svb");
    const [driverCapabilities, setDriverCapabilities] = useState<DriverCapability[] | null>(null);
    const [streaming, setStreaming] = useState(false);
    const [message, setMessage] = useState("No camera connected.");
    const [busy, setBusy] = useState(false);
    const [exposure, setExposure] = useState(1);
    const [lastCapture, setLastCapture] = useState<string | null>(null);

    const refresh = useCallback(async () => {
        try {
            const [p, f] = await Promise.all([
                daemonApi.getCameraParameters(),
                daemonApi.getCameraFormats(),
            ]);
            setParams(p);
            setFormats(f);
            setConnected(true);
            setMessage(`${p.length} controls available.`);
        } catch {
            setConnected(false);
            setParams([]);
            setFormats([]);
            setMessage("No camera connected.");
        }
    }, []);

    useEffect(() => {
        refresh();
    }, [refresh]);

    useEffect(() => {
        let cancelled = false;

        daemonApi
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
    }, []);

    const driverOptions = driverCapabilities ?? DRIVERS;
    const selectedDriver = driverOptions.find((d) => d.id === driver);
    const selectedDriverEnabled = selectedDriver?.enabled ?? true;

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
            const res = await daemonApi.connectCamera(driver);
            setMessage(`Connected: ${res.camera ?? driver}`);
            await refresh();
        } catch (e) {
            setMessage(e instanceof Error ? e.message : "Connect failed.");
        } finally {
            setBusy(false);
        }
    };

    const setParam = async (name: string, value: number) => {
        // Optimistic update, then reconcile with the value the hardware applied.
        setParams((prev) => prev.map((p) => (p.name === name ? { ...p, current: value } : p)));
        try {
            const res = await daemonApi.setCameraParameter(name, value);
            setParams((prev) => prev.map((p) => (p.name === name ? { ...p, current: res.value } : p)));
        } catch {
            setMessage(`Failed to set ${name}.`);
        }
    };

    const startStream = async () => {
        setBusy(true);
        try {
            const res = await daemonApi.startCameraStream(selectedFormat);
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
            await daemonApi.stopCameraStream();
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
                await daemonApi.startCameraStream(selectedFormat);
                setStreaming(true);
            }
            const gainParam = params.find((p) => p.name === "gain");
            const gain = gainParam ? Math.round(gainParam.current) : 0;
            const bin = formats[selectedFormat]?.bin ?? 1;
            const res = await daemonApi.capture({ exposure, gain, binning: bin });
            setLastCapture(res.path ?? res.id);
            setMessage(`Saved ${res.width}×${res.height} → ${res.path ?? res.id}`);
        } catch (e) {
            setMessage(e instanceof Error ? e.message : "Capture failed.");
        } finally {
            setBusy(false);
        }
    };

    const locked = mode === "auto" || busy;

    return (
        <div className="cam-panel">
            <div className="cam-head">
                <div className="cam-title">Camera Control</div>
                <div className="cam-mode" role="group" aria-label="Control mode">
                    <button
                        type="button"
                        className={`cam-mode-btn ${mode === "manual" ? "active" : ""}`}
                        onClick={() => setMode("manual")}
                    >
                        Manual
                    </button>
                    <button
                        type="button"
                        className={`cam-mode-btn ${mode === "auto" ? "active" : ""}`}
                        onClick={() => setMode("auto")}
                    >
                        Autonomous
                    </button>
                </div>
            </div>

            {mode === "auto" && (
                <div className="cam-note">
                    AI is driving the camera. Switch to Manual to take direct control at any time.
                </div>
            )}

            {!connected ? (
                <div className="cam-connect">
                    <select className="cam-select grow" value={driver} onChange={(e) => setDriver(e.target.value)}>
                        {driverOptions.map((d) => (
                            <option key={d.id} value={d.id} disabled={!d.enabled} title={d.reason}>
                                {d.label}
                                {d.enabled ? "" : " (not compiled)"}
                            </option>
                        ))}
                    </select>
                    <button type="button" className="cam-btn primary" onClick={connect} disabled={busy || !selectedDriverEnabled}>
                        Connect
                    </button>
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

            <div className="cam-status">{message}</div>
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
