import { useCallback, useEffect, useRef, useState } from "react";
import {
    achernarApi,
    usesDjangoRelay,
    type CamParam,
    type CameraDescriptor,
    type DriverCapability,
    type DiscoveryReport,
} from "../api/achernar";
import { getHardwareTransport } from "../lib/runtime";
import { useDaemonStatus } from "../hooks/useDaemonStatus";
import { captureRetentionSummary } from "./captureRetention";
import { canStartCameraPreview, captureCameraBundle } from "./captureCameraBundle";

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
    const liveStatus = useDaemonStatus();
    const liveCamera = liveStatus.data?.camera;
    const liveCameraRef = useRef(liveCamera);
    const descriptorInFlight = useRef(false);
    liveCameraRef.current = liveCamera;
    const [camera, setCamera] = useState<CameraDescriptor | null>(null);
    const [selectedFormat, setSelectedFormat] = useState(0);
    const [driver, setDriver] = useState("");
    const [driverCapabilities, setDriverCapabilities] = useState<DriverCapability[] | null>(null);
    const [driverReportFailed, setDriverReportFailed] = useState(false);
    const [discovery, setDiscovery] = useState<DiscoveryReport | null>(null);
    const [message, setMessage] = useState("No camera connected.");
    const [busy, setBusy] = useState(false);
    const [exposure, setExposure] = useState(1);
    const [lastCapture, setLastCapture] = useState<string | null>(null);
    const [captureWarnings, setCaptureWarnings] = useState<string[]>([]);
    const [captureRetention, setCaptureRetention] = useState<string | null>(null);
    /**
     * Which camera within the selected driver's enumeration. Set by picking a
     * device from a scan; 0 when the operator chose a driver by hand, which is
     * the right default for the single-camera case.
     */
    const [selectedCameraIndex, setSelectedCameraIndex] = useState(0);
    const params = camera?.controls ?? [];
    const formats = camera?.formats ?? [];
    const connected = liveCamera?.connected === true;
    const streaming = liveCamera?.streaming === true;
    const selectedCameraFormat = formats.find((format) => format.index === selectedFormat);
    const previewAvailable = canStartCameraPreview(camera, selectedCameraFormat);

    const refresh = useCallback(async () => {
        if (descriptorInFlight.current) return;
        descriptorInFlight.current = true;
        try {
            const descriptor = await achernarApi.getCameraDescriptor();
            const current = liveCameraRef.current;
            if (current?.connected !== true || (current.connection_id && current.connection_id !== descriptor.connection_id)) {
                setCamera(null);
                return;
            }
            setCamera(descriptor);
            const liveview = descriptor.formats.find((format) => format.mode === "liveview");
            setSelectedFormat(descriptor.streaming && descriptor.active_format_index !== null
                ? descriptor.active_format_index
                : (liveview?.index ?? descriptor.formats[0]?.index ?? 0));
            if (descriptor.connected) {
                if (descriptor.driver_id) setDriver(descriptor.driver_id);
                if (descriptor.device_index !== null) setSelectedCameraIndex(descriptor.device_index);
            }
            setMessage(descriptor.connected
                ? `${descriptor.controls.length} controls available.`
                : "No camera connected.");
        } catch (error) {
            setCamera(null);
            setMessage(error instanceof Error ? error.message : "Camera descriptor unavailable.");
        } finally {
            descriptorInFlight.current = false;
        }
    }, []);

    useEffect(() => {
        if (liveCamera?.connected !== true) {
            setCamera(null);
            setMessage(liveStatus.online ? "No camera connected." : "Camera state unavailable.");
            return;
        }
        void refresh();
    }, [refresh, liveCamera?.connected, liveCamera?.connection_id, liveStatus.online]);

    // A passive health probe or a just-completed connect can briefly own the
    // acquisition gate. Keep the authoritative connection visible and retry
    // the optional controls descriptor instead of offering a second connect.
    useEffect(() => {
        if (!connected || camera || busy) return;
        let cancelled = false;
        let retry: number;
        const attempt = async () => {
            await refresh();
            if (!cancelled) retry = window.setTimeout(attempt, 1000);
        };
        retry = window.setTimeout(attempt, 750);
        return () => { cancelled = true; window.clearTimeout(retry); };
    }, [connected, camera, busy, refresh]);

    useEffect(() => {
        let cancelled = false;

        achernarApi
            .getDriverCapabilities()
            .then((report) => {
                if (!cancelled) {
                    setDriverCapabilities(report.camera);
                    setDriverReportFailed(false);
                }
            })
            .catch(() => {
                if (!cancelled) {
                    setDriverCapabilities(null);
                    setDriverReportFailed(true);
                }
            });

        return () => {
            cancelled = true;
        };
    }, []);

    const driverOptions = driverCapabilities ?? [];
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
        if (driver) setDriver("");
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
            const res = await achernarApi.connectCamera(driver, undefined, selectedCameraIndex);
            setMessage(`Connected: ${res.camera ?? driver}`);
            await liveStatus.refetch();
        } catch (e) {
            setMessage(e instanceof Error ? e.message : "Connect failed.");
        } finally {
            setBusy(false);
        }
    };

    const disconnect = async () => {
        setBusy(true);
        try {
            await achernarApi.disconnect("camera");
            setCamera(null);
            setMessage("Camera disconnected.");
            await liveStatus.refetch();
        } catch (error) {
            setMessage(error instanceof Error ? error.message : "Disconnect failed.");
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
            const report = await achernarApi.discoverDevices();
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
        setBusy(true);
        setMessage(`Applying ${params.find((p) => p.name === name)?.label ?? humanize(name)}...`);
        try {
            const res = await achernarApi.setCameraParameter(name, value);
            // Refresh the body's labels and restrictions too: changing one
            // control can change the choices available for another.
            const controls = await achernarApi.getCameraParameters();
            setCamera((previous) => previous && {
                ...previous,
                controls,
            });
            const applied = controls.find((p) => p.name === name);
            const label = applied?.kind === "selection"
                ? applied.choices[Math.round(applied.current)]
                : applied?.kind === "bool"
                    ? applied.current > 0.5 ? "on" : "off"
                    : fmtVal(applied?.current ?? res.value);
            setMessage(`${applied?.label ?? humanize(name)}: ${label} (read back from camera).`);
        } catch (error) {
            // Restore camera-authored state after a refused write instead of
            // leaving an optimistic value looking like it was accepted.
            try {
                const controls = await achernarApi.getCameraParameters();
                setCamera((previous) => previous && { ...previous, controls });
            } catch {
                // Preserve the original setting failure if readback also fails.
            }
            setMessage(error instanceof Error ? error.message : `Failed to set ${name}.`);
        } finally {
            setBusy(false);
        }
    };

    const startStream = async () => {
        if (!previewAvailable) {
            setMessage("Select a live-view format for preview; dedicated still exposures use Capture bundle.");
            return;
        }
        setBusy(true);
        try {
            const res = await achernarApi.startCameraStream(selectedFormat);
            setCamera((previous) => previous && { ...previous, streaming: true, active_format_index: selectedFormat });
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
            await achernarApi.stopCameraStream();
            setCamera((previous) => previous && { ...previous, streaming: false });
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
        setCaptureRetention(null);
        try {
            // Send gain only if this body actually has a numeric `gain` option.
            // A DSLR exposes ISO instead, and the daemon reads `<= 0` as "leave
            // the operator's setting alone" — so sending 0 does not mean "no
            // opinion", it overrides whatever ISO they dialled in on the body.
            const gainParam = params.find((p) => p.name === "gain");
            const gain = gainParam ? Math.round(gainParam.current) : undefined;
            const res = await captureCameraBundle(
                achernarApi,
                camera,
                selectedCameraFormat,
                { exposure, gain },
                () => setCamera((previous) => previous && {
                    ...previous,
                    streaming: true,
                    active_format_index: selectedFormat,
                }),
            );
            setLastCapture(res.path ?? res.id);
            setCaptureRetention(captureRetentionSummary(res.retention));

            // Surface whatever the daemon flagged. These are real conditions —
            // an exposure the body could not reach, files left on the card, a
            // FITS that could not be derived — and a warning nobody sees is
            // the same as no warning.
            setCaptureWarnings(res.metadata?.warnings ?? []);

            // A dedicated capture applies its requested shutter/ISO on the
            // body. Keep the manual controls aligned with those settings.
            try {
                const controls = await achernarApi.getCameraParameters();
                setCamera((previous) => previous && { ...previous, controls });
            } catch {
                // A later status-read failure does not invalidate a saved capture.
            }

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
                    <select
                        className="cam-select grow"
                        value={driver}
                        onChange={(e) => setDriver(e.target.value)}
                        disabled={driverOptions.length === 0 || !liveStatus.online}
                    >
                        {driverOptions.length === 0 ? (
                            <option value="">
                                {driverReportFailed
                                    ? "Driver list unavailable - daemon unreachable"
                                    : "Reading driver list from daemon..."}
                            </option>
                        ) : (
                            <><option value="">Choose a camera driver</option>{driverOptions.map((d) => (
                                <option key={d.id} value={d.id} disabled={!d.enabled} title={d.reason}>
                                    {d.label}
                                    {d.enabled ? "" : " - unavailable"}
                                </option>
                            ))}</>
                        )}
                    </select>
                    <button type="button" className="cam-btn primary" onClick={connect} disabled={busy || !selectedDriverEnabled || !liveStatus.online}>
                        Connect
                    </button>
                    {!usesDjangoRelay() && (
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
                                    {f.format} · {f.width}×{f.height} · bin{f.bin}
                                    {f.mode !== "still" && (camera?.driver_id === "gphoto2"
                                        ? ` · preview cap ${f.framerate.toFixed(0)}fps`
                                        : ` · ${f.framerate.toFixed(0)}fps`)}
                                    {f.mode === "still" ? " · still (1 shutter/frame)" : " · live view"}
                                </option>
                            ))}
                        </select>
                        {streaming ? (
                            <button type="button" className="cam-btn" onClick={stopStream} disabled={busy}>
                                Stop
                            </button>
                        ) : (
                            <button
                                type="button"
                                className="cam-btn primary"
                                onClick={startStream}
                                disabled={locked || !previewAvailable}
                                title={previewAvailable ? "Start preview" : "Dedicated still exposures use Capture bundle; select live view for preview"}
                            >
                                Start
                            </button>
                        )}
                        <button type="button" className="cam-btn ghost" onClick={refresh} disabled={busy}>
                            Refresh
                        </button>
                        <button type="button" className="cam-btn ghost" onClick={disconnect} disabled={busy}>Disconnect</button>
                    </div>

                    {camera?.driver_id === "gphoto2" && <p className="cam-note">
                        Live-view speed depends on camera shutter and lighting. A long shutter can slow the preview;
                        try a shorter shutter for smoother framing, then restore your exposure settings before capture.
                    </p>}
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
                            Capture bundle
                        </button>
                    </div>

                    <div className="cam-note" role="note" aria-label="Capture retention policy">
                        <strong>Keep camera originals</strong>. Capture requests retain camera originals;
                        automatic deletion is unavailable. A persistent camera copy must be verified separately.
                    </div>

                    {lastCapture && (
                        <div className="cam-saved" title={lastCapture}>
                            Saved: <code>{lastCapture}</code>
                        </div>
                    )}

                    {captureRetention && (
                        <div className="cam-note" role="status" aria-label="Capture retention outcome">
                            {captureRetention}
                        </div>
                    )}

                    {captureWarnings.length > 0 && (
                        <ul className="cam-warnings" aria-label="Capture warnings">
                            {captureWarnings.map((w, i) => (
                                <li key={`${i}-${w.slice(0, 24)}`}>{w}</li>
                            ))}
                        </ul>
                    )}

                    {camera?.capabilities.supports_dedicated_still && selectedCameraFormat?.mode === "still" && (
                        <div className="cam-warnings" role="note">
                            Capture bundle takes one dedicated still exposure and saves its native files.
                            Select a live-view format to start a preview.
                        </div>
                    )}

                    {params.length > 0 && (
                        <div className="cam-params">
                            {params.filter((p) => p.name !== "keep_images").map((p) => (
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
    const label = param.label ?? humanize(param.name);
    const [draft, setDraft] = useState(String(param.current));
    useEffect(() => setDraft(String(param.current)), [param.current]);
    if (param.kind === "bool") {
        return (
            <label className="cam-param cam-param-bool">
                <span className="cam-param-label">{label}{param.read_only ? " · read-only" : ""}</span>
                <input
                    className="cam-switch"
                    aria-label={label}
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
                        {label}
                        {param.read_only ? " · read-only" : ""}
                    </span>
                    <span className="cam-param-val">
                        {param.choices[index] ?? `#${index}`}
                    </span>
                </div>
                <select
                    className="cam-select"
                    aria-label={label}
                    value={index}
                    disabled={disabled}
                    onChange={(e) => onChange(Number(e.target.value))}
                >
                    {param.choices.map((label, i) => (
                        <option key={`${label}-${i}`} value={i} disabled={/^unknown value\b/i.test(label)}>
                            {label}
                        </option>
                    ))}
                </select>
            </div>
        );
    }

    const step = param.step > 0 ? param.step : 1;
    const draftValue = Number(draft);
    const canApply = !disabled && draft.trim() !== "" && Number.isFinite(draftValue)
        && draftValue >= param.min && draftValue <= param.max && draftValue !== param.current;

    return (
        <div className="cam-param">
            <div className="cam-param-top">
                <span className="cam-param-label">
                    {label}{param.unit ? ` (${param.unit})` : ""}
                    {param.read_only ? " · read-only" : ""}
                </span>
                <span className="cam-param-val">{fmtVal(param.current)}</span>
            </div>
            {!param.read_only && (
                <div className="cam-param-ctl">
                    <input
                        className="cam-slider"
                        aria-label={`${label} slider`}
                        type="range"
                        min={param.min}
                        max={param.max}
                        step={step}
                        value={Number.isFinite(draftValue) ? draftValue : param.current}
                        disabled={disabled}
                        onChange={(e) => setDraft(e.target.value)}
                    />
                    <input
                        className="cam-num"
                        aria-label={label}
                        type="number"
                        min={param.min}
                        max={param.max}
                        step={step}
                        value={draft}
                        disabled={disabled}
                        onChange={(e) => setDraft(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === "Enter" && canApply) onChange(draftValue);
                        }}
                    />
                    <button type="button" className="cam-btn" disabled={!canApply} onClick={() => onChange(draftValue)}>
                        Apply
                    </button>
                </div>
            )}
        </div>
    );
}
