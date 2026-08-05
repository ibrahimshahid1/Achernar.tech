import { env } from "../env";
import { httpJson } from "./http";
import type { EquipmentState, CaptureParams, RelayCommandResult } from "../types/achernar";

/** A single manually-controllable camera option (from GET /camera/parameters). */
export interface CamParam {
    name: string;
    kind: string; // bool | number | msec | percent | kelvin | celsius | selection
    read_only: boolean;
    min: number;
    max: number;
    step: number;
    default: number;
    /**
     * The raw value the driver holds. For `kind === "selection"` this is an
     * INDEX into `choices`, not a physical quantity — index 17 is not
     * seventeen of anything.
     */
    current: number;
    /**
     * Human-readable labels for a `selection` option, in index order, so the
     * Nth entry is the value to send back for choice N. Empty for every other
     * kind.
     *
     * A DSLR exposes shutter and ISO as enumerations whose labels carry the
     * only physical meaning there is. Rendering the index instead gives the
     * operator a control they cannot read, which is not manual control.
     */
    choices: string[];
    /**
     * The label parsed into physical units — seconds for shutter, ISO speed for
     * ISO. Absent when the label does not parse ("bulb") or the option is not
     * one of those two. Display only: send `current` (the index) when setting.
     */
    current_value?: number;
}

/** A selectable stream format / binning mode (from GET /camera/formats). */
export interface CamFormat {
    index: number;
    format: string;
    width: number;
    height: number;
    bin: number;
    framerate: number;
    /**
     * How the driver produces frames in this format.
     *
     * `still` means one shutter actuation per frame. Streaming it on a DSLR
     * fires the mechanical shutter continuously — tens of thousands of
     * actuations across a session against a shutter rated for a few hundred
     * thousand. Prefer `liveview` for framing and focusing.
     */
    mode: "liveview" | "still";
}

export interface DriverCapability {
    id: string;
    label: string;
    compiled: boolean;
    enabled: boolean;
    reason?: string;
    option_hint?: string;
}

export interface DriverCapabilityReport {
    bridge_version: string;
    camera: DriverCapability[];
    mount: DriverCapability[];
    focuser: DriverCapability[];
}

/** A camera actually attached to a local bus, found by a discovery scan. */
export interface DiscoveredCamera {
    /** Driver id to pass to connectCamera (e.g. "svb"). */
    driver_id: string;
    driver_label: string;
    /** Index within that driver's enumeration. */
    index: number;
    name: string;
}

/**
 * Outcome of probing one driver. "empty" (no such camera attached) and
 * "failed" (the probe itself errored) are reported separately on purpose —
 * they mean very different things when troubleshooting in the dark.
 */
export interface DiscoveryProbe {
    driver_id: string;
    driver_label: string;
    status: "found" | "empty" | "not_compiled" | "failed" | "timeout" | "skipped";
    device_count: number;
    detail?: string;
}

export interface DiscoveryReport {
    cameras: DiscoveredCamera[];
    probes: DiscoveryProbe[];
    scanned_at: string;
    /** Caveats — what was deliberately not scanned, and why. */
    notes: string[];
}

/** Result of a single-shot capture (from POST /camera/capture). */
/** One artifact of a committed capture. A body set to RAW+JPEG yields two
 *  native assets plus a derived FITS — all belonging to the same exposure. */
export interface CapturedAsset {
    role:
        | "native_raw"
        | "camera_jpeg"
        | "sensor_fits"
        | "analysis_luma"
        | "thumbnail"
        | "calibrated_fits";
    media_type: string;
    /** Relative to the capture root, so the archive can be relocated. */
    relative_path: string;
    byte_length: number;
    sha256: string;
}

export interface CaptureResult {
    /** The capture attempt, allocated before the camera is commanded. */
    capture_id: string;
    /** The frames row; every asset below belongs to it. */
    id: string;
    session_id?: string;
    status: "committed";
    width: number;
    height: number;
    /** Directory holding the bundle, relative to the capture root. */
    relative_dir: string;
    assets: CapturedAsset[];
    /** @deprecated Single path cannot express a RAW+JPEG capture. Use `assets`. */
    path?: string;
    metadata: {
        /** What the hardware actually delivered, read back from the camera. */
        exposure: number;
        /** What was asked for. On a DSLR the two differ whenever the request
         *  falls between two steps of the shutter ladder. */
        requested_exposure: number;
        /** Signed difference between the two, in stops. */
        exposure_error_stops: number;
        gain: number;
        /** Present on bodies whose gain control is an ISO enumeration. */
        iso?: number;
        /** The camera's own shutter label, e.g. "1/125". */
        shutter?: string;
        temperature: number;
        /** Whether `temperature` was measured, estimated, or absent. A DSLR has
         *  no sensor thermometer, so it reports "missing" rather than a
         *  plausible-looking number. */
        temperature_state: "measured" | "estimated" | "missing";
        /** Start of exposure, from the driver's clock read immediately before
         *  the trigger — this is DATE-OBS, not a post-download timestamp. */
        timestamp: string;
        completed_at: string;
        /** Non-fatal problems worth showing the operator: an exposure the body
         *  could not reach, a control it refused, files left on the card. */
        warnings?: string[];
    };
}

/**
 * Daemon API (Edge Control)
 * PDF Section 5.1.1 - Daemon Endpoints
 */
export const daemonApi = {
    // GET /api/v1/status
    async getStatus(): Promise<EquipmentState> {
        return httpJson<EquipmentState>(`${env.daemonApiBase}/status`);
    },

    // GET /api/v1/drivers — build-time native hardware support report
    async getDriverCapabilities(): Promise<DriverCapabilityReport> {
        return httpJson<DriverCapabilityReport>(`${env.daemonApiBase}/drivers`);
    },

    /**
     * POST /api/v1/discover — scan local USB/SDK buses for attached cameras.
     *
     * Operator-initiated only (it instantiates vendor SDK drivers), and the
     * daemon refuses to scan while a camera is connected. Network transports
     * (Alpaca/INDI/INDIGO) are not scanned; they need an explicit address.
     */
    async discoverDevices(): Promise<DiscoveryReport> {
        return httpJson<DiscoveryReport>(`${env.daemonApiBase}/discover`, {
            method: "POST",
        });
    },

    // POST /api/v1/mount/slew
    async slewMount(payload: { ra: number; dec: number }): Promise<{ ok: boolean }> {
        return httpJson(`${env.daemonApiBase}/mount/slew`, {
            method: "POST",
            body: payload
        });
    },

    // POST /api/v1/camera/capture — single-shot, saves a FITS frame
    async capture(payload: CaptureParams): Promise<CaptureResult> {
        return httpJson<CaptureResult>(
            `${env.daemonApiBase}/camera/capture`,
            { method: "POST", body: payload }
        );
    },

    // POST /api/v1/camera/stream/start (optionally choose format/binning by index)
    async startCameraStream(formatIndex?: number): Promise<{ status: string; format: string; width: number; height: number; bin?: number }> {
        const query = typeof formatIndex === "number" ? `?format_index=${formatIndex}` : "";
        return httpJson(`${env.daemonApiBase}/camera/stream/start${query}`, {
            method: "POST",
        });
    },

    // POST /api/v1/camera/abort
    async stopCameraStream(): Promise<void> {
        await httpJson(`${env.daemonApiBase}/camera/abort`, { method: "POST" });
    },

    // GET /api/v1/camera/parameters — full manual control surface
    async getCameraParameters(): Promise<CamParam[]> {
        return httpJson<CamParam[]>(`${env.daemonApiBase}/camera/parameters`);
    },

    // POST /api/v1/camera/parameters — set one option (exposure, gain, cooler_target, ...)
    async setCameraParameter(name: string, value: number): Promise<{ status: string; name: string; value: number }> {
        return httpJson(`${env.daemonApiBase}/camera/parameters`, {
            method: "POST",
            body: { name, value },
        });
    },

    // GET /api/v1/camera/formats — selectable resolutions / binning modes
    async getCameraFormats(): Promise<CamFormat[]> {
        return httpJson<CamFormat[]>(`${env.daemonApiBase}/camera/formats`);
    },

    /**
     * POST /api/v1/connect — connect a camera driver (e.g. "svb" for SVBony).
     *
     * `index` selects which camera within that driver's enumeration, as
     * returned by `discoverDevices()`. It matters as soon as two cameras are
     * attached, which is the guiding + imaging configuration — without it the
     * daemon always opens camera 0 and the guider and the imager are the same
     * request.
     */
    async connectCamera(
        driver: string,
        option?: string,
        index?: number,
    ): Promise<{ status: string; camera?: string; cameras_found?: string[]; index?: number }> {
        return httpJson(`${env.daemonApiBase}/connect`, {
            method: "POST",
            body: { equipment_type: "camera", driver, option, index },
        });
    },

    // WS /api/v1/camera/stream
    cameraStreamWsUrl() {
        const url = new URL(env.daemonApiBase);
        url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
        url.pathname = `${url.pathname.replace(/\/$/, "")}/camera/stream`;
        url.search = "";
        return url.toString();
    },

    // POST /api/v1/focuser/move
    async moveFocuser(payload: { steps: number }): Promise<void> {
        await httpJson(`${env.daemonApiBase}/focuser/move`, {
            method: "POST",
            body: payload,
        });
    },

    async relayCommand(action: string, payload: Record<string, unknown> = {}): Promise<RelayCommandResult> {
        switch (action) {
            case "drivers.capabilities":
                return { ok: true, payload: await this.getDriverCapabilities() as unknown as Record<string, unknown> };
            case "equipment.status":
                return { ok: true, payload: await this.getStatus() as unknown as Record<string, unknown> };
            case "equipment.connect":
                return {
                    ok: true,
                    payload: await httpJson(`${env.daemonApiBase}/connect`, {
                        method: "POST",
                        body: payload,
                    }) as Record<string, unknown>,
                };
            case "camera.parameters":
                return { ok: true, payload: await this.getCameraParameters() as unknown as Record<string, unknown> };
            case "camera.formats":
                return { ok: true, payload: await this.getCameraFormats() as unknown as Record<string, unknown> };
            case "camera.set_parameter":
                return {
                    ok: true,
                    payload: await this.setCameraParameter(
                        String(payload.name ?? ""),
                        Number(payload.value ?? 0),
                    ) as unknown as Record<string, unknown>,
                };
            case "camera.start_stream":
                return {
                    ok: true,
                    payload: await this.startCameraStream(
                        typeof payload.format_index === "number" ? payload.format_index : undefined,
                    ) as unknown as Record<string, unknown>,
                };
            case "camera.abort":
                await this.stopCameraStream();
                return { ok: true, payload: { status: "stopped" } };
            case "camera.capture":
                return { ok: true, payload: await this.capture(payload as unknown as CaptureParams) as unknown as Record<string, unknown> };
            case "mount.slew":
                return {
                    ok: true,
                    payload: await this.slewMount({
                        ra: Number(payload.ra ?? 0),
                        dec: Number(payload.dec ?? 0),
                    }) as unknown as Record<string, unknown>,
                };
            case "mount.set_tracking":
                return {
                    ok: true,
                    payload: await httpJson(`${env.daemonApiBase}/mount/tracking`, {
                        method: "POST",
                        body: { enabled: Boolean(payload.enabled) },
                    }) as Record<string, unknown>,
                };
            case "mount.abort":
                return {
                    ok: true,
                    payload: await httpJson(`${env.daemonApiBase}/mount/abort`, {
                        method: "POST",
                    }) as Record<string, unknown>,
                };
            case "mount.sync":
                return {
                    ok: true,
                    payload: await httpJson(`${env.daemonApiBase}/mount/sync`, {
                        method: "POST",
                        body: {
                            ra: Number(payload.ra ?? 0),
                            dec: Number(payload.dec ?? 0),
                        },
                    }) as Record<string, unknown>,
                };
            case "focuser.move_relative":
                await this.moveFocuser({ steps: Number(payload.steps ?? 0) });
                return { ok: true, payload: { status: "moving", steps: Number(payload.steps ?? 0) } };
            case "focuser.move_absolute":
                return {
                    ok: true,
                    payload: await httpJson(`${env.daemonApiBase}/focuser/moveto`, {
                        method: "POST",
                        body: { position: Number(payload.position ?? 0) },
                    }) as Record<string, unknown>,
                };
            case "focuser.position":
                return {
                    ok: true,
                    payload: await httpJson(`${env.daemonApiBase}/focuser/position`) as Record<string, unknown>,
                };
            case "focuser.abort":
                return {
                    ok: true,
                    payload: await httpJson(`${env.daemonApiBase}/focuser/abort`, {
                        method: "POST",
                    }) as Record<string, unknown>,
                };
            default:
                throw new Error(`Unsupported local daemon action: ${action}`);
        }
    },
};
