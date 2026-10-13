import { env } from "../env";
import { httpJson } from "./http";
import type {
    CaptureParams,
    CaptureRetentionDecision,
    DaemonSummary,
    EquipmentState,
    ImagingSession,
    RelayCommandResult,
} from "../types/achernar";
import { getHardwareTransport } from "../lib/runtime";

/** A single manually-controllable camera option (from GET /camera/parameters). */
export interface CamParam {
    name: string;
    label?: string;
    unit?: string;
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

export interface CameraCapabilities {
    supports_stream: boolean;
    supports_dedicated_still: boolean;
    supports_exposure_abort: boolean;
    supports_cooling: boolean;
    supports_temperature: boolean;
    supports_hardware_binning: boolean;
    supports_roi: boolean;
}

/** One authoritative snapshot from GET /camera, including acquisition routing. */
export interface CameraDescriptor {
    connected: boolean;
    connection_id: string | null;
    name: string | null;
    driver_id: string | null;
    device_index: number | null;
    hardware_id: string | null;
    activity: "disconnected" | "connecting" | "idle" | "streaming" | "exposing"
        | "downloading" | "processing" | "disconnecting" | "error";
    streaming: boolean;
    active_format_index: number | null;
    operation_id: string | null;
    exposure_progress: number | null;
    temperature_c: number | null;
    cooler_on: boolean | null;
    last_error: string | null;
    capabilities: CameraCapabilities;
    formats: CamFormat[];
    controls: CamParam[];
    latest_frame: {
        stream_generation: number;
        counter: number;
        timestamp: number;
        width: number;
        height: number;
        format: string;
    } | null;
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
    /** Identity within this scan; USB enumeration can change after reconnect. */
    selection_key?: string;
    hardware_id?: string | null;
    transport?: string;
}

export type DiscoverableEquipment = "camera" | "mount" | "focuser" | "filter_wheel";

/** A device reported by an explicitly configured Alpaca server. */
export interface DiscoveredAuxiliaryDevice {
    driver_id: string;
    driver_label: string;
    index: number;
    name: string;
    selection_key: string;
    hardware_id: string | null;
    transport: string;
    availability: "reported";
    /** Includes the server's device number; preserve this exact connect option. */
    option: string;
}

export interface DiscoveryRequest {
    equipment_type?: DiscoverableEquipment | "all";
    driver?: "alpaca";
    /** An operator-supplied endpoint, never an invented network target. */
    option?: string;
}

export interface DeviceDiscoveryProbe {
    equipment_type: DiscoverableEquipment;
    driver_id?: string;
    driver_label?: string;
    status: string;
    device_count?: number;
    detail?: string;
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
    mounts?: DiscoveredAuxiliaryDevice[];
    focusers?: DiscoveredAuxiliaryDevice[];
    filter_wheels?: DiscoveredAuxiliaryDevice[];
    device_probes?: DeviceDiscoveryProbe[];
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
    /** Effective policy and camera-copy evidence, separate from the local archive. */
    retention: CaptureRetentionDecision;
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
/**
 * Command arguments are not readings — a missing one must fail loudly.
 *
 * These used to be `Number(payload.ra ?? 0)`. A dropped field did not error,
 * it became an instruction: `mount.sync` with no arguments told the mount its
 * current position was RA 0h Dec +0deg, silently corrupting the pointing model
 * for the rest of the session, and `focuser.move_absolute` commanded step 0,
 * which on an absolute focuser means driving toward a hard stop.
 */
function requireNumber(payload: Record<string, unknown>, key: string, action: string): number {
    const value = payload[key];
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new Error(`${action} requires a numeric "${key}" (got ${JSON.stringify(value)}).`);
    }
    return value;
}

function requireString(payload: Record<string, unknown>, key: string, action: string): string {
    const value = payload[key];
    if (typeof value !== "string" || value.length === 0) {
        throw new Error(`${action} requires a non-empty "${key}" (got ${JSON.stringify(value)}).`);
    }
    return value;
}

function requireBoolean(payload: Record<string, unknown>, key: string, action: string): boolean {
    const value = payload[key];
    if (typeof value !== "boolean") {
        throw new Error(`${action} requires a boolean "${key}" (got ${JSON.stringify(value)}).`);
    }
    return value;
}

export interface FocusRunResponse {
    run_id: string;
    state: string;
    best_position: number | null;
    best_hfr: number | null;
    samples_count: number;
}

export interface StartFocusRunRequest {
    method?: string;
    filter_name?: string;
    step_size?: number;
    /** "monotonic" (default) or "reversal". */
    sweep_pattern?: string;
    /** Positions either side of start. 3 = seven-point; campaign nights use 12. */
    sweep_radius_points?: number;
}

/** One entry of the daemon's published registry (GET /api/v1/commands). */
export interface CommandSpec {
    name: string;
    /** Method of the daemon's own route for this command. */
    method: "GET" | "POST";
    /** Path relative to /api/v1, with no leading slash. */
    path: string;
    description: string;
}

// ═════════════════════════════════════════════════════════════════════════
//  Transport
//
//  There used to be two clients — `daemon.ts` and `cloud.ts` — implementing
//  the same operations twice, with `getHardwareTransport()` choosing between
//  them at every call site. Every new capability had to be added in both, and
//  the two silently diverged: that is how autofocus ended up reachable on one
//  transport only.
//
//  Now one place decides how a command travels, and callers never know.
// ═════════════════════════════════════════════════════════════════════════

/**
 * The daemon publishes what it can do and where each command lives, the same
 * way it already publishes `GET /drivers`. Fetched once and reused, so the
 * action→route map is read rather than hand-written — before the daemon
 * published this, collapsing the two clients meant writing that map a third
 * time.
 */
let registryPromise: Promise<Map<string, CommandSpec>> | null = null;

export async function commandRegistry(): Promise<Map<string, CommandSpec>> {
    if (!registryPromise) {
        registryPromise = httpJson<CommandSpec[]>(`${env.daemonApiBase}/commands`)
            .then((specs) => new Map(specs.map((spec) => [spec.name, spec])))
            .catch((error) => {
                // Don't cache a failure: a daemon that starts later should work
                // without a page reload.
                registryPromise = null;
                throw error;
            });
    }
    return registryPromise;
}

/** Arguments the daemon takes as a query parameter rather than a body. */
const QUERY_PARAM_COMMANDS: Record<string, string> = {
    "camera.start_stream": "format_index",
};

/**
 * Deliver one command, by whichever transport this runtime uses.
 *
 * Browser → Django relay, which forwards `{action, payload}` verbatim.
 * Desktop → the daemon's own route for that action, read from its registry.
 */
export function usesDjangoRelay(): boolean {
    // `enableBrowserDirectDaemon` is the escape hatch for a browser on the same
    // LAN as the rig: nominally the Django transport, but pointed straight at
    // the daemon. It used to be checked at one call site only, so most commands
    // ignored it. Deciding it here means every command honours it.
    return getHardwareTransport() === "django" && !env.enableBrowserDirectDaemon;
}

async function send(
    action: string,
    payload: Record<string, unknown> = {},
): Promise<RelayCommandResult> {
    if (usesDjangoRelay()) {
        // Opening a DSLR, changing its configuration and downloading a still
        // can exceed the relay's default three-second acknowledgement window.
        // Wait for the result rather than treating a queued acknowledgement as
        // a failed operation while the camera continues in the background.
        const waitSeconds = action === "camera.capture" && typeof payload.exposure === "number"
            && Number.isFinite(payload.exposure)
            ? Math.min(140, Math.max(15, payload.exposure + 65))
            : action === "equipment.connect" || action === "equipment.discover" ? 90
                : action === "camera.set_parameter" ? 15 : undefined;
        return httpJson<RelayCommandResult>(`${env.cloudApiBase}/equipment/command/`, {
            method: "POST",
            body: { action, payload, ...(waitSeconds === undefined ? {} : { wait_seconds: waitSeconds }) },
        });
    }

    const spec = (await commandRegistry()).get(action);
    if (!spec) {
        throw new Error(
            `The daemon does not publish "${action}". See GET /api/v1/commands for what this build handles.`,
        );
    }

    const base = `${env.daemonApiBase}/${spec.path}`;

    if (spec.method === "GET") {
        return { ok: true, payload: await httpJson<Record<string, unknown>>(base) };
    }

    const queryKey = QUERY_PARAM_COMMANDS[action];
    if (queryKey) {
        const value = payload[queryKey];
        const url = value == null ? base : `${base}?${queryKey}=${value}`;
        return { ok: true, payload: await httpJson<Record<string, unknown>>(url, { method: "POST" }) };
    }

    return {
        ok: true,
        payload: await httpJson<Record<string, unknown>>(base, { method: "POST", body: payload }),
    };
}

/** Send a command and unwrap its payload, turning relay failures into throws. */
async function sendFor<T>(action: string, payload: Record<string, unknown> = {}): Promise<T> {
    const result = await send(action, payload);
    if (result.pending) {
        throw new Error(`${action} is still running on the daemon.`);
    }
    if (result.ok === false) {
        throw new Error(result.error ?? `${action} failed.`);
    }
    return (result.payload ?? {}) as T;
}

// ═════════════════════════════════════════════════════════════════════════
//  The client
// ═════════════════════════════════════════════════════════════════════════

export const achernarApi = {
    // ── Transport-agnostic: these work on either transport ───────────────

    /** Raw command escape hatch, for callers that already have an action name. */
    command: send,

    async getStatus(): Promise<EquipmentState> {
        return sendFor<EquipmentState>("equipment.status");
    },

    async getDriverCapabilities(): Promise<DriverCapabilityReport> {
        return sendFor<DriverCapabilityReport>("drivers.capabilities");
    },

    /** `index` selects which camera within the driver's enumeration. */
    async connectCamera(
        driver: string,
        option?: string,
        index?: number,
    ): Promise<{ status: string; camera?: string; cameras_found?: string[]; index?: number }> {
        return sendFor("equipment.connect", { equipment_type: "camera", driver, option, index });
    },

    async disconnect(equipmentType: string): Promise<Record<string, unknown>> {
        return sendFor("equipment.disconnect", { equipment_type: equipmentType });
    },

    async getCameraParameters(): Promise<CamParam[]> {
        return sendFor<CamParam[]>("camera.parameters");
    },

    async getCameraDescriptor(): Promise<CameraDescriptor> {
        return sendFor<CameraDescriptor>("camera.descriptor");
    },

    async setCameraParameter(name: string, value: number): Promise<{ status: string; name: string; value: number }> {
        requireString({ name }, "name", "camera.set_parameter");
        return sendFor("camera.set_parameter", { name, value: requireNumber({ value }, "value", "camera.set_parameter") });
    },

    async getCameraFormats(): Promise<CamFormat[]> {
        return sendFor<CamFormat[]>("camera.formats");
    },

    async startCameraStream(formatIndex?: number): Promise<{ status: string; format: string; width: number; height: number; bin?: number }> {
        return sendFor("camera.start_stream", typeof formatIndex === "number" ? { format_index: formatIndex } : {});
    },

    /** Strict stop: errors when no camera is connected. */
    async stopCameraStream(): Promise<void> {
        await sendFor("camera.stop_stream");
    },

    /** Idempotent stop: succeeds even when no camera is connected. */
    async abortCamera(): Promise<void> {
        await sendFor("camera.abort");
    },

    async capture(payload: CaptureParams): Promise<CaptureResult> {
        return sendFor<CaptureResult>("camera.capture", payload as unknown as Record<string, unknown>);
    },

    async cameraSnapshot(): Promise<{
        status: string;
        format: string;
        mime_type?: string;
        width: number;
        height: number;
        frame_counter?: number;
        timestamp?: number;
        data_base64: string;
    }> {
        return sendFor("camera.snapshot");
    },

    async slewMount(payload: { ra: number; dec: number }): Promise<Record<string, unknown>> {
        return sendFor("mount.slew", {
            ra: requireNumber({ ra: payload.ra }, "ra", "mount.slew"),
            dec: requireNumber({ dec: payload.dec }, "dec", "mount.slew"),
        });
    },

    async syncMount(payload: { ra: number; dec: number }): Promise<Record<string, unknown>> {
        return sendFor("mount.sync", {
            ra: requireNumber({ ra: payload.ra }, "ra", "mount.sync"),
            dec: requireNumber({ dec: payload.dec }, "dec", "mount.sync"),
        });
    },

    async setTracking(enabled: boolean): Promise<Record<string, unknown>> {
        // A missing `enabled` used to mean "stop tracking", which drifts the
        // target out of frame.
        requireBoolean({ enabled }, "enabled", "mount.set_tracking");
        return sendFor("mount.set_tracking", { enabled });
    },

    async abortMount(): Promise<Record<string, unknown>> {
        return sendFor("mount.abort");
    },

    async moveFocuser(payload: { steps: number }): Promise<Record<string, unknown>> {
        return sendFor("focuser.move_relative", {
            steps: requireNumber({ steps: payload.steps }, "steps", "focuser.move_relative"),
        });
    },

    async moveFocuserTo(position: number): Promise<Record<string, unknown>> {
        // Step 0 is a real position, so a missing argument must not default to it.
        return sendFor("focuser.move_absolute", {
            position: requireNumber({ position }, "position", "focuser.move_absolute"),
        });
    },

    async focuserPosition(): Promise<{ position: number | null }> {
        return sendFor("focuser.position");
    },

    async abortFocuser(): Promise<Record<string, unknown>> {
        return sendFor("focuser.abort");
    },

    // ── Daemon resources ─────────────────────────────────────────────────
    //
    // Discovery is relayed to the selected daemon. Frame sockets and the
    // autofocus APIs below retain their explicit local-daemon behavior.

    /**
     * Probes the selected daemon's buses or an operator-supplied endpoint.
     * The relay forwards this action to that daemon; it does not scan cloud
     * infrastructure or infer network addresses.
     */
    async discoverDevices(request?: DiscoveryRequest): Promise<DiscoveryReport> {
        if (usesDjangoRelay()) return sendFor<DiscoveryReport>("equipment.discover", { ...request });
        return httpJson<DiscoveryReport>(`${env.daemonApiBase}/discover`, {
            method: "POST",
            ...(request ? { body: request } : {}),
        });
    },

    /** Live frame socket. A WebSocket to the rig, not a command. */
    cameraStreamWsUrl(): string {
        const url = new URL(env.daemonApiBase);
        url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
        url.pathname = `${url.pathname.replace(/\/$/, "")}/camera/stream`;
        url.search = "";
        return url.toString();
    },

    /**
     * Autofocus. Deliberately not relayed — see the note above
     * `command_registry!` in the daemon's relay_commands.rs. Starting a sweep
     * from the cloud without FOCUS_* event forwarding gives you a run you
     * cannot watch, and the LAN case the direct API already serves.
     */
    async startFocusRun(req: StartFocusRunRequest = {}): Promise<FocusRunResponse> {
        return httpJson(`${env.daemonApiBase}/focus/run`, { method: "POST", body: req });
    },

    async abortFocusRun(): Promise<void> {
        await httpJson(`${env.daemonApiBase}/focus/abort`, { method: "POST" });
    },

    async getFocusStatus(): Promise<FocusRunResponse> {
        return httpJson(`${env.daemonApiBase}/focus/status`);
    },

    async getFocusHistory(): Promise<FocusRunResponse[]> {
        return httpJson(`${env.daemonApiBase}/focus/history`);
    },

    // ── Cloud only ───────────────────────────────────────────────────────
    //
    // Records Django keeps that no single daemon has.

    async listDaemons(): Promise<DaemonSummary[]> {
        return httpJson<DaemonSummary[]>(`${env.cloudApiBase}/equipment/daemons/`);
    },

    async listSessions(): Promise<ImagingSession[]> {
        return httpJson<ImagingSession[]>(`${env.cloudApiBase}/imaging/sessions/`);
    },
};
