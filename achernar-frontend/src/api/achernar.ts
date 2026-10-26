import { env } from "../env";
import { httpJson } from "./http";
import type {
    AlignmentMode,
    CaptureParams,
    CaptureRetentionDecision,
    DaemonSummary,
    DriveRate,
    EqFrame,
    EquipmentEvent,
    EquipmentState,
    FilterSlotView,
    GuidingBlock,
    ImagingSession,
    LinkState,
    Qualification,
    QualificationState,
    RelayCommandResult,
    ResumeBlock,
    SafetyGate,
    SafetyMonitorBlock,
    Site,
    WeatherBlock,
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
    /** D21. Absent from daemons older than the equipment layer. */
    qualification?: QualificationState;
}

/**
 * `GET /drivers`. The kinds beyond camera, mount and focuser are additive
 * (§2.12), so an older daemon simply omits them.
 */
export interface DriverCapabilityReport {
    bridge_version: string;
    camera: DriverCapability[];
    mount: DriverCapability[];
    focuser: DriverCapability[];
    filter_wheel?: DriverCapability[];
    guider?: DriverCapability[];
    weather?: DriverCapability[];
    safety_monitor?: DriverCapability[];
    rotator?: DriverCapability[];
    cover_calibrator?: DriverCapability[];
    switch?: DriverCapability[];
    dome?: DriverCapability[];
}

/** A connected device's identity (§2.6), as `/connect` and descriptors return it. */
export interface DeviceIdentityInfo {
    kind: string;
    driver: string;
    identity_key: string;
    connection_id: string;
    name: string;
    description?: string | null;
    driver_info?: string | null;
    driver_version?: string | null;
    interface_version?: number | null;
    unique_id?: string | null;
    /** Alpaca `/setup/v1/{type}/{n}/setup`. */
    setup_url?: string | null;
    qualification?: Qualification | null;
    endpoint?: string;
}

/** The mount's capability flags (§2.4 `MountCapabilities`). */
export interface MountCapabilities {
    can_park?: boolean;
    can_unpark?: boolean;
    can_set_park?: boolean;
    can_find_home?: boolean;
    /** False on every compliant Alpaca mount (O10); never gate on it. */
    can_slew?: boolean;
    /** GoTo is gated on this (O10). */
    can_slew_async?: boolean;
    can_slew_altaz?: boolean;
    can_slew_altaz_async?: boolean;
    can_sync?: boolean;
    can_sync_altaz?: boolean;
    can_set_tracking?: boolean;
    can_set_pier_side?: boolean;
    can_pulse_guide?: boolean;
    can_set_guide_rates?: boolean;
    can_move_axis?: boolean[];
    alignment_mode?: AlignmentMode | null;
    equatorial_system?: number | null;
    native_frame?: EqFrame;
    frame_reason?: string | null;
    does_refraction?: boolean | null;
    tracking_rates?: DriveRate[];
    interface_version?: number | null;
    device_state?: boolean;
    unsupported_members?: string[];
}

/** `GET mount` (§2.12). */
export interface MountDescriptor {
    identity: DeviceIdentityInfo;
    capabilities: MountCapabilities;
    frame: EqFrame;
    site: Site | null;
    can_flip: boolean | null;
    qualification?: Qualification | null;
    health?: { state?: LinkState; failure_class?: string | null; reason?: string | null };
    jog_deadman_ms?: number;
    max_pulse_ms?: number;
}

/** `GET filter-wheel`. */
export interface FilterWheelDescriptor {
    identity: DeviceIdentityInfo;
    slots: Array<Partial<FilterSlotView> & { index: number; name: string }>;
    position: number | null;
    moving: boolean;
    verified: boolean;
    verified_slot: number | null;
    qualification?: Qualification | null;
}

/** `GET guider`: the guiding block plus algorithm parameters (R25). */
export interface GuiderDescriptor {
    identity: DeviceIdentityInfo;
    guiding: GuidingBlock;
    algo_params?: Array<{ axis: string; name: string; value: number }>;
    actuation_disabled_reason?: string | null;
    dec_guide_mode?: string | null;
    qualification?: Qualification | null;
}

/** `GET safety`: the combined state of every monitor. */
export interface SafetyStatus {
    /** `none` when no monitor is configured. */
    state: SafetyGate;
    sources?: Array<{ connection_id: string; [key: string]: unknown }>;
    changed_at?: string | null;
    monitor?: SafetyMonitorBlock | null;
    resume_blocks?: ResumeBlock[];
    policy_running?: boolean;
    on_unsafe?: string;
}

/** Per-device e-stop result (§2.9). */
export interface EstopDeviceResult {
    kind: string;
    connection_id?: string | null;
    ok: boolean;
    error?: { class?: string; message?: string } | null;
    latency_ms?: number | null;
}

/** `POST equipment/estop`. */
export interface EstopReport {
    status: "stopped" | "partial";
    command_id?: string;
    results: EstopDeviceResult[];
    exposure?: EstopDeviceResult | null;
    block?: ResumeBlock;
    elapsed_ms?: number;
}

/** `GET equipment/events`. */
export interface EquipmentEventsPage {
    events: EquipmentEvent[];
    next_seq: number;
}

/** The accepted-command body the mount, focuser, wheel and guider routes return. */
export interface AcceptedCommand {
    status: string;
    command_id?: string;
    [key: string]: unknown;
}

/** Frames `/mount/slew`, `/mount/sync` and `/mount/center` accept. */
export type MountFrame = "mount" | "j2000" | "jnow";

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

/** `/discover` `equipment_type` values (§2.12), in `/drivers` key spelling. */
export type DiscoverableEquipment =
    | "camera"
    | "mount"
    | "focuser"
    | "filter_wheel"
    | "guider"
    | "weather"
    | "safety_monitor"
    | "rotator"
    | "cover_calibrator"
    | "switch"
    | "dome";

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
    /** `alpaca_broadcast` is refused unless the daemon enables it (D13). */
    driver?: "alpaca" | "alpaca_broadcast";
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
    // Additive arrays (§2.12); absent from older daemons.
    weather?: DiscoveredAuxiliaryDevice[];
    safety_monitors?: DiscoveredAuxiliaryDevice[];
    guiders?: DiscoveredAuxiliaryDevice[];
    rotators?: DiscoveredAuxiliaryDevice[];
    switches?: DiscoveredAuxiliaryDevice[];
    cover_calibrators?: DiscoveredAuxiliaryDevice[];
    domes?: DiscoveredAuxiliaryDevice[];
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

const MOUNT_FRAMES: readonly MountFrame[] = ["mount", "j2000", "jnow"];
export const DRIVE_RATES: readonly DriveRate[] = ["sidereal", "lunar", "solar", "king"];

/**
 * An equatorial target with its frame. RA is in hours [0, 24), Dec in
 * degrees [-90, 90]; the frame defaults to J2000, the frame catalogues and
 * plate solves use.
 */
function mountTarget(
    action: string,
    payload: { ra: number; dec: number; frame?: MountFrame },
): { ra: number; dec: number; frame: MountFrame } {
    const ra = requireNumber({ ra: payload.ra }, "ra", action);
    const dec = requireNumber({ dec: payload.dec }, "dec", action);
    if (ra < 0 || ra >= 24) throw new Error(`${action} requires ra in hours, 0 <= ra < 24 (got ${ra}).`);
    if (dec < -90 || dec > 90) throw new Error(`${action} requires dec in degrees, -90 <= dec <= 90 (got ${dec}).`);
    const frame = payload.frame ?? "j2000";
    if (!MOUNT_FRAMES.includes(frame)) {
        throw new Error(`${action} requires frame ${MOUNT_FRAMES.join(", ")} (got ${JSON.stringify(frame)}).`);
    }
    return { ra, dec, frame };
}

/** PHD2 settle criteria; the daemon's `[guider]` values fill any omitted field. */
export interface GuiderSettle {
    pixels?: number;
    time_s?: number;
    timeout_s?: number;
}

function settlePayload(action: string, settle: GuiderSettle | undefined): { settle?: GuiderSettle } {
    if (!settle) return {};
    const out: GuiderSettle = {};
    for (const key of ["pixels", "time_s", "timeout_s"] as const) {
        const value = settle[key];
        if (value === undefined) continue;
        if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
            throw new Error(`${action} requires a positive settle.${key} (got ${JSON.stringify(value)}).`);
        }
        out[key] = value;
    }
    return Object.keys(out).length ? { settle: out } : {};
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
    /**
     * False for commands that are unsafe at relay latency (jog, pulse guide;
     * D12): published for the direct transport, refused by the relay.
     */
    relay_allowed?: boolean;
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
 * A GET command's arguments, as the query string of its direct route.
 *
 * The relay hands the payload to the same handler as a body, so the direct
 * route has to carry it too or `equipment.events?since_seq=` would replay
 * the journal from the start on every poll. Only scalar values travel;
 * arrays are joined with commas (the daemon's `kinds` format).
 */
export function queryString(payload: Record<string, unknown>): string {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(payload)) {
        if (value == null) continue;
        if (Array.isArray(value)) {
            const parts = value.filter((part) => typeof part === "string" || typeof part === "number");
            if (parts.length) params.set(key, parts.join(","));
        } else if (typeof value === "string" || typeof value === "boolean"
            || (typeof value === "number" && Number.isFinite(value))) {
            params.set(key, String(value));
        }
    }
    return params.toString();
}

/** The equipment layer's abort paths (D19): each reports its own outcome. */
const EQUIPMENT_ABORTS = new Set(["mount.abort", "focuser.abort", "filter_wheel.abort", "guider.stop"]);

/**
 * How long the relay should wait for a result before answering "pending".
 *
 * The e-stop and the aborts wait up to 2 s on the daemon for each abort
 * path's own result (§2.9). Their outcome is the one thing the operator must
 * see, so they get room for that budget plus relay latency instead of the
 * relay's three-second default.
 */
function relayWaitSeconds(action: string, payload: Record<string, unknown>): number | undefined {
    if (action === "camera.capture" && typeof payload.exposure === "number" && Number.isFinite(payload.exposure)) {
        // Opening a DSLR, changing its configuration and downloading a still
        // can exceed the relay's default three-second acknowledgement window.
        return Math.min(140, Math.max(15, payload.exposure + 65));
    }
    if (action === "equipment.connect" || action === "equipment.discover") return 90;
    if (action === "camera.set_parameter") return 15;
    if (action === "equipment.estop") return 8;
    if (EQUIPMENT_ABORTS.has(action)) return 6;
    return undefined;
}

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
        // Wait for the result rather than treating a queued acknowledgement as
        // a failed operation while the device continues in the background.
        const waitSeconds = relayWaitSeconds(action, payload);
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
        const query = queryString(payload);
        return { ok: true, payload: await httpJson<Record<string, unknown>>(query ? `${base}?${query}` : base) };
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

    /**
     * GoTo. Coordinates are J2000 unless `frame` says otherwise (§2.12): the
     * daemon converts them to the mount's native frame, which is usually
     * JNow, about 22' away. Sending catalogue coordinates as "mount" would
     * put that error into every slew and every M-4 pointing sample.
     */
    async slewMount(payload: { ra: number; dec: number; frame?: MountFrame }): Promise<AcceptedCommand> {
        return sendFor("mount.slew", mountTarget("mount.slew", payload));
    },

    /** Tell the mount where it points; J2000 by default, as for slews. */
    async syncMount(payload: { ra: number; dec: number; frame?: MountFrame }): Promise<AcceptedCommand> {
        return sendFor("mount.sync", mountTarget("mount.sync", payload));
    },

    /** Identity, capabilities (the O10 gates), native frame, site and qualification. */
    async getMountDescriptor(): Promise<MountDescriptor> {
        return sendFor<MountDescriptor>("mount.descriptor");
    },

    /** Park, verified through AtPark. */
    async parkMount(): Promise<AcceptedCommand> {
        return sendFor("mount.park");
    },

    async unparkMount(): Promise<AcceptedCommand> {
        return sendFor("mount.unpark");
    },

    /** FindHome, verified through AtHome. */
    async findMountHome(): Promise<AcceptedCommand> {
        return sendFor("mount.find_home");
    },

    /** Slew, solve, sync or offset, re-slew until within tolerance (M-n). */
    async centerMount(payload: { ra: number; dec: number; frame?: MountFrame }): Promise<AcceptedCommand> {
        return sendFor("mount.center", mountTarget("mount.center", payload));
    },

    /** `to_mount` writes `[site]` to the mount; `from_mount` adopts the mount's site. */
    async syncMountSite(direction: "to_mount" | "from_mount"): Promise<AcceptedCommand> {
        if (direction !== "to_mount" && direction !== "from_mount") {
            throw new Error(`mount.site_sync requires direction "to_mount" or "from_mount" (got ${JSON.stringify(direction)}).`);
        }
        return sendFor("mount.site_sync", { direction });
    },

    async setMountTrackingRate(rate: DriveRate): Promise<AcceptedCommand> {
        if (!DRIVE_RATES.includes(rate)) {
            throw new Error(`mount.set_tracking_rate requires one of ${DRIVE_RATES.join(", ")} (got ${JSON.stringify(rate)}).`);
        }
        return sendFor("mount.set_tracking_rate", { rate });
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

    async setFocuserTempComp(enabled: boolean): Promise<AcceptedCommand> {
        requireBoolean({ enabled }, "enabled", "focuser.set_tempcomp");
        return sendFor("focuser.set_tempcomp", { enabled });
    },

    /**
     * Declare the origin of an emulated relative focuser. A profile write
     * only: nothing moves, and the daemon audits it.
     */
    async declareFocuserPosition(position: number): Promise<AcceptedCommand> {
        const value = requireNumber({ position }, "position", "focuser.set_position");
        if (!Number.isSafeInteger(value)) throw new Error(`focuser.set_position requires an integer step (got ${value}).`);
        return sendFor("focuser.set_position", { position: value });
    },

    // ── Filter wheel ─────────────────────────────────────────────────────

    async getFilterWheel(): Promise<FilterWheelDescriptor> {
        return sendFor<FilterWheelDescriptor>("filter_wheel.descriptor");
    },

    /** Runs the v6 §8.2.1 handshake. `slot` is 0-based. */
    async changeFilter(target: { slot: number } | { name: string }): Promise<AcceptedCommand> {
        if ("slot" in target) {
            const slot = requireNumber({ slot: target.slot }, "slot", "filter_wheel.change");
            if (!Number.isSafeInteger(slot) || slot < 0) {
                throw new Error(`filter_wheel.change requires a 0-based slot (got ${slot}).`);
            }
            return sendFor("filter_wheel.change", { slot });
        }
        return sendFor("filter_wheel.change", { name: requireString({ name: target.name?.trim() }, "name", "filter_wheel.change") });
    },

    /** Cancel the running handshake; the wheel itself has no halt. */
    async abortFilterChange(): Promise<AcceptedCommand> {
        return sendFor("filter_wheel.abort");
    },

    /** Copy the TOML seeds into the connected wheel's profile (D17), audited. */
    async importFilterSeed(): Promise<AcceptedCommand> {
        return sendFor("filter_wheel.import_seed");
    },

    // ── Guider (PHD2) ────────────────────────────────────────────────────

    async getGuider(): Promise<GuiderDescriptor> {
        return sendFor<GuiderDescriptor>("guider.descriptor");
    },

    async startGuiding(options: { recalibrate?: boolean; settle?: GuiderSettle } = {}): Promise<AcceptedCommand> {
        return sendFor("guider.start", {
            ...(options.recalibrate === undefined ? {} : { recalibrate: options.recalibrate }),
            ...settlePayload("guider.start", options.settle),
        });
    },

    /** The guider's abort (`stop_capture`); never refused while PHD2 answers. */
    async stopGuiding(): Promise<AcceptedCommand> {
        return sendFor("guider.stop");
    },

    async pauseGuiding(full = false): Promise<AcceptedCommand> {
        return sendFor("guider.pause", { full });
    },

    async resumeGuiding(): Promise<AcceptedCommand> {
        return sendFor("guider.resume");
    },

    /** Dither in guide-camera pixels; the daemon's `[guider]` defaults apply to omitted fields. */
    async dither(options: { amount_px?: number; ra_only?: boolean; settle?: GuiderSettle } = {}): Promise<AcceptedCommand> {
        if (options.amount_px !== undefined) {
            const amount = requireNumber({ amount_px: options.amount_px }, "amount_px", "guider.dither");
            if (amount <= 0) throw new Error(`guider.dither requires a positive amount_px (got ${amount}).`);
        }
        return sendFor("guider.dither", {
            ...(options.amount_px === undefined ? {} : { amount_px: options.amount_px }),
            ...(options.ra_only === undefined ? {} : { ra_only: options.ra_only }),
            ...settlePayload("guider.dither", options.settle),
        });
    },

    // ── Environment ──────────────────────────────────────────────────────

    async getWeather(): Promise<WeatherBlock> {
        return sendFor<WeatherBlock>("weather.status");
    },

    async getSafety(): Promise<SafetyStatus> {
        return sendFor<SafetyStatus>("safety.status");
    },

    // ── Equipment-wide: e-stop, resume blocks, journal ───────────────────

    /**
     * Emergency stop: every device's abort path within 2 s, degraded and
     * lost ones included (§2.9, D19). The camera exposure is aborted only
     * when asked. This is not a park.
     */
    async emergencyStop(options: { abort_exposure?: boolean } = {}): Promise<RelayCommandResult & { payload?: EstopReport }> {
        return send("equipment.estop", { abort_exposure: options.abort_exposure === true }) as Promise<RelayCommandResult & { payload?: EstopReport }>;
    },

    async getResumeBlocks(): Promise<ResumeBlock[]> {
        const body = await sendFor<{ blocks?: ResumeBlock[] }>("equipment.resume_blocks");
        return Array.isArray(body.blocks) ? body.blocks : [];
    },

    /** Operator clear, audited. Safety blocks clear only on a confirmed safe reading. */
    async clearResumeBlock(id: string): Promise<AcceptedCommand> {
        return sendFor("equipment.clear_resume_block", { id: requireString({ id }, "id", "equipment.clear_resume_block") });
    },

    /** The journal tail after `since_seq` (at most 500), optionally by event kind. */
    async getEquipmentEvents(query: { since_seq?: number; kinds?: string[]; limit?: number } = {}): Promise<EquipmentEventsPage> {
        const body = await sendFor<Partial<EquipmentEventsPage>>("equipment.events", {
            ...(query.since_seq === undefined ? {} : { since_seq: query.since_seq }),
            ...(query.kinds?.length ? { kinds: query.kinds.join(",") } : {}),
            ...(query.limit === undefined ? {} : { limit: query.limit }),
        });
        return {
            events: Array.isArray(body.events) ? body.events : [],
            next_seq: typeof body.next_seq === "number" ? body.next_seq : query.since_seq ?? 0,
        };
    },

    /**
     * Connect any equipment-layer device. `driver` is required (§2.12): the
     * relay no longer defaults it to the simulator.
     */
    async connectDevice(equipmentType: string, driver: string, option?: string): Promise<Record<string, unknown>> {
        return sendFor("equipment.connect", {
            equipment_type: requireString({ equipmentType }, "equipmentType", "equipment.connect"),
            driver: requireString({ driver }, "driver", "equipment.connect"),
            ...(option ? { option } : {}),
        });
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
