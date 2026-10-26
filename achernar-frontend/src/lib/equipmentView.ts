/**
 * View-model rules for the equipment-layer panels.
 *
 * Pure functions only, so every rule the panels depend on is tested in the
 * node test environment without a DOM. The one rule that runs through all of
 * it: a value the daemon did not measure renders as UNKNOWN, never as a
 * plausible number, and an unknown state is never "good".
 */
import type { MountCapabilities, MountDescriptor } from "../api/achernar";
import type {
    BusySet,
    DeviceHealthEntry,
    EqFrame,
    GuiderAppState,
    GuidingBlock,
    LinkState,
    MountBlock,
    ObservedField,
    Qualification,
    QualificationState,
    ResumeBlock,
    SafetyGate,
    SafetyMonitorBlock,
    WeatherBlock,
    WeatherFieldKey,
} from "../types/achernar";

/** Rendered wherever the daemon reported null: not measured, not zero. */
export const UNKNOWN = "--";

export type Tone = "good" | "warn" | "bad" | "muted";

/** Runs one API call and records its outcome in the console's command log. */
export type RunCommand = (label: string, call: () => Promise<unknown>) => Promise<void>;

// ── Numbers and times ─────────────────────────────────────────────────────

export function num(value: number | null | undefined, digits = 2, unit = ""): string {
    if (value == null || !Number.isFinite(value)) return UNKNOWN;
    return `${value.toFixed(digits)}${unit}`;
}

function sexagesimal(value: number, unitMarks: [string, string, string], signed: boolean, secondDigits: number): string {
    const sign = value < 0 ? "-" : signed ? "+" : "";
    const scale = 10 ** secondDigits;
    // Round once, on the smallest unit, so 59.99 s never prints as 60 s.
    let total = Math.round(Math.abs(value) * 3600 * scale);
    const seconds = (total % (60 * scale)) / scale;
    total = Math.floor(total / (60 * scale));
    const minutes = total % 60;
    const whole = Math.floor(total / 60);
    const pad = (n: number, width: number) => String(n).padStart(width, "0");
    const secondsText = secondDigits > 0 ? seconds.toFixed(secondDigits).padStart(3 + secondDigits, "0") : pad(seconds, 2);
    return `${sign}${pad(whole, 2)}${unitMarks[0]}${pad(minutes, 2)}${unitMarks[1]}${secondsText}${unitMarks[2]}`;
}

/** Right ascension in hours as `05h35m17.3s`; unknown stays unknown. */
export function hms(hours: number | null | undefined): string {
    if (hours == null || !Number.isFinite(hours)) return UNKNOWN;
    // Wrap after rounding to the printed 0.1 s, so 23h59m59.97s is 00h, not 24h.
    const tenths = ((Math.round(hours * 36_000) % 864_000) + 864_000) % 864_000;
    return sexagesimal(tenths / 36_000, ["h", "m", "s"], false, 1);
}

/** Declination in degrees as `+22°00'52"`; unknown stays unknown. */
export function dms(degrees: number | null | undefined): string {
    if (degrees == null || !Number.isFinite(degrees)) return UNKNOWN;
    return sexagesimal(degrees, ["°", "'", "\""], true, 0);
}

/**
 * One sexagesimal or decimal angle: `5.588`, `05:35:17.3`, `5 35 17.3`,
 * `05h35m17.3s`, `-05°23'28"`. Minutes and seconds must be below 60 and
 * only the first part may carry a sign. Null when the text is not one.
 */
function parseSexagesimal(text: string): number | null {
    const trimmed = text.trim();
    if (!trimmed) return null;
    if (/^[+-]?(\d+(\.\d*)?|\.\d+)$/.test(trimmed)) return Number(trimmed);
    const match = /^([+-])?\s*(\d+)\s*[:hd°\s]\s*(\d+(?:\.\d*)?)\s*(?:[:m'′\s]\s*(\d+(?:\.\d*)?)\s*[s"″]?)?\s*[m'′]?$/i.exec(trimmed);
    if (!match) return null;
    const [, sign, whole, minutes, seconds] = match;
    const m = Number(minutes);
    const s = seconds === undefined ? 0 : Number(seconds);
    if (m >= 60 || s >= 60) return null;
    if (seconds !== undefined && !Number.isInteger(m)) return null;
    const value = Number(whole) + m / 60 + s / 3600;
    return sign === "-" ? -value : value;
}

/** Right ascension in hours, 0 <= RA < 24; null for anything else. */
export function parseRa(text: string): number | null {
    if (/^\s*[+-]/.test(text)) return null;
    const value = parseSexagesimal(text);
    return value != null && Number.isFinite(value) && value >= 0 && value < 24 ? value : null;
}

/** Declination in degrees, -90 <= Dec <= 90; null for anything else. */
export function parseDec(text: string): number | null {
    const value = parseSexagesimal(text);
    return value != null && Number.isFinite(value) && value >= -90 && value <= 90 ? value : null;
}

/** A reading's age: `0.4 s`, `12 s`, `3 min`. */
export function ageText(ms: number | null | undefined): string {
    if (ms == null || !Number.isFinite(ms) || ms < 0) return UNKNOWN;
    if (ms < 1000) return `${(ms / 1000).toFixed(1)} s`;
    if (ms < 60_000) return `${Math.round(ms / 1000)} s`;
    if (ms < 3_600_000) return `${Math.round(ms / 60_000)} min`;
    return `${(ms / 3_600_000).toFixed(1)} h`;
}

export function yesNo(value: boolean | null | undefined, yes = "yes", no = "no"): string {
    if (value == null) return UNKNOWN;
    return value ? yes : no;
}

// ── Qualification and link state ──────────────────────────────────────────

/** D21 state from either the identity's `{state}` object or a driver entry's string. */
export function qualificationState(q: Qualification | QualificationState | null | undefined): QualificationState | null {
    if (!q) return null;
    if (typeof q === "string") return q;
    return typeof q.state === "string" ? q.state : null;
}

export function qualificationLabel(q: Qualification | QualificationState | null | undefined): string {
    switch (qualificationState(q)) {
        case "qualified": return "qualified";
        case "implemented_unqualified": return "unqualified";
        case "planned": return "planned";
        default: return "qualification unknown";
    }
}

/** Only a rig acceptance record makes a device `qualified` (D21). */
export function qualificationTone(q: Qualification | QualificationState | null | undefined): Tone {
    switch (qualificationState(q)) {
        case "qualified": return "good";
        case "implemented_unqualified": return "warn";
        default: return "muted";
    }
}

export function linkTone(state: LinkState | string | null | undefined): Tone {
    switch (state) {
        case "ready": return "good";
        case "degraded": case "connecting": return "warn";
        case "lost": return "bad";
        default: return "muted";
    }
}

/**
 * What the operator needs to know about one device's health: its link state,
 * failure class and the daemon's own reason. `last_error` is shown only when
 * it says something `reason` does not.
 */
export function healthDetail(health: DeviceHealthEntry | null | undefined): string | null {
    if (!health) return null;
    const parts: string[] = [];
    if (health.state && health.state !== "ready" && health.state !== "disconnected") parts.push(health.state);
    if (health.failure_class) parts.push(health.failure_class.replace(/_/g, " "));
    const reason = health.reason?.trim();
    const lastError = health.last_error?.trim();
    if (reason) parts.push(reason);
    if (lastError && lastError !== reason) parts.push(`last error: ${lastError}`);
    return parts.length ? parts.join(" · ") : null;
}

/**
 * Whether a device's abort path (mount abort, focuser halt, guider stop,
 * filter abort) is offered (D19).
 *
 * Pass the last status the daemon returned, not only a fresh one: a failed
 * or late status poll (for example a relay whose 16 command slots are all
 * busy, which aborts bypass) must not take the abort away. A device that is
 * connected, lost or degraded keeps it; a send that fails is reported.
 */
export function abortAvailable(block: { connected?: boolean | null } | null | undefined, health: DeviceHealthEntry | null | undefined): boolean {
    return block?.connected === true || health?.state === "lost" || health?.state === "degraded";
}

// ── Mount ─────────────────────────────────────────────────────────────────

export interface ActionGate {
    enabled: boolean;
    reason?: string;
}

/**
 * The frame the operator's GoTo, sync and center coordinates are sent in:
 * `j2000` (converted by the daemon) or `mount` (taken as mount-native and
 * passed unconverted, the legacy semantics).
 */
export type TargetFrame = "j2000" | "mount";

export interface MountActionGates {
    goto: ActionGate;
    sync: ActionGate;
    center: ActionGate;
    park: ActionGate;
    unpark: ActionGate;
    home: ActionGate;
    tracking: ActionGate;
    siteSync: ActionGate;
    /** Whether a target in the chosen frame can reach the mount at all. */
    frame: ActionGate;
    /** Never gated (O10, D19). */
    abort: ActionGate;
}

/**
 * The mount's native frame: the live status first, then the descriptor.
 * Null when neither reported one (not read yet, not a refusal).
 */
export function mountNativeFrame(mount: MountBlock | null | undefined, descriptor: MountDescriptor | null | undefined): EqFrame | null {
    return mount?.frame ?? descriptor?.frame ?? descriptor?.capabilities?.native_frame ?? null;
}

/**
 * A J2000 target needs a known native frame to convert into: the daemon
 * refuses it otherwise (M-a). Mount-native coordinates pass unconverted.
 */
function frameGate(native: EqFrame | null, reason: string | null | undefined, targetFrame: TargetFrame): ActionGate {
    if (targetFrame === "mount" || native !== "unknown") return { enabled: true };
    const why = reason?.trim() ? ` (${reason.trim().replace(/_/g, " ")})` : "";
    return {
        enabled: false,
        reason: `The mount's native frame is unknown${why}, so a J2000 target cannot be converted. `
            + "Set [mount] equatorial_system_override in the daemon config, or send the target as mount-native coordinates.",
    };
}

function capability(caps: MountCapabilities | undefined, key: keyof MountCapabilities, what: string): ActionGate {
    // An unread descriptor is not a refusal: the daemon checks every command
    // against the real capabilities anyway. Only a reported `false` gates.
    if (!caps || caps[key] === undefined) return { enabled: true };
    return caps[key] === true ? { enabled: true } : { enabled: false, reason: `The mount reports ${what} is not supported.` };
}

function both(a: ActionGate, b: ActionGate): ActionGate {
    return a.enabled ? b : a;
}

/**
 * Which mount controls the operator may use (O10, ASCOM scope-slew FAQ).
 *
 * GoTo is gated on CanSlewAsync, never on CanSlew: a compliant Alpaca mount
 * must report CanSlew false. Abort is never gated on any flag, nor on the
 * mount being parked, slewing or even lost (D19). Commands are otherwise
 * left to the daemon's gate, so a capability that was never read does not
 * disable a control.
 *
 * GoTo, sync and center with a J2000 target are disabled, with the
 * daemon's frame reason, while the native frame is unknown (an Alpaca mount
 * below InterfaceVersion 2 or with an unreadable EquatorialSystem, or the
 * interim OLS INDI adapter without an override); the operator may instead
 * send mount-native coordinates.
 */
export function mountActionGates(
    mount: MountBlock | null | undefined,
    descriptor: MountDescriptor | null | undefined,
    targetFrame: TargetFrame = "j2000",
): MountActionGates {
    const caps = descriptor?.capabilities;
    const parked = mount?.parked === true;
    const slewing = mount?.slewing === true;
    const parkedGate: ActionGate = parked ? { enabled: false, reason: "The mount is parked; unpark it first." } : { enabled: true };
    const slewingGate: ActionGate = slewing ? { enabled: false, reason: "The mount is slewing; wait or abort." } : { enabled: true };
    const frame = frameGate(mountNativeFrame(mount, descriptor), caps?.frame_reason, targetFrame);
    const goto = both(parkedGate, both(capability(caps, "can_slew_async", "asynchronous GoTo (CanSlewAsync)"), frame));
    return {
        goto,
        sync: both(parkedGate, both(capability(caps, "can_sync", "sync (CanSync)"), frame)),
        center: goto,
        park: both(parked ? { enabled: false, reason: "The mount is already parked." } : { enabled: true }, capability(caps, "can_park", "park (CanPark)")),
        unpark: both(mount?.parked === false ? { enabled: false, reason: "The mount is not parked." } : { enabled: true }, capability(caps, "can_unpark", "unpark (CanUnpark)")),
        home: both(parkedGate, capability(caps, "can_find_home", "find home (CanFindHome)")),
        tracking: both(parkedGate, capability(caps, "can_set_tracking", "tracking control (CanSetTracking)")),
        siteSync: slewingGate,
        frame,
        abort: { enabled: true },
    };
}

export function frameLabel(frame: string | null | undefined): string {
    switch (frame) {
        case "j2000": return "J2000";
        case "jnow_mean": return "JNow (mean)";
        case "jnow_apparent": return "JNow (apparent)";
        case "unknown": return "unknown frame";
        default: return UNKNOWN;
    }
}

export function alignmentLabel(mode: string | null | undefined): string {
    switch (mode) {
        case "alt_az": return "Alt-Az";
        case "polar": return "Polar (fork)";
        case "german_polar": return "German equatorial";
        default: return UNKNOWN;
    }
}

// ── Guider ────────────────────────────────────────────────────────────────

export interface GuiderActionGates {
    start: ActionGate;
    /** The guider's abort (`stop_capture`, D19). */
    stop: ActionGate;
    pause: ActionGate;
    resume: ActionGate;
    dither: ActionGate;
}

const ACTIVE_GUIDING: GuiderAppState[] = ["guiding", "calibrating", "lost_lock"];

export function guiderActionGates(guiding: GuidingBlock | null | undefined): GuiderActionGates {
    const state = guiding?.state ?? null;
    const known = state != null && state !== "unknown";
    const active = state != null && ACTIVE_GUIDING.includes(state);
    return {
        start: active ? { enabled: false, reason: "PHD2 is already guiding." } : { enabled: true },
        stop: { enabled: true },
        pause: state === "guiding" || state === "calibrating" ? { enabled: true }
            : { enabled: false, reason: known ? "PHD2 is not guiding." : "PHD2's state is unknown." },
        resume: state === "paused" ? { enabled: true } : { enabled: false, reason: "Guiding is not paused." },
        // The daemon also refuses a dither during an exposure (WP1A-7 check).
        dither: state === "guiding" ? { enabled: true } : { enabled: false, reason: "Dither needs PHD2 to be guiding." },
    };
}

export function guiderStateLabel(state: GuiderAppState | null | undefined): string {
    if (!state || state === "unknown") return "state unknown";
    return state === "lost_lock" ? "lost lock" : state;
}

export function guiderTone(state: GuiderAppState | null | undefined): Tone {
    switch (state) {
        case "guiding": return "good";
        case "calibrating": case "paused": case "looping": case "selected": return "warn";
        case "lost_lock": return "bad";
        default: return "muted";
    }
}

/** PHD2 `SettleDone.Status`: 0 is success; anything else is a failed settle. */
export function settleText(status: number | null | undefined): string {
    if (status == null) return "settle pending";
    return status === 0 ? "settled" : `settle failed (${status})`;
}

// ── Weather and safety ────────────────────────────────────────────────────

export const WEATHER_FIELDS: ReadonlyArray<{ key: WeatherFieldKey; label: string; unit: string; digits: number }> = [
    { key: "temperature_c", label: "Temperature", unit: " °C", digits: 1 },
    { key: "humidity_pct", label: "Humidity", unit: " %", digits: 0 },
    { key: "dew_point_c", label: "Dew point", unit: " °C", digits: 1 },
    { key: "pressure_hpa", label: "Pressure", unit: " hPa", digits: 0 },
    { key: "cloud_cover_pct", label: "Cloud cover", unit: " %", digits: 0 },
    { key: "sky_temperature_c", label: "Sky temperature", unit: " °C", digits: 1 },
    { key: "sky_quality_mag_arcsec2", label: "Sky quality", unit: " mag/arcsec²", digits: 2 },
    { key: "sky_brightness_lux", label: "Sky brightness", unit: " lux", digits: 3 },
    { key: "star_fwhm_arcsec", label: "Star FWHM", unit: "\"", digits: 2 },
    { key: "rain_rate_mm_h", label: "Rain rate", unit: " mm/h", digits: 1 },
    { key: "wind_speed_ms", label: "Wind speed", unit: " m/s", digits: 1 },
    { key: "wind_gust_ms", label: "Wind gust", unit: " m/s", digits: 1 },
    { key: "wind_direction_deg", label: "Wind direction", unit: "°", digits: 0 },
];

export interface WeatherRow {
    key: WeatherFieldKey;
    label: string;
    value: string;
    state: string;
    age: string;
    tone: Tone;
}

function fieldTone(field: ObservedField | undefined): Tone {
    if (!field || field.value == null) return "muted";
    return field.state === "fresh" ? "good" : "warn";
}

/**
 * Every ObservingConditions field the device reports, in a fixed order.
 * Fields the device does not implement are listed as such, never filled.
 */
export function weatherRows(weather: WeatherBlock | null | undefined): WeatherRow[] {
    const fields = weather?.fields ?? {};
    return WEATHER_FIELDS.map(({ key, label, unit, digits }) => {
        const field = fields[key];
        const state = field?.state ?? (field ? "unknown" : "not reported");
        return {
            key,
            label,
            value: field && field.value != null ? num(field.value, digits, unit) : UNKNOWN,
            state: state === "not_set" ? "not set" : state,
            age: ageText(field?.age_ms),
            tone: fieldTone(field),
        };
    });
}

export interface SafetyView {
    label: string;
    tone: Tone;
    detail: string;
}

/**
 * The combined safety state. A monitor that is not Ready, or whose reading
 * is stale, is not safe for new automatic actions (§2.8): `unknown` is
 * shown as a warning, never as safe.
 */
export function safetyView(gate: SafetyGate | null | undefined, monitor: SafetyMonitorBlock | null | undefined): SafetyView {
    if (gate === "unsafe" || monitor?.level === "unsafe") {
        return { label: "UNSAFE", tone: "bad", detail: monitor?.detail || "A safety monitor reports unsafe. Automatic actions are refused." };
    }
    if (gate === "unknown" || (monitor && monitor.is_safe == null)) {
        return { label: "UNKNOWN", tone: "warn", detail: "A monitor is configured but its state is unknown or stale; automatic actions are refused." };
    }
    if (gate === "safe" || monitor?.is_safe === true) {
        return { label: "SAFE", tone: "good", detail: "Every monitor reports safe." };
    }
    if (monitor?.level === "warning") {
        return { label: "WARNING", tone: "warn", detail: monitor.detail || "A safety monitor reports a warning level." };
    }
    if (gate === "none") {
        return { label: "NO MONITOR", tone: "muted", detail: "No safety monitor is connected; nothing gates on safety." };
    }
    return { label: UNKNOWN, tone: "muted", detail: "Safety state not reported." };
}

// ── Resume blocks, busy flags, e-stop ─────────────────────────────────────

/** A human title for each H22 block kind. */
export function resumeBlockTitle(block: ResumeBlock): string {
    const kind = block.kind;
    switch (kind.kind) {
        case "filter_handshake_incomplete": return "Filter handshake incomplete";
        case "outcome_unknown": return `Outcome unknown: ${kind.device.replace(/_/g, " ")} ${kind.op.replace(/_/g, " ")}`;
        case "safety_unsafe": return "Safety monitor reports unsafe";
        case "safety_unknown": return "Safety state unknown";
        case "guider_lost": return "Guider lost";
        case "mount_lost": return "Mount lost";
        case "limit_reached": return "Mount limit reached";
        case "estop": return "Emergency stop";
        default: return String((kind as { kind?: unknown }).kind ?? "Resume block").replace(/_/g, " ");
    }
}

/** `safety_*` blocks clear only on a confirmed safe reading; the rest by the operator. */
export function operatorClears(block: ResumeBlock): boolean {
    return block.kind.kind !== "safety_unsafe" && block.kind.kind !== "safety_unknown";
}

const BUSY_LABELS: Record<keyof BusySet, string> = {
    exposing: "exposing",
    autofocus: "autofocus",
    slewing: "slewing",
    settling: "settling",
    parking: "parking",
    flipping: "meridian flip",
    centering: "centering",
    dithering: "dithering",
    guider_settling: "guider settling",
    filter_changing: "filter change",
    cover_moving: "cover moving",
    calibrator_changing: "calibrator changing",
};

export function busyLabels(busy: BusySet | null | undefined): string[] {
    if (!busy) return [];
    return (Object.keys(BUSY_LABELS) as Array<keyof BusySet>).filter((key) => busy[key] === true).map((key) => BUSY_LABELS[key]);
}

export function estopSummary(report: {
    status?: string;
    results?: Array<{ kind: string; ok: boolean; latency_ms?: number | null; error?: { message?: string } | null }>;
    exposure?: { ok: boolean; error?: { message?: string } | null } | null;
    elapsed_ms?: number;
} | null | undefined): string {
    if (!report) return "No e-stop result.";
    const results = report.results ?? [];
    const parts = results.map((r) => `${r.kind.replace(/_/g, " ")} ${r.ok ? "stopped" : `NOT CONFIRMED${r.error?.message ? ` (${r.error.message})` : ""}`}`);
    if (report.exposure) parts.push(`exposure ${report.exposure.ok ? "aborted" : "NOT CONFIRMED"}`);
    const head = report.status === "stopped" ? "E-stop: every device confirmed" : "E-stop: some devices did not confirm";
    const body = parts.length ? parts.join(", ") : "no devices were connected";
    return `${head} — ${body}${report.elapsed_ms != null ? ` in ${report.elapsed_ms} ms` : ""}.`;
}
