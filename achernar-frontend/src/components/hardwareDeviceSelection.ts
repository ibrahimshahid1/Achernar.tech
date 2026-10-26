import type { DiscoverableEquipment, DiscoveredAuxiliaryDevice, DiscoveredCamera, DiscoveryReport, DriverCapability, DriverCapabilityReport } from "../api/achernar";
import type { DeviceBlockCommon, DeviceHealthEntry, EquipmentState, LinkState } from "../types/achernar";

/** A device kind the console lists, with where its status and health live. */
export interface DeviceKindSpec {
    kind: DiscoverableEquipment;
    label: string;
    /** The `/status` block key. The guider is `guiding` in telemetry too. */
    statusKey: "camera" | "mount" | "focuser" | "filter_wheel" | "guiding" | "weather" | "safety_monitor";
    /** The `device_health` key (the daemon's DeviceKind serde name). */
    healthKey: string;
    /** Discovery: a local bus scan, an explicit Alpaca server, or none (PHD2 is addressed). */
    discovery: "bus" | "alpaca" | "none";
}

/** The Tier 1 device list (§2.12), in NINA's connect order after the camera. */
export const DEVICE_KINDS: readonly DeviceKindSpec[] = [
    { kind: "camera", label: "Camera", statusKey: "camera", healthKey: "camera", discovery: "bus" },
    { kind: "filter_wheel", label: "Filter wheel", statusKey: "filter_wheel", healthKey: "filter_wheel", discovery: "alpaca" },
    { kind: "focuser", label: "Focuser", statusKey: "focuser", healthKey: "focuser", discovery: "alpaca" },
    { kind: "mount", label: "Mount", statusKey: "mount", healthKey: "mount", discovery: "alpaca" },
    { kind: "guider", label: "Guider", statusKey: "guiding", healthKey: "guider", discovery: "none" },
    { kind: "weather", label: "Weather", statusKey: "weather", healthKey: "weather", discovery: "alpaca" },
    { kind: "safety_monitor", label: "Safety monitor", statusKey: "safety_monitor", healthKey: "safety_monitor", discovery: "alpaca" },
];

/** Tier 2 kinds the daemon lists in `/drivers` but cannot connect yet. */
export const PLANNED_KINDS: ReadonlyArray<{ kind: keyof DriverCapabilityReport; label: string }> = [
    { kind: "rotator", label: "Rotator" },
    { kind: "cover_calibrator", label: "Cover calibrator" },
    { kind: "switch", label: "Switch" },
    { kind: "dome", label: "Dome" },
];

export function kindSpec(kind: DiscoverableEquipment): DeviceKindSpec | undefined {
    return DEVICE_KINDS.find((spec) => spec.kind === kind);
}

/** The status block for a kind, whatever key the daemon uses for it. */
export function deviceBlock(state: EquipmentState | undefined, spec: DeviceKindSpec): DeviceBlockCommon | null | undefined {
    if (!state) return undefined;
    if (spec.statusKey === "guiding") return state.guiding ?? state.guider;
    return state[spec.statusKey] as DeviceBlockCommon | null | undefined;
}

const LINK_STATES: readonly LinkState[] = ["connecting", "ready", "degraded", "lost", "disconnected"];

/**
 * The worker's link state (§2.8): the block's `link_state`, else the
 * registry's `device_health[kind].state`. A block's own `state` is not used,
 * because the guiding block's `state` is PHD2's AppState.
 */
export function deviceLinkState(device: DeviceBlockCommon | null | undefined, health: DeviceHealthEntry | null | undefined): LinkState | null {
    const raw = device?.link_state ?? health?.state ?? null;
    return raw && LINK_STATES.includes(raw) ? raw : null;
}

/** The selector heading. A lost device is not "not connected": its abort still runs (D19). */
export function connectionHeading(online: boolean, connected: boolean, state: LinkState | null): string {
    if (!online) return "Unknown";
    if (state === "lost") return "Lost";
    if (connected) return state === "degraded" ? "Degraded" : "Connected";
    return state === "connecting" ? "Connecting" : "Not connected";
}

/**
 * Polling measurements are not new connections; actual identities are.
 *
 * Each device contributes whether it is connected, its stable identity key
 * (§2.6; the display name only from a daemon that does not publish one) and
 * its connection id, so a reconnect of the same identity is still a new
 * connection.
 */
export function hardwareConnectionKey(state: EquipmentState | undefined, online: boolean): string {
    const identity = (device: DeviceBlockCommon | null | undefined) =>
        [device?.connected ?? null, device?.identity_key ?? device?.name ?? null, device?.connection_id ?? null];
    return JSON.stringify([online, state?.daemon_id ?? null, state?.hardware_id ?? null,
        ...DEVICE_KINDS.map((spec) => identity(deviceBlock(state, spec)))]);
}

export interface HardwareChoice {
    key: string;
    name: string;
    driver: string;
    driverLabel: string;
    index: number;
    option?: string;
}

function reportedDevices(type: DiscoverableEquipment, report: DiscoveryReport): Array<DiscoveredCamera | DiscoveredAuxiliaryDevice> | undefined {
    switch (type) {
        case "camera": return report.cameras;
        case "mount": return report.mounts;
        case "focuser": return report.focusers;
        case "filter_wheel": return report.filter_wheels;
        case "weather": return report.weather;
        case "safety_monitor": return report.safety_monitors;
        case "guider": return report.guiders;
        case "rotator": return report.rotators;
        case "switch": return report.switches;
        case "cover_calibrator": return report.cover_calibrators;
        case "dome": return report.domes;
    }
}

/** Only reported device instances become choices; driver capabilities do not. */
export function hardwareChoices(type: DiscoverableEquipment, report: DiscoveryReport | null): HardwareChoice[] {
    if (!report) return [];
    const devices = reportedDevices(type, report);
    const seen = new Set<string>();
    return (devices ?? []).flatMap((device) => {
        if (!device.name || !device.driver_id || !Number.isSafeInteger(device.index) || device.index < 0) return [];
        const key = device.selection_key || `${device.driver_id}:${device.index}`;
        if (seen.has(key)) return [];
        const option = "option" in device && typeof device.option === "string" ? device.option : undefined;
        if (type !== "camera" && (!option || !("availability" in device) || device.availability !== "reported")) return [];
        seen.add(key);
        return [{ key, name: device.name, driver: device.driver_id, driverLabel: device.driver_label || device.driver_id,
            index: device.index, ...(option ? { option } : {}) }];
    });
}

export function hardwareConnectPayload(type: DiscoverableEquipment, choice: HardwareChoice): Record<string, unknown> {
    // The discovered option already names the server and `#device=N`; the
    // index is the camera's position on its bus and harmless for the rest.
    return { equipment_type: type, driver: choice.driver, index: choice.index,
        ...(choice.option ? { option: choice.option } : {}) };
}

export function discoverySummary(type: DiscoverableEquipment, report: DiscoveryReport | null): string {
    if (!report) {
        return kindSpec(type)?.discovery === "none"
            ? "Not discovered: connect it by address below."
            : "Open the selector or refresh to discover devices.";
    }
    const choices = hardwareChoices(type, report);
    if (choices.length) return `${choices.length} ${choices.length === 1 ? "device reported" : "devices reported"}. Choose one, then Connect.`;
    const probes = type === "camera" ? report.probes : report.device_probes?.filter((probe) => probe.equipment_type === type);
    const significant = probes?.find((probe) => ["failed", "timeout", "manual_required", "unsupported", "skipped"].includes(probe.status));
    if (significant) return significant.detail || `${significant.status.replace(/_/g, " ")}: discovery did not report a selectable device.`;
    if (type !== "camera" && !probes) return "Device discovery is unavailable in this daemon report. Use the manual connection below.";
    return "No devices reported by this scan.";
}

export function endpointError(endpoint: string): string | null {
    if (!endpoint.trim()) return null;
    try {
        const url = new URL(endpoint.trim());
        if (!["http:", "https:"].includes(url.protocol) || !url.hostname || url.username || url.password) {
            return "Use an HTTP or HTTPS Alpaca server URL without embedded credentials.";
        }
        if (url.hash) return "Give the server URL only; the device number has its own field.";
        return null;
    } catch {
        return "Enter a complete Alpaca server URL, such as http://host:port.";
    }
}

// ── Manual connection: transport per driver capability entry (§2.12) ──────

/**
 * How a driver is addressed, from its `/drivers` entry:
 * - `alpaca`: `http(s)://host:port#device=N`;
 * - `indi`: `host:port`, plus `#device=<INDI device name>` when one is given;
 * - `phd2`: `host:port` (instance n listens on 4400 + n - 1);
 * - `sim`: no address, or the free option its hint describes;
 * - `option`: any other driver's free-form option, as its hint describes.
 */
export type Transport = "alpaca" | "indi" | "phd2" | "sim" | "option";

export function transportFor(driver: Pick<DriverCapability, "id">): Transport {
    switch (driver.id) {
        case "alpaca": return "alpaca";
        case "indi": return "indi";
        case "phd2": return "phd2";
        case "sim": case "simulator": return "sim";
        default: return "option";
    }
}

export interface TransportFields {
    url?: string;
    deviceNumber?: string;
    host?: string;
    port?: string;
    device?: string;
    option?: string;
}

export const INDI_DEFAULT_PORT = 7624;
export const PHD2_BASE_PORT = 4400;

function hostError(host: string): string | null {
    if (/\s/.test(host) || host.includes("/") || host.includes("#") || host.includes("@")) {
        return "Enter a host name or address only, such as localhost or 192.168.1.20.";
    }
    if (host.includes(":") && !/^\[[0-9a-fA-F:.]+\]$/.test(host)) {
        return "Put an IPv6 address in brackets, such as [::1].";
    }
    return null;
}

function portValue(text: string | undefined, fallback: number): number | string {
    const trimmed = (text ?? "").trim();
    if (!trimmed) return fallback;
    const port = Number(trimmed);
    if (!Number.isSafeInteger(port) || port < 1 || port > 65535) return "Enter a port from 1 to 65535.";
    return port;
}

/**
 * The `/connect` option for a transport, or the reason it cannot be built.
 *
 * Nothing here invents a network target: an Alpaca driver needs an explicit
 * server URL (an empty or "auto" option would ask the daemon to broadcast,
 * which it refuses by default, D13), and the INDI and PHD2 defaults are the
 * protocols' own localhost ports.
 */
export function connectOption(transport: Transport, fields: TransportFields): { option?: string; error?: string } {
    switch (transport) {
        case "alpaca": {
            const url = (fields.url ?? "").trim();
            if (!url) return { error: "Enter the Alpaca server URL, such as http://host:port." };
            const problem = endpointError(url);
            if (problem) return { error: problem };
            const numberText = (fields.deviceNumber ?? "").trim() || "0";
            const number = Number(numberText);
            if (!/^\d+$/.test(numberText) || !Number.isSafeInteger(number)) return { error: "The Alpaca device number is a whole number, 0 or more." };
            return { option: `${url.replace(/\/+$/, "")}#device=${number}` };
        }
        case "indi": {
            const host = (fields.host ?? "").trim() || "localhost";
            const hostProblem = hostError(host);
            if (hostProblem) return { error: hostProblem };
            const port = portValue(fields.port, INDI_DEFAULT_PORT);
            if (typeof port === "string") return { error: port };
            const device = (fields.device ?? "").trim();
            if (device.includes("#")) return { error: "An INDI device name cannot contain '#'." };
            return { option: `${host}:${port}${device ? `#device=${device}` : ""}` };
        }
        case "phd2": {
            const host = (fields.host ?? "").trim() || "localhost";
            const hostProblem = hostError(host);
            if (hostProblem) return { error: hostProblem };
            const port = portValue(fields.port, PHD2_BASE_PORT);
            if (typeof port === "string") return { error: port };
            return { option: `${host}:${port}` };
        }
        case "sim":
        case "option": {
            const option = (fields.option ?? "").trim();
            return option ? { option } : {};
        }
    }
}

/** The PHD2 port for a 1-based instance number (4400 + n - 1). */
export function phd2InstancePort(instance: number): number | null {
    if (!Number.isSafeInteger(instance) || instance < 1 || instance > 61136) return null;
    return PHD2_BASE_PORT + instance - 1;
}

export function manualConnectPayload(kind: DiscoverableEquipment, driver: DriverCapability, fields: TransportFields): { payload?: Record<string, unknown>; error?: string } {
    if (!driver.enabled) return { error: driver.reason || `${driver.label} cannot connect in this daemon build.` };
    const { option, error } = connectOption(transportFor(driver), fields);
    if (error) return { error };
    return { payload: { equipment_type: kind, driver: driver.id, ...(option ? { option } : {}) } };
}

/** The `/drivers` entries for a kind; an older daemon omits the new kinds. */
export function driversFor(report: DriverCapabilityReport | null | undefined, kind: DiscoverableEquipment): DriverCapability[] {
    if (!report) return [];
    const entries = (report as unknown as Record<string, DriverCapability[] | undefined>)[kind];
    return Array.isArray(entries) ? entries : [];
}

/** Planned kinds whose every driver is unavailable, with the daemon's reason. */
export function plannedKinds(report: DriverCapabilityReport | null | undefined): Array<{ label: string; reason: string }> {
    if (!report) return [];
    return PLANNED_KINDS.flatMap(({ kind, label }) => {
        const entries = driversFor(report, kind as DiscoverableEquipment);
        if (!entries.length || entries.some((entry) => entry.enabled)) return [];
        return [{ label, reason: entries[0].reason ?? "Not available in this daemon build." }];
    });
}
