import type { DiscoverableEquipment, DiscoveryReport } from "../api/achernar";
import type { EquipmentState } from "../types/achernar";

/** Polling measurements are not new connections; actual identities are. */
export function hardwareConnectionKey(state: EquipmentState | undefined, online: boolean): string {
    const identity = (device: { connected?: boolean; name?: string } | null | undefined) =>
        [device?.connected ?? null, device?.name ?? null];
    return JSON.stringify([online, state?.daemon_id ?? null, state?.hardware_id ?? null,
        identity(state?.camera), state?.camera?.connection_id ?? null,
        identity(state?.mount), identity(state?.focuser), identity(state?.filter_wheel)]);
}

export interface HardwareChoice {
    key: string;
    name: string;
    driver: string;
    driverLabel: string;
    index: number;
    option?: string;
}

/** Only reported device instances become choices; driver capabilities do not. */
export function hardwareChoices(type: DiscoverableEquipment, report: DiscoveryReport | null): HardwareChoice[] {
    if (!report) return [];
    const devices = type === "camera" ? report.cameras
        : type === "mount" ? report.mounts
            : type === "focuser" ? report.focusers : report.filter_wheels;
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
    if (type === "filter_wheel") throw new Error("Filter wheel connection is unavailable in this daemon build.");
    return { equipment_type: type, driver: choice.driver, index: choice.index,
        ...(choice.option ? { option: choice.option } : {}) };
}

export function discoverySummary(type: DiscoverableEquipment, report: DiscoveryReport | null): string {
    if (!report) return "Open the selector or refresh to discover devices.";
    const choices = hardwareChoices(type, report);
    if (choices.length) return `${choices.length} ${choices.length === 1 ? "device reported" : "devices reported"}. Choose one, then Connect.`;
    const probes = type === "camera" ? report.probes : report.device_probes?.filter((probe) => probe.equipment_type === type);
    const significant = probes?.find((probe) => ["failed", "timeout", "manual_required", "unsupported", "skipped"].includes(probe.status));
    if (significant) return significant.detail || `${significant.status.replace(/_/g, " ")}: discovery did not report a selectable device.`;
    if (type !== "camera" && !probes) return "Device discovery is unavailable in this daemon report. Use the advanced connection controls.";
    if (type === "filter_wheel") return "Filter wheel discovery and connection controls are unavailable in this daemon build.";
    return "No devices reported by this scan.";
}

export function endpointError(endpoint: string): string | null {
    if (!endpoint.trim()) return null;
    try {
        const url = new URL(endpoint.trim());
        if (!["http:", "https:"].includes(url.protocol) || !url.hostname || url.username || url.password) {
            return "Use an HTTP or HTTPS Alpaca server URL without embedded credentials.";
        }
        return null;
    } catch {
        return "Enter a complete Alpaca server URL, such as http://host:port.";
    }
}
