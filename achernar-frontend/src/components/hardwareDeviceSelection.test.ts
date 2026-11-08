import { describe, expect, it } from "vitest";
import type { DiscoveryReport, DriverCapability, DriverCapabilityReport } from "../api/achernar";
import type { EquipmentState } from "../types/achernar";
import {
    connectionHeading,
    connectOption,
    DEVICE_KINDS,
    deviceLinkState,
    discoverySummary,
    driversFor,
    endpointError,
    hardwareChoices,
    hardwareConnectPayload,
    hardwareConnectionKey,
    manualConnectPayload,
    phd2InstancePort,
    plannedKinds,
    transportFor,
} from "./hardwareDeviceSelection";

const emptyReport = (): DiscoveryReport => ({ cameras: [], mounts: [], focusers: [], filter_wheels: [],
    probes: [], device_probes: [], notes: [], scanned_at: "2026-10-12T16:00:00Z" });

describe("hardware selector device identity", () => {
    it("retires discovery on a new connection identity but not on ordinary telemetry polls", () => {
        const initial: EquipmentState = { connected: true, daemon_id: "rig", camera: {
            name: "Canon", connected: true, connection_id: "session-one", temperature: null, cooling: false,
        } };
        const key = hardwareConnectionKey(initial, true);
        expect(hardwareConnectionKey({ ...initial, camera: { ...initial.camera!, exposure_s: 4, streaming: true } }, true)).toBe(key);
        expect(hardwareConnectionKey({ ...initial, camera: { ...initial.camera!, connection_id: "session-two" } }, true)).not.toBe(key);
        expect(hardwareConnectionKey({ ...initial, camera: { ...initial.camera!, connected: false } }, true)).not.toBe(key);
        expect(hardwareConnectionKey({ ...initial, mount: { name: "New mount", connected: true, ra: null, dec: null, tracking: null } }, true)).not.toBe(key);
        expect(hardwareConnectionKey(initial, false)).not.toBe(key);
    });

    it("preserves a selected second camera and distinguishes equal indexes on different drivers", () => {
        const report = emptyReport();
        report.cameras = [
            { driver_id: "gphoto2", driver_label: "gphoto2", index: 0, name: "Canon", selection_key: "gphoto2:0" },
            { driver_id: "svb", driver_label: "SVB", index: 0, name: "Guide camera", selection_key: "svb:0" },
            { driver_id: "gphoto2", driver_label: "gphoto2", index: 1, name: "Second Canon", selection_key: "gphoto2:1" },
        ];
        const choices = hardwareChoices("camera", report);
        expect(choices.map((choice) => choice.key)).toEqual(["gphoto2:0", "svb:0", "gphoto2:1"]);
        expect(hardwareConnectPayload("camera", choices[2])).toEqual({ equipment_type: "camera", driver: "gphoto2", index: 1 });
    });

    it("preserves the reported Alpaca endpoint and device number instead of falling back to device zero", () => {
        const report = emptyReport();
        report.mounts = [{ driver_id: "alpaca", driver_label: "ASCOM Alpaca", index: 7, name: "Observatory mount",
            selection_key: "alpaca:mount:server:7", hardware_id: "mount-seven", transport: "alpaca", availability: "reported",
            option: "http://rig.local:11111#device=7" }];
        const [choice] = hardwareChoices("mount", report);
        expect(hardwareConnectPayload("mount", choice)).toEqual({ equipment_type: "mount", driver: "alpaca", index: 7,
            option: "http://rig.local:11111#device=7" });
        expect(hardwareChoices("focuser", report)).toEqual([]);
    });

    it("does not turn installed driver probes into attached-device choices", () => {
        const report = emptyReport();
        report.probes = [{ driver_id: "asi", driver_label: "ASI", status: "empty", device_count: 0 }];
        expect(hardwareChoices("camera", report)).toEqual([]);
        expect(discoverySummary("camera", report)).toBe("No devices reported by this scan.");
        // The filter wheel is connectable through the equipment layer (WP1B-5).
        expect(hardwareConnectPayload("filter_wheel", { key: "wheel", name: "Wheel", driver: "alpaca", driverLabel: "Alpaca", index: 0,
            option: "http://rig.local:11111#device=0" })).toEqual({ equipment_type: "filter_wheel", driver: "alpaca", index: 0,
            option: "http://rig.local:11111#device=0" });
    });

    it("rejects incomplete auxiliary targets, invalid indexes, and duplicate scan identities", () => {
        const report = emptyReport();
        report.cameras = [
            { driver_id: "svb", driver_label: "SVB", index: 0, name: "Camera" },
            { driver_id: "svb", driver_label: "SVB", index: 0, name: "Duplicate" },
            { driver_id: "svb", driver_label: "SVB", index: -1, name: "Invalid" },
        ];
        report.mounts = [{ driver_id: "alpaca", driver_label: "Alpaca", index: 2, name: "No target option",
            selection_key: "mount:2", hardware_id: null, transport: "alpaca", availability: "reported", option: "" }];
        expect(hardwareChoices("camera", report)).toHaveLength(1);
        expect(hardwareChoices("mount", report)).toEqual([]);
    });
});

describe("hardware selector truthful discovery states", () => {
    it("keeps a failed probe distinct from an empty bus", () => {
        const report = emptyReport();
        report.probes = [{ driver_id: "gphoto2", driver_label: "gphoto2", status: "timeout", device_count: 0, detail: "Camera probe timed out." }];
        expect(discoverySummary("camera", report)).toBe("Camera probe timed out.");
        expect(discoverySummary("camera", null)).toMatch(/discover/);
    });

    it("explains required manual endpoints and unsupported devices without inventing available hardware", () => {
        const report = emptyReport();
        report.device_probes = [
            { equipment_type: "mount", status: "manual_required", detail: "Provide the Alpaca server URL." },
            { equipment_type: "filter_wheel", status: "unsupported", detail: "Filter wheel discovery is unavailable." },
        ];
        expect(discoverySummary("mount", report)).toBe("Provide the Alpaca server URL.");
        expect(discoverySummary("filter_wheel", report)).toBe("Filter wheel discovery is unavailable.");
        expect(hardwareChoices("mount", report)).toEqual([]);
    });

    it("validates explicit HTTP endpoints without accepting embedded credentials or file URLs", () => {
        expect(endpointError("http://rig.local:11111")).toBeNull();
        expect(endpointError("https://observatory.example/alpaca")).toBeNull();
        expect(endpointError("")).toBeNull();
        expect(endpointError("rig.local:11111")).not.toBeNull();
        expect(endpointError("file:///etc/hosts")).not.toBeNull();
        expect(endpointError("http://user:password@rig.local")).not.toBeNull();
        // The device number has its own field; a fragment here would be ambiguous.
        expect(endpointError("http://rig.local:11111#device=2")).not.toBeNull();
    });
});

describe("device identity keys (§2.6)", () => {
    const mount = (overrides: Record<string, unknown>) => ({ name: "Mount", connected: true, ra: null, dec: null, tracking: null,
        connection_id: "c1", identity_key: "alpaca:esp32-1", ...overrides });

    it("tells two mounts with the same display name apart by identity key", () => {
        const a = hardwareConnectionKey({ connected: true, mount: mount({}) }, true);
        const b = hardwareConnectionKey({ connected: true, mount: mount({ identity_key: "alpaca:rig.local:11111/telescope/0" }) }, true);
        expect(a).not.toBe(b);
    });

    it("ignores readings, but not a reconnect of the same identity", () => {
        const key = hardwareConnectionKey({ connected: true, mount: mount({}) }, true);
        expect(hardwareConnectionKey({ connected: true, mount: mount({ ra: 5.5, slewing: true, age_ms: 12 }) }, true)).toBe(key);
        expect(hardwareConnectionKey({ connected: true, mount: mount({ connection_id: "c2" }) }, true)).not.toBe(key);
    });

    it("covers every Tier 1 kind, reading the guider from its `guiding` block", () => {
        const base: EquipmentState = { connected: true };
        const key = hardwareConnectionKey(base, true);
        for (const [field, block] of [
            ["filter_wheel", { connected: true, name: "Wheel", identity_key: "sim:filter_wheel" }],
            ["guiding", { connected: true, name: "PHD2", identity_key: "phd2:localhost:4400" }],
            ["weather", { connected: true, name: "Weather", identity_key: "sim:weather" }],
            ["safety_monitor", { connected: true, name: "Safety", identity_key: "sim:safety_monitor" }],
        ] as const) {
            expect(hardwareConnectionKey({ ...base, [field]: block }, true), field).not.toBe(key);
        }
        expect(DEVICE_KINDS.map((spec) => spec.kind)).toEqual(
            ["camera", "filter_wheel", "focuser", "mount", "guider", "weather", "safety_monitor"]);
    });
});

describe("link state and health (§2.8, D19)", () => {
    it("reads the worker link state, never PHD2's AppState", () => {
        expect(deviceLinkState({ link_state: "degraded" }, { connected: true, state: "ready" })).toBe("degraded");
        expect(deviceLinkState(null, { connected: false, state: "lost" })).toBe("lost");
        expect(deviceLinkState({ state: "guiding" } as never, null)).toBeNull();
    });

    it("shows a lost device as lost rather than merely not connected", () => {
        expect(connectionHeading(true, false, "lost")).toBe("Lost");
        expect(connectionHeading(true, true, "degraded")).toBe("Degraded");
        expect(connectionHeading(true, false, null)).toBe("Not connected");
        expect(connectionHeading(false, true, "ready")).toBe("Unknown");
    });
});

describe("new device kinds in discovery", () => {
    it("offers reported weather stations and safety monitors as choices", () => {
        const report = emptyReport();
        const device = (index: number, name: string) => ({ driver_id: "alpaca", driver_label: "Alpaca", index, name,
            selection_key: `alpaca:${name}:${index}`, hardware_id: null, transport: "alpaca", availability: "reported" as const,
            option: `http://rig.local:11111#device=${index}` });
        report.weather = [device(0, "Boltwood")];
        report.safety_monitors = [device(1, "Roof sensor")];
        expect(hardwareChoices("weather", report).map((choice) => choice.option)).toEqual(["http://rig.local:11111#device=0"]);
        expect(hardwareChoices("safety_monitor", report)[0].name).toBe("Roof sensor");
        expect(hardwareChoices("filter_wheel", report)).toEqual([]);
    });

    it("says the guider is addressed, not discovered", () => {
        expect(discoverySummary("guider", null)).toMatch(/by address/);
    });
});

describe("manual connection transports (§2.12)", () => {
    const driver = (id: string, overrides: Partial<DriverCapability> = {}): DriverCapability =>
        ({ id, label: id, compiled: true, enabled: true, qualification: "implemented_unqualified", ...overrides });

    it("picks the transport from the driver capability entry", () => {
        expect(["alpaca", "indi", "phd2", "sim", "simulator", "gphoto2"].map((id) => transportFor({ id })))
            .toEqual(["alpaca", "indi", "phd2", "sim", "sim", "option"]);
    });

    it("builds an Alpaca option with the device number and never an implicit broadcast", () => {
        expect(connectOption("alpaca", { url: "http://rig.local:11111/", deviceNumber: "3" })).toEqual({ option: "http://rig.local:11111#device=3" });
        expect(connectOption("alpaca", { url: "http://rig.local:11111" })).toEqual({ option: "http://rig.local:11111#device=0" });
        expect(connectOption("alpaca", { url: "" }).error).toMatch(/server URL/);
        expect(connectOption("alpaca", { url: "auto" }).error).toBeTruthy();
        expect(connectOption("alpaca", { url: "http://rig.local:11111", deviceNumber: "1.5" }).error).toMatch(/whole number/);
        expect(connectOption("alpaca", { url: "http://rig.local:11111", deviceNumber: "-1" }).error).toMatch(/whole number/);
    });

    it("builds INDI host:port with an optional device, defaulting to localhost:7624", () => {
        expect(connectOption("indi", {})).toEqual({ option: "localhost:7624" });
        expect(connectOption("indi", { host: "pi.local", port: "7625", device: "EQMod Mount" })).toEqual({ option: "pi.local:7625#device=EQMod Mount" });
        expect(connectOption("indi", { host: "[::1]" })).toEqual({ option: "[::1]:7624" });
        expect(connectOption("indi", { host: "::1" }).error).toMatch(/brackets/);
        expect(connectOption("indi", { host: "http://pi" }).error).toMatch(/host name/);
        expect(connectOption("indi", { port: "70000" }).error).toMatch(/port/);
    });

    it("builds PHD2 host:port, defaulting to instance 1 on localhost", () => {
        expect(connectOption("phd2", {})).toEqual({ option: "localhost:4400" });
        expect(connectOption("phd2", { host: "guide-pc", port: "4401" })).toEqual({ option: "guide-pc:4401" });
        expect(connectOption("phd2", { port: "0" }).error).toMatch(/port/);
        expect(phd2InstancePort(2)).toBe(4401);
        expect(phd2InstancePort(0)).toBeNull();
    });

    it("sends no option for a simulator unless one is given", () => {
        expect(connectOption("sim", {})).toEqual({});
        expect(connectOption("sim", { option: " 25000 " })).toEqual({ option: "25000" });
    });

    it("builds the connect payload and refuses a disabled driver with the daemon's reason", () => {
        expect(manualConnectPayload("filter_wheel", driver("sim"), {})).toEqual({ payload: { equipment_type: "filter_wheel", driver: "sim" } });
        expect(manualConnectPayload("guider", driver("phd2"), { port: "4402" })).toEqual({
            payload: { equipment_type: "guider", driver: "phd2", option: "localhost:4402" } });
        expect(manualConnectPayload("focuser", driver("moonlite", { enabled: false, reason: "Reached through indi_moonlite (O4)." }), {}))
            .toEqual({ error: "Reached through indi_moonlite (O4)." });
        expect(manualConnectPayload("mount", driver("alpaca"), {}).error).toMatch(/server URL/);
    });
});

describe("driver report kinds", () => {
    const planned = (id: string): DriverCapability => ({ id, label: id, compiled: false, enabled: false, reason: "Planned for Tier 2.", qualification: "planned" });
    const report = {
        bridge_version: "test", camera: [], mount: [], focuser: [],
        filter_wheel: [{ id: "sim", label: "Simulator", compiled: true, enabled: true }],
        rotator: [planned("alpaca"), planned("sim")], cover_calibrator: [planned("alpaca")], switch: [planned("alpaca")],
        dome: [{ id: "alpaca", label: "Alpaca", compiled: true, enabled: true }],
    } as DriverCapabilityReport;

    it("reads each kind's drivers and tolerates an older daemon without them", () => {
        expect(driversFor(report, "filter_wheel").map((entry) => entry.id)).toEqual(["sim"]);
        expect(driversFor(report, "guider")).toEqual([]);
        expect(driversFor(null, "mount")).toEqual([]);
    });

    it("lists only the kinds with no usable driver as planned", () => {
        expect(plannedKinds(report).map((entry) => entry.label)).toEqual(["Rotator", "Cover calibrator", "Switch"]);
        expect(plannedKinds(report)[0].reason).toBe("Planned for Tier 2.");
    });
});
