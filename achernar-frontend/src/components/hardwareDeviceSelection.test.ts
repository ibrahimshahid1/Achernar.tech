import { describe, expect, it } from "vitest";
import type { DiscoveryReport } from "../api/achernar";
import type { EquipmentState } from "../types/achernar";
import { discoverySummary, endpointError, hardwareChoices, hardwareConnectPayload, hardwareConnectionKey } from "./hardwareDeviceSelection";

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
        expect(() => hardwareConnectPayload("filter_wheel", { key: "wheel", name: "Wheel", driver: "alpaca", driverLabel: "Alpaca", index: 0 })).toThrow(/unavailable/);
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
    });
});
