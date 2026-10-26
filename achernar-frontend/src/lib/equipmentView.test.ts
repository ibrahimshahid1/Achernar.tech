import { describe, expect, it } from "vitest";
import type { MountDescriptor } from "../api/achernar";
import type { MountBlock, ResumeBlock } from "../types/achernar";
import {
    abortAvailable,
    ageText,
    busyLabels,
    dms,
    estopSummary,
    guiderActionGates,
    healthDetail,
    hms,
    mountActionGates,
    mountNativeFrame,
    num,
    operatorClears,
    parseDec,
    parseRa,
    qualificationLabel,
    qualificationTone,
    resumeBlockTitle,
    safetyView,
    UNKNOWN,
    weatherRows,
} from "./equipmentView";

const mount = (overrides: Partial<MountBlock> = {}): MountBlock => ({ connected: true, tracking: true, ra: 5, dec: 10, parked: false, slewing: false, ...overrides });
const descriptor = (capabilities: MountDescriptor["capabilities"]): MountDescriptor =>
    ({ identity: { kind: "mount", driver: "alpaca", identity_key: "alpaca:x", connection_id: "c", name: "Mount" }, capabilities, frame: "jnow_apparent", site: null, can_flip: false });

describe("unknown is never a number", () => {
    it("formats readings and keeps null as unknown, while 0 stays a reading", () => {
        expect(num(null)).toBe(UNKNOWN);
        expect(num(Number.NaN)).toBe(UNKNOWN);
        expect(num(0, 1, " °C")).toBe("0.0 °C");
        expect(hms(null)).toBe(UNKNOWN);
        expect(dms(undefined)).toBe(UNKNOWN);
        expect(ageText(null)).toBe(UNKNOWN);
    });

    it("prints sexagesimal coordinates and carries rounding into the next unit", () => {
        expect(hms(5.588)).toBe("05h35m16.8s");
        expect(hms(0)).toBe("00h00m00.0s");
        expect(hms(23.99999999)).toBe("00h00m00.0s");
        expect(hms(24.5)).toBe("00h30m00.0s");
        expect(hms(-1)).toBe("23h00m00.0s");
        expect(dms(-5.391)).toBe("-05°23'28\"");
        expect(dms(0)).toBe("+00°00'00\"");
        expect(dms(89.99999)).toBe("+90°00'00\"");
    });

    it("reports reading ages in a readable unit", () => {
        expect(ageText(400)).toBe("0.4 s");
        expect(ageText(12_400)).toBe("12 s");
        expect(ageText(180_000)).toBe("3 min");
    });
});

describe("coordinate input", () => {
    it("accepts decimal and sexagesimal RA within 0 <= RA < 24", () => {
        expect(parseRa("5.588")).toBeCloseTo(5.588, 9);
        expect(parseRa("05:35:16.8")).toBeCloseTo(5.588, 6);
        expect(parseRa("5 35 16.8")).toBeCloseTo(5.588, 6);
        expect(parseRa("05h35m16.8s")).toBeCloseTo(5.588, 6);
        expect(parseRa("05:30")).toBeCloseTo(5.5, 9);
        expect(parseRa("0")).toBe(0);
    });

    it("rejects RA out of range, signed, or with minutes or seconds of 60", () => {
        for (const text of ["24", "24:00:00", "-1", "+5", "05:60:00", "05:30:60", "abc", "", "5:30.5:10"]) {
            expect(parseRa(text), text).toBeNull();
        }
    });

    it("accepts a signed Dec within ±90, including -00:30", () => {
        expect(parseDec("-05:23:28")).toBeCloseTo(-5.391111, 5);
        expect(parseDec("-05°23'28\"")).toBeCloseTo(-5.391111, 5);
        expect(parseDec("+45")).toBe(45);
        expect(parseDec("-00:30:00")).toBeCloseTo(-0.5, 9);
        expect(parseDec("90")).toBe(90);
        expect(parseDec("90:00:01")).toBeNull();
        expect(parseDec("-91")).toBeNull();
    });
});

describe("mount controls follow O10 and D19", () => {
    it("gates GoTo on CanSlewAsync, not CanSlew: a compliant Alpaca mount reports CanSlew false", () => {
        const gates = mountActionGates(mount(), descriptor({ can_slew: false, can_slew_async: true, can_sync: true }));
        expect(gates.goto.enabled).toBe(true);
        expect(gates.center.enabled).toBe(true);
        const refused = mountActionGates(mount(), descriptor({ can_slew: true, can_slew_async: false }));
        expect(refused.goto).toEqual({ enabled: false, reason: expect.stringMatching(/CanSlewAsync/) });
    });

    it("never gates abort: not on capabilities, park, slewing or a missing status", () => {
        const none = descriptor({ can_slew_async: false, can_park: false, can_unpark: false, can_find_home: false, can_set_tracking: false });
        for (const block of [mount({ parked: true }), mount({ slewing: true }), null, undefined]) {
            expect(mountActionGates(block, none).abort.enabled).toBe(true);
        }
    });

    it("refuses motion on a parked mount, but allows unpark", () => {
        const gates = mountActionGates(mount({ parked: true }), descriptor({ can_slew_async: true, can_park: true, can_unpark: true, can_find_home: true, can_set_tracking: true }));
        expect(gates.goto.enabled).toBe(false);
        expect(gates.park.enabled).toBe(false);
        expect(gates.home.enabled).toBe(false);
        expect(gates.unpark.enabled).toBe(true);
        expect(mountActionGates(mount({ parked: false }), null).unpark.enabled).toBe(false);
    });

    it("gates park and home on CanPark and CanFindHome (the ESP32 profile has no home)", () => {
        const gates = mountActionGates(mount(), descriptor({ can_slew_async: true, can_park: true, can_find_home: false }));
        expect(gates.park.enabled).toBe(true);
        expect(gates.home.enabled).toBe(false);
    });

    it("leaves controls to the daemon's gate when capabilities were never read", () => {
        const gates = mountActionGates(mount(), null);
        expect([gates.goto, gates.sync, gates.park, gates.home, gates.tracking].every((gate) => gate.enabled)).toBe(true);
    });

    it("refuses a site write while the mount slews", () => {
        expect(mountActionGates(mount({ slewing: true }), null).siteSync.enabled).toBe(false);
    });

    it("refuses a J2000 GoTo, sync and center while the native frame is unknown, with the daemon's reason", () => {
        // The interim OLS INDI adapter without an override, or an Alpaca
        // mount below InterfaceVersion 2: the daemon cannot convert J2000.
        const unknown = { ...descriptor({ can_slew_async: true, can_sync: true, native_frame: "unknown", frame_reason: "ols_legacy_frame_unknown" }), frame: "unknown" as const };
        const gates = mountActionGates(mount({ frame: "unknown" }), unknown);
        for (const gate of [gates.goto, gates.sync, gates.center, gates.frame]) {
            expect(gate.enabled).toBe(false);
            expect(gate.reason).toMatch(/native frame is unknown \(ols legacy frame unknown\)/);
            expect(gate.reason).toMatch(/equatorial_system_override/);
            expect(gate.reason).toMatch(/mount-native/);
        }
        // Nothing else is gated on the frame, abort least of all.
        expect([gates.park, gates.home, gates.tracking, gates.siteSync, gates.abort].every((gate) => gate.enabled)).toBe(true);
        // The operator's explicit mount-native choice passes unconverted.
        const native = mountActionGates(mount({ frame: "unknown" }), unknown, "mount");
        expect([native.goto, native.sync, native.center, native.frame].every((gate) => gate.enabled)).toBe(true);
        // The live status alone is enough to know the frame is unknown.
        expect(mountActionGates(mount({ frame: "unknown" }), null).goto.enabled).toBe(false);
        // A known native frame converts; a frame never reported is left to the daemon.
        expect(mountActionGates(mount({ frame: "jnow_apparent" }), descriptor({ can_slew_async: true })).goto.enabled).toBe(true);
        expect(mountActionGates(mount(), null).goto.enabled).toBe(true);
    });

    it("takes the native frame from the live status, then the descriptor", () => {
        const d = descriptor({ native_frame: "jnow_mean" });
        expect(mountNativeFrame(mount({ frame: "j2000" }), d)).toBe("j2000");
        expect(mountNativeFrame(mount({ frame: null }), d)).toBe("jnow_apparent");
        expect(mountNativeFrame(mount(), { ...d, frame: undefined as unknown as MountDescriptor["frame"] })).toBe("jnow_mean");
        expect(mountNativeFrame(null, null)).toBeNull();
    });
});

describe("abort paths survive a failed status poll (D19)", () => {
    it("offers the abort for a connected, lost or degraded device, from the last known status", () => {
        expect(abortAvailable({ connected: true }, null)).toBe(true);
        expect(abortAvailable(null, { connected: false, state: "lost" })).toBe(true);
        expect(abortAvailable(undefined, { connected: false, state: "degraded" })).toBe(true);
        expect(abortAvailable({ connected: false }, { connected: false, state: "disconnected" })).toBe(false);
        expect(abortAvailable(null, null)).toBe(false);
    });
});

describe("guider controls", () => {
    it("dithers only while guiding and never disables stop", () => {
        expect(guiderActionGates({ state: "guiding" }).dither.enabled).toBe(true);
        for (const state of ["stopped", "looping", "calibrating", "paused", "unknown", null] as const) {
            expect(guiderActionGates({ state }).dither.enabled, String(state)).toBe(false);
            expect(guiderActionGates({ state }).stop.enabled).toBe(true);
        }
        expect(guiderActionGates(null).stop.enabled).toBe(true);
        expect(guiderActionGates({ state: "paused" }).resume.enabled).toBe(true);
        expect(guiderActionGates({ state: "guiding" }).start.enabled).toBe(false);
    });
});

describe("safety and weather", () => {
    it("never shows an unknown or stale monitor as safe", () => {
        expect(safetyView("unknown", null)).toMatchObject({ label: "UNKNOWN", tone: "warn" });
        expect(safetyView(undefined, { connected: true, level: "safe", is_safe: null })).toMatchObject({ label: "UNKNOWN" });
        expect(safetyView("safe", { connected: true, level: "safe", is_safe: true })).toMatchObject({ label: "SAFE", tone: "good" });
        expect(safetyView("unsafe", null)).toMatchObject({ label: "UNSAFE", tone: "bad" });
        expect(safetyView(undefined, { connected: true, level: "warning", is_safe: false })).toMatchObject({ label: "WARNING" });
        expect(safetyView("none", null)).toMatchObject({ label: "NO MONITOR", tone: "muted" });
    });

    it("lists every ObservingConditions field and never fills one the device did not report", () => {
        const rows = weatherRows({ connected: true, fields: {
            temperature_c: { value: 0, state: "fresh", age_ms: 1500 },
            humidity_pct: { value: null, state: "unsupported" },
        } });
        expect(rows).toHaveLength(13);
        const by = (key: string) => rows.find((row) => row.key === key)!;
        expect(by("temperature_c")).toMatchObject({ value: "0.0 °C", state: "fresh", age: "2 s", tone: "good" });
        expect(by("humidity_pct")).toMatchObject({ value: UNKNOWN, state: "unsupported", tone: "muted" });
        expect(by("wind_speed_ms")).toMatchObject({ value: UNKNOWN, state: "not reported" });
    });
});

describe("resume blocks, busy flags and e-stop", () => {
    const block = (kind: ResumeBlock["kind"]): ResumeBlock => ({ id: "b", kind, raised_at: "2026-10-26T20:00:00Z", reason: "r" });

    it("names each block and lets the operator clear all but the safety blocks", () => {
        expect(resumeBlockTitle(block({ kind: "outcome_unknown", device: "filter_wheel", op: "filter_handshake" }))).toBe("Outcome unknown: filter wheel filter handshake");
        expect(resumeBlockTitle(block({ kind: "estop" }))).toBe("Emergency stop");
        expect(operatorClears(block({ kind: "estop" }))).toBe(true);
        expect(operatorClears(block({ kind: "mount_lost" }))).toBe(true);
        expect(operatorClears(block({ kind: "safety_unsafe" }))).toBe(false);
        expect(operatorClears(block({ kind: "safety_unknown" }))).toBe(false);
    });

    it("lists the active busy flags", () => {
        expect(busyLabels(null)).toEqual([]);
        expect(busyLabels({ exposing: true, autofocus: false, slewing: true, settling: false, parking: false, flipping: false, centering: false,
            dithering: false, guider_settling: false, filter_changing: false, cover_moving: false, calibrator_changing: false })).toEqual(["exposing", "slewing"]);
    });

    it("says which devices did not confirm an e-stop", () => {
        expect(estopSummary({ status: "stopped", results: [{ kind: "mount", ok: true }], elapsed_ms: 40 }))
            .toBe("E-stop: every device confirmed — mount stopped in 40 ms.");
        expect(estopSummary({ status: "partial", results: [{ kind: "mount", ok: true }, { kind: "focuser", ok: false, error: { message: "timeout" } }] }))
            .toMatch(/some devices did not confirm — mount stopped, focuser NOT CONFIRMED \(timeout\)/);
        expect(estopSummary({ status: "stopped", results: [] })).toMatch(/no devices were connected/);
    });
});

describe("qualification and health", () => {
    it("reads qualification from an identity object or a driver entry", () => {
        expect(qualificationLabel({ state: "implemented_unqualified" })).toBe("unqualified");
        expect(qualificationLabel("planned")).toBe("planned");
        expect(qualificationLabel({ state: "qualified", record_id: "A23-1" })).toBe("qualified");
        expect(qualificationTone({ state: "qualified" })).toBe("good");
        expect(qualificationTone(null)).toBe("muted");
    });

    it("shows the failure class, the reason and a different last error once each", () => {
        expect(healthDetail({ connected: false, state: "lost", failure_class: "outcome_unknown", reason: "no reply", last_error: "no reply" }))
            .toBe("lost · outcome unknown · no reply");
        expect(healthDetail({ connected: true, state: "ready", reason: null, last_error: "old timeout" })).toBe("last error: old timeout");
        expect(healthDetail({ connected: true, state: "ready" })).toBeNull();
    });
});
