import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { DriverCapability, MountDescriptor } from "../api/achernar";
import type { ResumeBlock } from "../types/achernar";
import EquipmentCard from "./EquipmentCard";
import { FilterWheelControl } from "./filter";
import FocuserControls from "./focuser";
import GuiderPanel from "./guider";
import HardwareDeviceSelector from "./HardwareDeviceSelector";
import { DEVICE_KINDS } from "./hardwareDeviceSelection";
import { MountControls, MountStatus } from "./mount";
import { BusyFlags, EmergencyStopButton, ResumeBlockBanner, SafetyPanel } from "./safety";
import WeatherPanel from "./weather";

/**
 * The panels' safety-relevant rendering: which controls exist, which are
 * enabled, and that unknown readings never render as values. Rendered to
 * static markup, so no DOM environment is needed.
 */
const run = async () => undefined;
const noop = () => undefined;

/** The opening tag of the button whose text is `label`. */
function buttonTag(html: string, label: string): string {
    const index = html.indexOf(`>${label}</button>`);
    expect(index, `button "${label}"`).toBeGreaterThan(-1);
    return html.slice(html.lastIndexOf("<button", index), index + 1);
}
const isDisabled = (html: string, label: string) => /\sdisabled=""/.test(buttonTag(html, label));

const descriptor = (capabilities: MountDescriptor["capabilities"]): MountDescriptor => ({
    identity: { kind: "mount", driver: "alpaca", identity_key: "alpaca:esp32", connection_id: "c1", name: "ESP32",
        setup_url: "http://rig.local/setup/v1/telescope/0/setup" },
    capabilities,
    frame: "jnow_apparent",
    site: null,
    can_flip: false,
    qualification: { state: "implemented_unqualified" },
});

describe("emergency stop", () => {
    it("is one click, enabled whenever the daemon is reachable, and reports unconfirmed devices", () => {
        const html = renderToStaticMarkup(createElement(EmergencyStopButton, { online: true, pending: true, abortExposure: false,
            onAbortExposureChange: noop, onStop: noop,
            report: { status: "partial", results: [{ kind: "mount", ok: true }, { kind: "focuser", ok: false }] } }));
        // Still enabled while a previous stop is pending: a second press sends again.
        expect(isDisabled(html, "STOPPING…")).toBe(false);
        expect(html).toContain("focuser NOT CONFIRMED");
        const offline = renderToStaticMarkup(createElement(EmergencyStopButton, { online: false, pending: false, abortExposure: false,
            onAbortExposureChange: noop, onStop: noop }));
        expect(isDisabled(offline, "E-STOP")).toBe(true);
    });
});

describe("resume-block banner", () => {
    const block = (id: string, kind: ResumeBlock["kind"]): ResumeBlock => ({ id, kind, raised_at: "2026-10-26T20:00:00Z", reason: `reason ${id}` });

    it("renders nothing without blocks", () => {
        expect(renderToStaticMarkup(createElement(ResumeBlockBanner, { blocks: [], online: true, busy: false, onClear: noop }))).toBe("");
    });

    it("offers a clear only for operator-clearable blocks", () => {
        const html = renderToStaticMarkup(createElement(ResumeBlockBanner, { blocks: [block("a", { kind: "estop" }), block("b", { kind: "safety_unsafe" })],
            online: true, busy: false, onClear: noop }));
        expect(html).toContain("2 resume blocks");
        expect(html).toContain("Emergency stop");
        expect(html).toContain("Safety monitor reports unsafe");
        expect(html.match(/>Clear<\/button>/g)).toHaveLength(1);
        expect(html).toContain("clears on a confirmed safe reading");
    });

    it("shows active busy flags", () => {
        const html = renderToStaticMarkup(createElement(BusyFlags, { busy: { exposing: false, autofocus: false, slewing: true, settling: true, parking: false,
            flipping: false, centering: false, dithering: false, guider_settling: false, filter_changing: false, cover_moving: false, calibrator_changing: false } }));
        expect(html).toContain("Busy: slewing, settling");
    });
});

describe("mount panel", () => {
    const mount = { name: "ESP32", connected: true, tracking: true, ra: 5.588, dec: -5.391, parked: false, slewing: false, frame: "jnow_apparent" as const };

    it("disables GoTo without CanSlewAsync but keeps abort, and offers no jog or pulse control", () => {
        const html = renderToStaticMarkup(createElement(MountControls, { mount, descriptor: descriptor({ can_slew_async: false, can_park: true }),
            online: true, busy: true, run }));
        expect(isDisabled(html, "Slew")).toBe(true);
        // Abort is never gated, not even while another command is in flight.
        expect(isDisabled(html, "Abort mount motion")).toBe(false);
        expect(html).not.toMatch(/move-axis|Pulse guide<\/button>/);
        expect(html).toContain("send J2000");
        expect(html).toContain("Driver setup page");
    });

    it("keeps abort for a lost mount whose status block is gone (D19)", () => {
        const html = renderToStaticMarkup(createElement(MountControls, { mount: null, health: { connected: false, state: "lost" }, online: true, busy: false, run }));
        expect(isDisabled(html, "Abort mount motion")).toBe(false);
        expect(html).toContain("The mount is lost");
        const absent = renderToStaticMarkup(createElement(MountControls, { mount: null, online: true, busy: false, run }));
        expect(isDisabled(absent, "Abort mount motion")).toBe(true);
    });

    it("shows both frames and unknown readings as unknown", () => {
        const html = renderToStaticMarkup(createElement(MountStatus, { mount: { ...mount, ra_j2000: null, alt: null, pier_side: "east", alignment_mode: "polar", can_flip: false } }));
        expect(html).toContain("05h35m16.8s");
        expect(html).toContain("RA (JNow (apparent))");
        expect(html).toContain("Polar (fork)");
        expect(html).toMatch(/RA \(J2000\)<\/dt><dd>--<\/dd>/);
        expect(html).toMatch(/Altitude<\/dt><dd>--<\/dd>/);
        expect(html).toContain("not capable");
    });
});

describe("guider panel", () => {
    it("shows RMS as unknown without a pixel scale, and keeps stop enabled", () => {
        const html = renderToStaticMarkup(createElement(GuiderPanel, { guiding: { connected: true, name: "PHD2", state: "looping", rms_total: null,
            rms_total_px: 0.41, rms_ra_px: 0.3, rms_dec_px: 0.28, qualification: { state: "implemented_unqualified" } }, online: true, busy: true, run }));
        expect(html).toMatch(/RMS total<\/dt><dd>--<\/dd>/);
        expect(html).toContain("0.41 (RA 0.30, Dec 0.28)");
        expect(isDisabled(html, "Stop guiding")).toBe(false);
        expect(isDisabled(html, "Dither")).toBe(true);
        expect(html).toContain("UNQUALIFIED".toLowerCase());
    });

    it("reports a failed dither settle", () => {
        const html = renderToStaticMarkup(createElement(GuiderPanel, { guiding: { connected: true, state: "guiding", rms_total: 0.6,
            last_dither: { at: "2026-10-26T20:00:00Z", dx: 1.2, dy: -0.8, settle_status: 1 } }, online: true, busy: false, run }));
        expect(html).toContain("settle failed (1)");
        expect(isDisabled(html, "Dither")).toBe(false);
    });
});

describe("filter wheel, focuser and environment panels", () => {
    it("lists slots with profile offsets and marks the current one", () => {
        const html = renderToStaticMarkup(createElement(FilterWheelControl, { wheel: { connected: true, name: "Wheel", slot: 1, filter: "Ha",
            verified: true, slots: [{ index: 0, name: "L", offset: 0 }, { index: 1, name: "Ha", offset: 120 }, { index: 2, name: "OIII", offset: null }] },
        online: true, busy: false, run }));
        expect(html).toContain("slot 1 · Ha");
        expect(html).toContain("eq-current");
        expect(html).toMatch(/<td>OIII<\/td><td>--<\/td>/);
        expect(html).toContain("Import TOML seeds");
        expect(isDisabled(html, "Abort change")).toBe(false);
    });

    it("restricts an unknown-origin relative focuser and offers to declare its origin", () => {
        const html = renderToStaticMarkup(createElement(FocuserControls, { focuser: { connected: true, name: "ESP32 focuser", position: null, moving: null,
            absolute: false, position_source: "emulated_origin_unknown", temperature: null }, online: true, busy: false, run }));
        expect(html).toContain("The origin is unknown");
        expect(html).toContain("Declare origin");
        expect(isDisabled(html, "Move to position")).toBe(true);
        expect(html).toMatch(/Temperature<\/dt><dd>--<\/dd>/);
        expect(isDisabled(html, "Halt focuser")).toBe(false);
    });

    it("renders weather fields with their state and the safety state fail-safe", () => {
        const weather = renderToStaticMarkup(createElement(WeatherPanel, { online: true, weather: { connected: true, name: "Boltwood",
            fields: { temperature_c: { value: -2.5, state: "fresh", age_ms: 800 } } } }));
        expect(weather).toContain("-2.5 °C");
        expect(weather).toContain("not reported");
        const safety = renderToStaticMarkup(createElement(SafetyPanel, { online: true, summary: { state: "unknown" }, monitor: null }));
        expect(safety).toContain("UNKNOWN");
        expect(safety).not.toContain(">SAFE<");
    });
});

describe("device selector", () => {
    const drivers: DriverCapability[] = [
        { id: "phd2", label: "PHD2", compiled: true, enabled: true, qualification: "implemented_unqualified" },
        { id: "direct", label: "Direct guider", compiled: false, enabled: false, qualification: "planned", reason: "Tier 2." },
    ];
    const props = (kind: string, extra: Record<string, unknown> = {}) => ({
        spec: DEVICE_KINDS.find((spec) => spec.kind === kind)!, drivers, online: true, report: null, scanning: false, busy: false,
        discoveryAllowed: true, discover: run, resetDiscovery: noop, action: run, ...extra,
    });

    it("connects the guider by PHD2 host and port, with no discovery and planned drivers disabled", () => {
        const html = renderToStaticMarkup(createElement(HardwareDeviceSelector, props("guider")));
        expect(html).not.toContain("Refresh Guider devices");
        expect(html).toContain("<details class=\"hardware-manual-connect\" open=\"\">");
        expect(html).toContain("PHD2 host");
        expect(html).toContain("4400 + n − 1");
        expect(html).toMatch(/<option value="direct" disabled="">Direct guider \(planned\)<\/option>/);
    });

    it("shows a lost device as lost, with its reason, and lets it be disconnected", () => {
        const html = renderToStaticMarkup(createElement(HardwareDeviceSelector, props("mount", {
            device: null, health: { connected: false, state: "lost", failure_class: "timeout", reason: "no reply in 10 s" } })));
        expect(html).toContain(">Lost</strong>");
        expect(html).toContain("lost · timeout · no reply in 10 s");
        expect(isDisabled(html, "Disconnect")).toBe(false);
    });

    it("badges a connected device's qualification", () => {
        const html = renderToStaticMarkup(createElement(HardwareDeviceSelector, props("filter_wheel", {
            device: { connected: true, name: "Sim wheel", qualification: { state: "implemented_unqualified" } } })));
        expect(html).toContain("qual-badge");
        expect(html).toContain(">unqualified</span>");
    });

    it("shows the qualification badge on a card and keeps it out of the tag row", () => {
        const html = renderToStaticMarkup(createElement(EquipmentCard, { model: { id: "x", type: "guider", name: "PHD2", site: "s", status: "nominal",
            tags: ["guiding", "PHD2", "unqualified"] } }));
        expect(html).toContain("GDR");
        expect(html).toMatch(/qual-badge[^>]*>unqualified</);
        expect(html).not.toMatch(/class="tag">unqualified/);
    });
});
