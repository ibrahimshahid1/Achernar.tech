import { describe, expect, it } from "vitest";
import { cardsFromDaemon, UNKNOWN } from "./useEquipment";
import type { EquipmentState } from "../types/achernar";

/**
 * These moved here from Django when `status_to_equipment()` was deleted.
 *
 * The rule: a value the system did not measure must never render as one it
 * could plausibly have measured. Zero is a legitimate temperature, a
 * legitimate focuser step, and — as RA 0h / Dec +0deg — a legitimate sky
 * coordinate, so none of them can double as "unknown".
 */
const build = (status: Partial<EquipmentState>) =>
    cardsFromDaemon({ connected: true, ...status } as EquipmentState, "test", "test site");

describe("card construction: unmeasured values", () => {
    it("renders a null camera temperature as unknown, not 0.0 C", () => {
        const [card] = build({ camera: { name: "DSLR", connected: true, cooling: false, temperature: null } });
        expect(card.meta!.Temp).toBe(UNKNOWN);
    });

    it("still formats a real temperature", () => {
        const [card] = build({ camera: { name: "ASI", connected: true, cooling: false, temperature: -10.14 } });
        expect(card.meta!.Temp).toBe("-10.1 C");
    });

    it("reports 0.0 C as a reading, because it is one", () => {
        const [card] = build({ camera: { name: "ASI", connected: true, cooling: false, temperature: 0 } });
        expect(card.meta!.Temp).toBe("0.0 C");
    });

    it("renders an unknown mount position as unknown, not the origin", () => {
        const [card] = build({ mount: { name: "Mount", connected: true, ra: null, dec: null, tracking: null } });
        expect(card.meta!.RA).toBe(UNKNOWN);
        expect(card.meta!.Dec).toBe(UNKNOWN);
    });

    it("still formats a real pointing", () => {
        const [card] = build({ mount: { name: "Mount", connected: true, ra: 5.588, dec: -5.391, tracking: true } });
        expect(card.meta!.RA).toBe("5.59");
        expect(card.meta!.Dec).toBe("-5.39");
    });

    it("renders an unread focuser encoder as unknown, not step 0", () => {
        const [card] = build({ focuser: { name: "Focuser", connected: true, position: null, moving: false } });
        expect(card.meta!.Position).toBe(UNKNOWN);
    });

    it("reports focuser step 0 as a position, because it is one", () => {
        const [card] = build({ focuser: { name: "Focuser", connected: true, position: 0, moving: false } });
        expect(card.meta!.Position).toBe(0);
    });

    it("does not claim a mount is unparked when park state was never read", () => {
        const [card] = build({ mount: { name: "Mount", connected: true, ra: null, dec: null, tracking: null, parked: null } });
        expect(card.tags).toContain("park state unknown");
        expect(card.tags).toContain("tracking unknown");
        expect(card.tags).not.toContain("unparked");
    });

    it("removes devices the daemon reports as disconnected", () => {
        const cards = cardsFromDaemon(
            {
                connected: true,
                mount: { name: "Mount", connected: false, ra: null, dec: null, tracking: null },
                camera: { name: "Cam", connected: false, cooling: false, temperature: null },
                focuser: { name: "Foc", connected: false, position: null, moving: false },
            } as EquipmentState,
            "test",
            "test site",
        );
        expect(cards).toEqual([]);
    });

    it("builds no cards for hardware the daemon does not report at all", () => {
        expect(cardsFromDaemon({ connected: false } as EquipmentState, "test", "test site")).toEqual([]);
    });

    it("counts only explicitly connected devices, including a reported filter wheel", () => {
        expect(build({
            camera: { connected: true, name: "Camera", cooling: false, temperature: null },
            mount: { connected: false, name: "Old mount", ra: null, dec: null, tracking: null },
            focuser: null,
            filter_wheel: { connected: true, name: "Wheel", slot: 0 },
        }).map((card) => card.type)).toEqual(["camera", "filter-wheel"]);
        expect(build({ camera: { name: "Cached Canon", cooling: false, temperature: null } })).toEqual([]);
    });

    it("clears a previous camera when the next snapshot is null or stale", () => {
        const camera = { connected: true, name: "Canon", cooling: false, temperature: null };
        expect(build({ camera })).toHaveLength(1);
        expect(build({ camera: null })).toEqual([]);
        expect(build({ camera, connected: false })).toEqual([]);
        expect(build({ camera, telemetry_stale: true })).toEqual([]);
        expect(build({ camera, daemon_connected: false })).toEqual([]);
    });
});
