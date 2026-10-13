import { describe, expect, it } from "vitest";
import { freshEquipmentStatus } from "./useDaemonStatus";
import type { EquipmentState } from "../types/achernar";

const previous: EquipmentState = { connected: true, camera: { name: "Canon", connected: true, cooling: false, temperature: null } };

describe("live hardware status freshness", () => {
    it("hides React Query's last successful camera immediately after a failed poll", () => {
        expect(freshEquipmentStatus(previous, false, 1000, 1500)).toBe(previous);
        expect(freshEquipmentStatus(previous, true, 1000, 1500)).toBeUndefined();
    });

    it("expires cached hardware when a request hangs and provides no new snapshot", () => {
        expect(freshEquipmentStatus(previous, false, 1000, 5001)).toBeUndefined();
    });

    it("rejects stale relay snapshots even if their HTTP request just completed", () => {
        expect(freshEquipmentStatus({ ...previous, telemetry_stale: true }, false, 1000, 1000)).toBeUndefined();
        expect(freshEquipmentStatus({ ...previous, daemon_connected: false }, false, 1000, 1000)).toBeUndefined();
    });

    it("distinguishes a reachable daemon with no connected hardware from an unavailable daemon", () => {
        const empty = { connected: false, camera: null };
        expect(freshEquipmentStatus(empty, false, 1000, 1100)).toBe(empty);
        expect(freshEquipmentStatus(undefined, false, 0, 1100)).toBeUndefined();
    });
});
