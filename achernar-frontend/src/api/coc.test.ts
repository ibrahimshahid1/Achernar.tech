import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { cocApi, freshCocSample, validRoi, type CocStatus } from "./coc";
import { achernarApi } from "./achernar";

vi.mock("./achernar", () => ({ achernarApi: { command: vi.fn() } }));
const command = vi.mocked(achernarApi.command);
const context = { generation: 7, connection_id: "camera-a", width: 1920, height: 1080 };
const frame = { stream_generation: 7, width: 1920, height: 1080 };
const now = 100_000;
const status = () => ({ contract_version: "coc-segmentation-v1", shadow_only: true, runtime_eligible: false,
    enabled: true, status: "ready", models: [], latest: { sample_id: "frame-a", context, source_timestamp: 99 } } as unknown as CocStatus);
const fresh = (data = status(), failed = false, updated = 99_000, time = now,
    camera: string | null = "camera-a", streaming = true, geometry: typeof frame | undefined = frame) =>
    freshCocSample(data, failed, updated, time, camera, streaming, geometry);

beforeEach(() => command.mockReset());
afterEach(() => vi.restoreAllMocks());

describe("COC camera lifecycle", () => {
    it("shows only fresh frames matching the authoritative camera and stream", () => {
        expect(fresh()?.sample_id).toBe("frame-a");
        expect(fresh(status(), false, 99_000, now, "camera-b")).toBeNull();
        expect(fresh(status(), false, 99_000, now, "camera-a", false)).toBeNull();
        expect(fresh(status(), false, 99_000, now, "camera-a", true, { ...frame, stream_generation: 8 })).toBeNull();
        expect(fresh(status(), false, 99_000, now, "camera-a", true, { ...frame, width: 960 })).toBeNull();
    });
    it("hides outputs on network errors, stale polling or source clock errors", () => {
        expect(fresh(status(), true)).toBeNull();
        expect(fresh(status(), false, 94_000)).toBeNull();
        for (const timestamp of [94, 102, NaN]) {
            const data = status(); data.latest!.source_timestamp = timestamp;
            expect(fresh(data)).toBeNull();
        }
    });
    it("does not display cached disabled or waiting predictions", () => {
        const data = status(); data.enabled = false; expect(fresh(data)).toBeNull();
        data.enabled = true; data.status = "waiting_for_frame"; expect(fresh(data)).toBeNull();
    });
    it("requires an independent nonempty ROI and current camera context", async () => {
        expect(validRoi({ left: 0, top: 0, right: 1, bottom: 1 })).toBe(true);
        for (const right of [0, -1, 2, NaN]) expect(validRoi({ left: 0, top: 0, right, bottom: 1 })).toBe(false);
        await expect(cocApi.configure(true, "teacher", { left: 0, top: 0, right: 1, bottom: 1 }, null)).rejects.toThrow(/current frame/);
        expect(command).not.toHaveBeenCalled();
    });
});

describe("COC transport contract", () => {
    it("routes through the shared command adapter and validates research identity", async () => {
        command.mockResolvedValue({ ok: true, payload: { ...status() } });
        expect((await cocApi.status()).runtime_eligible).toBe(false);
        expect(command).toHaveBeenCalledWith("coc.status", {});
        command.mockResolvedValue({ ok: true, payload: { ...status(), runtime_eligible: true } });
        await expect(cocApi.status()).rejects.toThrow(/expected COC shadow contract/);
    });
    it("rejects pending and failed commands without showing them as successful", async () => {
        command.mockResolvedValue({ ok: false, pending: true }); await expect(cocApi.status()).rejects.toThrow(/pending/);
        command.mockResolvedValue({ ok: false, error: "expired frame" }); await expect(cocApi.save("frame-a", "review")).rejects.toThrow("expired frame");
    });
    it("freezes and saves the exact selected sample ID", async () => {
        command.mockResolvedValue({ ok: true, payload: { ...status().latest! } });
        await cocApi.freeze("frame-a");
        expect(command).toHaveBeenLastCalledWith("coc.freeze_sample", { sample_id: "frame-a" });
        command.mockResolvedValue({ ok: true, payload: { status: "saved", sample_id: "frame-a", path: "/data/frame-a", training_approved: false } });
        await cocApi.save("frame-a", "thin cloud, uncertain");
        expect(command).toHaveBeenLastCalledWith("coc.save_sample", { sample_id: "frame-a", notes: "thin cloud, uncertain" });
    });
});
