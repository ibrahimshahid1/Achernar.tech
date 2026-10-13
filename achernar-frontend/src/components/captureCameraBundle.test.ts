import { describe, expect, it, vi } from "vitest";
import type { CameraDescriptor, CamFormat } from "../api/achernar";
import { canStartCameraPreview, captureCameraBundle } from "./captureCameraBundle";

const sensorFormat: CamFormat = {
    index: 7,
    format: "Mono16",
    width: 1280,
    height: 960,
    bin: 2,
    framerate: -1,
    mode: "liveview",
};

function descriptor(dedicated = false, streaming = false): CameraDescriptor {
    return {
        connected: true,
        connection_id: "camera-connection",
        name: "Test camera",
        driver_id: "test",
        device_index: 0,
        hardware_id: null,
        activity: streaming ? "streaming" : "idle",
        streaming,
        active_format_index: streaming ? sensorFormat.index : null,
        operation_id: null,
        exposure_progress: null,
        temperature_c: null,
        cooler_on: null,
        last_error: null,
        capabilities: {
            supports_stream: true,
            supports_dedicated_still: dedicated,
            supports_exposure_abort: false,
            supports_cooling: false,
            supports_temperature: false,
            supports_hardware_binning: false,
            supports_roi: false,
        },
        controls: [],
        formats: [sensorFormat],
        latest_frame: null,
    };
}

function mockApi() {
    return {
        capture: vi.fn().mockResolvedValue({ capture_id: "archived-capture" }),
        startCameraStream: vi.fn().mockResolvedValue({ status: "streaming" }),
    };
}

describe("camera capture routing", () => {
    it.each(["liveview", "still", "no-format"] as const)(
        "captures a dedicated still without starting a %s stream or requesting preview binning",
        async (selection) => {
            const api = mockApi();
            const streamStarted = vi.fn();
            const format = selection === "no-format" ? undefined : { ...sensorFormat, mode: selection };
            const result = await captureCameraBundle(api, descriptor(true), format, { exposure: 3 }, streamStarted);
            expect(api.startCameraStream).not.toHaveBeenCalled();
            expect(streamStarted).not.toHaveBeenCalled();
            expect(api.capture).toHaveBeenCalledWith({
                exposure: 3,
                binning: 1,
                retention_policy: "retain_on_camera",
            });
            expect(result.capture_id).toBe("archived-capture");
        },
    );

    it("starts the exact published sensor format before capture and immediately reports the active stream", async () => {
        const api = mockApi();
        const events: string[] = [];
        api.startCameraStream.mockImplementation(async () => { events.push("start"); });
        api.capture.mockImplementation(async () => { events.push("capture"); return { capture_id: "sensor" }; });
        await captureCameraBundle(api, descriptor(), sensorFormat, { exposure: 3, gain: 20 }, () => events.push("streaming"));
        expect(events).toEqual(["start", "streaming", "capture"]);
        expect(api.startCameraStream).toHaveBeenCalledWith(7);
        expect(api.capture).toHaveBeenCalledWith({ exposure: 3, gain: 20, binning: 2, retention_policy: "retain_on_camera" });
    });

    it("uses an already running sensor stream without restarting it", async () => {
        const api = mockApi();
        await captureCameraBundle(api, descriptor(false, true), sensorFormat, { exposure: 3 }, vi.fn());
        expect(api.startCameraStream).not.toHaveBeenCalled();
        expect(api.capture).toHaveBeenCalledOnce();
    });

    it("does not capture or report streaming when sensor stream startup fails", async () => {
        const api = mockApi();
        api.startCameraStream.mockRejectedValue(new Error("Stream unavailable"));
        const streamStarted = vi.fn();
        await expect(captureCameraBundle(api, descriptor(), sensorFormat, { exposure: 3 }, streamStarted))
            .rejects.toThrow("Stream unavailable");
        expect(api.capture).not.toHaveBeenCalled();
        expect(streamStarted).not.toHaveBeenCalled();
    });

    it("refuses a disconnected descriptor before issuing commands", async () => {
        const api = mockApi();
        await expect(captureCameraBundle(api, { ...descriptor(), connected: false }, sensorFormat, { exposure: 3 }, vi.fn()))
            .rejects.toThrow("No camera connected");
        expect(api.startCameraStream).not.toHaveBeenCalled();
        expect(api.capture).not.toHaveBeenCalled();
    });

    it("requires a known sensor format even when a stream was reported active", async () => {
        const api = mockApi();
        await expect(captureCameraBundle(api, descriptor(false, true), undefined, { exposure: 3 }, vi.fn()))
            .rejects.toThrow("Select an available sensor stream format");
        expect(api.capture).not.toHaveBeenCalled();
    });

    it("disables still-stream startup only for a camera with a dedicated still capability", () => {
        const stillFormat: CamFormat = { ...sensorFormat, mode: "still", framerate: 0 };
        expect(canStartCameraPreview(descriptor(true), stillFormat)).toBe(false);
        expect(canStartCameraPreview(descriptor(true), sensorFormat)).toBe(true);
        expect(canStartCameraPreview(descriptor(), stillFormat)).toBe(true);
        expect(canStartCameraPreview(null, sensorFormat)).toBe(false);
    });
});
