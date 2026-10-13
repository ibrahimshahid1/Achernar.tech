import { describe, expect, it } from "vitest";
import { cameraPreviewWsUrl, decodePreviewFrame, PreviewFrameTracker, shouldShowPreview } from "./cameraPreview";

const localOptions = {
    pageUrl: "http://127.0.0.1:5173/run",
    daemonApiBase: "http://localhost:3030/api/v1",
    directDaemon: false,
};

describe("image-only camera transport", () => {
    it("selects the local binary socket for a same-machine browser", () => {
        expect(cameraPreviewWsUrl(localOptions)).toBe("ws://localhost:3030/api/v1/camera/stream");
    });

    it("recognizes IPv6 and IPv4 loopback addresses", () => {
        expect(cameraPreviewWsUrl({ ...localOptions, pageUrl: "http://[::1]:5173", daemonApiBase: "http://127.0.0.2:3030/api/v1/" }))
            .toBe("ws://127.0.0.2:3030/api/v1/camera/stream");
    });

    it.each([
        { pageUrl: "https://achernar.example/run" },
        { pageUrl: "http://192.168.1.5:5173/run" },
        { daemonApiBase: "http://192.168.1.6:3030/api/v1" },
        { daemonApiBase: "http://localhost.example:3030/api/v1" },
    ])("keeps remote browser previews on the relay by default: %o", (overrides) => {
        expect(cameraPreviewWsUrl({ ...localOptions, ...overrides })).toBeNull();
    });

    it("allows an explicitly configured remote preview without changing command mode", () => {
        expect(cameraPreviewWsUrl({ ...localOptions, pageUrl: "https://achernar.example/run", explicitWsUrl: "wss://rig.example/preview" }))
            .toBe("wss://rig.example/preview");
        expect(localOptions.directDaemon).toBe(false);
    });

    it("preserves explicitly selected direct daemon mode on a LAN", () => {
        expect(cameraPreviewWsUrl({ ...localOptions, daemonApiBase: "http://rig.local:3030/api/v1", directDaemon: true }))
            .toBe("ws://rig.local:3030/api/v1/camera/stream");
    });

    it.each(["http://rig.example/preview", "invalid"])("falls back for an invalid explicit socket: %s", (explicitWsUrl) => {
        expect(cameraPreviewWsUrl({ ...localOptions, explicitWsUrl })).toBeNull();
    });

    it("falls back when an HTTPS page would need an insecure socket", () => {
        expect(cameraPreviewWsUrl({ ...localOptions, pageUrl: "https://localhost:5173/run" })).toBeNull();
    });
});

function packet(timestamp = 1_791_572_034_123, width = 960, height = 640, format = 1): ArrayBuffer {
    const bytes = new ArrayBuffer(23);
    const header = new DataView(bytes);
    header.setUint32(0, width, true);
    header.setUint32(4, height, true);
    header.setBigUint64(8, BigInt(timestamp), true);
    header.setUint32(16, format, true);
    new Uint8Array(bytes, 20).set([0xff, 0xd8, 0xff]);
    return bytes;
}

describe("binary preview packets", () => {
    it("decodes the little-endian timestamp and JPEG payload without the header", () => {
        const decoded = decodePreviewFrame(packet());
        expect(decoded).toMatchObject({ width: 960, height: 640, timestampMs: 1_791_572_034_123 });
        expect(Array.from(new Uint8Array(decoded!.bytes))).toEqual([0xff, 0xd8, 0xff]);
    });

    it.each([new ArrayBuffer(0), new ArrayBuffer(20), packet(1, 0), packet(1, 1, 0), packet(1, 1, 1, 2)])(
        "ignores waiting, incomplete and unsupported packets",
        (bytes) => expect(decodePreviewFrame(bytes)).toBeNull(),
    );
});

describe("fresh frame accounting", () => {
    it("clears a last image on camera disconnect and when fresh frames stop", () => {
        expect(shouldShowPreview(true, 1000, 1500)).toBe(true);
        expect(shouldShowPreview(false, 1000, 1500)).toBe(false);
        expect(shouldShowPreview(true, 1000, 4001)).toBe(false);
        expect(shouldShowPreview(true, null, 1500)).toBe(false);
    });
    it("ignores duplicate and older timestamps across socket and snapshot deliveries", () => {
        const frames = new PreviewFrameTracker();
        expect(frames.accept(1000, 0)).toBe(true);
        expect(frames.accept(1000, 100, 1)).toBe(false);
        expect(frames.accept(999, 200, 0)).toBe(false);
        expect(frames.accept(1100, 300, 2)).toBe(true);
    });

    it("uses counters only when timestamp metadata is absent", () => {
        const frames = new PreviewFrameTracker();
        expect(frames.accept(undefined, 0, 5)).toBe(true);
        expect(frames.accept(undefined, 100, 5)).toBe(false);
        expect(frames.accept(undefined, 200, 4)).toBe(false);
        expect(frames.accept(undefined, 300, 6)).toBe(true);
        expect(frames.accept(undefined, 400)).toBe(false);
    });

    it("reports capture delivery rate and returns to zero when fresh frames stop", () => {
        const frames = new PreviewFrameTracker();
        for (let i = 0; i <= 10; i += 1) frames.accept(i * 100, i * 100);
        expect(frames.fps(1000)).toBe(10);
        // A 10 Hz socket retransmitting one frame does not count as 10 new frames.
        for (let i = 11; i <= 20; i += 1) frames.accept(1000, i * 100);
        expect(frames.fps(2000)).toBe(5);
        expect(frames.fps(3100)).toBe(0);
    });
});

// A newly mounted/reloaded component must reject a cached last frame by source
// clock, even when that is the first packet delivered on a new socket.
describe("source timestamp freshness", () => {
    it("rejects stale, future and missing source times after reload", async () => {
        const { sourcePreviewFresh } = await import("./cameraPreview");
        expect(sourcePreviewFresh(99_000,100_000)).toBe(true);
        for (const timestamp of [96_999,101_001,NaN,undefined]) expect(sourcePreviewFresh(timestamp,100_000)).toBe(false);
    });
});
