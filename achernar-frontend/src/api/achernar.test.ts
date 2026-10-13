import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * The transport adapter.
 *
 * This replaced two parallel clients that implemented the same operations
 * twice and drifted. The thing worth testing is that one action name reaches
 * the right place on either transport, and that the daemon-side route comes
 * from the daemon's published registry rather than a hand-written map.
 */

const REGISTRY = [
    { name: "equipment.status", method: "GET", path: "status", description: "Status." },
    { name: "camera.abort", method: "POST", path: "camera/abort", description: "Stop." },
    { name: "camera.start_stream", method: "POST", path: "camera/stream/start", description: "Start." },
    { name: "camera.capture", method: "POST", path: "camera/capture", description: "Capture." },
    { name: "camera.descriptor", method: "GET", path: "camera", description: "Camera descriptor." },
    { name: "focuser.position", method: "GET", path: "focuser/position", description: "Position." },
    { name: "mount.sync", method: "POST", path: "mount/sync", description: "Sync." },
];

let transport: "django" | "daemon" = "daemon";
let fetchMock: ReturnType<typeof vi.fn>;

vi.mock("../lib/runtime", () => ({
    getHardwareTransport: () => transport,
    isTauriRuntime: () => false,
}));

function jsonResponse(body: unknown) {
    return {
        ok: true,
        status: 200,
        text: async () => JSON.stringify(body),
    } as unknown as Response;
}

/** Fresh module each test: the registry is cached in module state. */
async function loadApi() {
    vi.resetModules();
    return import("./achernar");
}

beforeEach(() => {
    transport = "daemon";
    fetchMock = vi.fn(async (url: string) =>
        String(url).endsWith("/commands") ? jsonResponse(REGISTRY) : jsonResponse({ ok: true }),
    );
    vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
    vi.unstubAllGlobals();
});

const urlsCalled = () => fetchMock.mock.calls.map((c) => String(c[0]));
const callFor = (fragment: string) =>
    fetchMock.mock.calls.find((c) => String(c[0]).includes(fragment));

describe("daemon transport", () => {
    it("passes the requested discovery scope to the daemon without inventing an endpoint", async () => {
        const { achernarApi } = await loadApi();
        await achernarApi.discoverDevices({ equipment_type: "focuser" });
        const call = callFor("/discover")!;
        expect(JSON.parse(String(call[1]!.body))).toEqual({ equipment_type: "focuser" });
        expect(urlsCalled().some((url) => url.endsWith("/commands"))).toBe(false);
    });

    it("routes an action to the path the daemon publishes", async () => {
        const { achernarApi } = await loadApi();
        await achernarApi.abortCamera();

        expect(urlsCalled().some((u) => u.endsWith("/commands"))).toBe(true);
        expect(callFor("camera/abort")).toBeTruthy();
        expect(callFor("camera/abort")![1]).toMatchObject({ method: "POST" });
    });

    it("issues a GET for commands the daemon publishes as GET", async () => {
        const { achernarApi } = await loadApi();
        await achernarApi.focuserPosition();

        const call = callFor("focuser/position")!;
        expect(call[1]?.method ?? "GET").toBe("GET");
        expect(call[1]?.body).toBeUndefined();
    });

    it("fetches the registry once and reuses it", async () => {
        const { achernarApi } = await loadApi();
        await achernarApi.abortCamera();
        await achernarApi.getStatus();
        await achernarApi.focuserPosition();

        expect(urlsCalled().filter((u) => u.endsWith("/commands"))).toHaveLength(1);
    });

    it("refuses an action this build does not publish, naming where to look", async () => {
        const { achernarApi } = await loadApi();
        await expect(achernarApi.command("focus.run")).rejects.toThrow(/does not publish "focus\.run"/);
    });

    it("keeps the query-parameter quirk for stream start", async () => {
        const { achernarApi } = await loadApi();
        await achernarApi.startCameraStream(3);

        expect(callFor("camera/stream/start")![0]).toContain("?format_index=3");
    });

    it("omits the query parameter when no format was chosen", async () => {
        const { achernarApi } = await loadApi();
        await achernarApi.startCameraStream();

        expect(String(callFor("camera/stream/start")![0])).not.toContain("?");
    });

    it("does not cache a failed registry fetch", async () => {
        const { achernarApi, commandRegistry } = await loadApi();
        fetchMock.mockImplementationOnce(async () => {
            throw new Error("connection refused");
        });

        await expect(commandRegistry()).rejects.toThrow();
        // A daemon that starts later must work without a page reload.
        await expect(achernarApi.abortCamera()).resolves.toBeUndefined();
    });
});

describe("django transport", () => {
    it("relays scoped discovery with the explicit endpoint and waits for its result", async () => {
        transport = "django";
        const { achernarApi } = await loadApi();
        await achernarApi.discoverDevices({ equipment_type: "mount", driver: "alpaca", option: "http://rig.local:11111" });
        const body = JSON.parse(String(callFor("/equipment/command/")![1]!.body));
        expect(body).toEqual({ action: "equipment.discover", payload: {
            equipment_type: "mount", driver: "alpaca", option: "http://rig.local:11111",
        }, wait_seconds: 90 });
        expect(urlsCalled().some((url) => url.endsWith("/discover"))).toBe(false);
    });

    it("waits for a DSLR format probe with the body's self-timer enabled", async () => {
        transport = "django";
        const { achernarApi } = await loadApi();
        await achernarApi.connectCamera("gphoto2", undefined, 0);
        const body = JSON.parse(String(callFor("/equipment/command/")![1]!.body));
        expect(body.wait_seconds).toBe(90);
        expect(body.payload).toMatchObject({ driver: "gphoto2", index: 0 });
    });

    it("waits for DSLR control acknowledgement without changing the requested choice", async () => {
        transport = "django";
        const { achernarApi } = await loadApi();
        await achernarApi.setCameraParameter("iso", 10);
        const body = JSON.parse(String(callFor("/equipment/command/")![1]!.body));
        expect(body).toEqual({
            action: "camera.set_parameter",
            payload: { name: "iso", value: 10 },
            wait_seconds: 15,
        });
    });

    it("allows an exposure and native download to complete before returning a pending result", async () => {
        transport = "django";
        const { achernarApi } = await loadApi();
        await achernarApi.capture({ exposure: 20 });
        const body = JSON.parse(String(callFor("/equipment/command/")![1]!.body));
        expect(body.wait_seconds).toBe(85);
        expect(body.payload.exposure).toBe(20);
    });

    it("posts {action, payload} to the relay and never reads the registry", async () => {
        transport = "django";
        const { achernarApi } = await loadApi();
        await achernarApi.abortCamera();

        expect(urlsCalled().some((u) => u.endsWith("/commands"))).toBe(false);
        const call = callFor("/equipment/command/")!;
        expect(JSON.parse(String(call[1]!.body))).toEqual({ action: "camera.abort", payload: {} });
    });

    it("sends the same action name the daemon would have matched", async () => {
        transport = "django";
        const { achernarApi } = await loadApi();
        await achernarApi.syncMount({ ra: 5.5, dec: -5.4 });

        const body = JSON.parse(String(callFor("/equipment/command/")![1]!.body));
        expect(body).toEqual({ action: "mount.sync", payload: { ra: 5.5, dec: -5.4 } });
    });
});

describe("command arguments", () => {
    /**
     * A dropped field must not become an instruction. `mount.sync` with a
     * defaulted `ra` tells the mount its position is RA 0h, corrupting the
     * pointing model; `focuser.move_absolute` drives toward a hard stop.
     */
    it("rejects a missing mount coordinate instead of defaulting it to zero", async () => {
        const { achernarApi } = await loadApi();
        await expect(
            achernarApi.syncMount({ ra: undefined as unknown as number, dec: -5.4 }),
        ).rejects.toThrow(/requires a numeric "ra"/);
        expect(callFor("mount/sync")).toBeUndefined();
    });

    it("rejects a missing focuser position instead of commanding step 0", async () => {
        const { achernarApi } = await loadApi();
        await expect(
            achernarApi.moveFocuserTo(undefined as unknown as number),
        ).rejects.toThrow(/requires a numeric "position"/);
    });

    it("accepts a genuine zero, which is a real coordinate and a real step", async () => {
        const { achernarApi } = await loadApi();
        await expect(achernarApi.syncMount({ ra: 0, dec: 0 })).resolves.toBeTruthy();
    });

    it("rejects a missing tracking flag instead of silently stopping tracking", async () => {
        const { achernarApi } = await loadApi();
        await expect(
            achernarApi.setTracking(undefined as unknown as boolean),
        ).rejects.toThrow(/requires a boolean "enabled"/);
    });
});

describe("capture retention transport", () => {
    const retention = {
        requested_policy: "retain_on_camera",
        effective_policy: "retain_on_camera",
        policy_version: 1,
        camera_copy_state: "unknown",
        camera_files_deleted: false,
        deletion_eligible: false,
        reason: "Persistent camera storage has not been verified.",
    };

    it.each(["daemon", "django"] as const)("preserves the retain request and reported evidence over %s", async (mode) => {
        transport = mode;
        fetchMock.mockImplementation(async (url: string) => {
            if (String(url).endsWith("/commands")) return jsonResponse(REGISTRY);
            const payload = { retention };
            return jsonResponse(mode === "django" ? { ok: true, payload } : payload);
        });
        const { achernarApi } = await loadApi();
        const result = await achernarApi.capture({ exposure: 2, retention_policy: "retain_on_camera" });
        const call = callFor(mode === "django" ? "/equipment/command/" : "camera/capture")!;
        const body = JSON.parse(String(call[1]!.body));
        const payload = mode === "django" ? body.payload : body;
        expect(payload).toEqual({ exposure: 2, retention_policy: "retain_on_camera" });
        expect(result.retention).toEqual(retention);
    });

    it("surfaces the daemon's refusal of an unqualified deletion policy through the relay", async () => {
        transport = "django";
        fetchMock.mockImplementation(async () => jsonResponse({
            ok: false,
            error: "Camera deletion is disabled until storage qualification.",
        }));
        const { achernarApi } = await loadApi();
        await expect(achernarApi.capture({
            exposure: 2,
            retention_policy: "delete_after_verified_local_commit",
            retention_acknowledgement: { policy: "delete_after_verified_local_commit", policy_version: 1 },
        })).rejects.toThrow("Camera deletion is disabled until storage qualification.");
    });
});

describe("camera descriptor transport", () => {
    it.each(["daemon", "django"] as const)("reads the single camera capability snapshot over %s", async (mode) => {
        transport = mode;
        const descriptor = {
            connected: true,
            streaming: false,
            capabilities: { supports_dedicated_still: true, supports_stream: true },
            controls: [{ name: "iso", current: 4 }],
            formats: [{ index: 2, mode: "still" }],
        };
        fetchMock.mockImplementation(async (url: string) => {
            if (String(url).endsWith("/commands")) return jsonResponse(REGISTRY);
            return jsonResponse(mode === "django" ? { ok: true, payload: descriptor } : descriptor);
        });
        const { achernarApi } = await loadApi();
        expect(await achernarApi.getCameraDescriptor()).toEqual(descriptor);
        if (mode === "django") {
            const body = JSON.parse(String(callFor("/equipment/command/")![1]!.body));
            expect(body).toEqual({ action: "camera.descriptor", payload: {} });
        } else {
            const call = fetchMock.mock.calls.find((entry) => String(entry[0]).endsWith("/camera"));
            expect(call).toBeTruthy();
            expect(call![1]?.method ?? "GET").toBe("GET");
        }
        expect(callFor("camera/parameters")).toBeUndefined();
        expect(callFor("camera/formats")).toBeUndefined();
    });
});
