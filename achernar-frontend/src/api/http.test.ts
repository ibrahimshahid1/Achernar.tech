import { afterEach, describe, expect, it, vi } from "vitest";
import { HttpError, httpJson } from "./http";

afterEach(() => vi.unstubAllGlobals());

describe("camera error messages", () => {
    it.each([
        { error: "Could not set ISO: camera busy" },
        { error: { code: "camera_hardware_error", message: "Could not set ISO: camera busy" } },
        { detail: "Could not set ISO: camera busy" },
    ])("preserves the hardware failure from either transport", async (body) => {
        vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(body), { status: 502 })));
        await expect(httpJson("/camera/parameters")).rejects.toMatchObject({
            message: "Could not set ISO: camera busy",
            status: 502,
            bodyText: JSON.stringify(body),
        });
    });

    it("keeps the HTTP status when a proxy returns a non-JSON failure", async () => {
        vi.stubGlobal("fetch", vi.fn(async () => new Response("Unavailable", { status: 503 })));
        await expect(httpJson("/camera/parameters")).rejects.toBeInstanceOf(HttpError);
        await expect(httpJson("/camera/parameters")).rejects.toThrow("HTTP 503 for /camera/parameters");
    });
});
