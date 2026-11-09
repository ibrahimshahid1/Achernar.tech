import { describe, expect, it } from "vitest";
import {
    displayBackgroundLevel,
    displayBackgroundNoise,
    mapSharedObservation,
} from "./sharedObservation";

// Shapes follow the daemon's serde output for SharedObservation and
// BackgroundResult (achernar-daemon/src/vision/{shared_observation,background_model}.rs).
const measuredBackground = {
    schema_version: 1,
    method: "legacy_global_clip",
    method_version: 1,
    library: null,
    library_version: null,
    requested_method: "legacy_global_clip",
    config: {},
    config_sha256: "abc",
    level_status: "measured",
    noise_status: "measured",
    reasons: [],
    backend_error: null,
    units: "adu",
    coordinate_frame: "analysis_pixel",
    width: 64,
    height: 64,
    noise_source: "empirical_global_clip",
    support: { total_pixels: 4096, usable_pixels: 4096, clipped_sample_pixels: 4000, masks: {}, saturation_evidence: "unknown" },
    level: 812.4,
    level_definition: "median of ...",
    rms: 9.5,
    rms_definition: "population standard deviation ...",
    mesh: null,
    plane: null,
    plane_unavailable_reason: "no_spatial_model",
    bright_pixel_significance: null,
    saturation_fraction: null,
};

function observation(extra: Record<string, unknown>) {
    return {
        timestamp: "2026-11-09T00:00:00Z",
        status: "Valid",
        rejection_reasons: [],
        hfr_px: 2.1,
        star_count: 12,
        star_count_used: 9,
        cloud_probability: 0.1,
        cloud_event: "None",
        trailing_severity: 0.05,
        observation_valid: true,
        ...extra,
    };
}

describe("mapSharedObservation background contract", () => {
    it("keeps a versioned, measured background", () => {
        const obs = mapSharedObservation(
            observation({ background: measuredBackground, background_adu: 812.4, background_sigma: 9.5 }),
        )!;
        expect(obs.backgroundProvenance).toBe("current");
        expect(obs.background?.level).toBe(812.4);
        // A current payload without detection provenance has no star count.
        expect(obs.starCount).toBeNull();
        const detected = mapSharedObservation(
            observation({ background: measuredBackground, detection: { method: "sep_extract" } }),
        )!;
        expect(detected.starCount).toBe(12);
        expect(detected.detectionMethod).toBe("sep_extract");
        expect(obs.background?.gradientXPerPx).toBeNull();
        expect(displayBackgroundLevel(obs)).toEqual({ text: "812.4 adu", note: null });
        expect(displayBackgroundNoise(obs)).toEqual({ text: "9.50 adu", note: null });
    });

    it("renders an unavailable background as a dash with its reason, never 0", () => {
        const unavailable = {
            ...measuredBackground,
            method: "sep_spatial_mesh",
            requested_method: "sep_spatial_mesh",
            level_status: "unavailable",
            noise_status: "unavailable",
            reasons: ["frame_smaller_than_mesh_grid"],
            level: null,
            rms: null,
        };
        const obs = mapSharedObservation(
            observation({
                status: "BackgroundUnavailable",
                hfr_px: null,
                background: unavailable,
                background_adu: null,
                background_sigma: null,
                saturation_fraction: null,
                trailing_severity: 0,
                trailing_unavailable_reason: "background_unavailable",
            }),
        )!;
        expect(obs.backgroundAdu).toBeNull();
        expect(obs.backgroundSigma).toBeNull();
        expect(obs.saturationFraction).toBeNull();
        expect(obs.trailingSeverity).toBeNull();
        // Detection did not run: no star count, not a count of 0.
        expect(obs.starCount).toBeNull();
        expect(displayBackgroundLevel(obs)).toEqual({ text: "—", note: "frame_smaller_than_mesh_grid" });
        expect(displayBackgroundNoise(obs).text).toBe("—");
    });

    it("shows a measured level with unresolved noise without inventing a sigma", () => {
        const plateau = {
            ...measuredBackground,
            noise_status: "unresolved",
            reasons: ["zero_empirical_scatter"],
            level: 100,
            // Even if a producer sent a number, the status decides.
            rms: 0,
        };
        const obs = mapSharedObservation(observation({ background: plateau, background_adu: 100 }))!;
        expect(obs.background?.rms).toBeNull();
        expect(displayBackgroundLevel(obs).text).toBe("100.0 adu");
        expect(displayBackgroundNoise(obs)).toEqual({ text: "—", note: "noise unresolved, zero_empirical_scatter" });
    });

    it("labels a configured legacy fallback", () => {
        const fallback = {
            ...measuredBackground,
            requested_method: "sep_spatial_mesh",
            reasons: ["fallback_used", "backend_failure"],
        };
        const obs = mapSharedObservation(observation({ background: fallback }))!;
        expect(displayBackgroundLevel(obs).note).toBe("fallback from sep_spatial_mesh");
    });

    it("reads a pre-contract payload as legacy-unverified, keeping its stored numbers", () => {
        const obs = mapSharedObservation(
            observation({ background_adu: 0, sky_background_adu: 0, background_sigma: 0, snr_proxy_median: 0 }),
        )!;
        expect(obs.backgroundProvenance).toBe("legacy_unverified");
        expect(obs.backgroundAdu).toBe(0);
        expect(obs.starCount).toBe(12);
        expect(displayBackgroundLevel(obs)).toEqual({ text: "0.0 ADU", note: "legacy record, unverified" });
    });

    it("treats a payload without any background as absent rather than zero", () => {
        const obs = mapSharedObservation(observation({}))!;
        expect(obs.backgroundProvenance).toBe("absent");
        expect(obs.backgroundAdu).toBeNull();
        expect(displayBackgroundLevel(obs)).toEqual({ text: "—", note: "no background estimate" });
    });

    it("rejects non-objects", () => {
        expect(mapSharedObservation(null)).toBeNull();
        expect(mapSharedObservation("x")).toBeNull();
    });
});
