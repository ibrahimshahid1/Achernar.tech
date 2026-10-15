import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { CocSample } from "../api/coc";
import { cocCoverage } from "../lib/cocCoverage";
import CocCoverageSummary from "./CocCoverageSummary";

const summary = (overrides: Partial<CocSample["summary"]> = {}): CocSample["summary"] => ({
    status: "research_diagnostic", reason: null,
    cloud_area_fraction_of_roi: 0.5, contamination_area_fraction_of_roi: 0.2,
    cloud_fraction_of_classified_sky: 0.625,
    probability_calibrated: false, runtime_eligible: false, ...overrides,
});
const rendered = (value = summary()) => renderToStaticMarkup(createElement(CocCoverageSummary, { summary: value }));

describe("COC image-area cloud coverage", () => {
    it("labels classified-sky coverage separately from whole-region cloud and excluded area", () => {
        const html = rendered();
        expect(html).toContain("Cloud cover: 62.5% of classified sky");
        expect(html).toContain("Predicted cloud area: 50.0% of selected region");
        expect(html).toContain("Contamination / excluded area: 20.0% of selected region");
        expect(html).toContain("is not counted as clear sky");
        expect(html).toContain("without whole-sky or solid-angle calibration");
    });
    it("preserves valid zero and complete cloud coverage when usable sky exists", () => {
        expect(rendered(summary({ cloud_area_fraction_of_roi: 0, cloud_fraction_of_classified_sky: 0 })))
            .toContain("Cloud cover: 0.0% of classified sky");
        expect(rendered(summary({ cloud_area_fraction_of_roi: 0.8, cloud_fraction_of_classified_sky: 1 })))
            .toContain("Cloud cover: 100.0% of classified sky");
    });
    it.each(["valid_region_required", "empty_valid_region", "no_classified_sky", "contamination_exceeds_diagnostic_threshold", "uninformative_rgb_input"])
        ("shows unavailable on %s instead of turning an abstention into clear sky", (reason) => {
            const value = summary({ status: "abstained", reason, cloud_fraction_of_classified_sky: null });
            expect(rendered(value)).toContain("<strong>Cloud cover: Unavailable</strong>");
            // Even a malformed stale value must not override the authoritative abstention.
            expect(cocCoverage({ ...value, cloud_fraction_of_classified_sky: 0.625 }).cloudFraction).toBeNull();
        });
    it("rejects nonfinite, impossible and inconsistent denominator values", () => {
        for (const changes of [
            { cloud_fraction_of_classified_sky: NaN },
            { cloud_fraction_of_classified_sky: 1.1 },
            { contamination_area_fraction_of_roi: -0.1 },
            { contamination_area_fraction_of_roi: 1 },
            { cloud_area_fraction_of_roi: 0.9 },
            { cloud_fraction_of_classified_sky: 0.5 },
        ]) expect(cocCoverage(summary(changes)).cloudFraction).toBeNull();
    });
});
