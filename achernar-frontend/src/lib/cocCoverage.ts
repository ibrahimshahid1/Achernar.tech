import type { CocSample } from "../api/coc";

const fraction = (value: number | null): number | null =>
    typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1 ? value : null;

/** Coverage uses independent-ROI image pixels; contamination is excluded, never clear sky. */
export function cocCoverage(summary: CocSample["summary"]) {
    const cloudArea = fraction(summary.cloud_area_fraction_of_roi);
    const excludedArea = fraction(summary.contamination_area_fraction_of_roi);
    const classifiedCloud = fraction(summary.cloud_fraction_of_classified_sky);
    // Keep abstention authoritative, and fail closed on inconsistent/invalid
    // summaries instead of displaying an apparently precise coverage number.
    const consistent = cloudArea !== null && excludedArea !== null && excludedArea < 1
        && cloudArea + excludedArea <= 1 + 1e-9 && classifiedCloud !== null
        && Math.abs(classifiedCloud - cloudArea / (1 - excludedArea)) <= 1e-6;
    return {
        cloudFraction: summary.status === "research_diagnostic" && summary.reason === null && consistent
            ? classifiedCloud : null,
        cloudAreaFraction: cloudArea,
        excludedAreaFraction: excludedArea,
    };
}

export const coveragePercent = (value: number | null): string =>
    value === null ? "Unavailable" : `${(100 * value).toFixed(1)}%`;
