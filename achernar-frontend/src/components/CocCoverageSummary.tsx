import type { CocSample } from "../api/coc";
import { cocCoverage, coveragePercent } from "../lib/cocCoverage";

export default function CocCoverageSummary({ summary }: { summary: CocSample["summary"] }) {
    const coverage = cocCoverage(summary);
    return <div aria-label="Cloud cover summary">
        <p><strong>Cloud cover: {coveragePercent(coverage.cloudFraction)}{coverage.cloudFraction !== null ? " of classified sky" : ""}</strong></p>
        <p>Predicted cloud area: {coveragePercent(coverage.cloudAreaFraction)} of selected region · Contamination / excluded area: {coveragePercent(coverage.excludedAreaFraction)} of selected region</p>
        <p>Cloud cover divides cloud pixels by sky + cloud pixels inside your selected region. Contamination is excluded and is not counted as clear sky.</p>
        <p>These are image-area estimates for the selected camera view, without whole-sky or solid-angle calibration. Scores are uncalibrated; this is not a transparency measurement or weather safety assessment.</p>
    </div>;
}
