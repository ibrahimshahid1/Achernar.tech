import type { FocusSample } from "../../hooks/useFocusRun";

/**
 * RB-2 acceptance criteria, as written in docs/Achernar-Technical-Audit.md:
 *
 *   "Acceptance is a clean V or U, monotonic on each side, minimum interior
 *    to the sweep, >=15 stars at the minimum and >=8 at both ends."
 *
 * Implemented exactly as stated, with no invented tolerance. Monotonicity in
 * particular is checked strictly — seeing noise may well trip it on real data,
 * and if a tolerance is warranted that is a decision to make against real
 * sweeps, not one to bake in here from a guess. The violation count is
 * reported so the margin is visible rather than hidden behind a threshold.
 */

export const MIN_STARS_AT_MINIMUM = 15;
export const MIN_STARS_AT_ENDS = 8;

/** Abort threshold: "fewer than 8 stars at sweep centre". */
export const ABORT_STARS_AT_CENTRE = 8;

/** `pending` is not a soft failure — it means not yet determinable. */
export type CriterionState = "pass" | "fail" | "pending";

export interface Criterion {
    label: string;
    state: CriterionState;
    detail: string;
}

export interface Rb2Verdict {
    criteria: Criterion[];
    /** True while the sweep is still collecting; verdicts can still change. */
    provisional: boolean;
    /** Index into the position-sorted samples, or null if none yet. */
    minimumIndex: number | null;
    sorted: FocusSample[];
}

const PENDING_TOO_FEW = "Not enough samples yet";

export function evaluateRb2(samples: FocusSample[], runState: string): Rb2Verdict {
    const sorted = [...samples].sort((a, b) => a.position - b.position);
    const n = sorted.length;
    const provisional = runState === "running";

    if (n < 3) {
        return {
            sorted,
            minimumIndex: n > 0 ? argMinHfr(sorted) : null,
            provisional,
            criteria: [
                { label: "Minimum interior to sweep", state: "pending", detail: PENDING_TOO_FEW },
                { label: "Monotonic on each side", state: "pending", detail: PENDING_TOO_FEW },
                { label: `>=${MIN_STARS_AT_MINIMUM} stars at minimum`, state: "pending", detail: PENDING_TOO_FEW },
                { label: `>=${MIN_STARS_AT_ENDS} stars at both ends`, state: "pending", detail: PENDING_TOO_FEW },
            ],
        };
    }

    const minIdx = argMinHfr(sorted);
    const interior = minIdx > 0 && minIdx < n - 1;

    const leftViolations = countViolations(sorted, 0, minIdx, "decreasing");
    const rightViolations = countViolations(sorted, minIdx, n - 1, "increasing");
    const violations = leftViolations + rightViolations;

    const starsAtMin = sorted[minIdx].starCount;
    const firstStars = sorted[0].starCount;
    const lastStars = sorted[n - 1].starCount;
    const worstEnd = Math.min(firstStars, lastStars);

    return {
        sorted,
        minimumIndex: minIdx,
        provisional,
        criteria: [
            {
                label: "Minimum interior to sweep",
                state: interior ? "pass" : "fail",
                detail: interior
                    ? `Minimum at position ${sorted[minIdx].position}, ${minIdx} from the low end`
                    : `Minimum sits at the ${minIdx === 0 ? "low" : "high"} end — the sweep did not bracket focus`,
            },
            {
                label: "Monotonic on each side",
                state: violations === 0 ? "pass" : "fail",
                detail: violations === 0
                    ? "HFR falls to the minimum and rises after it"
                    : `${violations} reversal${violations === 1 ? "" : "s"} (${leftViolations} left, ${rightViolations} right)`,
            },
            {
                label: `>=${MIN_STARS_AT_MINIMUM} stars at minimum`,
                state: starsAtMin >= MIN_STARS_AT_MINIMUM ? "pass" : "fail",
                detail: `${starsAtMin} stars at the minimum`,
            },
            {
                label: `>=${MIN_STARS_AT_ENDS} stars at both ends`,
                state: worstEnd >= MIN_STARS_AT_ENDS ? "pass" : "fail",
                detail: `${firstStars} at low end, ${lastStars} at high end`,
            },
        ],
    };
}

function argMinHfr(sorted: FocusSample[]): number {
    let best = 0;
    for (let i = 1; i < sorted.length; i += 1) {
        if (sorted[i].hfr < sorted[best].hfr) best = i;
    }
    return best;
}

function countViolations(
    sorted: FocusSample[],
    from: number,
    to: number,
    direction: "decreasing" | "increasing",
): number {
    let violations = 0;
    for (let i = from; i < to; i += 1) {
        const delta = sorted[i + 1].hfr - sorted[i].hfr;
        if (direction === "decreasing" ? delta > 0 : delta < 0) violations += 1;
    }
    return violations;
}
