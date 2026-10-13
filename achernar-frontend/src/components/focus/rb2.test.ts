import { describe, expect, it } from "vitest";
import { evaluateRb2 } from "./rb2";
import type { FocusSample } from "../../hooks/useFocusRun";

const s = (position: number, hfr: number, starCount: number): FocusSample => ({ position, hfr, starCount });

const stateOf = (samples: FocusSample[], runState: string, label: string) =>
    evaluateRb2(samples, runState).criteria.find((c) => c.label.includes(label))!.state;

/** A textbook clean V, well populated. */
const cleanV = [
    s(1000, 5.1, 20), s(1100, 3.8, 24), s(1200, 2.4, 30),
    s(1300, 1.6, 40),
    s(1400, 2.5, 28), s(1500, 3.9, 22), s(1600, 5.3, 18),
];

describe("RB-2 acceptance", () => {
    it("passes every criterion on a clean, well-populated V", () => {
        const v = evaluateRb2(cleanV, "converged");
        expect(v.criteria.map((c) => c.state)).toEqual(["pass", "pass", "pass", "pass"]);
        expect(v.minimumIndex).toBe(3);
        expect(v.provisional).toBe(false);
    });

    it("fails when the sweep never bracketed focus", () => {
        const edge = [s(1000, 1.6, 40), s(1100, 2.4, 30), s(1200, 3.8, 24), s(1300, 5.1, 20)];
        expect(stateOf(edge, "converged", "interior")).toBe("fail");
    });

    it("fails when the star field at the minimum is too thin", () => {
        const thin = [s(1000, 5.1, 20), s(1100, 3.0, 18), s(1200, 1.6, 12), s(1300, 3.1, 19), s(1400, 5.2, 20)];
        expect(stateOf(thin, "converged", "at minimum")).toBe("fail");
        expect(stateOf(thin, "converged", "both ends")).toBe("pass");
    });

    it("fails when either end of the sweep is too thin", () => {
        const thin = [s(1000, 5.1, 20), s(1100, 3.0, 22), s(1200, 1.6, 30), s(1300, 3.1, 15), s(1400, 5.2, 5)];
        expect(stateOf(thin, "converged", "both ends")).toBe("fail");
    });

    it("fails monotonicity on a reversal, and counts it", () => {
        const bumpy = [s(1000, 5.1, 20), s(1100, 3.0, 22), s(1200, 1.6, 30), s(1300, 3.1, 25), s(1400, 2.9, 24), s(1500, 5.2, 20)];
        const criterion = evaluateRb2(bumpy, "converged").criteria.find((c) => c.label.includes("Monotonic"))!;
        expect(criterion.state).toBe("fail");
        // The margin is reported rather than hidden behind an invented tolerance.
        expect(criterion.detail).toMatch(/1 reversal/);
    });

    it("reports pending, not failure, when there is too little data to judge", () => {
        const two = [s(1000, 5.1, 20), s(1100, 3.0, 22)];
        expect(stateOf(two, "running", "interior")).toBe("pending");
        expect(stateOf(two, "running", "Monotonic")).toBe("pending");
        expect(stateOf(two, "running", "at minimum")).toBe("pending");
    });

    it("handles an empty sweep without inventing a minimum", () => {
        const v = evaluateRb2([], "running");
        expect(v.minimumIndex).toBeNull();
        expect(v.provisional).toBe(true);
    });

    it("sorts samples by position, so arrival order does not matter", () => {
        const shuffled = [cleanV[4], cleanV[0], cleanV[3], cleanV[6], cleanV[1], cleanV[5], cleanV[2]];
        const v = evaluateRb2(shuffled, "converged");
        expect(v.criteria.map((c) => c.state)).toEqual(["pass", "pass", "pass", "pass"]);
        expect(v.sorted.map((x) => x.position)).toEqual([1000, 1100, 1200, 1300, 1400, 1500, 1600]);
    });

    it("marks a running sweep provisional even when it currently looks good", () => {
        const v = evaluateRb2(cleanV, "running");
        expect(v.provisional).toBe(true);
        expect(stateOf(cleanV, "running", "interior")).toBe("pass");
    });

    it("does not divide by zero on a single sample or a flat curve", () => {
        expect(() => evaluateRb2([s(1200, 2.0, 30)], "running")).not.toThrow();
        const flat = [s(1000, 2.0, 30), s(1100, 2.0, 30), s(1200, 2.0, 30)];
        expect(() => evaluateRb2(flat, "converged")).not.toThrow();
    });
});
