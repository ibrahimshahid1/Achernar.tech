// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EmergencyStopButton } from "@/components/safety";

/**
 * The first component test that runs in a DOM. It opts into jsdom with the
 * docblock above, imports through the "@/" alias and renders with Testing
 * Library, so it also guards that wiring. Static-markup checks of the safety
 * panels stay in equipmentPanels.test.ts, in the default node environment.
 */
afterEach(cleanup);

const stopButton = () => screen.getByRole("button", { name: "Emergency stop: abort every device" });

describe("EmergencyStopButton in a DOM", () => {
    it("stops on a single click, with no confirmation step", () => {
        const onStop = vi.fn();
        render(<EmergencyStopButton reachable pending={false} abortExposure={false}
            onAbortExposureChange={() => undefined} onStop={onStop} />);
        fireEvent.click(stopButton());
        expect(onStop).toHaveBeenCalledTimes(1);
    });

    it("ignores clicks while no transport is configured", () => {
        const onStop = vi.fn();
        render(<EmergencyStopButton reachable={false} pending={false} abortExposure={false}
            onAbortExposureChange={() => undefined} onStop={onStop} />);
        expect(stopButton()).toHaveProperty("disabled", true);
        fireEvent.click(stopButton());
        expect(onStop).not.toHaveBeenCalled();
    });
});
