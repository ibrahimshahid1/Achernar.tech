import { useState, useCallback } from "react";
import { useDaemonWS, type DaemonWsMessage } from "./useDaemonWS";

export interface FocusSample {
    position: number;
    hfr: number;
    starCount: number;
}

export interface FocusRunState {
    runId: string | null;
    state: string;
    samples: FocusSample[];
    bestPosition: number | null;
    bestHfr: number | null;
    predictions: Array<{ model: string; offset: number; confidence: number }>;
    error: string | null;
}

/**
 * Subscribe to all FOCUS_* WebSocket events.
 */
export function useFocusRun() {
    const [focusState, setFocusState] = useState<FocusRunState>({
        runId: null,
        state: "idle",
        samples: [],
        bestPosition: null,
        bestHfr: null,
        predictions: [],
        error: null,
    });

    const handlers = {
        FOCUS_RUN_STARTED: (data: DaemonWsMessage) => {
            setFocusState({
                runId: data.run_id as string,
                state: "running",
                samples: [],
                bestPosition: null,
                bestHfr: null,
                predictions: [],
                error: null,
            });
        },
        FOCUS_RUN_STATE_CHANGED: (data: DaemonWsMessage) => {
            setFocusState((prev) => ({ ...prev, state: data.new_state as string }));
        },
        FOCUS_SAMPLE_CAPTURED: (data: DaemonWsMessage) => {
            setFocusState((prev) => ({
                ...prev,
                samples: [
                    ...prev.samples,
                    {
                        position: data.position as number,
                        hfr: data.hfr as number,
                        starCount: data.star_count as number,
                    },
                ],
            }));
        },
        FOCUS_PREDICTION_MADE: (data: DaemonWsMessage) => {
            setFocusState((prev) => ({
                ...prev,
                predictions: [
                    ...prev.predictions,
                    {
                        model: data.model as string,
                        offset: data.predicted_offset as number,
                        confidence: data.confidence as number,
                    },
                ],
            }));
        },
        FOCUS_RUN_CONVERGED: (data: DaemonWsMessage) => {
            setFocusState((prev) => ({
                ...prev,
                state: "converged",
                bestPosition: data.best_position as number,
                bestHfr: data.best_hfr as number,
            }));
        },
        FOCUS_RUN_FAILED: (data: DaemonWsMessage) => {
            setFocusState((prev) => ({
                ...prev,
                state: "failed",
                error: data.error as string,
            }));
        },
    };

    useDaemonWS(handlers);

    return focusState;
}
