import React from "react";

/** Focus subsystem status display. */
export function FocusStatus() {
    return <div id="focus-status">Focus Status — stub</div>;
}

/** Card showing details of a single focus run. */
export function FocusRunCard({ runId }: { runId: string }) {
    return <div id={`focus-run-${runId}`}>Focus Run {runId} — stub</div>;
}

/** Interactive focus curve chart (HFR vs position). */
export function FocusChart() {
    return <div id="focus-chart">Focus Chart — stub</div>;
}

/** Focus action buttons (start/abort autofocus). */
export function FocusActions() {
    return (
        <div id="focus-actions">
            <button id="btn-start-autofocus">Start Autofocus</button>
            <button id="btn-abort-autofocus">Abort</button>
        </div>
    );
}
