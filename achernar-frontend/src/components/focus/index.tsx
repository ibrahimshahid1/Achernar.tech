import { useState } from "react";
import { achernarApi } from "../../api/achernar";
import { useFocusRun, type FocusRunState, type FocusSample } from "../../hooks/useFocusRun";
import {
    evaluateRb2,
    ABORT_STARS_AT_CENTRE,
    MIN_STARS_AT_ENDS,
    MIN_STARS_AT_MINIMUM,
    type CriterionState,
} from "./rb2";

/**
 * Live focus-run view.
 *
 * The point of this is to tell a good sweep from a bad one *while you are
 * still outside and can redo it*. Everything here reads the daemon WebSocket
 * (`useFocusRun`), which already carried the full sample stream — only the
 * rendering was missing.
 *
 * Direct daemon only: `focus.*` is absent from the relay dispatcher and the
 * Django consumer drops every FOCUS_* event, so this cannot work through
 * achernar.com today. On a LAN it does not need to.
 */

const CHART_W = 640;
const CHART_H = 260;
const PAD = { top: 16, right: 16, bottom: 34, left: 46 };

function starTone(starCount: number): "good" | "warn" | "bad" {
    if (starCount >= MIN_STARS_AT_MINIMUM) return "good";
    if (starCount >= MIN_STARS_AT_ENDS) return "warn";
    return "bad";
}

/** Interactive focus curve chart (HFR vs position). */
export function FocusChart({ samples, runState }: { samples: FocusSample[]; runState: string }) {
    const { sorted, minimumIndex } = evaluateRb2(samples, runState);

    if (sorted.length === 0) {
        return (
            <div className="focus-chart-empty">
                No samples yet. The curve draws as the sweep collects.
            </div>
        );
    }

    const positions = sorted.map((s) => s.position);
    const hfrs = sorted.map((s) => s.hfr);
    const xMin = Math.min(...positions);
    const xMax = Math.max(...positions);
    const yMin = Math.min(...hfrs);
    const yMax = Math.max(...hfrs);

    // A single sample, or a perfectly flat curve, must not divide by zero.
    const xSpan = xMax - xMin || 1;
    const ySpan = yMax - yMin || 1;
    const plotW = CHART_W - PAD.left - PAD.right;
    const plotH = CHART_H - PAD.top - PAD.bottom;

    const x = (pos: number) => PAD.left + ((pos - xMin) / xSpan) * plotW;
    const y = (hfr: number) => PAD.top + plotH - ((hfr - yMin) / ySpan) * plotH;

    const path = sorted.map((s, i) => `${i === 0 ? "M" : "L"}${x(s.position)},${y(s.hfr)}`).join(" ");

    return (
        <svg
            className="focus-chart"
            viewBox={`0 0 ${CHART_W} ${CHART_H}`}
            role="img"
            aria-label={`Focus curve, ${sorted.length} samples, HFR against focuser position`}
        >
            {/* axes */}
            <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + plotH} className="focus-axis" />
            <line
                x1={PAD.left}
                y1={PAD.top + plotH}
                x2={PAD.left + plotW}
                y2={PAD.top + plotH}
                className="focus-axis"
            />

            <text x={PAD.left - 8} y={PAD.top + 4} className="focus-tick" textAnchor="end">
                {yMax.toFixed(2)}
            </text>
            <text x={PAD.left - 8} y={PAD.top + plotH} className="focus-tick" textAnchor="end">
                {yMin.toFixed(2)}
            </text>
            <text x={PAD.left} y={CHART_H - 10} className="focus-tick" textAnchor="start">
                {xMin}
            </text>
            <text x={PAD.left + plotW} y={CHART_H - 10} className="focus-tick" textAnchor="end">
                {xMax}
            </text>
            <text x={PAD.left + plotW / 2} y={CHART_H - 10} className="focus-axis-label" textAnchor="middle">
                focuser position
            </text>

            <path d={path} className="focus-curve" fill="none" />

            {minimumIndex !== null && (
                <line
                    x1={x(sorted[minimumIndex].position)}
                    y1={PAD.top}
                    x2={x(sorted[minimumIndex].position)}
                    y2={PAD.top + plotH}
                    className="focus-minimum-line"
                />
            )}

            {sorted.map((s, i) => (
                <circle
                    key={`${s.position}-${i}`}
                    cx={x(s.position)}
                    cy={y(s.hfr)}
                    r={i === minimumIndex ? 6 : 4}
                    className={`focus-point tone-${starTone(s.starCount)}${i === minimumIndex ? " is-minimum" : ""}`}
                >
                    <title>{`position ${s.position} · HFR ${s.hfr.toFixed(3)} · ${s.starCount} stars`}</title>
                </circle>
            ))}
        </svg>
    );
}

/** RB-2 acceptance checklist for the current run. */
export function FocusAcceptance({ samples, runState }: { samples: FocusSample[]; runState: string }) {
    const { criteria, provisional } = evaluateRb2(samples, runState);

    // Abort criterion, not an acceptance one: "fewer than 8 stars at sweep
    // centre" means stop and redo rather than keep and flag.
    const starved = samples.filter((s) => s.starCount < ABORT_STARS_AT_CENTRE).length;

    return (
        <div className="focus-acceptance">
            <div className="focus-acceptance-head">
                RB-2 acceptance
                {provisional && <span className="focus-provisional">provisional — sweep in progress</span>}
            </div>
            <ul className="focus-criteria">
                {criteria.map((c) => (
                    <li key={c.label} className={`focus-criterion state-${c.state}`}>
                        <span className="focus-criterion-mark">{markFor(c.state)}</span>
                        <span className="focus-criterion-label">{c.label}</span>
                        <span className="focus-criterion-detail">{c.detail}</span>
                    </li>
                ))}
            </ul>
            {starved > 0 && (
                <div className="focus-abort-warning">
                    {starved} sample{starved === 1 ? "" : "s"} below {ABORT_STARS_AT_CENTRE} stars — RB-2 aborts
                    rather than keeps a run this thin. Prefer dropping it to cleaning it.
                </div>
            )}
        </div>
    );
}

function markFor(state: CriterionState) {
    if (state === "pass") return "PASS";
    if (state === "fail") return "FAIL";
    return "--";
}

/** Focus subsystem status display. */
export function FocusStatus({ focus }: { focus: FocusRunState }) {
    return (
        <div className="focus-status">
            <span className={`focus-state state-${focus.state}`}>{focus.state}</span>
            <span className="focus-status-meta">{focus.samples.length} samples</span>
            {/* Null until the run converges — not 0, which would read as
                perfect focus at step zero. */}
            <span className="focus-status-meta">
                best {focus.bestPosition ?? "--"}
                {focus.bestHfr != null && ` @ HFR ${focus.bestHfr.toFixed(3)}`}
            </span>
            {focus.error && <span className="focus-error">{focus.error}</span>}
        </div>
    );
}

/** Card showing details of a single focus run. */
export function FocusRunCard({ runId }: { runId: string }) {
    return (
        <div className="focus-run-card" id={`focus-run-${runId}`}>
            <span className="focus-run-id">run {runId.slice(0, 8)}</span>
        </div>
    );
}

/** Focus action buttons (start/abort autofocus). */
export function FocusActions({ running }: { running: boolean }) {
    const [busy, setBusy] = useState(false);
    const [message, setMessage] = useState<string | null>(null);

    // These used to render with no onClick at all — two buttons that did
    // nothing, on every transport.
    const start = async () => {
        setBusy(true);
        setMessage(null);
        try {
            const run = await achernarApi.startFocusRun({ method: "classical" });
            setMessage(`Run ${run.run_id.slice(0, 8)} started.`);
        } catch (error) {
            setMessage(error instanceof Error ? error.message : "Could not start the run.");
        } finally {
            setBusy(false);
        }
    };

    const abort = async () => {
        setBusy(true);
        setMessage(null);
        try {
            await achernarApi.abortFocusRun();
            setMessage("Abort sent.");
        } catch (error) {
            setMessage(error instanceof Error ? error.message : "Could not abort the run.");
        } finally {
            setBusy(false);
        }
    };

    return (
        <div className="focus-actions" id="focus-actions">
            <button
                type="button"
                id="btn-start-autofocus"
                className="cam-btn primary"
                onClick={start}
                disabled={busy || running}
            >
                Start autofocus
            </button>
            <button
                type="button"
                id="btn-abort-autofocus"
                className="cam-btn"
                onClick={abort}
                disabled={busy || !running}
            >
                Abort
            </button>
            {message && <span className="focus-action-message">{message}</span>}
        </div>
    );
}

/** The whole focus panel: status, curve, acceptance, controls. */
export default function FocusPanel() {
    const focus = useFocusRun();
    const running = focus.state === "running";

    return (
        <div className="cam-panel focus-panel">
            <div className="cam-head">
                <div className="cam-title">Focus Run</div>
                <div className="cam-transport">Direct daemon</div>
            </div>
            <FocusStatus focus={focus} />
            <FocusChart samples={focus.samples} runState={focus.state} />
            <FocusAcceptance samples={focus.samples} runState={focus.state} />
            <FocusActions running={running} />
        </div>
    );
}
