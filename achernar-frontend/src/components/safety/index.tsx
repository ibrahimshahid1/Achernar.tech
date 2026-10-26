import type { EstopReport, SafetyStatus } from "../../api/achernar";
import {
    ageText,
    busyLabels,
    estopSummary,
    operatorClears,
    qualificationLabel,
    qualificationTone,
    resumeBlockTitle,
    safetyView,
} from "../../lib/equipmentView";
import type { BusySet, ResumeBlock, SafetyMonitorBlock } from "../../types/achernar";
import "../equipmentPanels.css";

/**
 * Emergency stop (§2.9): every device's abort path in parallel, degraded and
 * lost devices included, then the `estop` resume block. It is not a park.
 *
 * The button is enabled whenever the daemon is reachable. It never waits for
 * another command to finish and never asks for confirmation: a stop that
 * needs a second click is a slower stop.
 */
export function EmergencyStopButton({ online, pending, abortExposure, onAbortExposureChange, onStop, report, error }: {
    online: boolean;
    pending: boolean;
    abortExposure: boolean;
    onAbortExposureChange: (value: boolean) => void;
    onStop: () => void;
    report?: EstopReport | null;
    error?: string | null;
}) {
    return <div className="eq-estop">
        <button type="button" className="btn eq-estop-button" disabled={!online} onClick={onStop}
            aria-label="Emergency stop: abort every device">{pending ? "STOPPING…" : "E-STOP"}</button>
        <label className="eq-inline"><input type="checkbox" checked={abortExposure} disabled={!online} onChange={(event) => onAbortExposureChange(event.target.checked)} /> also abort the exposure</label>
        {report && <small className={`eq-hint tone-${report.status === "stopped" ? "good" : "bad"}`} role="status">{estopSummary(report)}</small>}
        {error && <small className="eq-hint tone-bad" role="alert">{error}</small>}
    </div>;
}

/**
 * Active resume blocks (H22): automatic captures and sequencing are refused
 * until each is cleared. Operator-clearable blocks get a clear button (the
 * daemon audits it); `safety_*` blocks clear only on a confirmed safe reading.
 */
export function ResumeBlockBanner({ blocks, online, busy, onClear }: {
    blocks: ResumeBlock[] | null | undefined;
    online: boolean;
    busy: boolean;
    onClear: (block: ResumeBlock) => void;
}) {
    if (!blocks?.length) return null;
    return <div className="eq-resume-banner" role="alert" aria-label="Resume blocks">
        <strong>{blocks.length === 1 ? "1 resume block" : `${blocks.length} resume blocks`}: automatic captures are refused.</strong>
        <ul>
            {blocks.map((block) => <li key={block.id}>
                <span className="eq-resume-title">{resumeBlockTitle(block)}</span>
                <span className="eq-hint"> · {new Date(block.raised_at).toLocaleTimeString()}{block.reason ? ` · ${block.reason}` : ""}</span>
                {operatorClears(block)
                    ? <button type="button" className="btn" disabled={!online || busy} onClick={() => onClear(block)}>Clear</button>
                    : <span className="eq-hint"> · clears on a confirmed safe reading</span>}
            </li>)}
        </ul>
    </div>;
}

/** Busy flags (H20), so the operator sees a critical transaction in progress. */
export function BusyFlags({ busy }: { busy: BusySet | null | undefined }) {
    const labels = busyLabels(busy);
    if (!labels.length) return null;
    return <span className="eq-busy" role="status">Busy: {labels.join(", ")}</span>;
}

/**
 * The combined safety state of every monitor, and this monitor's reading.
 * Unknown or stale is shown as a warning, never as safe (§2.8).
 */
export function SafetyPanel({ summary, monitor, online }: {
    summary?: SafetyStatus | null;
    monitor: SafetyMonitorBlock | null | undefined;
    online: boolean;
}) {
    if (!online) return <p className="eq-empty">Safety state unavailable.</p>;
    const view = safetyView(summary?.state, monitor ?? summary?.monitor ?? null);
    const reading = monitor ?? summary?.monitor ?? null;
    return <div className="eq-panel" aria-label="Safety">
        <div className="eq-toolbar">
            <span className={`eq-safety tone-${view.tone}`}>{view.label}</span>
            {reading?.qualification && <span className={`qual-badge tone-${qualificationTone(reading.qualification)}`}>{qualificationLabel(reading.qualification)}</span>}
            <span className="eq-hint">{view.detail}</span>
        </div>
        {reading && <dl className="eq-grid">
            <div className="eq-row"><dt>Monitor</dt><dd>{reading.name ?? "Safety monitor"}</dd></div>
            <div className="eq-row"><dt>Level</dt><dd>{reading.level ?? "unknown"}</dd></div>
            <div className="eq-row"><dt>Reading state</dt><dd>{reading.reading?.state ?? "unknown"}</dd></div>
            <div className="eq-row"><dt>Reading age</dt><dd>{ageText(reading.age_ms)}</dd></div>
        </dl>}
        {summary && <p className="eq-hint">On unsafe: {summary.on_unsafe ?? "not reported"}{summary.policy_running ? " · the unsafe policy is running" : ""}.</p>}
    </div>;
}
