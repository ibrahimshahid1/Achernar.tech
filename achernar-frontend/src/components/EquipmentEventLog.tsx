import type { EquipmentEvent } from "../types/achernar";
import "./equipmentPanels.css";

function payloadText(payload: unknown): string {
    if (payload == null) return "";
    try {
        const text = JSON.stringify(payload);
        return text.length > 160 ? `${text.slice(0, 157)}…` : text;
    } catch {
        return "";
    }
}

/**
 * The equipment journal tail (§4.2), newest first. Guide steps are hidden by
 * the caller unless asked for: PHD2 sends one per guide exposure.
 */
export default function EquipmentEventLog({ events, error, showGuideSteps, onShowGuideSteps }: {
    events: EquipmentEvent[];
    error?: string | null;
    showGuideSteps: boolean;
    onShowGuideSteps: (value: boolean) => void;
}) {
    return <div className="eq-panel" aria-label="Equipment events">
        <div className="eq-toolbar">
            <label className="eq-inline"><input type="checkbox" checked={showGuideSteps} onChange={(event) => onShowGuideSteps(event.target.checked)} /> show guide steps</label>
            {error && <span className="eq-hint tone-bad">{error}</span>}
        </div>
        {events.length === 0 ? <p className="eq-empty">No equipment events yet.</p> : <table className="eq-table eq-events">
            <thead><tr><th>Seq</th><th>Time</th><th>Device</th><th>Event</th><th>Detail</th></tr></thead>
            <tbody>{events.map((event) => <tr key={`${event.seq}-${event.event_id}`}>
                <td>{event.seq}</td>
                <td>{new Date(event.at_utc).toLocaleTimeString()}</td>
                <td>{event.device_kind?.replace(/_/g, " ") ?? "equipment"}</td>
                <td>{event.kind.replace(/_/g, " ")}</td>
                <td className="eq-detail">{payloadText(event.payload)}</td>
            </tr>)}</tbody>
        </table>}
    </div>;
}
