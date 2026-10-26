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
 * The equipment journal tail (§4.2, `GET equipment/events`), newest first.
 *
 * Guide steps are not in this journal: the daemon keeps them in a separate
 * ring and sends them only on the EQUIPMENT_EVENT stream, so this log never
 * shows them and offers no switch for them.
 */
export default function EquipmentEventLog({ events, error }: {
    events: EquipmentEvent[];
    error?: string | null;
}) {
    return <div className="eq-panel" aria-label="Equipment events">
        <div className="eq-toolbar">
            <span className="eq-hint">Journal events, newest first. Guide steps are not journaled and are not listed here.</span>
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
