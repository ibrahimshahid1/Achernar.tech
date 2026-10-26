import { ageText, qualificationLabel, qualificationTone, weatherRows } from "../../lib/equipmentView";
import type { WeatherBlock } from "../../types/achernar";
import "../equipmentPanels.css";

/**
 * ObservingConditions readings with each field's state and age (§2.13,
 * R18). NaN, ValueNotSet and NotImplemented arrive as null and render as
 * unknown, never as a value.
 */
export default function WeatherPanel({ weather, online }: { weather: WeatherBlock | null | undefined; online: boolean }) {
    if (!online || weather?.connected !== true) return <p className="eq-empty">No weather device connected.</p>;
    const rows = weatherRows(weather);
    return <div className="eq-panel" aria-label="Weather">
        <div className="eq-toolbar">
            <span className="eq-state">{weather.name ?? "Weather"}</span>
            {weather.qualification && <span className={`qual-badge tone-${qualificationTone(weather.qualification)}`}>{qualificationLabel(weather.qualification)}</span>}
            <span className="eq-hint">Published {ageText(weather.age_ms)} ago</span>
        </div>
        <table className="eq-table">
            <thead><tr><th>Field</th><th>Value</th><th>State</th><th>Age</th></tr></thead>
            <tbody>{rows.map((row) => <tr key={row.key}>
                <td>{row.label}</td>
                <td className={`tone-${row.tone}`}>{row.value}</td>
                <td>{row.state}</td>
                <td>{row.age}</td>
            </tr>)}</tbody>
        </table>
    </div>;
}
