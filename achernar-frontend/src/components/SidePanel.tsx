import { useTelemetryStore } from "../stores/telemetryStore";

export default function SidePanel() {
    const connected = useTelemetryStore((s) => s.connected);
    const equipment = useTelemetryStore((s) => s.equipment);
    const guiding = useTelemetryStore((s) => s.guiding);

    const camTemp = equipment?.camera?.temperature;
    const camStatus = equipment?.camera?.status;
    const mountRa = equipment?.mount?.ra;
    const mountDec = equipment?.mount?.dec;

    return (
        <aside className="side-panel">
            <div className="side-heading">
                <div className="side-title">Run Context</div>
                <div className="side-tag">
                    Telemetry: {connected ? "LIVE" : "OFFLINE"}
                </div>
            </div>

            <div className="metric-row">
                <span>Mount (RA/Dec)</span>
                <strong>
                    {mountRa?.toFixed?.(2) ?? "--"} / {mountDec?.toFixed?.(2) ?? "--"}
                </strong>
            </div>

            <div className="metric-row">
                <span>Camera</span>
                <strong>{camStatus ?? "--"}</strong>
            </div>

            <div className="metric-row">
                <span>Sensor Temp</span>
                <strong>{typeof camTemp === "number" ? `${camTemp.toFixed(1)}°C` : "--"}</strong>
            </div>

            <div className="metric-row">
                <span>Guiding RMS</span>
                <strong>{guiding?.rms_total?.toFixed?.(2) ?? "--"}"</strong>
            </div>

            <div className="side-actions">
                <button className="btn primary">Validate Night Configuration</button>
                <button className="btn">Export Instrument Matrix</button>
            </div>

            <div className="side-caption">
                The UI is "daemon-first": control + telemetry flow from the edge, while Django handles
                accounts, collaboration, and archives.
            </div>
        </aside>
    );
}
