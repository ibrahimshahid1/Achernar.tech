import type { DragEvent } from "react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import EquipmentCard from "../components/EquipmentCard";
import { useTelemetrySocket } from "../hooks/useTelemetrySocket";
import { useConnectedEquipment } from "../hooks/useEquipment";
import { useRunEquipmentSelection } from "../hooks/useRunEquipmentSelection";
import { getHardwareTransport } from "../lib/runtime";
import type { EquipmentCardModel } from "../types/achernar";

const QUEUE_ITEMS = [
    { target: "M42 Core", recipe: "L / 24 x 120 s" },
    { target: "Horsehead", recipe: "Ha / 18 x 300 s" },
    { target: "Flats", recipe: "Auto panel / dusk" },
];

const LOG_LINES = [
    { time: "20:24:12", text: "Django relay ready for daemon status" },
    { time: "20:25:51", text: "Camera stream control available" },
    { time: "20:26:20", text: "Run context waiting for equipment selection" },
];

function statusText(card: EquipmentCardModel) {
    if (card.status === "offline") {
        return "offline";
    }
    return card.tags[0] ?? "ready";
}

export default function Dashboard() {
    useTelemetrySocket();

    const navigate = useNavigate();
    const hardwareTransport = getHardwareTransport();
    const [dropActive, setDropActive] = useState(false);
    const { data: equipment = [], isLoading, error } = useConnectedEquipment();
    const {
        selectedIds,
        selectedCards,
        addEquipment,
        removeEquipment,
        clearSelection,
    } = useRunEquipmentSelection(equipment);

    const connectedCount = useMemo(
        () => equipment.filter((card) => card.status !== "offline").length,
        [equipment],
    );

    const camera = equipment.find((card) => card.type === "camera");
    const mount = equipment.find((card) => card.type === "mount");
    const focuser = equipment.find((card) => card.type === "focuser");
    const filterWheel = equipment.find((card) => card.type === "filter-wheel");

    const handleDragStart = (card: EquipmentCardModel, event: DragEvent<HTMLElement>) => {
        event.dataTransfer.setData("text/plain", card.id);
        event.dataTransfer.effectAllowed = "copy";
    };

    const handleDrop = (event: DragEvent<HTMLElement>) => {
        event.preventDefault();
        setDropActive(false);
        addEquipment(event.dataTransfer.getData("text/plain"));
    };

    const startRun = () => {
        if (selectedCards.length === 0) {
            return;
        }
        navigate("/run");
    };

    return (
        <main className="dashboard-layout">
            <section className="dashboard-main panel">
                <div className="dashboard-hero">
                    <div>
                        <div className="panel-title">Observatory Dashboard</div>
                        <h1>Connected equipment and run readiness</h1>
                        <div className="panel-subtitle">
                            Browser mode reads hardware through Django; desktop mode can talk to the daemon directly.
                        </div>
                    </div>
                    <div className="night-summary">
                        <strong>{connectedCount > 0 ? "Ready" : "Waiting"}</strong>
                        <span>{connectedCount} devices online</span>
                        <small>{hardwareTransport === "django" ? "Django relay" : "Direct daemon"}</small>
                    </div>
                </div>

                <section className="equipment-deck" aria-labelledby="equipmentTitle">
                    <div className="section-row">
                        <div>
                            <h2 id="equipmentTitle">Equipment</h2>
                            <p>Live daemon status, shaped for the current run context.</p>
                        </div>
                        <span className="source-chip">
                            {hardwareTransport === "django" ? "Django to daemon status" : "Direct daemon status"}
                        </span>
                    </div>

                    <div className="equipment-grid dashboard-equipment-grid">
                        {isLoading && (
                            <div className="loading-message">
                                <div className="spinner" />
                                <div>
                                    {hardwareTransport === "django"
                                        ? "Loading daemon state through Django..."
                                        : "Loading equipment from local daemon..."}
                                </div>
                            </div>
                        )}
                        {error && (
                            <div className="error-message">
                                <div>Error loading equipment</div>
                                <div className="error-details">{(error as Error).message}</div>
                                <div className="error-hint">
                                    {hardwareTransport === "django"
                                        ? "Make sure Django can reach the observatory daemon or its relay tunnel."
                                        : "Make sure the Rust daemon is running on port 3030."}
                                </div>
                            </div>
                        )}
                        {!isLoading && !error && equipment.length === 0 && (
                            <div className="empty-message">
                                <div>No camera, mount, focuser, or filter wheel connected</div>
                                <div className="empty-hint">
                                    Connect hardware through the daemon, then this dashboard will populate from live status.
                                </div>
                            </div>
                        )}
                        {!isLoading && !error && equipment.map((card) => (
                            <EquipmentCard
                                key={card.id}
                                model={card}
                                selected={selectedIds.includes(card.id)}
                                draggable={card.status !== "offline"}
                                onSelect={() => addEquipment(card.id)}
                                onDragStart={handleDragStart}
                            />
                        ))}
                    </div>
                </section>

                <section className="dashboard-bands" aria-label="Operational overview">
                    <article className="band-panel queue-panel">
                        <div className="section-row compact">
                            <h2>Observation Queue</h2>
                            <span className="count-chip">{QUEUE_ITEMS.length} targets</span>
                        </div>
                        <ol className="queue-list">
                            {QUEUE_ITEMS.map((item) => (
                                <li key={item.target}>
                                    <span>{item.target}</span>
                                    <strong>{item.recipe}</strong>
                                </li>
                            ))}
                        </ol>
                    </article>

                    <article className="band-panel logs-panel">
                        <div className="section-row compact">
                            <h2>Logs</h2>
                            <span className="count-chip">Live</span>
                        </div>
                        <div className="log-stream">
                            {LOG_LINES.map((line) => (
                                <p key={`${line.time}-${line.text}`}>
                                    <time>{line.time}</time> {line.text}
                                </p>
                            ))}
                        </div>
                    </article>

                    <article className="band-panel conditions-panel">
                        <div className="section-row compact">
                            <h2>Hardware Summary</h2>
                            <span className="count-chip caution">Check</span>
                        </div>
                        <div className="condition-row"><span>Camera</span><strong>{camera ? statusText(camera) : "missing"}</strong></div>
                        <div className="condition-row"><span>Mount</span><strong>{mount ? statusText(mount) : "missing"}</strong></div>
                        <div className="condition-row"><span>Focuser</span><strong>{focuser ? statusText(focuser) : "missing"}</strong></div>
                        <div className="condition-row"><span>Filter Wheel</span><strong>{filterWheel ? statusText(filterWheel) : "missing"}</strong></div>
                    </article>
                </section>
            </section>

            <aside className="run-context-panel" aria-labelledby="runContextTitle">
                <div className="run-context-header">
                    <div>
                        <h2 id="runContextTitle">Run Context</h2>
                        <p>Selected connected hardware for this imaging job.</p>
                    </div>
                    <span className={`context-valid ${selectedCards.length > 0 ? "good" : ""}`}>
                        {selectedCards.length > 0 ? "Valid" : "Empty"}
                    </span>
                </div>

                <div
                    className={`run-drop-zone ${dropActive ? "active" : ""}`}
                    onDragOver={(event) => {
                        event.preventDefault();
                        setDropActive(true);
                    }}
                    onDragLeave={() => setDropActive(false)}
                    onDrop={handleDrop}
                >
                    {selectedCards.length === 0 ? (
                        <div className="drop-hint">Drag connected equipment here or click a card.</div>
                    ) : selectedCards.map((card) => (
                        <div key={card.id} className="selected-equipment">
                            <span className="card-type">{card.type.slice(0, 3).toUpperCase()}</span>
                            <div>
                                <strong>{card.name}</strong>
                                <small>{card.summary ?? card.site}</small>
                            </div>
                            <button type="button" className="icon-button" onClick={() => removeEquipment(card.id)}>
                                x
                            </button>
                        </div>
                    ))}
                </div>

                <div className="run-fields">
                    <label>
                        Run name
                        <input type="text" defaultValue="M42 luminance checkout" />
                    </label>
                    <label>
                        Target
                        <input type="text" defaultValue="M42 Core" />
                    </label>
                    <label>
                        Capture recipe
                        <select defaultValue="L / 24 x 120 s">
                            <option>L / 24 x 120 s</option>
                            <option>Ha / 18 x 300 s</option>
                            <option>RGB / 12 x 180 s each</option>
                        </select>
                    </label>
                </div>

                <div className="run-context-actions">
                    <button type="button" className="btn ghost" onClick={clearSelection}>
                        Clear
                    </button>
                    <button type="button" className="btn primary start-run-btn" onClick={startRun} disabled={selectedCards.length === 0}>
                        Start Run Job
                    </button>
                </div>
            </aside>
        </main>
    );
}

