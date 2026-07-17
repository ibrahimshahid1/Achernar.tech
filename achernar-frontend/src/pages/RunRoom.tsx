import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import CameraControls from "../components/CameraControls";
import MjpegLiveView from "../components/MjpegLiveView";
import { cloudApi } from "../api/cloud";
import { daemonApi } from "../api/daemon";
import { useConnectedEquipment } from "../hooks/useEquipment";
import { useRunEquipmentSelection } from "../hooks/useRunEquipmentSelection";
import { useTelemetrySocket } from "../hooks/useTelemetrySocket";
import { getHardwareTransport } from "../lib/runtime";

type PanelId = "cameraPanel" | "mountPanel" | "focusPanel" | "filterPanel" | "logsPanel";

const TABS: Array<{ id: PanelId; label: string }> = [
    { id: "cameraPanel", label: "Camera" },
    { id: "mountPanel", label: "Mount" },
    { id: "focusPanel", label: "Focuser" },
    { id: "filterPanel", label: "Filter Wheel" },
    { id: "logsPanel", label: "Logs" },
];

export default function RunRoom() {
    useTelemetrySocket();

    const navigate = useNavigate();
    const hardwareTransport = getHardwareTransport();
    const { data: equipment = [] } = useConnectedEquipment();
    const { selectedCards } = useRunEquipmentSelection(equipment);
    const [activePanel, setActivePanel] = useState<PanelId>("cameraPanel");
    const [message, setMessage] = useState("");

    const camera = equipment.find((card) => card.type === "camera");
    const mount = equipment.find((card) => card.type === "mount");
    const focuser = equipment.find((card) => card.type === "focuser");
    const filterWheel = equipment.find((card) => card.type === "filter-wheel");

    const selectedSummary = useMemo(() => {
        if (selectedCards.length === 0) {
            return "No selected hardware";
        }
        return selectedCards.map((card) => card.name).join(" / ");
    }, [selectedCards]);

    const relayAction = async (action: string, payload: Record<string, unknown> = {}) => {
        setMessage(`Sending ${action}...`);
        try {
            const result = hardwareTransport === "django"
                ? await cloudApi.relayCommand(action, payload)
                : await daemonApi.relayCommand(action, payload);
            if (result.pending) {
                setMessage(`${action} queued on daemon`);
                return;
            }
            if (result.ok === false) {
                setMessage(result.error ?? `${action} failed`);
                return;
            }
            setMessage(`${action} accepted`);
        } catch (error) {
            setMessage(error instanceof Error ? error.message : `${action} failed`);
        }
    };

    return (
        <main className="run-room-page">
            <section className="run-room-shell">
                <header className="run-room-header">
                    <button type="button" className="btn" onClick={() => navigate("/dashboard")}>
                        Dashboard
                    </button>
                    <div>
                        <h1>M42 luminance checkout</h1>
                        <p>{selectedSummary}</p>
                    </div>
                    <div className="run-clock">
                        <strong>Running</strong>
                        <span>{hardwareTransport === "django" ? "Django relay" : "Direct daemon"}</span>
                    </div>
                </header>

                <div className="selected-run-strip" aria-label="Selected run equipment">
                    {selectedCards.length === 0 ? (
                        <div className="run-equipment-chip empty">Choose equipment on the dashboard first.</div>
                    ) : selectedCards.map((card) => (
                        <div key={card.id} className="run-equipment-chip">
                            <span className="card-type">{card.type.slice(0, 3).toUpperCase()}</span>
                            <div>
                                <strong>{card.name}</strong>
                                <small>{card.summary ?? card.site}</small>
                            </div>
                        </div>
                    ))}
                </div>

                <div className="equipment-tabbar" role="tablist" aria-label="Equipment controls">
                    {TABS.map((tab) => (
                        <button
                            key={tab.id}
                            type="button"
                            className={`tab-button ${activePanel === tab.id ? "active" : ""}`}
                            onClick={() => setActivePanel(tab.id)}
                        >
                            {tab.label}
                        </button>
                    ))}
                </div>

                <section className="imaging-workspace">
                    <aside className="run-side-rail">
                        <button
                            type="button"
                            className={`mini-window ${activePanel === "mountPanel" ? "active" : ""}`}
                            onClick={() => setActivePanel("mountPanel")}
                        >
                            <h3>Mount</h3>
                            <p>{mount?.summary ?? "No mount selected"}</p>
                        </button>
                        <button
                            type="button"
                            className={`mini-window ${activePanel === "focusPanel" ? "active" : ""}`}
                            onClick={() => setActivePanel("focusPanel")}
                        >
                            <h3>Focus</h3>
                            <p>{focuser?.summary ?? "No focuser selected"}</p>
                        </button>
                        <button
                            type="button"
                            className={`mini-window ${activePanel === "filterPanel" ? "active" : ""}`}
                            onClick={() => setActivePanel("filterPanel")}
                        >
                            <h3>Filter</h3>
                            <p>{filterWheel?.summary ?? "No filter wheel selected"}</p>
                        </button>
                    </aside>

                    <section className="run-camera-stage" aria-labelledby="runCameraTitle">
                        <div className="camera-stage-toolbar">
                            <div>
                                <h2 id="runCameraTitle">Live Camera</h2>
                                <span>{camera?.summary ?? "Waiting for camera status"}</span>
                            </div>
                            <div className="toolbar-actions">
                                <button type="button" className="btn" onClick={() => relayAction("camera.start_stream")}>
                                    Start Stream
                                </button>
                                <button type="button" className="btn" onClick={() => relayAction("camera.abort")}>
                                    Abort
                                </button>
                            </div>
                        </div>
                        <MjpegLiveView />
                    </section>

                    <aside className="run-control-pane">
                        <section className={`control-panel ${activePanel === "cameraPanel" ? "active" : ""}`}>
                            <CameraControls />
                        </section>
                        <section className={`control-panel ${activePanel === "mountPanel" ? "active" : ""}`}>
                            <h2>Mount Controls</h2>
                            <div className="control-readout"><span>Status</span><strong>{mount?.summary ?? "No mount"}</strong></div>
                            <div className="button-row">
                                <button type="button" className="btn" onClick={() => relayAction("mount.set_tracking", { enabled: true })}>
                                    Track
                                </button>
                                <button type="button" className="btn" onClick={() => relayAction("mount.set_tracking", { enabled: false })}>
                                    Stop Tracking
                                </button>
                                <button type="button" className="btn" onClick={() => relayAction("mount.abort")}>
                                    Abort
                                </button>
                            </div>
                        </section>
                        <section className={`control-panel ${activePanel === "focusPanel" ? "active" : ""}`}>
                            <h2>Focuser Controls</h2>
                            <div className="control-readout"><span>Status</span><strong>{focuser?.summary ?? "No focuser"}</strong></div>
                            <div className="button-row">
                                <button type="button" className="btn" onClick={() => relayAction("focuser.move_relative", { steps: -100 })}>
                                    -100
                                </button>
                                <button type="button" className="btn" onClick={() => relayAction("focuser.move_relative", { steps: -10 })}>
                                    -10
                                </button>
                                <button type="button" className="btn" onClick={() => relayAction("focuser.move_relative", { steps: 10 })}>
                                    +10
                                </button>
                                <button type="button" className="btn" onClick={() => relayAction("focuser.move_relative", { steps: 100 })}>
                                    +100
                                </button>
                            </div>
                        </section>
                        <section className={`control-panel ${activePanel === "filterPanel" ? "active" : ""}`}>
                            <h2>Filter Wheel</h2>
                            <div className="control-readout"><span>Status</span><strong>{filterWheel?.summary ?? "No filter wheel"}</strong></div>
                            <div className="filter-grid">
                                {["L", "R", "G", "B", "Ha", "OIII"].map((filter) => (
                                    <button key={filter} type="button" className="btn">
                                        {filter}
                                    </button>
                                ))}
                            </div>
                        </section>
                        <section className={`control-panel ${activePanel === "logsPanel" ? "active" : ""}`}>
                            <h2>Run Logs</h2>
                            <div className="log-stream compact">
                                <p><time>20:28:12</time> Run room opened</p>
                                <p><time>20:28:14</time> Camera controls bound to {hardwareTransport}</p>
                                {message && <p><time>now</time> {message}</p>}
                            </div>
                        </section>
                    </aside>
                </section>

                <footer className="bottom-dock" aria-label="Acquisition status">
                    <div className="dock-card"><span>Queue</span><strong>M42 Core / armed</strong></div>
                    <div className="dock-card"><span>Camera</span><strong>{camera?.summary ?? "No camera"}</strong></div>
                    <div className="dock-card"><span>Storage</span><strong>Local frames / sync idle</strong></div>
                    <div className="dock-card"><span>Relay</span><strong>{message || "Ready"}</strong></div>
                </footer>
            </section>
        </main>
    );
}

