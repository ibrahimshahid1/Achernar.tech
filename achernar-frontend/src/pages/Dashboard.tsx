import { useEffect, useMemo, useState } from "react";
import SidePanel from "../components/SidePanel";
import EquipmentCard from "../components/EquipmentCard";
import MjpegLiveView from "../components/MjpegLiveView";
import { cloudApi } from "../api/cloud";
import { daemonApi as localDaemonApi } from "../lib/api/daemon";
import { useTelemetrySocket } from "../lib/hooks/useTelemetrySocket";
import { useConnectedEquipment } from "../hooks/useEquipment";
import { getHardwareTransport } from "../lib/runtime";
import type { EquipmentCardModel, EquipmentType } from "../types/achernar";

const PILL_TYPES: Array<{ key: "all" | EquipmentType; label: string; icon: string }> = [
    { key: "all", label: "All", icon: "●" },
    { key: "mount", label: "Mounts", icon: "🜨" },
    { key: "focuser", label: "Focusers", icon: "⚙" },
    { key: "camera", label: "Cameras", icon: "📷" },
    { key: "filter-wheel", label: "Filter Wheels", icon: "⬤" },
    { key: "dome", label: "Domes", icon: "🏛" },
];

export default function Dashboard() {
    // Start telemetry as soon as Dashboard loads
    useTelemetrySocket();

    const [active, setActive] = useState<"all" | EquipmentType>("all");
    const [site, setSite] = useState<string>("all");
    const [query, setQuery] = useState("");
    const [relayMessage, setRelayMessage] = useState("");
    const hardwareTransport = getHardwareTransport();

    // Fetch connected equipment directly from the local daemon.
    const { data: allCards = [], isLoading, error } = useConnectedEquipment();

    const cards = useMemo(() => {
        return allCards
            .filter((c) => (active === "all" ? true : c.type === active))
            .filter((c) => (site === "all" ? true : c.site === site))
            .filter((c) => {
                const blob = `${c.name} ${c.site} ${c.tags.join(" ")}`.toLowerCase();
                return blob.includes(query.toLowerCase());
            });
    }, [allCards, active, site, query]);

    const uniqueSites = useMemo(() => {
        const s = new Set(allCards.map((c) => c.site));
        return ["all", ...Array.from(s)];
    }, [allCards]);

    useEffect(() => {
        // Keep it "dashboard-like": no heavy logic in first template version.
    }, []);

    const handleEquipmentAction = async (action: string, payload: Record<string, unknown> = {}) => {
        setRelayMessage(`Sending ${action}...`);
        try {
            const result = hardwareTransport === "django"
                ? await cloudApi.relayCommand(action, payload)
                : await localDaemonApi.relayCommand(action, payload);

            if (result.pending) {
                setRelayMessage(`${action} queued on daemon`);
                return;
            }
            if (result.ok === false) {
                setRelayMessage(result.error ?? `${action} failed`);
                return;
            }
            setRelayMessage(`${action} accepted`);
        } catch (error) {
            setRelayMessage(error instanceof Error ? error.message : `${action} failed`);
        }
    };

    return (
        <main className="layout">
            <section className="panel">
                <div className="panel-header">
                    <div>
                        <div className="panel-title">Instrument Deck</div>
                        <div className="panel-subtitle">
                            Linked mounts, focusers, cameras, filter wheels & domes across your observatories.
                        </div>
                    </div>

                    <div className="panel-controls">
                        {PILL_TYPES.map((p) => (
                            <div
                                key={p.key}
                                className={`pill ${active === p.key ? "active" : ""}`}
                                onClick={() => setActive(p.key)}
                                role="button"
                                tabIndex={0}
                            >
                                <span className="icon">{p.icon}</span> {p.label}
                            </div>
                        ))}
                    </div>
                </div>

                <div className="panel-toolbar">
                    <div className="toolbar-block">
                        <div className="filter-label">Scope / node</div>
                        <select className="filter-select" value={site} onChange={(e) => setSite(e.target.value)}>
                            {uniqueSites.map((s) => (
                                <option key={s} value={s}>
                                    {s === "all" ? "All observatories" : s}
                                </option>
                            ))}
                        </select>
                    </div>

                    <div className="toolbar-block">
                        <div className="filter-label">Search</div>
                        <input
                            className="filter-input"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="AP 1100GTO, tags, sites…"
                        />
                    </div>

                    {relayMessage && (
                        <div className="toolbar-block">
                            <div className="filter-label">Relay</div>
                            <div className="relay-status">{relayMessage}</div>
                        </div>
                    )}

                    <div className="toolbar-spacer" />

                    <div className="toolbar-block wide">
                        <MjpegLiveView />
                    </div>
                </div>

                <div className="equipment-grid">
                    {isLoading && (
                        <div className="loading-message">
                            <div className="spinner"></div>
                            <div>
                                {hardwareTransport === "django"
                                    ? "Loading daemon state through Django..."
                                    : "Loading equipment from local daemon..."}
                            </div>
                        </div>
                    )}
                    {error && (
                        <div className="error-message">
                            <div>⚠️ Error loading equipment</div>
                            <div className="error-details">{(error as Error).message}</div>
                            <div className="error-hint">
                                {hardwareTransport === "django"
                                    ? "Make sure Django is running on port 8000"
                                    : "Make sure the Rust daemon is running on port 3030"}
                            </div>
                        </div>
                    )}
                    {!isLoading && !error && cards.length === 0 && (
                        <div className="empty-message">
                            <div>No camera, mount, or focuser connected</div>
                            <div className="empty-hint">
                                {hardwareTransport === "django"
                                    ? "Django is relaying daemon state, but the daemon reports no connected hardware"
                                    : "The local daemon is online, but no supported hardware is connected"}
                            </div>
                        </div>
                    )}
                    {!isLoading && cards.map((c: EquipmentCardModel) => (
                        <EquipmentCard
                            key={c.id}
                            model={c}
                            onAction={handleEquipmentAction}
                        />
                    ))}
                </div>
            </section>

            <SidePanel />
        </main>
    );
}
