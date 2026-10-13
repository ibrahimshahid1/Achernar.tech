import { useNavigate } from "react-router-dom";
import { usesDjangoRelay } from "../api/achernar";
import { useDaemonStatus } from "../hooks/useDaemonStatus";

export default function TopNav() {
    const navigate = useNavigate();
    const status = useDaemonStatus();

    return (
        <div className="top-nav system-bar">
            <div className="brand" onClick={() => navigate("/dashboard")} role="button" tabIndex={0}>
                <div className="brand-orbit" />
                <div>
                    <div className="brand-title">Achernar Control</div>
                    <div className="brand-subtitle">Hardware test console</div>
                </div>
            </div>

            <div className="system-readouts" aria-label="System status">
                <div className={`status-pill ${status.online ? "" : "neutral"}`} role="status">{status.online ? "Daemon reachable" : status.isLoading ? "Checking daemon" : "Daemon unavailable"}</div>
                <div className="status-pill neutral">{usesDjangoRelay() ? "Django relay" : "Direct daemon"}</div>
            </div>
        </div>
    );
}
