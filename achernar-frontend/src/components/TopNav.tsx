import { useNavigate } from "react-router-dom";
import { getHardwareTransport } from "../lib/runtime";

export default function TopNav() {
    const navigate = useNavigate();
    const hardwareTransport = getHardwareTransport();

    return (
        <div className="top-nav system-bar">
            <div className="brand" onClick={() => navigate("/dashboard")} role="button" tabIndex={0}>
                <div className="brand-orbit" />
                <div>
                    <div className="brand-title">Achernar Control</div>
                    <div className="brand-subtitle">Observatory hardware console</div>
                </div>
            </div>

            <div className="system-readouts" aria-label="System status">
                <div className="status-pill"><span /> Daemon status live</div>
                <div className="status-pill neutral">{hardwareTransport === "django" ? "Django relay" : "Direct daemon"}</div>
                <div className="user-avatar">AL</div>
            </div>
        </div>
    );
}

