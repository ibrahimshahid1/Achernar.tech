import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { useMemo, useState } from "react";

function NavItem({ to, label }: { to: string; label: string }) {
    return (
        <NavLink
            to={to}
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
        >
            <span className="dot" />
            {label}
        </NavLink>
    );
}

export default function TopNav() {
    const [q, setQ] = useState("");
    const nav = useNavigate();
    const loc = useLocation();

    const placeholder = useMemo(
        () => "Search equipment, tags, or observatories…",
        []
    );

    return (
        <div className="top-nav">
            <div className="brand" onClick={() => nav("/dashboard")} role="button" tabIndex={0}>
                <div className="brand-orbit" />
                <div className="brand-title">Achernar Control</div>
            </div>

            <nav className="nav-links">
                <NavItem to="/dashboard" label="Dashboard" />
                <NavItem to="/equipment" label="Equipment" />
                <NavItem to="/queue" label="Observation Queue" />
                <NavItem to="/data-lab" label="Data Lab" />
                <NavItem to="/logs" label="Logs" />
            </nav>

            <div className="search-wrap">
                <span className="search-icon">🔍</span>
                <input
                    className="search-input"
                    value={q}
                    onChange={(e) => setQ(e.target.value)}
                    placeholder={placeholder}
                    onKeyDown={(e) => {
                        if (e.key === "Enter") {
                            nav("/equipment", { state: { q } });
                        }
                    }}
                />
            </div>

            <div className="user-block">
                <div className="status-pill">
                    <span />
                    {loc.pathname.includes("queue") ? "Queue: Armed" : "Night Run: Ready"}
                </div>
                <div className="user-avatar">AL</div>
            </div>
        </div>
    );
}
