import { useState } from "react";
import { achernarApi } from "../../api/achernar";
import { ageText, num, qualificationLabel, qualificationTone, UNKNOWN, yesNo, type RunCommand } from "../../lib/equipmentView";
import type { DeviceHealthEntry, FocuserBlock } from "../../types/achernar";
import "../equipmentPanels.css";

const UNRELIABLE_ORIGIN = new Set(["emulated_origin_unknown", "unknown_after_reconnect"]);

function Row({ label, value, tone }: { label: string; value: string; tone?: string }) {
    return <div className="eq-row"><dt>{label}</dt><dd className={tone ? `tone-${tone}` : undefined}>{value}</dd></div>;
}

/**
 * Focuser readings and manual moves (§2.12, §2.13). Every move passes the
 * daemon's gate: travel bounds, the exposure guard (A06) and, while a
 * relative focuser's origin is unknown, relative-only moves.
 *
 * `abortable` comes from the console's last known status, not the live
 * poll, so a failed or late poll never takes the halt away (D19).
 */
export default function FocuserControls({ focuser, health, online, abortable, busy, run }: {
    focuser: FocuserBlock | null | undefined;
    health?: DeviceHealthEntry | null;
    online: boolean;
    abortable: boolean;
    busy: boolean;
    run: RunCommand;
}) {
    const [position, setPosition] = useState("");
    const [origin, setOrigin] = useState("");
    const connected = online && focuser?.connected === true;
    const locked = busy || !connected;
    const unknownOrigin = focuser?.position_source != null && UNRELIABLE_ORIGIN.has(focuser.position_source);
    const absolute = focuser?.absolute === true;
    const positionValid = position.trim() !== "" && Number.isSafeInteger(Number(position)) && Number(position) >= 0
        && (focuser?.max_range == null || Number(position) <= focuser.max_range);
    const originValid = origin.trim() !== "" && Number.isSafeInteger(Number(origin));
    return <div className="eq-panel" aria-label="Focuser">
        <div className="eq-toolbar">
            <button type="button" className="btn eq-abort" disabled={!abortable} title="Halt out of band; never gated (D19)"
                onClick={() => void run("focuser.abort", () => achernarApi.abortFocuser())}>Halt focuser</button>
            {focuser?.qualification && <span className={`qual-badge tone-${qualificationTone(focuser.qualification)}`}>{qualificationLabel(focuser.qualification)}</span>}
        </div>
        {!connected && <p className="eq-empty">{!online
            ? abortable ? "Live focuser status is unavailable; halt is still sent to the daemon." : "Live focuser status is unavailable."
            : health?.state === "lost" ? "The focuser is lost; halt still reaches its abort path." : "No focuser connected."}</p>}
        {connected && <>
            <dl className="eq-grid">
                <Row label="Position" value={focuser?.position == null ? UNKNOWN : String(focuser.position)} />
                <Row label="Raw position" value={focuser?.raw_position == null ? UNKNOWN : String(focuser.raw_position)} />
                <Row label="Position source" value={focuser?.position_source?.replace(/_/g, " ") ?? UNKNOWN} tone={unknownOrigin ? "warn" : undefined} />
                <Row label="Moving" value={yesNo(focuser?.moving)} tone={focuser?.moving ? "warn" : undefined} />
                <Row label="Settling" value={yesNo(focuser?.settling)} />
                <Row label="Temperature" value={focuser?.temperature == null ? UNKNOWN
                    : `${num(focuser.temperature, 1, " °C")}${focuser.temperature_source ? ` (${focuser.temperature_source})` : ""}, ${ageText(focuser.temperature_age_ms)} old`} />
                <Row label="Temp. compensation" value={focuser?.temp_comp_available === false ? "not available" : yesNo(focuser?.temp_comp, "on", "off")} />
                <Row label="Step size" value={num(focuser?.step_size_um, 2, " µm")} />
                <Row label="Travel" value={focuser?.max_range == null ? UNKNOWN : `0 to ${focuser.max_range}${focuser.max_increment == null ? "" : `, ≤ ${focuser.max_increment} per move`}`} />
                <Row label="Mode" value={absolute ? "absolute" : focuser?.absolute === false ? "relative (emulated position)" : UNKNOWN} />
                <Row label="Backlash" value={focuser?.backlash_model ? `${focuser.backlash_model.model}${focuser.backlash_owner ? `, ${focuser.backlash_owner}` : ""}` : UNKNOWN} />
                <Row label="Reading age" value={ageText(focuser?.age_ms)} />
            </dl>
            {unknownOrigin && <p className="eq-hint tone-warn">The origin is unknown: only small relative moves are allowed and autofocus is refused until the position is declared.</p>}
            <div className="eq-toolbar">
                {[-100, -10, 10, 100].map((steps) => <button key={steps} type="button" className="btn" disabled={locked}
                    onClick={() => void run("focuser.move_relative", () => achernarApi.moveFocuser({ steps }))}>{steps > 0 ? "+" : ""}{steps} steps</button>)}
            </div>
            <div className="eq-toolbar">
                <label className="eq-inline">Absolute position <input className="cam-num" type="number" min="0" step="1" value={position} disabled={locked || !absolute} onChange={(event) => setPosition(event.target.value)} /></label>
                <button type="button" className="btn" disabled={locked || !absolute || !positionValid || unknownOrigin}
                    title={absolute ? undefined : "Absolute moves need an absolute focuser"}
                    onClick={() => void run("focuser.move_absolute", () => achernarApi.moveFocuserTo(Number(position)))}>Move to position</button>
            </div>
            {focuser?.temp_comp_available === true && <div className="eq-toolbar">
                <button type="button" className="btn" disabled={locked} onClick={() => void run("focuser.set_tempcomp", () => achernarApi.setFocuserTempComp(true))}>Temp. comp. on</button>
                <button type="button" className="btn" disabled={locked} onClick={() => void run("focuser.set_tempcomp", () => achernarApi.setFocuserTempComp(false))}>Temp. comp. off</button>
            </div>}
            {focuser?.absolute === false && <div className="eq-toolbar">
                <label className="eq-inline">Declare current position <input className="cam-num" type="number" step="1" value={origin} disabled={locked} onChange={(event) => setOrigin(event.target.value)} /></label>
                <button type="button" className="btn" disabled={locked || !originValid} title="Profile only: nothing moves; audited"
                    onClick={() => void run("focuser.set_position", () => achernarApi.declareFocuserPosition(Number(origin)))}>Declare origin</button>
            </div>}
        </>}
    </div>;
}
