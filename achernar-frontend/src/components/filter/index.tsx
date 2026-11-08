import { achernarApi } from "../../api/achernar";
import { ageText, qualificationLabel, qualificationTone, UNKNOWN, type RunCommand } from "../../lib/equipmentView";
import type { DeviceHealthEntry, FilterWheelBlock } from "../../types/achernar";
import "../equipmentPanels.css";

/**
 * Filter wheel position and control (§2.12, §3.4).
 *
 * A change runs the v6 §8.2.1 handshake on the daemon (wheel move, verify,
 * settle, focus offset, guiding pause), so this panel only names the target
 * slot. Offsets come from the identity profile; the `[filter_wheel]` TOML
 * values are seeds that apply only after an explicit, audited import (D17).
 *
 * The abort (cancel the handshake) stays available for a lost or degraded
 * wheel and through a failed or late status poll (D19): `abortable` comes
 * from the console's last known status.
 */
export function FilterWheelControl({ wheel, health, online, abortable, busy, run }: {
    wheel: FilterWheelBlock | null | undefined;
    health?: DeviceHealthEntry | null;
    online: boolean;
    abortable: boolean;
    busy: boolean;
    run: RunCommand;
}) {
    const connected = online && wheel?.connected === true;
    const abortButton = <button type="button" className="btn eq-abort" disabled={!abortable} title="Cancel the running handshake; the wheel itself has no halt"
        onClick={() => void run("filter_wheel.abort", () => achernarApi.abortFilterChange())}>Abort change</button>;
    if (!connected) {
        const text = !online
            ? abortable ? "Live filter wheel status is unavailable; abort is still sent to the daemon." : "Live filter wheel status is unavailable."
            : health?.state === "lost" ? "The filter wheel is lost; abort still cancels a running handshake." : "No filter wheel connected.";
        if (!abortable) return <p className="eq-empty">{text}</p>;
        return <div className="eq-panel" aria-label="Filter wheel">
            <div className="eq-toolbar">{abortButton}</div>
            <p className="eq-empty">{text}</p>
        </div>;
    }
    const slots = wheel?.slots ?? [];
    const moving = wheel?.moving === true;
    const locked = busy || moving;
    return <div className="eq-panel" aria-label="Filter wheel">
        <div className="eq-toolbar">
            <span className="eq-state">{moving ? "moving" : `slot ${wheel?.slot ?? UNKNOWN} · ${wheel?.filter ?? "filter unknown"}`}</span>
            <span className={`qual-badge tone-${wheel?.verified ? "good" : "warn"}`} title="The reported slot was read back after a handshake">{wheel?.verified ? "verified" : "unverified"}</span>
            {wheel?.qualification && <span className={`qual-badge tone-${qualificationTone(wheel.qualification)}`}>{qualificationLabel(wheel.qualification)}</span>}
            {abortButton}
        </div>
        {slots.length === 0 ? <p className="eq-empty">The wheel reported no slots.</p> : <table className="eq-table">
            <thead><tr><th>Slot</th><th>Filter</th><th>Offset (steps)</th><th /></tr></thead>
            <tbody>
                {slots.map((slot) => {
                    const current = slot.index === wheel?.slot;
                    return <tr key={slot.index} className={current ? "eq-current" : undefined}>
                        <td>{slot.index}</td>
                        <td>{slot.name || UNKNOWN}</td>
                        <td>{slot.offset ?? UNKNOWN}</td>
                        <td><button type="button" className="btn" disabled={locked || (current && wheel?.verified === true)}
                            onClick={() => void run("filter_wheel.change", () => achernarApi.changeFilter({ slot: slot.index }))}>{current ? "Re-verify" : "Change"}</button></td>
                    </tr>;
                })}
            </tbody>
        </table>}
        <div className="eq-toolbar">
            <button type="button" className="btn" disabled={busy} title="Copy the [filter_wheel] TOML seeds into this wheel's identity profile (audited)"
                onClick={() => void run("filter_wheel.import_seed", () => achernarApi.importFilterSeed())}>Import TOML seeds</button>
            <span className="eq-hint">Reading age {ageText(wheel?.age_ms)}. Changes are refused while an exposure integrates.</span>
        </div>
    </div>;
}

/** Filter bandit status and arm selection display. */
export function FilterBanditStatus() {
    return <div id="filter-bandit-status">Filter Bandit Status — stub</div>;
}
