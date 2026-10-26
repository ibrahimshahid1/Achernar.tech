import { useState } from "react";
import { achernarApi } from "../../api/achernar";
import {
    ageText,
    guiderActionGates,
    guiderStateLabel,
    guiderTone,
    num,
    qualificationLabel,
    qualificationTone,
    settleText,
    UNKNOWN,
    yesNo,
    type RunCommand,
} from "../../lib/equipmentView";
import type { DeviceHealthEntry, GuidingBlock } from "../../types/achernar";
import "../equipmentPanels.css";

function Row({ label, value, tone }: { label: string; value: string; tone?: string }) {
    return <div className="eq-row"><dt>{label}</dt><dd className={tone ? `tone-${tone}` : undefined}>{value}</dd></div>;
}

/**
 * Guider panel: PHD2's state, RMS, star and settle readings, the last
 * dither, and start, stop, pause, resume and dither (§2.12, §2.13).
 *
 * RMS in arcsec needs PHD2's pixel scale; without one it shows unknown and
 * the pixel RMS stays visible. Stop is the guider's abort (D19): `abortable`
 * comes from the console's last known status, not the live poll, so a failed
 * or late poll never takes it away.
 */
export default function GuiderPanel({ guiding, health, online, abortable, busy, run }: {
    guiding: GuidingBlock | null | undefined;
    health?: DeviceHealthEntry | null;
    online: boolean;
    abortable: boolean;
    busy: boolean;
    run: RunCommand;
}) {
    const [recalibrate, setRecalibrate] = useState(false);
    const [amount, setAmount] = useState("");
    const [raOnly, setRaOnly] = useState(false);
    const connected = online && guiding?.connected === true;
    const gates = guiderActionGates(guiding);
    const amountValue = amount.trim() === "" ? undefined : Number(amount);
    const amountValid = amountValue === undefined || (Number.isFinite(amountValue) && amountValue > 0);
    const locked = busy || !connected;
    const dither = guiding?.last_dither;
    return <div className="eq-panel" aria-label="Guider">
        <div className="eq-toolbar">
            <span className={`eq-state tone-${guiderTone(guiding?.state)}`}>{connected ? guiderStateLabel(guiding?.state) : "not connected"}</span>
            {guiding?.qualification && <span className={`qual-badge tone-${qualificationTone(guiding.qualification)}`}>{qualificationLabel(guiding.qualification)}</span>}
            <button type="button" className="btn eq-abort" disabled={!abortable} title="stop_capture; never refused while PHD2 answers (D19)"
                onClick={() => void run("guider.stop", () => achernarApi.stopGuiding())}>Stop guiding</button>
        </div>
        {!connected && <p className="eq-empty">{!online
            ? abortable ? "Live guider status is unavailable; stop is still sent to the daemon." : "Live guider status is unavailable."
            : health?.state === "lost" ? "PHD2 is unreachable; stop is still sent." : "No guider connected. Connect PHD2 by host and port above."}</p>}
        {connected && <>
            <dl className="eq-grid">
                <Row label="RMS total" value={num(guiding?.rms_total, 2, "\"")} />
                <Row label="RMS RA / Dec" value={guiding?.rms_ra_arcsec == null && guiding?.rms_dec_arcsec == null ? UNKNOWN
                    : `${num(guiding?.rms_ra_arcsec, 2, "\"")} / ${num(guiding?.rms_dec_arcsec, 2, "\"")}`} />
                <Row label="RMS (px)" value={guiding?.rms_total_px == null ? UNKNOWN
                    : `${num(guiding.rms_total_px, 2)} (RA ${num(guiding.rms_ra_px, 2)}, Dec ${num(guiding.rms_dec_px, 2)})`} />
                <Row label="Steps in window" value={guiding?.steps_n == null ? UNKNOWN : String(guiding.steps_n)} />
                <Row label="Last step dx / dy" value={guiding?.dx == null ? UNKNOWN : `${num(guiding.dx, 2)} / ${num(guiding.dy, 2)} px`} />
                <Row label="SNR" value={num(guiding?.snr, 1)} />
                <Row label="Star mass" value={num(guiding?.star_mass, 0)} />
                <Row label="HFD" value={num(guiding?.hfd, 2, " px")} />
                <Row label="Pixel scale" value={num(guiding?.pixel_scale, 3, "\"/px")} />
                <Row label="Calibrated" value={yesNo(guiding?.calibrated)} />
                <Row label="Settling" value={guiding?.settling ? (guiding.settle
                    ? `${num(guiding.settle.distance_px, 2)} px, ${num(guiding.settle.time_s, 0)}/${num(guiding.settle.settle_time_s, 0)} s${guiding.settle.star_locked ? "" : ", star not locked"}`
                    : "yes") : yesNo(guiding?.settling)} tone={guiding?.settling ? "warn" : undefined} />
                <Row label="Last dither" value={dither ? `${new Date(dither.at).toLocaleTimeString()} · dx ${num(dither.dx, 2)} dy ${num(dither.dy, 2)} px · ${settleText(dither.settle_status)}` : "none"}
                    tone={dither?.settle_status != null && dither.settle_status !== 0 ? "bad" : undefined} />
                <Row label="PHD2" value={[guiding?.phd2_version, guiding?.profile && `profile ${guiding.profile}`, guiding?.camera_name].filter(Boolean).join(" · ") || UNKNOWN} />
                <Row label="Reading age" value={ageText(guiding?.age_ms)} />
            </dl>
            <div className="eq-toolbar">
                <button type="button" className="btn" disabled={locked || !gates.start.enabled} title={gates.start.reason}
                    onClick={() => void run("guider.start", () => achernarApi.startGuiding({ recalibrate }))}>Start guiding</button>
                <label className="eq-inline"><input type="checkbox" checked={recalibrate} disabled={locked} onChange={(event) => setRecalibrate(event.target.checked)} /> recalibrate</label>
                <button type="button" className="btn" disabled={locked || !gates.pause.enabled} title={gates.pause.reason}
                    onClick={() => void run("guider.pause", () => achernarApi.pauseGuiding())}>Pause</button>
                <button type="button" className="btn" disabled={locked || !gates.resume.enabled} title={gates.resume.reason}
                    onClick={() => void run("guider.resume", () => achernarApi.resumeGuiding())}>Resume</button>
            </div>
            <div className="eq-toolbar">
                <label className="eq-inline">Dither <input className="cam-num" type="number" min="0" step="0.5" value={amount} placeholder="[guider] default" disabled={locked} onChange={(event) => setAmount(event.target.value)} /> guide px</label>
                <label className="eq-inline"><input type="checkbox" checked={raOnly} disabled={locked} onChange={(event) => setRaOnly(event.target.checked)} /> RA only</label>
                <button type="button" className="btn" disabled={locked || !gates.dither.enabled || !amountValid} title={gates.dither.reason}
                    onClick={() => void run("guider.dither", () => achernarApi.dither({ ...(amountValue === undefined ? {} : { amount_px: amountValue }), ra_only: raOnly }))}>Dither</button>
            </div>
            {!amountValid && <p className="eq-hint tone-warn">The dither amount is a positive number of guide-camera pixels.</p>}
            {!gates.dither.enabled && <p className="eq-hint">{gates.dither.reason} The daemon also refuses a dither while an exposure integrates.</p>}
        </>}
    </div>;
}
