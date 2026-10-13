import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { cocApi, freshCocSample, validRoi, type CocRoi, type CocSample } from "../api/coc";
import { getHardwareTransport } from "../lib/runtime";
import "./cloudValidation.css";

const percent = (value: number | null) => value === null ? "Unavailable" : `${(100 * value).toFixed(1)}%`;

export default function CloudValidationPanel({ cameraConnectionId, streaming, currentFrame }: {
    cameraConnectionId: string | null; streaming: boolean; currentFrame?: {stream_generation?: number; width?: number; height?: number};
}) {
    const [now, setNow] = useState(Date.now);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState("");
    const [saved, setSaved] = useState("");
    const [frozen, setFrozen] = useState<CocSample | null>(null);
    const [frozenAt, setFrozenAt] = useState(0);
    const [notes, setNotes] = useState("");
    const [showPrediction, setShowPrediction] = useState(false);
    const [selecting, setSelecting] = useState(false);
    const [start, setStart] = useState<{ x: number; y: number } | null>(null);
    const [roi, setRoi] = useState<CocRoi | null>(null);
    const query = useQuery({ queryKey: ["coc", "status", getHardwareTransport()], queryFn: cocApi.status,
        refetchInterval: 1000, refetchIntervalInBackground: true, retry: false, staleTime: 0 });
    useEffect(() => { const timer = window.setInterval(() => setNow(Date.now()), 500); return () => window.clearInterval(timer); }, []);
    const data = query.data;
    const live = freshCocSample(data, query.isError, query.dataUpdatedAt, now, cameraConnectionId, streaming, currentFrame);
    const frozenValid = frozen && !query.isError && now - query.dataUpdatedAt <= 4000 && data?.enabled && streaming
        && frozen.context.connection_id === cameraConnectionId && frozen.context.generation === currentFrame?.stream_generation
        && frozen.context.width === currentFrame?.width && frozen.context.height === currentFrame?.height && now - frozenAt <= 120000;
    const sample = frozenValid ? frozen : live;
    const generation = sample?.context.generation;
    const width = sample?.context.width;
    const height = sample?.context.height;
    useEffect(() => { setRoi(null); setStart(null); setSelecting(false); setFrozen(null); setNotes(""); }, [cameraConnectionId, generation, width, height]);
    const change = async (enabled: boolean, model: string, region: CocRoi | null) => {
        setBusy(true); setError(""); setFrozen(null); setNotes("");
        try { await cocApi.configure(enabled, model, region, region ? sample?.context ?? null : null); await query.refetch(); }
        catch (failure) { setError(failure instanceof Error ? failure.message : "COC configuration failed."); }
        finally { setBusy(false); }
    };
    const freeze = async () => {
        if (!live) return;
        setBusy(true); setError(""); setSaved(""); setNotes("");
        try { setFrozen(await cocApi.freeze(live.sample_id)); setRoi(null); setFrozenAt(Date.now()); setShowPrediction(false); }
        catch (failure) { setError(failure instanceof Error ? failure.message : "Frame freeze failed."); }
        finally { setBusy(false); }
    };
    const save = async () => {
        if (!sample || !frozenValid) return;
        setBusy(true); setError(""); setSaved("");
        try { const result = await cocApi.save(sample.sample_id, notes); setSaved(`Saved review sample: ${result.path}`); setNotes(""); setFrozen(null); }
        catch (failure) { setError(failure instanceof Error ? failure.message : "Sample save failed."); }
        finally { setBusy(false); }
    };
    const selectPoint = (event: React.PointerEvent<HTMLDivElement>) => {
        const box = event.currentTarget.getBoundingClientRect();
        return { x: Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)),
            y: Math.max(0, Math.min(1, (event.clientY - box.top) / box.height)) };
    };
    const shownRoi = frozenValid ? frozen!.roi : roi ?? (data?.roi_valid_for_current_camera ? data.roi : null);
    return <section className="coc-validation" aria-label="Cloud model local validation">
        <div className="section-row"><h2>Cloud model validation</h2><strong>Shadow only</strong></div>
        <p>Preview segmentation for local testing. Predictions cannot pause exposures or move equipment.</p>
        <div className="button-row">
            <label>Model <select className="cam-select" value={data?.model ?? "teacher"} disabled={busy || !data}
                onChange={(event) => void change(data?.enabled ?? false, event.target.value, null)}>
                {(data?.models ?? [{ id: "teacher", loaded: false, error: null }]).map((model) =>
                    <option key={model.id} value={model.id} disabled={!model.loaded}>{model.id === "teacher" ? "Teacher assisted" : "Human only"}{model.loaded ? "" : " (unavailable)"}</option>)}
            </select></label>
            <button className="btn" type="button" disabled={busy || !data || !data.models.find((m) => m.id === data.model)?.loaded}
                onClick={() => void change(!data?.enabled, data?.model ?? "teacher", null)}>{data?.enabled ? "Disable inference" : "Enable inference"}</button>
            <button className="btn" type="button" disabled={query.isFetching} onClick={() => void query.refetch()}>Refresh model status</button>
        </div>
        <p role="status">{query.isError ? "COC status unavailable; previous predictions hidden." : data?.reason ?? data?.status ?? "Checking COC model..."}</p>
        {data?.models.filter((model) => !model.loaded).map((model) => <p key={model.id}>{model.id}: {model.error}</p>)}
        {(error || query.error) && <p className="error-details">{error || (query.error instanceof Error ? query.error.message : "Status request failed.")}</p>}
        <div className="button-row">
            <button className="btn" type="button" disabled={busy || !live} onClick={() => void freeze()}>Freeze a review frame</button>
            {frozen && <button className="btn" type="button" onClick={() => { setFrozen(null); setNotes(""); }}>Return to live</button>}
        </div>
        {frozen && !frozenValid && <p>Frozen review expired or camera status changed. Freeze a fresh frame.</p>}
        {sample && <>
            {frozenValid && <p role="status">Frozen review frame · {new Date(sample.source_timestamp * 1000).toLocaleTimeString()} · save within two minutes.</p>}
            <p>{sample.context.width} × {sample.context.height} {sample.source_format} preview · frame {sample.frame_counter} · {sample.inference_ms.toFixed(1)} ms inference</p>
            <div className="coc-images">
                <figure><figcaption>Model input · drag a rectangle containing usable sky</figcaption>
                    <div className={`coc-roi-image ${selecting ? "selecting" : ""}`}
                        onPointerDown={(event) => { if (selecting) { event.currentTarget.setPointerCapture(event.pointerId); setStart(selectPoint(event)); setRoi(null); } }}
                        onPointerMove={(event) => { if (selecting && start) { const point = selectPoint(event); setRoi({ left: Math.min(start.x, point.x), top: Math.min(start.y, point.y), right: Math.max(start.x, point.x), bottom: Math.max(start.y, point.y) }); } }}
                        onPointerUp={() => { setStart(null); setSelecting(false); }}>
                        <img alt="Current rendered RGB input to COC" draggable={false} src={`data:image/png;base64,${sample.input_png_base64}`} />
                        {shownRoi && <div className="coc-roi-box" style={{ left: `${shownRoi.left * 100}%`, top: `${shownRoi.top * 100}%`, width: `${(shownRoi.right - shownRoi.left) * 100}%`, height: `${(shownRoi.bottom - shownRoi.top) * 100}%` }} />}
                    </div>
                </figure>
                {showPrediction && <figure><figcaption>Predicted overlay · blue sky, white cloud, orange contamination</figcaption><img alt="Uncalibrated COC segmentation overlay" src={`data:image/png;base64,${sample.overlay_png_base64}`} /></figure>}
            </div>
            <div className="button-row">
                <button className="btn" type="button" onClick={() => setShowPrediction(!showPrediction)}>{showPrediction ? "Hide prediction" : "Show prediction"}</button>
                <button className="btn" type="button" disabled={busy || !!frozenValid} onClick={() => { setSelecting(true); setRoi(null); }}>Select sky region</button>
                <button className="btn" type="button" disabled={busy || !!frozenValid || !roi || !validRoi(roi)} onClick={() => void change(true, data?.model ?? "teacher", roi)}>Apply selected region</button>
                <button className="btn" type="button" disabled={busy || !data?.roi} onClick={() => { setRoi(null); void change(true, data?.model ?? "teacher", null); }}>Clear region</button>
            </div>
            <p>Select the region from the input image, independently of the prediction. Reselect after changing framing. Without a region, maps are available and coverage summaries abstain.</p>
            {showPrediction && <div>
                <p>Diagnostic: {sample.summary.status}{sample.summary.reason ? ` · ${sample.summary.reason.replace(/_/g, " ")}` : ""}</p>
                <p>Cloud area in region: {percent(sample.summary.cloud_area_fraction_of_roi)} · Contamination: {percent(sample.summary.contamination_area_fraction_of_roi)}</p>
                <p>Scores are uncalibrated. These are not transparency classes or a weather safety assessment.</p>
            </div>}
            <label className="coc-notes">Review notes <textarea disabled={!frozenValid} maxLength={1000} value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="Visible clouds, uncertain regions, glow/foreground, camera settings or a framing change" /></label>
            <button className="btn" type="button" disabled={busy || !frozenValid} onClick={() => void save()}>Save frozen review sample</button>
            <p>Freeze a frame before writing notes. Saving records that fixed input, outputs, region, frame/model identity and notes. Samples remain unapproved for training; notes are not pixel-level ground truth. Review the input before showing predictions when assessing accuracy.</p>
        </>}
        {saved && <p role="status">{saved}</p>}
    </section>;
}
