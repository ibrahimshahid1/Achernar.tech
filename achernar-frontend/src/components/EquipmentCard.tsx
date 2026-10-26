import type { DragEvent } from "react";
import type { EquipmentCardModel } from "../types/achernar";

const TYPE_LABELS: Record<EquipmentCardModel["type"], string> = {
    mount: "MNT",
    focuser: "FOC",
    camera: "CAM",
    "filter-wheel": "FLT",
    guider: "GDR",
    weather: "WX",
    "safety-monitor": "SAF",
    dome: "DOM",
};

/** D21 tags the card builder adds; shown as a badge, not mixed into the tag row. */
const QUALIFICATION_TAGS = new Set(["qualified", "unqualified", "planned", "qualification unknown"]);

export default function EquipmentCard({
    model,
    onAction,
    selected = false,
    draggable = false,
    onSelect,
    onDragStart,
}: {
    model: EquipmentCardModel;
    onAction?: (action: string, payload?: Record<string, unknown>) => void;
    selected?: boolean;
    draggable?: boolean;
    onSelect?: (model: EquipmentCardModel) => void;
    onDragStart?: (model: EquipmentCardModel, event: DragEvent<HTMLElement>) => void;
}) {
    const metrics: NonNullable<EquipmentCardModel["metrics"]> = model.metrics
        ?? Object.entries(model.meta ?? {}).map(([label, value]) => ({ label, value }));
    const qualification = model.tags.find((tag) => QUALIFICATION_TAGS.has(tag));
    const tags = model.tags.filter((tag) => !QUALIFICATION_TAGS.has(tag));

    return (
        <article
            className={`card status-${model.status} ${selected ? "selected" : ""} ${onSelect ? "selectable" : ""}`}
            data-type={model.type}
            data-equipment-id={model.id}
            draggable={draggable}
            onClick={() => onSelect?.(model)}
            onDragStart={(event) => onDragStart?.(model, event)}
        >
            <div className="card-header">
                <div className="card-identity">
                    <div className="card-type">{TYPE_LABELS[model.type]}</div>
                    <div>
                        <div className="card-title">{model.name}</div>
                        <div className="card-subtitle">{model.site}</div>
                    </div>
                </div>
                <div className={`badge ${model.status}`}>{model.status.toUpperCase()}</div>
            </div>

            {model.summary && <div className="card-summary">{model.summary}</div>}

            {qualification && (
                <div className={`qual-badge tone-${qualification === "qualified" ? "good" : qualification === "unqualified" ? "warn" : "muted"}`}
                    title="D21: only a rig acceptance record makes a device qualified">
                    {qualification}
                </div>
            )}

            <div className="tag-row">
                {tags.slice(0, 4).map((t) => (
                    <span key={t} className="tag">
                        {t}
                    </span>
                ))}
            </div>

            {metrics.length > 0 && (
                <div className="mini-metrics">
                    {metrics
                        .slice(0, 4)
                        .map((metric) => (
                            <div key={metric.label} className={`mini-metric tone-${metric.tone ?? "muted"}`}>
                                <span>{metric.label}</span>
                                <strong>{String(metric.value)}</strong>
                            </div>
                        ))}
                </div>
            )}

            {onAction && (
                <div className="card-actions">
                    {model.type === "camera" && (
                            <button className="btn" type="button" onClick={(event) => {
                                event.stopPropagation();
                                onAction("camera.start_stream");
                            }}>
                                Start Stream
                            </button>
                        )}
                        {model.type === "mount" && (
                            <>
                            <button className="btn" type="button" onClick={(event) => {
                                event.stopPropagation();
                                onAction("mount.set_tracking", { enabled: true });
                            }}>
                                Track
                            </button>
                            <button className="btn" type="button" onClick={(event) => {
                                event.stopPropagation();
                                onAction("mount.abort");
                            }}>
                                Stop
                            </button>
                        </>
                    )}
                    {model.type === "focuser" && (
                        <>
                            <button className="btn" type="button" onClick={(event) => {
                                event.stopPropagation();
                                onAction("focuser.move_relative", { steps: -10 });
                            }}>
                                -10
                            </button>
                            <button className="btn" type="button" onClick={(event) => {
                                event.stopPropagation();
                                onAction("focuser.move_relative", { steps: 10 });
                            }}>
                                +10
                            </button>
                        </>
                    )}
                </div>
            )}

            <div className="card-source">
                <span>{model.source ?? "Equipment cache"}</span>
                <span>{model.id}</span>
            </div>
        </article>
    );
}
