import type { EquipmentCardModel } from "../types/achernar";

export default function EquipmentCard({
    model,
    onAction,
}: {
    model: EquipmentCardModel;
    onAction?: (id: string, action: "connect" | "details" | "calibrate") => void;
}) {
    return (
        <article className={`card status-${model.status}`} data-type={model.type}>
            <div className="card-header">
                <div>
                    <div className="card-title">{model.name}</div>
                    <div className="card-subtitle">{model.site}</div>
                </div>
                <div className={`badge ${model.status}`}>{model.status.toUpperCase()}</div>
            </div>

            <div className="tag-row">
                {model.tags.slice(0, 4).map((t) => (
                    <span key={t} className="tag">
                        {t}
                    </span>
                ))}
            </div>

            {model.meta && (
                <div className="mini-metrics">
                    {Object.entries(model.meta)
                        .slice(0, 3)
                        .map(([k, v]) => (
                            <div key={k} className="mini-metric">
                                <span>{k}</span>
                                <strong>{String(v)}</strong>
                            </div>
                        ))}
                </div>
            )}

            <div className="card-actions">
                <button className="btn primary" onClick={() => onAction?.(model.id, "connect")}>
                    Control
                </button>
                <button className="btn" onClick={() => onAction?.(model.id, "details")}>
                    Details
                </button>
            </div>
        </article>
    );
}
