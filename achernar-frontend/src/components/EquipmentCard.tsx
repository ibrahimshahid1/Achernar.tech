import type { EquipmentCardModel } from "../types/achernar";

const TYPE_LABELS: Record<EquipmentCardModel["type"], string> = {
    mount: "MNT",
    focuser: "FOC",
    camera: "CAM",
    "filter-wheel": "FLT",
    dome: "DOM",
};

export default function EquipmentCard({
    model,
}: {
    model: EquipmentCardModel;
}) {
    const metrics: NonNullable<EquipmentCardModel["metrics"]> = model.metrics
        ?? Object.entries(model.meta ?? {}).map(([label, value]) => ({ label, value }));

    return (
        <article className={`card status-${model.status}`} data-type={model.type}>
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

            <div className="tag-row">
                {model.tags.slice(0, 4).map((t) => (
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

            <div className="card-source">
                <span>{model.source ?? "Equipment cache"}</span>
                <span>{model.id}</span>
            </div>
        </article>
    );
}
