import type { EquipmentCardModel } from "../types/achernar";

export const seedEquipment: EquipmentCardModel[] = [
    {
        id: "eq-mount-001",
        type: "mount",
        name: "AP 1100GTO",
        site: "La Palma",
        status: "nominal",
        tags: ["tracking", "goto", "pier-west"],
        meta: { tracking: true, model: "Astro-Physics" },
    },
    {
        id: "eq-cam-001",
        type: "camera",
        name: "ASI6200MM Pro",
        site: "La Palma",
        status: "nominal",
        tags: ["mono", "cooling", "usb3"],
        meta: { temp: "-10.1°C", mode: "exposing" },
    },
    {
        id: "eq-focus-001",
        type: "focuser",
        name: "Moonlite CHL",
        site: "Siding Spring",
        status: "warning",
        tags: ["temp-drift", "backlash"],
        meta: { drift: "ΔT>3°C", steps: 12840 },
    },
    {
        id: "eq-fw-001",
        type: "filter-wheel",
        name: "EFW 7x2\"",
        site: "Atacama",
        status: "nominal",
        tags: ["Ha", "OIII", "SII"],
        meta: { slot: 3, filter: "Ha" },
    },
    {
        id: "eq-dome-001",
        type: "dome",
        name: "Robodome v2",
        site: "Atacama",
        status: "offline",
        tags: ["shutter", "az-sync"],
        meta: { reason: "link down" },
    },
];
