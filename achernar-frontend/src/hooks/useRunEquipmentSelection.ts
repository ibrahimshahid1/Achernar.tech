import { useCallback, useEffect, useMemo, useState } from "react";
import type { EquipmentCardModel } from "../types/achernar";

const STORAGE_KEY = "achernar.run.selectedEquipment";
const DEFAULT_TYPES = new Set(["camera", "mount"]);

function readStoredSelection(): string[] {
    if (typeof window === "undefined") {
        return [];
    }
    try {
        const parsed = JSON.parse(window.sessionStorage.getItem(STORAGE_KEY) || "[]");
        return Array.isArray(parsed) ? parsed.filter((id): id is string => typeof id === "string") : [];
    } catch {
        return [];
    }
}

function writeStoredSelection(ids: string[]) {
    if (typeof window === "undefined") {
        return;
    }
    try {
        window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(ids));
    } catch {
        // Session persistence is a convenience; selection still works in memory.
    }
}

function defaultSelection(cards: EquipmentCardModel[]) {
    return cards
        .filter((card) => card.status !== "offline" && DEFAULT_TYPES.has(card.type))
        .map((card) => card.id);
}

export function useRunEquipmentSelection(cards: EquipmentCardModel[]) {
    const [selectedIds, setSelectedIds] = useState<string[]>(() => readStoredSelection());
    const cardIds = useMemo(() => new Set(cards.map((card) => card.id)), [cards]);

    useEffect(() => {
        setSelectedIds((current) => {
            const pruned = current.filter((id) => cardIds.has(id));
            const next = pruned.length > 0 ? pruned : defaultSelection(cards);
            if (current.length === next.length && current.every((id, index) => id === next[index])) {
                return current;
            }
            writeStoredSelection(next);
            return next;
        });
    }, [cardIds, cards]);

    const selectedCards = useMemo(
        () => selectedIds
            .map((id) => cards.find((card) => card.id === id))
            .filter((card): card is EquipmentCardModel => Boolean(card)),
        [cards, selectedIds],
    );

    const addEquipment = useCallback((id: string) => {
        if (!cardIds.has(id)) {
            return;
        }
        setSelectedIds((current) => {
            const next = current.includes(id) ? current : [...current, id];
            writeStoredSelection(next);
            return next;
        });
    }, [cardIds]);

    const removeEquipment = useCallback((id: string) => {
        setSelectedIds((current) => {
            const next = current.filter((selectedId) => selectedId !== id);
            writeStoredSelection(next);
            return next;
        });
    }, []);

    const clearSelection = useCallback(() => {
        setSelectedIds([]);
        writeStoredSelection([]);
    }, []);

    return {
        selectedIds,
        selectedCards,
        addEquipment,
        removeEquipment,
        clearSelection,
    };
}
