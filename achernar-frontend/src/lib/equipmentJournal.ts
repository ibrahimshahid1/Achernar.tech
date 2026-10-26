import type { EquipmentEvent } from "../types/achernar";

export interface JournalPage {
    events: EquipmentEvent[];
    next_seq: number;
}

export interface JournalDrain {
    /** Every new event, oldest first. */
    events: EquipmentEvent[];
    /** The `since_seq` for the next poll. */
    nextSeq: number;
    /** The daemon's journal numbering went backwards (a restart): start again from 0. */
    reset: boolean;
}

/**
 * Read the equipment journal forward from `sinceSeq` (§4.2, `GET
 * equipment/events`).
 *
 * The daemon returns the oldest events after `since_seq`, at most one page
 * per request, so a first read of a long-running daemon drains full pages
 * until a short one says the live tail is reached. `maxPages` bounds one
 * poll; the next poll continues from `nextSeq`.
 */
export async function drainJournal(
    fetchPage: (sinceSeq: number, limit: number) => Promise<JournalPage>,
    sinceSeq: number,
    pageSize: number,
    maxPages: number,
): Promise<JournalDrain> {
    let since = sinceSeq;
    const events: EquipmentEvent[] = [];
    for (let pages = 0; pages < maxPages; pages += 1) {
        const page = await fetchPage(since, pageSize);
        if (page.next_seq < since) return { events: [], nextSeq: 0, reset: true };
        events.push(...page.events);
        since = page.next_seq;
        if (page.events.length < pageSize) break;
    }
    return { events, nextSeq: since, reset: false };
}

/** Newest first, capped: the display order of the events panel. */
export function newestFirst(previous: EquipmentEvent[], added: EquipmentEvent[], limit: number): EquipmentEvent[] {
    return [...[...added].reverse(), ...previous].slice(0, limit);
}
