import type { EquipmentEvent } from "../types/achernar";

export interface JournalPage {
    events: EquipmentEvent[];
    next_seq: number;
}

export interface JournalDrain {
    /** Every new event, oldest first, each with a seq above the cursor. */
    events: EquipmentEvent[];
    /** The `since_seq` for the next poll. */
    nextSeq: number;
}

/** The newest event a reader holds: its seq and the id that seq carried. */
export interface JournalAnchor {
    seq: number;
    event_id: string;
}

/**
 * Read the equipment journal forward from `sinceSeq` (§4.2, `GET
 * equipment/events`).
 *
 * The daemon returns the oldest events after `since_seq`, at most one page
 * per request, so a first read of a long-running daemon drains full pages
 * until a short one says the live tail is reached. `maxPages` bounds one
 * poll; the next poll continues from `nextSeq`.
 *
 * Only events above the cursor are kept, so a transport that drops
 * `since_seq` and replays the journal from the start (Django's localhost
 * fallback) adds nothing twice; a full page with nothing new ends the poll
 * instead of fetching the same page `maxPages` times.
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
        let added = 0;
        for (const event of page.events) {
            if (event.seq > since) {
                events.push(event);
                since = event.seq;
                added += 1;
            }
        }
        if (added === 0 || page.events.length < pageSize) break;
    }
    return { events, nextSeq: since };
}

/**
 * Whether the daemon's journal still holds `anchor`, the newest event this
 * reader has shown.
 *
 * The daemon answers `next_seq = since_seq` when nothing is newer, so a
 * restarted daemon that numbers below the cursor (a failed max-seq read
 * starts again at 1, a new or cleared data dir, a lost unflushed tail whose
 * seqs are reused) is invisible to the drain: it just looks idle. Asking for
 * the event at `anchor.seq` tells them apart: the same id means the same
 * journal; a different id, or no event there and none newer, means the
 * numbering restarted and the reader must read again from 0. Events newer
 * than the anchor mean the journal moved on, which the next drain reads.
 */
export async function journalContinues(
    fetchPage: (sinceSeq: number, limit: number) => Promise<JournalPage>,
    anchor: JournalAnchor,
): Promise<boolean> {
    const page = await fetchPage(Math.max(0, anchor.seq - 1), 1);
    const same = page.events.find((event) => event.seq === anchor.seq);
    if (same) return same.event_id === anchor.event_id;
    return page.events.some((event) => event.seq > anchor.seq);
}

/**
 * Newest first, capped: the display order of the events panel. An event
 * whose seq is not above every seq already held is dropped, so a replaying
 * transport never repeats a row (or a React key).
 */
export function newestFirst(previous: EquipmentEvent[], added: EquipmentEvent[], limit: number): EquipmentEvent[] {
    let highest = previous.reduce((max, event) => Math.max(max, event.seq), Number.NEGATIVE_INFINITY);
    const fresh: EquipmentEvent[] = [];
    for (const event of added) {
        if (event.seq > highest) {
            fresh.push(event);
            highest = event.seq;
        }
    }
    if (!fresh.length) return previous;
    return [...fresh.reverse(), ...previous].slice(0, limit);
}
