import { describe, expect, it } from "vitest";
import type { EquipmentEvent } from "../types/achernar";
import { drainJournal, journalContinues, newestFirst, type JournalPage } from "./equipmentJournal";

const event = (seq: number, id = `e${seq}`): EquipmentEvent => ({ seq, event_id: id, at_utc: "2026-10-26T20:00:00Z", kind: "weather_sample" });

/**
 * A journal served like `GET equipment/events`: the oldest events with
 * `seq > since`, at most `limit`, and `next_seq = since` when none is newer.
 */
function journal(events: EquipmentEvent[]) {
    const calls: number[] = [];
    const fetchPage = async (since: number, limit: number): Promise<JournalPage> => {
        calls.push(since);
        const page = events.filter((e) => e.seq > since).slice(0, limit);
        return { events: page, next_seq: page.length ? page[page.length - 1].seq : since };
    };
    return { calls, fetchPage };
}
const numbered = (count: number, from = 1, prefix = "e") => Array.from({ length: count }, (_, i) => event(from + i, `${prefix}${from + i}`));

describe("equipment journal paging", () => {
    it("drains full pages until it reaches the live tail", async () => {
        const { calls, fetchPage } = journal(numbered(1234));
        const drained = await drainJournal(fetchPage, 0, 500, 40);
        expect(calls).toEqual([0, 500, 1000]);
        expect(drained.events).toHaveLength(1234);
        expect(drained.nextSeq).toBe(1234);
    });

    it("continues from where the last poll stopped and asks once when nothing is new", async () => {
        const { calls, fetchPage } = journal(numbered(10));
        expect(await drainJournal(fetchPage, 10, 500, 40)).toEqual({ events: [], nextSeq: 10 });
        expect(calls).toEqual([10]);
        expect((await drainJournal(fetchPage, 7, 500, 40)).events.map((e) => e.seq)).toEqual([8, 9, 10]);
    });

    it("bounds one poll by the page budget", async () => {
        const { calls, fetchPage } = journal(numbered(5000));
        const drained = await drainJournal(fetchPage, 0, 500, 3);
        expect(calls).toHaveLength(3);
        expect(drained.nextSeq).toBe(1500);
    });

    it("adds nothing twice from a transport that drops since_seq and replays the journal", async () => {
        // Django's localhost fallback forwards GETs without their payload.
        const all = numbered(600);
        const calls: number[] = [];
        const replaying = async (since: number): Promise<JournalPage> => {
            calls.push(since);
            const page = all.slice(0, 500);
            return { events: page, next_seq: page[page.length - 1].seq };
        };
        const first = await drainJournal(replaying, 0, 500, 40);
        expect(first.events.map((e) => e.seq)).toEqual(all.slice(0, 500).map((e) => e.seq));
        // The second, identical page adds nothing and ends the poll: no 40 fetches.
        expect(calls).toEqual([0, 500]);
        const again = await drainJournal(replaying, first.nextSeq, 500, 40);
        expect(again).toEqual({ events: [], nextSeq: 500 });
    });
});

describe("journal restart detection", () => {
    it("confirms an idle journal that still holds the newest event shown", async () => {
        const { calls, fetchPage } = journal(numbered(900));
        expect(await journalContinues(fetchPage, { seq: 900, event_id: "e900" })).toBe(true);
        expect(calls).toEqual([899]);
    });

    it("sees a restart that numbers below the cursor, which the drain cannot", async () => {
        // A failed max-seq read starts again at 1: the drain from 900 only looks idle.
        const restarted = journal(numbered(5, 1, "n"));
        expect((await drainJournal(restarted.fetchPage, 900, 500, 40)).events).toEqual([]);
        expect(await journalContinues(restarted.fetchPage, { seq: 900, event_id: "e900" })).toBe(false);
    });

    it("sees reused seq numbers after a lost unflushed tail", async () => {
        const reused = journal([...numbered(895), ...numbered(10, 896, "r")]);
        expect(await journalContinues(reused.fetchPage, { seq: 900, event_id: "e900" })).toBe(false);
    });

    it("lets a journal that moved on continue, for the next drain to read", async () => {
        const moved = journal(numbered(20, 901, "m"));
        expect(await journalContinues(moved.fetchPage, { seq: 900, event_id: "e900" })).toBe(true);
    });

    it("is not fooled by a replaying transport that still holds the anchor", async () => {
        const all = numbered(600);
        const replaying = async (): Promise<JournalPage> => ({ events: all.slice(0, 500), next_seq: 500 });
        expect(await journalContinues(replaying, { seq: 500, event_id: "e500" })).toBe(true);
    });
});

describe("events display", () => {
    it("shows the newest first and keeps only the display limit", () => {
        const shown = newestFirst([event(3), event(2), event(1)], [event(4), event(5)], 4);
        expect(shown.map((e) => e.seq)).toEqual([5, 4, 3, 2]);
    });

    it("drops events it already holds when a page replays", () => {
        const previous = [event(3), event(2), event(1)];
        expect(newestFirst(previous, [event(1), event(2), event(3)], 200)).toBe(previous);
        const shown = newestFirst(previous, [event(2), event(3), event(4), event(4)], 200);
        expect(shown.map((e) => e.seq)).toEqual([4, 3, 2, 1]);
        expect(new Set(shown.map((e) => `${e.seq}-${e.event_id}`)).size).toBe(shown.length);
    });
});
