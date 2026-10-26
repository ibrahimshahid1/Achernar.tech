import { describe, expect, it } from "vitest";
import type { EquipmentEvent } from "../types/achernar";
import { drainJournal, newestFirst, type JournalPage } from "./equipmentJournal";

const event = (seq: number): EquipmentEvent => ({ seq, event_id: `e${seq}`, at_utc: "2026-10-26T20:00:00Z", kind: "weather_sample" });

/** A journal of `count` events served like `GET equipment/events`. */
function journal(count: number) {
    const calls: number[] = [];
    const fetchPage = async (since: number, limit: number): Promise<JournalPage> => {
        calls.push(since);
        const events = Array.from({ length: count }, (_, i) => event(i + 1)).filter((e) => e.seq > since).slice(0, limit);
        return { events, next_seq: events.length ? events[events.length - 1].seq : since };
    };
    return { calls, fetchPage };
}

describe("equipment journal paging", () => {
    it("drains full pages until it reaches the live tail", async () => {
        const { calls, fetchPage } = journal(1234);
        const drained = await drainJournal(fetchPage, 0, 500, 40);
        expect(calls).toEqual([0, 500, 1000]);
        expect(drained.events).toHaveLength(1234);
        expect(drained.nextSeq).toBe(1234);
        expect(drained.reset).toBe(false);
    });

    it("continues from where the last poll stopped and asks once when nothing is new", async () => {
        const { calls, fetchPage } = journal(10);
        expect((await drainJournal(fetchPage, 10, 500, 40))).toEqual({ events: [], nextSeq: 10, reset: false });
        expect(calls).toEqual([10]);
        expect((await drainJournal(fetchPage, 7, 500, 40)).events.map((e) => e.seq)).toEqual([8, 9, 10]);
    });

    it("bounds one poll by the page budget", async () => {
        const { calls, fetchPage } = journal(5000);
        const drained = await drainJournal(fetchPage, 0, 500, 3);
        expect(calls).toHaveLength(3);
        expect(drained.nextSeq).toBe(1500);
    });

    it("starts again when a restarted daemon numbers its journal from the beginning", async () => {
        const restarted = async (): Promise<JournalPage> => ({ events: [event(1)], next_seq: 1 });
        expect(await drainJournal(restarted, 900, 500, 40)).toEqual({ events: [], nextSeq: 0, reset: true });
    });

    it("shows the newest first and keeps only the display limit", () => {
        const shown = newestFirst([event(3), event(2), event(1)], [event(4), event(5)], 4);
        expect(shown.map((e) => e.seq)).toEqual([5, 4, 3, 2]);
    });
});
