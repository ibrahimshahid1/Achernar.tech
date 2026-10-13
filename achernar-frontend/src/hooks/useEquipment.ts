import { getHardwareTransport } from '../lib/runtime';
import { useDaemonStatus } from './useDaemonStatus';
import type { EquipmentCardModel, EquipmentState } from '../types/achernar';

/** Rendered wherever the daemon reported null: not measured, not zero. */
export const UNKNOWN = '--';

/**
 * Booleans from the daemon are tri-state now: true, false, or null for
 * "never read from hardware". `parked` in particular was hardcoded `false`
 * upstream, so "unparked" was being asserted about mounts nobody had asked.
 */
function boolText(value: boolean | null | undefined) {
    if (value == null) return UNKNOWN;
    return value ? 'on' : 'off';
}

function numText(value: number | null | undefined, digits = 2) {
    if (value == null) return UNKNOWN;
    return value.toFixed(digits);
}

/** Unknown is never 'good' and never 'warn' — it is simply not known. */
function boolTone(value: boolean | null | undefined, whenTrue: 'good' | 'warn', whenFalse: 'good' | 'warn' | 'muted') {
    if (value == null) return 'muted' as const;
    return value ? whenTrue : whenFalse;
}

export function cardsFromDaemon(
    status: EquipmentState,
    source: string,
    site: string,
): EquipmentCardModel[] {
    const cards: EquipmentCardModel[] = [];
    if (!status.connected || status.telemetry_stale || status.daemon_connected === false) return cards;

    if (status.mount?.connected === true) {
        // RA 0h / Dec +0deg is a real coordinate, so an unknown pointing must
        // not render as one.
        const ra = numText(status.mount.ra);
        const dec = numText(status.mount.dec);
        cards.push({
            id: 'daemon-mount',
            type: 'mount',
            name: status.mount.name ?? 'Local Mount',
            site,
            status: 'nominal',
            summary: `RA ${ra} / Dec ${dec}`,
            source,
            tags: [
                status.mount.tracking == null
                    ? 'tracking unknown'
                    : status.mount.tracking ? 'tracking' : 'tracking off',
                status.mount.parked == null
                    ? 'park state unknown'
                    : status.mount.parked ? 'parked' : 'unparked',
                'Achernar daemon',
            ],
            metrics: [
                { label: 'RA', value: ra },
                { label: 'Dec', value: dec },
                { label: 'Tracking', value: boolText(status.mount.tracking), tone: boolTone(status.mount.tracking, 'good', 'muted') },
                { label: 'Parked', value: boolText(status.mount.parked), tone: boolTone(status.mount.parked, 'warn', 'good') },
            ],
            meta: {
                RA: ra,
                Dec: dec,
                Tracking: boolText(status.mount.tracking),
            },
        });
    }

    if (status.camera?.connected === true) {
        // A DSLR has no sensor thermometer, so null is the correct answer
        // here and 0.0 C reads as a cold camera.
        const temperature = numText(status.camera.temperature, 1);
        const frame = status.camera.latest_frame;
        const cameraState = status.camera.streaming
            ? 'streaming'
            : status.camera.exposing
                ? 'exposing'
                : status.camera.status ?? 'idle';

        cards.push({
            id: 'daemon-camera',
            type: 'camera',
            name: status.camera.name ?? 'Local Camera',
            site,
            status: status.camera.status === 'error' ? 'warning' : 'nominal',
            summary: temperature === UNKNOWN
                ? cameraState
                : `${cameraState} at ${temperature} C`,
            source,
            tags: [cameraState, status.camera.cooling ? 'cooling' : 'ambient', 'Achernar daemon'],
            metrics: [
                { label: 'Mode', value: cameraState, tone: status.camera.streaming || status.camera.exposing ? 'good' : 'muted' },
                { label: 'Temp', value: temperature === UNKNOWN ? UNKNOWN : `${temperature} C` },
                { label: 'Exposure', value: status.camera.exposure_s == null ? '--' : `${Number(status.camera.exposure_s).toFixed(1)} s` },
                { label: 'Frame', value: frame?.width && frame?.height ? `${frame.width} x ${frame.height}` : boolText(status.camera.streaming), tone: status.camera.streaming ? 'good' : 'muted' },
            ],
            meta: {
                Temp: temperature === UNKNOWN ? UNKNOWN : `${temperature} C`,
                Cooling: status.camera.cooling,
                Stream: Boolean(status.camera.streaming),
            },
        });
    }

    if (status.focuser?.connected === true) {
        cards.push({
            id: 'daemon-focuser',
            type: 'focuser',
            name: status.focuser.name ?? 'Local Focuser',
            site,
            status: 'nominal',
            // Step 0 is a legitimate position; an unread encoder is not it.
            summary: `Position ${status.focuser.position ?? UNKNOWN}`,
            source,
            tags: [status.focuser.moving ? 'moving' : 'idle', 'Achernar daemon'],
            metrics: [
                { label: 'Position', value: status.focuser.position ?? UNKNOWN },
                { label: 'Motion', value: status.focuser.moving ? 'moving' : 'idle', tone: status.focuser.moving ? 'warn' : 'good' },
                { label: 'HFR', value: status.focuser.hfr ?? UNKNOWN },
                { label: 'Range', value: status.focuser.max_range ?? UNKNOWN },
            ],
            meta: {
                Position: status.focuser.position ?? UNKNOWN,
                Moving: status.focuser.moving,
            },
        });
    }

    if (status.filter_wheel?.connected === true) {
        const wheel = status.filter_wheel;
        cards.push({
            id: 'daemon-filter-wheel',
            type: 'filter-wheel',
            name: wheel.name ?? 'Filter Wheel',
            site,
            status: 'nominal',
            summary: `Filter ${wheel.filter ?? UNKNOWN} / slot ${wheel.slot ?? UNKNOWN}`,
            source,
            tags: [wheel.moving ? 'moving' : 'ready', 'filters', 'Achernar daemon'],
            metrics: [
                { label: 'Slot', value: wheel.slot ?? UNKNOWN },
                { label: 'Filter', value: wheel.filter ?? UNKNOWN },
                { label: 'Offset', value: wheel.offset ?? UNKNOWN },
                { label: 'Motion', value: wheel.moving ? 'moving' : 'stopped', tone: wheel.moving ? 'warn' : 'good' },
            ],
            meta: {
                Slot: wheel.slot ?? UNKNOWN,
                Filter: wheel.filter ?? UNKNOWN,
                Moving: Boolean(wheel.moving),
            },
        });
    }

    return cards;
}

/**
 * Currently connected equipment, as cards.
 *
 * `cardsFromDaemon` above is now the only place cards are built. Django used
 * to have its own copy in `status_to_equipment()` — UI presentation logic in
 * Python, on the far side of a network hop, where the frontend could not see
 * it and no test covered it. The two drifted on exactly the thing that
 * matters here: how an unmeasured value is rendered.
 *
 * The transport no longer branches: `achernarApi.getStatus()` fetches the same
 * daemon payload whether it travels by relay or direct HTTP.
 */
export function useConnectedEquipment() {
    const hardwareTransport = getHardwareTransport();
    const label = hardwareTransport === 'django' ? 'Django relay' : 'Direct daemon';
    const site = hardwareTransport === 'django' ? 'Browser relay' : 'Local daemon';

    const query = useDaemonStatus();
    return { ...query, data: query.data ? cardsFromDaemon(query.data, label, site) : [] };
}
