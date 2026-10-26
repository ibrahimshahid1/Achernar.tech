import { getHardwareTransport } from '../lib/runtime';
import { guiderStateLabel, guiderTone, qualificationLabel, safetyView, UNKNOWN } from '../lib/equipmentView';
import { useDaemonStatus } from './useDaemonStatus';
import type { DeviceBlockCommon, EquipmentCardModel, EquipmentState } from '../types/achernar';

/** Rendered wherever the daemon reported null: not measured, not zero. */
export { UNKNOWN };

/**
 * A degraded device is still connected, but its newest data is late or its
 * last poll failed (§2.8): the card says so instead of looking nominal.
 */
function cardStatus(block: DeviceBlockCommon): EquipmentCardModel['status'] {
    return block.link_state === 'degraded' || block.link_state === 'connecting' ? 'warning' : 'nominal';
}

/** The qualification tag (D21); absent from daemons that do not report it. */
function qualificationTags(block: DeviceBlockCommon): string[] {
    return block.qualification ? [qualificationLabel(block.qualification)] : [];
}

function motionText(value: boolean | null | undefined, moving: string, still: string) {
    if (value == null) return 'motion unknown';
    return value ? moving : still;
}

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
            status: cardStatus(status.mount),
            summary: `RA ${ra} / Dec ${dec}${status.mount.slewing === true ? ' (slewing)' : ''}`,
            source,
            tags: [
                status.mount.tracking == null
                    ? 'tracking unknown'
                    : status.mount.tracking ? 'tracking' : 'tracking off',
                status.mount.parked == null
                    ? 'park state unknown'
                    : status.mount.parked ? 'parked' : 'unparked',
                'Achernar daemon',
                ...qualificationTags(status.mount),
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
            status: cardStatus(status.focuser),
            // Step 0 is a legitimate position; an unread encoder is not it.
            summary: `Position ${status.focuser.position ?? UNKNOWN}`,
            source,
            // IsMoving never read is not "idle".
            tags: [motionText(status.focuser.moving, 'moving', 'idle'), 'Achernar daemon', ...qualificationTags(status.focuser)],
            metrics: [
                { label: 'Position', value: status.focuser.position ?? UNKNOWN },
                { label: 'Motion', value: motionText(status.focuser.moving, 'moving', 'idle'),
                    tone: status.focuser.moving == null ? 'muted' : status.focuser.moving ? 'warn' : 'good' },
                { label: 'Temp', value: status.focuser.temperature == null ? UNKNOWN : `${status.focuser.temperature.toFixed(1)} C` },
                { label: 'Range', value: status.focuser.max_range ?? UNKNOWN },
            ],
            meta: {
                Position: status.focuser.position ?? UNKNOWN,
                Moving: status.focuser.moving ?? UNKNOWN,
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
            status: cardStatus(wheel),
            summary: `Filter ${wheel.filter ?? UNKNOWN} / slot ${wheel.slot ?? UNKNOWN}`,
            source,
            tags: [wheel.moving ? 'moving' : 'ready', wheel.verified === true ? 'verified' : 'unverified', 'Achernar daemon',
                ...qualificationTags(wheel)],
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

    const guiding = status.guiding ?? status.guider;
    if (guiding?.connected === true) {
        // RMS in arcsec needs a pixel scale; without one it is unknown, not 0.
        const total = guiding.rms_total == null ? UNKNOWN : `${guiding.rms_total.toFixed(2)}"`;
        cards.push({
            id: 'daemon-guider',
            type: 'guider',
            name: guiding.name ?? 'Guider',
            site,
            status: cardStatus(guiding),
            summary: `${guiderStateLabel(guiding.state)} · RMS ${total}`,
            source,
            tags: [guiderStateLabel(guiding.state), 'PHD2', ...qualificationTags(guiding)],
            metrics: [
                { label: 'State', value: guiderStateLabel(guiding.state), tone: guiderTone(guiding.state) },
                { label: 'RMS', value: total },
                { label: 'SNR', value: guiding.snr == null ? UNKNOWN : guiding.snr.toFixed(1) },
                { label: 'Scale', value: guiding.pixel_scale == null ? UNKNOWN : `${guiding.pixel_scale.toFixed(2)}"/px` },
            ],
        });
    }

    if (status.weather?.connected === true) {
        const weather = status.weather;
        const temperature = weather.temperature == null ? UNKNOWN : `${weather.temperature.toFixed(1)} C`;
        const humidity = weather.humidity == null ? UNKNOWN : `${weather.humidity.toFixed(0)} %`;
        cards.push({
            id: 'daemon-weather',
            type: 'weather',
            name: weather.name ?? 'Weather',
            site,
            status: cardStatus(weather),
            summary: `${temperature} · RH ${humidity}`,
            source,
            tags: ['observing conditions', ...qualificationTags(weather)],
            metrics: [
                { label: 'Temp', value: temperature },
                { label: 'RH', value: humidity },
                { label: 'Dew pt', value: weather.dew_point == null ? UNKNOWN : `${weather.dew_point.toFixed(1)} C` },
                { label: 'Cloud', value: weather.fields?.cloud_cover_pct?.value == null ? UNKNOWN : `${weather.fields.cloud_cover_pct.value.toFixed(0)} %` },
            ],
        });
    }

    if (status.safety_monitor?.connected === true) {
        const monitor = status.safety_monitor;
        const view = safetyView(undefined, monitor);
        cards.push({
            id: 'daemon-safety-monitor',
            type: 'safety-monitor',
            name: monitor.name ?? 'Safety monitor',
            site,
            // Anything but a fresh safe reading is not nominal (§2.8 fail-safe).
            status: monitor.is_safe === true && cardStatus(monitor) === 'nominal' ? 'nominal' : 'warning',
            summary: view.label,
            source,
            tags: [view.label.toLowerCase(), ...qualificationTags(monitor)],
            metrics: [
                { label: 'Safety', value: view.label, tone: view.tone },
                { label: 'Level', value: monitor.level ?? UNKNOWN },
            ],
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
