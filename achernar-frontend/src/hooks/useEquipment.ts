import { useQuery } from '@tanstack/react-query';
import { cloudApi } from '../api/cloud';
import { daemonApi } from '../api/daemon';
import { getHardwareTransport } from '../lib/runtime';
import type { EquipmentCardModel, EquipmentState } from '../types/achernar';

function boolText(value: boolean | undefined) {
    return value ? 'on' : 'off';
}

function cardsFromDaemon(
    status: EquipmentState,
    source: string,
    site: string,
): EquipmentCardModel[] {
    const cards: EquipmentCardModel[] = [];

    if (status.mount) {
        const ra = Number(status.mount.ra ?? 0);
        const dec = Number(status.mount.dec ?? 0);
        cards.push({
            id: 'daemon-mount',
            type: 'mount',
            name: status.mount.name ?? 'Local Mount',
            site,
            status: status.mount.connected === false ? 'offline' : 'nominal',
            summary: `RA ${ra.toFixed(2)} / Dec ${dec.toFixed(2)}`,
            source,
            tags: [
                status.mount.tracking ? 'tracking' : 'tracking off',
                status.mount.parked ? 'parked' : 'unparked',
                'Achernar daemon',
            ],
            metrics: [
                { label: 'RA', value: ra.toFixed(2) },
                { label: 'Dec', value: dec.toFixed(2) },
                { label: 'Tracking', value: boolText(status.mount.tracking), tone: status.mount.tracking ? 'good' : 'muted' },
                { label: 'Parked', value: boolText(status.mount.parked), tone: status.mount.parked ? 'warn' : 'good' },
            ],
            meta: {
                RA: ra.toFixed(2),
                Dec: dec.toFixed(2),
                Tracking: status.mount.tracking,
            },
        });
    }

    if (status.camera) {
        const temperature = Number(status.camera.temperature ?? 0);
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
            status: status.camera.connected === false ? 'offline' : 'nominal',
            summary: `${cameraState} at ${temperature.toFixed(1)} C`,
            source,
            tags: [cameraState, status.camera.cooling ? 'cooling' : 'ambient', 'Achernar daemon'],
            metrics: [
                { label: 'Mode', value: cameraState, tone: status.camera.streaming || status.camera.exposing ? 'good' : 'muted' },
                { label: 'Temp', value: `${temperature.toFixed(1)} C` },
                { label: 'Cooling', value: boolText(status.camera.cooling), tone: status.camera.cooling ? 'good' : 'muted' },
                { label: 'Stream', value: boolText(status.camera.streaming), tone: status.camera.streaming ? 'good' : 'muted' },
            ],
            meta: {
                Temp: `${temperature.toFixed(1)} C`,
                Cooling: status.camera.cooling,
                Stream: Boolean(status.camera.streaming),
            },
        });
    }

    if (status.focuser) {
        cards.push({
            id: 'daemon-focuser',
            type: 'focuser',
            name: status.focuser.name ?? 'Local Focuser',
            site,
            status: status.focuser.connected === false ? 'offline' : 'nominal',
            summary: `Position ${status.focuser.position ?? 0}`,
            source,
            tags: [status.focuser.moving ? 'moving' : 'idle', 'Achernar daemon'],
            metrics: [
                { label: 'Position', value: status.focuser.position ?? 0 },
                { label: 'Motion', value: status.focuser.moving ? 'moving' : 'idle', tone: status.focuser.moving ? 'warn' : 'good' },
            ],
            meta: {
                Position: status.focuser.position,
                Moving: status.focuser.moving,
            },
        });
    }

    return cards;
}

/**
 * Hook to fetch currently connected equipment from the active runtime transport:
 * browser -> Django relay, Tauri -> direct local daemon.
 */
export function useConnectedEquipment() {
    const hardwareTransport = getHardwareTransport();

    return useQuery({
        queryKey: ['equipment', 'connected', hardwareTransport],
        queryFn: async () => {
            if (hardwareTransport === 'django') {
                const status = await cloudApi.getDaemonStatus();
                return cardsFromDaemon(status, 'Django relay', 'Browser relay');
            }

            const status = await daemonApi.getStatus();
            return cardsFromDaemon(status, 'Direct daemon', 'Local daemon');
        },
        refetchInterval: 2000,
        staleTime: 1000,
    });
}
