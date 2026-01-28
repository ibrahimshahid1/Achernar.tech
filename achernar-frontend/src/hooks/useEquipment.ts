import { useQuery } from '@tanstack/react-query';
import { cloudApi } from '../api/cloud';
import type { EquipmentCardModel } from '../types/achernar';

/**
 * Hook to fetch currently connected equipment from Django (daemon relay).
 * Per PDF Section 16.2: Django serves as IoT relay for daemon equipment data.
 */
export function useConnectedEquipment() {
    return useQuery({
        queryKey: ['equipment', 'connected'],
        queryFn: async () => {
            const equipment = await cloudApi.getConnectedEquipment();

            // Transform to EquipmentCardModel format
            return equipment as EquipmentCardModel[];
        },
        refetchInterval: 10000, // Poll every 10 seconds for updates
        staleTime: 5000, // Consider data stale after 5 seconds
    });
}
