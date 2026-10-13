import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { achernarApi } from "../api/achernar";
import { getHardwareTransport } from "../lib/runtime";
import type { EquipmentState } from "../types/achernar";

/**
 * Daemon Status Hook
 * React Query hook for daemon equipment state
 * Polls every 2 seconds for status updates
 */
export function freshEquipmentStatus(
    status: EquipmentState | undefined,
    failed: boolean,
    updatedAt: number,
    now: number,
): EquipmentState | undefined {
    if (!status || failed || !updatedAt || now - updatedAt > 4000 || status.telemetry_stale || status.daemon_connected === false) return undefined;
    return status;
}

export function useDaemonStatus() {
    const transport = getHardwareTransport();
    const [now, setNow] = useState(Date.now);
    useEffect(() => {
        const timer = window.setInterval(() => setNow(Date.now()), 1000);
        return () => window.clearInterval(timer);
    }, []);
    const query = useQuery({
        queryKey: ["daemon", "status", transport],
        queryFn: () => achernarApi.getStatus(),
        refetchInterval: 1000,
        staleTime: 0,
        retry: false,
        refetchOnMount: "always",
        refetchIntervalInBackground: true,
    });
    const data = freshEquipmentStatus(query.data, query.isError, query.dataUpdatedAt, now);
    return { ...query, data, online: data !== undefined };
}
