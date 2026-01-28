import { useQuery } from "@tanstack/react-query";
import { daemonApi } from "../api/daemon";

/**
 * Daemon Status Hook
 * React Query hook for daemon equipment state
 * Polls every 2 seconds for status updates
 */
export function useDaemonStatus() {
    return useQuery({
        queryKey: ["daemon", "status"],
        queryFn: () => daemonApi.getStatus(),
        refetchInterval: 2_000,
    });
}
