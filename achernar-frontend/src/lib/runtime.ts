export type HardwareTransport = "django" | "daemon";

export function isTauriRuntime() {
    return typeof window !== "undefined"
        && ("__TAURI_INTERNALS__" in window || "__TAURI__" in window);
}

export function getHardwareTransport(): HardwareTransport {
    const configured = import.meta.env.VITE_HARDWARE_TRANSPORT;

    if (configured === "django" || configured === "daemon") {
        return configured;
    }

    return isTauriRuntime() ? "daemon" : "django";
}
