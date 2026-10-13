interface PreviewConnectionOptions {
    pageUrl: string;
    daemonApiBase: string;
    directDaemon: boolean;
    explicitWsUrl?: string;
}

function isLoopback(hostname: string): boolean {
    return hostname === "localhost"
        || hostname === "[::1]"
        || /^127(?:\.\d{1,3}){3}$/.test(hostname);
}

/** Select only the image transport. Camera commands keep their configured route. */
export function cameraPreviewWsUrl(options: PreviewConnectionOptions): string | null {
    try {
        const page = new URL(options.pageUrl);
        const daemon = new URL(options.daemonApiBase);
        let preview: URL;
        if (options.explicitWsUrl) {
            preview = new URL(options.explicitWsUrl);
        } else {
            // A remote browser's localhost is not the telescope's localhost.
            if (!options.directDaemon && !(isLoopback(page.hostname) && isLoopback(daemon.hostname))) {
                return null;
            }
            if (daemon.protocol !== "http:" && daemon.protocol !== "https:") return null;
            preview = daemon;
            preview.protocol = daemon.protocol === "https:" ? "wss:" : "ws:";
            preview.pathname = `${preview.pathname.replace(/\/$/, "")}/camera/stream`;
            preview.search = "";
            preview.hash = "";
        }
        if (preview.protocol !== "ws:" && preview.protocol !== "wss:") return null;
        // Browsers block insecure sockets from an HTTPS page; use the relay.
        if (page.protocol === "https:" && preview.protocol !== "wss:") return null;
        return preview.toString();
    } catch {
        return null;
    }
}

export interface PreviewFrame {
    width: number;
    height: number;
    timestampMs: number;
    bytes: ArrayBuffer;
}

export function shouldShowPreview(cameraConnected: boolean, lastFreshAt: number | null, now: number): boolean {
    return cameraConnected && lastFreshAt !== null && now - lastFreshAt <= 3000;
}

/** Daemon packet: width:u32, height:u32, timestamp_ms:u64, format:u32, JPEG. */
export function decodePreviewFrame(packet: ArrayBuffer): PreviewFrame | null {
    if (packet.byteLength <= 20) return null;
    const header = new DataView(packet, 0, 20);
    const width = header.getUint32(0, true);
    const height = header.getUint32(4, true);
    const timestampMs = Number(header.getBigUint64(8, true));
    if (!width || !height || !Number.isSafeInteger(timestampMs) || header.getUint32(16, true) !== 1) {
        return null;
    }
    return { width, height, timestampMs, bytes: packet.slice(20) };
}

/** Count camera frames, rather than repeated deliveries of the latest image. */
export class PreviewFrameTracker {
    private lastTimestampMs = -Infinity;
    private lastCounter = -Infinity;
    private receivedAt: number[] = [];

    accept(timestampMs: number | undefined, now: number, counter?: number): boolean {
        if (timestampMs !== undefined && Number.isFinite(timestampMs)) {
            if (timestampMs <= this.lastTimestampMs) return false;
            this.lastTimestampMs = timestampMs;
        } else if (counter !== undefined && Number.isFinite(counter)) {
            if (counter <= this.lastCounter) return false;
        } else {
            // Without an identity there is no evidence this is a fresh frame.
            return false;
        }
        if (counter !== undefined && Number.isFinite(counter)) this.lastCounter = counter;
        this.receivedAt.push(now);
        this.prune(now);
        return true;
    }

    fps(now: number): number {
        this.prune(now);
        if (this.receivedAt.length < 2) return 0;
        const elapsed = now - this.receivedAt[0];
        return elapsed > 0 ? (this.receivedAt.length - 1) * 1000 / elapsed : 0;
    }

    private prune(now: number) {
        this.receivedAt = this.receivedAt.filter((received) => received >= now - 2000);
    }
}

/** Source age survives reloads; receiving an old JPEG does not make it fresh. */
export function sourcePreviewFresh(timestampMs: number | undefined, wallNow: number): boolean {
    return timestampMs !== undefined && Number.isFinite(timestampMs) && Number.isFinite(wallNow)
        && wallNow - timestampMs >= -1000 && wallNow - timestampMs <= 3000;
}
