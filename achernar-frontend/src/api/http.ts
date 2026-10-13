type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

export class HttpError extends Error {
    status: number;
    bodyText?: string;

    constructor(message: string, status: number, bodyText?: string) {
        super(message);
        this.status = status;
        this.bodyText = bodyText;
    }
}

export async function httpJson<TResponse>(
    url: string,
    opts: {
        method?: HttpMethod;
        body?: unknown;
        token?: string;
        signal?: AbortSignal;
        headers?: Record<string, string>;
    } = {}
): Promise<TResponse> {
    const res = await fetch(url, {
        method: opts.method ?? "GET",
        headers: {
            "Content-Type": "application/json",
            ...(opts.token ? { Authorization: `Bearer ${opts.token}` } : {}),
            ...(opts.headers ?? {}),
        },
        body: opts.body ? JSON.stringify(opts.body) : undefined,
        signal: opts.signal,
    });

    if (!res.ok) {
        const text = await res.text().catch(() => undefined);
        let detail: string | undefined;
        try {
            const body = JSON.parse(text ?? "{}");
            // Relay errors use a string; direct daemon errors carry a code
            // and message. Show the hardware failure on either transport.
            detail = typeof body.error === "string" ? body.error
                : typeof body.error?.message === "string" ? body.error.message
                    : typeof body.detail === "string" ? body.detail : undefined;
        } catch {
            // Non-JSON responses keep the status/URL fallback.
        }
        throw new HttpError(detail ?? `HTTP ${res.status} for ${url}`, res.status, text);
    }

    const text = await res.text();
    if (!text.trim()) {
        return {} as TResponse;
    }
    return JSON.parse(text) as TResponse;
}
