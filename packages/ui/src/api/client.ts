/**
 * Core Telescope Client Interface
 * 
 * Provides a unified API for controlling astronomy equipment
 * across both local daemon (Desktop) and cloud backend (Web).
 */

export interface CaptureParams {
    exposure: number;        // Exposure time in seconds
    gain?: number;           // Camera gain
    offset?: number;         // Camera offset
    binning?: number;        // Pixel binning (1, 2, 4, etc.)
    filter?: string;         // Filter name
    temperature?: number;    // Target sensor temperature in Celsius
}

export interface ImageData {
    id: string;
    data: ArrayBuffer | string;  // Raw FITS data or base64
    width: number;
    height: number;
    metadata: {
        exposure: number;
        gain: number;
        temperature: number;
        timestamp: string;
        [key: string]: any;
    };
}

export interface EquipmentStatus {
    connected: boolean;
    camera?: {
        name: string;
        temperature: number;
        cooling: boolean;
        exposing: boolean;
        streaming: boolean;  // Is live view active
    };
    mount?: {
        name: string;
        ra: number;
        dec: number;
        tracking: boolean;
        parked: boolean;
    };
    focuser?: {
        name: string;
        position: number;
        moving: boolean;
    };
}

/**
 * Live View Stream Frame
 * Low-resolution, high-framerate for focusing and framing
 */
export interface StreamFrame {
    data: ArrayBuffer;      // JPEG or PNG compressed frame
    width: number;
    height: number;
    timestamp: number;
    focusScore?: number;    // AI-computed focus quality
}

export interface FocuserAPI {
    move(steps: number): Promise<void>;
    moveTo(position: number): Promise<void>;
    getPosition(): Promise<number>;
    abort(): Promise<void>;
}

export interface MountAPI {
    slew(ra: number, dec: number): Promise<void>;
    slewToTarget(targetName: string): Promise<void>;
    sync(ra: number, dec: number): Promise<void>;
    park(): Promise<void>;
    unpark(): Promise<void>;
    abort(): Promise<void>;
    setTracking(enabled: boolean): Promise<void>;
}

/**
 * Main Telescope Client Interface
 * Implemented by both LocalClient and CloudClient
 */
export interface TelescopeClient {
    connect(): Promise<void>;
    disconnect(): Promise<void>;

    // High-res capture (FITS/RAW)
    capture(params: CaptureParams): Promise<ImageData>;
    abortCapture(): Promise<void>;

    // Live View Streaming (for focusing/framing)
    startStream(onFrame: (frame: StreamFrame) => void, onError?: (error: Error) => void): Promise<void>;
    stopStream(): Promise<void>;

    // MJPEG stream URL (alternative to WebSocket)
    getStreamUrl(): string;

    // Status
    getStatus(): Promise<EquipmentStatus>;

    // Sub-systems
    focuser: FocuserAPI;
    mount: MountAPI;
}

/**
 * Local Daemon Client (Desktop App)
 * Communicates with Rust daemon via HTTP and WebSocket
 */
export class LocalClient implements TelescopeClient {
    private baseUrl: string;
    private wsUrl: string;
    private streamSocket: WebSocket | null = null;

    constructor(port: number = 3030) {
        this.baseUrl = `http://127.0.0.1:${port}`;
        this.wsUrl = `ws://127.0.0.1:${port}`;
    }

    async connect(): Promise<void> {
        const response = await fetch(`${this.baseUrl}/connect`, {
            method: 'POST',
        });
        if (!response.ok) throw new Error('Failed to connect to daemon');
    }

    async disconnect(): Promise<void> {
        await this.stopStream();
        await fetch(`${this.baseUrl}/disconnect`, { method: 'POST' });
    }

    async capture(params: CaptureParams): Promise<ImageData> {
        const response = await fetch(`${this.baseUrl}/camera/capture`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        if (!response.ok) throw new Error('Capture failed');
        return response.json();
    }

    async abortCapture(): Promise<void> {
        await fetch(`${this.baseUrl}/camera/abort`, { method: 'POST' });
    }

    /**
     * Start live view stream via WebSocket
     * Low-latency binary frames for real-time focusing and AI inference
     */
    async startStream(
        onFrame: (frame: StreamFrame) => void,
        onError?: (error: Error) => void
    ): Promise<void> {
        if (this.streamSocket) {
            await this.stopStream();
        }

        return new Promise((resolve, reject) => {
            const ws = new WebSocket(`${this.wsUrl}/camera/stream`);
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => {
                this.streamSocket = ws;
                resolve();
            };

            ws.onmessage = (event) => {
                try {
                    // Parse binary frame data
                    const data = event.data as ArrayBuffer;
                    const view = new DataView(data);

                    // Frame format: [width(4)][height(4)][timestamp(8)][focusScore(4)][imageData(...)]
                    const width = view.getUint32(0, true);
                    const height = view.getUint32(4, true);
                    const timestamp = Number(view.getBigUint64(8, true));
                    const focusScore = view.getFloat32(16, true);
                    const imageData = data.slice(20);

                    onFrame({
                        data: imageData,
                        width,
                        height,
                        timestamp,
                        focusScore: focusScore > 0 ? focusScore : undefined,
                    });
                } catch (error) {
                    onError?.(error as Error);
                }
            };

            ws.onerror = (_event) => {
                const error = new Error('WebSocket error');
                onError?.(error);
                reject(error);
            };

            ws.onclose = () => {
                this.streamSocket = null;
            };
        });
    }

    async stopStream(): Promise<void> {
        if (this.streamSocket) {
            this.streamSocket.close();
            this.streamSocket = null;
        }
    }

    /**
     * Get MJPEG stream URL (alternative to WebSocket)
     * Can be used directly in <img> or <video> element
     */
    getStreamUrl(): string {
        return `${this.baseUrl}/camera/stream/mjpeg`;
    }

    async getStatus(): Promise<EquipmentStatus> {
        const response = await fetch(`${this.baseUrl}/status`);
        if (!response.ok) throw new Error('Failed to get status');
        return response.json();
    }

    focuser: FocuserAPI = {
        move: async (steps: number) => {
            await fetch(`${this.baseUrl}/focuser/move`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ steps }),
            });
        },
        moveTo: async (position: number) => {
            await fetch(`${this.baseUrl}/focuser/moveto`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ position }),
            });
        },
        getPosition: async () => {
            const response = await fetch(`${this.baseUrl}/focuser/position`);
            const data = await response.json();
            return data.position;
        },
        abort: async () => {
            await fetch(`${this.baseUrl}/focuser/abort`, { method: 'POST' });
        },
    };

    mount: MountAPI = {
        slew: async (ra: number, dec: number) => {
            await fetch(`${this.baseUrl}/mount/slew`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ra, dec }),
            });
        },
        slewToTarget: async (targetName: string) => {
            await fetch(`${this.baseUrl}/mount/slew-target`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target: targetName }),
            });
        },
        sync: async (ra: number, dec: number) => {
            await fetch(`${this.baseUrl}/mount/sync`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ra, dec }),
            });
        },
        park: async () => {
            await fetch(`${this.baseUrl}/mount/park`, { method: 'POST' });
        },
        unpark: async () => {
            await fetch(`${this.baseUrl}/mount/unpark`, { method: 'POST' });
        },
        abort: async () => {
            await fetch(`${this.baseUrl}/mount/abort`, { method: 'POST' });
        },
        setTracking: async (enabled: boolean) => {
            await fetch(`${this.baseUrl}/mount/tracking`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled }),
            });
        },
    };
}

/**
 * Cloud Backend Client (Web App)
 * Communicates with Django REST API and Django Channels WebSocket
 */
export class CloudClient implements TelescopeClient {
    private baseUrl: string;
    private wsUrl: string;
    private streamSocket: WebSocket | null = null;

    constructor(baseUrl: string = '/api/v1') {
        this.baseUrl = baseUrl;
        // Convert http(s) to ws(s) for WebSocket
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.wsUrl = `${protocol}//${window.location.host}/ws`;
    }

    async connect(): Promise<void> {
        const response = await fetch(`${this.baseUrl}/equipment/connect`, {
            method: 'POST',
            credentials: 'include',
        });
        if (!response.ok) throw new Error('Failed to connect');
    }

    async disconnect(): Promise<void> {
        await this.stopStream();
        await fetch(`${this.baseUrl}/equipment/disconnect`, {
            method: 'POST',
            credentials: 'include',
        });
    }

    async capture(params: CaptureParams): Promise<ImageData> {
        const response = await fetch(`${this.baseUrl}/imaging/capture`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(params),
        });
        if (!response.ok) throw new Error('Capture failed');
        return response.json();
    }

    async abortCapture(): Promise<void> {
        await fetch(`${this.baseUrl}/imaging/abort`, {
            method: 'POST',
            credentials: 'include',
        });
    }

    /**
     * Start live view stream via Django Channels WebSocket
     */
    async startStream(
        onFrame: (frame: StreamFrame) => void,
        onError?: (error: Error) => void
    ): Promise<void> {
        if (this.streamSocket) {
            await this.stopStream();
        }

        return new Promise((resolve, reject) => {
            const ws = new WebSocket(`${this.wsUrl}/camera/stream/`);
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => {
                this.streamSocket = ws;
                // Send start command
                ws.send(JSON.stringify({ command: 'start_stream' }));
                resolve();
            };

            ws.onmessage = (event) => {
                try {
                    if (typeof event.data === 'string') {
                        // JSON message (metadata or error)
                        const msg = JSON.parse(event.data);
                        if (msg.error) {
                            onError?.(new Error(msg.error));
                        }
                    } else {
                        // Binary frame data
                        const data = event.data as ArrayBuffer;
                        const view = new DataView(data);

                        const width = view.getUint32(0, true);
                        const height = view.getUint32(4, true);
                        const timestamp = Number(view.getBigUint64(8, true));
                        const focusScore = view.getFloat32(16, true);
                        const imageData = data.slice(20);

                        onFrame({
                            data: imageData,
                            width,
                            height,
                            timestamp,
                            focusScore: focusScore > 0 ? focusScore : undefined,
                        });
                    }
                } catch (error) {
                    onError?.(error as Error);
                }
            };

            ws.onerror = (_event) => {
                const error = new Error('WebSocket error');
                onError?.(error);
                reject(error);
            };

            ws.onclose = () => {
                this.streamSocket = null;
            };
        });
    }

    async stopStream(): Promise<void> {
        if (this.streamSocket) {
            this.streamSocket.send(JSON.stringify({ command: 'stop_stream' }));
            this.streamSocket.close();
            this.streamSocket = null;
        }
    }

    /**
     * Get MJPEG stream URL from Django
     */
    getStreamUrl(): string {
        return `${this.baseUrl}/camera/stream/mjpeg`;
    }

    async getStatus(): Promise<EquipmentStatus> {
        const response = await fetch(`${this.baseUrl}/equipment/status`, {
            credentials: 'include',
        });
        if (!response.ok) throw new Error('Failed to get status');
        return response.json();
    }

    focuser: FocuserAPI = {
        move: async (steps: number) => {
            await fetch(`${this.baseUrl}/focuser/move`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ steps }),
            });
        },
        moveTo: async (position: number) => {
            await fetch(`${this.baseUrl}/focuser/moveto`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ position }),
            });
        },
        getPosition: async () => {
            const response = await fetch(`${this.baseUrl}/focuser/position`, {
                credentials: 'include',
            });
            const data = await response.json();
            return data.position;
        },
        abort: async () => {
            await fetch(`${this.baseUrl}/focuser/abort`, {
                method: 'POST',
                credentials: 'include',
            });
        },
    };

    mount: MountAPI = {
        slew: async (ra: number, dec: number) => {
            await fetch(`${this.baseUrl}/mount/slew`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ ra, dec }),
            });
        },
        slewToTarget: async (targetName: string) => {
            await fetch(`${this.baseUrl}/mount/slew-target`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ target: targetName }),
            });
        },
        sync: async (ra: number, dec: number) => {
            await fetch(`${this.baseUrl}/mount/sync`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ ra, dec }),
            });
        },
        park: async () => {
            await fetch(`${this.baseUrl}/mount/park`, {
                method: 'POST',
                credentials: 'include',
            });
        },
        unpark: async () => {
            await fetch(`${this.baseUrl}/mount/unpark`, {
                method: 'POST',
                credentials: 'include',
            });
        },
        abort: async () => {
            await fetch(`${this.baseUrl}/mount/abort`, {
                method: 'POST',
                credentials: 'include',
            });
        },
        setTracking: async (enabled: boolean) => {
            await fetch(`${this.baseUrl}/mount/tracking`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ enabled }),
            });
        },
    };
}
