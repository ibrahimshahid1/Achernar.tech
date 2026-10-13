import { useEffect, useRef, useState } from "react";
import { achernarApi } from "../api/achernar";
import { env } from "../env";
import { cameraPreviewWsUrl, decodePreviewFrame, PreviewFrameTracker, shouldShowPreview, sourcePreviewFresh } from "../lib/cameraPreview";
import { getHardwareTransport } from "../lib/runtime";

type StreamState = "idle" | "connecting" | "waiting" | "live" | "error";

export default function MjpegLiveView({ cameraConnected, cameraConnectionId }: { cameraConnected: boolean; cameraConnectionId: string | null }) {
    const [imageUrl, setImageUrl] = useState<string | null>(null);
    const [streamState, setStreamState] = useState<StreamState>("idle");
    const [message, setMessage] = useState("Connect a camera and start the daemon stream.");
    const [connectionKey, setConnectionKey] = useState(0);
    const [frameSize, setFrameSize] = useState({ width: 0, height: 0 });
    const [fps, setFps] = useState(0);
    const [previewTransport, setPreviewTransport] = useState("snapshot");
    const imageUrlRef = useRef<string | null>(null);
    const hardwareTransport = getHardwareTransport();
    const useDirectDaemon = hardwareTransport === "daemon" || env.enableBrowserDirectDaemon;
    const previewWsUrl = cameraPreviewWsUrl({
        pageUrl: window.location.href,
        daemonApiBase: env.daemonApiBase,
        directDaemon: useDirectDaemon,
        explicitWsUrl: env.cameraPreviewWs,
    });

    useEffect(() => {
        let disposed = false;
        let ws: WebSocket | null = null;
        let socketActive = false;
        let snapshotActive = false;
        let snapshotTimer: number | undefined;
        let connectionTimer: number | undefined;
        let lastFreshAt: number | null = null;
        const frames = new PreviewFrameTracker();
        setImageUrl(null);
        setFps(0);
        if (!cameraConnected) {
            setStreamState("idle");
            setMessage("No camera connected.");
            return;
        }
        setStreamState("connecting");
        setMessage(previewWsUrl ? "Connecting to daemon preview..." : "Connecting to relayed preview...");

        const setPreviewImage = (bytes: BlobPart, mimeType: string, width: number, height: number) => {
            const nextUrl = URL.createObjectURL(new Blob([bytes], { type: mimeType }));
            const previousUrl = imageUrlRef.current;
            imageUrlRef.current = nextUrl;
            setImageUrl(nextUrl);
            setStreamState("live");
            setFrameSize({ width, height });
            lastFreshAt = performance.now();

            if (previousUrl) {
                URL.revokeObjectURL(previousUrl);
            }
        };

        const clearPreview = () => {
            if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
            imageUrlRef.current = null;
            lastFreshAt = null;
            setImageUrl(null);
            setFps(0);
            setStreamState("waiting");
            setMessage("Waiting for fresh camera frames.");
        };

        const startSnapshots = () => {
            if (disposed || snapshotActive) return;
            snapshotActive = true;
            socketActive = false;
            window.clearTimeout(connectionTimer);
            ws?.close();
            setPreviewTransport(useDirectDaemon ? "snapshot" : "relay");
            setStreamState("waiting");
            setMessage("Waiting for relayed preview frames.");

            const pollSnapshot = async () => {
                try {
                    const snapshot = await achernarApi.cameraSnapshot();
                    if (disposed || !snapshot.data_base64) return;
                    const timestampMs = snapshot.timestamp === undefined
                        ? undefined : Math.floor(snapshot.timestamp * 1000);
                    if (!sourcePreviewFresh(timestampMs, Date.now())) { clearPreview(); return; }
                    if (!frames.accept(timestampMs, performance.now(), snapshot.frame_counter)) return;
                    const raw = window.atob(snapshot.data_base64);
                    const bytes = new Uint8Array(raw.length);
                    for (let i = 0; i < raw.length; i += 1) {
                        bytes[i] = raw.charCodeAt(i);
                    }
                    setPreviewImage(
                        bytes,
                        snapshot.mime_type || "image/jpeg",
                        snapshot.width,
                        snapshot.height,
                    );
                } catch {
                    if (disposed) return;
                    setStreamState("waiting");
                    setMessage("Waiting for relayed preview frames.");
                } finally {
                    // Wait for the previous response before polling again. Slow
                    // relay requests must never overlap or reorder the images.
                    if (!disposed) snapshotTimer = window.setTimeout(pollSnapshot, 750);
                }
            };

            void pollSnapshot();
        };

        const statsTimer = window.setInterval(() => {
            const now = performance.now();
            setFps(frames.fps(now));
            if (lastFreshAt !== null && !shouldShowPreview(cameraConnected, lastFreshAt, now)) clearPreview();
        }, 500);

        if (previewWsUrl) {
            try {
                ws = new WebSocket(previewWsUrl);
                socketActive = true;
                ws.binaryType = "arraybuffer";
                setPreviewTransport("direct preview");
                connectionTimer = window.setTimeout(startSnapshots, 3000);
                ws.onopen = () => {
                    if (disposed || !socketActive) return;
                    window.clearTimeout(connectionTimer);
                    setStreamState("waiting");
                    setMessage("Waiting for camera frames.");
                };
                ws.onerror = startSnapshots;
                ws.onclose = startSnapshots;
                ws.onmessage = (event) => {
                    if (disposed || !socketActive || !(event.data instanceof ArrayBuffer)) return;
                    const frame = decodePreviewFrame(event.data);
                    if (!frame) {
                        clearPreview();
                        return;
                    }
                    if (!sourcePreviewFresh(frame.timestampMs, Date.now())) { clearPreview(); return; }
                    if (!frames.accept(frame.timestampMs, performance.now())) return;
                    setPreviewImage(frame.bytes, "image/jpeg", frame.width, frame.height);
                };
            } catch {
                startSnapshots();
            }
        } else {
            startSnapshots();
        }

        return () => {
            disposed = true;
            window.clearTimeout(snapshotTimer);
            window.clearTimeout(connectionTimer);
            window.clearInterval(statsTimer);
            ws?.close();
            if (imageUrlRef.current) {
                URL.revokeObjectURL(imageUrlRef.current);
                imageUrlRef.current = null;
            }
        };
    }, [connectionKey, previewWsUrl, useDirectDaemon, cameraConnected, cameraConnectionId]);

    const startStream = async () => {
        setStreamState("connecting");
        setMessage("Starting camera stream...");
        try {
            const result = await achernarApi.startCameraStream();
            setMessage(`Stream started: ${result.width} x ${result.height} ${result.format}`);
            setStreamState("waiting");
            setConnectionKey((key) => key + 1);
        } catch (error) {
            setStreamState("error");
            setMessage(error instanceof Error ? error.message : "Could not start camera stream.");
        }
    };

    return (
        <div className="liveview">
            <div className="liveview-title">Live View</div>
            <div className="liveview-frame">
                {imageUrl && cameraConnected ? (
                    <img src={imageUrl} alt="Live camera stream" />
                ) : (
                    <div className={`liveview-placeholder ${streamState}`}>
                        <span>{streamState === "live" ? "Live" : message}</span>
                    </div>
                )}
            </div>
            <div className="liveview-footer">
                <span className={`liveview-status ${streamState}`}>
                    {streamState === "live"
                        ? `${frameSize.width} x ${frameSize.height} · ${fps.toFixed(1)} fresh fps · ${previewTransport}`
                        : message}
                </span>
                <button className="liveview-action" type="button" onClick={startStream} disabled={!cameraConnected}>
                    {streamState === "error" ? "Retry" : "Start"}
                </button>
            </div>
        </div>
    );
}
