import { useEffect, useRef, useState } from "react";
import { cloudApi } from "../api/cloud";
import { daemonApi } from "../api/daemon";
import { getHardwareTransport } from "../lib/runtime";

type StreamState = "idle" | "connecting" | "waiting" | "live" | "error";

export default function MjpegLiveView() {
    const [imageUrl, setImageUrl] = useState<string | null>(null);
    const [streamState, setStreamState] = useState<StreamState>("idle");
    const [message, setMessage] = useState("Connect a camera and start the daemon stream.");
    const [connectionKey, setConnectionKey] = useState(0);
    const imageUrlRef = useRef<string | null>(null);
    const hardwareTransport = getHardwareTransport();

    useEffect(() => {
        // The live view always connects straight to the daemon's frame socket
        // (it is CORS-permissive on :3030), in both browser and desktop modes.
        let disposed = false;
        setStreamState("connecting");
        setMessage("Connecting to daemon stream...");

        const ws = new WebSocket(daemonApi.cameraStreamWsUrl());
        ws.binaryType = "arraybuffer";

        ws.onopen = () => {
            if (disposed) return;
            setStreamState("waiting");
            setMessage("Waiting for camera frames.");
        };

        ws.onerror = () => {
            if (disposed) return;
            setStreamState("error");
            setMessage("Daemon stream is unavailable.");
        };

        ws.onclose = () => {
            if (disposed) return;
            setStreamState("error");
            setMessage("Daemon stream closed.");
        };

        ws.onmessage = (event) => {
            if (disposed) return;
            if (!(event.data instanceof ArrayBuffer) || event.data.byteLength <= 20) {
                return;
            }

            const header = new DataView(event.data, 0, 20);
            const width = header.getUint32(0, true);
            const height = header.getUint32(4, true);

            if (!width || !height) {
                setStreamState("waiting");
                setMessage("Waiting for camera frames.");
                return;
            }

            const frame = event.data.slice(20);
            const nextUrl = URL.createObjectURL(new Blob([frame], { type: "image/jpeg" }));
            const previousUrl = imageUrlRef.current;
            imageUrlRef.current = nextUrl;
            setImageUrl(nextUrl);
            setStreamState("live");
            setMessage(`${width} x ${height} live frame`);

            if (previousUrl) {
                URL.revokeObjectURL(previousUrl);
            }
        };

        return () => {
            disposed = true;
            ws.close();
            if (imageUrlRef.current) {
                URL.revokeObjectURL(imageUrlRef.current);
                imageUrlRef.current = null;
            }
        };
    }, [connectionKey, hardwareTransport]);

    const startStream = async () => {
        setStreamState("connecting");
        setMessage("Starting camera stream...");
        try {
            const result = hardwareTransport === "django"
                ? await cloudApi.startCameraStream()
                : await daemonApi.startCameraStream();
            setMessage(`Stream started: ${result.width} x ${result.height} ${result.format}`);
            setStreamState("waiting");
            if (hardwareTransport === "daemon") {
                setConnectionKey((key) => key + 1);
            }
        } catch (error) {
            setStreamState("error");
            setMessage(error instanceof Error ? error.message : "Could not start camera stream.");
        }
    };

    return (
        <div className="liveview">
            <div className="liveview-title">Live View</div>
            <div className="liveview-frame">
                {imageUrl ? (
                    <img src={imageUrl} alt="Live camera stream" />
                ) : (
                    <div className={`liveview-placeholder ${streamState}`}>
                        <span>{streamState === "live" ? "Live" : message}</span>
                    </div>
                )}
            </div>
            <div className="liveview-footer">
                <span className={`liveview-status ${streamState}`}>{message}</span>
                <button className="liveview-action" type="button" onClick={startStream}>
                    {streamState === "error" ? "Retry" : "Start"}
                </button>
            </div>
        </div>
    );
}
