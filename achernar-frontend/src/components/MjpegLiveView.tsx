import { daemonApi } from "../lib/api/daemon";

export default function MjpegLiveView() {
    const src = daemonApi.mjpegUrl({ fps: 10, scale: 0.6 });

    return (
        <div className="liveview">
            <div className="liveview-title">Live View</div>
            <div className="liveview-frame">
                <img src={src} alt="Live MJPEG Stream" />
            </div>
            <div className="liveview-caption">Edge stream • MJPEG • low-latency</div>
        </div>
    );
}
