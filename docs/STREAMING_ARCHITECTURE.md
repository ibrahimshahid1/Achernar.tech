# Live preview and shared vision observations

This document describes the current daemon transport and the proposed runtime
integration in [Achernar v5](../achernar_v5.tex) and the
[compute-aware runtime ADR](adr/compute-aware-model-runtime.md). Architecture
snapshot: 2026-10-03. Preview delivery, science capture and measurement
publication have different contracts and cadences.

## Current endpoints and data paths

The routes in [`main.rs`](../achernar-daemon/src/main.rs) are the source of truth.
The daemon defaults to `127.0.0.1:3030`.

| Endpoint | Purpose | Current behavior |
| --- | --- | --- |
| `POST /api/v1/camera/capture` | Science or verification capture | Acquires and persists a capture bundle with provenance; returns asset metadata |
| `POST /api/v1/camera/stream/start` | Start acquisition | Starts the connected camera with an advertised format |
| `POST /api/v1/camera/stream/stop` | Stop acquisition | Stops the camera stream through the acquisition control path |
| `GET /api/v1/camera/stream` | Preview WebSocket | Encodes the latest frame as JPEG on a nominal 100 ms sender tick |
| `POST /api/v1/camera/snapshot` | Snapshot | Returns a JPEG snapshot and metadata from the current frame |
| `GET /api/v1/camera/stream/mjpeg` | Reserved compatibility route | Returns a text placeholder; multipart MJPEG is not implemented |
| `GET /ws/telemetry` | Telemetry WebSocket | Sends status telemetry and separate `VISION_UPDATE` events |

Opening a preview socket does not start camera acquisition. With no frame, or a
format the preview encoder cannot handle, the current sender emits a 20-byte
zero packet. A connected camera or configured simulator is required for images.

The camera callback populates `AppState.latest_frame`. The preview sender and
the vision loop each read that latest-frame slot. The preview sender does not
wait for FQE, and its JPEG payload is not the scientific input contract. The
vision path derives luminance from the camera frame, computes classical
measurements, and optionally evaluates FQE crops.

Preview uses a nominal 10 Hz timer. Vision is launched at 1 Hz in current
`main.rs`; the loop accepts a rate parameter and uses `MissedTickBehavior::Skip`.
Neither number is a measured latency guarantee. A camera can produce frames
more slowly than either consumer, so source-frame freshness must be distinguished
from message and observation publication rates.

## Binary preview protocol

Each binary message has a 20-byte little-endian header followed by JPEG bytes:

| Byte offset | Type | Meaning |
| --- | --- | --- |
| 0 | `u32` | Image width |
| 4 | `u32` | Image height |
| 8 | `u64` | Camera frame Unix timestamp in milliseconds |
| 16 | `u32` | Payload format; `1` means JPEG in the current sender |
| 20 | bytes | JPEG payload |

The field at byte 16 is not an HFR or `focusScore`. Clients should reject short
packets, handle zero/empty waiting packets, validate the format and use the source
timestamp when reporting fresh-frame rate. The current header has no model
identity or observation identifier; clients must not assume every preview image
has a newly computed observation attached to it.

The existing shared client in
[`packages/ui/src/api/client.ts`](../packages/ui/src/api/client.ts) still contains
legacy local routes and interprets the format field as `focusScore`. That is a
known implementation mismatch, not the current wire contract. The main web
preview component in
[`MjpegLiveView.tsx`](../achernar-frontend/src/components/MjpegLiveView.tsx)
consumes the JPEG payload after the 20-byte header. Its name does not imply the
HTTP MJPEG endpoint is implemented.

Local and cloud transport paths need separate acceptance tests; this document
does not claim identical behavior or zero transport latency. See
[TESTING.md](../TESTING.md) for direct daemon inspection without the legacy
shared-client parser.

## Shared observations and FQE shadow inference

[`vision/vision_loop.rs`](../achernar-daemon/src/vision/vision_loop.rs) publishes
[`SharedObservation`](../achernar-daemon/src/vision/shared_observation.rs) to the
application state and broadcast channel. The telemetry socket wraps each
observation as:

```json
{
  "type": "VISION_UPDATE",
  "observation": { "status": "…", "hfr_px": null, "fqe_hfr_px": null }
}
```

This is a partial envelope example, not a complete observation schema. The
important fields are classical HFR in pixels, validity/rejection reasons, used
star counts, background/trailing/cloud information, and optional FQE shadow
measurements. Missing measurements remain `null`; they must not become a
spuriously good zero or an arbitrary reciprocal focus score. HFD and fitted
FWHM also remain distinct quantities.

[`vision/fqe_runner.rs`](../achernar-daemon/src/vision/fqe_runner.rs) is the
current neural inference path. It already batches the frame's usable 64×64
grayscale crops in one ONNX Runtime call, then robustly aggregates the
predictions. It does not register an accelerator execution provider. Calibration
is associated with the exact artifact and includes per-star and post-pooling
scales. Its published fields are `fqe_hfr_px`, `fqe_sigma_log_hfr`,
`fqe_star_count`, `fqe_model_version`, and `fqe_calibrated`.

FQE is shadow-only in the current loop. Missing models or inference failures
leave its fields absent and preserve the classical measurements used by
autofocus. The checked-in random reference model is not eligible for automatic
selection. Cloud and trailing observations currently come from classical or
heuristic processing, not qualified neural alternatives.

The loop also stores an
[`ObservedFrame`](../achernar-daemon/src/vision/observed_frame.rs) envelope tying
the exact camera frame to its observation for autofocus provenance. Camera
generation checks and the publication barrier prevent a disconnected/replaced
stream from republishing an obsolete observation. Preserve these protections
when moving inference to a worker.

## Proposed compute-aware runtime integration

The following is the implementation target, not current behavior. Application
state will own a runtime with per-family handles. Discovery and qualification
will choose an immutable artifact, execution provider/options and input/batch
profile. Controllers will consume semantic observations without deciding which
GPU or model variant should run.

The FQE adapter must own versioned preprocessing and postprocessing so a larger
crop or multiscale variant cannot silently reuse the current 64×64 contract.
Preserve existing batching and measured uncertainty. Only scientific variants
validated against the same frozen splits and quality gates can be compared;
local performance measurements cannot promote a model.

For the first implementation, start with classical measurements and any already
qualified shadow FQE baseline, qualify candidates sequentially in an allowed
background window, and pin the selected candidate for the observing session.
FQE remains shadow-only in that release even if a bundle is marked promoted.
Classical operation must work without any neural artifact. Future autonomous
use additionally requires the applicable subsystem and supervisor gates;
promotion alone never enables control.

Move inference behind a bounded worker queue. Prefer a current eligible frame
over accumulating obsolete frames, preserve acquisition/control responsiveness,
and define how overload produces a skipped neural result or classical fallback.
At publication, recheck camera generation and frame freshness and atomically
associate the output with the runtime that produced it. Do not relabel a stale
prediction as a fresh result during a fallback or model transition.

The measurement budget must include preprocessing, transfer, inference,
aggregation, postprocessing, scheduling delay and publication. Qualify the
admitted set of families under preview encoding, camera acquisition and capture persistence;
individually passing models may fail together. Reserve memory and latency
headroom for the rest of the rig. There is no universal `<20 ms` focus-score
target or promise that a GPU always improves end-to-end latency.

Add structured runtime provenance to observations or an explicitly linked
envelope: family/variant, artifact hash, provider, precision, contract/input
profile, batch size, end-to-end latency, selection identity and fallback reason.
Persist selection/fallback transitions and aggregate performance windows rather
than every crop's timing forever. Any wire-protocol extension needs an explicit
schema/version and compatibility test; the current binary format field must
not be repurposed again.

Later work may add automatic demotion after sustained constraint violations.
That requires hysteresis, hold-down time and transitions at safe frame
boundaries. Opportunistic promotion during an observing session is outside the
initial design.

## Validation boundaries

The [testing guide](../TESTING.md#proposed-compute-aware-runtime-acceptance)
defines golden-vector parity, setup/warmed latency, p50/p95/p99, deadline misses,
memory accounting, realistic batch profiles, sustained/co-load tests, cache
invalidation and provider-failure behavior. The
[build guide](../CROSS_PLATFORM_BUILD.md#proposed-execution-provider-qualification-matrix)
tracks proposed provider/device qualification. A smooth preview, a green source
build or a successful ONNX session alone does not establish an accepted runtime
or an improved scientific model.
