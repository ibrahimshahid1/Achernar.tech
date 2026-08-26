# Phase 1 Camera Control API

This document describes the camera contract implemented by `achernar-daemon` today. It is a current-state reference, not a proposal. The direct daemon HTTP/WebSocket API is canonical; relay transports should preserve the same payload semantics where they expose the same operation.

## Control and frame pipeline

```text
HTTP or relay command
  -> shared daemon control implementation
  -> acquisition mutex for camera-mutating operations
  -> OlsCamera Rust wrapper
  -> OpenLiveStacker C++/FFI bridge
  -> simulator, Alpaca, vendor SDK, gphoto2, UVC, INDI, INDIGO, or watch-folder driver

driver frame callback
  -> owned FrameData tagged with stream_generation + driver frame_counter
  -> single-slot latest_frame mailbox
       -> live-view WebSocket and snapshot
       -> sensor-camera capture freshness gate and FITS persistence
       -> 1 Hz vision loop and telemetry/vision updates

dedicated still capture (currently the DSLR/mirrorless path)
  -> camera-authored RAW/JPEG bundle
  -> optional derived FITS
  -> transactional capture store
```

The daemon owns one camera at a time. Discovery, connect, disconnect, stream start/stop, parameter writes, autofocus, descriptor/control/format SDK reads, and the whole capture operation share one asynchronous acquisition domain. A capture holds ownership until hardware acquisition and persistence finish, so another command cannot change settings mid-exposure. Status, snapshot, and preview remain non-blocking reads of cached/owned frame state.

The stream itself continues after `start` releases the mutex. Parameter writes, connect, disconnect, and stream start queue behind the current operation. Stream stop uses a non-waiting lock: during an exposure it returns a typed `409 exposure_abort_unsupported`, and during another exclusive operation it returns `409 camera_busy`. Phase 1 has no true exposure cancellation.

## HTTP and WebSocket surface

The default HTTP prefix is `/api/v1`. The daemon also exposes telemetry at `/ws/telemetry`.

| Method | Path | Current behavior |
| --- | --- | --- |
| `GET` | `/api/v1/drivers` | Reports drivers compiled/enabled in this daemon build. |
| `POST` | `/api/v1/discover` | Operator-triggered scan of local ASI, SVBony, ToupTek, UVC, and gphoto2 buses. Network transports are not scanned. A connected camera causes all probes to be skipped. |
| `POST` | `/api/v1/connect` | Connects equipment; use `equipment_type: "camera"` for this contract. |
| `POST` | `/api/v1/disconnect` | Disconnects one equipment type or all equipment and returns a JSON result. |
| `GET` | `/api/v1/status` | Lightweight cached equipment/lifecycle status; it does not call the camera SDK. |
| `GET` | `/api/v1/camera` | Frontend-ready camera descriptor. |
| `GET` | `/api/v1/camera/formats` | Current connection's selectable acquisition formats. |
| `GET` | `/api/v1/camera/parameters` | Current camera's supported control descriptors and values. |
| `POST` | `/api/v1/camera/parameters` | Sets one control and returns its hardware readback. |
| `POST` | `/api/v1/camera/stream/start?format_index=N` | Starts the selected format. If omitted, the daemon prefers MJPEG, then another previewable format, then the first advertised format. An explicit index is always honored exactly. |
| `POST` | `/api/v1/camera/stream/stop` | Stops the stream. |
| `POST` | `/api/v1/camera/abort` | Compatibility stream-stop alias; idempotent while disconnected. It does **not** abort an exposure. |
| `GET` + upgrade | `/api/v1/camera/stream` | Binary WebSocket live view. It does not start the hardware stream. |
| `POST` | `/api/v1/camera/snapshot` | Returns the latest previewable frame as base64 JPEG. |
| `GET` | `/api/v1/camera/stream/mjpeg` | Placeholder text only; this is not a multipart MJPEG endpoint. |
| `POST` | `/api/v1/camera/capture` | Takes and commits one physical exposure. |
| `GET` + upgrade | `/ws/telemetry` | Text JSON telemetry every second plus `VISION_UPDATE` events. |

### Discovery and connection

Use `/drivers` as the source of truth for what the running build can use. Do not infer support from a hard-coded list.

Each discovered local camera includes `driver_id`, `driver_label`, driver-local `index`, vendor `name`, a scan-scoped `selection_key` such as `uvc:0`, nullable `hardware_id`, and `transport: "local"`. Send `driver_id` and `index` back to connect. Neither the key nor index is a persistent physical identity.

`POST /api/v1/connect` accepts:

```json
{
  "equipment_type": "camera",
  "driver": "svb",
  "option": null,
  "index": 0
}
```

`equipment_type` defaults to `camera`; `index` defaults to `0`. The accepted camera driver IDs/aliases in the handler are `sim`, `alpaca`, `asi`, `svb`, `toup`, `wdir`/`watchdir`, `gphoto2`/`dslr`, `uvc`/`webcam`, `indi`, and `indigo`. `option` is driver-specific. A successful camera connection returns:

```json
{
  "status": "connected",
  "connection_id": "uuid",
  "driver": "svb",
  "camera": "vendor-reported name",
  "index": 0,
  "cameras_found": ["vendor-reported name"]
}
```

`connection_id` changes on every successful connection. `device_index` is scoped to one driver's current enumeration and is not a durable hardware identity.

Disconnect accepts an optional body such as `{"equipment_type":"camera"}`. A missing body defaults to `all`; disconnecting an already-disconnected camera is successful and idempotent.

```json
{ "status": "disconnected", "equipment_type": "camera" }
```

## Camera descriptor

`GET /api/v1/camera` is the single discovery/configuration payload for a camera UI:

```ts
type CameraDescriptor = {
  connected: boolean;
  connection_id: string | null;
  name: string | null;
  driver_id: string | null;
  device_index: number | null;
  hardware_id: null;
  activity: CameraActivity;
  streaming: boolean;
  active_format_index: number | null;
  operation_id: string | null;
  exposure_progress: number | null; // estimated fraction, 0..1
  temperature_c: number | null;
  cooler_on: boolean | null;
  last_error: string | null;
  capabilities: {
    supports_stream: boolean;
    supports_dedicated_still: boolean;
    supports_exposure_abort: false;
    supports_cooling: boolean;
    supports_temperature: boolean;
    supports_hardware_binning: boolean;
    supports_roi: boolean;
  };
  formats: FormatInfo[];
  controls: CameraControl[];
  latest_frame: {
    stream_generation: number;
    counter: number;
    timestamp: number; // Unix seconds
    width: number;
    height: number;
    format: string;
  } | null;
};
```

When no camera object exists, this endpoint still returns a descriptor with empty `formats` and `controls`; `activity` may be `error` after a failed connection. When a camera is present, the endpoint calls the driver to read formats and controls. It returns `409 camera_busy` rather than entering the SDK during another camera operation. Use `/status` or telemetry for frequent lifecycle polling.

### Lifecycle/activity

`CameraActivity` has these stable snake-case wire values:

```text
disconnected | connecting | idle | streaming | exposing | downloading | processing | disconnecting | error
```

Current transitions are:

- connect: `connecting -> idle` or `error`;
- stream start: `idle -> streaming`;
- capture: `idle|streaming -> exposing -> processing -> idle|streaming`;
- stream stop: `streaming -> idle` or `error`;
- disconnect: `* -> disconnecting -> disconnected`.

`downloading` is reserved because the current native still call combines shutter and download into one blocking operation. `processing` covers FITS generation and durable persistence after bytes arrive. `exposure_progress` exists only while `activity == "exposing"`; it is a wall-clock estimate based on the requested exposure duration, is clamped to `0..1`, and does not represent download or persistence progress. `operation_id` remains the capture ID until the operation finishes.

`GET /api/v1/status` uses different legacy field names (`status`, `temperature`, `cooling`, `progress`) and returns `camera: null` when no successful camera connection exists. `/ws/telemetry` uses that lightweight shape and additionally includes the latest frame's generation, counter, dimensions, and format.

## Acquisition formats

Both `GET /api/v1/camera/formats` and `descriptor.formats` return:

```ts
type FormatInfo = {
  index: number;
  id: string;              // "format-<index>"
  format: string;          // driver/bridge enum debug name
  width: number;           // pixels
  height: number;          // pixels
  bin: number;
  roi_num: number;
  roi_den: number;
  framerate: number;       // frames/second
  previewable: boolean;
  capture_supported: boolean;
  mode: "liveview" | "still";
};
```

Send the returned `index` unchanged to `stream/start`; do not reconstruct a format from dimensions. The index and generated ID are scoped to the current camera connection. Starting without an index selects `0`.

`mode` is currently derived from the driver-advertised frame rate: positive values and OLS's `-1` “unknown rate” sentinel are `liveview`; zero is `still`. `previewable` is true only for `Mjpeg`, `Rgb24`, `Mono8`, and `Raw8`. `capture_supported` is true for those formats plus `Mono16` and `Raw16`. Stream start itself does not reject an unpreviewable or capture-unsupported format; the caller must use these flags for its intended operation.

Starting an already-running stream is idempotent when no new format is requested or the requested index is already active. Selecting a different format while streaming returns `409`; stop first.

A new start returns `status`, `already_running: false`, `format_index`, `format`, dimensions, bin/ROI values, and `mode`. The idempotent response contains only `status`, `already_running: true`, and `format_index`. Both stop routes return:

```json
{ "status": "stopped", "was_streaming": true, "exposure_aborted": false }
```

## Camera controls

The control surface is generated from the connected driver. A `CameraControl` is:

```ts
type CameraControl = {
  name: string;  // stable snake_case daemon key
  label: string;
  group: "acquisition" | "cooling" | "image" | "advanced";
  unit?: "ms" | "degC" | "K" | "percent" | "driver_units" | "ISO";
  kind: "bool" | "number" | "msec" | "percent" |
        "kelvin" | "celsius" | "selection";
  read_only: boolean;
  min: number;
  max: number;
  step: number;
  default: number;
  current: number;
  choices: string[];
  current_value?: number;
};
```

The daemon may recognize more controls than a particular camera exposes. Always render the returned array. Current stable names are:

```text
auto_exposure, auto_wb, auto_focus, exposure, white_balance, wb_red, wb_blue,
focus, gain, gamma, brightness, contrast, temperature, cooler_target, cooler_on,
fan_on, cooler_power, average_bin, black_level, conv_gain_hcg, conv_gain_hdr,
low_noise, high_fullwell, iso, shutter, viewfinder, capture_target, keep_images,
capture_delay, zoom, live_stretch
```

`capturetarget` is accepted as a write alias, but descriptors always emit `capture_target`.

### Units and value semantics

| Control names | `unit` | Wire value |
| --- | --- | --- |
| `exposure`, `capture_delay` | `ms` | milliseconds |
| `temperature`, `cooler_target` | `degC` or `K` | derived from the driver's physical option type |
| `cooler_power` | `percent` | driver-reported percent |
| `gain`, `black_level`, `brightness`, `contrast`, `gamma`, `white_balance`, `wb_red`, `wb_blue` | `driver_units` | opaque driver units |
| `iso` | `ISO` | selection index in `current`; parsed ISO speed in `current_value` when possible |
| all others | omitted | interpret from `kind`, labels, and bounds |

For `kind == "selection"`, `current` is an integer index and `choices[current]` is the display label. `shutter.current_value` is parsed seconds and `iso.current_value` is parsed ISO speed when the label is numeric. `current_value` is display-only; send the selection index.

This is intentionally different from the capture request: generic control `exposure` is milliseconds, while `POST /camera/capture` uses seconds.

Set one control with:

```json
{ "name": "gain", "value": 120 }
```

The backend rejects unknown/unsupported/read-only controls, non-finite or out-of-range values, booleans other than `0`/`1`, and non-integer selection values. It does not enforce alignment to `step`. Drivers may round or clamp, so the response readback is authoritative:

```json
{
  "status": "ok",
  "name": "gain",
  "requested_value": 120,
  "value": 120,
  "applied_value": 120
}
```

## Streaming, preview, and frame identity

Starting a new stream clears all previous preview and vision observations and advances a daemon-owned `stream_generation`. Stop, disconnect, and connection reset also invalidate observations. The callback checks generation both before copying and before publishing, so a late callback from an old camera/stream is discarded.

`latest_frame` is a single overwrite slot, not a queue or history. A frame's identity is the pair `(stream_generation, counter)`:

- `stream_generation` is daemon-owned and protects reconnect/restart boundaries;
- `counter` is driver-owned, signed 32-bit, and may reset or wrap;
- frame/snapshot/descriptor timestamps are floating-point Unix seconds.

### Live-view WebSocket

`GET /api/v1/camera/stream` upgrades to a WebSocket. The server samples and sends the current mailbox at roughly 10 Hz, so it can repeat the same camera frame. Each binary message is:

| Byte range | Type | Meaning |
| --- | --- | --- |
| `0..4` | little-endian `u32` | width |
| `4..8` | little-endian `u32` | height |
| `8..16` | little-endian `u64` | Unix timestamp in milliseconds |
| `16..20` | little-endian `u32` | `1`, meaning JPEG/MJPEG |
| `20..` | bytes | JPEG payload |

When no previewable frame exists, the server sends exactly 20 zero bytes. MJPEG input passes through; RGB24 and 8-bit mono/raw are JPEG-encoded at quality 80. Other input formats produce the zero header. The packet does not include generation or counter, so deduplicate repeated preview packets by timestamp if needed and use the descriptor/snapshot/telemetry for stronger identity.

`POST /api/v1/camera/snapshot` returns the current frame once:

```json
{
  "status": "ok",
  "format": "JPEG",
  "mime_type": "image/jpeg",
  "width": 1280,
  "height": 960,
  "frame_counter": 42,
  "timestamp": 1786590000.25,
  "data_base64": "..."
}
```

## Capture

Request:

```ts
type CaptureRequest = {
  exposure: number;         // required, seconds
  gain?: number;            // non-negative integer: gain or requested ISO
  offset?: number;          // applied through black_level and read back, or rejected
  binning?: number;         // must match the active hardware format, or is rejected
  filter?: string;          // archive metadata only; no wheel command
  temperature?: number;     // best-effort cooler target on streamed sensor path
  session_id?: string;
  capture_type?: string;    // defaults to "light"; not validated in Phase 1
};
```

The handler requires a finite positive `exposure`. It does not yet validate that `capture_type` belongs to the documented light/dark/flat/bias/focus/preview/verification set, so callers should validate that field.

There are two acquisition paths:

1. A camera with `supports_dedicated_still == true` uses its blocking still API and does not sample `latest_frame`. It preserves all camera-authored RAW/JPEG files, attempts a derived FITS without making derivation failure fatal, commits the bundle, and only then asks the camera to delete its originals.
2. Other sensor cameras capture from the active stream. The stream must already be running. The daemon verifies that requested binning matches the active format, disables auto exposure, applies exposure/gain/offset, reads the hardware values back, discards the first post-write completion because it may already have been in flight, and accepts only a later counter from the same stream generation. With an empty mailbox, the first observed counter establishes the baseline and the next counter is accepted. Failure to prove freshness returns `504` after `(3 * applied_exposure + 20)` seconds, clamped to 20–900 seconds.

For the streamed path, continuous exposure is written in milliseconds and read back into seconds. Gain uses a continuous `gain` control when available or the nearest enumerated ISO. An offset request requires a writable `black_level` control. A binning mismatch returns `422` instead of being recorded as false metadata. A `temperature` request is attempted only when the camera exposes `cooler_target`, and rejection is a warning rather than a capture failure.

Capture runs in a daemon-owned task. If the browser closes or the HTTP request is abandoned during an exposure, the camera transaction continues to completion, keeps exclusive ownership, and commits or records its failure rather than being cancelled halfway through.

A successful response represents a committed capture:

```ts
type CaptureResult = {
  capture_id: string;       // attempt ID, allocated before hardware is touched
  id: string;               // durable frame row ID
  session_id?: string;
  status: "committed";
  width: number;
  height: number;
  relative_dir: string;
  assets: Array<{
    role: string;           // native_raw, camera_jpeg, sensor_fits, ...
    media_type: string;
    relative_path: string;
    byte_length: number;
    sha256: string;
  }>;
  path?: string;            // deprecated canonical-asset compatibility field
  metadata: {
    exposure: number;       // applied seconds
    requested_exposure: number;
    exposure_error_stops: number;
    gain: number;
    iso?: number;
    shutter?: string;
    temperature: number | null;
    temperature_state: "measured" | "estimated" | "missing";
    timestamp: string;      // exposure start, RFC 3339
    completed_at: string;   // RFC 3339
    warnings?: string[];
  };
};
```

Use `assets`, not deprecated `path`; one exposure can produce multiple canonical artifacts. Show `warnings` to the operator. When temperature is unavailable, the current serializer emits `temperature: null` and `temperature_state: "missing"`.

## Errors

Newer camera-control handlers use this JSON envelope:

```json
{
  "error": {
    "code": "camera_not_connected",
    "message": "No camera is connected"
  }
}
```

| HTTP | Stable code | Emitted for |
| --- | --- | --- |
| `400` | `invalid_camera_format` | Out-of-range stream format index. |
| `400` | `unknown_camera_control` | Unrecognized control name. |
| `409` | `camera_stream_already_running` | Attempt to switch format without stopping. |
| `409` | `camera_busy` | Descriptor/control/format/stop request conflicts with an exclusive operation. |
| `409` | `exposure_abort_unsupported` | Stream stop requested during an exposure that cannot be interrupted. |
| `409` | `camera_reconnect_required` | A stop failure left the native stream state uncertain. |
| `409` | `camera_control_read_only` | Write to read-only control. |
| `412` | `camera_not_connected` | Operation requires a connected camera. |
| `422` | `unsupported_camera_control` | Recognized control absent on this camera. |
| `422` | `camera_control_out_of_range` | Non-finite/out-of-bounds value. |
| `422` | `invalid_camera_control_value` | Invalid bool or selection representation. |
| `502` | `camera_has_no_formats` | Connected camera reports no formats. |
| `502` | `camera_hardware_error` | Driver format/control/start/stop failure. |

Connect, disconnect, descriptor, capture, format, control, and stream lifecycle failures use the structured envelope. Snapshot still uses bare `404`/`415`, and older non-camera equipment endpoints retain their legacy status-only errors. A generic client should therefore still fall back to HTTP status when no JSON envelope is present. `last_error` in descriptor/status provides additional detail for lifecycle and capture failures.

## Relay parity

The daemon cloud relay recognizes `drivers.capabilities`, `equipment.status`, `equipment.connect`, `equipment.disconnect`, `camera.descriptor`, `camera.parameters`, `camera.formats`, `camera.set_parameter`, `camera.start_stream`, `camera.stop_stream`/`camera.abort`, `camera.capture`, and `camera.snapshot`. Camera connect/disconnect, descriptor, format/control reads, start, stop, parameter writes, and capture delegate to the same implementations as direct HTTP. Relay errors remain strings rather than the direct HTTP envelope. Local bus discovery is not exposed through relay.

## Phase 1 limitations and frontend guidance

- There is one daemon-owned camera and one active format. Phase 1 cannot own a guider and imaging camera simultaneously.
- `hardware_id` is always `null`; key UI state by `connection_id`, not `device_index` or camera name.
- There is no exposure abort. Camera `abort`/stream stop reports `exposure_aborted: false` and returns immediately with `409 exposure_abort_unsupported` during an exposure.
- There is no filter-wheel action in this camera API. `CaptureRequest.filter` is provenance metadata only.
- Capture-time offset is applied through `black_level`; unsupported offset is rejected. Capture-time binning is a consistency assertion against the selected hardware format—select binning/ROI through a returned format index first.
- Prefer a format with `mode == "liveview" && previewable` for framing/focus. For sensor capture, also require `capture_supported`; a dedicated-still camera does not require an active stream for capture.
- Do not hard-code drivers, controls, units, choices, formats, or capabilities. Read `/drivers` and the camera descriptor, and send returned indexes unchanged.
- Render selection labels, but submit selection indexes. Reconcile every write with the returned hardware value.
- Treat activity and streaming as separate facts. Disable conflicting mutations optimistically, but rely on backend serialization for correctness.
- Use `/status` or telemetry for frequent progress/status updates. Refresh the descriptor after connect, stream-format changes, and parameter writes.
- The preview WebSocket repeats a single-slot latest frame and has no backpressure/history semantics. Revoke replaced browser object URLs and deduplicate when downstream work must run once per camera frame.
- The MJPEG URL is not implemented, and 16-bit/raw formats are not browser-preview encoded.
- Local discovery is explicit, sequential, limited to local buses, and skipped while connected. Network drivers require an explicit `option` supplied at connect time.
- The daemon API has permissive CORS and no authentication layer; its default bind host is loopback. Do not expose it on an untrusted network without an authenticated boundary.

## Implementation anchors

- Routes and WebSocket packet framing: `achernar-daemon/src/main.rs`
- Descriptor, controls, formats, stream, snapshot, capture, freshness, and errors: `achernar-daemon/src/api/camera.rs`
- Global connection/status endpoints: `achernar-daemon/src/api/mod.rs`
- Structured camera errors: `achernar-daemon/src/api/api_error.rs`
- Lifecycle, acquisition mutex, generation invalidation, and frame mailbox: `achernar-daemon/src/state.rs`
- Driver/FFI wrapper and callback generation checks: `achernar-daemon/src/hardware/camera.rs`
- Discovery/build capability reports: `achernar-daemon/src/hardware/discovery.rs`, `achernar-daemon/src/hardware/capabilities.rs`
- Relay action mapping: `achernar-daemon/src/relay_commands.rs`
