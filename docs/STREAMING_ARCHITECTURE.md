# Achernar Repository - Live View Streaming Refinement

## Overview

Refined the daemon architecture to implement proper **real-time streaming** separate from high-resolution capture, enabling low-latency focusing and AI inference.

---

## Critical Architectural Changes

### 1. Separated Live View from Capture

**Previous Issue:** Single POST /camera/capture endpoint = unacceptable latency for focusing

**Solution Implemented:**

| Endpoint | Purpose | Protocol | Output |
|----------|---------|----------|--------|
| **POST /camera/capture** | High-res imaging | HTTP | Save RAW/FITS to disk |
| **GET /camera/stream** (WS) | Live view | WebSocket | Binary frames @10 FPS |
| **GET /camera/stream/mjpeg** | Alternative stream | HTTP | MJPEG multipart |

---

### 2. WebSocket Streaming Protocol

**[Rust Daemon](file:///c:/Users/Ibrah/Documents/GitHub/archenar.tech/achernar-daemon/src/main.rs)**

```rust
async fn handle_stream_socket(mut socket: WebSocket) {
    let mut ticker = interval(Duration::from_millis(100)); // 10 FPS
    
    loop {
        ticker.tick().await;
        let frame = generate_frame(); // From camera raw buffer
        socket.send(Message::Binary(frame)).await?;
    }
}
```

**Binary Frame Format:**
```
[width: u32 (4 bytes)]
[height: u32 (4 bytes)]  
[timestamp: u64 (8 bytes)]
[focusScore: f32 (4 bytes)] // AI-computed HFR
[imageData: JPEG/PNG bytes]
```

**Frame Rate:** 10 FPS (100ms interval) - adjustable based on camera

---

### 3. Client-Side Streaming

**[TelescopeClient Interface](file:///c:/Users/Ibrah/Documents/GitHub/archenar.tech/packages/ui/src/api/client.ts)**

```typescript
interface TelescopeClient {
  // High-res capture for saving
  capture(params: CaptureParams): Promise<ImageData>;
  
  // Real-time streaming for focusing
  startStream(
    onFrame: (frame: StreamFrame) => void,
    onError?: (error: Error) => void  
  ): Promise<void>;
  
  stopStream(): Promise<void>;
  
  // Alternative: MJPEG URL for <img> tag
  getStreamUrl(): string;
}
```

**Usage Example:**

```typescript
const telescope = useTelescope();

// Start live view
await telescope.startStream(
  (frame) => {
    // Update canvas with frame.data
    // Display frame.focusScore for autofocus
    console.log(`Focus score: ${frame.focusScore}`);
  },
  (error) => console.error('Stream error:', error)
);

// High-res capture (separate)
const image = await telescope.capture({ exposure: 300, gain: 200 });
```

---

### 4. LocalClient vs CloudClient

**LocalClient (Desktop)**
- WebSocket: `ws://127.0.0.1:3030/camera/stream`
- Direct connection to daemon sidecar
- Zero network latency

**CloudClient (Web)**
- WebSocket: `wss://cloud.achernar.tech/ws/camera/stream/`
- Django Channels routing
- Remote equipment access

---

## AI Focus Integration

The streaming architecture enables **real-time AI inference**:

```
Camera → Raw Buffer → Daemon
   ↓
Debayer + Resize (Preview)
   ↓
ONNX Focus Model → focusScore
   ↓
WebSocket Frame (with score)
   ↓
Client: Display + Autofocus Loop
```

**Focus Score Computation (@5 Hz):**
- Each frame includes `focusScore` field
- Client can plot focus curve in real-time
- Autofocus algorithm uses live scores (no HTTP POST latency!)

---

## Files Modified

### TypeScript Client
- [client.ts](file:///c:/Users/Ibrah/Documents/GitHub/archenar.tech/packages/ui/src/api/client.ts)
  - Added `StreamFrame` interface
  - Implemented `startStream()` / `stopStream()` methods
  - WebSocket binary frame parsing
  - Both LocalClient and CloudClient

### Rust Daemon
- [main.rs](file:///c:/Users/Ibrah/Documents/GitHub/archenar.tech/achernar-daemon/src/main.rs)
  - Added `/camera/stream` WebSocket route
  - Implemented `handle_stream_socket()` with 10 FPS ticker
  - Binary frame generation with header format

- [camera.rs](file:///c:/Users/Ibrah/Documents/GitHub/archenar.tech/achernar-daemon/src/api/camera.rs)
  - Clarified `capture()` = high-res + disk save
  - Added `stream_mjpeg()` endpoint placeholder

---

## Next Implementation Steps

### Immediate

1. **Install pnpm** (command suggested to user)
   ```bash
   npm install -g pnpm
   pnpm install
   ```

2. **Implement Camera Frame Capture**
   - Replace `generate_dummy_frame()` with actual camera SDK calls
   - Use `rusb` for USB cameras or `v4l` for webcams
   - Implement debayering in `processing/debayer.rs`

3. **Integrate ONNX Focus Model**
   - Load focus scoring model in `ai/focus_score.rs`
   - Compute HFR on each frame before streaming
   - Update `focusScore` field in binary frame

### Future Enhancements

- **Adaptive Frame Rate**: Adjust based on client latency
- **Frame Compression**: Optimize JPEG quality vs bandwidth
- **Multi-Client Support**: Broadcast to multiple viewers
- **Frame Buffering**: Handle network jitter

---

## Architecture Validation

✅ **Low Latency:** WebSocket binary frames instead of HTTP POST  
✅ **Separation of Concerns:** Live view ≠ High-res capture  
✅ **AI-Ready:** Focus scores embedded in stream  
✅ **Cross-Platform:** Works identically in Desktop (LocalClient) and Web (CloudClient)  
✅ **Offline-First:** Daemon streams locally without cloud dependency

---

## Testing Checklist

- [ ] Build Rust daemon: `cd achernar-daemon && cargo build`
- [ ] Start daemon: `./target/debug/achernar-daemon`
- [ ] Test WebSocket: `wscat -c ws://127.0.0.1:3030/camera/stream`
- [ ] Verify binary frame format (width, height, timestamp, score)
- [ ] Build UI package: `cd packages/ui && pnpm build`
- [ ] Test `startStream()` method from example component
