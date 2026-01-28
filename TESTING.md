# Achernar - Testing Guide

Complete testing guide for the Achernar astronomy imaging system.

---

## Prerequisites

### Required Software

1. **Node.js 18+** - [Download](https://nodejs.org/)
2. **pnpm 8+** - Install: `npm install -g pnpm`
3. **Python 3.11+** - [Download](https://www.python.org/)
4. **Rust 1.70+** - [Install](https://rustup.rs/)
5. **Docker Desktop** - [Download](https://www.docker.com/products/docker-desktop/) (optional, for full stack)

### Windows-Specific
- **Visual Studio Build Tools** (for native modules)
- **PostgreSQL 15+** (if not using Docker)

---

## Step 1: Install Dependencies

### Install pnpm globally
```powershell
npm install -g pnpm
```

### Install workspace packages
```powershell
cd c:\Users\Ibrah\Documents\GitHub\archenar.tech
pnpm install
```

**Expected Output:**
```
Progress: resolved X, reused Y, downloaded Z, added W
packages/ui: Running postinstall script, done in Xs
Done in Xs
```

### Install Python dependencies
```powershell
pip install -r requirements.txt
```

---

## Step 2: Test Rust Daemon Build

### Build the daemon
```powershell
cd achernar-daemon
cargo build
```

**Expected Output:**
```
Compiling achernar-daemon v0.1.0
Finished dev [unoptimized + debuginfo] target(s) in X.XXs
```

### Run the daemon
```powershell
cargo run
```

**Expected Output:**
```
Starting Achernar Daemon v0.1.0
Daemon listening on http://127.0.0.1:3030
```

**Keep this terminal open!**

---

## Step 3: Test WebSocket Streaming

### Option A: Using wscat (Node.js tool)

Install wscat:
```powershell
npm install -g wscat
```

Connect to stream:
```powershell
wscat -c ws://127.0.0.1:3030/camera/stream -b
```

**Expected:** Binary data frames streaming at ~10 FPS

### Option B: Using Browser Console

Open browser console and run:
```javascript
const ws = new WebSocket('ws://127.0.0.1:3030/camera/stream');
ws.binaryType = 'arraybuffer';

ws.onmessage = (event) => {
  const data = event.data;
  const view = new DataView(data);
  
  const width = view.getUint32(0, true);
  const height = view.getUint32(4, true);
  const timestamp = Number(view.getBigUint64(8, true));
  const focusScore = view.getFloat32(16, true);
  
  console.log({ width, height, timestamp, focusScore });
};
```

**Expected Output:**
```
{ width: 640, height: 480, timestamp: 1702582800000, focusScore: 0 }
{ width: 640, height: 480, timestamp: 1702582800100, focusScore: 0 }
...
```

---

## Step 4: Test Shared UI Package

### Build the UI library
```powershell
cd packages\ui
pnpm build
```

**Expected Output:**
```
vite v5.0.8 building for production...
✓ X modules transformed.
dist/index.js  XX.XX kB
✓ built in XXXms
```

### Verify exports
```powershell
node -e "console.log(Object.keys(require('./dist/index.js')))"
```

**Expected:**
```
[ 'LocalClient', 'CloudClient', 'TelescopeProvider', 'useTelescope', 'CameraControl' ]
```

---

## Step 5: Test Backend (Docker)

### Start all services
```powershell
cd c:\Users\Ibrah\Documents\GitHub\archenar.tech
docker-compose up -d
```

**Expected Output:**
```
[+] Running 5/5
 ✔ Container achernar_postgres  Started
 ✔ Container achernar_redis     Started
 ✔ Container achernar_backend   Started
 ✔ Container achernar_celery    Started
 ✔ Container achernar_web       Started
```

### Check service health
```powershell
docker-compose ps
```

**Expected: All services `Up` and healthy**

### View backend logs
```powershell
docker-compose logs backend
```

**Expected:**
```
Daphne running on 0.0.0.0:8000
```

### Run migrations
```powershell
docker-compose exec backend python manage.py migrate
```

---

## Step 6: Access Applications

### Daemon Health Check
```powershell
curl http://127.0.0.1:3030/health
```

**Expected:** `OK`

### Backend API
Open browser: http://localhost:8000

**Expected:** Django welcome page or API root

### Frontend Dev Server
Open browser: http://localhost:5173

**Expected:** Vite dev server running

---

## Step 7: Test Live Streaming Client

Create a test HTML file to verify streaming:

**test-stream.html**
```html
<!DOCTYPE html>
<html>
<head>
  <title>Achernar Stream Test</title>
</head>
<body>
  <h1>Live View Stream Test</h1>
  <canvas id="preview" width="640" height="480"></canvas>
  <div id="stats"></div>
  
  <script type="module">
    import { LocalClient } from 'http://localhost:5173/@fs/c:/Users/Ibrah/Documents/GitHub/archenar.tech/packages/ui/dist/index.js';
    
    const client = new LocalClient(3030);
    const canvas = document.getElementById('preview');
    const ctx = canvas.getContext('2d');
    const stats = document.getElementById('stats');
    
    let frameCount = 0;
    let lastTime = Date.now();
    
    await client.startStream(
      (frame) => {
        frameCount++;
        const now = Date.now();
        const fps = 1000 / (now - lastTime);
        lastTime = now;
        
        stats.innerHTML = `
          Frame: ${frameCount}<br>
          FPS: ${fps.toFixed(1)}<br>
          Size: ${frame.width}x${frame.height}<br>
          Focus: ${frame.focusScore || 'N/A'}
        `;
        
        // Decode and display frame.data (JPEG)
        const blob = new Blob([frame.data], { type: 'image/jpeg' });
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
          ctx.drawImage(img, 0, 0);
          URL.revokeObjectURL(url);
        };
        img.src = url;
      },
      (error) => console.error('Stream error:', error)
    );
  </script>
</body>
</html>
```

---

## Troubleshooting

### TypeScript Errors

**Issue:** `Cannot find module 'react'`

**Solution:**
```powershell
cd packages\ui
pnpm install
```

### Daemon Won't Start

**Issue:** `Address already in use`

**Solution:** Kill existing process on port 3030
```powershell
# Find process
netstat -ano | findstr :3030

# Kill process (replace PID)
taskkill /PID <PID> /F
```

### Docker Issues

**Issue:** Services not starting

**Solution:** Check logs
```powershell
docker-compose logs
docker-compose down
docker-compose up -d
```

### Database Migrations

**Issue:** `no such table`

**Solution:**
```powershell
docker-compose exec backend python manage.py migrate
docker-compose exec backend python manage.py createsuperuser
```

---

## Next Steps

### 1. Implement Camera Hardware
- Add actual camera SDK integration in `achernar-daemon/src/hardware/usb_camera.rs`
- Replace `generate_dummy_frame()` with real camera capture
- Test with ZWO ASI or ToupTek cameras

### 2. Add AI Focus Scoring
- Load ONNX model in `achernar-daemon/src/ai/focus_score.rs`
- Compute HFR (Half Flux Radius) on each frame
- Update `focusScore` field in stream

### 3. Build Desktop App
- Migrate `achernar-desktop` to `packages/desktop`
- Configure Tauri sidecar for daemon bundling
- Test daemon spawn/shutdown

### 4. Migrate Web Frontend
- Move `achernar-frontend` to `packages/web`
- Update imports to use `@achernar/ui`
- Test CloudClient with Django backend

### 5. Production Deployment
- Build production Docker images
- Configure HTTPS and WSS
- Set up PostgreSQL with PostGIS
- Deploy to cloud platform

---

## Verification Checklist

- [ ] Rust daemon builds successfully
- [ ] WebSocket stream delivers frames @ 10 FPS
- [ ] Shared UI package exports all components
- [ ] Docker services all healthy
- [ ] Frontend dev server accessible
- [ ] Backend API responds
- [ ] TypeScript no errors after `pnpm install`
- [ ] Live stream displays in browser
- [ ] Focus scores populated (after AI implementation)

---

## Performance Benchmarks

**Target Metrics:**
- Stream latency: < 100ms
- Frame rate: 10 FPS (adjustable)
- Focus score computation: < 20ms per frame
- High-res capture: < 5s for 300s exposure

**Monitor with:**
```bash
# CPU usage
docker stats

# Network throughput
netstat -e
```
