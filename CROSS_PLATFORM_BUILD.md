# Cross-Platform Build Instructions

## ✅ Fixed Issues

### Rust (Cargo.toml)
- **Linux-only deps** now isolated: `indi`, `v4l` only compile on Linux
- **Windows support** added: `ascom-alpaca` for telescope/camera control
- **Modern versions**: axum 0.7, tokio 1.35
- **Optional features**: Hardware support is now opt-in via features

### JavaScript (package.json)
- **Removed**: `fits.js` / `fitsjs` (unmaintained, 404 errors)
- **Added**: `@observerly/fits` (modern, maintained FITS library)

---

## 🚀 Build on Windows

### 1. Build Rust Daemon (Windows)

```powershell
cd achernar-daemon

# Basic build (no optional hardware)
cargo build

# With ASCOM support (Windows telescopes)
cargo build --features ascom-support

# Run the daemon
cargo run
```

**Expected Output:**
```
Starting Achernar Daemon v0.1.0
Daemon listening on http://127.0.0.1:3030
```

### 2. Build UI Package

```powershell
cd ..\packages\ui

# Install dependencies (will now work without FITS errors)
pnpm install

# Build the library
pnpm build
```

---

## 📦 Optional Hardware Features

The daemon now has **optional** hardware support:

### Windows
```powershell
# Build with ASCOM Alpaca support
cargo build --features ascom-support
```

### Linux
```powershell
# Build with INDI protocol support
cargo build --features indi-support

# Build with both
cargo build --features indi-support
```

### No Hardware (Testing)
```powershell
# Build without any hardware dependencies (works on all platforms)
cargo build
```

This allows you to:
- ✅ Build on Windows without Linux deps
- ✅ Test the WebSocket streaming without hardware
- ✅ Add hardware support only when needed

---

## 🔧 FITS File Handling

**Old (broken):**
```typescript
import FITS from 'fits.js'; // 404 error
```

**New (working):**
```typescript
import { FITS } from '@observerly/fits';

// Parse FITS file
const fits = new FITS(arrayBuffer);
const header = fits.getHeader();
const data = fits.getDataUnit();
```

---

## ⚠️ Known Platform Differences

| Feature | Windows | Linux | macOS |
|---------|---------|-------|-------|
| **ASCOM Alpaca** | ✅ | ❌ | ❌ |
| **INDI Protocol** | ❌ | ✅ | ✅ |
| **USB (rusb)** | ✅ | ✅ | ✅ |
| **Serial (serialport)** | ✅ | ✅ | ✅ |
| **V4L Webcams** | ❌ | ✅ | ❌ |

**Recommendation:** Build without features on Windows, test the streaming architecture, then add `--features ascom-support` when you connect real hardware.

---

## 📝 Next Steps

1. **Build and test:**
   ```powershell
   cd achernar-daemon
   cargo build
   cargo run
   ```

2. **Install UI deps:**
   ```powershell
   cd ..\packages\ui
   pnpm install
   pnpm build
   ```

3. **Test WebSocket stream:**
   ```powershell
   # Should work without any hardware connected
   wscat -c ws://127.0.0.1:3030/camera/stream -b
   ```

The daemon will send dummy frames for testing. You can implement actual camera integration later using ASCOM on Windows.
