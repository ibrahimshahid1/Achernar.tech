# Achernar - Unified Astronomy Imaging and Control System

**A comprehensive AI-enhanced platform for amateur and professional astronomers combining offline-first design with intelligent cloud capabilities.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://reactjs.org/)

## 🌟 Overview

Achernar is a unified astronomy imaging and control system that democratizes high-quality astrophotography and research through a hybrid AI architecture. The system combines edge computing for real-time operations with cloud computing for collaboration, archiving, and large-scale model training.

**Key Innovation**: Deploy quantized ONNX models on the edge for sub-millisecond decision making (focus, cloud detection) while utilizing a novel tri-modal embedding space (Image, Time-Series, Spectra) in the cloud for advanced recommendation engines and anomaly detection.

## ✨ Key Features

### Hardware & Control
- **Universal Hardware Support**: INDI, ASCOM Alpaca, direct USB/Serial control
- **Real-Time Processing**: Hardware-accelerated debayering and H.264/MJPEG encoding
- **Professional Calibration**: Automatic dark, flat, and dark-flat frame integration
- **Plate Solving**: Local astrometry with blind solve capability

### AI/ML Intelligence

#### Edge AI (Real-Time, <50ms)
- **Focus Scoring**: ViT-Tiny ONNX models with uncertainty quantification
- **Cloud Detection**: MobileNetV3-Small binary classifier with confidence thresholds
- **Trailing Detection**: CNN-based guiding issue detection
- **Drift Tracking**: Kalman filter + ARIMA for predictive refocusing
- **Settings Recommender**: Contextual bandit (LinUCB) for exposure/gain optimization

#### Cloud AI (Heavy Compute)
- **Tri-Modal Embeddings**: Unified 256-D space for Image ↔ Light Curve ↔ Spectrum ↔ Metadata
- **Target Ranking**: Optimization-based scheduling (OR-Tools) with SNR prediction
- **Quality Scoring**: Calibrated regression with SHAP explainability
- **Anomaly Detection**: Isolation Forest / One-Class SVM for transient discovery
- **Active Learning**: Continuous improvement from user feedback

### Design Principles
- ✅ **Scientific Integrity First**: Never alter raw science data with ML
- ✅ **Uncertainty-Aware**: All predictions include confidence scores
- ✅ **Explainable Decisions**: Critical recommendations show reasoning (SHAP, attention maps)
- ✅ **Graceful Degradation**: System works without AI; classical methods as fallback
- ✅ **Privacy-Preserving**: Optional federated learning for collaborative improvement
- ✅ **Continuous Learning**: User feedback loops improve models over time

### Platform Features
- **Offline-First Design**: Full functionality without internet connectivity
- **Cloud Synchronization**: Automatic session sync when connectivity is restored
- **Multi-Platform Support**: Desktop app (Tauri), web interface, mobile PWA
- **Remote Operations**: Cloud-mediated IoT relay + WebRTC for low-latency control
- **Field Deployable**: Runs on Raspberry Pi, Jetson Nano, Intel NUC, or laptops

## 🏗️ Architecture

### System Layers
```mermaid
graph TB
    subgraph "Equipment Layer"
        CAM[Cameras<br/>ZWO/ToupTek/DSLR]
        MOUNT[Mounts<br/>INDI/ASCOM/Serial]
        FOCUS[Focusers<br/>INDI/Moonlite]
        FILTER[Filter Wheels<br/>INDI/USB]
        SENSOR[Sensors<br/>Arduino/BME280]
    end
    
    subgraph "Achernar Daemon (Rust + C++)"
        HAL[Hardware Abstraction<br/>libusb/INDI/tokio]
        IPE[Image Processing<br/>OpenCV/libraw]
        EDGE_AI[Edge AI<br/>ONNX Runtime]
        CTRL[Control Systems<br/>State Machine]
        STORE[Local Storage<br/>SQLite + FITS]
        SERVER[Daemon Server<br/>Axum/WebSocket]
    end
    
    subgraph "Network Layer"
        WIFI[WiFi AP<br/>hostapd]
        VPN[VPN/Tailscale]
        INET[Internet]
    end
    
    subgraph "Client Devices"
        DESKTOP[Desktop App<br/>Tauri 2.0]
        WEB[Web Browser<br/>React PWA]
        TABLET[Tablet<br/>Touch UI]
    end
    
    subgraph "Cloud Platform (AWS/GCP)"
        API[API Gateway<br/>Django + Channels]
        CLOUD_AI[AI/ML Services<br/>PyTorch/FAISS]
        PROC[Processing<br/>Celery/Redis]
        DB[(PostgreSQL<br/>S3/GCS)]
    end
    
    CAM --> HAL
    MOUNT --> HAL
    FOCUS --> HAL
    FILTER --> HAL
    SENSOR --> HAL
    
    HAL --> IPE
    IPE --> EDGE_AI
    EDGE_AI --> CTRL
    CTRL --> STORE
    STORE --> SERVER
    
    SERVER --> WIFI
    WIFI --> VPN
    VPN --> INET
    
    WIFI --> DESKTOP
    WIFI --> WEB
    WIFI --> TABLET
    
    INET --> API
    API --> CLOUD_AI
    API --> PROC
    PROC --> DB
    CLOUD_AI --> DB
```

### Edge vs Cloud: Division of Labor

| **Edge (Must Be Fast: <50ms)** | **Cloud (Heavier Compute)** |
|--------------------------------|----------------------------|
| Focus/cloud classifiers | Target ranking/scheduling |
| Noise estimation | Tri-modal embedding training |
| Kalman drift tracking | Anomaly detection at scale |
| Contextual bandits (settings) | Global personalization |
| Classical stats (HFR, background) | Model retraining |
| Uncertainty quantification | A/B testing & canary deployments |
| Graceful fallback to rules | Heavy analytics & SHAP |
| | Federated learning aggregation |

## 📁 Repository Structure
```
achernar.tech/
├── packages/
│   ├── ui/                     # Shared React components (@achernar/ui)
│   │   ├── components/         # Reusable UI components
│   │   ├── hooks/             # Custom React hooks
│   │   └── types/             # TypeScript interfaces
│   ├── web/                    # Cloud frontend (Vite + React)
│   │   └── src/
│   │       ├── pages/         # Route pages
│   │       └── services/      # API clients
│   └── desktop/                # Desktop app (Tauri 2.0)
│       ├── src/               # React frontend
│       └── src-tauri/         # Rust backend
│
├── achernar-daemon/            # Rust daemon (core system)
│   ├── src/
│   │   ├── hardware/          # INDI/USB/Serial drivers
│   │   ├── imaging/           # C++ FFI, OpenCV processing
│   │   ├── ai/                # ONNX Runtime inference
│   │   ├── control/           # State machine & sequences
│   │   ├── storage/           # SQLite + FITS I/O
│   │   └── server/            # Axum web server + WebSocket
│   ├── models/                # ONNX model files
│   └── Cargo.toml
│
├── achernar_backend/           # Django backend
│   ├── api/                   # REST API (DRF)
│   ├── realtime/              # Django Channels (WebSocket)
│   ├── ml/                    # AI/ML cloud services
│   │   ├── embeddings/        # Tri-modal encoder training
│   │   ├── recommender/       # Bayesian hierarchical models
│   │   └── anomaly/           # Isolation Forest / autoencoders
│   ├── processing/            # Celery tasks
│   └── manage.py
│
├── ml/                         # ML training pipeline
│   ├── data/                  # Feature store (Parquet/DuckDB)
│   ├── training/              # PyTorch Lightning modules
│   │   ├── focus/             # Focus score regressor
│   │   ├── cloud/             # Cloud classifier
│   │   └── embeddings/        # Tri-modal CLIP-style
│   ├── export/                # QAT → ONNX export
│   └── notebooks/             # Exploratory analysis
│
├── docs/
│   ├── API.md                 # API documentation
│   ├── HARDWARE.md            # Hardware setup guide
│   ├── AI_ARCHITECTURE.md     # ML system design
│   └── DEPLOYMENT.md          # Production deployment
│
├── docker-compose.yml          # Development environment
├── pnpm-workspace.yaml         # Monorepo configuration
├── achernar.toml.example       # Daemon configuration template
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and **pnpm** 8+
- **Python** 3.11+
- **Rust** 1.70+ (with `cargo`)
- **Docker** and **Docker Compose**
- **PostgreSQL** 15+ (or use Docker)
- **Redis** 6+ (or use Docker)

### Development Setup

1. **Clone the repository**
```bash
   git clone https://github.com/yourusername/achernar.tech.git
   cd achernar.tech
```

2. **Install dependencies**
```bash
   # Install pnpm if needed
   npm install -g pnpm
   
   # Install all workspace packages
   pnpm install
   
   # Install Python dependencies
   pip install -r requirements.txt
   
   # Install Rust (if not already installed)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

3. **Set up environment**
```bash
   cp .env.example .env
   cp achernar.toml.example achernar.toml
   # Edit .env and achernar.toml with your configuration
```

4. **Start services with Docker**
```bash
   docker-compose up -d
   # This starts PostgreSQL, Redis, and Celery workers
```

5. **Run database migrations**
```bash
   python manage.py migrate
   python manage.py createsuperuser  # Create admin account
```

6. **Download AI models** (optional for development)
```bash
   cd achernar-daemon/models
   ./download_models.sh  # Downloads pre-trained ONNX models
```

7. **Start the development servers**
```bash
   # Terminal 1: Django backend
   python manage.py runserver
   
   # Terminal 2: Rust daemon
   cd achernar-daemon
   cargo run --release
   
   # Terminal 3: Web frontend
   cd packages/web
   pnpm dev
   
   # Terminal 4: Celery workers
   celery -A achernar_backend worker -l info
```

8. **Access the applications**
   - **Web frontend**: http://localhost:5173
   - **Django backend**: http://localhost:8000
   - **Django admin**: http://localhost:8000/admin
   - **Daemon API**: http://localhost:3030/api/v1
   - **WebSocket telemetry**: ws://localhost:3030/ws/telemetry

### Building the Daemon
```bash
cd achernar-daemon
cargo build --release

# The binary will be at: target/release/achernar-daemon
```

### Building the Desktop App
```bash
cd packages/desktop
pnpm tauri build

# Outputs:
# - Linux: .deb, .AppImage
# - macOS: .dmg, .app
# - Windows: .msi, .exe
```

### Running Tests
```bash
# Rust tests
cd achernar-daemon
cargo test

# Python tests
pytest

# Frontend tests
pnpm test
```

## 🛠️ Technology Stack

### Frontend
- **React 18** with TypeScript
- **Vite** for fast HMR and builds
- **TanStack Query** for data fetching and caching
- **Zustand** for transient UI state
- **Tailwind CSS** for styling
- **d3.js** for astronomical visualizations
- **fits.js** for FITS file handling
- **Three.js** for 3D star field rendering

### Backend
- **Django 4.2+** with REST Framework
- **Django Channels** for WebSocket support
- **Celery** for background tasks and ML training
- **PostgreSQL 15+** with full-text search
- **Redis Cluster** for caching and message broker
- **S3/GCS** for object storage with CDN
- **FAISS** for vector similarity search (IVF-PQ)

### Daemon (Rust)
- **Axum** web framework
- **tokio** async runtime
- **tokio-tungstenite** for WebSocket
- **libindi-rust** for INDI protocol
- **rusb** for USB camera support
- **tokio-serial** for mount control
- **OpenCV** (via FFI) for image processing
- **ONNX Runtime** for AI inference
- **SQLite 3.40+** with WAL mode
- **fitsio-rs** for FITS I/O

### Desktop App
- **Tauri 2.0** (Rust + WebView)
- **React** (shared from `@achernar/ui`)
- **Process management** for daemon sidecar

### AI/ML Stack
- **PyTorch Lightning** for training
- **ONNX Runtime** for edge inference
- **LightGBM/XGBoost** for gradient boosting
- **FAISS** for embedding retrieval
- **OR-Tools** for optimization
- **MLflow** for experiment tracking
- **DVC** for data versioning
- **Optuna** for hyperparameter tuning
- **SHAP** for explainability

## 📊 AI/ML System Design

### Edge Models (Real-Time)

| Model | Purpose | Architecture | Latency | Uncertainty |
|-------|---------|--------------|---------|-------------|
| Focus Scorer | HFR prediction | ViT-Tiny (INT8) | <50ms | MC Dropout |
| Cloud Classifier | Clear vs cloudy | MobileNetV3-Small | <30ms | Temperature scaling |
| Trailing Detector | Guiding issues | CNN on crops | <40ms | Ensemble |
| Settings Bandit | Exposure/gain | LinUCB | <10ms | UCB bounds |

### Cloud Models (Heavy Compute)

| Model | Purpose | Architecture | Update Frequency |
|-------|---------|--------------|------------------|
| Tri-Modal Embeddings | Unified representation | ViT + Transformer + 1D ResNet | Weekly |
| Target Ranker | Tonight's schedule | LightGBM + OR-Tools | Daily |
| Quality Scorer | Frame curation | Calibrated regression | Daily |
| Anomaly Detector | Transient discovery | Isolation Forest | Weekly |

### Training Pipeline
```mermaid
graph LR
    A[Raw Data] --> B[Feature Store<br/>Parquet/DuckDB]
    B --> C[Weak Labeling<br/>Snorkel-style]
    C --> D[PyTorch Lightning<br/>Mixed Precision]
    D --> E[QAT Export<br/>ONNX INT8]
    E --> F[Edge Deployment<br/>ONNX Runtime]
    F --> G[User Feedback<br/>Active Learning]
    G --> A
```

### Metrics We Track

#### Edge Models
- **Focus Regressor**: MAE vs HFR, Spearman ρ, latency, calibration error (ECE)
- **Cloud Classifier**: AUROC, precision@high-recall, F1 score
- **Trailing Detector**: Precision/recall, false positive rate

#### Recommender Systems
- **Exposure/Gain**: Off-policy eval (IPS/DR), uplift vs baseline, user acceptance rate
- **Filter Choice**: Accuracy on held-out sessions, regret, user override rate

#### System-Level
- **User Engagement**: Session duration, repeat usage, feature adoption
- **Data Quality**: % of subs kept vs rejected, median SNR improvement
- **Scientific Impact**: Publications using Achernar data (long-term)

## 🌐 API Documentation

### Daemon API (Local)

**Base URL**: `http://localhost:3030/api/v1`

#### Get Equipment Status
```http
GET /status
```

**Response**:
```json
{
  "mount": "tracking",
  "camera": "idle",
  "focuser": "connected",
  "temperature": 22.4,
  "humidity": 45.2
}
```

#### Capture Image
```http
POST /camera/capture
Content-Type: application/json

{
  "exposure": 30.0,
  "gain": 120,
  "count": 5,
  "filter": "Ha"
}
```

**Response**:
```json
{
  "job_id": "job_123",
  "status": "queued"
}
```

#### Mount Slew
```http
POST /mount/slew
Content-Type: application/json

{
  "ra": 14.22,
  "dec": -32.05
}
```

#### Live View Stream
```http
GET /stream/mjpeg?fps=10&scale=0.5
```

### WebSocket Telemetry

**URL**: `ws://localhost:3030/ws/telemetry`

**Message Format**:
```json
{
  "type": "TELEMETRY_UPDATE",
  "timestamp": 1702931000,
  "payload": {
    "mount": {
      "ra": 14.22,
      "dec": -32.05,
      "pier_side": "west"
    },
    "camera": {
      "temperature": -10.1,
      "status": "exposing",
      "progress": 0.45
    },
    "guiding": {
      "rms_total": 0.35,
      "dx": 0.1,
      "dy": -0.2
    },
    "ai": {
      "focus_hfr": 2.34,
      "cloud_prob": 0.05,
      "recommended_exposure": 45.0
    }
  }
}
```

### Cloud API

See [API Documentation](./docs/API.md) for complete cloud API reference.

## 📚 Configuration

### Daemon Configuration (`achernar.toml`)
```toml
[server]
host = "0.0.0.0"
port = 3030
log_level = "info"

[equipment]
profile_name = "RedCat51_Setup"

[equipment.camera]
driver = "zwo_asi"
id = "ASI533MC_Pro"
gain_default = 100
offset = 50

[equipment.mount]
driver = "indi"
device_name = "iOptron HEM27"
host = "localhost"
port = 7624

[ai]
enable_edge_inference = true
model_path = "./models/onnx"
cloud_threshold = 0.75
uncertainty_threshold = 0.85

[storage]
raw_path = "/mnt/astro_data/raw"
archive_path = "/mnt/astro_data/archive"
```

### Environment Variables (`.env`)
```bash
# Django
DJANGO_SECRET_KEY=your-secret-key
DJANGO_DEBUG=True
DATABASE_URL=postgresql://user:pass@localhost:5432/achernar

# Redis
REDIS_URL=redis://localhost:6379/0

# AWS (optional)
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_STORAGE_BUCKET_NAME=achernar-data

# ML
MLFLOW_TRACKING_URI=http://localhost:5000
```

## 🔧 Hardware Setup

### Supported Hardware

#### Cameras
- **ZWO ASI** series (via ZWO SDK)
- **ToupTek** series (via ToupTek SDK)
- **DSLR** (via libgphoto2)
- **V4L2** webcams

#### Mounts
- **INDI-compatible** mounts
- **ASCOM Alpaca** (via bridge)
- **Serial/TCP** direct control

#### Focusers
- **INDI** drivers
- **Moonlite** protocol
- **USB/Serial** direct

#### Filter Wheels
- **INDI** drivers
- **USB HID**
- **Serial** commands

### Minimum Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Raspberry Pi 4 (4GB) | Intel NUC 11th gen |
| RAM | 4 GB | 8 GB+ |
| Storage | 64 GB SSD | 256 GB+ NVMe |
| GPU | None | NVIDIA Jetson Nano |
| Network | WiFi or Ethernet | Dual-band WiFi |

See [Hardware Setup Guide](./docs/HARDWARE.md) for detailed instructions.

## 🚢 Deployment

### Production Deployment
```bash
# Build optimized binaries
cargo build --release --target x86_64-unknown-linux-gnu

# Build Docker images
docker-compose -f docker-compose.prod.yml build

# Deploy to cloud
./scripts/deploy.sh production
```

### Edge Device Deployment
```bash
# Cross-compile for Raspberry Pi
cargo build --release --target armv7-unknown-linux-gnueabihf

# Package with systemd service
./scripts/package_daemon.sh

# Install on device
scp target/armv7-unknown-linux-gnueabihf/release/achernar-daemon pi@raspberrypi:~
ssh pi@raspberrypi 'sudo ./install_daemon.sh'
```

See [Deployment Guide](./docs/DEPLOYMENT.md) for complete instructions.

## 🧪 6-8 Week R&D Roadmap

### Week 1-2: Foundation
- ✅ Implement classical metrics (HFR, background, elongation)
- ✅ Train tiny focus classifier/regressor (MobileNetV3-Small)
- ✅ Export ONNX INT8 with calibration
- ✅ Unit tests for <50ms latency
- 🎯 **Deliverable**: Focus score in real-time UI

### Week 3-4: Quality Control
- 🔄 Cloud/obstruction classifier with confidence
- 🔄 Contextual bandit v0 (LinUCB)
- 🔄 MLflow/DVC setup
- 🔄 FAISS prototype for retrieval
- 🎯 **Deliverable**: Auto-reject with explainable alerts

### Week 5-6: Embeddings & Ranking
- ⏳ Bimodal embedding (image+metadata)
- ⏳ "Similar subs" retrieval
- ⏳ Target ranking with OR-Tools
- ⏳ Model versioning in SQLite
- 🎯 **Deliverable**: "Find similar framings", tonight's targets ranked

### Week 7-8: Multi-Modal & Feedback
- ⏳ Light-curve encoder + classification
- ⏳ Calibrated quality scoring (Platt/Isotonic)
- ⏳ User feedback collection (thumbs up/down)
- ⏳ Active learning pipeline v0
- 🎯 **Deliverable**: User corrects AI, sees improvement

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards

- **Rust**: Run `cargo fmt` and `cargo clippy` before committing
- **Python**: Follow PEP 8, use `black` and `flake8`
- **TypeScript**: Use `prettier` and `eslint`
- **Commits**: Follow [Conventional Commits](https://www.conventionalcommits.org/)

## 🐛 Known Issues & Limitations

- **Edge Models**: Require manual download (not bundled to keep repo size small)
- **INDI**: Some drivers may have compatibility issues on macOS
- **Cloud Sync**: Large FITS files (>100MB) may timeout on slow connections
- **WebRTC**: Symmetric NAT requires TURN server (not yet configured)

See [GitHub Issues](https://github.com/yourusername/achernar.tech/issues) for complete list.

## 📖 Documentation

- [Development Guide](./docs/DEVELOPMENT.md)
- [API Documentation](./docs/API.md)
- [AI/ML Architecture](./docs/AI_ARCHITECTURE.md)
- [Hardware Setup](./docs/HARDWARE.md)
- [Deployment Guide](./docs/DEPLOYMENT.md)
- [Contributing Guidelines](./CONTRIBUTING.md)

## 🙏 Acknowledgments

- **INDI Protocol** maintainers for the astronomy device standard
- **Astropy** community for scientific Python tools
- **PyTorch** and **ONNX** teams for ML infrastructure
- **Amateur astronomy community** for invaluable feedback
- All open-source contributors

## 📄 License

This project is licensed under the MIT License - see [LICENSE](./LICENSE) for details.

## 📧 Contact

- **Project Lead**: Ibrahim Shahid
- **Email**: contact@achernar.tech
- **Documentation**: https://docs.achernar.tech
- **GitHub**: https://github.com/yourusername/achernar.tech

## 🌟 Comparison: Achernar vs Traditional Software

| Feature | Traditional | Achernar |
|---------|------------|----------|
| Offline Operation | Limited/None | Full functionality |
| AI Guidance | None | Real-time, uncertainty-aware |
| Model Updates | N/A | Active learning from users |
| Explainability | Manual inspection | Automated SHAP/attention |
| Multi-Modal Learning | Separate tools | Unified embedding space |
| Privacy | Data uploaded | Federated learning option |
| Deployment | Desktop-only | Edge + Cloud + PWA |
| Hardware Support | Vendor-specific | Universal (INDI/ASCOM/USB) |

---

**Built with ❤️ for the astronomy community by astronomers, for astronomers.**

*"Achernar: Where classical astronomy meets modern AI."*