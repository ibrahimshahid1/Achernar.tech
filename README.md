# Achernar - Unified Astronomy Imaging and Control System

**A comprehensive AI-enhanced platform for amateur and professional astronomers combining offline-first design with intelligent cloud capabilities.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://reactjs.org/)

## 🌟 Overview

Achernar is a unified astronomy imaging and control system that democratizes high-quality astrophotography and research through a hybrid AI architecture. The system combines edge computing for real-time operations with cloud computing for collaboration, archiving, and large-scale model training.

**Local inference direction**: Run models beside the equipment and qualify each deployment against the host's measured performance. The implemented neural path is a CPU-default, batched Focus Quality Estimator (FQE) in shadow mode. The checked-in graph is an ineligible reference artifact with random weights; it is not a trained observing model.

The [v5 architecture specification](achernar_v5.tex) and [compute-aware runtime ADR](docs/adr/compute-aware-model-runtime.md) describe the next design: choose a validated model variant and execution provider within memory and subsystem timing budgets. This is planned work; a detected GPU does not yet enable accelerated inference or a larger model.

## ✨ Key Features

### Hardware & Control
- **Universal Hardware Support**: INDI, ASCOM Alpaca, direct USB/Serial control
- **Real-Time Processing**: Hardware-accelerated debayering and H.264/MJPEG encoding
- **Professional Calibration**: Automatic dark, flat, and dark-flat frame integration
- **Plate Solving**: Local astrometry with blind solve capability

### AI/ML Intelligence

#### Local intelligence: current and planned

- **Current FQE infrastructure**: Batched 64×64 grayscale star crops, two-head log-HFR inference, robust pooling and hash-bound uncertainty calibration. Predictions are shadow telemetry; classical HFR remains the autofocus authority.
- **Current cloud/trailing paths**: Heuristic/classical implementations. Trained neural COC and TED deployments remain planned.
- **Current drift tracking**: An optional three-state Kalman forecaster publishes advisory refocus estimates; it does not move the focuser.
- **Planned compute-aware runtime**: Compare qualified model variants using measured end-to-end latency, memory use and family-specific scientific quality. NVIDIA, Jetson and other accelerator support require explicit build and device validation.
- **Fixed-compute methods**: Control, Kalman estimation and small statistical/bandit models retain their own contracts; available GPU memory does not change their scientific meaning.

#### Cloud AI (Heavy Compute)
- **Tri-Modal Embeddings**: AstroM3 integration for light curves, spectra and metadata; the proposed image/telemetry retrofit needs explicit alignment training and validation before it can share the embedding space.
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
- **Field deployment targets**: Raspberry Pi, Jetson Nano, Intel NUC and laptops; validated build/provider combinations and local measurements determine available model capabilities.

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

| **Local (Subsystem-Specific Deadlines)** | **Cloud (Heavier Compute)** |
|--------------------------------|----------------------------|
| FQE shadow inference; planned qualified COC/TED variants | Target ranking/scheduling |
| Noise estimation | Tri-modal embedding training |
| Kalman drift tracking | Anomaly detection at scale |
| Contextual bandits (settings) | Global personalization |
| Classical stats (HFR, background) | Model retraining |
| Uncertainty quantification | A/B testing & canary deployments |
| Graceful fallback to rules | Heavy analytics & SHAP |
| | Federated learning aggregation |

This table describes the division of responsibility, including planned features.
The v5 design measures preprocessing, transfers, inference and postprocessing
against each subsystem's deadline. A single global latency number is not a
hardware-support guarantee.

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
│   │   ├── processing/        # Calibration, debayering, stacking, plate solving
│   │   ├── vision/            # Shared observations, preprocessing, FQE runner
│   │   ├── models/            # Registry and generic runner scaffolding
│   │   ├── ai/                # Legacy AI placeholders; not the active FQE path
│   │   ├── control/           # State machine & sequences
│   │   ├── storage/           # SQLite + FITS I/O
│   │   └── api/               # Axum routes + WebSocket
│   ├── achernar.example.toml  # Configuration parsed by src/config.rs
│   └── Cargo.toml
│
├── backend/                    # Django backend
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
│   ├── export/                # Precision-specific ONNX export
│   └── notebooks/             # Exploratory analysis
│
├── models/                     # Versioned artifacts and metadata; reference FQE only
├── docs/
│   ├── adr/compute-aware-model-runtime.md  # Planned local runtime contract
│   ├── STREAMING_ARCHITECTURE.md           # Frame/observation path
│   └── Achernar-Technical-Audit.md         # Dated implementation audit
├── achernar_v5.tex             # Revised architecture source; distinguishes status
│
├── docker-compose.yml          # Development environment
├── pnpm-workspace.yaml         # Monorepo configuration
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
   cp achernar-daemon/achernar.example.toml achernar-daemon/achernar.toml
   # Edit .env and achernar-daemon/achernar.toml with your configuration
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

6. **Review the model status** (optional for development)

   See [ML development](ml/README.md) for the FQE reference artifact and training workflow. There is no published trained FQE/COC/TED model download in this repository. The daemon skips the random reference artifact during automatic resolution and continues with classical measurements. To discover eligible local bundles from the daemon directory, set `ACHERNAR_MODELS_DIR=../models`; explicit FQE paths are configured under `[focus.models]`.

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

### Local model status and v5 direction

| Family/method | Current status | v5 direction |
|---------------|----------------|--------------|
| FQE | CPU-default ONNX runner; 64×64 star batches; random reference excluded from auto-selection; shadow output only | Train comparable small/larger variants, qualify each artifact/provider path, preserve calibrated log-HFR semantics |
| Cloud/obstruction (COC) | Heuristic path; no usable checked-in neural artifact | Add trained variants after data, quality and promotion gates exist |
| Trailing (TED) | Classical/placeholder path | Validate any CNN extension against classical baselines |
| Focus drift | Optional three-state Kalman advisory model | Remain a fixed-compute estimator with rig-specific state |
| Control and bandits | Separate control/statistical contracts | Preserve safety limits and semantics independently of GPU capacity |

No cross-device latency or accuracy improvement is claimed by these entries.
The planned runtime candidate is an immutable artifact plus execution provider,
provider options and input/batch profile. Selection is pinned initially for the
observing session; classical processing remains available when no candidate
qualifies. Proposed introspection endpoints are `GET /api/v1/ai/models` and
`GET /api/v1/ai/runtime`; these endpoints are not implemented yet.

### Cloud Models (Heavy Compute)

These are design targets. The listed refresh cadences are proposed, and do not
describe a currently validated production service schedule.

| Model | Purpose | Architecture | Update Frequency |
|-------|---------|--------------|------------------|
| Tri-Modal Embeddings | Unified representation | ViT + Transformer + 1D ResNet | Weekly |
| Target Ranker | Tonight's schedule | LightGBM + OR-Tools | Daily |
| Quality Scorer | Frame curation | Calibrated regression | Daily |
| Anomaly Detector | Transient discovery | Isolation Forest | Weekly |

### Proposed Training and Deployment Pipeline

This diagram includes planned infrastructure. Precision-specific artifacts need
their own quality and deployed-parity evidence before shadow qualification.

```mermaid
graph LR
    A[Raw Data] --> B[Feature Store<br/>Parquet/DuckDB]
    B --> C[Weak Labeling<br/>Snorkel-style]
    C --> D[PyTorch Lightning<br/>Mixed Precision]
    D --> E[Validated Artifact Export<br/>FP32 / FP16 / INT8]
    E --> F[Qualified Shadow Deployment<br/>ONNX Runtime]
    F --> G[User Feedback<br/>Active Learning]
    G --> A
```

### Metrics We Track

#### Edge Models
- **Focus Regressor**: HFR error in pixels, log-HFR error, within-run ranking, held-out prediction-interval coverage and width, and end-to-end latency.
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

The current [router](achernar-daemon/src/main.rs) and typed handlers define the
request/response contracts. Inspect capabilities and device state before
issuing hardware commands; a route's existence does not establish that the
connected driver supports the operation.

| Route relative to `/api/v1` | Purpose |
|---|---|
| `GET /status` | Equipment state and observations, with unknown values kept distinct from valid zeros |
| `GET /drivers` | Compiled driver capabilities and availability |
| `GET /commands` | Commands currently published through the relay |
| `POST /camera/capture` | Physical capture and asset provenance; see the [camera handler](achernar-daemon/src/api/camera.rs) |
| `GET /camera/stream` | Binary preview WebSocket; acquisition must be started separately |
| `GET /camera/stream/mjpeg` | Placeholder route; multipart MJPEG is not implemented |
| `/focus/*` | Current focus run, abort, status, history and drift routes |

Use the [testing guide](TESTING.md) for current request examples and the
[streaming contract](docs/STREAMING_ARCHITECTURE.md) for frame parsing.

### WebSocket Telemetry

**URL**: `ws://localhost:3030/ws/telemetry`

The socket publishes equipment telemetry and separate `VISION_UPDATE`
observations. The [telemetry builder](achernar-daemon/src/telemetry.rs) and
[shared observation type](achernar-daemon/src/vision/shared_observation.rs)
define the current fields. FQE predictions remain optional shadow values;
missing guider or model measurements must not be displayed as a successful zero.

### Cloud API

The [Django URL configuration](backend/urls.py) lists mounted services. Relay
support is narrower than the local daemon router; consult the published command
registry and [streaming documentation](docs/STREAMING_ARCHITECTURE.md).

## 📚 Configuration

### Daemon Configuration (`achernar.toml`)

The authoritative working example is
[`achernar-daemon/achernar.example.toml`](achernar-daemon/achernar.example.toml).
The current daemon supports the following relevant subset:

```toml
[daemon]
port = 3030
log_level = "info"

[focus.models]
# Optional explicit path to a validated local FQE bundle; output stays shadow.
# quality_model_path = "../models/focus_quality/<version>/model.onnx"

[storage]
database_path = "achernar.db"
```

The v5 policy below is a **design example**, not current configuration. Keep it
commented until the runtime and parser support are implemented:

```toml
# [ai]
# model_selection = "auto"
# compute_budget = "balanced"  # efficient | balanced | quality
# FQE output initially remains shadow-only; control authority is a separate gate.
```

The selection policy does not confer control authority. Future live use requires
artifact promotion, valid calibration and local qualification; a manual pin or
explicit path must not bypass those gates in the v5 runtime.

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

### Compute-aware deployment targets

| Target | Intended starting point | Required validation |
|--------|-------------------------|---------------------|
| Raspberry Pi | Classical processing and compact CPU candidates | aarch64 packaging, shared RAM reserve, sustained end-to-end timing |
| Jetson Nano | Compact CPU baseline; candidate accelerated variants | Compatible provider build, supported operators, power mode and shared-memory budget |
| NUC or laptop CPU | CPU variants that meet the observing workload budget | Thread settings, co-load, memory and tail latency |
| Laptop with discrete GPU | Candidate larger or higher-resolution models | Supported provider/driver, actual graph placement, memory reserve, quality and timing gates |

These are validation targets, not a completed support matrix or promised model
tiers. A GPU alone does not establish eligibility, and extra compute improves
scientific quality only when a separately validated variant demonstrates it.
See the [build and hardware qualification guide](CROSS_PLATFORM_BUILD.md).

## 🚢 Deployment

### Current build and release status

Use the [current build instructions](CROSS_PLATFORM_BUILD.md) for native builds
and platform prerequisites. The repository contains a development
[`docker-compose.yml`](docker-compose.yml), but no production Compose file,
`scripts/deploy.sh`, or `scripts/package_daemon.sh`. Earlier examples using those
paths described intended automation, not executable release procedures.

Raspberry Pi, Jetson, and accelerated laptop releases require target-specific
native libraries, provider packaging, and the validation matrix in the build
guide. Cross-compiling a Rust binary alone does not qualify a hardware release.

## 🧪 Compute-aware runtime roadmap

The [runtime ADR](docs/adr/compute-aware-model-runtime.md) defines the acceptance
gates. This documentation revision does not implement the following phases.

1. **Contracts**: Versioned manifest, family semantics, lifecycle states,
   preprocessing/golden vectors and subsystem budgets.
2. **FQE runtime**: CPU baseline, provider-aware candidate abstraction,
   end-to-end benchmark cache, configuration and status API. Keep output shadow.
3. **Scientific validation**: Train small and larger FQE variants using identical
   frozen real-data splits; establish quality, calibration and subgroup gates.
4. **Accelerator validation**: Test NVIDIA Linux and Jetson builds under sustained
   observing load before declaring support or model tiers.
5. **Additional families**: Add COC after its dataset and neural implementation
   exist; defer TED/BGE expansion and other providers until validated.
6. **Runtime adaptation**: Add conservative demotion, frame-boundary handover and
   hysteresis after session-pinned selection is reliable.

## 🤝 Contributing

Contributions are welcome. Follow the applicable checks in [TESTING.md](TESTING.md)
and include implementation status and validation evidence in the change description.

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

- **Local models**: The checked-in FQE graph is a random, ineligible reference. Neural COC/TED artifacts, compute-aware selection and accelerator qualification are planned.
- **INDI**: Some drivers may have compatibility issues on macOS
- **Cloud Sync**: Large FITS files (>100MB) may timeout on slow connections
- **WebRTC**: Symmetric NAT requires TURN server (not yet configured)

See [GitHub Issues](https://github.com/yourusername/achernar.tech/issues) for complete list.

## 📖 Documentation

- [v5 Architecture Source](./achernar_v5.tex)
- [v5 Architecture PDF](./achernar_v5.pdf)
- [Architecture Document Guide and Rebuild Instructions](./docs/ARCHITECTURE_DOCUMENTS.md)
- [Compute-aware Model Runtime ADR](./docs/adr/compute-aware-model-runtime.md)
- [ML Development and Artifact Status](./ml/README.md)
- [Cross-platform Builds](./CROSS_PLATFORM_BUILD.md)
- [Testing](./TESTING.md)
- [Streaming and Observation Contracts](./docs/STREAMING_ARCHITECTURE.md)
- [Compute-aware Runtime Review](./docs/compute-aware-runtime-review-2026-10-03.md)

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
