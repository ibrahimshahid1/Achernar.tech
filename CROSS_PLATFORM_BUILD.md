# Cross-platform builds and model runtimes

This guide describes the build paths present in the repository and the proposed
compute-aware runtime in [Achernar v5](achernar_v5.tex) and the
[runtime ADR](docs/adr/compute-aware-model-runtime.md). The runtime proposal is
not an accelerator support announcement. A successful build, a detected GPU,
and a qualified model are separate results.

Architecture snapshot: 2026-10-03. The current daemon creates an ONNX Runtime
session for FQE without explicitly registering an accelerator execution
provider. Its existing serving path is CPU inference on batches of 64×64 star
crops, in shadow mode alongside classical measurements. Automatic provider
selection, benchmark caching, and the proposed `[ai]` configuration are not yet
implemented.

## Current build prerequisites

The daemon includes a C++ camera bridge. An empty Cargo feature list does not
remove that dependency or produce a pure-Rust simulator build. See
[`build.rs`](achernar-daemon/build.rs) and the bridge's
[`CMakeLists.txt`](achernar-daemon/ols-bridge/CMakeLists.txt) for the build contract.

| Component | Requirements and source of truth |
| --- | --- |
| Rust daemon | Rust stable, CMake 3.21+, C++17 toolchain, OpenCV 4 and TIFF; additional USB/serial dependencies on the target OS |
| Linux native dependencies | The CI job installs `cmake`, `g++`, `pkg-config`, `libopencv-dev`, `libtiff-dev`, `libusb-1.0-0-dev`, and `libudev-dev` |
| macOS native dependencies | C++ toolchain and packages required by the bridge; its CMake configuration probes Homebrew's `opencv@4` and `libtiff` paths |
| Windows native dependencies | Visual Studio 2022 C++ tools, CMake, and vcpkg; checked-in scripts restore and stage native DLLs |
| Optional camera capabilities | Depend on bridge configuration, vendor SDKs and native libraries. LibRaw availability determines whether supported native RAW captures can also be decoded into FITS |
| Web/UI workspace | Node.js and pnpm as declared in [`package.json`](package.json); CI currently uses Node.js 20 |

The Cargo manifest's `indi-support` feature is commented out. Do not use the old
`cargo build --features indi-support` instructions. The optional `ascom-support`
Cargo feature is not evidence that a particular camera or telescope is
validated; inspect the daemon's compiled-driver report and the applicable
hardware test.

## Build and inspect the current daemon

On Linux or macOS, after installing the native dependencies, run from the
repository root:

```bash
cargo build --manifest-path achernar-daemon/Cargo.toml
cargo test --manifest-path achernar-daemon/Cargo.toml
cargo run --manifest-path achernar-daemon/Cargo.toml
```

The daemon reads `ACHERNAR_CONFIG`, or `./achernar.toml` relative to its working
directory. Keep that working directory and the configured storage paths
deliberate when comparing runs. The default bind address is `127.0.0.1:3030`;
`ACHERNAR_HOST` and `ACHERNAR_PORT` can override it. Startup logs report the
bridge ABI and RAW decoding capability.

From another terminal:

```bash
curl http://127.0.0.1:3030/health
curl http://127.0.0.1:3030/api/v1/drivers
curl http://127.0.0.1:3030/api/v1/status
```

`/health` returns `OK`; it does not establish camera, model, or accelerator
readiness. Opening the preview WebSocket does not connect a camera or start
acquisition. Use the UI or the applicable hardware smoke test first.

On Windows, use the checked-in scripts from the repository root:

```powershell
.\scripts\windows\bootstrap.ps1
.\scripts\windows\build.ps1 -Configuration Debug -RunTests
.\scripts\windows\run-daemon.ps1 -Configuration Debug -Port 3030
```

Follow [Windows rig-agent and SV105 testing](docs/windows-rig-agent-testing.md)
for prerequisite installation, simulator acceptance and physical-camera tests.
The scripts and `.github/workflows/windows-daemon.yml` define an x64 MSVC build
and simulator test; their presence alone is not a passing run or a physical
hardware qualification.

For the UI workspace:

```bash
pnpm install
pnpm -C packages/ui build
pnpm -C achernar-frontend exec tsc --noEmit -p tsconfig.app.json
```

The shared UI client's legacy preview parser still differs from the daemon's
current protocol. See [streaming architecture](docs/STREAMING_ARCHITECTURE.md)
before using that client as a protocol acceptance test.

## Proposed execution-provider qualification matrix

All accelerator rows below are planned qualification targets. There are no
documented accelerator Cargo feature names or install commands yet. Packaging
must establish a compatible ONNX Runtime build, provider libraries, driver and
device combination before runtime probing can qualify an artifact.

| Target | Current evidence in this checkout | Proposed qualification path |
| --- | --- | --- |
| Linux x86-64 CPU | CPU session path; Linux CI is configured for format/compile checks | Keep as the reference CPU path; add repeatable release-binary correctness and performance evidence |
| Windows x86-64 CPU | Windows build, Rust tests and simulator workflow are configured | Record a successful workflow run and device acceptance separately |
| macOS CPU | Platform-specific bridge build code; no equivalent checked-in hardware qualification matrix | Validate the exact OS, architecture and packaged runtime |
| Raspberry Pi 4/5, Linux aarch64 | Intended deployment class; no checked-in device qualification | Qualify the CPU baseline and native dependencies on each supported OS/device combination |
| NVIDIA laptop/desktop, Linux x86-64 | No accelerator registration in the serving path | First accelerator candidate: a tested CUDA provider build; qualify TensorRT separately if adopted |
| Original Jetson Nano, Linux aarch64 | No qualified daemon/provider distribution | Investigate compatibility with its exact JetPack, CUDA, TensorRT and ONNX Runtime stack before promising accelerator support |
| Jetson Orin Nano, Linux aarch64 | Separate, unqualified target | Validate independently of the original Nano; its results cannot certify the older device |
| Windows GPU | No DirectML or CUDA serving integration | Later provider-specific packages and tests; GPU presence alone does not establish either provider |
| Apple silicon | No CoreML serving integration | Later CoreML provider qualification, including operator placement and shared-memory measurements |
| Intel accelerator | No OpenVINO serving integration | Optional later provider qualification when a concrete deployment requires it |

CoreML is the candidate ONNX Runtime provider in the Apple row. Do not label a
separate, unimplemented `Metal` execution provider as supported. Report the
actual provider and observed graph placement, including CPU fallback for
unsupported nodes.

Device names are advisory inputs to discovery. Selection must operate on an
artifact, provider, provider options, and input/batch profile that passed local
correctness and resource checks. No row implies that a named device can run a
particular `Nano`, `Medium`, or `Large` scientific model variant.

## Evidence required for a supported runtime package

Before changing a matrix row from planned to qualified, retain the binary/build
identity, OS/architecture, device identity, driver, runtime/provider versions and
options, artifact and contract hashes, and power/thread configuration. Run the
[testing protocol](TESTING.md#proposed-compute-aware-runtime-acceptance) on the
packaged binary, including sustained operation with camera acquisition,
preview encoding and other representative rig workloads.

Record session/engine setup separately from warmed end-to-end inference. A
TensorRT engine or similar compiled cache must be keyed to its relevant device,
software and artifact configuration and rebuilt when incompatible. Benchmark
results and engine caches belong in the local data directory; they are not
immutable model-bundle contents or portable promises of another machine's
performance.

Release evidence must include successful fallback after provider failure and
the measured memory reserve for the rest of the daemon. On devices with shared
CPU/GPU memory, identify overlapping measurements and do not add them together
as though they were independent allocations. Source-only CI does not replace
these checks.
