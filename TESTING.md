# Achernar testing guide

This guide separates checks available in the current repository from acceptance
criteria for the proposed runtime in [Achernar v5](achernar_v5.tex) and the
[compute-aware runtime ADR](docs/adr/compute-aware-model-runtime.md). Architecture
snapshot: 2026-10-03. Commands below exercise existing code; the proposed selector,
benchmark cache and runtime-status API do not yet have executable test commands.

## Prerequisites

Use the toolchain and native dependency instructions in
[CROSS_PLATFORM_BUILD.md](CROSS_PLATFORM_BUILD.md). A daemon test build compiles
the C++ camera bridge even without optional Cargo features. Install web/backend
dependencies only for the components being tested:

```bash
pnpm install
python -m pip install -r requirements.txt
```

Use a dedicated Python environment for ML work:

```bash
python -m pip install -r ml/requirements.txt
```

CI currently uses Rust stable, Node.js 20 and Python 3.11. The root package
manifest declares the supported Node.js/pnpm constraints. Do not interpret an
old minimum Rust version or a dependency installation as proof that a complete
native build works on a target device.

## Current component checks

Run from the repository root with the relevant dependencies installed:

```bash
cargo fmt --manifest-path achernar-daemon/Cargo.toml --all --check
cargo check --manifest-path achernar-daemon/Cargo.toml
cargo test --manifest-path achernar-daemon/Cargo.toml
python -m unittest discover -s ml/tests
python -m unittest discover -s tools
pnpm -C achernar-frontend exec tsc --noEmit -p tsconfig.app.json
pnpm -C packages/ui build
python manage.py check
```

Backend checks use the configured development database/cache. For a local
SQLite check, set `USE_SQLITE=1` in the environment as CI does. Check the actual
command exit status and record the commit and toolchain; test counts in a dated
audit describe that audit, not the current run.

The FQE tests cover normalization, grouped/temporal splits, aggregation,
calibration and artifact handling. The checked-in reference graph is random and
ineligible for automatic selection. Passing inference on that graph verifies
the interface, not focus-estimation quality. Current FQE outputs are shadow
predictions; classical HFR remains the control authority.

`.github/workflows/ci.yml` configures gating frontend type checks, Django checks,
and daemon formatting/compilation. Several other jobs are advisory. The Windows
workflow configures a Rust test suite and simulator API smoke test. Inspect a
run's results before describing a gate as passed; neither workflow establishes
accelerator performance or physical-camera acceptance.

## Current daemon and acquisition smoke test

Start the daemon using the platform build guide. Its default local address is
`http://127.0.0.1:3030`:

```bash
curl http://127.0.0.1:3030/health
curl http://127.0.0.1:3030/api/v1/drivers
curl http://127.0.0.1:3030/api/v1/status
```

`/health` should return `OK`. Inspect the compiled-driver report separately.
Acquisition tests need a connected camera or the simulator; the daemon does not
generate dummy science frames merely because a WebSocket is open. Windows has
an existing isolated simulator/camera test in
[docs/windows-rig-agent-testing.md](docs/windows-rig-agent-testing.md), using
`scripts/windows/smoke-api.ps1`.

After connecting a test camera and starting acquisition through the UI or the
smoke script, inspect the binary preview in a browser console that can reach the
local daemon:

```javascript
const preview = new WebSocket('ws://127.0.0.1:3030/api/v1/camera/stream');
preview.binaryType = 'arraybuffer';
preview.onmessage = ({ data }) => {
  if (!(data instanceof ArrayBuffer) || data.byteLength < 20) return;
  const header = new DataView(data, 0, 20);
  const width = header.getUint32(0, true);
  const height = header.getUint32(4, true);
  const timestampMs = Number(header.getBigUint64(8, true));
  const format = header.getUint32(16, true);
  if (!width || !height || data.byteLength === 20) {
    console.log('Waiting for a previewable frame');
    return;
  }
  if (format !== 1) throw new Error(`Unexpected preview format: ${format}`);
  console.log({ width, height, timestampMs, jpegBytes: data.byteLength - 20 });
};
// Call preview.close() when finished.
```

Offset 16 contains an integer format code, not a focus score. The payload begins
at byte 20 and is JPEG. The nominal sender tick is 100 ms, but repeated camera
timestamps can occur when acquisition is slower. Count distinct source frames
when measuring fresh-frame rate; socket message rate is a different metric.

Inspect measurements on the separate telemetry stream:

```javascript
const telemetry = new WebSocket('ws://127.0.0.1:3030/ws/telemetry');
telemetry.onmessage = ({ data }) => {
  const event = JSON.parse(data);
  if (event.type === 'VISION_UPDATE') {
    const obs = event.observation;
    console.log({
      status: obs.status,
      hfrPx: obs.hfr_px,
      usedStars: obs.star_count_used,
      shadowHfrPx: obs.fqe_hfr_px,
      shadowCalibrated: obs.fqe_calibrated,
    });
  }
};
// Call telemetry.close() when finished.
```

The current startup rate is 1 Hz for the vision loop. Validity and nullable
measurement fields matter: a missing HFR must remain `null`, and a missing or
failed FQE must leave classical measurements available. The FQE runner already
batches a frame's usable 64×64 crops in a single ONNX call.

The shared `packages/ui/src/api/client.ts` still uses legacy local routes and
parses the preview format field as `focusScore`. Do not use that parser as the
current protocol oracle. The direct examples above and the daemon encoder are
the reference for this documentation revision; client repair is separate work.
The `/api/v1/camera/stream/mjpeg` route is a text placeholder, not a multipart
MJPEG service.

## Optional backend stack

The checked-in Compose file defines its services and development configuration.
From the repository root:

```bash
docker compose up -d
docker compose ps
docker compose logs backend
docker compose exec backend python manage.py migrate
```

Inspect actual service health and logs. A web/backend stack test does not certify
the host camera bridge, GPU access in a container, or a model's accuracy. Use the
same isolation and data-directory discipline as the native acquisition tests.

## Proposed compute-aware runtime acceptance

This section defines tests to add with the runtime implementation. It does not
claim these tests or runtime endpoints already exist. Each result must identify
the scientific variant, immutable artifact hash, preprocessing/postprocessing
contract, precision, execution provider/options and input/batch profile.
The first runtime release keeps FQE shadow-only, including when a bundle is
marked promoted. Classical operation is the unconditional baseline; no trained
neural baseline is assumed to be available.

### Correctness and lifecycle

1. Reject missing/tampered artifacts, incompatible schemas or tensor contracts,
   revoked artifacts and mismatched calibration. Record a specific rejection
   reason. A pinned path or explicit provider must not bypass these gates.
2. Run immutable golden vectors through Python/export and the deployed Rust
   path. Compare preprocessing, raw outputs, aggregation, calibrated uncertainty
   and final semantics with declared per-artifact tolerances. Include blank,
   non-finite, edge, saturated and representative real-star inputs.
3. Require all accepted scientific variants to use the same frozen evaluation
   split and relevant device/sky/optics slices. Local latency qualification
   cannot establish scientific quality. A reference artifact is never a
   production model; shadow eligibility does not grant control authority.
4. Check deterministic candidate ordering, hard resource gates, per-family
   pins/disablement, and baseline behavior when no promoted candidate qualifies.
   Verify every eligible installed variant remains visible to selection.
5. Invalidate qualification after relevant artifact, contract, software, device,
   provider-option or workload changes. Corrupt/incomplete cache records must
   trigger fresh qualification rather than silently authorizing a candidate.

### End-to-end performance protocol

Use a release build on the actual deployment machine. Declare the subsystem
deadline, observation cadence, memory reserve, fixture distribution, warm-up
criterion, sample count, repetitions and duration before comparing candidates.
Record sample count and uncertainty for percentile estimates; a few fast
invocations do not establish a reliable p99.

For deadline admission, measure from work admission to publication, including
queue/scheduling delay, preprocessing, host/device transfers, inference, output
synchronization, aggregation and postprocessing. Synchronize asynchronous
accelerator work before stopping the timer. Report warmed p50/p95/p99, maximum
observed latency, throughput, errors and deadline misses. Stage timings and
source-frame age at publication supplement that complete measurement so an
apparently fast kernel cannot conceal stale observations.

Record session creation, model loading, engine compilation and first-result
latency separately from warmed measurements. Verify startup can publish the safe
baseline while a more expensive candidate is qualified; do not hide setup cost
inside an unexplained warm-up discard.

Use realistic FQE star-count distributions and batch sizes, including empty and
large batches, rather than only one crop. Test candidates sequentially to avoid
an artificial out-of-memory result from loading the whole catalog at once.
Measure each family alone and under the joint workload the selector will admit.
Keep scientific test-set labels out of runtime tuning; performance fixtures and
golden correctness fixtures need declared provenance and permitted use.

Record peak process RSS, accelerator allocations and system memory pressure with
the measurement method. On Jetson and Apple silicon, CPU and accelerator memory
can share a pool; identify overlapping counters and avoid double counting.
Include model loading and engine-building peaks as well as steady-state usage.

Run both idle and representative co-load profiles: camera acquisition, preview
JPEG encoding, capture persistence, control tasks and other active model
families. Add a sustained run long enough to characterize warm steady state and
thermal/power throttling, recording duration, temperatures/clocks when available,
power mode, AC/battery state and missed deadlines. An unavailable sensor is
`unknown`, not evidence that throttling did not occur.

There is no universal `<20 ms` or `<50 ms` pass threshold. Allocate each
subsystem's complete latency budget from its cadence and control requirements,
with explicit headroom. A 300-second exposure also cannot complete in five
seconds; report exposure time, readout, transfer, persistence and total capture
time separately.

### Qualification record and cache key

Store the artifact and contract hashes, daemon build, schema/benchmark protocol
and qualification-policy versions, ORT/provider versions and chain/options,
achieved placement, driver, OS/architecture, device identity, CPU ISA/thread settings, precision,
input/batch profile, power mode and workload/reservation profile. Keep
performance measurements in the local data store, separate from
immutable model metadata. Record the measured constraints and rejection reasons,
not only the name of the winner.

### Failure, backpressure and session stability

Exercise unavailable providers, unsupported operators, session-load failures,
out-of-memory conditions, invalid/non-finite outputs and missed deadlines. Verify
that fallback preserves semantic units and validity, publishes the correct
runtime identity and does not reuse a stale neural prediction as a fresh result.
Record actual graph/provider placement where partial CPU fallback is possible.
An application timeout does not guarantee cancellation of native provider work.
Verify that late results are discarded and still-running work remains accounted
for before admitting a replacement; otherwise a fallback can overcommit memory.

The first implementation should pin selection for the observing session. Test a
bounded inference queue, frame-generation invalidation on reconnect, and atomic
publication at a frame boundary. The model used for an observation must be the
one recorded with that observation. Existing camera/control operations must
remain responsive while qualification runs in the background.

If later work adds automatic demotion, add tests for hysteresis, hold-down time,
safe fallback transition and recovery. Do not treat those later mechanisms as
part of current acceptance. Persist selection/fallback events and aggregated
latency windows; avoid unbounded storage growth from per-crop telemetry.

## Reporting results

Attach the exact build/configuration and machine details, command outcomes,
fixture hashes, correctness tolerances, raw benchmark observations or a durable
report, and unresolved limits. State separately whether the result establishes
source compilation, simulator behavior, physical hardware behavior, deployment
parity, or scientific quality. Update a hardware/provider matrix row only after
its required evidence has passed.
