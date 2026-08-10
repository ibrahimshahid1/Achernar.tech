# Achernar — Market Position & Next Implementation Steps

Written 2026-08-07. Supersedes the market discussion; complements `HANDOFF-next-session.md`.
Repo state below was verified against code, not inferred from docs.

---

## 1. The market as it stands

### Price tiers (verified, Aug 2026)

| Tier | Product | Price | Notes |
|---|---|---|---|
| Free / OSS | NINA, Ekos/KStars, PHD2, Siril, ASTAP | $0 | Mature, broad hardware coverage |
| Hardware-bundled | ZWO ASIAIR Plus | $299 one-time | **ZWO gear only** |
| Prosumer automation | Voyager | €129 + €29/yr | Closest competitor; markets "AI driven automation" |
| Prosumer automation | Sequence Generator Pro | paid license | Scripted sequencing |
| **Institutional** | **ACP Expert** | **$1,495** | Dispatch scheduler, PinPoint, remote observatory |
| Institutional | TheSkyX Professional | paid | TPoint, dome/camera control |
| Service | iTelescope membership | $40–160/mo | End-user access to hosted rigs |
| Service | DSP Remote hosting | from $300/mo | Per-rig hosting |

Smart telescope hardware market: **$221M (2025) → $360M (2032), 7.28% CAGR.**

### The white space

Every product above automates through **scripts and rules**. Voyager's DragScript and
RoboTarget are sophisticated, but they are deterministic scripting with heuristics —
not learned control. Nobody in this market ships:

- learned closed-loop control with **calibrated uncertainty**
- a **deterministic safety supervisor** bounding model actions
- **shadow-mode evaluation** with explicit promotion criteria
- **cross-subsystem shared perception** (one vision loop feeding focus + FW + mount)

v4 §7 and §8 specify all four. That is a genuine gap, not a marketing angle.

### Two structural openings

**1. ACP Expert at $1,495 is the price anchor.** It is ~10× the prosumer tier, and it
sells on one thing: unattended operation. Institutions demonstrably pay that for
automation built on 20-year-old dispatch scheduling. This validates the pricing thesis
for the autonomy layer far better than any hobbyist subscription would.

**2. ASIAIR is ZWO-only.** Achernar's INDI / ASCOM / direct-USB layer is vendor-neutral,
which is the one thing ZWO structurally cannot copy without cannibalising camera sales.
**Caveat: `hardware/indi_client.rs` is currently a 12-line stub.** The moat is specified
and not yet built.

---

## 2. Repo state (verified against code)

### Built — 19,626 LOC Rust

| Module | LOC | State |
|---|---|---|
| `hardware/` | 3,547 | camera, focuser, mount, filter wheel, OLS FFI, gphoto2 DSLR, raw_decode, discovery |
| `vision/` | 3,115 | **vision_loop at 1–5 Hz, SharedObservation bus (§8.1 complete)**, classical HFR/star detect, background, trailing, FQE runner |
| `api/` | 3,021 | camera, focus, focuser, mount, websocket |
| `focus/` | 2,553 | agent state machine, classical solver, backlash, drift, features, policy |
| `storage/` | 1,944 | 15 repos, migrations 001–005 |
| `mount/` | 693 | guiding bandit, dither policy, plate-solve trigger, pointing model, safety |
| `filter_wheel/` | 466 | SNR scorer, bandit, planner, controller, safety |
| `control/` | 436 | **safety_guard, action_arbiter, control_supervisor, ownership** |
| `models/` | 374 | registry with SHA-256 + `runtime_eligible` gating |

Plus `ml/` at 4,501 LOC (excl. venv): dataset builders, FQE + offset training, ONNX
export, eval, replay harness, synthetic PSF, AstroM3 loader with checkpoints downloaded.

**The §7/§8 control scaffolding is essentially complete.**

### The single blocker: no trained models, no data

- `models/focus_quality/1.0.0/metadata.json`:
  `"trained": false`, `"weights": "random"`, `"runtime_eligible": false`,
  `"promotion_status": "never — integration placeholder only"`
- `focus_offset/`, `cloud_classifier/`, `trailing_detector/` have **no `training_info` block at all**
  → they fail `is_runtime_eligible()` (`models/model_registry.rs:55`) as well
- `frames` table: **0 rows**. `focus_samples`: 7. `ml/data/`: a README with 4 unchecked TODOs

The architecture is correctly refusing to run itself. `model_registry.rs:55` and
`vision/fqe_runner.rs:325` both gate on `runtime_eligible`, so every model path stays
dark until real weights exist. That is the design working — but it means **the flywheel
has never turned once.**

### Genuinely missing, on the critical path

- `processing/` is **58 LOC total**. `debayer.rs`, `calibration.rs`, `stacking.rs` are
  4-line `pub struct` + TODO. `plate_solver.rs` always returns `Unsupported`.
- `hardware/indi_client.rs` — 12-line stub (see moat note above)
- `ai/` (18 LOC) is dead code, superseded by `vision/`. Delete it.

---

## 3. Next steps, in order

**1. Compile.** `HANDOFF-next-session.md` says the capture work was never compiled — the
sandbox has no Rust toolchain. `cargo build && cargo test` on the Mac. Expect signature
mismatches from the capture rework. Nothing below is real until this passes.

**2. Turn the flywheel once.** Capture → `frames` gets its first row. It is currently 0.
This single number is the gate on every model in §7.6.

**3. Debayer + calibration** (`processing/`, currently 8 LOC of TODO). FQE consumes 64×64
crops; without debayering, an OSC or DSLR frame yields HFR measured on a Bayer mosaic —
i.e. garbage labels. This directly blocks dataset quality, so it comes before any training.

**4. Focus sweep harness → first real dataset.** Automated ramp across focuser positions,
capture at each, classical HFR as the label (§9.1 bootstrapping). One clear night yields
a few thousand labeled crops. `ml/data_pipeline/build_focus_quality_dataset.py` already
exists to consume it.

**5. Train FQE for real.** The whole chain exists: `ml/train/train_focus_quality.py` →
`calibrate_uncertainty.py` → `export/export_quality_onnx.py` → `write_model_metadata.py`.
Flip metadata to `trained: true`, `runtime_eligible: true`, `promotion_status: "shadow"`.
That one change lights up the shadow path already wired at `vision/vision_loop.rs:120`.

**6. Shadow-mode evaluation.** `ml/eval/replay_offset_policy.py` exists. Compare FQE
against classical HFR over logged sessions; apply §7.10 promotion criteria.

**7. INDI client.** The vendor-neutrality moat — and what makes step 2 work on gear you
don't personally own.

### Why this order

Steps 1–3 are prerequisites for possessing any usable data at all. Steps 4–6 are one
complete lap of **train → export → shadow → promote**.

FQE goes first not because focus matters most, but because it is the **only** model where
classical bootstrapping (§9.1) produces labels for free. Every other model in §7.6 needs
either human labelling or nights of policy rollout.

Once one lap is closed for FQE, every remaining model is the same lap with a different
label source. That lap is the thing worth proving once, carefully.
