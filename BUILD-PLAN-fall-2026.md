# Achernar — Build & Data Plan
**Aug 7 2026 → Jan 1 2027** · Z114 + SV105 + DSLR (semester loan) + a focuser you build

Portfolio freeze: **Nov 1**. Full scope: **Jan 1**.

---

## 1. The finding that decides the focuser design

I checked OLS before recommending anything. `OpenLiveStacker/src/focuser.cpp` is
**46 lines**, and its entire driver list is:

```cpp
return {{ "alpaca:remote", "auto" }};
```

That's it. One driver. There is no Moonlite driver, no USB-serial driver, no
INDI focuser driver in OLS — even though the Rust side
(`hardware/focuser.rs`) has enum variants that map to `"moonlite"` and `"usb"`.
Those names point at OLS drivers that **don't exist**, which is exactly what
`capabilities.rs` reports when it marks them `disabled`.

So there are two paths:

| Path | Work required |
|---|---|
| Arduino + Moonlite protocol | Write a new OLS C++ focuser driver, wire it through the bridge ABI, rebuild. **Days of work in the riskiest part of the codebase.** |
| **ESP32 + Alpaca** | **Zero new C++ or Rust.** `alpaca_focuser.cpp` (178 lines) is already implemented and already compiled. |

**Decision: ESP32, not Arduino Nano.** Flash **myFP2ESP32** — the ESP32
successor to myFocuserPro2, which natively speaks ALPACA, ASCOM, INDI and TCP/IP,
and has 315k+ downloads. Your focuser becomes a WiFi device on the network, the
daemon's existing Alpaca path connects to it, and you write no glue at all.

The ESP32 devboard is ~$8 versus ~$5 for a Nano. That $3 buys you a week.

Bonus: the Alpaca focuser interface includes `Temperature` and `TempComp`
properties as standard, so the DS18B20 probe is exposed through the same API —
which closes **D5** (no environment sensor) at the same time as **D3** (no
focuser position). One build, two blocking defects.

---

## 2. The observing trick that makes an untracked Dob work

The Z114 has no tracking, so exposures trail at 15.041·cos(δ) arcsec/sec. At
0.66"/px (Z114 + 2× Barlow + SV105), on the celestial equator, you'd be limited
to ~44 ms before the trail exceeds one pixel — which would corrupt every HFR
measurement.

**Point at Polaris instead.** δ = +89.35°, so cos δ = 0.011:

| Field | Trail rate | 500 ms exposure |
|---|---|---|
| Equator (δ = 0°) | 22.8 px/s | 11.4 px — unusable |
| δ = +40° | 17.4 px/s | 8.7 px — unusable |
| **Polaris (δ = +89.35°)** | **0.26 px/s** | **0.13 px — negligible** |

At the pole the sky is effectively stationary. An untracked tabletop Dob becomes
a tracked telescope for focus-sweep purposes, for free. You re-centre roughly
every 30–45 minutes as Polaris walks its 39′-radius circle around the pole.

**Every focus sweep in this plan targets the north celestial pole region.** Use
field stars around Polaris rather than Polaris itself where possible — mag 2.0
will clip an 8-bit sensor at best focus, and a clipped core reads as
artificially *sharp*.

---

## 3. Rig assignment

| Rig | Hardware | Job |
|---|---|---|
| **R2** | Z114 + 2× Barlow + SV105 + your focuser | **Primary.** Focus sweeps → FQE, FOP, FDF |
| **R1** | Z114 + SV105, native | TED trailing ladder (untracked = free exact labels) |
| **R4** | DSLR + kit lens + tripod + intervalometer | COC, runs unattended every night from week 1 |
| **Bench** | DSLR indoors | Photon transfer curves → Model 6 |

**SV105 is the primary instrument, not the DSLR.** Three reasons:

1. It streams through **Path A** (`start_stream` → `vision_loop`), which is the
   mature, 1254-line, already-working path. The DSLR goes through **Path B**,
   which has never been compiled.
2. A 25-position sweep at 200 ms × 3 frames is ~2–3 minutes including moves.
   With a DSLR at a 3–6 s gphoto2 round trip it's 8–10 minutes. **Runs are the
   binding constraint on every focus model**, so this is a 3–4× multiplier.
3. **Weight.** A T6S + T-ring is ~700 g hanging off a plastic 1.25″
   rack-and-pinion focuser on a tabletop Dob. It will sag and slip. The SV105 is
   ~100 g.

The semester camera loan is still valuable — it just goes to R4 and the bench,
not on the telescope.

---

## 4. What actually gets finished

Realistic clear nights, Piedmont NC: ~12 usable sessions by Nov 1, ~25 by Jan 1.
August and September are humid and cloudy (TCO doesn't even run public nights in
summer); October is decent; **November and December are the best months** —
cold, dry, post-frontal, and 14 hours of darkness.

### By Nov 1 — the portfolio freeze

- [ ] **Model 6 (Background/SNR) — COMPLETE.** Measured gain, read noise, black
      level, saturation per body per ISO. Bench work, weather-independent.
- [ ] **Motorized focuser — built, calibrated, and driving from the daemon over
      Alpaca.** With measured steps/µm, backlash, and repeatability.
- [ ] **Model 1 (FQE) — trained on your own data**, grouped-split, evaluated
      against the classical HFR baseline, exported to ONNX, running in the
      daemon's shadow path. Target 60–80 focus runs.
- [ ] **Model 5 (TED) — dataset complete** with physically-computed labels.
      2 nights.
- [ ] **Model 4 (COC) — dataset accumulating**, 15+ labelled nights.

### Nov 1 → Jan 1

- [ ] **Model 2 (FOP)** — needs ~200 runs. At 20+ runs/night automated, that's
      6–8 more nights. Reachable only because the focuser is motorized.
- [ ] **Model 3 (FDF)** — one or two dedicated thermal nights. December cooling
      curves are ideal.
- [ ] **Model 4 (COC) — trained.**
- [ ] Whatever the Miroshnichenko thread turns into, on a separate track.

### Cut, deliberately

- **Model 7 (Exposure bandit)** — needs a tracked mount and 300 s exposures.
- **Model 8 (FOR)** — needs a filter wheel. Already descoped.
- **Model 9 (TME edge)** — needs plate solving plus 10k solved frames.

**Five of nine models, and they're the five that form one coherent story:** the
complete focus stack, plus the two sky-condition models, sitting on sensor
constants you measured yourself. That reads far better than nine half-finished
models.

---

## 5. The parallel track — UNCG

**Rule: this track may not block the build.** It has a response latency measured
in weeks and an outcome outside your control. Plan Phases 0–4 as though UNCG
will never reply. If it does, it's upside.

Prof. Miroshnichenko is **Director of the Three College Observatory** — 0.81 m
Ritchey-Chrétien on an equatorial mount, largest telescope in NC, dark site at
Graham, ~45 min from Greensboro — and he's already said there's data to work on.

Three asks, in this order:

1. **The archive project (his benefit).** His 2025 *Galaxies* paper builds
   photometric criteria from optical + near-IR colour indices, finds ~40 B[e]
   candidates, then confirms spectroscopically. Replacing hand-built cuts with
   ranking in a learned embedding space is a direct extension — and it's
   *falsifiable against work he's already published*, which is why it's a real
   proposal rather than a pitch.
2. **The autofocus and engineering logs (your benefit).** Every observatory
   running ACP/MaxIm/NINA writes (focuser position, HFD, temperature) triples
   nightly and throws them away. They are exactly the FOP and FDF training set.
   TCO has been running a binary programme since 2013. **One zip file could
   supply, in a single transfer, the hundreds of runs that would otherwise take
   twenty nights.** Be honest that this serves your project, not his.
3. **Engineering time (small, later).** Not "may I observe" — "may I run a
   defocus sequence on a bright star during engineering time." Fifteen minutes,
   no dark time needed. Offer to take the nights nobody wants.

Second contact: **Dr. Alicia Aarnio**, same department — stellar magnetic fields,
young stars, large datasets, and she runs an undergraduate research group. Better
fit for the software/ML side specifically.

**What this track does and doesn't do:**

| | |
|---|---|
| **Could unblock** | FOP, FDF (if the logs exist). TME, via the échelle archive. Background/SNR, via their calibration frames. |
| **Cannot unblock** | **COC** — they don't observe through cloud. Nobody does. |
| **Does not replace** | The DIY focuser. Data you can't re-collect on demand isn't a substitute for an instrument you control. |

## 6. Fallbacks and alternative routes

Defined as triggers, so you're not re-deciding under pressure in October.

| Route | Cost | Activate when |
|---|---|---|
| **LCO public archive** | free | **Now.** No login, no application. Real FITS with headers; exercises the ingest path while it's cloudy. |
| **UNCG / TCO** | free | **Now.** One email, runs in parallel. |
| LCO Global Sky Clubs | free | If the Durham club will sponsor an application — 10 hrs on the 0.4 m network. |
| Skynet (UNC-CH) | ~$65 | If you want a **scriptable API** to write a daemon driver against. ~20 telescopes; access via the tuition-free course. |
| ZWO EAF | $179 | If the DIY focuser build fails twice. Same outcome, no build risk, `indi_asi_focuser`. |
| Seestar S30 / S50 | $349 / $499 | If the build slips past October. Native Alpaca exposes camera + mount + **focuser** — drop-in rig, zero new code. But 2.4"/px and fixed optics, so it's an *integration* rig, not an FQE rig. |
| iTelescope | ~$50/hr | Only after the focus stack is done, and only for FOR + the bandit. Its ACP directive set has **no absolute focuser position command**, so it can never help FQE or FOP. |
| **GoTo mount** | ~$200 | **Never.** Solves tracking; tracking isn't the blocker. Buys the bandit and nothing else, while $40 of focuser buys FOP + FDF. |
| Raleigh loaner | $25 + hassle | Never. Residency proof and in-person pickup for gear worse than TCO's. |

**The one-line version:** own gear plus a $40 focuser covers six of nine models.
Everything in this table buys models that were already descoped.


## 7. Phases

### Phase 0 — Aug 7–21 · Software foundation (no sky required)

The gate. Nothing downstream is real until this passes.

- [ ] **BP-0:** `brew install libraw libgphoto2 libusb opencv libtiff`, then
      `cargo build`. Expect signature mismatches from the never-compiled capture
      work — fix against a clean tree first.
- [ ] Save `curl -s localhost:8080/drivers` to `docs/`. This is your ground truth.
- [x] **D1 fixed** — the builder now reads `saturation_adu` and drops frames it
      cannot gate. `--allow-missing-saturation` exists for synthetic fixtures and
      stamps the manifest `permissive`. Tests in `ml/tests/test_dataset_builder.py`.
- [x] **D2 downgraded to latent** — `capture()` computes no quality at all, so
      FQE labels come only from the autofocus path, which *is* plumbed
      (`api/focus.rs:561`). Fix when still-capture starts measuring.
- [ ] **Run `python3 -m unittest discover -s tools -p 'test_*.py'`** on the Mac,
      plus `ml/tests/` — one dataset-builder test needs `sep`, which wouldn't
      install in the sandbox.
- [ ] **Measure your sensors:** `tools/photon_transfer.py` (see `tools/README.md`).
      Indoors, any weather. Produces the `[camera] saturation_adu` line, and
      finishes Model 6.
- [ ] **BP-1:** probe the SV105 through the `uvc` driver. Confirm it opens on
      macOS and reports an **8-bit** container. If libuvc can't claim it
      (AVFoundation may), you need a Linux host — find out now, not in October.
- [x] **Sidecar importer built** (`tools/import_session.py`) — closes G2/G4/G5.
      Migration `006_environment_log.sql` adds `environment_log` and
      `disturbance_log`. Sky labels land in `frame_measurements` beside where a
      COC model's predictions will go.
- [ ] **Start RB-5 (COC) immediately.** Body + kit lens + tripod + intervalometer,
      one frame every 30 s, label every 10 min. Runs on cloudy nights. It needs
      more distinct nights than anything else, so it has to start first.
      Import the next morning with `--dry-run` first.

### Phase 1 — Aug 14–Sep 4 · Build the focuser

- [ ] Measure the Z114 focuser: **travel per full knob revolution**, with
      calipers. Everything else derives from this number.
- [ ] Size the drive: CFZ at f/3.95 is 4.88·λ·N² ≈ **42 µm**. Spec wants step
      size ≤ CFZ/5 ≈ **8 µm**. With a 3:1 GT2 reduction (20T → 60T) and 1/16
      microstepping you'll land around 2–3 µm/step — comfortably inside, with
      torque to spare.
- [ ] CAD and print the bracket. **PETG or ASA, not PLA** — PLA creeps under
      belt tension and won't survive a car trunk in the sun.
- [ ] Wire: ESP32 + DRV8825 (or TMC2209 for quiet microstepping) + NEMA 17 +
      DS18B20 temperature probe.
- [ ] Flash **myFP2ESP32**, configure Station mode on your WiFi.
- [ ] Verify from a laptop: `curl http://<esp32>/api/v1/focuser/0/position`
- [ ] Connect the daemon: `POST /connect` with focuser driver `alpaca`, then
      `POST /focuser/moveto`. **No code changes should be required.** If they
      are, that's the finding.

### Phase 2 — Sep 4–18 · Calibrate on the bench

The artificial star is what makes this phase weather-proof.

- [ ] Build an artificial star: LED behind a pinhole, 20 m away (across a yard
      or a long hallway). At 20 m the Z114 focuses ~10 mm further out than
      infinity — check that's inside your focuser travel.
- [ ] Print a **Bahtinov mask** for the Z114. It gives an independent,
      human-readable best-focus indicator — free ground truth to check the
      model against, and a good portfolio detail.
- [ ] **BP-5:** measure steps/µm against a dial indicator, measure backlash by
      approaching a position from each direction, measure repeatability over 20
      round trips. Write to `focuser_profiles`.
- [ ] Run full automated 25-position sweeps indoors on the artificial star.
      **This debugs the entire sweep → capture → HFR → database loop with zero
      clear nights consumed.**

> Artificial-star runs are **engineering data, not training data** — no
> atmosphere, no seeing, one star. Use them to validate the pipeline and the
> mechanics. Do not put them in the FQE training set.

### Phase 3 — Sep 18–Dec 15 · Observe

Every clear night, in priority order:

1. **Focus sweeps on the pole** (RB-2/RB-3). 25 positions, 3 frames, always
   approaching from the same direction. **One run in three should be a reversal
   run**, and some should start ±150 steps out — otherwise FOP's direction and
   history features have nothing to learn from.
2. **One thermal night** (RB-4): start at dusk, autofocus every 30 min, don't
   touch focus between runs, log ambient every 15 min. Pick a night forecast to
   drop ≥8 °C. This single night is the entire FDF dataset.
3. **Two trailing nights** (RB-6): Z114 native, tracking off, exposure ladder
   sampled densely between 0.05 and 0.25 s, at three declinations.
4. **COC every night regardless**, including the bad ones.

Target: **60–80 runs by Nov 1, 200+ by Christmas.**

### Phase 4 — Oct 20–Nov 1, then Dec 15–Jan 1 · Train and write up

- [ ] Build the dataset, grouped by run, split by run *and* night.
- [ ] Train the FQE, calibrate uncertainty, export ONNX, publish the bundle.
- [ ] **Evaluate against the classical baseline.** If the model doesn't beat
      classical HFR on a held-out night, say so — that's a real result and it's
      more credible than a claim.
- [ ] Portfolio assets (below).

---

## 8. Print list

| Part | Material | Note |
|---|---|---|
| NEMA 17 bracket for the Z114 focuser | PETG/ASA | The one part that must be stiff |
| 60T GT2 pulley + belt tensioner | PETG | 3:1 reduction off a 20T motor pulley |
| ESP32 + driver enclosure | PETG | Strain relief matters more than looks |
| **Bahtinov mask** | PLA fine | Independent focus ground truth |
| Artificial star housing | PLA fine | Pinhole + LED + diffuser |
| Dial indicator mount | PETG | For BP-5 only |
| Manual-override knob | PETG | Keep hand focus working — Rule 1 |

An enclosed printer handles ASA well, which is the better choice if the rig
lives in a car.

---

## 9. Portfolio framing

The strongest single artifact is a plot: **your model's predicted best-focus
position versus the true best-focus position, on data collected by a focuser you
designed, printed, wired, and flashed.** That's a closed loop across CAD,
mechanics, firmware, a standard protocol, a Rust daemon, and a trained network —
and it is *measured*, not asserted.

Things worth capturing as you go, because you can't reconstruct them later:

- Photos of every iteration of the bracket, including the ones that failed
- The first V-curve, even if it's ugly
- Backlash measurement before and after compensation
- A short video of a sweep running unattended
- The Bahtinov mask agreeing (or disagreeing) with the model

Document failures. A bracket that flexed and how you found out is more
convincing than a bracket that worked first try.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| libuvc can't open the SV105 on macOS | Test in **week 1**. Fallback: Linux VM with USB passthrough |
| `cargo build` fails badly on the capture path | Path A doesn't need it — FQE work can proceed on streaming alone |
| Z114's plastic focuser has huge backlash | Always approach from one direction; measure it in BP-5; the daemon has `focus/backlash.rs` |
| Bracket flexes under belt tension | PETG/ASA, short belt span, print the failed one first and find out |
| August/September clouds | Phases 0–2 need no sky at all; that's 6 weeks of real progress in the worst weather |
| Not enough runs for FOP by Jan 1 | Report it as "N runs collected, model in progress" — honest beats inflated |

---

## 11. Do this week

1. `cargo build` on the Mac.
2. Order: ESP32 devboard, NEMA 17, TMC2209 or DRV8825, DS18B20, GT2 belt +
   pulleys, 2× Barlow, dial indicator. **Under $100 total.**
3. Put a camera on a tripod tonight and start the COC campaign — it needs the
   most nights and it works when it's cloudy.
4. Measure the Z114 focuser's travel per knob revolution.
