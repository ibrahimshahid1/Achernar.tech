# Equipment runbook: interop, rig acceptance and qualification

This runbook is the rig acceptance procedure of the native equipment layer
(`src/equipment/`). It is the only procedure that can turn a device identity
from `implemented_unqualified` into `qualified` (decision D21, §5.1 rule 10 of
the equipment design spec). Simulator results, CI and the interop tests are
recorded separately from it and never qualify anything on their own.

It covers:

1. what is tested automatically, and how to run it (CI, ESP32 host simulator,
   ASCOM OmniSim);
2. preparing a rig session;
3. the acceptance checklist per device kind: what each check proves, the
   route or command to run, and the pass criteria;
4. how a device identity becomes qualified, and how the result is recorded;
5. a worked example: the R50 + Z114 rig with the ESP32 Alpaca mount;
6. Windows COM drivers through ASCOM Remote;
7. INDI site and time;
8. PHD2 setup;
9. the physical autofocus gate (D18);
10. known gaps found by the interop work.

Conventions used below:

```sh
D=http://127.0.0.1:3030/api/v1        # the daemon ([daemon] port, or ACHERNAR_PORT)
J='Content-Type: application/json'
DB=achernar.db                         # [storage] database_path
```

Every motion route returns on accept with `{status, command_id, ...}`;
completion arrives as events (`GET $D/equipment/events?since_seq=N&kinds=a,b`)
and in `GET $D/status`. Errors are `{error: {code, message}}` with the HTTP
status of their class: 412 not connected, 501 unsupported, 400 invalid value,
409 invalid operation or ownership conflict, 422 safety rejected, 423 resume
block active (automatic sources only), 504 timeout, 502 transport or driver.

---

## 1. Automated tests

### 1.1 CI (every OS, no hardware, loopback only)

```sh
cargo test --locked                                   # the whole suite
cargo test --locked -- equipment::testing::interop    # the interop scenario
```

`src/equipment/testing/interop.rs` holds the CI scenario
`the_tier1_rig_runs_end_to_end_through_the_daemon_routes`. It serves the
daemon's real routes on `127.0.0.1:0` and drives, over HTTP:

- the WP1A-4 kinematic mount in its **ESP32-like profile** (InterfaceVersion
  4, Polar, Topocentric, constant SideOfPier 0, CanSlew false with
  CanSlewAsync true, park, no home), behind real Alpaca HTTP through the
  `testing/alpaca_server.rs` simulator adapter (O11 l);
- the WP1A-5 **Platform 6 focuser** (InterfaceVersion 3, no DeviceState,
  absolute, a temperature probe), declared firmware-owned backlash through its
  identity profile;
- the PHD2 mock, and the filter-wheel, weather and safety simulators.

It checks: connecting moves nothing; a J2000 slew is converted to the mount's
JNow frame and settles (`slew_commanded → started → arrived → settled`);
tracking on and off; a focuser move is a single device move with read-back;
a filter change runs the handshake; guiding, guide steps and a dither through
PHD2; a pulse is refused while PHD2 guides; a slew stops guiding first (D20);
unsafe weather drives the safety policy in the §5.7 order (stop guiding,
abort slew, park); abort and e-stop reach the device; resume blocks refuse an
automatic capture with 423; a park is verified through AtPark and a park that
never sets AtPark fails; and every v6 ground-truth table the session feeds
gets rows (`device_connections`, `equipment_events`, `control_audit_logs`,
`focuser_moves`, `filter_switch_events`, `guide_steps`, `dither_events`,
`exposure_device_windows`, `guider_state_log`, `pointing_samples`,
`environment_log`, `device_profiles`), with the policy's stop guiding, abort
slew and park audited under the safety source in that order, and the
exposure window's guide RMS matching a reference computed from PHD2's steps.
The camera stays on OLS; for the exposure-window tables the capture path's
own hooks stand in for a camera, because no CI fixture drives an OLS capture.

Two sibling tests use the same daemon and the same ESP32-like mount over HTTP:

- `manual_motion_reaches_the_alpaca_mount_through_the_daemon_routes`: a jog
  through `mount/move-axis` on a mount that reports CanSlew false (O10: jog is
  gated on CanMoveAxis), stopped by its dead-man (`jog_deadman_lapsed`), kept
  moving by renewals, stopped at once by rate 0; a tertiary-axis jog refused
  with 501 and nothing sent; a sync (`sync_applied` with a new generation,
  read back); a pulse seen as `is_pulse_guiding` true, then false; a jog
  refused with 422 while parked.
- `a_stalled_alpaca_mount_degrades_and_a_silent_one_is_lost`: one 4 s stall
  of the DeviceState poll makes the mount `degraded`, never `lost`;
  `/mount/abort` stops a GoTo while it is degraded; a server that stops
  answering makes it `lost` within `lost_after_s`, and `/mount/abort` still
  runs its path on the lost handle (checks M10 and M11 in simulation).

Two known gaps (§10 items 1 and 2) are asserted as they are today, so the
fix of each fails the scenario at that step and its message says what to
require instead.

`native_alpaca_drivers_conform_against_the_served_simulators` holds the native
Alpaca telescope and focuser drivers to the WP1A-2 conformance suites over the
same HTTP adapter.

### 1.2 Local interop (`#[ignore]`, run on demand)

From `achernar-daemon/`. The firmware folder is local-only and lives in the
main checkout, so the recipe finds the main checkout through git; it works
from a worktree too:

```sh
ROOT=$(git rev-parse --path-format=absolute --git-common-dir)/..
ACHERNAR_MOUNT_SIM=$ROOT/mount_cad_esp_32_firmware/achernar-mount-esp32/host_sim/build/achernar-mount-sim \
ACHERNAR_OMNISIM=$HOME/.local/opt/ascom-omnisim/omnisim \
cargo test --locked -- --ignored interop --nocapture
```

Each test starts its process on free ports and kills it on every exit path
(including a failed assertion). A test skips, and passes, only when its
variable is **unset**; it prints `ACHERNAR_MOUNT_SIM is not set; skipping
this local-only interop test` (visible with `--nocapture`). A variable that
is set but does not name an existing file fails the test with the path, so a
mistyped path cannot pass. A pass with a variable unset proves nothing about
that fixture: for the O2 gate (deleting the OLS mount and focuser) and for a
rig session, run with both variables set and check the output for the
host-sim and OmniSim lines (`host sim: ...`, `OmniSim on port ...`). An
OmniSim log is removed after a pass and kept, with its path printed, after a
failure.

- **ESP32 host simulator** (`ACHERNAR_MOUNT_SIM`, macOS and Linux). The
  firmware's portable core in real time. Build it with
  `make -C mount_cad_esp_32_firmware/achernar-mount-esp32/host_sim` (the
  folder is local-only and gitignored). The tests start it with
  `--alpaca-port 0 --discovery-port 0 --lx200-port 0 --lat --lon --elev
  --unique-id` and read the port from its banner
  (`achernar-mount-sim 0.1.0 alpaca=127.0.0.1:PORT ...`). With the Dec axis
  present the native Alpaca mount runs end to end through the daemon
  (connect, status fields, J2000 GoTo and settle, abort mid-GoTo, tracking,
  a pulse, a jog stopped by its dead-man and one stopped by rate 0, a sync,
  park and unpark with a jog refused while parked, e-stop, disconnect) and
  passes the telescope conformance suite. With the Dec axis absent, a GoTo
  outside the firmware's 1° Dec window is refused with
  `409 mount_invalid_operation` and the firmware's message, a Dec jog is
  refused with `501 mount_unsupported` (the firmware reports
  CanMoveAxis(1) false), and the mount stays still.
- **ASCOM OmniSim** (`ACHERNAR_OMNISIM`, owner decision O9). The launcher
  must honour `OMNISIM_PORT`. Every native Alpaca driver (Telescope,
  Focuser, FilterWheel, ObservingConditions, SafetyMonitor) passes its
  conformance suite against its OmniSim device; the daemon connects all five
  through `/connect` and drives each one (GoTo, abort, pulse, jog, sync,
  park, focuser move, filter change, weather and safety reads, and an e-stop
  of a running GoTo that reaches the mount, focuser and filter wheel and
  stops the mount). Rotator,
  CoverCalibrator, Switch and Dome have no native driver until Tier 2
  (WP2-5): the client connects to and reads each one, and the daemon refuses
  them with 501. OmniSim runs are serialised, because every instance also
  binds UDP 32227 for discovery. One OmniSim v0.5.0 departure from the ASCOM
  standard is reported, not failed: its telescope answers an unknown `Action`
  with InvalidOperation (0x40B) instead of ActionNotImplemented (0x40C).
  OmniSim devices can be set to a lower interface version for Platform 6
  checks; that setting persists in OmniSim's own configuration, so restore it
  afterwards.

### 1.3 The in-process simulators

Every kind has an in-process simulator, selectable as driver `sim`:

```sh
curl -s -X POST $D/connect -H "$J" -d '{"equipment_type":"mount","driver":"sim"}'
curl -s -X POST $D/connect -H "$J" -d '{"equipment_type":"focuser","driver":"sim","option":"25000"}'
curl -s -X POST $D/connect -H "$J" -d '{"equipment_type":"filter_wheel","driver":"sim"}'
curl -s -X POST $D/connect -H "$J" -d '{"equipment_type":"weather","driver":"sim"}'
curl -s -X POST $D/connect -H "$J" -d '{"equipment_type":"safety_monitor","driver":"sim"}'
```

Simulator identities are `sim:{kind}` and their `device_connections` rows
have `driver = 'sim'`, so simulator runs never mix with hardware evidence.
Simulator runs never update a real rig's drift model or focuser scale (D18).

---

## 2. Preparing a rig session

1. **Build and test** on the machine that runs the rig: `cargo build --locked`
   and the CI suite (§1.1). Note the commit (`git rev-parse HEAD`); it goes
   into every qualification record.
2. **Configuration** (`achernar.toml`, or `ACHERNAR_CONFIG`). Nothing in it
   connects, scans or broadcasts (D13). Set at least:
   - `[site] latitude_deg`, `longitude_deg` (east positive), `elevation_m`,
     all three or none. Without a site, geometry features report "site
     unknown" and automatic slews are refused when the target altitude cannot
     be computed.
   - `[mount.safety] min_altitude`, and a `horizon_profile_path` if the
     horizon is obstructed.
   - Leave `[equipment] allow_broadcast_discovery = false`; address devices
     explicitly.
3. **Identity profiles** hold every rig-specific number (D17): backlash and
   who owns it, travel bounds, settle times, filter offsets, the temperature
   probe's role. TOML values are behaviour defaults and seeds only. Profiles
   are keyed by the identity key of §2.6 (`alpaca:{UniqueID}`, or
   `alpaca:{host}:{port}/{type}/{n}` without one). Connect once, read
   `identity.identity_key` from the `/connect` reply, then declare what the
   device needs (§3.2 shows the focuser recipe).
4. **Evidence.** Note the journal position before the first check
   (`sqlite3 $DB 'SELECT MAX(seq) FROM equipment_events'`, or the largest
   `seq` that `GET $D/equipment/events` returns); the journal range from there
   to the end of the session is the run's evidence, together with the
   database file.
5. **Safety first.** Before a safety monitor is trusted for unattended use,
   exercise the real park on the real mount (check M7 and S4): with
   `park_on_safety_violation = true` a confirmed unsafe reading parks the
   mount.

---

## 3. Acceptance checklist per device kind

Run each check against the real device, through the daemon's routes (or the
UI, which calls the same routes). A check passes only when every criterion in
its row holds. Record the measured values (latencies, errors, positions) with
the result. "Event X" means `GET $D/equipment/events?since_seq=<mark>&kinds=X`
shows it after a mark taken just before the check.

### 3.1 Mount (R1-R11, §3.2 M-a to M-k and M-n)

| # | Check | Proves | Run | Pass |
|---|---|---|---|---|
| M1 | Connect moves nothing | D16; identity from the management API; Platform 6/7 connect | Note the scope's position; `POST $D/connect {"equipment_type":"mount","driver":"alpaca","option":"http://<host>:<port>#device=<n>"}` | 200 with `identity.unique_id` and `qualification.state = implemented_unqualified`; the scope did not move, tracking and park state unchanged; a `device_connections` row with the identity key, driver and interface version |
| M2 | Status fields | §2.13, §5.7 item 2 | `GET $D/status` within 2 s of the connect | `mount` has `slewing`, `pier_side` (a string or null), `parked`, `alt`, `az`, `lst_h`, `frame`, `alignment_mode`, `can_flip`; `frame` matches the mount's EquatorialSystem; `lst_h` within 0.1 s of a reference LST |
| M3 | Site and clock | M-b | Look for events `site_mismatch` and `clock_mismatch` after the connect | None, or each one explained and resolved with `[site] sync` / `[mount] time_sync` (§7) |
| M4 | J2000 GoTo and settle | M-c, D10, O10 (GoTo gated on CanSlewAsync) | `POST $D/mount/slew {"ra":<h>,"dec":<deg>,"frame":"j2000"}` to a target well above the horizon | Events `slew_commanded`, `slew_started`, `slew_arrived`, `slew_settled` in order; `slew_settled.error_arcmin` within `[mount] arrival_tolerance_arcmin`; a plate solve of a frame taken after the settle lands within the mount's pointing accuracy (record the error) |
| M5 | Abort mid-GoTo | D4, D19, O10 (never gated) | Start a long GoTo, then `POST $D/mount/abort` | `abort_path_ran: true`; `slewing` false within 2 s (record the latency); event `slew_ended` with outcome `aborted`; the scope physically stopped |
| M6 | Tracking | M-e | `POST $D/mount/tracking {"enabled":false}`, then `true` | Status follows each write; event `tracking_changed`; with tracking on, two solves 10 min apart agree within the mount's tracking error |
| M7 | Park and unpark, verified | M-f | `POST $D/mount/park`, then `POST $D/mount/unpark` | Event `parked` only after AtPark reads true; the scope is at its physical park marks; `unparked` after AtPark reads false. A mount without CanPark answers 501 |
| M8 | Pulse guide | M-i, D20 | `POST $D/mount/pulse-guide {"direction":"east","duration_ms":300}` (and 1000 ms, to watch it in `GET $D/status`); repeat while PHD2 guides | Event `pulse_guide`; `mount.is_pulse_guiding` true while it runs and false after (when the mount reports IsPulseGuiding); the scope moved by about duration × guide rate; refused with 422 while PHD2 is guiding or calibrating |
| M9 | Limits | M-k, H13 | Temporarily raise `[mount.safety] min_altitude` above a target's altitude, restart, request that slew | 422 `safety_rejected` (event `safety_rejected`), no motion. Restore the value |
| M10 | Degraded, not lost | §2.8 | Interrupt the network for less than `lost_after_s` (default 10 s) | `device_health.mount.state` goes `degraded`, then `ready`; never `lost`; nothing parks |
| M11 | Lost, abort still runs | §2.8, D19 | Power the mount controller off | `lost` within `lost_after_s`; `POST $D/mount/abort` still answers (`abort_path_ran`), and only a retired handle answers 412 |
| M12 | Disconnect never parks | §2.6, D16 | `POST $D/disconnect {"equipment_type":"mount"}` while idle; then while slewing | Idle: disconnected, scope untouched. Slewing: 409 unless `"force": true`, which aborts first |
| M13 | Sync | M-g | With tracking on, take a frame, plate-solve it, then `POST $D/mount/sync {"ra":<h>,"dec":<deg>,"frame":"j2000"}` with the solved position | `[mount] sync_mode = "mount"` (default): event `sync_applied` with `before`, `after` and a `generation` one higher than the last sync's; `mount.ra_j2000`/`dec_j2000` read back within 5″ of the synced position; a solve of the next frame agrees within the mount's pointing accuracy (record the error). `sync_mode = "offset"`: `sync_applied` with `after: null` and the generation unchanged; the mount itself is not synced. Refused while parked |
| M14 | Jog and its dead-man | M-h, O10 point 3 (jog gated on CanMoveAxis, never on CanSlew), D19 | `POST $D/mount/move-axis {"axis":0,"rate_deg_s":<r>}` with `r` inside the mount's AxisRates; do not repeat it. Then start a jog again, repeat the request within `renew_within_ms` for a few seconds, and send `{"axis":0,"rate_deg_s":0}`. Then park and request a jog. Then a jog on an axis the mount does not move (axis 1 on a Dec-less mount) | The first reply has `status: moving` and `renew_within_ms` = `[mount] jog_deadman_ms` (default 1000); the axis moves; without renewal it stops within `jog_deadman_ms` plus one poll, with events `move_axis` (rate r) then `move_axis` (rate 0) and `operation_ended` failed with `jog_deadman_lapsed`. Renewed, it keeps moving; rate 0 stops it at once (`status: stopped`, `abort_path_ran: true`). Parked: 422 `safety_rejected`, no motion. An axis without CanMoveAxis: 501 `mount_unsupported`, nothing sent. A mount reporting CanSlew false still jogs |
| M15 | Center | M-n, H12, D20 | With a camera and the plate solver (ASTAP) configured, `POST $D/mount/center {"ra":<h>,"dec":<deg>,"frame":"j2000"}` (optional `tolerance_arcmin`, `max_attempts`, `exposure_s`, `gain`, `binning` override `[mount.center]`) | Accepted with a `command_id`; the busy set shows `centering`; PHD2 stopped before each slew; event `centered` with `error_arcmin` within the tolerance and the attempts used (or `center_failed` with its reason after `max_attempts`); one `pointing_samples` row with purpose `center` per solve. An abort or e-stop stops the loop's later motion |

### 3.2 Focuser (R12-R16, §3.3 F-a to F-i; A23 bench items)

Declare the device's facts in its profile first (D17). A focuser whose
firmware compensates backlash, with a probe reading the ambient air:

```sh
KEY='alpaca:<UniqueID>'   # from the /connect reply
sqlite3 $DB "INSERT INTO device_profiles (identity_key, kind, profile_json, updated_at)
  VALUES ('$KEY', 'focuser', '{}', strftime('%Y-%m-%dT%H:%M:%fZ','now'))
  ON CONFLICT(identity_key) DO NOTHING;
  UPDATE device_profiles SET
    profile_json = json_patch(profile_json, json('{
      \"backlash_owner\": \"firmware\", \"temperature_role\": \"ambient\",
      \"focuser\": {\"backlash_owner\": \"firmware\", \"temperature_role\": \"ambient\"}}')),
    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
  WHERE identity_key = '$KEY';"
```

The declaration is written both at the top level (where the focuser session
reads it today) and under `focuser` (the layout `profile.rs` documents); see
§10. Reconnect the focuser after a profile change.

| # | Check | Proves | Run | Pass |
|---|---|---|---|---|
| F1 | Connect and capabilities | §2.6, F-g | `POST $D/connect {"equipment_type":"focuser","driver":"alpaca","option":"..."}`; `GET $D/focuser` (the descriptor) and `GET $D/status` (its `focuser` block) | Descriptor: `capabilities.absolute`, `capabilities.max_step`, `capabilities.max_increment` and `capabilities.step_size_um` (a driver claim) as the device declares; `backlash_owner`, `backlash_model` and `temperature_role` from the profile. Status block: the same facts as `absolute`, `max_range`, `max_increment`, `step_size_um`, `backlash_owner` and `temperature_source`. No move was sent |
| F2 | Absolute move with read-back | F-a, A23 | `POST $D/focuser/moveto {"position":N}` | Status `position == N` once `active_op` is null; one `focuser_moves` row with `raw_start`, `raw_end == N`, `outcome = succeeded`; events `focuser_move_commanded`, `focuser_move_started`, `focuser_move_stopped`, `focuser_settled` |
| F3 | Backlash ownership | F-e | The move of F2, and a reversal | Firmware-owned: exactly one device move per request (no overshoot), `backlash_model.model = none`. Achernar-owned: the overshoot or absolute sequence of the profile, with logical and raw positions recorded |
| F4 | Repeatability (A23) | A23 | 10 approaches to one position from each side; read the position and focus a star (HFR) each time | Read-back equals the target every time; the HFR spread is within the seeing; record the spread in steps |
| F5 | Travel bounds | D9, F-a | `POST $D/focuser/moveto` past `[focus.safety] max_position` (or the profile's travel) | 422; nothing moved |
| F6 | No move while exposing | A06 | Start a long exposure, then a move | 422 until the exposure ends |
| F7 | Halt | F-c | `POST $D/focuser/abort` during a long move | The motor stops; event `focuser_halted` or `focuser_move_stopped` with the stopped position |
| F8 | Temperature | F-f, H6 | Compare `focuser.temperature` with a reference thermometer over an hour | Agreement within the probe's stated accuracy; `environment_log` rows with `source = 'probe'` and the device identity |
| F9 | Reconnect and origin | F-b, F-h, §2.15 | Power-cycle the controller while connected | Event `device_reconnected`; a relative focuser goes `unknown_after_reconnect` and refuses absolute moves; an absolute one re-reads its position |

### 3.3 Filter wheel (R17, §3.4 W-a to W-c)

| # | Check | Proves | Run | Pass |
|---|---|---|---|---|
| W1 | Connect and reconcile | W-b, D17 | Connect; `GET $D/filter-wheel` | Slot count from the device; names from the profile, else the device; TOML seeds not applied (`filter_config_mismatch` reports a difference) |
| W2 | Change handshake | W-c, v6 §8.2.1 | `POST $D/filter-wheel/change {"slot":k}` | Events `filter_change_requested`, `filter_moving`, `filter_arrived`, `filter_settled`, `filter_change_ended` in order; `slot == k` and `verified: true`; a `filter_switch_events` row with `handshake_ok`; the physical slot is k |
| W3 | Focus offsets | W-b, W-c step 9 | Offsets come from the identity profile, else the device's own `FocusOffsets` (a wheel whose driver declares offsets moves the focuser on every change; OmniSim's does). TOML seeds reach the profile only through `POST $D/filter-wheel/import-seed` (explicit, audited). Change between slots with different offsets | Event `filter_offset_applied` with the relative steps; the focuser moved by exactly the difference |
| W4 | Refused while exposing | §2.9 | A change during an exposure | 422 |

### 3.4 Guider: PHD2 (R24-R25, §3.5)

| # | Check | Proves | Run | Pass |
|---|---|---|---|---|
| G1 | Connect politely | G-a, D16 | `POST $D/connect {"equipment_type":"guider","driver":"phd2","option":"<host>:4400"}` | `guiding` block non-null in status; PHD2's equipment untouched (no `set_connected` unless `[guider] connect_equipment = true`) |
| G2 | Guide | G-c, G-h, G-i | `POST $D/guider/start {}` | Settle completes (`settle_done`); `guide_steps` rows match PHD2's GuideStep log unmodified; `rms_total` in arcsec agrees with PHD2's display |
| G3 | Dither | G-f, H10 | `POST $D/guider/dither {}` between exposures | Events `dithered`, `settle_done`; a `dither_events` row with the settle status; after the next capture, `next_frame_id` is filled |
| G4 | Stop around slews | D20 | Slew while guiding | The slew's reply has `was_guiding: true`; PHD2 stopped before the mount moved; guiding is restarted by the operator (Tier 1) |
| G5 | Loss | G-a | Close PHD2 | Device lost; `guider_lost` resume block |
| G6 | Guide-camera role | G-l | Try to connect the guide camera as Achernar's camera | `camera_role_conflict`, refused unless forced. Not wired yet (§10 item 6): until it is, never connect the guide camera to Achernar |

### 3.5 Weather and safety monitor (R18-R19, §3.9, §3.10)

| # | Check | Proves | Run | Pass |
|---|---|---|---|---|
| E1 | Weather fields | §3.9, D5 | `GET $D/weather` | Every implemented field with its value and age; unimplemented fields `null` with state `unsupported`, never 0; `environment_log` rows every `[weather] log_interval_s` |
| S1 | No monitor | §3.10 | `GET $D/safety` before connecting one | `state: none`; nothing triggers |
| S2 | Safe | §3.10 | Connect the monitor | `state: safe` while the device reads safe |
| S3 | One bad poll never parks | §2.8, §9.3 | Interrupt the monitor's link briefly | `state` goes `unknown` (automatic actions refused), the policy does not run, nothing parks |
| S4 | Confirmed unsafe | §3.10, O11 h | Make the monitor read unsafe (its test input, or cover the rain sensor) with the mount slewing and PHD2 guiding | In order: `safety_unsafe` resume block, PHD2 stopped, exposure aborted (if `abort_exposure_on_unsafe`), slew aborted, mount parked (or tracking off per `[safety] on_unsafe`); each action audited under the safety source; the mount physically at park |
| S5 | Manual resume | §3.10 | Make the monitor safe again; clear the `safety_unsafe` block with `POST $D/equipment/resume-blocks/clear {"id":...}` | Refused (409) while unsafe; allowed once safety reads safe. **Cannot pass yet**: today the route refuses a safety block even when safe (§10 item 1); record S5 as failed until that is fixed |

### 3.6 Camera hooks (OLS cameras; H1, H17, H22)

| # | Check | Proves | Run | Pass |
|---|---|---|---|---|
| C1 | Exposure window | H1, D7 | `POST $D/camera/capture {"exposure":S}` with the mount tracking | The frame's `exposure_device_windows` row is `finalised = 1` with mount, focuser, filter and guiding fields; `acquisition_json.equipment` present; FITS keywords present, a missing value omitted rather than 0 |
| C2 | Clock quality | §2.14, A06 | Inspect the row's `clock_quality` | gphoto2 stills `host_trigger_bracket`; stream cameras `host_arrival`; never `driver_exposure_start` unless the driver provides it |
| C3 | Resume blocks | H22 | With any block active, `POST $D/camera/capture {"exposure":1,"source":{"kind":"trigger","rule":"acceptance"}}` | 423; an operator capture runs and its provenance carries `resume_block_active` |
| C4 | Pointing sample | H12, M-4 | Plate-solve a tracked frame (`POST $D/platesolve/solve`) | A `pointing_samples` row with the mount frame, JNow flavour, pier side and sync generation |

### 3.7 Whole rig

| # | Check | Proves | Run | Pass |
|---|---|---|---|---|
| X1 | E-stop | §2.9, D19 | With the mount slewing and the focuser moving, `POST $D/equipment/estop {}` (the UI's e-stop button and the relay action `equipment.estop` call the same code; the relay path with every slot busy is covered by the CI spine tests) | Every device's result within 2 s (`elapsed_ms`); AbortSlew, MoveAxis 0 and tracking off at the mount (`estop_stops_tracking`), Halt at the focuser, `stop_capture` at PHD2; an `estop` resume block the operator clears with `POST $D/equipment/resume-blocks/clear {"id":...}` |
| X2 | Shutdown | §2.15 | Ctrl+C the daemon during a focuser move | The move halts, a jog stops, an Achernar slew aborts; the mount is not parked and PHD2 keeps guiding; open `device_connections` rows close with `end_class = shutdown` |
| X3 | Default opens nothing | D13 | Start the daemon with no `/connect` | Nothing connected, no broadcast (`scripts/windows/smoke-api.ps1` passes on Windows) |

### 3.8 Not yet acceptable

Rotator, cover calibrator, switch and dome have no native driver until Tier 2
(WP2-5); `/connect` answers 501 for them and `/drivers` lists them as
`planned`. The INDI focuser, wheel, weather and safety drivers and the native
INDI mount are Tier 2; LX200 and MoonLite devices are reached through
indiserver or ASCOM Remote (O4).

---

## 4. How a device identity becomes qualified (D21)

Qualification is per **device identity**, not per driver. A record names the
identity key and the exact driver and firmware it was earned with:

1. The device's kind passes every check of §3 that applies to it, on the
   real hardware, in one session, with the measured values recorded.
2. The run's evidence is kept: the database file (or a copy), the journal
   range (`seq` from/to), the daemon commit, and any frames and solves used.
3. The operator writes a qualification record into the identity's profile,
   under `qualification.<record_id>`:

   ```sh
   KEY='alpaca:<UniqueID>'
   REC='mount-20261107-1'
   sqlite3 $DB "UPDATE device_profiles SET
     profile_json = json_patch(profile_json, json('{\"qualification\": {\"$REC\": {
       \"procedure\": \"EQUIPMENT_RUNBOOK.md\",
       \"result\": \"pass\",
       \"kind\": \"mount\",
       \"identity_key\": \"$KEY\",
       \"driver\": \"alpaca\",
       \"driver_info\": \"<from device_connections>\",
       \"driver_version\": \"<from device_connections>\",
       \"interface_version\": 4,
       \"daemon_commit\": \"<git rev-parse HEAD>\",
       \"checks\": {\"M1\": \"pass\", \"M2\": \"pass\", \"M5\": \"pass, 840 ms\"},
       \"evidence\": {\"database\": \"<path>\", \"seq_from\": 0, \"seq_to\": 0},
       \"operator\": \"<name>\",
       \"at\": \"<UTC, ISO 8601>\"}}}')),
     updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
     WHERE identity_key = '$KEY';"
   ```

   The `qualification` section is a map keyed by record id
   (`profile::patch::qualification_record`), so later records never erase
   earlier ones. Write it between sessions, or while the daemon runs (the
   merge happens inside SQLite and loses nothing).
4. A record stays valid only for the identity, driver version and interface
   version it names. A firmware or driver update, a different ClientID-shared
   bridge in front of the device, or a different connection path means a new
   run.
5. Publication. `Qualification::Qualified { record_id }` is what `/connect`,
   `/drivers`, `/status` and `device_connections.qualification` would then
   report for that identity. **Today no code reads these records yet** (§10
   item 3), so a qualified device still shows `implemented_unqualified` until
   the spine publishes them. Record them anyway: the record, not the badge, is
   the evidence.

Nothing else may write `qualified`: not a driver, not a simulator run, not
CI. The conformance suite enforces that drivers never report it
(`only_rig_acceptance_qualifies`).

---

## 5. Worked example: the R50 + Z114 rig with the ESP32 Alpaca mount

This is the generic procedure applied to one rig. Nothing in the daemon knows
this hardware (D15): the mount is an ordinary ASCOM Alpaca telescope, the
camera an ordinary OLS gphoto2 camera, the guider an ordinary PHD2.

**The rig.** A Canon EOS R50 on the Z114 tube, on the printed fork mount whose
ESP32 firmware (`mount_cad_esp_32_firmware/achernar-mount-esp32`) serves an
ITelescopeV4 on port 11111; an SV105 guide camera in PHD2; optionally the
planned ESP32 focuser.

**What the mount declares, and what the checks should show:**

| Fact (from the device) | Consequence in Achernar | Checks |
|---|---|---|
| InterfaceVersion 4, DeviceState | Platform 7 `connect`/`connecting`; one `devicestate` GET per poll | M1, M2 |
| EquatorialSystem 1 (Topocentric) | `frame = jnow_apparent`; J2000 slews and solves are converted (about 22′ in 2026) | M2, M4, C4 |
| AlignmentMode 1 (Polar, a fork) | `alignment_mode = polar`, `can_flip = false` (D14) | M2 |
| SideOfPier constant 0 | `pier_side = "east"` always; no flip logic | M2 |
| CanSlew false, CanSlewAsync true | Correct for Alpaca (O10): GoTo uses the async slew only | M4 |
| CanPark true, CanSetPark false, CanFindHome false | Park is the power-on reference (HA 0 h, Dec +90°); home answers 501 | M7 |
| Guide rate 0.5× sidereal | Read from the mount; `[mount] guide_rate = 0.5` is only a fallback | M8 |
| Dec motor absent (`CFG_DEC_INSTALLED 0`) | GoTos more than 1° from the synced Dec and N/S pulses are refused by the firmware: expect `409 mount_invalid_operation` with the firmware's message, and no motion. The firmware reports CanMoveAxis(1) false, so a Dec jog is refused by Achernar with `501 mount_unsupported` before anything is sent; RA jogs work although CanSlew is false (O10) | M4, M8, M14 |
| Firmware limits HA ±5.5 h, Dec −10°…+90° | Enforced in firmware as well; Achernar's own limits (M9) refuse first | M9 |
| One connection per ClientID; motion stops only when the last client leaves | Achernar and PHD2 can share the mount; disconnecting Achernar never stops PHD2's guiding (D16) | M12, G1 |
| Clock from SNTP (needs internet) or a client | Without internet, set `[mount] time_sync = true` (UTC written once at connect when off by ≥ 10 s) or set the clock from another client; check M3 | M3 |

**Session outline.**

1. Power the mount at its park marks. Start PHD2 with the SV105 and the mount
   (PHD2 connects to the mount on its own; for a Dec-less mount set PHD2's
   Dec guide mode to Off).
2. `POST $D/connect {"equipment_type":"mount","driver":"alpaca","option":"http://achernar-mount.local:11111#device=0"}`
   (or the mount's IP). Run M1-M3: `frame` jnow_apparent, `alignment_mode`
   polar, `can_flip` false, `pier_side` east, `parked` true.
3. Unpark (M7 part 2). With the Dec motor absent: point the tube in Dec by
   hand, capture with the R50 (`POST $D/camera/capture`), plate-solve,
   `POST $D/mount/sync {"ra":..,"dec":..,"frame":"j2000"}` (M13), then run M4
   with a target within 1° of that Dec, and confirm the firmware's refusal
   for a target outside it.
4. M5 abort, M6 tracking, M8 pulses (E/W only without Dec), M14 jog (axis 0;
   axis 1 refused without Dec), M15 center once the camera and solver work,
   M9 limits, M10 and M11 by switching the mount's Wi-Fi off and on, M12.
5. PHD2 checks G1-G5 with the SV105 (G6: never connect the SV105 as the main
   camera).
6. Camera checks C1-C4 with the R50 through OLS `gphoto2`: windows with
   `clock_quality = host_trigger_bracket`.
7. X1 e-stop; then park (M7) and confirm the marks.
8. Record one qualification record per identity that passed (§4): the mount
   (`alpaca:<the firmware's UniqueID>`), the PHD2 instance
   (`phd2:<host>:4400/<profile>`), and later the ESP32 focuser after F1-F9.

The ESP32 focuser, once built, declares InterfaceVersion 3 (the Platform 6
path, no DeviceState), an absolute position, a DHT22 read through the
standard `temperature` member, and firmware-owned backlash: declare
`backlash_owner = firmware` and `temperature_role = ambient` in its profile
(§3.2) and run F1-F9; F4 is the A23 repeatability sweep.

---

## 6. Windows COM drivers through ASCOM Remote

There is no native COM layer (O5). A Windows mount, focuser, wheel, weather
station or safety monitor with only a COM (ASCOM Platform) driver is reached
as an Alpaca device:

1. Install ASCOM Remote (from the ASCOM Initiative) on the Windows PC that
   owns the device.
2. In the Remote Server setup, add one device per driver (for example
   Telescope 0 → the mount's COM driver), choose the listening port (11111 by
   default), and allow connections from other hosts if Achernar runs on
   another machine. Start the server.
3. In Achernar: `POST $D/discover {"equipment_type":"mount","driver":"alpaca","option":"http://<pc>:11111"}`
   lists what the server offers; connect with
   `{"equipment_type":"mount","driver":"alpaca","option":"http://<pc>:11111#device=0"}`.
4. The identity key is the UniqueID ASCOM Remote reports for that device. It
   is the bridge's identity in front of the driver, so qualify it as such:
   the record names ASCOM Remote's version and the COM driver's
   `DriverInfo`/`DriverVersion` (both appear in `device_connections`).
5. Interface versions and capabilities are whatever the COM driver
   implements; a Platform 6 driver takes the `Connected=true` path. Run the
   same checklist.

ASCOM Remote and Alpaca have no authentication: keep them on a trusted
network and never expose them to the internet.

---

## 7. INDI site and time

- In Tier 1 an INDI mount is reached only through the OLS `ols_indi` adapter
  (O3), and only in builds where the OLS bridge was compiled with libindi
  (not the default macOS or Windows builds); `/drivers` says which. Native
  INDI drivers arrive in Tier 2. The option is `host:port#device=<INDI device
  name>`.
- INDI mounts often expect the client to set the time and site
  (`TIME_UTC`, `GEOGRAPHIC_COORD`). For them, set `[site] sync = "to_mount"`
  (written while the mount is not slewing, then read back and verified) and
  `[mount] time_sync = true` (written once at connect, before any motion, only
  when the mount's clock is off by at least `time_tolerance_s`). The defaults
  stay passive (`verify_only`, `time_sync = false`): connecting writes
  nothing unless asked (D16).
- `CONNECTION` is device-wide in INDI. Achernar leaves it alone on disconnect
  (`[equipment] indi_disconnect_device = false`), so PHD2 or KStars keep the
  device.
- LX200 and MoonLite devices (including the ESP32 firmware's LX200 TCP port
  4030) are reached through `indi_lx200generic` and `indi_moonlite`, not a
  native serial driver (O4). Serial ports have one owner: let indiserver own
  it, and point PHD2 at the same indiserver.

---

## 8. PHD2 setup

1. In PHD2: connect the guide camera and the mount (PHD2's own ASCOM, INDI or
   Alpaca mount driver, or ST4 on camera), set the focal length and pixel size
   in the profile (the arcsec RMS uses PHD2's pixel scale), and calibrate.
2. Enable the event server: Tools → Enable Server. Instance n listens on port
   `4400 + n − 1`.
3. In Achernar:
   `POST $D/connect {"equipment_type":"guider","driver":"phd2","option":"<host>:4400"}`.
   Achernar reads PHD2's state and never switches PHD2's equipment on or off
   unless `[guider] connect_equipment = true`.
4. Guiding is PHD2's: `POST $D/guider/start`, `stop`, `pause`, `resume` and
   `dither` are thin commands to its server. Dither and `set_algo_param` run
   only between exposures, and dither only while guiding.
5. One guiding authority per mount (D20): while PHD2 guides or calibrates,
   Achernar refuses its own pulses; before an Achernar slew, park or center,
   PHD2 is stopped and the operator restarts it (Tier 1).
6. The guide camera belongs to PHD2: do not connect it as Achernar's camera
   (G-l).
7. Aggressiveness parameter names depend on PHD2's algorithm and are
   discovered (`get_algo_param_names`); leave `aggressiveness_param_*` empty
   unless discovery picks the wrong one. Policy writes stay off
   (`allow_policy_actuation = false`, Tier 2).

---

## 9. The physical autofocus gate (D18)

Automatic physical autofocus stays closed. `POST $D/focus/run` runs only with
the OLS simulator camera and a simulator focuser
(`validate_simulation_focus_run` in `src/api/focus.rs`); with real hardware it
answers `412 focus_unqualified` ("pending A03/A06/A13/A15/A23
qualification"). DSLR still focus (the R50) is refused separately: autofocus
needs a dedicated archived still exposure, which the still-stream callbacks do
not provide. Simulator runs are marked `training_label_eligible: false` and
never update a rig's drift model or focuser scale.

Lifting the gate is a separate, reviewed code change after the evidence
exists, never a configuration flag. The evidence this runbook produces for it:

- **A23** (the real focuser and mount protocols): F1-F9 on the real focuser,
  F4's repeatability sweep (approach and reversal), and M1-M15 on the mount;
- then the **A06/A13** focus timing trace (exposure identity after settle,
  unknown settings and pauses) with the real camera, and the **A15** real
  stellar sweep (annuli and blends), each as defined in
  `docs/sequential-remediation-plan-2026-10-08.md`.

The v6 autonomy ladder (v6 §7.11) and the policy actuation defaults (M-1
`set_algo_param`, automatic cloud pause, automatic resume after unsafe) stay
off for the same reason.

---

## 10. Known gaps found by the interop work

These were found while building and reviewing the interop tests. Each is
filed as a change note with its owner. The CI scenario asserts items 1 and 2
as they are today, so the fix of each fails the scenario at that step with a
message saying what to require instead; the others are noted where they
matter (§3.2, §4).

1. **Safety resume route.** `POST equipment/resume-blocks/clear` refuses every
   `safety_unsafe` and `safety_unknown` block, even once safety reads safe: it
   calls the generic block clear, and nothing calls
   `SafetySupervisor::request_resume`. Until the route reaches it, a safety
   block lasts until the daemon restarts (blocks are not persisted), and
   automatic captures stay refused with 423 meanwhile; check S5 cannot pass.
2. **`telemetry_log` (H19).** Nothing in production records the 1 Hz
   per-device telemetry series yet, so `telemetry_log` stays empty. TME and
   AstroM3Edge (v6 Model 9) and replay need it: until it is written, that v6
   data path has no ground truth from a session.
3. **Qualification records are not published.** No code reads
   `profile.qualification` records into `DeviceIdentity.qualification`
   (§4 step 5).
4. **Focuser profile layout.** The focuser session reads `backlash_owner`,
   `temperature_role`, `travel`, `settle_ms` and `relative_position` at the
   top level of the profile, while `profile.rs` documents a `focuser`
   section and the recorder's lost-position safeguard (§2.15) writes
   `focuser.relative_position`, which the session never reads. Declare
   focuser values in both places (§3.2) until the two agree. For a
   **relative** focuser the safeguard therefore does not work: after a
   dropped position update and a reconnect, a stale top-level origin can be
   restored. Until this is fixed, do not count F9 as passed for a relative
   focuser, and re-establish its origin by hand after a reconnect
   (`POST $D/focuser/set-position`).
5. **Unknown Action on Alpaca drivers.** The drivers pass a device's error for
   an unknown `Action` through as is; OmniSim's telescope answers 0x40B, so
   the conformance rule `unknown_action_is_unsupported` cannot hold for it.
   Checking `SupportedActions` first would make the result device-independent.
6. **Guide-camera role check (G-l).** `guider::camera_role_conflict` exists
   and is tested, but nothing calls it when a camera connects, so connecting
   the guide camera as the main camera is not refused.
