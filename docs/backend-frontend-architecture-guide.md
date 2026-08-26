# Backend & Frontend — Architectural Repair Guide

**Written 2026-08-26.** Companion to `docs/Achernar-Technical-Audit.md`. Every problem below was reproduced against a running Django server and a running daemon, not inferred from reading.

**Read this first:** none of this blocks the observing campaign. The data path is browser-free — daemon → SQLite → `tools/import_session.py` → dataset builder. Do this work on cloudy nights and daylight hours. If you have a clear sky, close this file and go outside.

---

## The one idea

There is exactly one contract in this system: **the daemon's HTTP API**. Everything else is a transport that carries it.

Right now that contract is written down in **four** places, by hand, with no test that they agree:

| Where | What it declares | Count |
|---|---|---|
| `achernar-daemon/src/main.rs` | HTTP routes | 34 |
| `achernar-daemon/src/relay_commands.rs` | relay command names | 21 |
| `backend/apps/equipment/views.py::_local_command_fallback` | Django's action whitelist | 16 |
| `achernar-frontend/src/api/{daemon,cloud}.ts` | two client surfaces | ~20 each |

They already disagree. Django's whitelist is missing five commands the daemon handles (`equipment.disconnect`, `camera.descriptor`, `camera.stop_stream`, `mount.sync`, `focuser.position`), and **`focus.*` is missing from all of them** — autofocus exists only on direct HTTP, so a browser going through achernar.com cannot start a sweep.

Every problem in this guide is a consequence of that one fact. Fix the contract duplication and most of the list stops recurring.

---

## Backend

### B1 — Django invents hardware that isn't there *(highest severity)*

`GET /api/equipment/connected/` with no daemon reachable returns four seeded rows — "ASI6200MM Pro", "AP 1100GTO", "MoonLite NiteCrawler", "ZWO EFW 7x50" at "Backyard Obs" — each with `connected: true` and nothing marking them as fake. Verified live.

This is the same class of bug as the `max_adu` defect you already fixed in the daemon: a plausible value standing in for an unknown one. The daemon spent 1,254 lines learning to say `null`; the API in front of it says "ASI6200MM Pro".

**Fix.** Delete the seed fallback from the `connected` action. When no daemon is live, return `[]` with an explicit status, and let the UI render an empty state. Keep `seed_data.py` as a fixture script if you want demo data, but never as a fallback inside a live endpoint. If you want a demo mode, make it an explicit query parameter (`?demo=1`) so it can never be reached by accident.

### B2 — The translation layer fabricates zeros

`status_to_equipment()` does:

```python
'Temp': f'{float(camera.get("temperature") or 0):.1f} C',
'Position': int(focuser.get("position") or 0),
'RA': round(float(mount.get("ra") or 0), 2),
```

Verified live: the daemon returned `temperature: null` and Django rendered `"Temp": "0.0 C"`. A DSLR has no sensor thermometer, so `null` is the *correct* answer, and 0.0 °C is a lie that reads as a cold camera.

**Fix.** Two parts, and the second is the real one.

1. Immediately: propagate `None` and let the UI render "—".
2. Architecturally: **Django should not be reshaping daemon payloads at all.** `status_to_equipment` exists to turn daemon status into UI cards, which means UI presentation logic lives in Python, on the far side of a network hop, where the frontend can't see it and no test covers it. Move card construction into the frontend and have Django relay the daemon's payload verbatim. That deletes ~70 lines and closes this bug permanently rather than one field at a time.

### B3 — Autofocus is unreachable through the cloud

`focus.*` appears in neither `relay_commands.rs` nor Django's whitelist. `/focus/run`, `/focus/status`, `/focus/history`, `/focus/drift` are direct-HTTP-only.

Decide which this is:

- **Intentional** — the rig agent owns focus; the cloud watches. Then say so in `relay_commands.rs` with a comment naming the reason, and remove `useFocusRun` from the Django transport path so the UI doesn't offer a button that can't work.
- **A gap** — then add the six `focus.*` arms to the daemon dispatcher. They are thin: `start_focus_run`, `abort_focus_run`, `get_focus_status`, `get_focus_history`, `get_drift_prediction`, `get_drift_history` already exist as functions; the relay arm just calls them.

Note that a focus run takes minutes, and `send_relay_command` waits 3 seconds then returns `202 Accepted` with `pending: true`. So exposing focus over the relay also means the UI needs to poll `focus.status` rather than await the run — which is the right shape anyway, and is how `/focus/run` already behaves on the daemon side (it spawns a detached task).

### B4 — Unknown actions produce a misleading error

In `command()`, when `_local_command_fallback` returns `None` (action not in the whitelist), the code falls through to `send_relay_command(daemon, ...)` where `daemon` is `None`, producing:

```
503  "No connected daemon is available for relay commands."
```

Verified: I got this for `focus.run` **while the daemon was up and answering**. The message names the wrong cause, which is how an afternoon disappears.

**Fix.** Return `400 Bad Request` with the action name and the list of supported actions. An unsupported verb and an absent daemon are different failures.

### B5 — Make the command list one artifact

This is the fix that prevents B3 and B4 from coming back.

**Minimum version (an hour):** add a test that reads the match arms out of `relay_commands.rs` and asserts Django's whitelist covers them. Ugly, effective, catches drift the day it happens.

**Better version:** have the daemon expose its own command list — `GET /api/v1/commands` returning `[{name, method, path, description}]`, generated from the same table the dispatcher matches on. Django then relays anything on that list without a whitelist at all, and the frontend can render available controls from it. One source, three consumers, no drift possible.

The daemon already does exactly this for drivers (`GET /drivers` → `driver_capability_report()`), and the frontend already consumes it. This is the same pattern applied to commands.

### B6 — Delete the nine empty apps

`accounts`, `atlas`, `community`, `data_management`, `imaging`, `observatory`, `planning`, `processing` are 18-line `startapp` scaffolds — no models, no migrations, no URLs. They are in `INSTALLED_APPS`, so they load on every request and appear in every architecture diagram as though they exist.

**Fix.** Delete them. Re-create with `startapp` when you actually build one. An empty app is a claim, and right now `INSTALLED_APPS` claims sixteen subsystems where seven exist.

Keep: `core`, `equipment`, `realtime`, `sessions`, `ml`, `sync`. `ai_engine` is a judgment call — its `embeddings/models.py` is real (79 lines of ORM models for 512-D AstroM3 vectors, plus a pgvector migration), while `quality/`, `recommender/` and `anomaly/` are nine files of docstring-plus-TODO. Keep the app, delete the three empty sub-packages until there's something to put in them.

### B7 — Decide what Celery is for, then either build it or remove it

`celery_app.py` configures autodiscovery. `backend/tasks/__init__.py` declares five tasks — `export_training_data`, `refresh_embeddings`, `distribute_model`, `sync_daemon_telemetry`, `generate_session_report` — and every one raises `NotImplementedError`. `docker-compose.yml` starts a worker with nothing to run.

Of the five, only one is on the critical path in the next six months: **`distribute_model`**, which is how a trained ONNX bundle would reach a rig. And even that is better done as a pull — the daemon already has `GET /api/sync/available-models/`, which works and returns `{"models": []}`, so the daemon can poll and download rather than the cloud pushing.

**Fix.** Delete `tasks/__init__.py` and the compose service. Keep `celery_app.py` if you like, but an infrastructure component with zero real tasks is a maintenance cost and a misleading signal. Add it back when there is a job that genuinely must run out-of-band.

### B8 — Decide whether the cloud stores frames at all

`POST /api/sync/upload-session/` returns 501. `apps/sessions` defines `ImagingSession` and `Frame` models that mirror the daemon's SQLite schema, with `file_url` fields for S3 — and nothing populates them.

Before writing the upload endpoint, answer the prior question: **does the cloud need the pixels, or only the metadata?** Your campaign's frames live on the rig and get processed there. Uploading FITS to S3 costs money and solves a problem you do not currently have (multi-site collaboration, remote processing).

The defensible near-term scope is **metadata sync only**: sessions, frames rows, measurements, no pixels. That makes `upload-session` a bulk JSON insert, keeps `Frame.file_url` empty and honest, and gives you a cloud view of what was collected without an object store. Write that, and put a comment on `file_url` saying pixels stay on the rig until there is a reason for them not to.

### B9 — Security is dev-grade and should be labelled as such

A shared `achernar-dev-token` with a hardcoded default in two files, `AllowAny` on every equipment endpoint, `# TODO: Authenticate user` in `TelemetryConsumer`, `DEBUG = True` and `ALLOWED_HOSTS = ['*']` in dev settings.

This is fine for a rig on your own LAN and **not fine on a public host**. Don't fix it now — fix it before anything is exposed. What to do today is one line: put a comment at the top of `settings/prod.py` stating the three things that must be true before a public deployment (per-user auth on equipment endpoints, per-daemon ownership checks, real tokens), so future-you can't deploy it by accident.

---

## Frontend

### F1 — Delete the four helpers that call routes that don't exist

`cloudApi.listEquipmentMatrix`, `listAtlasTargets`, `listDocs`, `getExposureRecommendation` call `/api/equipment/matrix/`, `/api/atlas/targets/`, `/api/docs/`, `/api/ai-engine/recommend/exposure`. I confirmed all four return **404**. None is called from any component.

They're harmless today and actively misleading tomorrow — they read as "the cloud has an atlas and a recommender". Delete them. Re-add when the endpoint exists.

### F2 — Collapse the two transports into one client

`api/daemon.ts` (375 lines) and `api/cloud.ts` (172 lines) implement the same operations twice, and `getHardwareTransport()` picks between them at every call site:

```ts
const cameraApi = hardwareTransport === "django" ? cloudApi : daemonApi;
```

This is the frontend's copy of the duplication problem in B5. Every new capability must be added twice, and the two can silently diverge — which is how `focus` ended up available on one transport only.

**Fix.** One `achernarApi` object with one method per operation, and a single transport adapter underneath that decides how a call is delivered:

```ts
// One place decides HOW a command travels. Callers never know.
async function send(action: string, payload = {}) {
  return transport === "django"
    ? postJson(`${cloudBase}/equipment/command/`, { action, payload })
    : postJson(`${daemonBase}/${routeFor(action)}`, payload);
}
```

The relay already speaks a uniform `{action, payload}` shape, so the Django side is a single function. The daemon side needs an action→route map, which is the same table B5 proposes the daemon publish — so do B5 first and this becomes a lookup rather than a hand-written map.

### F3 — Stop hardcoding the driver list

`CameraControls.tsx` opens with a hardcoded `DRIVERS` array of ten drivers all marked `compiled: true, enabled: true`. The daemon already answers this correctly at runtime — I verified `GET /drivers` on this Mac reports `indi`, `indigo` and `wdir` as **not compiled**, which the hardcoded array claims are available.

The component does fetch `driverCapabilities` and prefers it when present. So the constant is a fallback that is wrong by construction. **Delete it** and render a loading state until the real report arrives. A control that offers a driver the build doesn't contain produces a failure the operator cannot diagnose.

### F4 — Make the stub pages honest or absent

`Queue.tsx`, `Equipment.tsx`, and the `mount/`, `imaging/`, `cloud/`, `settings/` component modules return placeholder text — several literally render `"— stub"`. They're in the router, so they're reachable.

Either remove them from the router until they do something, or give them a real empty state that says what the page will do and what's required first. A page that says "Mount Status — stub" to an operator at 2 a.m. is worse than a 404.

### F5 — Decide what the frontend is for, this year

The honest scope question. Your Windows testing doc already states the product boundary: the rig agent is the product, the site is the control surface. The campaign runs without a browser at all.

So the frontend's job for the next six months is narrow and worth stating explicitly somewhere in the repo:

1. **Live view + camera control** — works today, `CameraControls.tsx` and `MjpegLiveView.tsx` are real.
2. **A focus-run view** — the V-curve, live, as it's collected. This is the highest-value thing you could build in the UI, because it's how you tell a good sweep from a bad one *while you're still outside and can redo it*. RB-2's acceptance criteria (clean V, interior minimum, ≥15 stars at minimum, ≥8 at ends) are checkable on a chart in one glance.
3. **Nothing else** until data collection is running.

Item 2 is the one piece of frontend work that would actually pay for itself this autumn.

---

## Order of work

Grouped by what they cost and what they buy.

**One sitting each, do first — these are deletions:**
- F1 delete four dead helpers
- F3 delete hardcoded `DRIVERS`
- B6 delete nine empty apps
- B7 delete Celery tasks + compose service
- B4 return 400 instead of a misleading 503

Deleting misleading code is the highest-value-per-minute work in the repo right now, and none of it can break anything that works.

**Half a day each, do next — these are correctness:**
- B1 remove the seed fallback (this one is a genuine correctness bug)
- B2 stop coercing `null` → `0.0`
- B5 minimum version: a drift test between the daemon dispatcher and Django's whitelist

**A weekend each, when you want the architecture fixed:**
- B5 full version: `GET /api/v1/commands` as the single contract
- F2 collapse the two clients onto that contract
- B2 architectural half: move card construction out of Django
- B3 decide and implement the focus-over-relay question

**When it's actually needed, not before:**
- B8 metadata-only session sync
- B9 real auth, before any public exposure
- F5 item 2: the live V-curve view — the exception, worth doing *during* the campaign because it improves the data you collect

**Sequencing note.** B5 is the keystone. Doing it before F2 turns F2 from "write a mapping by hand" into "read the mapping the daemon publishes". Doing F2 first means writing that map twice.
