# Achernar Frontend Draft

Static prototype for the requested dashboard-to-run-room flow.

Open `index.html` in a browser to review the dashboard concept. Click
`Start Run Job` to move to `run.html`, which is the camera-centered run room.
The draft is intentionally separate from the React app and does not change
production behavior.

`daemon-status.sample.json` mirrors the browser-facing shape we want Django to
relay from the local daemon: camera, mount, focuser, and filter wheel status.
The static JS renders equipment cards from that fixture and falls back to the
same data inline if the file is opened without a local server.

## What This Tests

- A dashboard-first home for equipment, queue, logs, conditions, and run context.
- Equipment cards that represent daemon-backed status surfaced through Django.
- A Django-relay-shaped daemon status fixture rendered into live equipment
  cards.
- Drag or click equipment into the run context.
- Start Run Job persists selected equipment and opens a separate run page.
- Camera controls live on the right pane.
- Mount, focuser, filter wheel, and logs are available as top tabs, side windows,
  and bottom status panels around the camera view.

## Not Included Yet

- No AI planning/chat flow.
- No production React integration.
- No real API calls from this static draft.
