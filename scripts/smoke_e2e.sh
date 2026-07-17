#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Achernar end-to-end smoke test.
#
# Exercises both control loops against a running stack (start it first with
# ./start_local.sh) using the OpenLiveStacker *sim* camera driver — no physical
# hardware required:
#
#   1. Daemon direct path :  health/status → sim camera/focuser → stream → focus
#   2. Django relay path  :  status/command → stream relay → preview snapshot
#
# Usage:   ./scripts/smoke_e2e.sh
# Env:     DAEMON_URL (http://127.0.0.1:3030)  DJANGO_URL (http://127.0.0.1:8000)
#          SIM_DIR (…/OpenLiveStacker_Drivers/OpenLiveStacker/sim)
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAEMON="${DAEMON_URL:-http://127.0.0.1:3030}"
DJANGO="${DJANGO_URL:-http://127.0.0.1:8000}"
SIM_DIR="${SIM_DIR:-$ROOT/OpenLiveStacker_Drivers/OpenLiveStacker/sim}"

PASS=0; FAIL=0
BODY=""
HTTP=""
ok()   { echo "  ✓ $1"; PASS=$((PASS+1)); }
bad()  { echo "  ✗ $1"; FAIL=$((FAIL+1)); }

# get <url>                 -> body in $BODY, http code in $HTTP
# post <url> <json>         -> body in $BODY, http code in $HTTP
get()  { local b; b="$(curl -sS -w '\n%{http_code}' "$1" 2>/dev/null || true)"; HTTP="${b##*$'\n'}"; BODY="${b%$'\n'*}"; }
post() { local b; b="$(curl -sS -w '\n%{http_code}' -H 'Content-Type: application/json' -d "$2" "$1" 2>/dev/null || true)"; HTTP="${b##*$'\n'}"; BODY="${b%$'\n'*}"; }
jhas() { python3 -c "import sys,json;d=json.load(sys.stdin);print('1' if ($2) else '')" 2>/dev/null <<<"$1"; }
json_camera_connect() {
  python3 -c 'import json,sys; print(json.dumps({"equipment_type":"camera","driver":"sim","option":sys.argv[1]}))' "$SIM_DIR"
}

echo "── Daemon direct path ($DAEMON) ─────────────────────────────"
get "$DAEMON/health"; body="$BODY"
[ "$HTTP" = "200" ] && ok "GET /health ($body)" || bad "GET /health (http=$HTTP) — is the daemon running?"

get "$DAEMON/api/v1/status"; body="$BODY"
[ "$HTTP" = "200" ] && ok "GET /api/v1/status" || bad "GET /api/v1/status (http=$HTTP)"

echo "  connecting sim camera (SIM_DIR=$SIM_DIR)…"
post "$DAEMON/api/v1/connect" "$(json_camera_connect)"; body="$BODY"
[ "$HTTP" = "200" ] && ok "POST /api/v1/connect (sim) → $(printf '%s' "$body" | head -c 120)" \
                    || bad "POST /api/v1/connect (http=$HTTP) $body"

post "$DAEMON/api/v1/connect" '{"equipment_type":"focuser","driver":"sim","option":"10000"}'; body="$BODY"
[ "$HTTP" = "200" ] && ok "POST /api/v1/connect (sim focuser) → $(printf '%s' "$body" | head -c 120)" \
                    || bad "POST /api/v1/connect focuser (http=$HTTP) $body"

get "$DAEMON/api/v1/status"; body="$BODY"
if [ -n "$(jhas "$body" 'd.get("camera")')" ]; then ok "status now reports a camera"; else bad "status has no camera after connect"; fi
if [ -n "$(jhas "$body" 'd.get("focuser")')" ]; then ok "status now reports a focuser"; else bad "status has no focuser after connect"; fi

post "$DAEMON/api/v1/camera/stream/start" '{}'; body="$BODY"
{ [ "$HTTP" = "200" ] || [ "$HTTP" = "202" ]; } && ok "POST /api/v1/camera/stream/start" || bad "stream/start (http=$HTTP) $body"

echo "  running deterministic autofocus on sim stream…"
post "$DAEMON/api/v1/focus/run" '{"method":"classical","step_size":50}'; body="$BODY"
if [ "$HTTP" = "200" ] && [ -n "$(jhas "$body" 'd.get("state") == "Converged" and d.get("best_position") is not None')" ]; then
  ok "POST /api/v1/focus/run converged → $(printf '%s' "$body" | head -c 120)"
else
  bad "focus/run (http=$HTTP) $body"
fi

echo "── Django relay path ($DJANGO) ──────────────────────────────"
get "$DJANGO/api/sync/available-models/"; body="$BODY"
[ "$HTTP" = "200" ] && ok "GET /api/sync/available-models/" || bad "available-models (http=$HTTP)"

get "$DJANGO/api/equipment/connected/"; body="$BODY"
[ "$HTTP" = "200" ] && ok "GET /api/equipment/connected/ (relay or localhost fallback)" || bad "equipment/connected (http=$HTTP)"

get "$DJANGO/api/equipment/daemon-status/"; body="$BODY"
{ [ "$HTTP" = "200" ] || [ "$HTTP" = "503" ]; } && ok "GET /api/equipment/daemon-status/ (http=$HTTP)" || bad "daemon-status (http=$HTTP)"

# Generic relay command. This must be a real relay response; 503 means the
# reverse tunnel is missing and the cloud/browser path is not proven.
post "$DJANGO/api/equipment/command/" '{"action":"equipment.status","wait_seconds":5}'; body="$BODY"
if [ "$HTTP" = "200" ] && [ -n "$(jhas "$body" 'd.get("ok") is True and d.get("payload", {}).get("connected") is True')" ]; then
  ok "POST /api/equipment/command/ relayed equipment.status"
else
  bad "command relay (http=$HTTP) $body"
fi

post "$DJANGO/api/equipment/camera/start-stream/" '{}'; body="$BODY"
[ "$HTTP" = "200" ] && ok "POST /api/equipment/camera/start-stream/ via relay" || bad "relay stream/start (http=$HTTP) $body"

sleep 1
post "$DJANGO/api/equipment/camera/snapshot/" '{}'; body="$BODY"
if [ "$HTTP" = "200" ] && [ -n "$(jhas "$body" 'd.get("data_base64") and d.get("width", 0) > 0 and d.get("height", 0) > 0')" ]; then
  ok "POST /api/equipment/camera/snapshot/ returns relayed preview"
else
  bad "relay snapshot (http=$HTTP) $body"
fi

echo "─────────────────────────────────────────────────────────────"
echo "Smoke test: $PASS passed, $FAIL failed."
[ "$FAIL" -eq 0 ]
