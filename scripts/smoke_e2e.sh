#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Achernar end-to-end smoke test.
#
# Exercises both control loops against a running stack (start it first with
# ./start_local.sh) using the OpenLiveStacker *sim* camera driver — no physical
# hardware required:
#
#   1. Daemon direct path :  /health → /status → connect(sim) → status → stream
#   2. Django relay path  :  /api/equipment/connected → daemon-status → command
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
ok()   { echo "  ✓ $1"; PASS=$((PASS+1)); }
bad()  { echo "  ✗ $1"; FAIL=$((FAIL+1)); }

# get <url>                 -> body on stdout, http code in $HTTP
# post <url> <json>         -> body on stdout, http code in $HTTP
get()  { local b; b="$(curl -sS -w '\n%{http_code}' "$1" 2>/dev/null)"; HTTP="${b##*$'\n'}"; printf '%s' "${b%$'\n'*}"; }
post() { local b; b="$(curl -sS -w '\n%{http_code}' -H 'Content-Type: application/json' -d "$2" "$1" 2>/dev/null)"; HTTP="${b##*$'\n'}"; printf '%s' "${b%$'\n'*}"; }
jhas() { python3 -c "import sys,json;d=json.load(sys.stdin);print('1' if ($2) else '')" 2>/dev/null <<<"$1"; }

echo "── Daemon direct path ($DAEMON) ─────────────────────────────"
body="$(get "$DAEMON/health")"
[ "$HTTP" = "200" ] && ok "GET /health ($body)" || bad "GET /health (http=$HTTP) — is the daemon running?"

body="$(get "$DAEMON/api/v1/status")"
[ "$HTTP" = "200" ] && ok "GET /api/v1/status" || bad "GET /api/v1/status (http=$HTTP)"

echo "  connecting sim camera (SIM_DIR=$SIM_DIR)…"
body="$(post "$DAEMON/api/v1/connect" "{\"equipment_type\":\"camera\",\"driver\":\"sim\",\"option\":\"$SIM_DIR\"}")"
[ "$HTTP" = "200" ] && ok "POST /api/v1/connect (sim) → $(printf '%s' "$body" | head -c 120)" \
                    || bad "POST /api/v1/connect (http=$HTTP) $body"

body="$(get "$DAEMON/api/v1/status")"
if [ -n "$(jhas "$body" 'd.get("camera")')" ]; then ok "status now reports a camera"; else bad "status has no camera after connect"; fi

body="$(post "$DAEMON/api/v1/camera/stream/start" '{}')"
{ [ "$HTTP" = "200" ] || [ "$HTTP" = "202" ]; } && ok "POST /api/v1/camera/stream/start" || bad "stream/start (http=$HTTP) $body"

echo "── Django relay path ($DJANGO) ──────────────────────────────"
body="$(get "$DJANGO/api/sync/available-models/")"
[ "$HTTP" = "200" ] && ok "GET /api/sync/available-models/" || bad "available-models (http=$HTTP)"

body="$(get "$DJANGO/api/equipment/connected/")"
[ "$HTTP" = "200" ] && ok "GET /api/equipment/connected/ (relay or localhost fallback)" || bad "equipment/connected (http=$HTTP)"

body="$(get "$DJANGO/api/equipment/daemon-status/")"
{ [ "$HTTP" = "200" ] || [ "$HTTP" = "503" ]; } && ok "GET /api/equipment/daemon-status/ (http=$HTTP)" || bad "daemon-status (http=$HTTP)"

# Generic relay command (works once the daemon's reverse tunnel is registered;
# 200 = relayed result, 503 = no tunnel yet — both are acceptable here).
body="$(post "$DJANGO/api/equipment/command/" '{"action":"equipment.status"}')"
{ [ "$HTTP" = "200" ] || [ "$HTTP" = "202" ] || [ "$HTTP" = "503" ]; } && ok "POST /api/equipment/command/ (http=$HTTP)" || bad "command relay (http=$HTTP) $body"

echo "─────────────────────────────────────────────────────────────"
echo "Smoke test: $PASS passed, $FAIL failed."
[ "$FAIL" -eq 0 ]
