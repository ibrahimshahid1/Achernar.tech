#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Achernar — start the full local stack (Django + Rust daemon + React frontend).
#
#   ./start_local.sh
#
# Defaults to SQLite + the in-memory Channels layer so no Postgres/Redis is
# required for local development. Set USE_SQLITE= (empty) and provide DB_* env
# to use Postgres instead. Ctrl+C stops everything.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Stop only processes started by this launcher, preserving other open work.
CHILD_PIDS=()
stop_tree() {
  local pid="$1" child
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do stop_tree "$child"; done
  kill -TERM "$pid" 2>/dev/null || true
}
cleanup() { for pid in "${CHILD_PIDS[@]}"; do stop_tree "$pid"; done; }
trap cleanup EXIT
trap 'exit 130' INT TERM

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

for port in 8000 3030 5173; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is already in use. Reuse that stack or stop its owner before launching another."
    exit 1
  fi
done
# Hash-verified research bundles are loaded but inference remains off until
# explicitly enabled in the camera panel. Environment overrides are preserved.
if [ -f "$ROOT/ml/artifacts/coc/human-teacher-tiny-v1/manifest.json" ]; then
  export ACHERNAR_COC_BUNDLE="${ACHERNAR_COC_BUNDLE:-$ROOT/ml/artifacts/coc/human-teacher-tiny-v1}"
fi
if [ -f "$ROOT/ml/artifacts/coc/human-tiny-v1/manifest.json" ]; then
  export ACHERNAR_COC_BASELINE_BUNDLE="${ACHERNAR_COC_BASELINE_BUNDLE:-$ROOT/ml/artifacts/coc/human-tiny-v1}"
fi
export DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-backend.settings.dev}"

# Shared relay token (must match between Django and the daemon).
export ACHERNAR_RELAY_DEV_TOKEN="${ACHERNAR_RELAY_DEV_TOKEN:-achernar-dev-token}"
# SQLite by default for friction-free local dev (see backend/settings/dev.py).
export USE_SQLITE="${USE_SQLITE:-1}"
export ACHERNAR_DJANGO_HOST="${ACHERNAR_DJANGO_HOST:-127.0.0.1}"
export ACHERNAR_FRONTEND_HOST="${ACHERNAR_FRONTEND_HOST:-127.0.0.1}"

wait_for() { # wait_for <url> <name> [timeout_s]
  local url="$1" name="$2" timeout="${3:-40}" i=0
  printf "  waiting for %s" "$name"
  until curl -fsS -o /dev/null "$url" 2>/dev/null; do
    i=$((i+1)); [ "$i" -ge "$timeout" ] && { echo " … readiness failed"; return 1; }
    printf "."; sleep 1
  done
  echo " ✓"
}

# ── 1. Django backend (ASGI via daphne — required for the WS relay) ──────────
echo "[1/3] Django backend…"
if   [ -f venv/bin/activate ];          then source venv/bin/activate
elif [ -f backend/.venv/bin/activate ]; then source backend/.venv/bin/activate
else echo "  (no virtualenv found; using system python)"; fi
export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
python manage.py migrate --noinput
python -m daphne -b "$ACHERNAR_DJANGO_HOST" -p 8000 backend.asgi:application &
CHILD_PIDS+=("$!")
wait_for "http://127.0.0.1:8000/api/sync/available-models/" "Django :8000"

# ── 2. Rust daemon (registers a reverse tunnel with Django) ──────────────────
echo "[2/3] Rust daemon…"
export ACHERNAR_HOST="${ACHERNAR_HOST:-127.0.0.1}"
export ACHERNAR_PORT="${ACHERNAR_PORT:-3030}"
export ACHERNAR_CLOUD_HTTP="${ACHERNAR_CLOUD_HTTP:-http://127.0.0.1:8000}"
export ACHERNAR_CLOUD_WS="${ACHERNAR_CLOUD_WS:-ws://127.0.0.1:8000}"
export ACHERNAR_HARDWARE_ID="${ACHERNAR_HARDWARE_ID:-achernar-local-daemon}"
( cd achernar-daemon && cargo run --locked ) &
CHILD_PIDS+=("$!")
wait_for "http://127.0.0.1:3030/health" "daemon :3030" 120

# ── 3. React frontend ────────────────────────────────────────────────────────
echo "[3/3] React frontend…"
if [ "${ACHERNAR_NATIVE:-0}" = "1" ]; then
  pnpm --dir achernar-desktop tauri dev &
else
(
  cd achernar-frontend
  if [ -x node_modules/.bin/vite ]; then
    node_modules/.bin/vite --host "$ACHERNAR_FRONTEND_HOST"
  else
    pnpm dev --host "$ACHERNAR_FRONTEND_HOST"
  fi
) &
fi
CHILD_PIDS+=("$!")
wait_for "http://127.0.0.1:5173" "frontend :5173" 120

cat <<EOF

All services started.
  Django    : http://localhost:8000
  Frontend  : http://localhost:5173   (browser → Django relay path)
  Daemon    : http://localhost:3030   (desktop → direct path)

Check the COC model and relay endpoints in another terminal:
  ./scripts/check_coc_local.sh

Press Ctrl+C to stop all services.
EOF

wait
