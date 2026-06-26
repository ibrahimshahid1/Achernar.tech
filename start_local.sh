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
set -uo pipefail

# Kill the whole process group on exit so background services stop cleanly.
trap "trap - SIGTERM && kill -- -$$ 2>/dev/null" SIGINT SIGTERM EXIT

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Shared relay token (must match between Django and the daemon).
export ACHERNAR_RELAY_DEV_TOKEN="${ACHERNAR_RELAY_DEV_TOKEN:-achernar-dev-token}"
# SQLite by default for friction-free local dev (see backend/settings/dev.py).
export USE_SQLITE="${USE_SQLITE:-1}"

wait_for() { # wait_for <url> <name> [timeout_s]
  local url="$1" name="$2" timeout="${3:-40}" i=0
  printf "  waiting for %s" "$name"
  until curl -fsS -o /dev/null "$url" 2>/dev/null; do
    i=$((i+1)); [ "$i" -ge "$timeout" ] && { echo " … not ready (continuing)"; return 1; }
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
python manage.py runserver 0.0.0.0:8000 &
wait_for "http://127.0.0.1:8000/api/sync/available-models/" "Django :8000"

# ── 2. Rust daemon (registers a reverse tunnel with Django) ──────────────────
echo "[2/3] Rust daemon…"
export ACHERNAR_HOST="${ACHERNAR_HOST:-127.0.0.1}"
export ACHERNAR_PORT="${ACHERNAR_PORT:-3030}"
export ACHERNAR_CLOUD_HTTP="${ACHERNAR_CLOUD_HTTP:-http://127.0.0.1:8000}"
export ACHERNAR_CLOUD_WS="${ACHERNAR_CLOUD_WS:-ws://127.0.0.1:8000}"
export ACHERNAR_HARDWARE_ID="${ACHERNAR_HARDWARE_ID:-achernar-local-daemon}"
( cd achernar-daemon && cargo run ) &
wait_for "http://127.0.0.1:3030/health" "daemon :3030" 120

# ── 3. React frontend ────────────────────────────────────────────────────────
echo "[3/3] React frontend…"
( cd achernar-frontend && pnpm dev ) &
wait_for "http://127.0.0.1:5173" "frontend :5173"

cat <<EOF

All services started.
  Django    : http://localhost:8000
  Frontend  : http://localhost:5173   (browser → Django relay path)
  Daemon    : http://localhost:3030   (desktop → direct path)

Smoke-test the relay end-to-end in another terminal:
  ./scripts/smoke_e2e.sh

Press Ctrl+C to stop all services.
EOF

wait
