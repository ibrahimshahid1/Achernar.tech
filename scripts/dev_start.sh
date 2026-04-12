#!/usr/bin/env bash
# scripts/dev_start.sh — Start all Achernar services for local development
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "╔═══════════════════════════════════════════╗"
echo "║   Achernar Local Dev Server               ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# Activate Python venv
source "$ROOT_DIR/backend/.venv/bin/activate" 2>/dev/null || true

cleanup() {
    echo ""
    echo "Shutting down all services..."
    kill $(jobs -p) 2>/dev/null || true
    wait 2>/dev/null
    echo "All services stopped."
}
trap cleanup EXIT

# ── Django backend ──
echo "▶ Starting Django backend on :8000..."
USE_SQLITE=1 python "$ROOT_DIR/manage.py" runserver 0.0.0.0:8000 &
DJANGO_PID=$!
echo "  PID: $DJANGO_PID"

# ── Frontend dev server ──
echo "▶ Starting frontend dev server on :5173..."
cd "$ROOT_DIR/achernar-frontend"
npm run dev -- --host &
FRONTEND_PID=$!
echo "  PID: $FRONTEND_PID"

# ── Daemon (if Rust is available) ──
if command -v cargo &> /dev/null; then
    echo "▶ Starting Rust daemon on :3030..."
    cd "$ROOT_DIR/achernar-daemon"
    cargo run &
    DAEMON_PID=$!
    echo "  PID: $DAEMON_PID"
else
    echo "⚠ Skipping daemon — Rust toolchain not found"
fi

echo ""
echo "════════════════════════════════════════════"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo "  Daemon:   http://localhost:3030"
echo "════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop all services."
wait
