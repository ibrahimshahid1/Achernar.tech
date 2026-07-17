#!/usr/bin/env bash
# scripts/dev_setup.sh — One-shot development environment setup
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "╔═══════════════════════════════════════════╗"
echo "║   Achernar Dev Environment Setup          ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# ── Python backend ──
echo "▶ Setting up Python backend..."
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "  ⚠ $PYTHON_BIN not found; falling back to python3"
    PYTHON_BIN="python3"
fi
if [ ! -d "$ROOT_DIR/venv" ]; then
    "$PYTHON_BIN" -m venv "$ROOT_DIR/venv"
fi
source "$ROOT_DIR/venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r "$ROOT_DIR/requirements.txt"
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    print(f"  ⚠ Active venv is Python {sys.version.split()[0]}; Python 3.11 is the supported backend dev target.")
PY
echo "  ✓ Python environment ready"

# ── Frontend ──
echo "▶ Setting up frontend..."
cd "$ROOT_DIR/achernar-frontend"
if [ ! -d "node_modules" ]; then
    npm install
fi
echo "  ✓ Frontend dependencies ready"

# ── Rust daemon (check only) ──
echo "▶ Checking Rust toolchain..."
if command -v cargo &> /dev/null; then
    echo "  ✓ Rust toolchain found: $(rustc --version)"
else
    echo "  ⚠ Rust not found. Install via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
fi

# ── Environment file ──
if [ ! -f "$ROOT_DIR/.env" ]; then
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    echo "  ✓ Created .env from .env.example"
else
    echo "  ✓ .env already exists"
fi

echo ""
echo "Setup complete! Run 'scripts/dev_start.sh' to start all services."
