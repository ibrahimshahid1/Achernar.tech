#!/usr/bin/env bash
# scripts/check_all.sh — Run all verification checks across all layers
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "╔═══════════════════════════════════════════╗"
echo "║   Achernar — Full Verification Suite      ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

PASS=0
FAIL=0
SKIP=0

check() {
    local name="$1"
    shift
    echo -n "  $name ... "
    if output=$("$@" 2>&1); then
        echo "✓"
        PASS=$((PASS + 1))
    else
        echo "✗"
        echo "    $output" | head -5
        FAIL=$((FAIL + 1))
    fi
}

skip() {
    local name="$1"
    local reason="$2"
    echo "  $name ... ⏭ ($reason)"
    SKIP=$((SKIP + 1))
}

# ── Rust ──
echo "▶ Rust Daemon"
if command -v cargo &> /dev/null; then
    check "cargo check" bash -c "cd '$ROOT_DIR/achernar-daemon' && cargo check --workspace 2>&1"
else
    skip "cargo check" "Rust toolchain not installed"
fi

# ── Django ──
echo "▶ Django Backend"
source "$ROOT_DIR/backend/.venv/bin/activate" 2>/dev/null || true
check "manage.py check" bash -c "cd '$ROOT_DIR' && USE_SQLITE=1 python manage.py check 2>&1"

# ── Frontend ──
echo "▶ Frontend"
check "tsc --noEmit" bash -c "cd '$ROOT_DIR/achernar-frontend' && npx tsc --noEmit 2>&1"

# ── ML ──
echo "▶ ML Training"
check "model imports" bash -c "cd '$ROOT_DIR' && python -c 'from ml.models import focus_quality_cnn' 2>&1"

# ── Summary ──
echo ""
echo "════════════════════════════════════════════"
echo "  Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "════════════════════════════════════════════"

exit $FAIL
