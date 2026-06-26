#!/usr/bin/env bash
# Download AstroM3 pretrained weights from HuggingFace into ml/checkpoints/astrom3/.
#
#   ./scripts/download_astrom3.sh            # core set (CLIP base + 4 classifiers)
#   ./scripts/download_astrom3.sh --set all  # + the 5 seed variants
#
# Needs Python 3.10+ and network access to huggingface.co.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -c "import huggingface_hub" 2>/dev/null || {
  echo "Installing huggingface_hub…"
  python3 -m pip install --quiet huggingface_hub
}

exec python3 ml/astrom3/download.py "$@"
