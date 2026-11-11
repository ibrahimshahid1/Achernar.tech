"""Real-frame qualification of the background estimators (BGE v2).

Real frames have no pixel truth. This package qualifies a background method
on real camera data with truth-free evidence (pull distributions, difference
pulls, negated-residual detections, injection-recovery, real darks with a
simulated sky, focus-sweep fits, repeatability, status failures) and paired
comparisons against the legacy method, all split by observing night, with
thresholds fixed before any hold-out night is analysed and a one-shot,
ledgered hold-out run. See docs/bge-real-frame-qualification.md (local
working doc) and the module docstrings.

Modules: ``manifest`` (bge-real-frames-v2), ``sidecar`` (field log),
``builders`` (directory + log, import_session DB, daemon DB + capture
archive), ``split`` (night split rule), ``methods`` (Python and Rust method
variants), ``masks``, ``realmetrics``, ``injection``, ``dark_injection``,
``stats``, ``thresholds``, ``holdout``, ``engine``, ``run_qualification``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

QUALIFICATION_VERSION = "bge-real-qualification-v1"
MANIFEST_SCHEMA_V1 = "bge-real-frames-v1"
MANIFEST_SCHEMA_V2 = "bge-real-frames-v2"
SIDECAR_SCHEMA = "bge-real-sidecar-v1"
SPLIT_RULE_SCHEMA = "bge-real-split-rule-v1"
THRESHOLDS_SCHEMA_V2 = "bge-real-thresholds-v2"
RESULTS_SCHEMA = "bge-real-qualification-results-v1"
LEDGER_SCHEMA = "bge-real-holdout-ledger-v1"

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[3]
DEFAULT_THRESHOLDS = PACKAGE_DIR / "thresholds-real-v1.json"
DEFAULT_SPLIT_RULE = PACKAGE_DIR / "split-rule-v1.json"
DEFAULT_LEDGER = PACKAGE_DIR / "holdout_ledger.jsonl"

#: The representation the daemon analyses (``processing/debayer.rs``
#: ``PROCESSING_VERSION``): bilinear CFA reconstruction, mean of R, G, B,
#: float32 linear ADU, with bad_pixel/invalid/edge/native_saturation masks.
#: Archived per capture as ``analysis/linear.fits`` + ``analysis/<mask>.fits``
#: + ``analysis/analysis.json`` (``storage/analysis_assets.rs``).
DAEMON_ANALYSIS_REPRESENTATION = "achernar-linear-bilinear-rgb-mean-v1"

FRAME_TYPES = ("light", "dark", "bias", "flat")
SPLITS = ("development", "holdout")
MASK_NAMES = ("bad_pixel", "invalid", "edge", "native_saturation")


def canonical_json(obj) -> str:
    """Canonical JSON identical to ``ml.eval.bge.canonical_json``."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def config_sha256(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
