"""Background-estimation evaluation harness (BGE-09, BGE-14, BGE-22, BGE-23).

Seeded simulations with known background, noise and sources, a bridge to the
Rust daemon's ``bge_eval_bridge`` test, Python SEP reference methods, metrics
against simulated truth, and one benchmark command. See README.md.

Results from these fixtures are correctness checks under controlled,
simulated assumptions. They are not measured real-camera accuracy.
"""

import hashlib
import json

#: Version of the fixture generators. Bump when any rendered pixel can change.
GENERATOR_VERSION = "bge-fixtures-v1"
#: Version of the metric definitions and result layout.
HARNESS_VERSION = "bge-eval-harness-v1"
RESULTS_SCHEMA = "bge-eval-results-v1"
BRIDGE_SCHEMA = "bge-eval-bridge-v1"


def canonical_json(obj) -> str:
    """Canonical JSON identical to the daemon's ``canonical_json``."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def config_sha256(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(array) -> str:
    """sha256 of an array's C-order bytes (dtype and shape are not hashed)."""
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
