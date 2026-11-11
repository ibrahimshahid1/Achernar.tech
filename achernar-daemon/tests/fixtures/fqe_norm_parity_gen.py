"""Expected values of the daemon's FQE normalization/inference parity tests.

Regenerates the constants in achernar-daemon/src/vision/preprocess.rs:

* ``normalization_is_bit_identical_to_the_training_normalization``: sha256 of
  each fixture crop (float32 little-endian) and of its
  ``ml/preprocess/normalize.py::robust_normalize`` output;
* ``reference_graph_inference_matches_python_onnxruntime``: ``(mu_log_hfr,
  log_var)`` from Python onnxruntime (CPU provider) running
  ``models/focus_quality/1.0.0/model.onnx`` on the normalized crops.

Reference environment (FQE run 20261111T0038, package D): numpy 2.5.4,
onnxruntime 1.31.0. Run from the repository's ml/ directory:

    PYTHONDONTWRITEBYTECODE=1 <venv>/bin/python -B \
        ../achernar-daemon/tests/fixtures/fqe_norm_parity_gen.py           # print
    ... fqe_norm_parity_gen.py --check   # compare with the constants in preprocess.rs

The crops use integer arithmetic only (plus one IEEE float32 division by 3 for
the odd crops), so Rust's ``shared_fixture_crop`` builds bit-identical inputs:
an integer LCG for noise, an integer Lorentzian-like star, RGB-mean thirds on
odd crops, a noise-only crop (k=3), an exactly flat crop (k=4, hi <= lo) and a
50 000-ADU sky (k=6, float32 precision). Keep this file and
``shared_fixture_crop`` in step.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "ml"))
from preprocess.normalize import robust_normalize  # noqa: E402

MASK = (1 << 64) - 1
N_CROPS = 7
CROP = 64
GRAPH = REPO / "models" / "focus_quality" / "1.0.0" / "model.onnx"
RUST_TEST = REPO / "achernar-daemon" / "src" / "vision" / "preprocess.rs"


class Lcg:
    def __init__(self, seed: int) -> None:
        self.state = seed & MASK

    def next(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & MASK
        return self.state >> 33


def crop(k: int, n: int = CROP) -> np.ndarray:
    rng = Lcg(0x5EED + k)
    sky = 100 + 37 * k if k != 6 else 50_000
    amp = 400 * (k + 1) if k not in (3, 4) else 0
    if k == 6:
        amp = 15_000
    noise_mod = 41 if k != 4 else 1
    cx, cy = 32 + (k % 3) - 1, 32 - (k % 2)
    width2 = 2 + 3 * k
    out = np.empty((n, n), dtype=np.float32)
    for y in range(n):
        for x in range(n):
            r2 = (x - cx) ** 2 + (y - cy) ** 2
            star = (amp * width2) // (width2 + r2)
            out[y, x] = np.float32(sky + (rng.next() % noise_mod) + star)
    if k % 2 == 1:
        out = out / np.float32(3.0)
    return out


def sha256_f32(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def generate(graph: Path) -> tuple[list[tuple[str, str]], list[tuple[float, float]]]:
    import onnxruntime as ort

    crops = [crop(k) for k in range(N_CROPS)]
    normalized = [robust_normalize(c)[0] for c in crops]
    hashes = []
    for c, z in zip(crops, normalized):
        assert z.dtype == np.float32
        hashes.append((sha256_f32(c), sha256_f32(z)))
    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    batch = np.stack(normalized)[:, None].astype(np.float32)
    mu, log_var = session.run(None, {"image": batch})
    outputs = [
        (float(np.ravel(mu)[k]), float(np.ravel(log_var)[k])) for k in range(N_CROPS)
    ]
    return hashes, outputs


def rust_constants(path: Path) -> tuple[list[tuple[str, str]], list[tuple[float, float]]]:
    """The expected values as written in preprocess.rs's two tests."""
    source = path.read_text()
    norm_test = source.split("fn normalization_is_bit_identical_to_the_training_normalization")[1]
    norm_test = norm_test.split("#[test]")[0]
    digests = re.findall(r'"([0-9a-f]{64})"', norm_test)
    hashes = list(zip(digests[0::2], digests[1::2]))
    ort_test = source.split("fn reference_graph_inference_matches_python_onnxruntime")[1]
    ort_test = ort_test.split("let python")[1].split("];")[0]
    number = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    pairs = re.findall(rf"\(\s*{number}\s*,\s*{number}\s*\)", ort_test)
    return hashes, [(float(a), float(b)) for a, b in pairs]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--graph", type=Path, default=GRAPH)
    parser.add_argument("--check", action="store_true", help="compare with preprocess.rs")
    args = parser.parse_args()

    import onnxruntime as ort

    hashes, outputs = generate(args.graph)
    print(f"numpy {np.__version__}, onnxruntime {ort.__version__}, graph {args.graph}")
    for k, ((raw, norm), (mu, log_var)) in enumerate(zip(hashes, outputs)):
        print(f"k={k} raw_sha={raw} norm_sha={norm} mu={mu!r} log_var={log_var!r}")
    if not args.check:
        return 0

    rust_hashes, rust_outputs = rust_constants(RUST_TEST)
    ok = rust_hashes == hashes and len(rust_outputs) == N_CROPS
    if rust_hashes != hashes:
        print("MISMATCH: sha256 constants differ from preprocess.rs", file=sys.stderr)
    for k, ((mu, log_var), (rust_mu, rust_log_var)) in enumerate(zip(outputs, rust_outputs)):
        # The Rust literals are Python reprs of float32 outputs widened to float64.
        if abs(mu - rust_mu) > 1e-12 or abs(log_var - rust_log_var) > 1e-12:
            ok = False
            print(f"MISMATCH: crop {k} onnxruntime output differs", file=sys.stderr)
    print("check:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
