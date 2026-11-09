"""Subprocess worker: time one Python method on a raw float32 frame.

Runs in its own process so ``ru_maxrss`` is the peak of this workload only:
``python -m ml.eval.bge.timing_worker --pixels F --width W --height H --method M``.
Prints one JSON object.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time


def _rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)  # macOS bytes, Linux KiB


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixels", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--method", required=True, choices=["python_sep", "python_global_scalar"])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args(argv)

    import numpy as np

    rss_start = _rss_bytes()
    image = np.fromfile(args.pixels, dtype="<f4").reshape(args.height, args.width)
    rss_loaded = _rss_bytes()
    from ml.eval.bge import python_methods as M

    def once():
        t0 = time.perf_counter()
        if args.method == "python_sep":
            est = M.python_sep_background(image, {})
        else:
            est = M.python_global_scalar(image, {})
        t1 = time.perf_counter()
        meas = M.detect_and_measure(image, {}, est)
        t2 = time.perf_counter()
        return {"background_ms": 1e3 * (t1 - t0), "detect_measure_ms": 1e3 * (t2 - t1),
                "total_ms": 1e3 * (t2 - t0)}, est, meas

    warmup, _, _ = once()
    runs = []
    for _ in range(args.repeats):
        timing, est, meas = once()
        runs.append(timing)
    totals = sorted(r["total_ms"] for r in runs)

    def pct(q):
        return float(np.percentile(totals, q))

    print(json.dumps({
        "method": args.method,
        "width": args.width,
        "height": args.height,
        "repeats": args.repeats,
        "warmup": warmup,
        "runs": runs,
        "total_ms_median": pct(50),
        "total_ms_p95": pct(95),
        "background_ms_median": float(np.median([r["background_ms"] for r in runs])),
        "detect_measure_ms_median": float(np.median([r["detect_measure_ms"] for r in runs])),
        "peak_rss_bytes": _rss_bytes(),
        "rss_at_start_bytes": rss_start,
        "peak_rss_after_load_bytes": rss_loaded,
        "background_status": est.status,
        "noise_status": est.noise_status,
        "detection_status": meas["detection"]["status"],
        "n_objects": None if meas["candidates"] is None else len(meas["candidates"]),
        "n_accepted": meas["n_accepted"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
