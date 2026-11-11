"""Real camera frames for a future run (BGE-22 third comparison).

Real frames have no pixel-level truth. This adapter can load a manifest of
frames and run the Python methods to record *observations* (statuses, levels,
noise, counts, HFR, repeatability across repeated frames), but it refuses to
emit accuracy or pass/fail without a thresholds file that was declared
before the run, and it never opens a frame marked ``"split": "holdout"``
unless hold-out use is requested explicitly *and* such a thresholds file is
supplied. The benchmark command does not call it; no hold-out data is read.

Manifest (``bge-real-frames-v1``)::

    {"schema": "bge-real-frames-v1",
     "frames": [{"id": str, "path": str (relative to the manifest), "format": "fits" | "f32",
                 "width": int, "height": int (f32 only), "hdu": int (fits, default 0),
                 "masks": {"bad_pixel"|"invalid"|"edge"|"native_saturation": path to uint8 .u8},
                 "split": "development" | "holdout", "group": str (repeat/sweep group, optional),
                 "focus_position": number (optional), "camera": {...}, "exposure_s": number}]}

Thresholds (``bge-real-thresholds-v1``)::

    {"schema": "bge-real-thresholds-v1", "predeclared_utc": ISO-8601, "manifest_sha256": hex,
     "metrics": {"<metric>": {"max": v} | {"min": v}}}

Supported metrics: ``status_failure_rate`` (fraction of frames whose
background is unavailable), ``hfr_repeatability_cv_max`` (largest coefficient
of variation of frame HFR within a group).

``bge-real-frames-v2`` manifests (nights, frame types, daemon analysis
bundles; ``ml/eval/bge/qualification/manifest.py``) are also readable here:
their light frames are evaluated the same way (``repeat_group`` as the group).
The full real-frame qualification (pull, difference pull, negated residual,
injection-recovery, dark trials, focus sweeps, paired legacy comparisons,
thresholds R1-R17b and the one-shot hold-out ledger) is
``python -m ml.eval.bge.qualification.run_qualification``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

MANIFEST_SCHEMA = "bge-real-frames-v1"
MANIFEST_SCHEMA_V2 = "bge-real-frames-v2"
THRESHOLDS_SCHEMA = "bge-real-thresholds-v1"
SUPPORTED_METRICS = ("status_failure_rate", "hfr_repeatability_cv_max")


class RealFrameRefusal(RuntimeError):
    """Raised instead of producing a claim the inputs cannot support."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") == MANIFEST_SCHEMA_V2:
        from .qualification.manifest import validate_manifest

        validate_manifest(manifest)
        return manifest
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"expected schema {MANIFEST_SCHEMA!r} or {MANIFEST_SCHEMA_V2!r}")
    for f in manifest.get("frames", []):
        if f.get("split") not in ("development", "holdout"):
            raise ValueError(f"frame {f.get('id')!r}: split must be 'development' or 'holdout'")
    return manifest


def load_thresholds(path, manifest_path, now=None) -> dict:
    th = json.loads(Path(path).read_text())
    if th.get("schema") != THRESHOLDS_SCHEMA:
        raise RealFrameRefusal(f"thresholds file does not declare {THRESHOLDS_SCHEMA!r}")
    try:
        declared = dt.datetime.fromisoformat(th["predeclared_utc"])
    except (KeyError, ValueError) as exc:
        raise RealFrameRefusal(f"thresholds file has no valid predeclared_utc: {exc}") from exc
    if declared.tzinfo is None:
        raise RealFrameRefusal("predeclared_utc must carry a timezone")
    now = now or dt.datetime.now(dt.timezone.utc)
    if declared >= now:
        raise RealFrameRefusal("thresholds are not predeclared (timestamp is not in the past)")
    if th.get("manifest_sha256") != _sha256(manifest_path):
        raise RealFrameRefusal("thresholds were declared for a different manifest (sha256 mismatch)")
    unknown = set(th.get("metrics", {})) - set(SUPPORTED_METRICS)
    if unknown:
        raise RealFrameRefusal(f"unsupported threshold metrics: {sorted(unknown)}")
    return th


def select_frames(manifest: dict, include_holdout: bool = False) -> list:
    return [f for f in manifest["frames"] if (f["split"] == "development" or include_holdout)
            and f.get("frame_type", "light") == "light"]


def load_frame(entry: dict, base: Path):
    if "frame_type" in entry:  # bge-real-frames-v2 entry
        from .qualification.manifest import load_frame as load_v2

        return load_v2(entry, base)
    path = base / entry["path"]
    if entry["format"] == "fits":
        from astropy.io import fits

        with fits.open(path, memmap=False) as hdul:
            image = np.asarray(hdul[entry.get("hdu", 0)].data, dtype=np.float32)
    elif entry["format"] == "f32":
        image = np.fromfile(path, dtype="<f4").reshape(entry["height"], entry["width"])
    else:
        raise ValueError(f"unknown format {entry['format']!r}")
    masks = {k: np.fromfile(base / v, dtype=np.uint8).reshape(image.shape) for k, v in entry.get("masks", {}).items()}
    return np.ascontiguousarray(image), masks


def evaluate(manifest_path, thresholds_path=None, include_holdout: bool = False) -> dict:
    """Observations for development frames; pass/fail only with predeclared thresholds."""
    from . import python_methods as PM

    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    thresholds = None
    if thresholds_path is not None:
        thresholds = load_thresholds(thresholds_path, manifest_path)
    if include_holdout and thresholds is None:
        raise RealFrameRefusal("hold-out frames require a predeclared thresholds file")
    frames = select_frames(manifest, include_holdout)
    rows = []
    for entry in frames:
        image, masks = load_frame(entry, manifest_path.parent)
        est = PM.python_sep_background(image, masks)
        meas = PM.detect_and_measure(image, masks, est)
        rows.append({"id": entry["id"], "split": entry["split"], "group": entry.get("group", entry.get("repeat_group")),
                     "focus_position": entry.get("focus_position"), "background": est.summary(),
                     "detection_status": meas["detection"]["status"],
                     "n_candidates": None if meas["candidates"] is None else len(meas["candidates"]),
                     "n_accepted": meas["n_accepted"], "frame_hfr_median_px": meas["frame_hfr_median_px"]})
    observed = {"status_failure_rate": (sum(r["background"]["level_status"] != "measured" for r in rows) / len(rows))
                if rows else None}
    groups = {}
    for r in rows:
        if r["group"] and r["frame_hfr_median_px"] is not None:
            groups.setdefault(r["group"], []).append(r["frame_hfr_median_px"])
    cvs = [float(np.std(v) / np.mean(v)) for v in groups.values() if len(v) >= 2 and np.mean(v) > 0]
    observed["hfr_repeatability_cv_max"] = max(cvs) if cvs else None
    out = {"manifest": str(manifest_path), "n_frames_evaluated": len(rows),
           "n_holdout_frames_skipped": sum(f["split"] == "holdout" for f in manifest["frames"]) if not include_holdout else 0,
           "observations": observed, "frames": rows,
           "interpretation": "operational reliability/generalisation on real frames; no pixel-level truth"}
    if thresholds is None:
        out.update(status="observations_only", pass_fail=None, accuracy=None,
                   reason="no predeclared thresholds file: accuracy and pass/fail are not emitted")
        return out
    verdicts = {}
    for name, rule in thresholds["metrics"].items():
        v = observed.get(name)
        if v is None:
            verdicts[name] = {"status": "not_computable", "value": None, "rule": rule}
        elif "max" in rule:
            verdicts[name] = {"status": "pass" if v <= rule["max"] else "fail", "value": v, "rule": rule}
        else:
            verdicts[name] = {"status": "pass" if v >= rule["min"] else "fail", "value": v, "rule": rule}
    out.update(status="evaluated_against_predeclared_thresholds", pass_fail=verdicts, accuracy=None,
               thresholds_predeclared_utc=thresholds["predeclared_utc"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--thresholds", type=Path, default=None)
    ap.add_argument("--include-holdout", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    result = evaluate(args.manifest, args.thresholds, args.include_holdout)
    args.out.write_text(json.dumps(result, indent=1, default=float))
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
