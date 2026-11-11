"""Manifest ``bge-real-frames-v2``: what was observed, and the exact analysis input.

Top level::

    {"schema": "bge-real-frames-v2", "created_utc": ISO, "builder": {...},
     "analysis_input": {"representation": str, "loader": str, "note": str},
     "split_rule": {"path": str, "sha256": hex}, "nights": [...],
     "frames": [frame, ...], "excluded": [{"source": str, "reason": str, ...}]}

Frame (``path``/``sha256`` are the analysis intensity the methods read)::

    id, path, sha256, format ("daemon_bundle" | "fits" | "f32" | "import_luma"),
    hdu (fits), width/height (f32), masks {name: {"path", "format": "fits"|"u8", "sha256"}},
    analysis_input {"representation", "descriptor_path", "descriptor_sha256",
                    "processing_version", "config_sha256", "units"},
    frame_type (light|dark|bias|flat), night_id, split, timestamp_utc,
    rig_id, camera, optical_train, iso, gain, exposure_s, sensor_temp_c,
    ambient_temp_c, focus_position, focuser_stepsize_um, focal_ratio,
    focus_run_id, sweep_direction, repeat_group, pointing {...}, notes,
    saturation_adu, gain_e_per_adu, source {path, sha256, role}, provenance {...},
    flags [...]

Paths are relative to the manifest's directory unless absolute. The loader
verifies every sha256 before returning pixels. ``import_luma`` frames are read
with ``tools/import_session.load_luma_with_container`` (the repository's own
loader); daemon bundles and FITS with Astropy. RAW is never decoded here
(``tools/import_session.py`` gives the reason): a RAW-only capture needs the
daemon's analysis bundle.

A ``bge-real-frames-v1`` manifest is still accepted; :func:`load_manifest`
upgrades it in memory (frame_type light, night_id from ``group`` absent ->
``"unknown-night"``) and marks it ``upgraded_from_v1``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import (DAEMON_ANALYSIS_REPRESENTATION, FRAME_TYPES, MANIFEST_SCHEMA_V1, MANIFEST_SCHEMA_V2, MASK_NAMES,
               SPLITS, REPO_ROOT, sha256_file)

FORMATS = ("daemon_bundle", "fits", "f32", "import_luma")
MASK_FORMATS = ("fits", "u8")
OPTIONAL_FIELDS = (
    "timestamp_utc", "rig_id", "camera", "optical_train", "iso", "gain", "exposure_s", "sensor_temp_c",
    "ambient_temp_c", "focus_position", "focuser_stepsize_um", "focal_ratio", "focus_run_id", "sweep_direction",
    "repeat_group", "pointing", "notes", "saturation_adu", "gain_e_per_adu", "source", "provenance", "flags",
    "hdu", "width", "height", "masks", "analysis_input", "truth",
)


class ManifestError(ValueError):
    pass


class ChecksumMismatch(ManifestError):
    pass


def validate_frame(f: dict) -> list:
    errors = []
    fid = f.get("id")
    for key in ("id", "path", "sha256", "format", "frame_type", "night_id", "split"):
        if f.get(key) in (None, ""):
            errors.append(f"frame {fid!r}: missing {key}")
    if f.get("format") not in FORMATS:
        errors.append(f"frame {fid!r}: format {f.get('format')!r} not in {FORMATS}")
    if f.get("frame_type") not in FRAME_TYPES:
        errors.append(f"frame {fid!r}: frame_type {f.get('frame_type')!r} not in {FRAME_TYPES}")
    if f.get("split") not in SPLITS:
        errors.append(f"frame {fid!r}: split {f.get('split')!r} not in {SPLITS}")
    if f.get("format") == "f32" and not (f.get("width") and f.get("height")):
        errors.append(f"frame {fid!r}: f32 needs width and height")
    rep = (f.get("analysis_input") or {}).get("representation")
    if not rep:
        errors.append(f"frame {fid!r}: analysis_input.representation is required")
    for name, m in (f.get("masks") or {}).items():
        if name not in MASK_NAMES:
            errors.append(f"frame {fid!r}: unknown mask {name!r}")
        if not isinstance(m, dict) or m.get("format") not in MASK_FORMATS or not m.get("path"):
            errors.append(f"frame {fid!r}: mask {name!r} needs path and format in {MASK_FORMATS}")
    unknown = set(f) - set(OPTIONAL_FIELDS) - {"id", "path", "sha256", "format", "frame_type", "night_id", "split"}
    if unknown:
        errors.append(f"frame {fid!r}: unknown fields {sorted(unknown)}")
    return errors


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA_V2:
        raise ManifestError(f"expected schema {MANIFEST_SCHEMA_V2!r}")
    errors = []
    ids = set()
    for f in manifest.get("frames", []):
        errors += validate_frame(f)
        if f.get("id") in ids:
            errors.append(f"duplicate frame id {f.get('id')!r}")
        ids.add(f.get("id"))
    nights = {}
    for f in manifest.get("frames", []):
        nights.setdefault(f.get("night_id"), set()).add(f.get("split"))
    for night, splits in nights.items():
        if len(splits) > 1:
            errors.append(f"night {night!r} is split across {sorted(map(str, splits))}: nights are never split")
    if errors:
        raise ManifestError("; ".join(errors[:20]) + (f" (+{len(errors) - 20} more)" if len(errors) > 20 else ""))


def upgrade_v1(manifest: dict) -> dict:
    """A v1 manifest as v2 frames, in memory. v1 has no night: every frame
    goes to ``unknown-night``, which makes night-level claims impossible
    (stated in the result)."""
    frames = []
    for f in manifest.get("frames", []):
        entry = {
            "id": f["id"], "path": f["path"], "sha256": f.get("sha256") or "unverified-v1",
            "format": f["format"], "frame_type": "light", "night_id": "unknown-night", "split": f["split"],
            "analysis_input": {"representation": "undeclared-v1"},
            "flags": ["upgraded_from_v1"],
        }
        for key in ("hdu", "width", "height", "focus_position", "camera", "exposure_s"):
            if key in f:
                entry[key] = f[key]
        if f.get("group"):
            entry["repeat_group"] = f["group"]
        if f.get("masks"):
            entry["masks"] = {k: {"path": v, "format": "u8"} for k, v in f["masks"].items()}
        frames.append(entry)
    return {"schema": MANIFEST_SCHEMA_V2, "upgraded_from": MANIFEST_SCHEMA_V1, "frames": frames, "excluded": [],
            "analysis_input": {"representation": "undeclared-v1"}}


def load_manifest(path) -> dict:
    """Read and validate a v2 manifest (or upgrade a v1 one)."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") == MANIFEST_SCHEMA_V1:
        manifest = upgrade_v1(manifest)
    validate_manifest(manifest)
    return manifest


def resolve(base: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(base) / p


def _verify(path: Path, expected: str | None, what: str) -> None:
    if not expected or expected == "unverified-v1":
        return
    actual = sha256_file(path)
    if actual != expected:
        raise ChecksumMismatch(f"{what}: sha256 {actual[:12]}... != manifest {expected[:12]}... ({path})")


def _read_fits_2d(path: Path, hdu=None) -> np.ndarray:
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        if hdu is not None:
            data = hdul[int(hdu)].data
        else:
            data = next((h.data for h in hdul if getattr(h, "data", None) is not None and h.data.ndim == 2), None)
        if data is None or data.ndim != 2:
            raise ManifestError(f"{path}: no 2-D image HDU")
        return np.array(data)


def import_session_module():
    """``tools/import_session.py`` as a module (night rule, luma loader)."""
    import importlib.util
    import sys

    tool = REPO_ROOT / "tools" / "import_session.py"
    name = "_achernar_import_session"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, tool)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def load_frame(entry: dict, base, verify: bool = True):
    """``(image float32 C-contiguous, masks {name: uint8 0/1})`` for one frame."""
    base = Path(base)
    path = resolve(base, entry["path"])
    if verify:
        _verify(path, entry.get("sha256"), f"frame {entry['id']}")
    fmt = entry["format"]
    masks = {}
    if fmt in ("fits", "daemon_bundle"):
        image = _read_fits_2d(path, entry.get("hdu"))
    elif fmt == "f32":
        image = np.fromfile(path, dtype="<f4").reshape(int(entry["height"]), int(entry["width"]))
    elif fmt == "import_luma":
        loaded = import_session_module().load_luma_with_container(path)
        if loaded is None:
            raise ManifestError(f"frame {entry['id']}: import_session loader could not read {path}")
        image = loaded.data
        if loaded.channel_clip_mask is not None:
            masks["native_saturation"] = np.asarray(loaded.channel_clip_mask, dtype=np.uint8)
    else:
        raise ManifestError(f"unknown format {fmt!r}")
    image = np.ascontiguousarray(image, dtype=np.float32)
    for name, m in (entry.get("masks") or {}).items():
        mpath = resolve(base, m["path"])
        if verify:
            _verify(mpath, m.get("sha256"), f"frame {entry['id']} mask {name}")
        if m["format"] == "fits":
            arr = _read_fits_2d(mpath)
        else:
            arr = np.fromfile(mpath, dtype=np.uint8).reshape(image.shape)
        if arr.shape != image.shape:
            raise ManifestError(f"frame {entry['id']} mask {name}: shape {arr.shape} != image {image.shape}")
        masks[name] = np.ascontiguousarray(arr != 0, dtype=np.uint8)
    return image, masks


def load_truth(entry: dict, base):
    """Synthetic-validation truth maps (tests only; real frames have none)."""
    truth = entry.get("truth")
    if not truth:
        return None
    base = Path(base)
    out = {}
    for key in ("background", "noise"):
        if truth.get(key):
            out[key] = np.ascontiguousarray(_read_fits_2d(resolve(base, truth[key])), dtype=np.float32)
    return out


def frames_by_night(manifest: dict, split: str | None = None) -> dict:
    out = {}
    for f in manifest["frames"]:
        if split is None or f["split"] == split:
            out.setdefault(f["night_id"], []).append(f)
    return out


def representation(entry: dict) -> str:
    return (entry.get("analysis_input") or {}).get("representation") or "undeclared"


def is_daemon_representation(entry: dict) -> bool:
    return representation(entry) == DAEMON_ANALYSIS_REPRESENTATION
