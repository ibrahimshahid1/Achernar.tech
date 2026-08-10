"""
build_focus_quality_dataset — Section 7.7.5.

Turns preserved autofocus frames into an FQE training set. It joins
`focus_samples` to `frames` (via the `frame_id` that Phase 1 now populates),
loads each 16-bit PNG, detects stars, and labels each star crop with a
SEP/DAOPHOT-style HFR (labels/psf_measure), grouped by run for leakage-free
splitting.

Output: an .npz with
    crops     (N, S, S) float32  robust-normalized to [0, 1]
    log_hfr   (N,) float32        log(HFR_px) regression target
    hfr_px    (N,) float32        raw HFR label
    run_id    (N,) int64          integer-encoded run (ranking key)
    run_key   (N,) str            original run UUID (split/provenance key)
    frame_id  (N,) str            exact preserved frame UUID
    rig_id / camera_id / filter_name / night_id
                                  deployment-strata provenance
    position  (N,) int64          focuser position
    star_snr  (N,) float32        per-star aperture SNR proxy

REQUIRED DATA (the wall): this needs a populated `achernar.db` + a `frames/`
directory produced by real autofocus runs on real rigs. Until that exists the
builder has nothing to read; validate it against a synthetic fixture (see
ml/tests/test_dataset_builder.py). A *shippable* model additionally needs
coverage across rigs, filters, seeing, SNR and focus offsets — not a single rig.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Matches `saturation_threshold` in achernar-daemon/src/vision/background.rs:
# sensor response goes non-linear below the hard clip, so the gate sits at 98%.
SATURATION_FRACTION = 0.98

# A container ceiling at or below this is an 8-bit stream, where the container
# *is* the clip point — mirroring the 8-bit arm of `resolve_saturation_adu` in
# vision/quality.rs. Above it we are looking at a 16-bit container, whose
# ceiling says nothing about where the sensor actually saturates.
EIGHT_BIT_MAX_ADU = 255.0


def saturation_level_from_quality(quality_json: str) -> float | None:
    """SEP saturation threshold in ADU for this frame, or ``None`` if unknown.

    The daemon already writes what we need. `api/focus.rs` puts both
    ``saturation_adu`` and ``saturation_gate_applied`` into ``quality_json``,
    with a comment stating they are there "so the dataset builder can filter on
    it". This function is that filter.

    The previous implementation read ``max_adu`` — the *container* ceiling — and
    fell back to 64000. On a 14-bit body delivered in a 16-bit container the
    sensor clips near 16383 while that rule put the gate at 64230, so no star
    was ever rejected. Clipped cores have artificially flat profiles and
    therefore measure *smaller* HFR than the truth, which biased the brightest,
    highest-weight samples toward "sharper than reality".

    Note the deliberate asymmetry with the daemon: at runtime an unknown
    saturation point yields an infinite threshold (reject nothing, and report
    the gate as not performed — the honest thing to do when you must keep
    observing). At training time the honest thing is the opposite: drop the
    frame, because an ungated label cannot be distinguished from a clean one
    once it is in the dataset.
    """
    q = _quality_dict(quality_json)

    # An explicit "the gate did not run" is decisive regardless of any value.
    if q.get("saturation_gate_applied") is False:
        return None

    declared = q.get("saturation_adu")
    if declared is not None:
        try:
            value = float(declared)
        except (TypeError, ValueError):
            return None
        if value > 0.0:
            return value * SATURATION_FRACTION
        return None

    # No declared value. Only an 8-bit container can supply one on its own.
    max_adu = q.get("max_adu")
    if max_adu is not None:
        try:
            ceiling = float(max_adu)
        except (TypeError, ValueError):
            return None
        if 0.0 < ceiling <= EIGHT_BIT_MAX_ADU:
            return ceiling * SATURATION_FRACTION

    return None


# Back-compat alias; the old private name is still referenced by older notebooks.
_saturation_level_from_quality = saturation_level_from_quality


def _quality_dict(raw: str) -> dict:
    try:
        value = json.loads(raw) if raw else {}
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_spatial_sources(sources, width: int, height: int, limit: int = 16):
    """Choose high-SNR valid stars across a 4x4 field grid before filling spare
    slots. This avoids training almost entirely on the bright central field while
    retaining a deterministic, inspectable selection policy."""
    ranked = sorted(
        ((index, source) for index, source in enumerate(sources) if source.valid),
        key=lambda item: (-item[1].snr, -item[1].flux, item[1].y, item[1].x),
    )
    selected, selected_indices, occupied = [], set(), set()
    for index, source in ranked:
        cell_x = min(3, max(0, int(4.0 * source.x / max(width, 1))))
        cell_y = min(3, max(0, int(4.0 * source.y / max(height, 1))))
        cell = (cell_x, cell_y)
        if cell in occupied:
            continue
        selected.append(source)
        selected_indices.add(index)
        occupied.add(cell)
        if len(selected) >= limit:
            return selected
    for index, source in ranked:
        if index not in selected_indices:
            selected.append(source)
            if len(selected) >= limit:
                break
    return selected


def build_dataset(
    db_path: str,
    output_path: str,
    crop_size: int = 64,
    min_snr: float = 5.0,
    max_stars_per_frame: int = 16,
    allow_missing_saturation: bool = False,
) -> dict:
    """Build the FQE dataset from a daemon SQLite DB + its frame PNGs.

    Frames whose saturation point is unknown are dropped by default; see
    :func:`saturation_level_from_quality`. ``allow_missing_saturation`` restores
    the old permissive behaviour for synthetic fixtures only, and stamps the
    manifest so a dataset built that way can never be mistaken for a clean one.
    """
    from PIL import Image

    from labels.psf_measure import extract_centered_crop, measure_frame_sep
    from preprocess.normalize import robust_normalize

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT fs.run_id AS run_key, fs.position AS position, fs.hfr AS classical_hfr_px,
               fs.frame_id AS frame_id,
               f.local_path AS local_path, f.checksum_sha256 AS checksum_sha256,
               f.quality_json AS quality_json, f.filter_name AS frame_filter_name,
               f.temperature AS frame_temperature, f.exposure_s AS exposure_s,
               f.gain AS gain, f.captured_at AS frame_captured_at,
               fr.rig_id AS rig_id, fr.filter_name AS run_filter_name,
               fr.started_at AS run_started_at
        FROM focus_samples fs
        JOIN frames f ON f.id = fs.frame_id
        LEFT JOIN focus_runs fr ON fr.id = fs.run_id
        WHERE fs.frame_id IS NOT NULL
        ORDER BY fs.run_id, fs.position
        """
    ).fetchall()
    con.close()

    run_index: dict[str, int] = {}
    columns: dict[str, list] = {
        name: []
        for name in (
            "crops", "log_hfr", "hfr_px", "run_id", "position", "star_snr",
            "run_key", "frame_id", "frame_checksum", "rig_id", "camera_id",
            "filter_name", "captured_at", "night_id", "label_source",
            "field_x_norm", "field_y_norm", "classical_frame_hfr_px",
            "exposure_s", "gain", "temperature",
        )
    }
    frames_read = 0
    frames_skipped_unknown_saturation = 0
    stars_kept = 0
    sources_detected = 0
    sources_valid = 0
    db_dir = Path(db_path).resolve().parent

    for row in rows:
        raw_path = row["local_path"]
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = db_dir / path
        if not path.exists():
            continue
        expected_checksum = row["checksum_sha256"] or ""
        actual_checksum = _sha256(path)
        if expected_checksum and actual_checksum != expected_checksum:
            raise RuntimeError(
                f"Frame checksum mismatch for {row['frame_id']}: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )

        # Decide this before opening the image: a frame we cannot gate is a
        # frame we cannot label, so there is no point measuring it.
        sat = saturation_level_from_quality(row["quality_json"])
        if sat is None:
            if not allow_missing_saturation:
                frames_skipped_unknown_saturation += 1
                continue
            # Permissive mode: no gate at all, matching the daemon's runtime
            # behaviour of rejecting nothing rather than inventing a threshold.
            sat = float("inf")

        with Image.open(path) as image:
            img = np.asarray(image).astype(np.float64)
        if img.ndim != 2:
            continue
        frames_read += 1
        sources, _frame_label = measure_frame_sep(
            img,
            saturation_level=sat,
            min_snr=min_snr,
            fit_fwhm=False,
        )
        sources_detected += len(sources)
        sources_valid += sum(source.valid for source in sources)

        quality = _quality_dict(row["quality_json"])
        run_key = str(row["run_key"])
        rid = run_index.setdefault(run_key, len(run_index))
        rig_id_value = str(row["rig_id"] or quality.get("rig_id") or "unknown-rig")
        camera_id = str(quality.get("camera_id") or "unknown-camera")
        filter_name = str(
            row["run_filter_name"]
            or row["frame_filter_name"]
            or quality.get("filter_name")
            or "unknown-filter"
        )
        captured_at = str(row["frame_captured_at"] or row["run_started_at"] or "")
        night_id = captured_at[:10] if len(captured_at) >= 10 else "unknown-night"

        for source in _select_spatial_sources(
            sources, img.shape[1], img.shape[0], max_stars_per_frame
        ):
            crop = extract_centered_crop(img, source.x, source.y, crop_size)
            if crop is None or source.hfr_px is None or not np.isfinite(source.hfr_px):
                continue
            norm, _ = robust_normalize(crop)
            columns["crops"].append(norm.astype(np.float32))
            columns["log_hfr"].append(np.log(source.hfr_px))
            columns["hfr_px"].append(source.hfr_px)
            columns["run_id"].append(rid)
            columns["position"].append(int(row["position"]))
            columns["star_snr"].append(source.snr)
            columns["run_key"].append(run_key)
            columns["frame_id"].append(str(row["frame_id"]))
            columns["frame_checksum"].append(actual_checksum)
            columns["rig_id"].append(rig_id_value)
            columns["camera_id"].append(camera_id)
            columns["filter_name"].append(filter_name)
            columns["captured_at"].append(captured_at)
            columns["night_id"].append(night_id)
            columns["label_source"].append("sep_flux_radius")
            columns["field_x_norm"].append(source.x / max(img.shape[1] - 1, 1))
            columns["field_y_norm"].append(source.y / max(img.shape[0] - 1, 1))
            columns["classical_frame_hfr_px"].append(float(row["classical_hfr_px"]))
            columns["exposure_s"].append(
                float(row["exposure_s"]) if row["exposure_s"] is not None else np.nan
            )
            columns["gain"].append(float(row["gain"]) if row["gain"] is not None else np.nan)
            columns["temperature"].append(
                float(row["frame_temperature"])
                if row["frame_temperature"] is not None
                else np.nan
            )
            stars_kept += 1

    if not columns["crops"]:
        hint = (
            f"Has any real autofocus run been recorded yet?"
            if not frames_skipped_unknown_saturation
            else (
                f"{frames_skipped_unknown_saturation} frame(s) were dropped because "
                "their sensor saturation point is unknown. Set `[camera] "
                "saturation_adu` in the daemon config (measure it with "
                "tools/photon_transfer.py) and re-capture, or pass "
                "--allow-missing-saturation for synthetic fixtures only."
            )
        )
        raise RuntimeError(f"No usable star crops found in {db_path}. {hint}")

    dataset = {
        "crops": np.stack(columns["crops"]).astype(np.float32),
        "log_hfr": np.asarray(columns["log_hfr"], dtype=np.float32),
        "hfr_px": np.asarray(columns["hfr_px"], dtype=np.float32),
        "run_id": np.asarray(columns["run_id"], dtype=np.int64),
        "position": np.asarray(columns["position"], dtype=np.int64),
        "star_snr": np.asarray(columns["star_snr"], dtype=np.float32),
        "field_x_norm": np.asarray(columns["field_x_norm"], dtype=np.float32),
        "field_y_norm": np.asarray(columns["field_y_norm"], dtype=np.float32),
        "classical_frame_hfr_px": np.asarray(
            columns["classical_frame_hfr_px"], dtype=np.float32
        ),
        "exposure_s": np.asarray(columns["exposure_s"], dtype=np.float32),
        "gain": np.asarray(columns["gain"], dtype=np.float32),
        "temperature": np.asarray(columns["temperature"], dtype=np.float32),
    }
    for name in (
        "run_key", "frame_id", "frame_checksum", "rig_id", "camera_id",
        "filter_name", "captured_at", "night_id", "label_source",
    ):
        dataset[name] = np.asarray(columns[name], dtype=np.str_)
    # The original run UUID is the minimum leakage boundary. Keep it explicit so
    # registry files do not depend on a rebuild-specific integer encoding.
    dataset["split_group"] = dataset["run_key"].copy()
    np.savez_compressed(output_path, **dataset)
    output = Path(output_path)
    manifest = {
        "dataset": str(output),
        "sha256": _sha256(output),
        "label_method": "SEP mesh background + deblending + aperture-normalized flux_radius",
        "frames_read": frames_read,
        "frames_skipped_unknown_saturation": frames_skipped_unknown_saturation,
        "sources_detected": sources_detected,
        "sources_valid": sources_valid,
        "stars_kept": stars_kept,
        "runs": len(run_index),
        "crop_size": crop_size,
        "max_stars_per_frame": max_stars_per_frame,
        # Load-bearing provenance: a permissive build has ungated labels in it
        # and must never be promoted or compared against a gated one.
        "saturation_gate": "permissive" if allow_missing_saturation else "enforced",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"✓ {output_path}: {stars_kept} stars from {frames_read} frames "
        f"across {len(run_index)} runs"
    )
    if frames_skipped_unknown_saturation:
        print(
            f"  ! {frames_skipped_unknown_saturation} frame(s) dropped: sensor "
            "saturation point unknown (set [camera] saturation_adu)"
        )
    if allow_missing_saturation:
        print("  ! PERMISSIVE saturation gate — synthetic fixtures only, never promote")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Build FQE training dataset")
    parser.add_argument("--db", required=True, help="Path to the daemon achernar.db")
    parser.add_argument("--output", default="focus_quality_ds_v1.npz")
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--max-stars-per-frame", type=int, default=16)
    parser.add_argument(
        "--allow-missing-saturation",
        action="store_true",
        help=(
            "Keep frames whose sensor saturation point is unknown. Synthetic "
            "fixtures only — the resulting labels are ungated and the manifest "
            "is stamped 'permissive'."
        ),
    )
    args = parser.parse_args()
    build_dataset(
        args.db,
        args.output,
        args.crop_size,
        max_stars_per_frame=args.max_stars_per_frame,
        allow_missing_saturation=args.allow_missing_saturation,
    )


if __name__ == "__main__":
    main()
