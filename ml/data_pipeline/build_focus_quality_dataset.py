"""
build_focus_quality_dataset — Section 7.7.5.

Turns preserved autofocus frames into an FQE training set. It joins
`focus_samples` to `frames` (via the `frame_id` that Phase 1 now populates),
loads each 16-bit PNG, detects stars, and labels each star crop with a
SEP/DAOPHOT-style HFR (labels/psf_measure), grouped by run for leakage-free
splitting.

Output: an .npz (plus ``<name>.manifest.json``) with
    crops     (N, S, S) float32  robust-normalized to [0, 1]
    log_hfr   (N,) float32        log(HFR_px) regression target
    hfr_px    (N,) float32        raw HFR label
    run_id    (N,) int64          integer-encoded run (ranking key)
    run_key   (N,) str            original run UUID (split/provenance key)
    frame_id  (N,) str            exact preserved frame UUID
    rig_id / camera_id / filter_name / night_id
                                  deployment-strata provenance
    position  (N,) int64          focuser position
    star_snr  (N,) float32        per-star aperture SNR (background-noise-only proxy)
    label_version / background_config_sha256 / detection_config_sha256 (N,) str
                                  per-row label provenance (see below)
    saturation_source (N,) str    "declared_sensor_saturation" | "container_ceiling_8bit" | "none"
    local_background_adu / local_rms_adu (N,) float32
                                  SEP B(x,y) and R(x,y) at the star

LABEL VERSIONS. Labels from this builder carry ``LABEL_VERSION`` per row and
in the manifest, together with the SEP version and the background/detection
configurations and their sha256. Frozen datasets built before that (e.g.
``focus_quality_synth_v1.npz``, ``focus_quality_ds_v1.npz``) have none of it
and are ``legacy_unverified``: they are never overwritten (a different default
output path, and an explicit refusal to replace an existing dataset or
manifest), and :func:`load_qualified_dataset` — the one loader a *qualified*
evaluation may use — excludes their rows.

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
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Matches `saturation_threshold` in achernar-daemon/src/vision/background.rs:
# sensor response goes non-linear below the hard clip, so the gate sits at 98%.
SATURATION_FRACTION = 0.98

# An 8-bit container (max_adu <= 255) is treated as its own clip point
# (decision D-011): 255 is the real ceiling of the *recorded* data, so a star
# reaching it is clipped in the data whatever the sensor did, and the rule errs
# toward rejecting stars. Its provenance is recorded as "container_ceiling_8bit"
# — a property of the container, not a native sensor saturation level. No
# container rule exists for 16-bit data: a 14-bit body in a 16-bit container
# clips near 16383 and nothing in the container says so. (The Rust
# `resolve_saturation_adu` this once mirrored was removed in 98f6a2b6; the
# daemon has no container arm.)
EIGHT_BIT_MAX_ADU = 255.0

SATURATION_SOURCE_DECLARED = "declared_sensor_saturation"
SATURATION_SOURCE_CONTAINER_8BIT = "container_ceiling_8bit"
# A saturation_adu written with a saturation_source this builder does not
# recognise: the level is still applied as a gate, but it is never called a
# declared sensor value.
SATURATION_SOURCE_UNRECOGNIZED = "unrecognized_saturation_source"

# Label provenance. Bump LABEL_VERSION whenever psf_measure.LABEL_SCHEMA_VERSION,
# the SEP pin, or the selection policy of this builder changes meaning.
LABEL_VERSION = "fqe_sep_spatial_v2"
LEGACY_UNVERIFIED = "legacy_unverified"
QUALIFIED = "qualified"
PERMISSIVE_SATURATION = "unqualified_permissive_saturation"
DEFAULT_OUTPUT = "focus_quality_ds_v2/focus_quality_ds_v2.npz"


class LegacyUnverifiedLabels(RuntimeError):
    """A qualified evaluation was asked to use labels without v2 provenance."""


def saturation_decision_from_quality(quality_json: str) -> tuple[float | None, str | None]:
    """``(threshold_adu, source)`` for a frame; ``(None, None)`` when unknown.

    ``source`` is :data:`SATURATION_SOURCE_DECLARED` for a declared sensor
    value or :data:`SATURATION_SOURCE_CONTAINER_8BIT` for the 8-bit container
    rule. See :func:`saturation_level_from_quality` for the policy.

    A ``saturation_source`` written next to ``saturation_adu`` (by
    tools/import_session.py) is honoured: ``container_ceiling_8bit`` stays a
    container ceiling, and any other unrecognised value becomes
    :data:`SATURATION_SOURCE_UNRECOGNIZED`. Without the key (the daemon, which
    writes its configured sensor value, and pre-v2 records) ``saturation_adu``
    is a declared sensor value.
    """
    q = _quality_dict(quality_json)

    # An explicit "the gate did not run" is decisive regardless of any value.
    if q.get("saturation_gate_applied") is False:
        return None, None

    declared = q.get("saturation_adu")
    if declared is not None:
        try:
            value = float(declared)
        except (TypeError, ValueError):
            return None, None
        if not (value > 0.0 and np.isfinite(value)):
            return None, None
        written = q.get("saturation_source")
        if written is None or written == SATURATION_SOURCE_DECLARED:
            source = SATURATION_SOURCE_DECLARED
        elif written == SATURATION_SOURCE_CONTAINER_8BIT:
            source = SATURATION_SOURCE_CONTAINER_8BIT
        else:
            source = SATURATION_SOURCE_UNRECOGNIZED
        return value * SATURATION_FRACTION, source

    # No declared value. Only an 8-bit container can supply one on its own.
    max_adu = q.get("max_adu")
    if max_adu is not None:
        try:
            ceiling = float(max_adu)
        except (TypeError, ValueError):
            return None, None
        if 0.0 < ceiling <= EIGHT_BIT_MAX_ADU:
            return ceiling * SATURATION_FRACTION, SATURATION_SOURCE_CONTAINER_8BIT

    return None, None


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

    The 8-bit container rule (``max_adu <= 255``) is a container ceiling, not
    a sensor saturation level; :func:`saturation_decision_from_quality`
    returns that provenance alongside the threshold.
    """
    return saturation_decision_from_quality(quality_json)[0]


# Back-compat alias; the old private name is still referenced by older notebooks.
_saturation_level_from_quality = saturation_level_from_quality


def resolve_night_id(frame_night_id, run_night_id, captured_at: str) -> str:
    """The observing night a row belongs to.

    Night is a **split key** — COC splits on it and the FQE reports it as a
    deployment stratum — so getting it wrong silently invalidates a
    generalisation number.

    Prefer the stored column. `frames.night_id` and `focus_runs.night_id` are
    written by the daemon (migration 007) and by `tools/import_session.py`, both
    of which apply the local-sunset-date rule in `storage/night.rs`; and for a
    hand-imported sweep the observer's field card is a better authority on which
    night it was than a file timestamp.

    Only when neither column is populated fall back to the timestamp — and then
    apply the same noon rule rather than slicing the date. A plain
    ``captured_at[:10]`` puts a 23:50 sweep and the 00:10 sweep that follows it
    on different nights, which is exactly the split-corrupting behaviour that
    rule exists to prevent.
    """
    for stored in (frame_night_id, run_night_id):
        if stored:
            text = str(stored).strip()
            if text:
                return text

    if len(captured_at) < 10:
        return "unknown-night"
    try:
        moment = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except ValueError:
        return captured_at[:10]
    if moment.tzinfo is not None:
        moment = moment.astimezone()
    if moment.hour < 12:
        moment = moment - timedelta(days=1)
    return moment.strftime("%Y-%m-%d")



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


def _manifest_path(output: Path) -> Path:
    return output.with_suffix(".manifest.json")


def _read_manifest(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def label_provenance_status(manifest: dict | None) -> str:
    """Dataset-level label status: :data:`QUALIFIED`, :data:`LEGACY_UNVERIFIED`
    or :data:`PERMISSIVE_SATURATION`.

    Qualified means: the current :data:`LABEL_VERSION`, the pinned SEP version,
    both SEP configuration objects present and matching their recorded sha256,
    and an enforced saturation gate. Anything missing is legacy_unverified —
    old fields are not reinterpreted.
    """
    from labels.psf_measure import SEP_PINNED_VERSION, config_sha256

    if not manifest or manifest.get("label_version") != LABEL_VERSION:
        return LEGACY_UNVERIFIED
    if manifest.get("sep_version") != SEP_PINNED_VERSION:
        return LEGACY_UNVERIFIED
    for kind in ("background", "detection"):
        config = manifest.get(f"{kind}_config")
        digest = manifest.get(f"{kind}_config_sha256")
        if not isinstance(config, dict) or not digest or config_sha256(config) != digest:
            return LEGACY_UNVERIFIED
    if manifest.get("saturation_gate") != "enforced":
        return PERMISSIVE_SATURATION
    return QUALIFIED


def qualified_row_mask(dataset: dict, manifest: dict | None) -> np.ndarray:
    """Per-row mask of labels a qualified evaluation may use.

    A row qualifies only if the dataset qualifies (:func:`label_provenance_status`)
    and the row itself carries the current label version and the manifest's
    configuration hashes. Rows without these columns are legacy_unverified.
    """
    n = len(dataset["hfr_px"])
    if label_provenance_status(manifest) != QUALIFIED:
        return np.zeros(n, dtype=bool)
    try:
        version = np.asarray(dataset["label_version"]).astype(str)
        background = np.asarray(dataset["background_config_sha256"]).astype(str)
        detection = np.asarray(dataset["detection_config_sha256"]).astype(str)
    except KeyError:
        return np.zeros(n, dtype=bool)
    return (
        (version == LABEL_VERSION)
        & (background == manifest["background_config_sha256"])
        & (detection == manifest["detection_config_sha256"])
    )


def load_qualified_dataset(dataset_path: str) -> tuple[dict, dict]:
    """The loader a *qualified* FQE evaluation must use (BGE-20).

    Returns ``(dataset, report)`` with only qualified rows. Rows lacking v2
    provenance — every row of a dataset without a manifest, or built before
    ``LABEL_VERSION`` — are excluded and counted as legacy_unverified; a
    dataset with no qualified row raises :class:`LegacyUnverifiedLabels`.
    Exploratory scripts that read an .npz directly (e.g. ``eval/baselines.py``,
    ``eval/eval_quality.py`` today) are not qualified evaluations.
    """
    path = Path(dataset_path)
    with np.load(path, allow_pickle=False) as npz:
        data = {key: npz[key] for key in npz.files}
    manifest = _read_manifest(_manifest_path(path))
    status = label_provenance_status(manifest)
    mask = qualified_row_mask(data, manifest)
    n = len(mask)
    report = {
        "dataset": str(path),
        "dataset_status": status,
        "rows_total": int(n),
        "rows_qualified": int(mask.sum()),
        "rows_excluded": int(n - mask.sum()),
        "rows_legacy_unverified": int(n - mask.sum()) if status != PERMISSIVE_SATURATION else 0,
    }
    if not mask.any():
        raise LegacyUnverifiedLabels(
            f"{path}: no qualified labels ({status}; {n} row(s) excluded). Rebuild with "
            f"build_focus_quality_dataset ({LABEL_VERSION}) for a qualified evaluation."
        )
    filtered = {key: (value[mask] if getattr(value, "shape", ()) and value.shape[0] == n else value)
                for key, value in data.items()}
    return filtered, report


def _refuse_overwrite(output: Path, overwrite: bool) -> None:
    """Never replace a frozen dataset or manifest.

    Without ``overwrite`` any existing output or manifest is an error. With it,
    only a dataset whose manifest already declares the current LABEL_VERSION
    may be replaced; a legacy (or manifest-less) artifact never is.
    """
    manifest_path = _manifest_path(output)
    existing = [p for p in (output, manifest_path) if p.exists()]
    if not existing:
        return
    if overwrite:
        manifest = _read_manifest(manifest_path)
        if manifest and manifest.get("label_version") == LABEL_VERSION:
            return
        raise FileExistsError(
            f"refusing to overwrite {', '.join(map(str, existing))}: not a {LABEL_VERSION} dataset "
            "(frozen/legacy artifacts are never replaced; choose a new --output)"
        )
    raise FileExistsError(
        f"refusing to overwrite existing {', '.join(map(str, existing))}; choose a new --output "
        "(or pass overwrite=True for a dataset built by this label version)"
    )


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
    overwrite: bool = False,
) -> dict:
    """Build the FQE dataset from a daemon SQLite DB + its frame PNGs.

    Frames whose saturation point is unknown are dropped by default; see
    :func:`saturation_level_from_quality`. ``allow_missing_saturation`` restores
    the old permissive behaviour for synthetic fixtures only, and stamps the
    manifest so a dataset built that way can never be mistaken for a clean one.

    Labels come from ``psf_measure.measure_frame_sep`` with its default
    (explicit, hashed) SEP configuration. Frames whose SEP background is
    unavailable or whose noise is unresolved yield no label and are counted
    by reason. An existing output or manifest is never overwritten (see
    :func:`_refuse_overwrite`).
    """
    from PIL import Image

    from labels.psf_measure import (
        LABEL_SCHEMA_VERSION,
        SepBackgroundParams,
        SepDetectionParams,
        extract_centered_crop,
        measure_frame_sep,
        require_sep,
    )
    from preprocess.normalize import robust_normalize

    output = Path(output_path)
    if output.suffix != ".npz":
        output = output.with_name(output.name + ".npz")
    _refuse_overwrite(output, overwrite)
    sep_version = require_sep().__version__
    background_params = SepBackgroundParams()
    detection_params = SepDetectionParams()
    background_config = background_params.config(sep_version)
    detection_config = detection_params.config(sep_version)
    background_sha = background_params.config_sha256(sep_version)
    detection_sha = detection_params.config_sha256(sep_version)

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    # `night_id` arrived in migration 007. A database written before it is still
    # perfectly readable — the night is then derived from the timestamp — so the
    # column is selected only when present rather than made a hard requirement.
    # A dataset builder that refuses to open last month's database is a worse
    # failure than the imprecise grouping it was trying to avoid.
    def _has_column(table: str, column: str) -> bool:
        return any(
            row[1] == column for row in con.execute(f"PRAGMA table_info({table})")
        )

    frame_night = (
        "f.night_id AS frame_night_id"
        if _has_column("frames", "night_id")
        else "NULL AS frame_night_id"
    )
    run_night = (
        "fr.night_id AS run_night_id"
        if _has_column("focus_runs", "night_id")
        else "NULL AS run_night_id"
    )

    rows = con.execute(
        f"""
        SELECT fs.run_id AS run_key, fs.position AS position, fs.hfr AS classical_hfr_px,
               fs.frame_id AS frame_id,
               f.local_path AS local_path, f.checksum_sha256 AS checksum_sha256,
               f.quality_json AS quality_json, f.filter_name AS frame_filter_name,
               f.temperature AS frame_temperature, f.exposure_s AS exposure_s,
               f.gain AS gain, f.captured_at AS frame_captured_at,
               {frame_night},
               fr.rig_id AS rig_id, fr.filter_name AS run_filter_name,
               fr.started_at AS run_started_at, {run_night}
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
            "label_version", "background_config_sha256", "detection_config_sha256",
            "saturation_source", "local_background_adu", "local_rms_adu",
        )
    }
    frames_read = 0
    frames_skipped_unknown_saturation = 0
    frames_skipped_ineligible = 0
    frames_background_unavailable: Counter = Counter()
    # Background measured but sep.extract raised (e.g. pixstack/deblend
    # overflow in a crowded field): no rows, counted here with the error text.
    frames_detection_backend_failure = 0
    detection_backend_errors: Counter = Counter()
    frames_by_saturation_source: Counter = Counter()
    frames_by_saturation_gate: Counter = Counter()
    rejection_counts: Counter = Counter()
    sep_flag_counts: Counter = Counter()
    stars_kept = 0
    sources_detected = 0
    sources_valid = 0
    db_dir = Path(db_path).resolve().parent

    for row in rows:
        quality = _quality_dict(row["quality_json"])
        # Simulation/preview archives exercise storage but do not supply
        # scientific labels. The fixture-only saturation override must not
        # override an explicit acquisition eligibility decision.
        if (
            quality.get("training_label_eligible") is False
            or quality.get("acquisition_domain") in {"simulation", "preview"}
        ):
            frames_skipped_ineligible += 1
            continue
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
        sat, saturation_source = saturation_decision_from_quality(row["quality_json"])
        if sat is None:
            if not allow_missing_saturation:
                frames_skipped_unknown_saturation += 1
                continue
            # Permissive mode: no gate at all, matching the daemon's runtime
            # behaviour of rejecting nothing rather than inventing a threshold.
            # measure_frame_sep records the gate as "not_performed".
        frames_by_saturation_source[saturation_source or "none"] += 1

        with Image.open(path) as image:
            img = np.asarray(image).astype(np.float64)
        if img.ndim != 2:
            continue
        frames_read += 1
        sources, frame_label = measure_frame_sep(
            img,
            saturation_level=sat,
            background_params=background_params,
            detection_params=detection_params,
            min_snr=min_snr,
            fit_fwhm=False,
        )
        frames_by_saturation_gate[frame_label.saturation_gate] += 1
        background = frame_label.background
        if background.get("level_status") != "measured" or background.get("noise_status") != "measured":
            for reason in background.get("reasons") or ["unknown"]:
                frames_background_unavailable[reason] += 1
            continue
        if frame_label.detection.get("status") != "measured":
            frames_detection_backend_failure += 1
            detection_backend_errors[str(frame_label.detection.get("backend_error"))] += 1
            continue
        sources_detected += len(sources)
        sources_valid += sum(source.valid for source in sources)
        rejection_counts.update(frame_label.rejection_counts)
        sep_flag_counts.update(frame_label.sep_flag_counts)

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
        night_id = resolve_night_id(
            row["frame_night_id"], row["run_night_id"], captured_at
        )

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
            columns["label_version"].append(LABEL_VERSION)
            columns["background_config_sha256"].append(background_sha)
            columns["detection_config_sha256"].append(detection_sha)
            columns["saturation_source"].append(saturation_source or "none")
            columns["local_background_adu"].append(source.local_background)
            columns["local_rms_adu"].append(source.local_rms)
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
        if frames_skipped_ineligible:
            hint += f" {frames_skipped_ineligible} frame(s) were excluded by acquisition eligibility."
        if frames_background_unavailable:
            hint += f" SEP background unavailable: {dict(frames_background_unavailable)}."
        if frames_detection_backend_failure:
            hint += f" SEP detection failed on {frames_detection_backend_failure} frame(s): {dict(detection_backend_errors)}."
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
        "local_background_adu": np.asarray(columns["local_background_adu"], dtype=np.float32),
        "local_rms_adu": np.asarray(columns["local_rms_adu"], dtype=np.float32),
    }
    for name in (
        "run_key", "frame_id", "frame_checksum", "rig_id", "camera_id",
        "filter_name", "captured_at", "night_id", "label_source",
        "label_version", "background_config_sha256", "detection_config_sha256",
        "saturation_source",
    ):
        dataset[name] = np.asarray(columns[name], dtype=np.str_)
    # The original run UUID is the minimum leakage boundary. Keep it explicit so
    # registry files do not depend on a rebuild-specific integer encoding.
    dataset["split_group"] = dataset["run_key"].copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dataset)
    from labels.psf_measure import SEP_WIDTH_DEFINITION, SNR_DEFINITION_BACKGROUND_ONLY

    manifest = {
        "dataset": str(output),
        "sha256": _sha256(output),
        "label_version": LABEL_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "label_method": "sep_spatial_mesh+sep_extract+sep_flux_radius",
        "label_method_description": (
            "SEP 1.4.1 mesh background B(x,y)/R(x,y), local-noise extraction with deblending, "
            "aperture-normalized flux_radius on data - B(x,y)"
        ),
        "sep_version": sep_version,
        "background_config": background_config,
        "background_config_sha256": background_sha,
        "detection_config": detection_config,
        "detection_config_sha256": detection_sha,
        "noise_source": "empirical_mesh_rms",
        "units": "adu",
        "width_definition": dict(SEP_WIDTH_DEFINITION),
        "snr_definition": SNR_DEFINITION_BACKGROUND_ONLY,
        "fwhm_fitted": False,
        "legacy_policy": (
            f"rows without label_version={LABEL_VERSION} and matching config hashes are "
            f"{LEGACY_UNVERIFIED} and are excluded by load_qualified_dataset"
        ),
        "frames_read": frames_read,
        "frames_skipped_unknown_saturation": frames_skipped_unknown_saturation,
        "frames_skipped_ineligible": frames_skipped_ineligible,
        "frames_background_unavailable": dict(frames_background_unavailable),
        "frames_detection_backend_failure": frames_detection_backend_failure,
        "detection_backend_errors": dict(detection_backend_errors),
        "frames_by_saturation_source": dict(frames_by_saturation_source),
        "frames_by_saturation_gate": dict(frames_by_saturation_gate),
        "sources_detected": sources_detected,
        "sources_valid": sources_valid,
        "rejection_counts": dict(rejection_counts),
        "sep_flag_counts": dict(sep_flag_counts),
        "stars_kept": stars_kept,
        "runs": len(run_index),
        "crop_size": crop_size,
        "max_stars_per_frame": max_stars_per_frame,
        # Load-bearing provenance: a permissive build has ungated labels in it
        # and must never be promoted or compared against a gated one.
        "saturation_gate": "permissive" if allow_missing_saturation else "enforced",
    }
    _manifest_path(output).write_text(json.dumps(manifest, indent=2))
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
    # A v2 default distinct from the frozen focus_quality_ds_v1.npz; an existing
    # output is never overwritten.
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
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
