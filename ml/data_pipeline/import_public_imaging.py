"""Bounded FQE/TED/BGE starter intake, without inventing focus telemetry.

Run from the repository root::

    python -m ml.data_pipeline.import_public_imaging --root ml/data/public --seed 42

The simulator writes native-pixel, trainer-compatible FQE crops,
32-pixel TED observations, and full-scene background tests.  A separately
identified HST ACS single exposure supplies a *diagnostic*, not three randomly
split sets of its correlated stars.  No model is trained by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np

from ml.data_pipeline.public_common import (
    download_file, grouped_split, sha256_file, write_json, write_jsonl,
)
from ml.preprocess.normalize import robust_normalize


SIM_VERSION = "achernar-analytic-prf-v1"
HST_ROOT = "jd5702jpq"
HST_FILENAME = HST_ROOT + "_flt.fits"
HST_URL = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/" + HST_FILENAME
HST_SIZE = 21_216_960
HST_DOC = "https://hst-docs.stsci.edu/acsdhb/chapter-3-acs-calibration-pipeline/3-5-calacs-examples"
HST_POLICY = "https://stdatu.stsci.edu/publishing/data-use"
HST_ATTRIBUTION = "https://archive.stsci.edu/publishing/data-attributions"
HST_DQ_DOC = "https://www.stsci.edu/hst/instrumentation/acs/data-analysis/dq-flag-definitions"


def aperture_hfr(signal: np.ndarray, center: tuple[float, float], radius: float) -> float | None:
    """Half signed aperture flux with SEP's five-subpixel aperture integration.

    The noiseless input is already integrated over pixels.  Negative noisy
    residuals are retained; clipping them would introduce a positive noise
    pedestal.  This finite-aperture reference is not an infinite analytic HFR
    or a physical best-focus label. ``None`` means an invalid measurement.
    """
    import sep

    y, x = np.indices(signal.shape, dtype=np.float64)
    r = np.hypot(x - center[0], y - center[1])
    selected = (r <= radius) & np.isfinite(signal)
    if selected.sum() < 4:
        return None
    total = np.asarray(signal, dtype=np.float64)[selected].sum()
    if not np.isfinite(total) or total <= 0:
        return None
    if not np.isfinite(signal).all():
        return None
    result, flags = sep.flux_radius(np.ascontiguousarray(signal, dtype=np.float64),
                                   np.asarray([center[0]], dtype=float),
                                   np.asarray([center[1]], dtype=float),
                                   np.asarray([radius], dtype=float), 0.5, subpix=5)
    if flags[0] or not np.isfinite(result[0]) or result[0] <= 0:
        return None
    return float(result[0])


def moment_shape(signal: np.ndarray, *, angle_threshold: float = 0.02) -> tuple[float, float | None]:
    """Finite-frame second-moment ellipticity and axial image angle.

    Angles run from +x toward +y (array row index), modulo pi.  They are not RA
    or Dec.  Round and empty images have no angle, represented by None/NaN.
    """
    f = np.clip(np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    total = float(f.sum())
    if total <= 0:
        return 0.0, None
    y, x = np.indices(f.shape)
    cx, cy = float((f*x).sum()/total), float((f*y).sum()/total)
    covariance = np.array([
        [(f*(x-cx)**2).sum(), (f*(x-cx)*(y-cy)).sum()],
        [(f*(x-cx)*(y-cy)).sum(), (f*(y-cy)**2).sum()],
    ]) / total
    values, vectors = np.linalg.eigh(covariance)
    ellipticity = float(1.0 - np.sqrt(max(values[0], 0) / max(values[1], 1e-15)))
    if ellipticity < angle_threshold:
        return ellipticity, None
    major = vectors[:, 1]
    return ellipticity, float(np.arctan2(major[1], major[0]) % np.pi)


def render_profile(size: int, params: dict, oversample: int = 3) -> np.ndarray:
    """Integrate an analytic Gaussian/Moffat profile and motion path per pixel.

    Midpoint quadrature approximates a top-hat pixel response.  Profiles are
    normalized by their analytic infinite-plane integral, *not* by the stamp
    sum, preserving edge truncation.  Motion is integrated before photon noise.
    """
    coord = (np.arange(size * oversample) + 0.5) / oversample - 0.5
    xx, yy = np.meshgrid(coord, coord)
    cx, cy = params["center"]
    pa, width = params["optical_pa_rad"], params["width_px"]
    ratio = params["axis_ratio"]
    co, si = np.cos(pa), np.sin(pa)
    motion = params["motion"]
    length, angle = params["trail_length_px"], params["trail_pa_image_rad"]
    if motion == "none":
        path = [(0.0, 0.0)]
    elif motion == "linear":
        path = [(t*length*np.cos(angle), t*length*np.sin(angle)) for t in np.linspace(-0.5, 0.5, 13)]
    elif motion == "curved":
        path = [(
            t*length*np.cos(angle) - (t*t - 0.1)*length*0.5*np.sin(angle),
            t*length*np.sin(angle) + (t*t - 0.1)*length*0.5*np.cos(angle),
        ) for t in np.linspace(-0.5, 0.5, 13)]
    else:
        # A fixed deterministic 2D jitter path; angle is not a straight trail.
        path = [(length*0.25*np.cos(t), length*0.17*np.sin(2*t)) for t in np.linspace(0, 2*np.pi, 13, endpoint=False)]
    result = np.zeros_like(xx)
    for dx, dy in path:
        u = ((xx-cx-dx)*co + (yy-cy-dy)*si) / width
        v = (-(xx-cx-dx)*si + (yy-cy-dy)*co) / (width*ratio)
        r2 = u*u + v*v
        if params["profile"] == "gaussian":
            profile = np.exp(-0.5*r2) / (2*np.pi*width*width*ratio)
        else:
            beta = params["moffat_beta"]
            profile = (1+r2)**(-beta) * (beta-1) / (np.pi*width*width*ratio)
        result += profile / len(path)
    integrated = result.reshape(size, oversample, size, oversample).mean(axis=(1, 3))
    return integrated * params["flux_electrons"]


def _params(rng: np.random.Generator, size: int) -> dict:
    edge = bool(rng.random() < 0.06)
    center = [size/2-0.5+rng.uniform(-0.45, 0.45), size/2-0.5+rng.uniform(-0.45, 0.45)]
    if edge:
        center[int(rng.integers(2))] = float(rng.choice([3.0, size-4.0]))
    motion = str(rng.choice(["none", "linear", "curved", "jitter"], p=[0.40, 0.40, 0.10, 0.10]))
    return {
        "profile": str(rng.choice(["gaussian", "moffat"])),
        "width_px": float(np.exp(rng.uniform(np.log(0.8), np.log(4.5)))),
        "moffat_beta": float(rng.uniform(2.2, 5.0)),
        "axis_ratio": float(rng.uniform(0.65, 1.0)) if rng.random() < 0.30 else 1.0,
        "optical_pa_rad": float(rng.uniform(0, np.pi)),
        "center": center,
        "flux_electrons": float(np.exp(rng.uniform(np.log(700), np.log(180_000)))),
        "motion": motion,
        "trail_length_px": 0.0 if motion == "none" else float(rng.uniform(0.2, 13.0)),
        "trail_pa_image_rad": float(rng.uniform(0, np.pi)),
        "edge_case": edge,
        "blended": bool(rng.random() < 0.07),
    }


def _background(rng: np.random.Generator, size: int) -> tuple[np.ndarray, float, dict]:
    sky = float(np.exp(rng.uniform(np.log(3), np.log(1000))))
    gx, gy = rng.uniform(-0.35, 0.35, 2) * sky
    y, x = np.indices((size, size), dtype=np.float64)
    result = np.maximum(sky + gx*(x/size-0.5) + gy*(y/size-0.5), 0.01)
    read_noise = float(rng.uniform(1.5, 12.0))
    return result, read_noise, {"sky_electrons": sky, "gradient_x_per_frame": float(gx), "gradient_y_per_frame": float(gy), "read_noise_electrons": read_noise}


def _detector(rng, signal, background, read_noise, *, corrupt=False):
    expected = np.maximum(signal + background, 0)
    bias, ceiling = 500.0, 65_535.0
    unclipped = rng.poisson(expected).astype(np.float64) + rng.normal(0, read_noise, expected.shape) + bias
    saturated = unclipped >= ceiling * 0.98
    raw = np.rint(np.clip(unclipped, 0, ceiling)).astype(np.float32)
    bad = np.zeros(raw.shape, dtype=bool)
    if corrupt:
        y, x = raw.shape[0]//2, raw.shape[1]//2
        bad[y, x] = True
        raw[y, x] = np.nan
    variance = expected + read_noise*read_noise
    return raw, saturated, bad, variance.astype(np.float32), bias, ceiling


def _write_npz(path: Path, arrays: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _fqe_arrays(crops, references, measurements, records):
    n = len(crops)
    ids = [r["source_id"] for r in records]
    # Compatibility fields are singleton *sample* keys, never inferred focus
    # runs. RankNet has zero valid pairs even if the legacy default is enabled.
    return {
        "crops": np.asarray(crops, dtype=np.float32).reshape(n, 64, 64),
        "hfr_px": np.asarray(references, dtype=np.float32),
        "log_hfr": np.log(np.asarray(references, dtype=np.float32)),
        "reference_hfr_px": np.asarray(references, dtype=np.float32),
        "measured_hfr_px": np.asarray(measurements, dtype=np.float32),
        "run_id": np.arange(n, dtype=np.int64),
        "run_key": np.asarray(ids, dtype="U128"),
        "run_key_semantics": np.full(n, "singleton_sample_not_focus_run", dtype="U40"),
        "ranking_eligible": np.zeros(n, dtype=bool),
        "split_group": np.asarray([r["split_group"] for r in records], dtype="U100"),
        "split": np.asarray([r["split"] for r in records], dtype="U10"),
        "frame_id": np.asarray([r["frame_id"] for r in records], dtype="U128"),
        "star_snr": np.asarray([r["star_snr"] for r in records], dtype=np.float32),
        "rig_id": np.asarray([r["rig_id"] for r in records], dtype="U64"),
        "camera_id": np.asarray([r["camera_id"] for r in records], dtype="U64"),
        "filter_name": np.asarray([r["filter_name"] for r in records], dtype="U32"),
        "night_id": np.asarray([r["night_id"] for r in records], dtype="U32"),
    }


def _frozen_registry(dataset_path: Path, arrays: dict, seed: int) -> dict:
    splits = {name: np.flatnonzero(arrays["split"] == label) for name, label in [("train", "train"), ("val", "validation"), ("test", "test")]}
    if any(index.size == 0 for index in splits.values()):
        raise ValueError("Synthetic feasibility dataset needs nonempty family-grouped splits; increase --families")
    registry = {
        "dataset_sha256": sha256_file(dataset_path), "n_samples": len(arrays["crops"]),
        "group_field": "split_group", "strategy": "stable_ancestor_hash", "seed": seed,
        "counts": {name: len(index) for name, index in splits.items()},
        "groups": {name: sorted(set(arrays["split_group"][index].tolist())) for name, index in splits.items()},
        "indices": {name: index.tolist() for name, index in splits.items()},
    }
    write_json(dataset_path.with_suffix(".splits.json"), registry)
    return registry


def prepare_synthetic(root: Path, seed=42, families=800, variants=3, scene_count=32) -> dict:
    if families < 12 or variants < 1 or variants > 20 or scene_count < 0:
        raise ValueError("Use at least 12 families, 1–20 variants, and nonnegative scene count")
    output = root / "stellar_sim_v1"
    output.mkdir(parents=True, exist_ok=True)
    (output/'generator_source.py').write_bytes(Path(__file__).read_bytes())
    normalization_source=Path(sys.modules[robust_normalize.__module__].__file__)
    (output/'normalization_source.py').write_bytes(normalization_source.read_bytes())
    rng = np.random.default_rng(seed)
    groups = [f"{SIM_VERSION}:seed{seed}:family{i:05d}" for i in range(families)]
    split_map = grouped_split(groups, seed=seed)
    rows, accepted_rows, accepted_crops, references, measured = [], [], [], [], []
    raw64, reference64, masks64, ted32, ted_ref32 = [], [], [], [], []
    ted_ell, ted_angle, ted_angle_valid, ted_valid = [], [], [], []
    full_scenes, full_backgrounds, full_variances, full_masks, scene_rows = [], [], [], [], []
    for family, ancestor in enumerate(groups):
        params = _params(rng, 64)
        primary = render_profile(64, params)
        signal = primary.copy()
        if params["blended"]:
            neighbor = dict(params, center=[params["center"][0]+rng.uniform(4, 9), params["center"][1]+rng.uniform(-4, 4)], flux_electrons=params["flux_electrons"]*rng.uniform(0.3, 0.9), blended=False)
            signal += render_profile(64, neighbor)
        ref_hfr = aperture_hfr(primary, tuple(params["center"]), 24.0)
        for variant in range(variants):
            background, read_noise, noise_params = _background(rng, 64)
            corrupt = bool(rng.random() < 0.015)
            raw, sat, bad, variance, bias, ceiling = _detector(rng, signal, background, read_noise, corrupt=corrupt)
            measured_hfr = aperture_hfr(raw-background-bias, tuple(params["center"]), 24.0)
            ell, angle = moment_shape(primary[16:48, 16:48])
            yy, xx = np.indices((64, 64))
            aperture = np.hypot(xx-params["center"][0], yy-params["center"][1]) <= 24
            snr = float(primary[aperture].sum() / np.sqrt(variance[aperture].sum()))
            rejections = []
            if params["edge_case"]:
                rejections.append("edge_truncated")
            if params["blended"]:
                rejections.append("blended")
            if sat[aperture].any():
                rejections.append("saturated")
            if bad.any() or not np.isfinite(raw).all():
                rejections.append("invalid_pixel")
            if snr < 8:
                rejections.append("low_snr")
            if ell > 0.35:
                rejections.append("elongated")
            if ref_hfr is None or ref_hfr <= 0 or measured_hfr is None or measured_hfr <= 0:
                rejections.append("invalid_hfr")
            frame_id = f"sim-seed{seed}-f{family:05d}-v{variant:02d}"
            row = {
                "source_id": frame_id, "ancestor_id": ancestor, "frame_id": frame_id,
                "split_group": ancestor, "split": split_map[ancestor],
                "task": ["FQE", "TED", "BGE"], "training_eligible": not rejections,
                "fqe_eligible": not rejections, "fqe_rejections": rejections,
                "license": "project-generated-synthetic; repository terms apply",
                "is_fully_simulated": True, "is_augmented_real": False,
                "rig_id": "synthetic-no-physical-rig", "camera_id": "synthetic-linear-mono-electron-detector",
                "filter_name": "not_modeled", "night_id": "synthetic-no-observing-night",
                "units": "electrons with fixed 500-electron numeric bias; gain=1 electron/numeric unit",
                "star_snr": snr, "parameters": params, "noise": noise_params,
                "saturation_numeric_level": ceiling*0.98, "bias_numeric_level": bias,
                "reference_hfr_px": ref_hfr, "measured_hfr_px": measured_hfr,
                "measured_hfr_method": "sep_subpix5_radius24_signed_flux_with_oracle_background_v1",
                "reference_hfr_method": "sep_subpix5_pixel_integrated_noiseless_primary_radius24_v1",
                "ranking_eligible": False, "focuser_position": None, "signed_focus_offset": None,
                "raw_path": "observations.npz", "array_index": len(rows),
            }
            raw64.append(raw)
            reference64.append(primary.astype(np.float32))
            masks64.append((bad.astype(np.uint8) + 2*sat.astype(np.uint8)))
            ted32.append(raw[16:48, 16:48])
            ted_ref32.append(primary[16:48, 16:48].astype(np.float32))
            ted_ell.append(ell)
            ted_angle.append(np.nan if angle is None else angle)
            ted_angle_valid.append(angle is not None)
            ted_valid.append(not any(r in rejections for r in ("edge_truncated", "blended", "saturated", "invalid_pixel", "low_snr")))
            if not rejections:
                row["fqe_index"] = len(accepted_crops)
                accepted_crops.append(robust_normalize(raw)[0])
                references.append(ref_hfr)
                measured.append(measured_hfr)
                accepted_rows.append(row)
            rows.append(row)
        if family < min(scene_count, families):
            scene = np.zeros((128, 128), dtype=np.float64)
            star_mask = np.zeros(scene.shape, dtype=bool)
            yy, xx = np.indices(scene.shape)
            for _ in range(int(rng.integers(2, 18))):
                sp = _params(rng, 128)
                sp["center"] = rng.uniform(0, 127, 2).tolist()
                sp["blended"] = False
                scene += render_profile(128, sp)
                star_mask |= np.hypot(xx-sp["center"][0], yy-sp["center"][1]) < max(5, sp["width_px"]*5+sp["trail_length_px"])
            bg, read, scene_noise = _background(rng, 128)
            raw, sat, bad, var, bias, ceiling = _detector(rng, scene, bg, read, corrupt=(family % 13 == 0))
            full_scenes.append(raw)
            full_backgrounds.append((bg+bias).astype(np.float32))
            full_variances.append(var)
            full_masks.append(bad.astype(np.uint8)+2*sat.astype(np.uint8)+4*star_mask.astype(np.uint8))
            scene_rows.append({"source_id": f"sim-scene-{family:05d}", "ancestor_id": ancestor, "split_group": ancestor, "split": split_map[ancestor], "training_eligible": False, "task": "BGE_classical_benchmark", "noise": scene_noise})
    fqe = _fqe_arrays(accepted_crops, references, measured, accepted_rows)
    fqe_artifact = _write_npz(output / "fqe.npz", fqe)
    registry = _frozen_registry(output / "fqe.npz", fqe, seed)
    observations_artifact = _write_npz(output / "observations.npz", {
        "raw_crops": np.asarray(raw64, dtype=np.float32),
        "noiseless_primary": np.asarray(reference64, dtype=np.float32),
        "pixel_flags": np.asarray(masks64, dtype=np.uint8),
        "fqe_eligible": np.asarray([r["fqe_eligible"] for r in rows]),
        "split_group": np.asarray([r["split_group"] for r in rows]),
        "split": np.asarray([r["split"] for r in rows]),
    })
    ted_artifact = _write_npz(output / "ted.npz", {
        "crops": np.asarray([robust_normalize(x)[0] for x in ted32], dtype=np.float32),
        "raw_crops": np.asarray(ted32, dtype=np.float32),
        "noiseless_primary": np.asarray(ted_ref32, dtype=np.float32),
        "reference_ellipticity": np.asarray(ted_ell, dtype=np.float32),
        "reference_pa_image_rad": np.asarray(ted_angle, dtype=np.float32),
        "reference_angle_valid": np.asarray(ted_angle_valid, dtype=bool),
        "training_eligible": np.asarray(ted_valid, dtype=bool),
        "trail_length_px": np.asarray([r["parameters"]["trail_length_px"] for r in rows], dtype=np.float32),
        "motion_type": np.asarray([r["parameters"]["motion"] for r in rows]),
        "applied_linear_pa_image_rad": np.asarray([r["parameters"]["trail_pa_image_rad"] if r["parameters"]["motion"] == "linear" else np.nan for r in rows], dtype=np.float32),
        "split_group": np.asarray([r["split_group"] for r in rows]),
        "split": np.asarray([r["split"] for r in rows]),
    })
    bge_artifact = _write_npz(output / "bge.npz", {
        "images": np.asarray(full_scenes, dtype=np.float32).reshape(-1, 128, 128),
        "background_truth": np.asarray(full_backgrounds, dtype=np.float32).reshape(-1, 128, 128),
        "variance_truth_preclip": np.asarray(full_variances, dtype=np.float32).reshape(-1, 128, 128),
        "pixel_flags": np.asarray(full_masks, dtype=np.uint8).reshape(-1, 128, 128),
        "split_group": np.asarray([r["split_group"] for r in scene_rows]),
        "split": np.asarray([r["split"] for r in scene_rows]),
    })
    write_jsonl(output / "records.jsonl", rows)
    write_jsonl(output / "bge_records.jsonl", scene_rows)
    manifest = {
        "dataset_id": "stellar_sim_v1", "generator": SIM_VERSION, "seed": seed,
        "generator_source_sha256": sha256_file(output/'generator_source.py'),
        "normalization_source_sha256":sha256_file(output/'normalization_source.py'), "numpy_version": np.__version__,
        "families": families, "variants_per_family": variants, "observations": len(rows),
        "fqe_accepted": len(accepted_rows), "fqe_split_counts": registry["counts"],
        "fqe_rejections": dict(Counter(reason for r in rows for reason in r["fqe_rejections"])),
        "ted_eligible": int(sum(ted_valid)), "bge_scenes": len(scene_rows),
        "artifacts": [fqe_artifact, observations_artifact, ted_artifact, bge_artifact],
        "preprocessing": "native pixel scale; FQE 64x64 / TED 32x32; per-crop NumPy linear percentile 1/99 [0,1]",
        "mask_bits": {"1": "invalid detector pixel", "2": "98 percent saturation gate", "4": "synthetic star/source region (BGE only)"},
        "reference_definition": "noiseless pixel-integrated primary; finite aperture radius24; SEP five-subpixel signed curve of growth",
        "coordinate_convention": "+x columns, +y rows; axial PA radians [0,pi), round PA undefined; no RA/Dec mapping",
        "training_config_required": {"lambda_rank": 0, "split_registry": "fqe.splits.json"},
        "ranking_policy": "run_id/run_key are singleton sample compatibility keys, not focus runs; no positions or direction labels",
        "limitations": [
            "Analytic midpoint-quadrature generator, not GalSim or a calibrated R50/Z114 optical model.",
            "Synthetic mono electron-space profiles omit CFA, true diffraction, sensor pattern noise and real optical calibration.",
            "Measured noisy HFR uses known simulation centroid/background and is an oracle-background diagnostic, not a deployed measurement.",
            "Noiseless finite-aperture HFR and noisy measured HFR are distinct; neither is a best-focus motor label.",
            "Preclip variance is Poisson plus Gaussian read variance; integer quantization contributes additional variance and clipping changes the distribution.",
            "A family and every derivative share one frozen split across FQE, TED, BGE; no independent-nights or cross-rig qualification claim.",
        ],
        "license": "project-generated synthetic; repository terms apply", "status": "prepared_for_feasibility_experiments_not_model_qualification",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def prepare_hst(root: Path, *, max_sources=2000, product=HST_ROOT,
                dataset_name="stellar_fits_pilot", expected_size=HST_SIZE) -> dict:
    """One public ACS/WFC FLT exposure with preserved SCI/ERR/DQ.

    All its crops share a single untouched diagnostic split. The calibrated
    electrons and ERR pipeline estimates are never described as raw Canon ADU
    or physical noise ground truth. Reject masked cores and saturation; retain
    only explicitly identified experimental references with <=2% masked wings.
    """
    from astropy.io import fits
    import sep

    output = root / dataset_name
    output.mkdir(parents=True, exist_ok=True)
    filename = product + "_flt.fits"
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/" + filename
    source = download_file(url, output / "originals" / filename, max_bytes=180_000_000, expected_size=expected_size)
    with fits.open(source["path"], memmap=False, checksum=True) as hdus:
        primary = hdus[0].header
        sci_hdu, err_hdu, dq_hdu = hdus["SCI", 1], hdus["ERR", 1], hdus["DQ", 1]
        image = np.array(sci_hdu.data, dtype=np.float32, order="C")
        error = np.array(err_hdu.data, dtype=np.float32, order="C")
        dq = np.array(dq_hdu.data, dtype=np.uint16, order="C")
        if image.ndim != 2 or min(image.shape) < 64 or image.shape != error.shape or image.shape != dq.shape:
            raise ValueError("HST pilot geometry/SCI ERR DQ contract changed")
        if primary.get("INSTRUME") != "ACS" or primary.get("ROOTNAME", "").strip() != product or sci_hdu.header.get("BUNIT", "").strip() != "ELECTRONS":
            raise ValueError("HST product identity/native unit contract changed")
        header_summary = {"primary": {k: primary.get(k) for k in ("ROOTNAME", "FILENAME", "TARGNAME", "INSTRUME", "DETECTOR", "APERTURE", "PROPOSID", "DATE-OBS", "TIME-OBS", "EXPTIME", "FILTER1", "FILTER2", "SUBARRAY", "CCDGAIN", "CAL_VER", "BIASCORR", "DARKCORR", "FLATCORR", "PCTECORR", "DQICORR")}, "science": {k: sci_hdu.header.get(k) for k in ("EXTNAME", "EXTVER", "BUNIT", "NAXIS1", "NAXIS2", "CCDCHIP", "LTV1", "LTV2", "LTM1_1", "LTM2_2")}, "hdu_inventory": [{"index": i, "name": h.name, "shape": list(h.data.shape) if getattr(h.data, "shape", None) is not None else None} for i, h in enumerate(hdus)]}
        write_json(output / "headers.json", header_summary)
        (output / "primary_header.txt").write_text(primary.tostring(sep="\n", endcard=True, padding=False)+"\n")
        (output / "science_header.txt").write_text(sci_hdu.header.tostring(sep="\n", endcard=True, padding=False)+"\n")
    mask = (dq != 0) | ~np.isfinite(image) | ~np.isfinite(error) | (error <= 0)
    if mask.all():raise ValueError("HST SCI1 contains no valid analysis pixels")
    cleaned = image.copy()
    finite_median = float(np.median(image[~mask]))
    cleaned[mask] = finite_median
    bkg = sep.Background(cleaned, mask=mask)
    residual = np.ascontiguousarray(cleaned-bkg.back())
    # ERR propagates calibration noise; it is a pipeline estimate, not truth.
    safe_error = np.maximum(error, 1e-6)
    safe_error[mask] = max(float(np.median(error[~mask])), 1.0)
    # Full-frame globular-cluster fields can exceed SEP's default 300k
    # active-pixel stack. Bound allocation by this image; keep the threshold
    # and masks identical, and restore the process-global setting afterwards.
    previous_stack = sep.get_extract_pixstack()
    try:
        sep.set_extract_pixstack(max(previous_stack, image.size))
        objects = sep.extract(residual, 6.0, err=safe_error, mask=mask, minarea=5, deblend_nthresh=32, deblend_cont=0.005)
    finally:
        sep.set_extract_pixstack(previous_stack)
    x, y = objects["x"], objects["y"]
    rmax = np.clip(6*objects["a"], 6, 16)
    hfr, hfr_flags = sep.flux_radius(residual, x, y, rmax, 0.5, mask=mask, subpix=5)
    flux, flux_error, aperture_flags = sep.sum_circle(residual, x, y, np.maximum(3*objects["a"], 4), err=safe_error, mask=mask, subpix=5)
    ranked = sorted(range(len(objects)), key=lambda i: -float(flux[i] / max(flux_error[i], 1e-6)))[:max_sources]
    rows, fqe_rows, crops, refs, measured, raw_crops, ted_crops = [], [], [], [], [], [], []
    for i in ranked:
        obj = objects[i]
        cx, cy = int(np.floor(x[i]+0.5)), int(np.floor(y[i]+0.5))
        ell = float(1-obj["b"]/max(obj["a"], 1e-6))
        snr = float(flux[i]/max(flux_error[i], 1e-6))
        rejections = []
        if cx < 32 or cy < 32 or cx+32 > image.shape[1] or cy+32 > image.shape[0]:
            rejections.append("edge_truncated")
        if int(obj["flag"]) & sep.OBJ_MERGED:
            rejections.append("blended")
        if int(obj["flag"]) & sep.OBJ_TRUNC:
            rejections.append("edge_truncated")
        aperture = float(rmax[i])
        ix0, ix1 = max(0, int(x[i]-aperture-1)), min(image.shape[1], int(x[i]+aperture+2))
        iy0, iy1 = max(0, int(y[i]-aperture-1)), min(image.shape[0], int(y[i]+aperture+2))
        yy, xx = np.mgrid[iy0:iy1, ix0:ix1]
        local_aperture = np.hypot(xx-x[i], yy-y[i]) <= aperture
        local_dq = dq[iy0:iy1, ix0:ix1][local_aperture]
        if np.any(local_dq & (256 | 2048)):
            rejections.append("saturated_dq")
        local_mask = mask[iy0:iy1, ix0:ix1]
        masked_fraction = float(local_mask[local_aperture].mean())
        core = np.hypot(xx-x[i], yy-y[i]) <= max(2, 1.5*float(obj["a"]))
        if local_mask[core].any() or masked_fraction > 0.02:
            rejections.append("quality_masked_core_or_wings")
        neighbors = np.hypot(x-x[i], y-y[i])
        if np.any((neighbors > 0) & (neighbors < max(8, aperture))):
            rejections.append("crowded_aperture")
        if ell > 0.35:
            rejections.append("elongated")
        if snr < 15 or not np.isfinite(snr):
            rejections.append("low_snr")
        fatal_flags = ~int(sep.APER_HASMASKED)
        if int(hfr_flags[i]) & fatal_flags or int(aperture_flags[i]) & fatal_flags or not np.isfinite(hfr[i]) or hfr[i] <= 0:
            rejections.append("invalid_aperture_measurement")
        row = {
            "source_id": f"HST:{product}:SCI1:source{i:05d}", "ancestor_id": f"HST:{product}",
            "frame_id": product, "split_group": f"HST:{product}", "split": "test",
            "task": ["FQE_diagnostic", "TED_unlabeled_observation", "BGE_pipeline_comparison"],
            "training_eligible": False, "fqe_quality_accepted": not rejections,
            "rejections": sorted(set(rejections)), "ranking_eligible": False,
            "license": "public-domain HST mission data; retain STScI/NASA/ESA acknowledgement",
            "source_path": "originals/" + filename, "source_sha256": source["sha256"],
            "masked_aperture_fraction": masked_fraction,
            "label_quality_tier": "strict_unmasked_aperture" if masked_fraction == 0 else "mask_aware_experimental",
            "mask_aware_reference_qualified": False,
            "x_native_pixel": float(x[i]), "y_native_pixel": float(y[i]),
            "crop_origin_xy": [cx-32, cy-32], "units": "ELECTRONS; calibrated FLT, not raw ADU",
            "hfr_px": float(hfr[i]) if np.isfinite(hfr[i]) else None,
            "hfr_method": "sep.flux_radius; masked background-subtracted; rmax=clip(6a,6,16); subpix5",
            "hfr_aperture_radius_px": aperture, "measured_ellipticity": ell, "star_snr": snr,
            "rig_id": "HST-ACS-WFC", "camera_id": "HST-ACS-WFC-CCD1",
            "filter_name": ";".join(str(header_summary["primary"][k]) for k in ("FILTER1","FILTER2") if header_summary["primary"][k] and not str(header_summary["primary"][k]).startswith("CLEAR")), "night_id": header_summary["primary"]["DATE-OBS"],
            "focuser_position": None, "signed_focus_offset": None, "tracking_error_label": None,
        }
        if "edge_truncated" not in rejections:
            crop = image[cy-32:cy+32, cx-32:cx+32]
            row["observation_index"] = len(raw_crops)
            raw_crops.append(crop)
            ted_crops.append(crop[16:48, 16:48])
            if not rejections:
                row["fqe_index"] = len(crops)
                model_crop = crop.copy()
                crop_mask = mask[cy-32:cy+32, cx-32:cx+32]
                model_crop[crop_mask] = bkg.back()[cy-32:cy+32, cx-32:cx+32][crop_mask]
                crops.append(robust_normalize(model_crop)[0])
                refs.append(float(hfr[i]))
                measured.append(float(hfr[i]))
                fqe_rows.append(row)
        rows.append(row)
    artifacts = [_write_npz(output / "frame.npz", {"science_electrons": image, "error_pipeline_electrons": error, "dq": dq, "valid_mask": ~mask, "sep_background_estimate_electrons": bkg.back().astype(np.float32)})]
    artifacts.append(_write_npz(output / "fqe_diagnostic.npz", _fqe_arrays(crops, refs, measured, fqe_rows)))
    artifacts.append(_write_npz(output / "observations.npz", {"raw_crops64": np.asarray(raw_crops, dtype=np.float32).reshape(-1, 64, 64), "raw_crops32": np.asarray(ted_crops, dtype=np.float32).reshape(-1, 32, 32)}))
    write_jsonl(output / "records.jsonl", rows)
    manifest = {
        "dataset_id": dataset_name, "source_id": "mast:HST/product/"+filename,
        "source": dict(source, path="originals/"+filename),
        "license": "public-domain HST mission data", "license_source": HST_POLICY,
        "source_documentation": HST_DOC, "attribution_source": HST_ATTRIBUTION, "dq_documentation": HST_DQ_DOC,
        "instrument": "HST ACS/WFC", "product_type": "single calibrated FLT exposure; not drizzled/coadded; CTE correction omitted",
        "native_units": "ELECTRONS", "native_shape": list(image.shape), "science_hdu": ["SCI", 1], "error_hdu": ["ERR", 1], "dq_hdu": ["DQ", 1],
        "source_count_detected": len(objects), "source_count_inspected": len(rows), "fqe_quality_accepted": len(fqe_rows),
        "training_eligible": False, "split": "test", "split_group": "HST:"+product,
        "rejections": dict(Counter(reason for row in rows for reason in row["rejections"])),
        "quality_mask_policy": "mask all nonzero DQ/nonfinite/nonpositive ERR; reject masked core or >2% masked aperture, saturation bits256|2048 always reject; remaining masked-wing labels experimental",
        "crop_imputation": "masked input pixels replaced by SEP background before model normalization; originals and DQ retained",
        "strict_saturation_policy": "pipeline DQ flags used instead of invented scalar threshold; no Canon calibration inferred",
        "artifacts": artifacts, "sep_version": sep.__version__, "numpy_version": np.__version__,
        "limitations": ["One correlated exposure, frozen as diagnostic test only; cannot establish cross-night or cross-rig transfer.", "HST calibrated electrons and pipeline ERR do not calibrate Canon R50 ADU/gain/read noise.", "Crowded-field detections include nonstellar contaminants; quality gates are conservative and not a human morphology audit.", "SEP labels are measurement references, not independent physical width truth; no focus positions or tracking causes available.", "FLT subarray retains optical distortion, possible residual striping, and no CTE correction; no pixel area map applied."],
        "status": "prepared_external_diagnostic_not_training_campaign",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("ml/data/public"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--families", type=int, default=800)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--scene-count", type=int, default=32)
    parser.add_argument("--max-real-sources", type=int, default=2000)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--real-only", action="store_true")
    args = parser.parse_args(argv)
    if args.synthetic_only and args.real_only:
        parser.error("--synthetic-only and --real-only are mutually exclusive")
    result = {}
    if not args.real_only:
        manifest = prepare_synthetic(args.root, args.seed, args.families, args.variants, args.scene_count)
        result["synthetic"] = {k: manifest[k] for k in ("observations", "fqe_accepted", "fqe_split_counts", "ted_eligible", "bge_scenes")}
    if not args.synthetic_only:
        manifest = prepare_hst(args.root, max_sources=args.max_real_sources)
        result["real_fits"] = {k: manifest[k] for k in ("source_count_detected", "source_count_inspected", "fqe_quality_accepted", "training_eligible")}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
