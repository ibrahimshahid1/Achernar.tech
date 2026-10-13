#!/usr/bin/env python3
"""Independent A03 numeric oracle and phase/color characterization.

The predeclared acceptance protocol is docs/a03-validation-protocol-2026-10-12.md.
NumPy is the sole required external dependency. SEP is an optional, explicitly
reported measurement comparison. The oracle never imports production code.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL = "a03-cfa-validation-v1-2026-10-12"
SEED = 20261012
MAX_ULP = 4
PATTERNS = {
    "rggb": ((0, 1), (1, 2)),
    "bggr": ((2, 1), (1, 0)),
    "grbg": ((1, 0), (2, 1)),
    "gbrg": ((1, 2), (0, 1)),
}
KERNEL_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 4
KERNEL_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float64) / 4
DTYPES = {"u8": "u1", "u16": "u2", "f32": "f4"}


def convolve_constant(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Independent sparse-plane linear operator; compare interiors only."""
    height, width = array.shape
    padded = np.pad(array, 1)
    result = np.zeros((height, width), dtype=np.float64)
    for y in range(3):
        for x in range(3):
            result += kernel[y, x] * padded[y : y + height, x : x + width]
    return result


def channel_map(shape: tuple[int, int], pattern: str, origin: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape)
    table = np.asarray(PATTERNS[pattern])
    return table[(y + origin[1]) % 2, (x + origin[0]) % 2]


def reference(samples: np.ndarray, descriptor: dict[str, Any], masks: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    invalid = ~np.isfinite(values)
    invalid |= np.asarray(masks.get("invalid", np.zeros(values.size)), dtype=bool).reshape(values.shape)
    bad = np.asarray(masks.get("bad_pixel", np.zeros(values.size)), dtype=bool).reshape(values.shape)
    saturation = np.asarray(masks.get("native_saturation", np.zeros(values.size)), dtype=bool).reshape(values.shape)
    threshold = descriptor["saturation"]
    if threshold is not None:
        saturation |= values >= threshold["adu"]
    values = np.where(np.isfinite(values), values, 0)
    if descriptor["color"] == "mono":
        rgb = np.repeat(values[..., None], 3, axis=2)
        edge = np.zeros(values.shape, dtype=bool)
        derived = {"bad_pixel": bad, "invalid": invalid, "native_saturation": saturation}
    else:
        cfa = descriptor["color"]["bayer"]
        channels = channel_map(values.shape, cfa["pattern"], (cfa["origin_x"], cfa["origin_y"]))
        rgb = np.stack([
            convolve_constant(values * (channels == c), KERNEL_G if c == 1 else KERNEL_RB)
            for c in range(3)
        ], axis=2)
        derived = {}
        for name, native in [("bad_pixel", bad), ("invalid", invalid), ("native_saturation", saturation)]:
            propagated = np.zeros(values.shape, dtype=bool)
            for c in range(3):
                kernel = KERNEL_G if c == 1 else KERNEL_RB
                propagated |= convolve_constant(native * (channels == c), kernel) > 0
            derived[name] = propagated
        edge = np.zeros(values.shape, dtype=bool)
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    derived["edge"] = edge
    # Production invalidity includes bad support: invalid products are NaN,
    # while the distinct bad mask preserves the exclusion's original reason.
    derived["invalid"] |= derived["bad_pixel"]
    return {"rgb": rgb.astype(np.float32), "pixels": rgb.mean(axis=2).astype(np.float32), "masks": derived}


def encode_case(identifier: str, samples: np.ndarray, *, pattern: str | None = "rggb",
                origin: tuple[int, int] = (0, 0), kind: str = "u16", endian: str = "little",
                padding: int = 0, masks: dict[str, Any] | None = None,
                saturation: float | None = None) -> dict[str, Any]:
    height, width = samples.shape
    dtype = np.dtype(("<" if endian == "little" else ">") + DTYPES[kind])
    rows = samples.astype(dtype)
    data = b"".join(row.tobytes() + bytes([0xA5]) * padding for row in rows)
    color: Any = "mono" if pattern is None else {
        "bayer": {"pattern": pattern, "origin_x": origin[0], "origin_y": origin[1]}
    }
    descriptor = {
        "schema_version": 1, "width": width, "height": height,
        "row_stride_bytes": width * dtype.itemsize + padding,
        "sample_type": kind, "byte_order": endian, "color": color,
        "domain": "linear_residual_adu" if kind == "f32" else "native_adu",
        "geometry": {"analysis_to_sensor": [[1., 0., float(origin[0])], [0., 1., float(origin[1])]],
                     "coordinate_system": "native_sensor"},
        "saturation": None if saturation is None else {"adu": saturation, "source": "synthetic_protocol_v1"},
        "calibration_ids": ["synthetic-residual-v1"] if kind == "f32" else [],
    }
    return {"id": identifier, "descriptor": descriptor, "data": list(data), "masks": masks or {}}


def production(binary: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = "".join(json.dumps(case, allow_nan=False, separators=(",", ":")) + "\n" for case in cases)
    completed = subprocess.run([str(binary.resolve())], input=payload, text=True, capture_output=True, check=True)
    outputs = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    if len(outputs) != len(cases):
        raise RuntimeError(f"fixture returned {len(outputs)} lines for {len(cases)} cases")
    if [o.get("id") for o in outputs] != [c["id"] for c in cases]:
        raise RuntimeError("fixture output identity/order mismatch")
    return outputs


def ulp_distance(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    def ordered(value: np.ndarray) -> np.ndarray:
        bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32).astype(np.int64)
        return np.where(bits & 0x80000000, 0x80000000 - (bits & 0x7fffffff), bits + 0x80000000)
    return np.abs(ordered(actual) - ordered(expected))


def numerical_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any] | None]]:
    y, x = np.indices((9, 11))
    requests, expected = [], []
    for pattern in PATTERNS:
        for origin in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            # Distinct channel planes and nonlinear texture ensure a mislabeled
            # CFA or channel permutation cannot pass by reproducing a ramp.
            channels = channel_map((9, 11), pattern, origin)
            samples = np.asarray([700., 1300., 2900.])[channels] + (x * y * 7 + x * x * 3) % 113
            case = encode_case(f"channel-texture-{pattern}-{origin}", samples, pattern=pattern, origin=origin)
            requests.append(case)
            expected.append(reference(samples, case["descriptor"], case["masks"]))
            for kind in DTYPES:
                samples = ((x * 17 + y * 11) % (251 if kind == "u8" else 601)).astype(float)
                if kind == "u16":
                    samples *= 107
                elif kind == "f32":
                    samples = samples - 100 + .25
                for endian in ["little", "big"]:
                    for padding in [0, 5]:
                        identifier = f"numeric-{pattern}-{origin}-{kind}-{endian}-pad{padding}"
                        case = encode_case(identifier, samples, pattern=pattern, origin=origin, kind=kind,
                                           endian=endian, padding=padding)
                        requests.append(case)
                        expected.append(reference(samples, case["descriptor"], case["masks"]))
    for kind in DTYPES:
        samples = (x + y * 3).astype(float) - (10.25 if kind == "f32" else 0)
        case = encode_case(f"mono-{kind}", samples, pattern=None, kind=kind, padding=3)
        requests.append(case)
        expected.append(reference(samples, case["descriptor"], case["masks"]))
    for name, transform in [
        ("rotation", [[0., -2., 20.], [2., 0., 5.]]),
        ("reflection", [[-1., 0., 30.], [0., 1., 4.]]),
        ("anisotropic", [[1., 0., 2.], [0., 2., 3.]]),
    ]:
        samples = (x + y).astype(float)
        case = encode_case(f"geometry-{name}", samples, pattern=None)
        case["descriptor"]["geometry"]["analysis_to_sensor"] = transform
        requests.append(case)
        expected.append(reference(samples, case["descriptor"], case["masks"]))
    # Threshold boundary and independently marked support, including nonfinite f32 bytes.
    for pattern in PATTERNS:
        samples = np.full((9, 11), 100., dtype=float)
        samples[3, 3], samples[3, 4], samples[3, 5] = 999, 1000, 1001
        masks = {name: [0] * samples.size for name in ["bad_pixel", "invalid", "native_saturation"]}
        masks["bad_pixel"][4 * 11 + 5] = 1
        masks["invalid"][5 * 11 + 7] = 1
        masks["native_saturation"][6 * 11 + 6] = 1
        case = encode_case(f"masks-{pattern}", samples, pattern=pattern, masks=masks, saturation=1000)
        requests.append(case)
        expected.append(reference(samples, case["descriptor"], case["masks"]))
    samples = np.full((9, 11), -2.25)
    samples[4, 5] = np.nan
    samples[5, 6] = np.inf
    case = encode_case("nonfinite-residual", samples, kind="f32")
    requests.append(case)
    expected.append(reference(samples, case["descriptor"], case["masks"]))
    # Invalid descriptors should produce an explicit unavailable result, not analysis.
    base = encode_case("invalid-base", np.full((9, 11), 100.))
    for name, mutate in [
        ("truncated-buffer", lambda c: c["data"].pop()),
        ("short-stride", lambda c: c["descriptor"].update(row_stride_bytes=2)),
        ("unknown-color", lambda c: c["descriptor"].update(color="unknown")),
        ("unknown-version", lambda c: c["descriptor"].update(schema_version=999)),
        ("bad-mask-size", lambda c: c["masks"].update(bad_pixel=[1])),
        ("nonbinary-mask", lambda c: c["masks"].update(bad_pixel=[2] * 99)),
        ("threshold-above-container", lambda c: c["descriptor"].update(saturation={"adu": 65536, "source": "test"})),
        ("singular-transform", lambda c: c["descriptor"]["geometry"].update(analysis_to_sensor=[[0., 0., 0.], [0., 0., 0.]])),
    ]:
        case = json.loads(json.dumps(base))
        case["id"] = name
        mutate(case)
        requests.append(case)
        expected.append(None)
    residual = encode_case("residual-without-lineage", np.full((9, 11), -2.25), kind="f32")
    residual["descriptor"]["calibration_ids"] = []
    requests.append(residual)
    expected.append(None)
    residual = encode_case("residual-with-native-threshold", np.full((9, 11), -2.25), kind="f32", saturation=1000)
    requests.append(residual)
    expected.append(None)
    return requests, expected


def check_numerical(binary: Path) -> dict[str, Any]:
    requests, expected = numerical_corpus()
    outputs = production(binary, requests)
    failures, max_ulp, checked_pixels = [], 0, 0
    for request, output, oracle in zip(requests, outputs, expected):
        identifier = request["id"]
        if oracle is None:
            if output.get("ok") or not output.get("error"):
                failures.append({"id": identifier, "reason": "invalid input did not fail explicitly"})
            continue
        if not output.get("ok"):
            failures.append({"id": identifier, "reason": output.get("error", "unavailable")})
            continue
        shape = (request["descriptor"]["height"], request["descriptor"]["width"])
        valid = np.ones(shape, dtype=bool)
        for name, wanted in oracle["masks"].items():
            actual = np.asarray(output["masks"][name], dtype=bool).reshape(shape)
            if not np.array_equal(actual, wanted):
                failures.append({"id": identifier, "reason": f"{name} propagation differs", "pixels": int(np.sum(actual != wanted))})
            valid &= ~wanted
        for name in ["rgb", "pixels"]:
            wanted = oracle[name]
            actual = np.asarray(output[name], dtype=np.float32).reshape(wanted.shape)
            a, b = actual[valid], wanted[valid]
            if not np.all(np.isfinite(a)):
                failures.append({"id": identifier, "reason": f"nonfinite unmasked {name}"})
                continue
            error = int(np.max(ulp_distance(a, b), initial=0))
            checked_pixels += a.size
            max_ulp = max(max_ulp, error)
            limit = 0 if request["descriptor"]["color"] == "mono" else MAX_ULP
            if error > limit:
                failures.append({"id": identifier, "reason": f"{name} exceeds {limit} ULP", "max_ulp": error})
        descriptor = output["descriptor"]
        if descriptor.get("training_eligible") is not False:
            failures.append({"id": identifier, "reason": "analysis was not explicitly excluded from training"})
        # The complete source descriptor must be represented, preserving the native map.
        source = descriptor.get("sensor", descriptor.get("sensor_descriptor"))
        if source != request["descriptor"]:
            failures.append({"id": identifier, "reason": "sensor descriptor was not preserved exactly"})
        if descriptor.get("geometry") != request["descriptor"]["geometry"]:
            failures.append({"id": identifier, "reason": "analysis geometry was not preserved"})
        if not descriptor.get("processing_version") or not descriptor.get("intensity_definition"):
            failures.append({"id": identifier, "reason": "analysis processing/intensity identity missing"})
        source_sha256 = hashlib.sha256(bytes(request["data"])).hexdigest()
        if descriptor.get("source_sha256") != source_sha256:
            failures.append({"id": identifier, "reason": "source bytes hash differs"})
        wanted_mask_hashes = {name: hashlib.sha256(bytes(values)).hexdigest() for name, values in request["masks"].items()}
        if descriptor.get("native_mask_sha256") != wanted_mask_hashes:
            failures.append({"id": identifier, "reason": "native mask lineage hash differs"})
        if len(descriptor.get("config_sha256", "")) != 64:
            failures.append({"id": identifier, "reason": "config identity is missing"})
        transform = np.asarray(request["descriptor"]["geometry"]["analysis_to_sensor"])
        geometry = output["geometry_checks"]
        for field, point in [("mapped_origin", [0., 0., 1.]), ("mapped_unit_x", [1., 0., 1.]), ("mapped_unit_y", [0., 1., 1.])]:
            if not np.array_equal(geometry[field], transform @ point):
                failures.append({"id": identifier, "reason": f"{field} geometry differs"})
        matrix = transform[:, :2]
        gram = matrix.T @ matrix
        expected_scale = math.sqrt(gram[0, 0]) if np.array_equal(gram, np.eye(2) * gram[0, 0]) else None
        if geometry["isotropic_native_scale"] != expected_scale:
            failures.append({"id": identifier, "reason": "scalar geometry eligibility differs"})
    return {"cases": len(requests), "checked_values": checked_pixels, "max_float32_ulp": max_ulp,
            "ulp_limit": MAX_ULP, "passed": not failures, "failures": failures}


def integrated_gaussian(size: int, center: tuple[float, float], sigma: float) -> np.ndarray:
    coordinate = np.arange(size, dtype=float)
    denominator = sigma * math.sqrt(2)
    def integral(c: float) -> np.ndarray:
        return np.asarray([.5 * (math.erf((v + .5 - c) / denominator) - math.erf((v - .5 - c) / denominator))
                           for v in coordinate])
    return np.outer(integral(center[1]), integral(center[0]))


@lru_cache(maxsize=32)
def aperture_geometry(shape: tuple[int, int], center: tuple[float, float]) -> tuple[Any, ...]:
    """Cache only aperture geometry; image measurements remain independent."""
    radius, subpix = 12., 16
    y, x = np.indices(shape)
    x, y = x - center[0], y - center[1]
    vicinity = (np.abs(x) <= radius + .5) & (np.abs(y) <= radius + .5)
    px, py = x[vicinity], y[vicinity]
    offsets = (np.arange(subpix) + .5) / subpix - .5
    sx, sy = np.meshgrid(offsets, offsets)
    radii = np.hypot(px[:, None] + sx.ravel(), py[:, None] + sy.ravel())
    inside = radii.ravel() <= radius
    distances = radii.ravel()[inside]
    order = np.argsort(distances, kind="stable")
    pixel_indices = np.repeat(np.arange(len(px)), subpix * subpix)[inside][order]
    overlap = np.mean(radii <= radius, axis=1)
    affected = np.hypot(x, y) < radius + math.sqrt(.5)
    return vicinity, px, py, overlap, pixel_indices, distances[order], affected


def aperture_metrics(image: np.ndarray, center: tuple[float, float], mask: np.ndarray | None = None) -> dict[str, Any]:
    """Fixed-radius 12 aperture, 16x16 equal-area pixel-overlap sampling."""
    vicinity, px, py, overlap, indices, distances, affected = aperture_geometry(image.shape, center)
    if mask is not None and np.any(mask & affected):
        return {"status": "masked_aperture", "centroid": None, "flux": None, "hfr": None}
    values = image[vicinity]
    cumulative = np.cumsum(values[indices] / 256)
    flux = float(cumulative[-1])
    if not math.isfinite(flux) or flux <= 0:
        return {"status": "nonpositive_flux", "centroid": None, "flux": flux, "hfr": None}
    # First crossing is declared; noisy negative residuals are kept, not clipped.
    crossing = np.flatnonzero(cumulative >= .5 * flux)
    hfr = float(distances[crossing[0]]) if len(crossing) else None
    pixel_weights = values * overlap
    centroid = [float(center[0] + np.sum(pixel_weights * px) / flux),
                float(center[1] + np.sum(pixel_weights * py) / flux)]
    return {"status": "measured", "centroid": centroid, "flux": flux, "hfr": hfr}


def optical_cases():
    rng = np.random.default_rng(SEED)
    colors = {"neutral": (1, 1, 1), "red": (4, 1, .25), "blue": (.25, 1, 4), "green": (.1, 1, .1)}
    count = 0
    for pattern in PATTERNS:
        for origin in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            for phase_y in [0, .25, .5, .75]:
                for phase_x in [0, .25, .5, .75]:
                    center = (32 + phase_x, 32 + phase_y)
                    for sigma in [.6, 1., 1.5, 2., 3.]:
                        profile = integrated_gaussian(65, center, sigma)
                        channels = channel_map(profile.shape, pattern, origin)
                        for color_name, amplitudes in colors.items():
                            native = 128 + 10000 * np.asarray(amplitudes)[channels] * profile
                            truth = 10000 * np.mean(amplitudes) * profile
                            for stratum in ["noiseless", "noise", "clipped"]:
                                samples = native.copy()
                                if stratum == "noise":
                                    samples = rng.poisson(samples).astype(float) + rng.normal(0, 3, samples.shape)
                                threshold = 1000 if stratum == "clipped" else None
                                samples = np.rint(np.clip(samples, 0, threshold or 65535))
                                identifier = f"optical-{count}"
                                count += 1
                                request = encode_case(identifier, samples, pattern=pattern, origin=origin, saturation=threshold)
                                request["include_rgb"] = False
                                meta = {"id": identifier, "pattern": pattern, "origin": list(origin), "phase": [phase_x, phase_y],
                                        "sigma": sigma, "color": color_name, "stratum": stratum,
                                        "center": center, "truth": truth, "samples": samples}
                                yield request, meta


def characterize(binary: Path) -> dict[str, Any]:
    try:
        import sep
    except ImportError:
        sep = None
    # Keep the CLI transport bounded rather than constructing an enormous JSON payload.
    results = []
    corpus = iter(optical_cases())
    truth_cache = {}
    while batch := list(itertools.islice(corpus, 64)):
        requests, metadata = zip(*batch)
        outputs = production(binary, list(requests))
        for request, output, meta in zip(requests, outputs, metadata):
            result = {k: v for k, v in meta.items() if k not in ["truth", "samples", "center"]}
            if not output.get("ok"):
                result.update(status="analysis_unavailable", reason=output.get("error"))
                results.append(result)
                continue
            image = np.asarray(output["pixels"], dtype=float).reshape((65, 65)) - 128
            mask = np.zeros((65, 65), dtype=bool)
            for native_mask in output["masks"].values():
                mask |= np.asarray(native_mask, dtype=bool).reshape((65, 65))
            measured = aperture_metrics(image, meta["center"], mask)
            truth_key = (meta["center"], meta["sigma"], meta["color"])
            if truth_key not in truth_cache:
                truth_cache[truth_key] = aperture_metrics(meta["truth"], meta["center"])
            truth = truth_cache[truth_key]
            result.update(measurement=measured, truth_measurement=truth,
                          continuous_hfr=meta["sigma"] * math.sqrt(2 * math.log(2)))
            oracle = reference(meta["samples"], request["descriptor"], request["masks"])
            result["independent_reconstruction"] = aperture_metrics(oracle["pixels"].astype(float) - 128, meta["center"], mask)
            if measured["status"] == "measured":
                result["centroid_error_native_pixels"] = math.dist(measured["centroid"], meta["center"])
                result["hfr_minus_pixel_truth"] = measured["hfr"] - truth["hfr"]
                result["hfr_minus_continuous_truth"] = measured["hfr"] - result["continuous_hfr"]
                result["relative_aperture_flux_error"] = measured["flux"] / truth["flux"] - 1
                if sep is not None:
                    radii, flags = sep.flux_radius(np.ascontiguousarray(image, dtype=np.float64),
                        np.array([meta["center"][0]]), np.array([meta["center"][1]]), np.array([12.]), .5,
                        mask=mask, subpix=16)
                    result["sep"] = {"hfr": float(radii[0]), "flag": int(flags[0])}
            results.append(result)
    summaries = []
    for sigma in [.6, 1., 1.5, 2., 3.]:
        for color in ["neutral", "red", "blue", "green"]:
            for stratum in ["noiseless", "noise", "clipped"]:
                group = [r for r in results if r["sigma"] == sigma and r["color"] == color and r["stratum"] == stratum]
                measured = [r for r in group if "hfr_minus_pixel_truth" in r]
                summary = {"sigma": sigma, "color": color, "stratum": stratum, "cases": len(group), "measured_cases": len(measured)}
                for metric in ["centroid_error_native_pixels", "hfr_minus_pixel_truth", "relative_aperture_flux_error"]:
                    values = [r[metric] for r in measured]
                    summary[metric] = None if not values else {"min": min(values), "median": float(np.median(values)), "max": max(values)}
                summaries.append(summary)
    return {"cases": len(results), "optical_pass_fail_threshold": None,
            "sep": {"available": sep is not None, "version": None if sep is None else sep.__version__, "subpix": 16, "rmax": 12},
            "stratified_summary": summaries, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numerical-only", action="store_true")
    args = parser.parse_args()
    report = {"protocol": PROTOCOL, "seed": SEED, "python": platform.python_version(), "numpy": np.__version__,
              "fixture_binary_sha256": hashlib.sha256(args.fixture_binary.read_bytes()).hexdigest(),
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "numerical": check_numerical(args.fixture_binary)}
    if not args.numerical_only and report["numerical"]["passed"]:
        report["optical"] = characterize(args.fixture_binary)
    if hashlib.sha256(args.fixture_binary.read_bytes()).hexdigest() != report["fixture_binary_sha256"]:
        raise RuntimeError("fixture binary changed during validation; run again against an immutable copy")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "numerical": report["numerical"],
                      "optical_cases": report.get("optical", {}).get("cases", 0)}, indent=2))
    return 0 if report["numerical"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
