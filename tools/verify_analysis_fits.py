#!/usr/bin/env python3
"""Independently read an exported production RAW crop with Astropy/NumPy.

Input: ACHERNAR_RAW_FIXTURE_OUTPUT from the ignored Rust FITS integration test.
Requires Astropy and NumPy. No network or hardware access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import astropy
from astropy.io import fits
import numpy as np

from validate_cfa_analysis import reference, ulp_distance


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdus:
        hdus.verify("exception")
        assert len(hdus) == 1
        return hdus[0].data.copy(), hdus[0].header.copy()


def verify(directory: Path) -> dict:
    bundle = json.loads((directory / "analysis/analysis.json").read_text())
    descriptor = bundle["analysis"]
    sensor_descriptor = descriptor["sensor"]
    info = json.loads((directory / "raw_info.json").read_text())
    metadata = info["metadata"]
    height, width = sensor_descriptor["height"], sensor_descriptor["width"]
    byte_order = "<" if sensor_descriptor["byte_order"] == "little" else ">"
    native = np.fromfile(directory / "sensor.u16", dtype=byte_order + "u2").reshape(height, width)
    sensor, header = read(directory / "derived/image.fits")
    np.testing.assert_array_equal(sensor, native)
    assert header["BITPIX"] == 16 and header["BZERO"] == 32768 and header["BSCALE"] == 1
    assert header["RAWVER"] == metadata["decoder_version"]
    assert header["RAWX0"] == metadata["crop_x"] and header["RAWY0"] == metadata["crop_y"]
    assert header["RAWW"] == metadata["raw_width"] and header["RAWH"] == metadata["raw_height"]
    assert header["BLKTABN"] == len(metadata["black_table"])
    assert [header["BLKROWS"], header["BLKCOLS"]] == info["cblack_pattern"]
    for index, value in enumerate(metadata["black_table"]):
        assert header[f"BLKC{index:04}"] == value
    assert header["NONLIN"] == "UNMEAS"
    if info["white_level"] > 0:
        assert header["SATURATE"] == info["white_level"]
    assert "EXPTIME" not in header and "DATE-OBS" not in header
    assert descriptor["source_sha256"] == digest(directory / "sensor.u16")
    assert bundle["source_samples_sha256"] == descriptor["source_sha256"]
    assert bundle["parent"]["sha256"] == digest(directory / "derived/image.fits")
    assert bundle["sensor_decode"]["metadata"] == metadata
    assert not descriptor["training_eligible"] and not bundle["training_label_eligible"]
    np.testing.assert_array_equal(descriptor["geometry"]["analysis_to_sensor"],
                                  [[1, 0, metadata["crop_x"]], [0, 1, metadata["crop_y"]]])
    expected = reference(native, sensor_descriptor, {})
    masks = {}
    for record in bundle["masks"]:
        path = directory / "analysis" / record["filename"]
        assert digest(path) == record["sha256"]
        mask, mask_header = read(path)
        assert mask_header["BITPIX"] == 8 and mask.shape == (height, width)
        assert set(np.unique(mask)).issubset({0, 1})
        np.testing.assert_array_equal(mask, expected["masks"][record["kind"]])
        masks[record["kind"]] = mask
    assert set(masks) == {"bad_pixel", "invalid", "edge", "native_saturation"}
    intensity_path = directory / "analysis" / bundle["intensity"]["filename"]
    assert digest(intensity_path) == bundle["intensity"]["sha256"]
    intensity, analysis_header = read(intensity_path)
    assert analysis_header["BITPIX"] == -32 and "BZERO" not in analysis_header
    assert "EXPTIME" not in analysis_header and "DATE-OBS" not in analysis_header
    assert intensity.dtype.kind == "f" and intensity.dtype.itemsize == 4
    valid = ~np.logical_or.reduce([mask.astype(bool) for mask in masks.values()])
    errors = ulp_distance(intensity[valid], expected["pixels"][valid])
    maximum_ulp = int(errors.max()) if len(errors) else 0
    assert maximum_ulp <= 4
    negative, negative_header = read(directory / "negative.fits")
    assert negative_header["BITPIX"] == -32 and "BZERO" not in negative_header
    np.testing.assert_array_equal(negative.ravel()[:3], np.asarray([-4.25, .125, 65535], dtype=np.float32))
    assert np.isnan(negative.ravel()[3])
    return {"passed": True, "astropy": astropy.__version__, "numpy": np.__version__,
            "shape": [height, width], "native_values_checked": native.size,
            "analysis_values_checked": int(valid.sum()), "maximum_float32_ulp": maximum_ulp,
            "decoder_version": metadata["decoder_version"], "make": info["make"], "model": info["model"],
            "sensor_fits_sha256": digest(directory / "derived/image.fits"),
            "descriptor_sha256": digest(directory / "analysis/analysis.json"),
            "qualification": "file_and_computational_contract_only; optical_and_device_profiles_unqualified"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
