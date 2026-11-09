"""Build the Rust-vs-Python SEP parity fixture (parity_v1).

Run with the reference environment (sep==1.4.1, numpy 2.5.4, pinned in
ml/requirements-public-data.lock.txt):

    <venv>/bin/python achernar-daemon/sep/tests/fixtures/make_parity_fixture.py

It deterministically synthesises one float32 frame, runs the PyPI sep 1.4.1
wheel on it exactly as the daemon adapter runs the vendored C library, and
writes inputs and reference outputs to parity_v1/. `--check` regenerates in
memory and compares with the committed files byte for byte.

Frame: 192 x 160 px; sky = linear + quadratic gradient (ADU-like units);
Gaussian read-like noise sigma 8 from numpy default_rng(SEED); ten sources
(Gaussian and Moffat), including a close double, one cut by the left border,
one touching the masked rectangle's edge and a bright one inside it; a masked
rectangle that leaves one 32x32 mesh below SEP's 50 % support, with NaN pixels
inside the mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import sep

SEED = 20261108
WIDTH, HEIGHT = 192, 160
SKY_SIGMA = 8.0
BACKGROUND = {"bw": 32, "bh": 32, "fw": 3, "fh": 3, "fthresh": 0.0, "maskthresh": 0.0}
EXTRACT = {
    "thresh": 5.0,
    "err": "background_rms_map",
    "minarea": 5,
    "filter_kernel": [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
    "filter_type": "conv",
    "deblend_nthresh": 32,
    "deblend_cont": 0.005,
    "clean": True,
    "clean_param": 1.0,
    "segmentation_map": True,
    "residual": "data - back(), masked pixels set to 0.0",
}
# (x, y, profile, width parameter(s), peak above sky)
SOURCES = [
    (30.3, 25.7, "gauss", (1.8,), 900.0),
    (150.2, 40.4, "moffat", (2.5, 3.0), 1200.0),
    (70.6, 110.1, "gauss", (2.2,), 600.0),
    (120.0, 130.5, "moffat", (2.0, 2.5), 800.0),
    (100.4, 60.2, "gauss", (1.6,), 1500.0),  # close double, primary
    (105.6, 61.0, "gauss", (1.6,), 1000.0),  # close double, secondary
    (1.2, 90.3, "gauss", (1.8,), 1000.0),  # cut by the left border
    (175.5, 140.8, "gauss", (1.5,), 400.0),
    (55.0, 150.0, "moffat", (1.8, 3.0), 500.0),
    (127.6, 75.3, "gauss", (1.7,), 700.0),  # touches the mask's left edge
    (150.0, 75.0, "gauss", (2.0,), 20000.0),  # inside the mask (clipped core)
]
MASK_RECT = (130, 170, 60, 90)  # x0, x1, y0, y1 (half-open)
NAN_PIXELS = [(140, 70), (145, 80), (160, 85)]  # (x, y), all inside the mask
CATALOG_FIELDS = [
    "thresh", "npix", "tnpix", "xmin", "xmax", "ymin", "ymax", "x", "y",
    "x2", "y2", "xy", "errx2", "erry2", "errxy", "a", "b", "theta",
    "cxx", "cyy", "cxy", "cflux", "flux", "cpeak", "peak",
    "xcpeak", "ycpeak", "xpeak", "ypeak", "flag",
]
INT_FIELDS = {"npix", "tnpix", "xmin", "xmax", "ymin", "ymax", "xcpeak", "ycpeak", "xpeak", "ypeak", "flag"}


RUST_TARGET_OS = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}


def c_library() -> str:
    if platform.system() == "Darwin":
        return f"Apple libSystem (macOS {platform.mac_ver()[0]})"
    name, version = platform.libc_ver()
    return f"{name} {version}".strip() or platform.system()


def synthesise() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    sky = 1000.0 + 0.8 * x - 0.5 * y + 0.004 * (x - 96.0) ** 2 + 0.003 * (y - 80.0) ** 2
    rng = np.random.default_rng(SEED)
    image = sky + rng.normal(0.0, SKY_SIGMA, size=sky.shape)
    for x0, y0, profile, shape, peak in SOURCES:
        r2 = (x - x0) ** 2 + (y - y0) ** 2
        if profile == "gauss":
            (sigma,) = shape
            image += peak * np.exp(-r2 / (2.0 * sigma * sigma))
        else:
            alpha, beta = shape
            image += peak * (1.0 + r2 / (alpha * alpha)) ** (-beta)
    data = image.astype("<f4")
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    x0, x1, y0, y1 = MASK_RECT
    mask[y0:y1, x0:x1] = 1
    for px, py in NAN_PIXELS:
        assert mask[py, px] == 1
        data[py, px] = np.nan
    return np.ascontiguousarray(data), mask


def reference(data: np.ndarray, mask: np.ndarray) -> dict:
    bkg = sep.Background(
        data,
        mask=mask,
        maskthresh=BACKGROUND["maskthresh"],
        bw=BACKGROUND["bw"],
        bh=BACKGROUND["bh"],
        fw=BACKGROUND["fw"],
        fh=BACKGROUND["fh"],
        fthresh=BACKGROUND["fthresh"],
    )
    back = bkg.back()
    rms = bkg.rms()
    assert back.dtype == np.float32 and rms.dtype == np.float32
    usable = (mask == 0) & np.isfinite(data)
    # The Rust safe API rejects non-positive unmasked noise; the fixture
    # must be a valid input for both sides.
    assert np.all(rms[usable] > 0.0), "fixture RMS map not positive on usable pixels"
    resid = data - back
    resid[mask != 0] = 0.0
    objects, segmap = sep.extract(
        resid,
        EXTRACT["thresh"],
        err=rms,
        mask=mask,
        minarea=EXTRACT["minarea"],
        filter_kernel=np.array(EXTRACT["filter_kernel"], dtype=np.float32),
        filter_type=EXTRACT["filter_type"],
        deblend_nthresh=EXTRACT["deblend_nthresh"],
        deblend_cont=EXTRACT["deblend_cont"],
        clean=EXTRACT["clean"],
        clean_param=EXTRACT["clean_param"],
        segmentation_map=EXTRACT["segmentation_map"],
    )
    catalog = []
    for row in objects:
        catalog.append(
            {
                name: (int(row[name]) if name in INT_FIELDS else float(row[name]))
                for name in CATALOG_FIELDS
            }
        )
    return {
        "back": back.astype("<f4"),
        "rms": rms.astype("<f4"),
        "segmap": segmap.astype("<i4"),
        "globalback": float(bkg.globalback),
        "globalrms": float(bkg.globalrms),
        "catalog": catalog,
    }


def render(data: np.ndarray, mask: np.ndarray, ref: dict) -> dict[str, bytes]:
    files = {
        "input.f32": data.tobytes(),
        "mask.u8": mask.tobytes(),
        "back.f32": ref["back"].tobytes(),
        "rms.f32": ref["rms"].tobytes(),
        "segmap.i32": ref["segmap"].tobytes(),
    }
    files["catalog.json"] = (json.dumps(ref["catalog"], indent=1) + "\n").encode()
    # Same values, one object per line, for the dependency-free Rust test;
    # repr() round-trips every float64 exactly.
    lines = ["\t".join(CATALOG_FIELDS)]
    for obj in ref["catalog"]:
        lines.append("\t".join(repr(obj[name]) for name in CATALOG_FIELDS))
    files["catalog.tsv"] = ("\n".join(lines) + "\n").encode()
    files["globals.tsv"] = (
        f"globalback\t{ref['globalback']!r}\nglobalrms\t{ref['globalrms']!r}\n"
    ).encode()
    script = Path(__file__).read_bytes()
    meta = {
        "fixture": "parity_v1",
        "generator": "achernar-daemon/sep/tests/fixtures/make_parity_fixture.py",
        "generator_sha256": hashlib.sha256(script).hexdigest(),
        "sep_version": sep.__version__,
        "sep_c_version_string": getattr(sep, "sep_version_string", None),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        # Deblended children depend on the C library's rand_r (deblend.c
        # gatherup); tests compare these with the platform they run on.
        "rust_target_os": RUST_TARGET_OS.get(platform.system(), platform.system().lower()),
        "c_library": c_library(),
        "deblend_rng": "C library rand_r(&randseed), randseed = 1 per sep_extract (deblend.c gatherup)",
        "seed": SEED,
        "width": WIDTH,
        "height": HEIGHT,
        "layout": "row-major, little-endian; x is the column (fast) index",
        "sky": "1000 + 0.8*x - 0.5*y + 0.004*(x-96)^2 + 0.003*(y-80)^2",
        "sky_noise_sigma": SKY_SIGMA,
        "sources": [list(s[:3]) + [list(s[3]), s[4]] for s in SOURCES],
        "mask_rect_x0_x1_y0_y1": list(MASK_RECT),
        "nan_pixels_xy": [list(p) for p in NAN_PIXELS],
        "background": BACKGROUND,
        "extract": EXTRACT,
        "globalback": ref["globalback"],
        "globalrms": ref["globalrms"],
        "objects": len(ref["catalog"]),
        "sha256": {name: hashlib.sha256(blob).hexdigest() for name, blob in sorted(files.items())},
    }
    files["meta.json"] = (json.dumps(meta, indent=1, sort_keys=True) + "\n").encode()
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with committed files")
    args = parser.parse_args()
    if sep.__version__ != "1.4.1":
        print(f"sep {sep.__version__} is not the pinned 1.4.1", file=sys.stderr)
        return 2
    data, mask = synthesise()
    files = render(data, mask, reference(data, mask))
    out = Path(__file__).resolve().parent / "parity_v1"
    if args.check:
        stale = [n for n, blob in files.items() if not (out / n).is_file() or (out / n).read_bytes() != blob]
        print("stale: " + ", ".join(stale) if stale else "parity_v1 is up to date")
        return 1 if stale else 0
    out.mkdir(exist_ok=True)
    for name, blob in files.items():
        (out / name).write_bytes(blob)
    total = sum(len(b) for b in files.values())
    print(f"wrote {len(files)} files, {total} bytes, {len(json.loads(files['catalog.json']))} objects to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
