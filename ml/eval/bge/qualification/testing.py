"""Synthetic fixtures for the qualification tests and self-checks (never real data).

Frames are generated in a tmp dir: Gaussian sky noise on a known background
with Gaussian stars, written as FITS (or as daemon-like analysis bundles), so
the oracle method has exact truth.
"""

from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from ml.eval.bge import scenarios as SC
from ml.eval.bge.qualification import DAEMON_ANALYSIS_REPRESENTATION, REPO_ROOT

REP = DAEMON_ANALYSIS_REPRESENTATION


def write_fits(path, data, dtype=np.float32, header=None):
    from astropy.io import fits

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=dtype))
    for k, v in (header or {}).items():
        hdu.header[k] = v
    hdu.writeto(path, overwrite=True)
    return path


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sky_frame(seed, shape=(192, 256), sky=1000.0, sigma=10.0, n_stars=12, psf_sigma=1.6, gradient=(0.05, 0.03),
              flux=(5e3, 1e5), min_sep=30.0):
    """(image, B, R, sources) with Gaussian stars on a linear-gradient sky."""
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    b = sky + gradient[0] * (xx - w / 2) + gradient[1] * (yy - h / 2)
    r = np.full(shape, sigma)
    stars = []
    if n_stars:
        xy = SC.place_uniform(rng, n_stars, w, h, 16, min_sep)
        fl = SC.log_uniform(rng, flux[0], flux[1], n_stars)
        psf = {"model": "gaussian", "sigma_px": psf_sigma}
        stars = [(x, y, float(f), psf) for (x, y), f in zip(xy, fl)]
    src, sources = SC.add_stars(shape, stars)
    image = b + src + rng.standard_normal(shape) * r
    return image.astype(np.float32), b.astype(np.float32), r.astype(np.float32), sources


def dark_frame(seed, shape=(192, 256), offset=500.0, sigma=5.0, hot=((10, 20), (100, 50))):
    rng = np.random.default_rng(seed)
    image = offset + rng.standard_normal(shape) * sigma
    for y, x in hot:
        image[y, x] += 2000.0
    return image.astype(np.float32)


def write_bundle(bundle_dir, image, masks=None, capture=None, processing_version=REP):
    """A daemon-like capture bundle: analysis/linear.fits, masks, analysis.json, manifest.v1.json."""
    bundle_dir = Path(bundle_dir)
    a = bundle_dir / "analysis"
    write_fits(a / "linear.fits", image)
    records = []
    for kind in ("bad_pixel", "invalid", "edge", "native_saturation"):
        m = (masks or {}).get(kind, np.zeros(image.shape, np.uint8))
        p = write_fits(a / f"{kind}.fits", m, dtype=np.uint8)
        records.append({"kind": kind, "role": "analysis_mask", "filename": f"{kind}.fits", "sha256": sha(p),
                        "flagged_value": 1})
    descriptor = {"schema_version": 1,
                  "analysis": {"processing_version": processing_version, "config_sha256": "c" * 64, "units": "adu",
                               "sensor": {"saturation": {"adu": 16383.0, "source": "test"}}},
                  "parent": {"role": "sensor_fits", "filename": "image.fits", "sha256": "0" * 64},
                  "intensity": {"role": "analysis_fits", "filename": "linear.fits", "sha256": sha(a / "linear.fits"),
                                "dtype": "float32"},
                  "masks": records, "calibration_state": "unapplied", "training_label_eligible": False}
    (a / "analysis.json").write_text(json.dumps(descriptor, indent=1))
    if capture is not None:
        (bundle_dir / "manifest.v1.json").write_text(json.dumps(capture, indent=1))
    return bundle_dir


def migrate(db_path):
    """A DB with the daemon's schema (achernar-daemon/migrations, applied in order)."""
    con = sqlite3.connect(db_path)
    for p in sorted(glob.glob(str(REPO_ROOT / "achernar-daemon" / "migrations" / "*.sql"))):
        con.executescript(Path(p).read_text())
    con.commit()
    return con


def iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).isoformat()


def focus_hfr_sigma(p, p0=1000.0, s0=1.3, k=0.012):
    return float(np.hypot(s0, k * (p - p0)))
