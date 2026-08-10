#!/usr/bin/env python3
"""
photon_transfer — measure gain, read noise, black level and saturation (BP-2).

This is Model 6 of the specification (Section 7.6.6, Background / SNR
Estimator). That model is not trained; it needs *constants*. `READNOISE` and
`SENSOR_SATURATION` appear literally in the spec's own listing, and nothing in
the repository measures them.

It is also the fix for D1's other half. `[camera] saturation_adu` is commented
out in achernar.example.toml, and the config file's own warning is correct: a
wrong value is worse than none. This tool is how you get a right one.

Everything here happens indoors, in any weather, with a lens cap and a laptop
screen. It is the only substantial piece of the plan that never needs a clear
night, which is why it goes first.

METHOD

For a pair of flats at identical illumination, the mean carries signal and the
*difference* carries only noise — differencing cancels fixed-pattern
non-uniformity, which would otherwise dominate the variance and make the gain
come out far too small:

    g = [(S1 + S2) - (B1 + B2)] / [var(S1 - S2) - var(B1 - B2)]        e-/ADU

    read_noise = g * std(B1 - B2) / sqrt(2)                             e-

Saturation is where the variance-versus-signal curve turns over, NOT the largest
value in the file — that is the container ceiling, and confusing the two is
exactly the bug this tool exists to prevent.

BAYER CHANNELS

The four CFA channels have different sensitivities, so mixing them inflates the
variance and corrupts the gain. Frames are subsampled by 2x2 phase and each
phase measured separately.

Usage
-----
    # RAW (needs rawpy: pip install rawpy)
    python3 tools/photon_transfer.py \
        --bias bias/ --flats flats/ --camera "Canon EOS 760D" --iso 800 \
        --out docs/device-profiles/

    # Already-converted 16-bit TIFF/PNG of the CFA plane
    python3 tools/photon_transfer.py --bias bias/ --flats flats/ ...

Flats must be in pairs at each illumination level. Pairing is by sorted order,
so shoot them as level1a, level1b, level2a, level2b, ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAW_SUFFIXES = {".cr2", ".cr3", ".nef", ".arw", ".rw2", ".orf", ".raf", ".dng", ".pef"}


# ─── Loading ─────────────────────────────────────────────────────────────────


def load_cfa_plane(path: Path) -> np.ndarray:
    """Return the undemosaiced sensor plane as float64 ADU.

    Refuses to guess: a demosaiced RGB image has had neighbouring pixels mixed
    into every sample, which destroys the pixel-to-pixel independence the whole
    method assumes.
    """
    suffix = path.suffix.lower()

    if suffix in RAW_SUFFIXES:
        try:
            import rawpy
        except ImportError as exc:
            raise RuntimeError(
                f"{path.name} is a RAW file and rawpy is not installed.\n"
                "  pip install rawpy\n"
                "or convert first, preserving the CFA plane and the black level:\n"
                "  dcraw -D -4 -T <file>        # -D = no demosaic, -4 = linear 16-bit"
            ) from exc
        with rawpy.imread(str(path)) as raw:
            return raw.raw_image_visible.astype(np.float64)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to read TIFF/PNG frames") from exc

    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim != 2:
        raise RuntimeError(
            f"{path.name} has {arr.ndim} channels — it looks demosaiced. "
            "Photon transfer needs the raw CFA plane (dcraw -D -4 -T)."
        )
    return arr.astype(np.float64)


def bayer_phase(plane: np.ndarray, phase: tuple[int, int]) -> np.ndarray:
    y0, x0 = phase
    return plane[y0::2, x0::2]


def central_region(arr: np.ndarray, size: int = 512) -> np.ndarray:
    """Centre crop. Vignetting and amplifier glow both live at the edges."""
    h, w = arr.shape
    half = size // 2
    cy, cx = h // 2, w // 2
    return arr[
        max(0, cy - half) : min(h, cy + half),
        max(0, cx - half) : min(w, cx + half),
    ]


# ─── Measurement ─────────────────────────────────────────────────────────────


def pair_statistics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(mean of the two frames, variance of their difference)."""
    mean = float(np.mean(a) + np.mean(b)) / 2.0
    # var(a - b) = var(a) + var(b) = 2 * var(single frame)
    diff_var = float(np.var(a - b, ddof=1))
    return mean, diff_var


def fit_linear_region(
    signal: np.ndarray, variance: np.ndarray, max_deviation: float = 0.10
) -> tuple[float, float, int]:
    """Fit var = signal/g + var_read over the linear part of the curve.

    Returns (gain, read_variance_adu2, n_points_used). Points are added in
    increasing signal order for as long as they stay within `max_deviation` of
    the running fit; the first point that departs marks the onset of
    non-linearity, i.e. the knee.
    """
    order = np.argsort(signal)
    signal, variance = signal[order], variance[order]

    best = None
    for n in range(3, len(signal) + 1):
        s, v = signal[:n], variance[:n]
        slope, intercept = np.polyfit(s, v, 1)
        if slope <= 0:
            break
        predicted = slope * s + intercept
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(predicted - v) / np.maximum(np.abs(v), 1e-9)
        if np.nanmax(rel) > max_deviation:
            break
        best = (1.0 / slope, intercept, n)
    if best is None:
        slope, intercept = np.polyfit(signal[:3], variance[:3], 1)
        if slope <= 0:
            raise RuntimeError(
                "Variance does not increase with signal. Something is applying a "
                "non-linear transform — usually a tone curve, or a demosaiced input."
            )
        best = (1.0 / slope, intercept, 3)
    return best


def saturation_from_knee(signal: np.ndarray, variance: np.ndarray, n_linear: int) -> float:
    """Signal level where the PTC turns over.

    Above full well the variance collapses (every pixel pinned at the same
    value), so the turnover is the honest estimate. If the ladder never reached
    saturation we say so by returning the highest level measured, and the caller
    reports it as a lower bound.
    """
    order = np.argsort(signal)
    signal, variance = signal[order], variance[order]
    if n_linear >= len(signal):
        return float(signal[-1])
    peak = int(np.argmax(variance))
    return float(signal[max(peak, n_linear - 1)])


def measure(
    bias_paths: list[Path],
    flat_paths: list[Path],
    phase: tuple[int, int] = (0, 0),
    crop: int = 512,
) -> dict:
    if len(bias_paths) < 2:
        raise RuntimeError("Need at least 2 bias frames (20 recommended)")
    if len(flat_paths) < 6 or len(flat_paths) % 2:
        raise RuntimeError(
            f"Need an even number of flats, at least 6 (got {len(flat_paths)}). "
            "Shoot 2 frames at each of >=3 illumination levels."
        )

    biases = [central_region(bayer_phase(load_cfa_plane(p), phase), crop) for p in bias_paths]
    black_level = float(np.mean([np.mean(b) for b in biases]))

    # Read noise from the difference of the first bias pair: differencing
    # removes the fixed offset pattern, leaving temporal noise only.
    bias_mean, bias_diff_var = pair_statistics(biases[0], biases[1])
    read_noise_adu = float(np.sqrt(bias_diff_var / 2.0))

    signals, variances, levels = [], [], []
    ordered = sorted(flat_paths)
    for i in range(0, len(ordered), 2):
        a = central_region(bayer_phase(load_cfa_plane(ordered[i]), phase), crop)
        b = central_region(bayer_phase(load_cfa_plane(ordered[i + 1]), phase), crop)
        mean, diff_var = pair_statistics(a, b)
        signals.append(mean - black_level)
        variances.append((diff_var - bias_diff_var) / 2.0)
        levels.append({"files": [ordered[i].name, ordered[i + 1].name],
                       "mean_adu": mean, "signal_adu": mean - black_level})

    signal = np.asarray(signals)
    variance = np.asarray(variances)
    gain, read_var_adu2, n_linear = fit_linear_region(signal, variance)
    knee_signal = saturation_from_knee(signal, variance, n_linear)
    saturation_adu = knee_signal + black_level
    reached_saturation = n_linear < len(signal)

    return {
        "gain_e_per_adu": gain,
        "read_noise_adu": read_noise_adu,
        "read_noise_e": read_noise_adu * gain,
        "black_level_adu": black_level,
        "saturation_adu": saturation_adu,
        "saturation_is_lower_bound": not reached_saturation,
        "full_well_e": knee_signal * gain,
        "linear_points_used": int(n_linear),
        "levels_measured": len(signal),
        "bayer_phase": list(phase),
        "crop_px": crop,
        "ptc": {"signal_adu": signal.tolist(), "variance_adu2": variance.tolist()},
        "levels": levels,
    }


# ─── Reporting ───────────────────────────────────────────────────────────────


def plot_ptc(result: dict, out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    s = np.asarray(result["ptc"]["signal_adu"])
    v = np.asarray(result["ptc"]["variance_adu2"])
    n = result["linear_points_used"]
    order = np.argsort(s)
    s, v = s[order], v[order]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(s, v, "o", label="measured")
    fit = s[:n] / result["gain_e_per_adu"]
    ax.plot(s[:n], fit, "-", label=f"linear fit, g = {result['gain_e_per_adu']:.3f} e-/ADU")
    ax.axvline(result["saturation_adu"] - result["black_level_adu"], ls="--",
               color="crimson", label="saturation")
    ax.set_xlabel("signal above black (ADU)")
    ax.set_ylabel("variance (ADU$^2$)")
    ax.set_title("Photon transfer curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def report(result: dict, camera: str, iso: float | None) -> str:
    bound = " (LOWER BOUND — ladder never saturated)" if result["saturation_is_lower_bound"] else ""
    return "\n".join([
        f"  camera            {camera}" + (f"  ISO {iso:g}" if iso else ""),
        f"  gain              {result['gain_e_per_adu']:.4f} e-/ADU",
        f"  read noise        {result['read_noise_e']:.2f} e-  ({result['read_noise_adu']:.2f} ADU)",
        f"  black level       {result['black_level_adu']:.1f} ADU",
        f"  saturation        {result['saturation_adu']:.0f} ADU{bound}",
        f"  full well         {result['full_well_e']:.0f} e-",
        f"  linear points     {result['linear_points_used']} of {result['levels_measured']}",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Photon transfer curve (BP-2)")
    parser.add_argument("--bias", required=True, type=Path, help="Directory of bias frames")
    parser.add_argument("--flats", required=True, type=Path, help="Directory of flat pairs")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--iso", type=float)
    parser.add_argument("--out", type=Path, default=Path("docs/device-profiles"))
    parser.add_argument("--phase", default="0,0", help="Bayer phase as y,x (default 0,0)")
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--all-phases", action="store_true",
                        help="Measure all four CFA phases separately")
    args = parser.parse_args()

    def frames_in(d: Path) -> list[Path]:
        return sorted(p for p in d.iterdir() if p.is_file() and not p.name.startswith("."))

    bias_paths, flat_paths = frames_in(args.bias), frames_in(args.flats)
    phases = [(0, 0), (0, 1), (1, 0), (1, 1)] if args.all_phases else [
        tuple(int(v) for v in args.phase.split(","))
    ]

    results = {}
    for phase in phases:
        result = measure(bias_paths, flat_paths, phase=phase, crop=args.crop)
        results[f"{phase[0]},{phase[1]}"] = result
        print(f"\nBayer phase {phase}:")
        print(report(result, args.camera, args.iso))

    args.out.mkdir(parents=True, exist_ok=True)
    slug = args.camera.lower().replace(" ", "-")
    stem = f"{slug}-iso{args.iso:g}" if args.iso else slug
    profile = {
        "camera": args.camera,
        "iso": args.iso,
        "method": "photon transfer, paired flats, difference variance",
        "tool": "tools/photon_transfer.py",
        "bias_frames": len(bias_paths),
        "flat_frames": len(flat_paths),
        "phases": results,
    }
    profile_path = args.out / f"{stem}.json"
    profile_path.write_text(json.dumps(profile, indent=2))

    plotted = plot_ptc(next(iter(results.values())), args.out / f"{stem}-ptc.png")

    primary = next(iter(results.values()))
    print(f"\n✓ wrote {profile_path}")
    if plotted:
        print(f"✓ wrote {args.out / f'{stem}-ptc.png'}")
    print("\nPut this in achernar.toml — and nowhere else, per the config's own warning")
    print("that a wrong value is worse than none:\n")
    print("[camera]")
    print(f"saturation_adu = {primary['saturation_adu']:.0f}")
    if primary["saturation_is_lower_bound"]:
        print(
            "\n! Your flat ladder never reached saturation, so that number is a "
            "lower bound.\n  Re-shoot with brighter levels before trusting it."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
