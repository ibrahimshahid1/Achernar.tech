#!/usr/bin/env python3
"""
Download AstroM3 pretrained weights from the HuggingFace Hub.

    python ml/astrom3/download.py                 # core set (CLIP base + 4 classifiers)
    python ml/astrom3/download.py --set all       # core + the 5 seed variants
    python ml/astrom3/download.py --models AstroM3-CLIP AstroM3-CLIP-all
    python ml/astrom3/download.py --out /custom/path

Weights land in ml/checkpoints/astrom3/<RepoName>/ (git-ignored). Each repo is
~66 MB (16.5M F32 params); the per-modality classifiers are smaller.

Requires: pip install huggingface_hub
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "ml" / "checkpoints" / "astrom3"
ORG = "AstroMLCore"

CORE = ["AstroM3-CLIP", "AstroM3-CLIP-all", "AstroM3-CLIP-photo", "AstroM3-CLIP-spectra", "AstroM3-CLIP-meta"]
SEEDS = ["AstroM3-CLIP-42", "AstroM3-CLIP-0", "AstroM3-CLIP-66", "AstroM3-CLIP-12", "AstroM3-CLIP-123"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Download AstroM3 weights from HuggingFace.")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    ap.add_argument("--set", choices=["core", "seeds", "all"], default="core")
    ap.add_argument("--models", nargs="*", help="explicit repo names under AstroMLCore/ (overrides --set)")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("error: huggingface_hub not installed.  pip install huggingface_hub", file=sys.stderr)
        return 2

    if args.models:
        names = args.models
    elif args.set == "core":
        names = CORE
    elif args.set == "seeds":
        names = SEEDS
    else:
        names = CORE + SEEDS

    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(names)} model(s) -> {out_root}")

    failures = []
    for name in names:
        repo = f"{ORG}/{name}"
        dest = out_root / name
        print(f"  ↓ {repo}")
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=str(dest),
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.md"],
            )
            print(f"    ✓ {dest.relative_to(ROOT)}")
        except Exception as e:  # noqa: BLE001
            print(f"    ✗ {repo}: {type(e).__name__}: {e}")
            failures.append(repo)

    if failures:
        print(f"\n{len(failures)} download(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nAll downloads complete. Load with ml/astrom3/loader.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
