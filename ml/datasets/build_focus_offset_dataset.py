"""
build_focus_offset_dataset — Section 7.7.5.
Builds training dataset for the Focus Offset MLP model.
"""
import argparse
import numpy as np


def build_dataset(data_dir: str, output_path: str):
    """Extract feature vectors + offset targets from focus run data."""
    raise NotImplementedError("Implement offset dataset building")


def main():
    parser = argparse.ArgumentParser(description="Build offset training dataset")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="focus_offset_dataset.parquet")
    args = parser.parse_args()
    build_dataset(args.data_dir, args.output)


if __name__ == "__main__":
    main()
