"""Calibrate model uncertainty estimates (temperature scaling)."""
import argparse


def calibrate(model_path: str, val_data: str, output_path: str):
    raise NotImplementedError("Implement temperature scaling calibration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    calibrate(args.model, args.val_data, args.output)
