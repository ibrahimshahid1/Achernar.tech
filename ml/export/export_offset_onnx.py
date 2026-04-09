"""Export focus offset MLP to ONNX."""
import argparse


def export(model_path: str, output_path: str):
    raise NotImplementedError("Implement offset MLP ONNX export")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export(args.model, args.output)
