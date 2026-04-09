"""
export_quality_onnx — Section 7.6.1 listing 9.
Export FQE model to ONNX with INT8 QAT quantization.
"""
import argparse


def export(model_path: str, output_path: str, quantize: bool = True):
    raise NotImplementedError("Implement ONNX export with INT8 QAT — Section 7.6.1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()
    export(args.model, args.output, not args.no_quantize)
