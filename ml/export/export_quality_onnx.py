"""
export_quality_onnx — Section 7.6.1, Listing 9.

Export the two-head FQE to ONNX and verify FP32 parity against PyTorch before any
quantization. Per the research note, establish FP32 parity first; QAT/INT8 comes
later and must not be trusted until the FP32 path is bit-for-bit sane.

Usage:
    python -m export.export_quality_onnx --model focus_quality_fqe.pt \
        --output focus_quality_1.0.0.onnx
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def export(model_path: str, output_path: str, crop_size: int = 64, quantize: bool = False) -> dict:
    import torch

    from models.focus_quality_cnn import FocusQualityEstimator

    model = FocusQualityEstimator()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, 1, crop_size, crop_size)
    # Legacy (TorchScript) exporter: produces a graph the ORT dynamic quantizer
    # and the Rust `ort` runtime both consume cleanly. The dynamo exporter (torch
    # 2.13 default) emits AveragePool axes as constants that ORT's version
    # converter rejects during INT8 quantization.
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["image"],
        output_names=["mu_log_hfr", "log_var"],
        dynamic_axes={"image": {0: "batch"}, "mu_log_hfr": {0: "batch"}, "log_var": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    # --- FP32 parity check against onnxruntime ---
    import onnxruntime as ort

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = rng.random((8, 1, crop_size, crop_size)).astype(np.float32)
    with torch.no_grad():
        t_mu, t_lv = model(torch.as_tensor(x))
    o_mu, o_lv = sess.run(None, {"image": x})

    mu_err = float(np.max(np.abs(t_mu.numpy() - o_mu)))
    lv_err = float(np.max(np.abs(t_lv.numpy() - o_lv)))
    parity_ok = mu_err < 1e-4 and lv_err < 1e-4
    print(f"ONNX FP32 parity: max|Δmu|={mu_err:.2e}  max|Δlog_var|={lv_err:.2e}  ok={parity_ok}")

    result = {
        "onnx_path": output_path,
        "mu_max_abs_err": mu_err,
        "log_var_max_abs_err": lv_err,
        "parity_ok": parity_ok,
    }

    # FP32 parity is the promotion gate — a mismatch means the exported graph does
    # not compute what PyTorch does, so fail loudly instead of shipping it.
    if not parity_ok:
        raise RuntimeError(
            f"ONNX FP32 parity failed (max|Δmu|={mu_err:.2e}, max|Δlog_var|={lv_err:.2e}); "
            "refusing to emit an unfaithful model."
        )

    if quantize:
        # INT8 is a nice-to-have; a failure here must not sink the FP32 export,
        # which is the promotion gate (FP32 parity before QAT/INT8).
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            q_path = output_path.replace(".onnx", ".int8.onnx")
            quantize_dynamic(output_path, q_path, weight_type=QuantType.QInt8)
            print(f"✓ Wrote dynamic-INT8 model -> {q_path} (validate accuracy separately before use)")
            result["int8_path"] = q_path
        except Exception as err:  # noqa: BLE001
            print(f"⚠ INT8 quantization skipped ({type(err).__name__}: {err}); FP32 export is valid.")
            result["int8_error"] = str(err)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    print("RESULT:", export(args.model, args.output, args.crop_size, args.quantize))
