"""
onnx_export — export trained PyTorch models to the Achernar edge layout.

Edge model layout (spec Section 7.6.10) consumed by the daemon's
`models/model_registry.rs` + `models/onnx_runner.rs`:

    models/<model_name>/<version>/
        model.onnx          # the exported graph
        metadata.json       # schema written by ml/export/write_model_metadata.py

This module needs PyTorch (`pip install -r ml/requirements.txt`). To produce
shape-correct *reference* ONNX models without training (or without torch), use
ml/pipelines/build_reference_models.py instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

try:
    import torch
except Exception:  # torch is optional at import time
    torch = None  # type: ignore

from .write_model_metadata import write_metadata


def export_to_onnx(
    model,
    sample_inputs: "tuple[torch.Tensor, ...]",
    output_path: str | Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: dict | None = None,
    opset: int = 17,
) -> Path:
    """Export a torch model to ONNX, then validate it with onnx + onnxruntime."""
    if torch is None:
        raise RuntimeError("PyTorch is required for export_to_onnx (pip install -r ml/requirements.txt)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()

    # Default: make the batch dim dynamic for every input and output.
    if dynamic_axes is None:
        dynamic_axes = {n: {0: "batch"} for n in list(input_names) + list(output_names)}

    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(sample_inputs),
            str(output_path),
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )
    _validate(output_path, sample_inputs, input_names)
    return output_path


def export_to_edge(
    model,
    sample_inputs: "tuple[torch.Tensor, ...]",
    models_root: str | Path,
    model_name: str,
    version: str,
    task: str,
    input_format: dict,
    output_format: dict,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: dict | None = None,
    training_info: dict | None = None,
) -> Path:
    """Export into models/<name>/<version>/model.onnx and write metadata.json."""
    out_dir = Path(models_root) / model_name / version
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.onnx"
    export_to_onnx(model, sample_inputs, model_path, input_names, output_names, dynamic_axes)
    write_metadata(
        str(model_path), model_name, version, task,
        framework="onnx", input_format=input_format, output_format=output_format,
        training_info=training_info,
    )
    return model_path


def _validate(model_path: Path, sample_inputs, input_names) -> None:
    """Best-effort structural + runtime validation of an exported model."""
    try:
        import onnx
        onnx.checker.check_model(str(model_path))
    except Exception as e:  # pragma: no cover - advisory
        print(f"  ! onnx.checker skipped/failed: {e}")
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        feed = {}
        for name, t in zip(input_names, sample_inputs):
            arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
            feed[name] = arr.astype(np.float32)
        outs = sess.run(None, feed)
        print(f"  ✓ onnxruntime OK, output shapes: {[o.shape for o in outs]}")
    except Exception as e:  # pragma: no cover - advisory
        print(f"  ! onnxruntime validation skipped/failed: {e}")
