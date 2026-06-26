"""
build_reference_models — emit shape-correct *reference* ONNX edge models WITHOUT
training or PyTorch (uses onnx + numpy only).

Why: the daemon's model_registry.rs / onnx_runner.rs and the
/api/v1/ai/analyze contract can be wired and tested against valid ONNX files
that have the *correct input/output contracts* long before the real models are
trained. These reference graphs have random weights — they are placeholders for
integration, not for accuracy. Replace them with real exports from
ml/export/onnx_export.py once trained.

Writes, per spec Section 7.6.10:
    models/<name>/<version>/model.onnx
    models/<name>/<version>/metadata.json

Run:  python ml/pipelines/build_reference_models.py
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.export.write_model_metadata import write_metadata  # noqa: E402

MODELS_ROOT = ROOT / "models"
OPSET = 17
RNG = np.random.default_rng(7)


def _init(name: str, shape: tuple[int, ...]) -> onnx.TensorProto:
    """Small random weight initializer (Xavier-ish)."""
    fan_in = shape[0]
    w = (RNG.standard_normal(shape) / np.sqrt(fan_in)).astype(np.float32)
    return numpy_helper.from_array(w, name=name)


def _linear(prefix, x, in_dim, out_dim, inits, nodes, bias=True):
    """y = x @ W (+ b). Appends initializers/nodes; returns output name."""
    w = f"{prefix}_W"
    inits.append(_init(w, (in_dim, out_dim)))
    mm = f"{prefix}_mm"
    nodes.append(helper.make_node("MatMul", [x, w], [mm]))
    if not bias:
        return mm
    b = f"{prefix}_b"
    inits.append(numpy_helper.from_array(np.zeros((out_dim,), np.float32), name=b))
    out = f"{prefix}_out"
    nodes.append(helper.make_node("Add", [mm, b], [out]))
    return out


def build_focus_quality(img=64):
    """Grayscale star crop -> focus quality score in [0, 1]."""
    nodes, inits = [], []
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 1, img, img])
    score = helper.make_tensor_value_info("score", TensorProto.FLOAT, ["batch", 1])
    nodes.append(helper.make_node("Flatten", ["image"], ["flat"], axis=1))
    logits = _linear("fq", "flat", img * img, 1, inits, nodes)
    nodes.append(helper.make_node("Sigmoid", [logits], ["score"]))
    graph = helper.make_graph(nodes, "focus_quality", [image], [score], inits)
    return _finish(graph), {
        "inputs": {"image": ["batch", 1, img, img]},
        "outputs": {"score": ["batch", 1]},
        "notes": "Higher score = better focus. Reference weights are random.",
    }


def _finish(graph) -> onnx.ModelProto:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9  # compatible with onnxruntime 1.16+
    onnx.checker.check_model(model)
    return model


def _verify(path: pathlib.Path, io: dict) -> None:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {}
    for name, shape in io["inputs"].items():
        dims = [2 if (isinstance(d, str) or d is None) else d for d in shape]  # batch=2
        feed[name] = RNG.standard_normal(dims).astype(np.float32)
    outs = sess.run(None, feed)
    shapes = {o.name: list(s.shape) for o, s in zip(sess.get_outputs(), outs)}
    print(f"    onnxruntime OK -> {shapes}")


SPECS = [
    ("focus_quality", "1.0.0", "focus_quality", build_focus_quality),
]


def main():
    MODELS_ROOT.mkdir(exist_ok=True)
    for name, version, task, builder in SPECS:
        out_dir = MODELS_ROOT / name / version
        out_dir.mkdir(parents=True, exist_ok=True)
        model, io = builder()
        model_path = out_dir / "model.onnx"
        onnx.save(model, str(model_path))
        print(f"[{name} {version}] wrote {model_path.relative_to(ROOT)}")
        _verify(model_path, io)
        # Preserve an existing curated metadata.json; otherwise write a fresh one.
        meta_path = out_dir / "metadata.json"
        if meta_path.exists():
            print(f"    keeping existing {meta_path.relative_to(ROOT)} (curated)")
        else:
            write_metadata(
                str(model_path), name, version, task,
                framework="onnx", input_format=io["inputs"], output_format=io["outputs"],
                training_info={"status": "reference", "trained": False, "weights": "random"},
            )
    print("\nReference edge models ready under models/. Replace with trained exports when available.")


if __name__ == "__main__":
    main()
