"""Export an immutable research COC bundle, requiring real-image ONNX parity."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from ml.coc import CLASS_NAMES, CONTRACT_VERSION, ONNX_PRECISION
from ml.coc.checkpoint import load_checkpoint
from ml.coc.data import HumanCloudDataset, inspect_human_dataset
from ml.coc.inference import preprocess_image
from ml.data_pipeline.public_common import sha256_file, write_json
from ml.export.stable_cloud_norm import stabilize_normalization


def export(checkpoint, output, parity_image, human_root=None):
    import onnx
    import onnxruntime as ort
    model, state = load_checkpoint(checkpoint)
    training_dir = Path(checkpoint).parent
    run_path, quality_path = training_dir / "run.json", training_dir / "quality.json"
    run = json.loads(run_path.read_text())
    quality = json.loads(quality_path.read_text())
    if sha256_file(run_path) != state["run_sha256"] or quality["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("Checkpoint provenance and quality report are not bound")
    if quality.get("runtime_eligible") is not False or quality.get("final_test_used") is not False:
        raise ValueError("Expected a validation-only research candidate")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh immutable export directory")
    size = state["size"]
    image, original_size = preprocess_image(parity_image, size)
    # Legacy exporter explicitly selected: fixed spatial contract, dynamic batch.
    torch.onnx.export(model, torch.from_numpy(image), str(output / "model.onnx"),
                      input_names=["image"], output_names=["logits"], opset_version=17,
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    normalization = stabilize_normalization(output / "model.onnx",
        expected_count=sum(isinstance(module, torch.nn.GroupNorm) for module in model.modules()))
    onnx.checker.check_model(str(output / "model.onnx"))
    session = ort.InferenceSession(str(output / "model.onnx"), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(17)
    vectors = {"real_image": image, "black": np.zeros_like(image), "white": np.ones_like(image),
               "random_batch2": rng.random((2, 3, size, size), dtype=np.float32)}
    parity = []
    golden = {}
    for name, x in vectors.items():
        with torch.inference_mode():
            reference = model(torch.from_numpy(x)).numpy()
        actual = session.run(["logits"], {"image": x})[0]
        error = float(np.max(np.abs(reference - actual)))
        passed = bool(np.allclose(reference, actual, atol=1e-4, rtol=1e-4))
        parity.append({"case": name, "max_abs_error": error, "passed": passed})
        golden[name + "_image"] = x
        golden[name + "_logits"] = reference
    human_root = human_root or run["config"]["human_root"]
    identity = inspect_human_dataset(human_root)
    if identity["records_sha256"] != run["dataset"]["records_sha256"]:
        raise ValueError("Export validation differs from the frozen training split revision")
    validation = HumanCloudDataset(human_root, "validation", size=size)
    validation_parity = []
    for item in validation:
        x = item["image"][None].numpy()
        with torch.inference_mode():
            reference = model(torch.from_numpy(x)).numpy()
        actual = session.run(["logits"], {"image": x})[0]
        validation_parity.append({"sample_id": item["sample_id"],
            "max_abs_error": float(np.max(np.abs(reference - actual))),
            "passed": bool(np.allclose(reference, actual, atol=1e-4, rtol=1e-4)),
            "argmax_pixel_differences": int((reference.argmax(1) != actual.argmax(1)).sum())})
    write_json(output / "parity.json", {"atol": 1e-4, "rtol": 1e-4, "cases": parity,
        "real_image_sha256": sha256_file(parity_image), "original_size": original_size,
        "runtime": ort.__version__, "provider": "CPUExecutionProvider", "normalization": normalization,
        "validation_records_sha256": identity["records_sha256"], "validation_cases": validation_parity,
        "final_test_used": False})
    if not all(row["passed"] for row in parity + validation_parity):
        raise RuntimeError("ONNX numerical parity failed; no valid manifest will be emitted")
    np.savez_compressed(output / "golden.npz", **golden)
    write_json(output / "quality.json", quality)
    write_json(output / "training_run.json", run)
    supplemental = training_dir / "supplemental_provenance.json"
    if supplemental.exists():
        provenance = json.loads(supplemental.read_text())
        snapshot = training_dir / "training_source_snapshot.zip"
        if (provenance["run_sha256"] != sha256_file(run_path)
                or provenance["source_snapshot_sha256"] != sha256_file(snapshot)):
            raise ValueError("Supplemental training provenance changed")
        shutil.copyfile(supplemental, output / supplemental.name)
        shutil.copyfile(snapshot, output / snapshot.name)
    write_json(output / "preprocess.json", {"contract_version": CONTRACT_VERSION, "input": "rendered RGB uint8",
        "channels": "RGB", "layout": "BCHW", "dtype": "float32", "normalization": "divide by 255",
        "size": [size, size], "resize": "Pillow bilinear, square; categorical ROI nearest",
        "exif_orientation": "must already be applied", "linear_ADU_input": False})
    write_json(output / "postprocess.json", {"class_order": list(CLASS_NAMES), "pixel_scores": "uncalibrated softmax",
        "valid_region": "independent explicit binary camera/content mask; required for summaries",
        "summary": "cloud area within ROI, contamination fraction, abstention; no transparency/five classes",
        "contamination_abstention_threshold": 0.3, "threshold_qualified": False, "control_authority": False})
    files = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
    manifest = {"schema_version": 1, "contract_version": CONTRACT_VERSION,
        "family": "cloud_segmentation", "task": "cloud_segmentation", "variant": state["variant"],
        "created_at": datetime.now(timezone.utc).isoformat(), "class_order": list(CLASS_NAMES), "image_size": size,
        "input": {"image": ["batch", 3, size, size]}, "output": {"logits": ["batch", 3, size, size]},
        "precision": ONNX_PRECISION, "framework": "onnx", "opset": 17, "files_sha256": files,
        "checkpoint_sha256": sha256_file(checkpoint), "runtime_eligible": False, "promotion_status": "research",
        "export_sources_sha256": {str(p): sha256_file(p) for p in
            [Path(__file__), Path("ml/export/stable_cloud_norm.py"), Path("ml/coc/checkpoint.py"),
             Path("ml/coc/inference.py"), Path("ml/coc/predict.py"), Path("ml/coc/__init__.py")]},
        "daemon_integration": "not installed; separate offline research bundle, not daemon manifest v2"}
    write_json(output / "manifest.json", manifest)
    return {"output": str(output), "parity": parity, "runtime_eligible": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--parity-image", required=True, help="Rendered training/validation image; never final-test selection")
    p.add_argument("--human-root", help="Override relocated human adapter for complete validation parity")
    a = p.parse_args()
    print(json.dumps(export(a.checkpoint, a.output, a.parity_image, a.human_root)))


if __name__ == "__main__":
    main()
