"""Run a checkpoint or verified ONNX bundle on a rendered RGB sky image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ml.coc import CLASS_NAMES, CONTRACT_VERSION, ONNX_PRECISION
from ml.coc.checkpoint import load_checkpoint
from ml.coc.inference import load_region, preprocess_image, summarize_logits
from ml.data_pipeline.public_common import sha256_file, write_json


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("contract_version") != CONTRACT_VERSION or manifest.get("class_order") != list(CLASS_NAMES):
        raise ValueError("Unsupported ONNX bundle contract")
    if manifest.get("runtime_eligible") is not False or manifest.get("promotion_status") != "research":
        raise ValueError("This CLI only accepts research bundles")
    size = manifest.get("image_size")
    if type(size) is not int or size < 32 or size % 16:
        raise ValueError("Invalid bundle image size")
    if (manifest.get("task") != "cloud_segmentation" or manifest.get("family") != "cloud_segmentation"
            or manifest.get("framework") != "onnx" or manifest.get("precision") != ONNX_PRECISION
            or manifest.get("input") != {"image": ["batch", 3, size, size]}
            or manifest.get("output") != {"logits": ["batch", 3, size, size]}):
        raise ValueError("Unsupported bundle tensor/task contract")
    files = manifest.get("files_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("Missing bundle integrity map")
    for member, digest in files.items():
        if not isinstance(member, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Malformed bundle integrity map")
        path = (root / member).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != digest:
            raise ValueError("Bundle member integrity failed")
    if "model.onnx" not in manifest["files_sha256"]:
        raise ValueError("Missing hash-bound ONNX graph")
    return manifest


def predict(image_path, output, checkpoint=None, bundle=None, roi=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh prediction output folder")
    if bool(checkpoint) == bool(bundle):
        raise ValueError("Choose exactly one checkpoint or bundle")
    if bundle:
        import onnxruntime as ort
        manifest = verify_bundle(bundle)
        size = manifest["image_size"]
        session = ort.InferenceSession(str(Path(bundle) / "model.onnx"), providers=["CPUExecutionProvider"])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "image" or inputs[0].type != "tensor(float)"
                or inputs[0].shape != ["batch", 3, size, size]
                or len(outputs) != 1 or outputs[0].name != "logits" or outputs[0].type != "tensor(float)"):
            raise ValueError("ONNX graph does not match the bundle input/output contract")
        forward = lambda x: session.run(["logits"], {"image": x})[0]
        artifact = {"onnx_sha256": sha256_file(Path(bundle) / "model.onnx"), "backend": "onnxruntime_cpu"}
    else:
        model, state = load_checkpoint(checkpoint)
        size = state["size"]
        def forward(x):
            with torch.inference_mode():
                return model(torch.from_numpy(x)).numpy()
        artifact = {"checkpoint_sha256": sha256_file(checkpoint), "backend": "pytorch_cpu"}
    image, original_size = preprocess_image(image_path, size)
    region = load_region(roi, size, original_size) if roi else None
    logits = forward(image)
    if logits.shape != (1, 3, size, size) or logits.dtype != np.float32:
        raise ValueError("Model output does not match the declared FP32 segmentation contract")
    summary, classes, scores = summarize_logits(logits[0], region)
    if region is not None and region.any() and np.all(np.ptp(image[0][:, region], axis=1) <= 1e-6):
        summary.update(status="abstained", reason="uninformative_rgb_input",
                       cloud_area_fraction_of_roi=None, cloud_fraction_of_classified_sky=None,
                       contamination_area_fraction_of_roi=None, mean_soft_cloud_share_of_sky=None)
    summary.update(artifact=artifact, input_sha256=sha256_file(image_path), original_size=original_size,
                   model_image_size=size, roi_sha256=sha256_file(roi) if roi else None,
                   limitation="Source-domain segmentation diagnostic; local R50 transfer, whole-frame decision thresholds and calibration unvalidated.")
    Image.fromarray(classes).save(output / "classes.png")
    np.savez_compressed(output / "scores.npz", scores=scores)
    colors = np.array([[50, 120, 220], [240, 240, 240], [230, 110, 40]], dtype=np.uint8)
    rgb = np.rint(image[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    overlay = (0.55 * rgb + 0.45 * colors[classes]).astype(np.uint8)
    Image.fromarray(overlay).save(output / "overlay.png")
    write_json(output / "prediction.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--bundle")
    p.add_argument("--image", required=True)
    p.add_argument("--roi", help="Independent binary camera/content region in original image coordinates")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(predict(a.image, a.output, a.checkpoint, a.bundle, a.roi)))


if __name__ == "__main__":
    main()
