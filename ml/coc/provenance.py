"""Preserve executable training/transform sources alongside run fingerprints."""
from importlib.metadata import version
from pathlib import Path
import zipfile

from ml.data_pipeline.public_common import sha256_file, write_json


def capture_sources(output, sources, repo_root):
    output, repo_root = Path(output), Path(repo_root).resolve()
    with zipfile.ZipFile(output / "training_source_snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            path = Path(source).resolve()
            archive.write(path, str(path.relative_to(repo_root)))
    write_json(output / "supplemental_provenance.json", {
        "run_sha256": sha256_file(output / "run.json"), "captured_before_optimization": True,
        "source_snapshot_sha256": sha256_file(output / "training_source_snapshot.zip"),
        "versions": {p: version(p) for p in ["torch", "numpy", "pillow", "onnx", "onnxruntime"]},
        "transform_sources_sha256": {str(Path(p).resolve().relative_to(repo_root)): sha256_file(p) for p in sources},
    })
