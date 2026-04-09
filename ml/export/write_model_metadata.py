"""
write_model_metadata — Section 7.6.10.
Writes metadata.json for each exported model with the full schema.
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime


def write_metadata(
    model_path: str,
    model_name: str,
    version: str,
    task: str,
    framework: str = "onnx",
    input_format: dict | None = None,
    output_format: dict | None = None,
    training_info: dict | None = None,
) -> dict:
    """Write metadata.json alongside a model file."""
    model_file = Path(model_path)
    sha256 = hashlib.sha256(model_file.read_bytes()).hexdigest() if model_file.exists() else "PLACEHOLDER"

    metadata = {
        "model_name": model_name,
        "version": version,
        "task": task,
        "framework": framework,
        "input_format": input_format or {},
        "output_format": output_format or {},
        "sha256": sha256,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "training_info": training_info,
    }

    out_path = model_file.parent / "metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2))
    print(f"✓ Wrote {out_path}")
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    write_metadata(args.model, args.name, args.version, args.task)
