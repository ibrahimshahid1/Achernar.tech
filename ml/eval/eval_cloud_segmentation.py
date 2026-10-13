"""Evaluate a frozen COC checkpoint. Final-test evaluation needs an explicit flag."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from ml.coc.checkpoint import load_checkpoint
from ml.coc.data import HumanCloudDataset, inspect_human_dataset
from ml.data_pipeline.public_common import sha256_file, write_json
from ml.train.train_cloud_segmentation import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--human-root", default="ml/data/public/lenghu_sky8_human")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--allow-final-test", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args()
    if a.split == "test" and not a.allow_final_test:
        p.error("The final test is reserved; pass --allow-final-test only for a frozen final evaluation")
    output = Path(a.output)
    if output.exists():
        p.error("Use a fresh evaluation output path")
    model, state = load_checkpoint(a.checkpoint)
    run_path = Path(a.checkpoint).parent / "run.json"
    if sha256_file(run_path) != state["run_sha256"]:
        raise ValueError("Checkpoint training provenance changed")
    run = json.loads(run_path.read_text())
    identity = inspect_human_dataset(a.human_root)
    if identity["records_sha256"] != run["dataset"]["records_sha256"]:
        raise ValueError("Frozen split revision differs from the training run")
    data = HumanCloudDataset(a.human_root, a.split, size=state["size"])
    metrics = evaluate(model, DataLoader(data, batch_size=a.batch_size), "cpu")
    report = {"checkpoint_sha256": sha256_file(a.checkpoint), "dataset": identity, "split": a.split,
              "metrics": metrics, "final_test_used": a.split == "test", "runtime_eligible": False}
    write_json(output, report)
    print(json.dumps({"output": str(output), "split": a.split, "overall": metrics["overall"]}))


if __name__ == "__main__":
    main()
