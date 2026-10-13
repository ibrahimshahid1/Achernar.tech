"""Strict safe-tensor-container loading for COC research checkpoints."""
from pathlib import Path

import torch

from ml.coc import CLASS_NAMES, CONTRACT_VERSION
from ml.models.cloud_classifier import CloudSegmenter


def load_checkpoint(path):
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if state.get("contract_version") != CONTRACT_VERSION or state.get("class_order") != list(CLASS_NAMES):
        raise ValueError("Unsupported COC checkpoint task or class order")
    if not isinstance(state.get("size"), int) or state["size"] < 32 or state["size"] % 16:
        raise ValueError("Invalid checkpoint input size")
    model = CloudSegmenter(variant=state["variant"])
    model.load_state_dict(state["state_dict"], strict=True)
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise ValueError("Nonfinite checkpoint parameters")
    return model.eval(), state
