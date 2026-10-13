"""Partial human supervision and explicitly separate COC teacher distillation.

Ignored pixels are not clear-sky examples. Teacher confidence is a selection
score, not calibrated probability or independent human ground truth.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .cloud_classifier import IGNORE_INDEX


def _validate_logits(logits: torch.Tensor, name: str) -> None:
    if logits.ndim != 4 or logits.shape[1] != 3 or min(logits.shape) < 1:
        raise ValueError(f"{name} must have nonempty shape [B, 3, H, W]")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError(f"{name} must contain finite floating point logits")


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean CE over human-labeled pixels, with a safe all-ignored zero.

    With class weights, the denominator is the sum of the selected target
    weights, matching PyTorch weighted cross entropy. Weights must be positive;
    zero weights would silently turn a labeled class into ignored supervision.
    """
    _validate_logits(logits, "logits")
    if targets.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
        raise ValueError("targets must have shape [B, H, W] matching logits")
    if targets.dtype != torch.long:
        raise ValueError("targets must be int64 class labels")
    if targets.device != logits.device:
        raise ValueError("targets and logits must be on the same device")
    supported = ((targets >= 0) & (targets < 3)) | (targets == IGNORE_INDEX)
    if not bool(supported.all()):
        raise ValueError("targets may contain only sky=0, cloud=1, contamination=2, ignore=255")
    if class_weights is not None:
        if class_weights.shape != (3,) or class_weights.device != logits.device:
            raise ValueError("class_weights must be a three-value tensor on the logits device")
        if not class_weights.is_floating_point() or not bool(torch.isfinite(class_weights).all()):
            raise ValueError("class_weights must contain finite floating point values")
        if bool((class_weights <= 0).any()):
            raise ValueError("class_weights must be strictly positive")
        class_weights = class_weights.to(dtype=logits.dtype)

    if not bool((targets != IGNORE_INDEX).any()):
        # An empty mean would be NaN. Keep the graph connected so backward()
        # works while every student logit receives exactly zero gradient.
        return logits.reshape(-1)[0] * 0.0
    return F.cross_entropy(logits, targets, weight=class_weights, ignore_index=IGNORE_INDEX)


def masked_teacher_kl(
    student: torch.Tensor,
    teacher: torch.Tensor,
    content_mask: torch.Tensor,
    temperature: float = 2.0,
    confidence_threshold: float = 0.8,
) -> tuple[torch.Tensor, int]:
    """Distill only selected producer-content pixels; return loss and pixel count.

    Student logits are bilinearly resized to the teacher grid before softmax.
    The confidence gate uses the *untempered*, detached teacher softmax. KL uses
    temperature-softened distributions and the conventional temperature² factor.
    Teacher tensors never receive gradients. No teacher examples are evaluated
    as human ground truth by this objective.
    """
    _validate_logits(student, "student")
    _validate_logits(teacher, "teacher")
    if student.shape[0] != teacher.shape[0] or student.device != teacher.device:
        raise ValueError("student and teacher must share batch size and device")
    if content_mask.shape != (teacher.shape[0], teacher.shape[2], teacher.shape[3]):
        raise ValueError("content_mask must have shape [B, teacher_H, teacher_W]")
    if content_mask.dtype != torch.bool or content_mask.device != teacher.device:
        raise ValueError("content_mask must be boolean and on the teacher device")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be finite and in [0, 1]")

    reference = teacher.detach().to(dtype=student.dtype)
    selected = content_mask & (F.softmax(reference, dim=1).amax(dim=1) >= confidence_threshold)
    selected_count = int(selected.sum().item())
    if selected_count == 0:
        return student.reshape(-1)[0] * 0.0, 0

    resized = F.interpolate(student, size=teacher.shape[-2:], mode="bilinear", align_corners=False)
    pixel_kl = F.kl_div(
        F.log_softmax(resized / temperature, dim=1),
        F.softmax(reference / temperature, dim=1),
        reduction="none",
    ).sum(dim=1)
    return pixel_kl[selected].mean() * temperature * temperature, selected_count
