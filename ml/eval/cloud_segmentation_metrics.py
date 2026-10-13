"""Streaming COC metrics on annotated pixels, with explicit episode strata.

Input predictions are class indices, not logits or probabilities. These metrics
do not infer whole-sky coverage, transparency, or observing-pause correctness
from partial region annotations.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


CLASS_ORDER = ("sky", "cloud", "contamination")
IGNORE_INDEX = 255
NUM_CLASSES = len(CLASS_ORDER)


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _mean_defined(values) -> float | None:
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def confusion_metrics(confusion: np.ndarray) -> dict:
    """Summarize a [truth, prediction] confusion matrix without NaN values.

    Recall needs true support, precision needs predicted support, and IoU needs
    nonempty union. Balanced accuracy averages recall over supported classes;
    absent true classes are explicitly listed rather than counted as successes.
    """
    matrix = np.asarray(confusion)
    if matrix.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError("COC confusion matrix must be 3 by 3")
    if not np.issubdtype(matrix.dtype, np.integer) or (matrix < 0).any():
        raise ValueError("Confusion counts must be nonnegative integers")
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    per_class = {}
    for index, name in enumerate(CLASS_ORDER):
        true_positive = int(matrix[index, index])
        actual = int(support[index])
        selected = int(predicted[index])
        union = actual + selected - true_positive
        per_class[name] = {
            "support": actual,
            "predicted_support": selected,
            "iou": _ratio(true_positive, union),
            "precision": _ratio(true_positive, selected),
            "recall": _ratio(true_positive, actual),
        }
    labeled = int(matrix.sum())
    return {
        "confusion_matrix": matrix.tolist(),
        "labeled_pixel_count": labeled,
        "pixel_accuracy": _ratio(int(np.trace(matrix)), labeled),
        "mean_iou": _mean_defined(row["iou"] for row in per_class.values()),
        "balanced_accuracy": _mean_defined(row["recall"] for row in per_class.values()),
        "balanced_accuracy_supported_classes": [
            name for name, row in per_class.items() if row["support"] > 0
        ],
        "per_class": per_class,
    }


def _metadata(value: str | Sequence[str] | None, count: int, name: str):
    if value is None:
        return [None] * count
    if isinstance(value, str):
        if count != 1:
            raise ValueError(f"{name} must provide one value per batch image")
        values = [value]
    else:
        values = list(value)
    if len(values) != count or any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} must contain one nonempty string per image")
    return values


def _empty_accumulator():
    return {
        "confusion": np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64),
        "image_count": 0,
        "images_with_labels": 0,
        "total_pixel_count": 0,
    }


def _accumulate(accumulator, confusion, total_pixels):
    accumulator["confusion"] += confusion
    accumulator["image_count"] += 1
    accumulator["images_with_labels"] += int(confusion.sum() > 0)
    accumulator["total_pixel_count"] += int(total_pixels)


def _summarize(accumulator):
    metrics = confusion_metrics(accumulator["confusion"])
    total = accumulator["total_pixel_count"]
    metrics.update({
        "image_count": accumulator["image_count"],
        "images_with_labels": accumulator["images_with_labels"],
        "total_pixel_count": total,
        "ignored_pixel_count": total - metrics["labeled_pixel_count"],
        "annotated_pixel_fraction": _ratio(metrics["labeled_pixel_count"], total),
    })
    return metrics


class CloudSegmentationMetrics:
    """Accumulate integer [H,W] or [B,H,W] COC predictions and targets.

    ``update(prediction, target, split_groups=..., camera_eras=...)`` accepts a
    metadata string for a single image or one string per batch image. Repeated
    episode/era names accumulate together across calls. Missing metadata omits
    that stratum, while all images still contribute to ``overall``. No tensors
    or per-pixel outputs are retained.
    """

    def __init__(self):
        self._overall = _empty_accumulator()
        self._episodes = {}
        self._camera_eras = {}

    def update(self, prediction, target, *, split_groups=None, camera_eras=None):
        predictions = np.asarray(prediction)
        targets = np.asarray(target)
        if predictions.shape != targets.shape or targets.ndim not in (2, 3):
            raise ValueError("Prediction and target must have matching [H,W] or [B,H,W] shapes")
        if not np.issubdtype(predictions.dtype, np.integer) or not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("Prediction and target must contain integer class indices")
        if not np.isin(targets, [0, 1, 2, IGNORE_INDEX]).all():
            raise ValueError("Target contains an unknown COC class")
        if targets.ndim == 2:
            predictions, targets = predictions[None], targets[None]
        groups = _metadata(split_groups, len(targets), "split_groups")
        eras = _metadata(camera_eras, len(targets), "camera_eras")
        valid = targets != IGNORE_INDEX
        if not np.isin(predictions[valid], [0, 1, 2]).all():
            raise ValueError("Prediction contains an unknown class on an annotated pixel")
        for pred, truth, keep, group, era in zip(predictions, targets, valid, groups, eras, strict=True):
            # Predictions outside annotated support cannot affect any metric.
            encoded = NUM_CLASSES * truth[keep].astype(np.int64) + pred[keep].astype(np.int64)
            matrix = np.bincount(encoded, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
            _accumulate(self._overall, matrix, truth.size)
            if group is not None:
                accumulator = self._episodes.setdefault(group, _empty_accumulator())
                _accumulate(accumulator, matrix, truth.size)
            if era is not None:
                accumulator = self._camera_eras.setdefault(era, _empty_accumulator())
                _accumulate(accumulator, matrix, truth.size)
        return self

    def compute(self) -> dict:
        """Return JSON-safe pixel, episode, and camera-era metrics.

        Pixel totals are pooled; episode macro metrics weight each supplied
        episode equally. Neither supplies a confidence interval or proves that
        filename groups represent independent nights or unseen camera eras.
        """
        episodes = {key: _summarize(value) for key, value in sorted(self._episodes.items())}
        eras = {key: _summarize(value) for key, value in sorted(self._camera_eras.items())}
        return {
            "class_order": list(CLASS_ORDER),
            "ignore_index": IGNORE_INDEX,
            "confusion_orientation": "rows=truth, columns=prediction",
            "metric_scope": "annotated pixels only; no transparency or whole-sky decision labels inferred",
            "overall": _summarize(self._overall),
            "by_episode": episodes,
            "by_camera_era": eras,
            "episode_macro": {
                "episode_count": len(episodes),
                "episodes_with_labels": sum(row["labeled_pixel_count"] > 0 for row in episodes.values()),
                "mean_iou": _mean_defined(row["mean_iou"] for row in episodes.values()),
                "balanced_accuracy": _mean_defined(row["balanced_accuracy"] for row in episodes.values()),
                "per_class_iou": {
                    name: _mean_defined(row["per_class"][name]["iou"] for row in episodes.values())
                    for name in CLASS_ORDER
                },
            },
        }
