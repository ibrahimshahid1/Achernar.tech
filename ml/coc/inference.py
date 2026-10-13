"""Explicit COC research preprocessing and region summaries (no five-class mapping)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from ml.coc import CLASS_NAMES, CONTRACT_VERSION


def preprocess_image(path, size):
    with Image.open(path) as source:
        # Public training inputs are rendered RGB, not linear RAW/FITS or ADU.
        if source.mode not in {"RGB", "RGBA", "L", "P"}:
            raise ValueError("Provide a rendered 8-bit sky image, not a sensor/calibration asset")
        if source.getexif().get(274, 1) != 1:
            raise ValueError("Apply and record EXIF orientation before COC inference")
        original_size = list(source.size)
        rgb = source.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        image = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
    return image[None], original_size


def load_region(path, size, original_size):
    with Image.open(path) as source:
        if list(source.size) != original_size:
            raise ValueError("ROI must match original image geometry")
        region = np.asarray(source.convert("L"))
        if not np.isin(region, [0, 1, 255]).all():
            raise ValueError("ROI must be binary, 0 outside and 1/255 inside")
        region = np.asarray(Image.fromarray((region > 0).astype(np.uint8)).resize(
            (size, size), Image.Resampling.NEAREST)).astype(bool)
    return region


def summarize_logits(logits, valid_region=None, max_contamination=0.3):
    logits = np.asarray(logits)
    if logits.ndim != 3 or logits.shape[0] != 3 or min(logits.shape) < 1 or not np.isfinite(logits).all():
        raise ValueError("Expected finite logits [3,H,W]")
    if not 0 <= max_contamination <= 1:
        raise ValueError("Invalid diagnostic contamination threshold")
    shifted = logits.astype(np.float64) - logits.max(axis=0, keepdims=True)
    scores = np.exp(shifted)
    scores /= scores.sum(axis=0, keepdims=True)
    classes = scores.argmax(axis=0).astype(np.uint8)
    summary = {"contract_version": CONTRACT_VERSION, "class_order": list(CLASS_NAMES),
               "status": "abstained", "reason": "valid_region_required", "valid_region_pixels": 0,
               "cloud_area_fraction_of_roi": None, "cloud_fraction_of_classified_sky": None,
               "contamination_area_fraction_of_roi": None, "mean_soft_cloud_share_of_sky": None,
               "probability_calibrated": False, "transparency": None, "five_class_condition": None,
               "runtime_eligible": False, "promotion_status": "research"}
    if valid_region is None:
        return summary, classes, scores.astype(np.float32)
    region = np.asarray(valid_region)
    if region.dtype != bool or region.shape != classes.shape:
        raise ValueError("Valid region must be boolean [H,W], independent of predictions")
    count = int(region.sum())
    summary["valid_region_pixels"] = count
    if count == 0:
        summary["reason"] = "empty_valid_region"
        return summary, classes, scores.astype(np.float32)
    sky, cloud, contamination = [int(((classes == i) & region).sum()) for i in range(3)]
    summary["cloud_area_fraction_of_roi"] = cloud / count
    summary["contamination_area_fraction_of_roi"] = contamination / count
    if sky + cloud == 0:
        summary["reason"] = "no_classified_sky"
    elif contamination / count > max_contamination:
        summary["reason"] = "contamination_exceeds_diagnostic_threshold"
    else:
        summary.update(status="research_diagnostic", reason=None,
                       cloud_fraction_of_classified_sky=cloud / (sky + cloud),
                       mean_soft_cloud_share_of_sky=float(scores[1, region].sum() /
                                                         (scores[0, region] + scores[1, region]).sum()))
    return summary, classes, scores.astype(np.float32)
