"""COC v1 research segmentation; no equipment-control authority."""

CLASS_NAMES = ("sky", "cloud", "contamination")
IGNORE_INDEX = 255
CONTRACT_VERSION = "coc-segmentation-v1"
ONNX_PRECISION = {"weights": "FP32", "input": "FP32", "output": "FP32", "normalization_arithmetic": "FP64"}
