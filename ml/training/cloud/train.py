"""
ML Training Pipeline - Cloud Classifier

PyTorch Lightning module for cloud detection.
PDF Section 6.2.4 - Cloud/Obstruction Classifier
"""

# TODO: Implement PyTorch Lightning training module
# - MobileNetV3-Small binary classifier
# - Input: Full frame or downscaled
# - Output: clear vs clouded/obstructed + confidence
# - Loss: BCE + temperature scaling for calibration
# - Augmentations: blur, fog synthesis, contrast reduction
# - Hard negative mining
