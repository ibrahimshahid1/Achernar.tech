"""
ML models — tracks model versions and active learning feedback.
"""
from django.db import models


class ModelVersion(models.Model):
    """Registered ML model version available for download by daemons."""
    name = models.CharField(max_length=100)
    version = models.CharField(max_length=50)
    task = models.CharField(max_length=100)
    framework = models.CharField(max_length=50, default='onnx')
    sha256 = models.CharField(max_length=64)
    file_url = models.URLField(help_text="S3/GCS URL of the model file")
    metadata = models.JSONField(default=dict)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['name', 'version']
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} v{self.version}"


class ActiveLearningFeedback(models.Model):
    """Human feedback for active learning loop."""
    model_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE)
    input_data = models.JSONField()
    predicted_output = models.JSONField()
    human_label = models.JSONField()
    confidence = models.FloatField(null=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
