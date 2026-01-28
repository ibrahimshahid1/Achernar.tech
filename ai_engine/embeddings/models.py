"""
Tri-Modal Embedding Models

Handles Image ↔ Light Curve ↔ Spectrum ↔ Metadata embeddings.
PDF Section 6.6 - Tri-Modal Embedding
"""
from django.db import models
from django.contrib.postgres.fields import ArrayField
import uuid


class EmbeddingModel(models.Model):
    """Tracks trained embedding models (tri-modal CLIP-style)"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    name = models.CharField(max_length=255)
    version = models.CharField(max_length=50)
    
    # Model architecture
    image_encoder = models.CharField(max_length=100, help_text="e.g., ViT-Small, ConvNeXt-Tiny")
    timeseries_encoder = models.CharField(max_length=100, help_text="e.g., Transformer, InceptionTime")
    spectra_encoder = models.CharField(max_length=100, help_text="e.g., 1D ResNet")
    
    # Training metadata
    embedding_dim = models.IntegerField(default=256)
    training_samples = models.IntegerField()
    training_loss = models.FloatField(null=True, blank=True)
    
    # Deployment status
    is_deployed = models.BooleanField(default=False)
    deployed_at = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} v{self.version}"


class ObjectEmbedding(models.Model):
    """Stores embeddings for astronomical objects"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    model = models.ForeignKey(EmbeddingModel, on_delete=models.CASCADE)
    
    # Object metadata
    object_name = models.CharField(max_length=255, blank=True)
    catalog_id = models.CharField(max_length=100, blank=True)
    object_type = models.CharField(max_length=100, blank=True)
    
    # Embedding vector (256-D)
    embedding = ArrayField(models.FloatField(), size=256)
    
    # Source modality
    modality = models.CharField(max_length=20, choices=[
        ('image', 'Image'),
        ('timeseries', 'Time-Series'),
        ('spectra', 'Spectra'),
        ('multimodal', 'Multi-Modal'),
    ])
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['object_type', 'modality']),
        ]
    
    def __str__(self):
        return f"{self.object_name or self.catalog_id} ({self.modality})"
