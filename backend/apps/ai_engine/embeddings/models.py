"""
AstroM3 embedding models.

Stores the 512-D shared embedding vectors produced by AstroM3 / AstroM3Edge.
"""
from django.db import models
import uuid


ASTROM3_EMBEDDING_DIM = 512
ASTROM3_MODALITY_CHOICES = [
    ('image', 'Image'),
    ('photometry', 'Photometry'),
    ('telemetry', 'Telemetry'),
    ('timeseries', 'Time-Series'),
    ('spectra', 'Spectra'),
    ('metadata', 'Metadata'),
    ('multimodal', 'Multi-Modal'),
]


class EmbeddingModel(models.Model):
    """Tracks trained embedding model versions."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    name = models.CharField(max_length=255)
    version = models.CharField(max_length=50)
    
    # Model architecture
    image_encoder = models.CharField(max_length=100, blank=True, help_text="Edge only, e.g. AstroM3Edge ImageEncoder")
    timeseries_encoder = models.CharField(max_length=100, help_text="e.g., AstroM3 Informer")
    spectra_encoder = models.CharField(max_length=100, blank=True, help_text="Cloud only, e.g. AstroM3 GalSpecNet")
    metadata_encoder = models.CharField(max_length=100, default="AstroM3 MetaModel")
    
    # Training metadata
    embedding_dim = models.IntegerField(default=ASTROM3_EMBEDDING_DIM)
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
    
    # Embedding vector (512-D for AstroM3 / AstroM3Edge). Stored as JSON for
    # local SQLite compatibility; PostgreSQL deployments additionally enable
    # pgvector via migrations for indexed vector search.
    embedding = models.JSONField(default=list)
    
    # Source modality
    modality = models.CharField(max_length=20, choices=ASTROM3_MODALITY_CHOICES)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['object_type', 'modality'], name='ai_obj_emb_type_mod_idx'),
        ]
    
    def __str__(self):
        return f"{self.object_name or self.catalog_id} ({self.modality})"
