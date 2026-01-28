"""
Core Django App - Shared Models

Base abstract models for reuse across all apps.
"""
from django.db import models
import uuid


class TimestampedModel(models.Model):
    """
    Abstract base model with automatic timestamp tracking.
    
    Usage:
        class MyModel(TimestampedModel):
            name = models.CharField(max_length=255)
    """
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        abstract = True


class UUIDModel(models.Model):
    """
    Abstract base model with UUID primary key.
    
    Usage:
        class MyModel(UUIDModel, TimestampedModel):
            name = models.CharField(max_length=255)
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    class Meta:
        abstract = True


class SoftDeleteModel(models.Model):
    """
    Abstract base model with soft delete capability.
    
    Usage:
        queryset = MyModel.objects.filter(is_deleted=False)
    """
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        abstract = True
    
    def soft_delete(self):
        """Mark object as deleted without removing from database"""
        from django.utils import timezone
        self.is_deleted = True
        self.deleted_at = timezone.now()
        self.save()
