"""
Sync models — tracks daemon registration and sync state.
"""
from django.db import models
from django.conf import settings


class DaemonRegistration(models.Model):
    """A registered daemon instance for syncing."""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    daemon_id = models.CharField(max_length=64, unique=True)
    hostname = models.CharField(max_length=255, blank=True)
    last_sync = models.DateTimeField(null=True, blank=True)
    auth_token = models.CharField(max_length=255)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Daemon {self.daemon_id} ({self.hostname})"


class SyncEvent(models.Model):
    """Log of sync operations between daemon and cloud."""
    DIRECTIONS = [('upload', 'Upload'), ('download', 'Download')]
    TYPES = [
        ('session', 'Session'),
        ('frames', 'Frames'),
        ('model', 'Model'),
        ('telemetry', 'Telemetry'),
    ]

    daemon = models.ForeignKey(DaemonRegistration, on_delete=models.CASCADE)
    direction = models.CharField(max_length=20, choices=DIRECTIONS)
    sync_type = models.CharField(max_length=20, choices=TYPES)
    items_count = models.IntegerField(default=0)
    success = models.BooleanField(default=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
