"""
Realtime Models

Tracks daemon connections, user sessions, and telemetry state.

PDF Section 16.2.1 - IoT Relay Architecture
"""
from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
import uuid

User = get_user_model()


class DaemonConnection(models.Model):
    """
    Tracks active daemon connections to the cloud IoT relay.
    
    Each daemon maintains a persistent WebSocket connection and
    is identified by a unique UUID.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        related_name='daemon_connections',
        null=True,
        blank=True,
    )
    
    # Connection metadata
    daemon_version = models.CharField(max_length=50, blank=True)
    hardware_id = models.CharField(max_length=255, help_text="Unique hardware identifier")
    
    # Connection state
    is_connected = models.BooleanField(default=False)
    connected_at = models.DateTimeField(null=True, blank=True)
    disconnected_at = models.DateTimeField(null=True, blank=True)
    last_heartbeat = models.DateTimeField(null=True, blank=True)
    gateway_channel_name = models.CharField(max_length=255, blank=True)
    
    # Location info (optional)
    site_name = models.CharField(max_length=255, blank=True)
    latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-connected_at']
        indexes = [
            models.Index(fields=['user', 'is_connected']),
            models.Index(fields=['last_heartbeat']),
        ]
    
    def __str__(self):
        status = "Connected" if self.is_connected else "Disconnected"
        owner = self.user.username if self.user else "dev-relay"
        return f"{owner} - {self.hardware_id} ({status})"
    
    def mark_connected(self, channel_name=''):
        """Mark daemon as connected"""
        self.is_connected = True
        self.connected_at = timezone.now()
        self.last_heartbeat = timezone.now()
        self.gateway_channel_name = channel_name
        self.save()
    
    def mark_disconnected(self):
        """Mark daemon as disconnected"""
        self.is_connected = False
        self.disconnected_at = timezone.now()
        self.save()
    
    def update_heartbeat(self):
        """Update last heartbeat timestamp"""
        self.last_heartbeat = timezone.now()
        self.save(update_fields=['last_heartbeat'])


class UserSession(models.Model):
    """
    Tracks active user sessions (web clients) for routing telemetry.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='websocket_sessions')
    daemon = models.ForeignKey(DaemonConnection, on_delete=models.SET_NULL, null=True, blank=True)
    
    # Session metadata
    channel_name = models.CharField(max_length=255, unique=True, help_text="Django Channels channel name")
    user_agent = models.TextField(blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    # Session state
    is_active = models.BooleanField(default=True)
    connected_at = models.DateTimeField(auto_now_add=True)
    disconnected_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-connected_at']
        indexes = [
            models.Index(fields=['user', 'is_active']),
            models.Index(fields=['channel_name']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - Session {self.id}"


class TelemetrySnapshot(models.Model):
    """
    Stores the latest telemetry state from each daemon (Digital Twin).
    
    Used for restoring state when users reconnect.
    PDF Section 16.2.1 - Redis Cache (Digital Twin)
    """
    daemon = models.OneToOneField(DaemonConnection, on_delete=models.CASCADE, related_name='telemetry')
    
    # Telemetry data (JSON)
    mount_state = models.JSONField(default=dict, blank=True)
    camera_state = models.JSONField(default=dict, blank=True)
    focuser_state = models.JSONField(default=dict, blank=True)
    filter_wheel_state = models.JSONField(default=dict, blank=True)
    device_health = models.JSONField(default=dict, blank=True)
    guiding_state = models.JSONField(default=dict, blank=True)
    ai_metrics = models.JSONField(default=dict, blank=True)
    
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Telemetry for {self.daemon.hardware_id}"
