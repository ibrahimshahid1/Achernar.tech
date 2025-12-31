from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()


class EquipmentType(models.TextChoices):
    """Equipment type choices matching frontend types"""
    MOUNT = 'mount', 'Mount'
    CAMERA = 'camera', 'Camera'
    FOCUSER = 'focuser', 'Focuser'
    FILTER_WHEEL = 'filter-wheel', 'Filter Wheel'
    DOME = 'dome', 'Dome'


class ConnectionStatus(models.TextChoices):
    """Connection status matching frontend badge styles"""
    NOMINAL = 'nominal', 'Nominal'
    WARNING = 'warning', 'Warning'
    OFFLINE = 'offline', 'Offline'


class ConnectedEquipment(models.Model):
    """
    Represents currently connected equipment state from daemon.
    This is a cache/relay of daemon equipment status.
    Updated via daemon WebSocket messages or periodic sync.
    Per PDF Section 16.2: IoT Relay Architecture
    """
    # Equipment identification (from daemon)
    equipment_id = models.CharField(max_length=100, unique=True, primary_key=True)
    equipment_type = models.CharField(max_length=20, choices=EquipmentType.choices)
    name = models.CharField(max_length=200)
    
    # Connection info
    connected = models.BooleanField(default=False)
    driver = models.CharField(max_length=50, blank=True)  # 'indi', 'ascom', 'zwo_asi', etc.
    
    # Location/site
    site = models.CharField(max_length=100, default='Local')
    
    # Status
    status = models.CharField(
        max_length=20, 
        choices=ConnectionStatus.choices, 
        default=ConnectionStatus.OFFLINE
    )
    
    # Tags for filtering/display
    tags = models.JSONField(default=list, blank=True)
    
    # Metadata (telemetry snapshot)
    meta = models.JSONField(default=dict, blank=True)
    
    # Timestamps
    last_seen = models.DateTimeField(auto_now=True)
    first_seen = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['equipment_type', 'name']
        verbose_name = 'Connected Equipment'
        verbose_name_plural = 'Connected Equipment'
    
    def __str__(self):
        status_icon = '🟢' if self.connected else '🔴'
        return f"{status_icon} {self.name} ({self.equipment_type})"
