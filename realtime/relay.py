"""
IoT Relay Logic

Helper functions for managing daemon connections and routing.
PDF Section 16.2.1 - IoT Relay Architecture
"""
from typing import Optional, List
from django.core.cache import cache
from .models import DaemonConnection, TelemetrySnapshot


def get_active_daemons(user) -> List[DaemonConnection]:
    """Get all active daemon connections for a user"""
    return DaemonConnection.objects.filter(
        user=user,
        is_connected=True
    ).select_related('user')


def get_daemon_telemetry(daemon_id: str) -> Optional[dict]:
    """
    Get latest telemetry for a daemon.
    
    First checks Redis cache (fast), falls back to database.
    """
    # Try cache first
    cache_key = f'telemetry:{daemon_id}'
    cached_telemetry = cache.get(cache_key)
    
    if cached_telemetry:
        return cached_telemetry
    
    # Fall back to database
    try:
        daemon = DaemonConnection.objects.get(id=daemon_id)
        snapshot = TelemetrySnapshot.objects.filter(daemon=daemon).first()
        
        if snapshot:
            telemetry = {
                'mount': snapshot.mount_state,
                'camera': snapshot.camera_state,
                'guiding': snapshot.guiding_state,
                'ai': snapshot.ai_metrics,
            }
            
            # Cache for 30 seconds
            cache.set(cache_key, telemetry, 30)
            return telemetry
    
    except DaemonConnection.DoesNotExist:
        pass
    
    return None


def is_daemon_healthy(daemon_id: str) -> bool:
    """
    Check if daemon is healthy based on last heartbeat.
    
    A daemon is considered unhealthy if no heartbeat in 60 seconds.
    """
    from django.utils import timezone
    from datetime import timedelta
    
    try:
        daemon = DaemonConnection.objects.get(id=daemon_id)
        
        if not daemon.is_connected:
            return False
        
        if not daemon.last_heartbeat:
            return False
        
        # Check if heartbeat is recent (within 60 seconds)
        threshold = timezone.now() - timedelta(seconds=60)
        return daemon.last_heartbeat > threshold
    
    except DaemonConnection.DoesNotExist:
        return False
