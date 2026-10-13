"""Connection and snapshot freshness shared by relay HTTP and WebSocket reads."""
import os
from datetime import timedelta

from django.utils import timezone


DAEMON_STALE_SECONDS = int(os.getenv('ACHERNAR_DAEMON_STALE_SECONDS', '30'))
TELEMETRY_STALE_SECONDS = int(os.getenv('ACHERNAR_TELEMETRY_STALE_SECONDS', '30'))
HARDWARE_FIELDS = {
    'camera': 'camera_state',
    'mount': 'mount_state',
    'focuser': 'focuser_state',
    'filter_wheel': 'filter_wheel_state',
}


def daemon_is_live(daemon):
    return bool(
        daemon and daemon.is_connected and daemon.last_heartbeat
        and daemon.last_heartbeat >= timezone.now() - timedelta(seconds=DAEMON_STALE_SECONDS)
    )


def connected_hardware(value):
    """A name or an old descriptor alone is never evidence of a connection."""
    return value if isinstance(value, dict) and value.get('connected') is True else None


def snapshot_payload(daemon, snapshot=None):
    daemon_connected = daemon_is_live(daemon)
    fresh = bool(
        daemon_connected and snapshot and snapshot.updated_at
        and snapshot.updated_at >= timezone.now() - timedelta(seconds=TELEMETRY_STALE_SECONDS)
    )
    payload = {
        name: connected_hardware(getattr(snapshot, field)) if fresh else None
        for name, field in HARDWARE_FIELDS.items()
    }
    payload.update({
        'connected': any(payload.values()),
        'daemon_connected': daemon_connected,
        'telemetry_stale': not fresh,
        'device_health': snapshot.device_health if fresh else {},
        'guiding': snapshot.guiding_state if fresh else {},
        'ai': snapshot.ai_metrics if fresh else {},
        'updated_at': snapshot.updated_at.isoformat() if snapshot else None,
    })
    return payload
