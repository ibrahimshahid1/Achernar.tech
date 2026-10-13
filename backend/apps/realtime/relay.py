"""Helpers for reading the current daemon relay state."""
from typing import Optional, List
from .models import DaemonConnection, TelemetrySnapshot
from .state import daemon_is_live, snapshot_payload


def get_active_daemons(user) -> List[DaemonConnection]:
    return [
        daemon for daemon in DaemonConnection.objects.filter(user=user, is_connected=True)
        if daemon_is_live(daemon)
    ]


def get_daemon_telemetry(daemon_id: str) -> Optional[dict]:
    try:
        daemon = DaemonConnection.objects.get(id=daemon_id)
    except DaemonConnection.DoesNotExist:
        return None
    snapshot = TelemetrySnapshot.objects.filter(daemon=daemon).first()
    # A second cache could outlive an explicit device or gateway disconnect.
    return snapshot_payload(daemon, snapshot)


def is_daemon_healthy(daemon_id: str) -> bool:
    daemon = DaemonConnection.objects.filter(id=daemon_id).first()
    return daemon_is_live(daemon)
