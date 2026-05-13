"""
Sync views — endpoints for daemon ↔ cloud sync operations.
"""
import os
import uuid

from django.utils import timezone
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from apps.ml.models import ModelVersion
from apps.realtime.models import DaemonConnection


DEFAULT_RELAY_DEV_TOKEN = "achernar-dev-token"


def expected_relay_token():
    return os.getenv("ACHERNAR_RELAY_DEV_TOKEN", DEFAULT_RELAY_DEV_TOKEN)


def request_relay_token(request):
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return (
        request.headers.get("X-Achernar-Relay-Token")
        or request.data.get("dev_token")
        or request.data.get("auth_token")
        or request.data.get("token")
    )


def websocket_url(request, path):
    scheme = "wss" if request.is_secure() else "ws"
    return f"{scheme}://{request.get_host()}{path}"


def not_implemented(feature):
    return Response(
        {
            "status": "not_implemented",
            "feature": feature,
            "detail": "This cloud sync endpoint is mounted for local integration but is not implemented yet.",
        },
        status=status.HTTP_501_NOT_IMPLEMENTED,
    )


@api_view(['POST'])
def upload_session(request):
    """Receive a session + frames from a daemon."""
    return not_implemented("session upload")


@api_view(['GET'])
def available_models(request):
    """List model versions available for daemon download."""
    models = ModelVersion.objects.filter(is_active=True).values(
        "name",
        "version",
        "task",
        "framework",
        "sha256",
        "file_url",
        "metadata",
    )
    return Response({"models": list(models)})


@api_view(['POST'])
def register_daemon(request):
    """Register a new daemon instance."""
    if request_relay_token(request) != expected_relay_token():
        return Response(
            {"status": "forbidden", "detail": "Invalid relay token."},
            status=status.HTTP_403_FORBIDDEN,
        )

    daemon_id = request.data.get("daemon_id")
    daemon = None
    if daemon_id:
        try:
            daemon_uuid = uuid.UUID(str(daemon_id))
        except ValueError:
            return Response(
                {"status": "error", "detail": "daemon_id must be a UUID."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        daemon, _ = DaemonConnection.objects.get_or_create(
            id=daemon_uuid,
            defaults={
                "hardware_id": request.data.get("hardware_id") or f"daemon-{daemon_uuid}",
            },
        )
    else:
        hardware_id = request.data.get("hardware_id") or request.data.get("hostname") or "local-dev-daemon"
        daemon = DaemonConnection.objects.filter(hardware_id=hardware_id).order_by("-created_at").first()
        if daemon is None:
            daemon = DaemonConnection.objects.create(hardware_id=hardware_id)

    daemon.hardware_id = request.data.get("hardware_id") or daemon.hardware_id
    daemon.daemon_version = request.data.get("daemon_version", daemon.daemon_version)
    daemon.site_name = request.data.get("site_name", daemon.site_name)
    daemon.last_heartbeat = timezone.now()
    daemon.save(update_fields=["hardware_id", "daemon_version", "site_name", "last_heartbeat", "updated_at"])

    gateway_path = f"/ws/gateway/{daemon.id}/"
    telemetry_path = f"/ws/telemetry/{daemon.id}/"
    return Response(
        {
            "status": "registered",
            "daemon_id": str(daemon.id),
            "hardware_id": daemon.hardware_id,
            "gateway_url": websocket_url(request, gateway_path),
            "telemetry_url": websocket_url(request, telemetry_path),
        }
    )
