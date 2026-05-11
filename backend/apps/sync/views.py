"""
Sync views — endpoints for daemon ↔ cloud sync operations.
"""
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from apps.ml.models import ModelVersion


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
    return not_implemented("daemon registration")
