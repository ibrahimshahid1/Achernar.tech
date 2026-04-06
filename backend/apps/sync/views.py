"""
Sync views — endpoints for daemon ↔ cloud sync operations.
"""
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status


@api_view(['POST'])
def upload_session(request):
    """Receive a session + frames from a daemon."""
    raise NotImplementedError("Implement session upload sync endpoint")


@api_view(['GET'])
def available_models(request):
    """List model versions available for daemon download."""
    raise NotImplementedError("Implement model download list endpoint")


@api_view(['POST'])
def register_daemon(request):
    """Register a new daemon instance."""
    raise NotImplementedError("Implement daemon registration endpoint")
