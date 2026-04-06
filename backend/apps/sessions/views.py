"""
Sessions views — REST API for session and frame management.
GET  /api/sessions/           — list sessions with pagination
GET  /api/sessions/{id}/      — session detail + frames
"""
from rest_framework import viewsets
from .models import ImagingSession, Frame
from .serializers import ImagingSessionSerializer, FrameSerializer


class ImagingSessionViewSet(viewsets.ModelViewSet):
    queryset = ImagingSession.objects.all()
    serializer_class = ImagingSessionSerializer


class FrameViewSet(viewsets.ModelViewSet):
    queryset = Frame.objects.all()
    serializer_class = FrameSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        session_id = self.request.query_params.get('session_id')
        if session_id:
            qs = qs.filter(session_id=session_id)
        return qs
