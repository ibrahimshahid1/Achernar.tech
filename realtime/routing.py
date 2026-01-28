"""
WebSocket URL Routing for Realtime App

Defines WebSocket URL patterns for Django Channels.
"""
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # Daemon persistent connection endpoint
    # wss://api.achernar.io/ws/gateway/<daemon_id>/
    re_path(r'ws/gateway/(?P<daemon_id>[0-9a-f-]+)/$', consumers.DaemonConsumer.as_asgi()),
    
    # Web client telemetry subscription endpoint
    # ws://localhost:8000/ws/telemetry/<daemon_id>/
    re_path(r'ws/telemetry/(?P<daemon_id>[0-9a-f-]+)/$', consumers.TelemetryConsumer.as_asgi()),
]
