"""
Realtime Django App Configuration

Handles WebSocket connections, IoT relay, and real-time telemetry
routing between Rust daemons and web clients.

PDF Reference: Section 16.2 - Cloud-Mediated Remote Operations
"""
from django.apps import AppConfig


class RealtimeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.realtime'
    verbose_name = 'Real-time Communication'
