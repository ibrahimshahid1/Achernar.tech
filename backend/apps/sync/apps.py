"""
Sync app — daemon ↔ cloud sync for sessions, models, and telemetry.
"""
from django.apps import AppConfig

class SyncConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.sync'
    verbose_name = 'Daemon Sync'
