"""
Sessions app — mirrors daemon SQLite session/frame schema.
Stores imaging session metadata uploaded from the daemon.
"""
from django.apps import AppConfig

class SessionsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.sessions'
    verbose_name = 'Imaging Sessions'
