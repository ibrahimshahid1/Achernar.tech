from django.contrib import admin
from .models import DaemonRegistration, SyncEvent

@admin.register(DaemonRegistration)
class DaemonRegistrationAdmin(admin.ModelAdmin):
    list_display = ['daemon_id', 'hostname', 'user', 'last_sync', 'is_active']

@admin.register(SyncEvent)
class SyncEventAdmin(admin.ModelAdmin):
    list_display = ['daemon', 'direction', 'sync_type', 'success', 'created_at']
