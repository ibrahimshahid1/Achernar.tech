from django.contrib import admin
from .models import ImagingSession, Frame

@admin.register(ImagingSession)
class ImagingSessionAdmin(admin.ModelAdmin):
    list_display = ['target_name', 'started_at', 'status', 'rig_id']
    list_filter = ['status']

@admin.register(Frame)
class FrameAdmin(admin.ModelAdmin):
    list_display = ['session', 'frame_type', 'filter_name', 'exposure_s', 'hfr', 'accepted']
    list_filter = ['frame_type', 'accepted']
