"""
Sessions models — Django models mirroring the daemon SQLite schema.
Section 5.2: Sessions and frames for cloud storage/collaboration.
"""
from django.db import models
from django.conf import settings


class ImagingSession(models.Model):
    """An imaging session synced from the daemon."""
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('completed', 'Completed'),
        ('aborted', 'Aborted'),
    ]

    daemon_id = models.CharField(max_length=64, unique=True, help_text="UUID from daemon SQLite")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    name = models.CharField(max_length=255, blank=True)
    rig_id = models.CharField(max_length=64, blank=True)
    target_name = models.CharField(max_length=255, blank=True)
    target_ra = models.FloatField(null=True, blank=True)
    target_dec = models.FloatField(null=True, blank=True)
    location_lat = models.FloatField(null=True, blank=True)
    location_lon = models.FloatField(null=True, blank=True)
    started_at = models.DateTimeField()
    ended_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-started_at']

    def __str__(self):
        return f"{self.target_name or 'Unnamed'} — {self.started_at:%Y-%m-%d}"


class Frame(models.Model):
    """A single captured frame within a session."""
    FRAME_TYPES = [
        ('light', 'Light'),
        ('dark', 'Dark'),
        ('flat', 'Flat'),
        ('bias', 'Bias'),
        ('dark_flat', 'Dark Flat'),
    ]

    daemon_id = models.CharField(max_length=64, unique=True)
    session = models.ForeignKey(ImagingSession, on_delete=models.CASCADE, related_name='frames')
    frame_type = models.CharField(max_length=20, choices=FRAME_TYPES, default='light')
    filter_name = models.CharField(max_length=50, blank=True)
    exposure_s = models.FloatField()
    gain = models.IntegerField(null=True, blank=True)
    temperature = models.FloatField(null=True, blank=True)
    hfr = models.FloatField(null=True, blank=True)
    fwhm = models.FloatField(null=True, blank=True)
    star_count = models.IntegerField(null=True, blank=True)
    background_adu = models.FloatField(null=True, blank=True)
    quality_score = models.FloatField(null=True, blank=True)
    cloud_prob = models.FloatField(null=True, blank=True)
    accepted = models.BooleanField(default=True)
    file_url = models.URLField(blank=True, help_text="S3/GCS URL of the uploaded FITS/XISF file")
    thumbnail_url = models.URLField(blank=True)
    captured_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['captured_at']

    def __str__(self):
        return f"{self.frame_type} {self.filter_name} {self.exposure_s}s"
