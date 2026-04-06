from django.contrib import admin
from .models import ModelVersion, ActiveLearningFeedback

@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ['name', 'version', 'task', 'is_active', 'created_at']

@admin.register(ActiveLearningFeedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ['model_version', 'confidence', 'created_at']
