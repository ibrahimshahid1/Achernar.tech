"""
ML app — model serving endpoints and active learning feedback collection.
"""
from django.apps import AppConfig

class MlConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.ml'
    verbose_name = 'ML Model Management'
