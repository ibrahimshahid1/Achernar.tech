"""
Achernar settings package.

Default to dev settings when imported directly.
Override with DJANGO_SETTINGS_MODULE=backend.settings.prod for production.
"""

# When `backend.settings` is imported (without .dev or .prod suffix),
# default to development settings for backwards compatibility.
from .dev import *  # noqa: F401, F403
