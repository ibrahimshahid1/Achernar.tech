"""
Achernar Django settings — Development overrides.

Usage:
    DJANGO_SETTINGS_MODULE=backend.settings.dev python manage.py runserver
"""

from .base import *  # noqa: F401, F403

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-j@p(&xu#@z4nrk#b(h#^na5$%3mw@7wdg$!+ow7v9t8@a!q!7a'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']

# Database — SQLite for easy local dev, PostgreSQL via env vars
if os.getenv('USE_SQLITE'):
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': os.getenv('DB_NAME', 'achernar'),
            'USER': os.getenv('DB_USER', 'postgres'),
            'PASSWORD': os.getenv('DB_PASSWORD', 'postgres'),
            'HOST': os.getenv('DB_HOST', 'localhost'),
            'PORT': os.getenv('DB_PORT', '5432'),
        }
    }

# CORS — wide open for local development
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True

# Channels — use in-memory layer if no Redis
if not os.getenv('REDIS_URL'):
    CHANNEL_LAYERS = {
        'default': {
            'BACKEND': 'channels.layers.InMemoryChannelLayer',
        },
    }
else:
    CHANNEL_LAYERS = {
        'default': {
            'BACKEND': 'channels_redis.core.RedisChannelLayer',
            'CONFIG': {
                'hosts': [os.getenv('REDIS_URL')],
            },
        },
    }
