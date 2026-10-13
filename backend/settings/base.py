"""
Achernar Django settings — Base configuration.
Shared settings for all environments (dev, prod).

For more information on this file, see
https://docs.djangoproject.com/en/5.2/topics/settings/
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent.parent


# Application definition

INSTALLED_APPS = [
    'daphne',  # Must be first for ASGI support
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Third party
    'rest_framework',
    'corsheaders',
    'channels',

    # Achernar. Every app below has models, views or both. Apps that were
    # `startapp` scaffolds with no models, migrations or URLs were removed —
    # INSTALLED_APPS is a claim about what exists, and it should stay true.
    # Re-create with `startapp` when there is something to put in one.
    'apps.core',           # Shared utilities and base models
    'apps.equipment',      # Equipment database and daemon command relay
    'apps.realtime',       # WebSocket/IoT relay for daemon connections
    'apps.sessions.apps.SessionsConfig',  # Imaging session/frame sync from daemon
    'apps.ai_engine',      # Cloud AI services (AstroM3 embeddings)
    'apps.ml',             # ML model serving and active learning
    'apps.sync',           # Daemon ↔ cloud sync
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'backend.wsgi.application'
ASGI_APPLICATION = 'backend.asgi.application'


# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.2/howto/static-files/

STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Celery was removed: it was configured and running a worker, but every task
# in backend/tasks raised NotImplementedError and no @shared_task existed
# anywhere in the repo. Re-add it when there is a job that genuinely must run
# out-of-band. Note that model distribution is better as a daemon-side pull
# via GET /api/sync/available-models/ than as a cloud-side push.

# Django REST Framework
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 100,
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
    ],
}

# Cache
# -----------------------------------------------------------------------------
# The daemon command relay hands a COMMAND_RESULT from the ASGI WebSocket
# consumer (apps.realtime.consumers.DaemonConsumer) back to the synchronous DRF
# view (apps.equipment.views.send_relay_command) through the Django cache. That
# handoff only works if every server process shares the SAME cache. With the
# default per-process LocMemCache this works for a single ASGI process, but
# silently breaks the moment the deployment runs more than one process/replica
# (the polling HTTP worker never sees the result the WS worker wrote, so every
# relay command times out as "pending"). Whenever REDIS_URL is configured we
# therefore back the cache with Redis so the relay stays correct under scaling;
# otherwise we fall back to a named LocMemCache for single-process local dev.
if os.getenv('REDIS_URL'):
    CACHES = {
        'default': {
            'BACKEND': 'django.core.cache.backends.redis.RedisCache',
            'LOCATION': os.getenv('REDIS_URL'),
        }
    }
else:
    CACHES = {
        'default': {
            'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
            'LOCATION': 'achernar-relay',
        }
    }
