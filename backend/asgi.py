"""
ASGI config for Achernar backend project.

Enables Django Channels for WebSocket support.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

# Initialize Django ASGI application early
django_asgi_app = get_asgi_application()

# Import WebSocket routing after Django is initialized
import apps.realtime.routing

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(
            apps.realtime.routing.websocket_urlpatterns
        )
    ),
})

