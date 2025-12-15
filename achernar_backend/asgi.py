"""
ASGI config for achernar_backend project.
Enables Django Channels for WebSocket support.
"""

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'achernar_backend.settings')

django_asgi_app = get_asgi_application()

# Import routing after Django setup
# from observatory import routing as observatory_routing

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    # WebSocket routing will be added here
    # "websocket": AuthMiddlewareStack(
    #     URLRouter(
    #         observatory_routing.websocket_urlpatterns
    #     )
    # ),
})
