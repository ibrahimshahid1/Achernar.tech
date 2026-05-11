from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('apps.core.urls')),
    path('api/equipment/', include('apps.equipment.urls')),
    path('api/imaging/', include('apps.sessions.urls')),
    path('api/ml/', include('apps.ml.urls')),
    path('api/sync/', include('apps.sync.urls')),
]
