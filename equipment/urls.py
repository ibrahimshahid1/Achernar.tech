from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ConnectedEquipmentViewSet

router = DefaultRouter()
router.register(r'', ConnectedEquipmentViewSet, basename='equipment')

urlpatterns = [
    path('', include(router.urls)),
]
