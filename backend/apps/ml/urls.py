from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'models', views.ModelVersionViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('feedback/', views.submit_feedback, name='submit-feedback'),
]
