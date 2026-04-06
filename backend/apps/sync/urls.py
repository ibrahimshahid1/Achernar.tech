from django.urls import path
from . import views

urlpatterns = [
    path('upload-session/', views.upload_session, name='upload-session'),
    path('available-models/', views.available_models, name='available-models'),
    path('register/', views.register_daemon, name='register-daemon'),
]
