from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from .views import ApiRootView, HelloView

urlpatterns = [
    path('', ApiRootView.as_view(), name='api-root'),
    path('hello/', HelloView.as_view(), name='hello'),
]

urlpatterns = format_suffix_patterns(urlpatterns)
