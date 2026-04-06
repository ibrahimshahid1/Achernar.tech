"""
ML views — model serving and feedback endpoints.
GET  /api/models/             — available model versions
POST /api/feedback/           — active learning label submission
"""
from rest_framework import viewsets, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import ModelVersion, ActiveLearningFeedback
from rest_framework import serializers


class ModelVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelVersion
        fields = '__all__'


class FeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = ActiveLearningFeedback
        fields = '__all__'


class ModelVersionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ModelVersion.objects.filter(is_active=True)
    serializer_class = ModelVersionSerializer


@api_view(['POST'])
def submit_feedback(request):
    """POST /api/feedback/ — active learning label submission."""
    serializer = FeedbackSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
