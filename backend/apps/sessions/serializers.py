from rest_framework import serializers
from .models import ImagingSession, Frame

class FrameSerializer(serializers.ModelSerializer):
    class Meta:
        model = Frame
        fields = '__all__'

class ImagingSessionSerializer(serializers.ModelSerializer):
    frames = FrameSerializer(many=True, read_only=True)
    frame_count = serializers.IntegerField(source='frames.count', read_only=True)

    class Meta:
        model = ImagingSession
        fields = '__all__'
