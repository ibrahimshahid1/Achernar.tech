from rest_framework import serializers
from .models import ImagingSession, Frame

class FrameSerializer(serializers.ModelSerializer):
    background_provenance = serializers.CharField(read_only=True)

    def validate(self, attrs):
        """Reject background values the database constraints would refuse
        (400 rather than an IntegrityError): a versioned row needs both
        statuses, and carries a level/sigma only when it was measured."""
        def value(name):
            if name in attrs:
                return attrs[name]
            return getattr(self.instance, name, None) if self.instance else None

        if value('background_schema_version') is not None:
            errors = {}
            for status in ('background_level_status', 'background_noise_status'):
                if value(status) is None:
                    errors[status] = 'required when background_schema_version is set'
            if value('background_adu') is not None and value('background_level_status') != 'measured':
                errors['background_adu'] = 'must be null unless background_level_status is measured'
            if value('background_sigma') is not None and value('background_noise_status') != 'measured':
                errors['background_sigma'] = 'must be null unless background_noise_status is measured'
            if errors:
                raise serializers.ValidationError(errors)
        return attrs

    class Meta:
        model = Frame
        fields = '__all__'

class ImagingSessionSerializer(serializers.ModelSerializer):
    frames = FrameSerializer(many=True, read_only=True)
    frame_count = serializers.IntegerField(source='frames.count', read_only=True)

    class Meta:
        model = ImagingSession
        fields = '__all__'
