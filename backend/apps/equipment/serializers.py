from rest_framework import serializers
from .models import ConnectedEquipment


class ConnectedEquipmentSerializer(serializers.ModelSerializer):
    """
    Serializer for connected equipment.
    Transforms database model to frontend EquipmentCardModel format.
    """
    # Rename fields to match frontend expectations
    id = serializers.CharField(source='equipment_id')
    type = serializers.CharField(source='equipment_type')
    
    class Meta:
        model = ConnectedEquipment
        fields = [
            'id',
            'type',
            'name',
            'site',
            'status',
            'tags',
            'meta',
            'connected',
            'driver',
            # `last_seen` was recorded by the model and omitted here, which
            # removed the only fact that separates a live reading from a
            # six-month-old one. A consumer cannot tell fresh from stale
            # without it, so it ships.
            'last_seen',
        ]
        read_only_fields = ['id', 'last_seen']

    def to_representation(self, instance):
        """
        Customize output to match frontend EquipmentCardModel exactly.

        Connection state is derived from freshness, never taken from the
        stored flag alone: `connected` is the last value written by
        POST /api/equipment/sync, and nothing expires it.
        """
        data = super().to_representation(instance)

        fresh = instance.is_fresh and self.context.get('live_source_available', True)
        data['connected'] = bool(instance.connected) and fresh
        data['stale'] = not fresh
        if not fresh:
            # Whatever it was last recorded as, we cannot vouch for it now.
            data['status'] = 'offline'

        # Ensure meta includes connection info for display
        if not data.get('meta'):
            data['meta'] = {}

        data['meta']['driver'] = instance.driver
        data['meta']['connected'] = data['connected']
        data['meta']['last_seen'] = data['last_seen']

        return data
