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
        ]
        read_only_fields = ['id']
    
    def to_representation(self, instance):
        """
        Customize output to match frontend EquipmentCardModel exactly
        """
        data = super().to_representation(instance)
        
        # Ensure meta includes connection info for display
        if not data.get('meta'):
            data['meta'] = {}
        
        data['meta']['driver'] = instance.driver
        data['meta']['connected'] = instance.connected
        
        return data
