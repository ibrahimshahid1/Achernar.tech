from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny  # For now, no auth required for local testing
from django.utils import timezone
from .models import ConnectedEquipment
from .serializers import ConnectedEquipmentSerializer


class ConnectedEquipmentViewSet(viewsets.ModelViewSet):
    """
    API for connected equipment (daemon relay).
    Per PDF Section 16.2: IoT Relay Architecture
    """
    queryset = ConnectedEquipment.objects.all()
    serializer_class = ConnectedEquipmentSerializer
    permission_classes = [AllowAny]  # TODO: Add proper auth later
    
    @action(detail=False, methods=['get'], url_path='connected')
    def connected(self, request):
        """
        GET /api/equipment/connected/
        Returns currently connected equipment for dashboard display.
        This is the main endpoint the frontend will call.
        """
        # Get all equipment, ordered by type
        equipment = self.get_queryset().order_by('equipment_type', 'name')
        serializer = self.get_serializer(equipment, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['post'], url_path='sync')
    def sync(self, request):
        """
        POST /api/equipment/sync
        Sync endpoint for daemon to push equipment updates.
        Expected payload: { "equipment": [...] }
        
        This will be called by the daemon to update Django's cache
        of connected equipment.
        """
        equipment_data = request.data.get('equipment', [])
        
        updated_ids = []
        for item in equipment_data:
            equipment_id = item.get('id') or item.get('equipment_id')
            
            # Update or create equipment record
            obj, created = ConnectedEquipment.objects.update_or_create(
                equipment_id=equipment_id,
                defaults={
                    'equipment_type': item.get('equipment_type', item.get('type')),
                    'name': item.get('name'),
                    'connected': item.get('connected', False),
                    'driver': item.get('driver', ''),
                    'site': item.get('site', 'Local'),
                    'status': self._determine_status(item),
                    'tags': item.get('tags', []),
                    'meta': item.get('meta', {}),
                }
            )
            updated_ids.append(equipment_id)
        
        # Mark equipment not in the update as offline
        ConnectedEquipment.objects.exclude(
            equipment_id__in=updated_ids
        ).update(
            connected=False,
            status='offline'
        )
        
        return Response({
            'status': 'ok',
            'updated': len(updated_ids),
            'timestamp': timezone.now().isoformat()
        })
    
    def _determine_status(self, item):
        """Determine status from item data"""
        if not item.get('connected'):
            return 'offline'
        
        # Check for warnings in meta or status fields
        meta = item.get('meta', {})
        if 'warning' in str(meta).lower() or item.get('status') == 'warning':
            return 'warning'
        
        return 'nominal'
