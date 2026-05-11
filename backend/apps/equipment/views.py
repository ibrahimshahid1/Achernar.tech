from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny  # For now, no auth required for local testing
from django.utils import timezone
import json
import os
import urllib.error
import urllib.request
from .models import ConnectedEquipment
from .serializers import ConnectedEquipmentSerializer


def daemon_api_base():
    return os.getenv('ACHERNAR_DAEMON_API', 'http://127.0.0.1:3030/api/v1').rstrip('/')


def fetch_daemon_status(timeout=1.5):
    url = f'{daemon_api_base()}/status'
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None


def post_daemon(path, payload=None, timeout=5):
    data = json.dumps(payload or {}).encode('utf-8')
    request = urllib.request.Request(
        f'{daemon_api_base()}/{path.lstrip("/")}',
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode('utf-8')
        return json.loads(body) if body else {}


def status_to_equipment(status_payload):
    cards = []

    mount = status_payload.get('mount')
    if mount:
        cards.append({
            'id': 'daemon-mount',
            'type': 'mount',
            'name': mount.get('name') or 'Local Mount',
            'site': 'Local daemon',
            'status': 'nominal',
            'tags': [
                'tracking' if mount.get('tracking') else 'tracking off',
                'parked' if mount.get('parked') else 'unparked',
                'OpenLiveStacker',
            ],
            'meta': {
                'RA': round(float(mount.get('ra') or 0), 2),
                'Dec': round(float(mount.get('dec') or 0), 2),
                'Tracking': bool(mount.get('tracking')),
                'source': 'django-daemon-relay',
            },
            'connected': True,
            'driver': 'OpenLiveStacker',
        })

    camera = status_payload.get('camera')
    if camera:
        camera_state = (
            'streaming' if camera.get('streaming')
            else 'exposing' if camera.get('exposing')
            else 'idle'
        )
        cards.append({
            'id': 'daemon-camera',
            'type': 'camera',
            'name': camera.get('name') or 'Local Camera',
            'site': 'Local daemon',
            'status': 'nominal',
            'tags': [camera_state, 'cooling' if camera.get('cooling') else 'ambient', 'OpenLiveStacker'],
            'meta': {
                'Temp': f'{float(camera.get("temperature") or 0):.1f} C',
                'Cooling': bool(camera.get('cooling')),
                'Stream': bool(camera.get('streaming')),
                'source': 'django-daemon-relay',
            },
            'connected': True,
            'driver': 'OpenLiveStacker',
        })

    focuser = status_payload.get('focuser')
    if focuser:
        cards.append({
            'id': 'daemon-focuser',
            'type': 'focuser',
            'name': focuser.get('name') or 'Local Focuser',
            'site': 'Local daemon',
            'status': 'nominal',
            'tags': ['moving' if focuser.get('moving') else 'idle', 'OpenLiveStacker'],
            'meta': {
                'Position': int(focuser.get('position') or 0),
                'Moving': bool(focuser.get('moving')),
                'source': 'django-daemon-relay',
            },
            'connected': True,
            'driver': 'OpenLiveStacker',
        })

    return cards


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
        daemon_status = fetch_daemon_status()
        if daemon_status is not None:
            return Response(status_to_equipment(daemon_status))

        # Get all equipment, ordered by type
        equipment = self.get_queryset().order_by('equipment_type', 'name')
        serializer = self.get_serializer(equipment, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'], url_path='daemon-status')
    def daemon_status(self, request):
        """
        GET /api/equipment/daemon-status/
        Browser-facing relay to the local daemon status endpoint.
        """
        daemon_status = fetch_daemon_status()
        if daemon_status is None:
            return Response(
                {'status': 'offline', 'detail': 'Local daemon is not reachable from Django.'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response(daemon_status)

    @action(detail=False, methods=['post'], url_path='camera/start-stream')
    def start_camera_stream(self, request):
        """
        POST /api/equipment/camera/start-stream/
        Browser-facing command relay to the local daemon.
        """
        try:
            return Response(post_daemon('camera/stream/start'))
        except urllib.error.HTTPError as exc:
            return Response(
                {'status': 'error', 'detail': exc.reason},
                status=exc.code,
            )
        except OSError:
            return Response(
                {'status': 'offline', 'detail': 'Local daemon is not reachable from Django.'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
    
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
