from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny  # For now, no auth required for local testing
from django.utils import timezone
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.core.cache import cache
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from .models import ConnectedEquipment
from .serializers import ConnectedEquipmentSerializer
from apps.realtime.models import DaemonConnection, TelemetrySnapshot


COMMAND_WAIT_SECONDS = 3.0
COMMAND_POLL_SECONDS = 0.1
COMMAND_RESULT_TTL_SECONDS = 60


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


def command_result_key(command_id):
    return f"achernar:command_result:{command_id}"


def frame_preview_key(daemon_id):
    return f"achernar:frame_preview:{daemon_id}"


def active_daemon(daemon_id=None):
    qs = DaemonConnection.objects.all().order_by('-last_heartbeat', '-connected_at', '-created_at')
    if daemon_id:
        return qs.filter(id=daemon_id).first()
    return qs.filter(is_connected=True).first() or qs.first()


def snapshot_to_status(daemon, snapshot=None):
    snapshot = snapshot or TelemetrySnapshot.objects.filter(daemon=daemon).first()
    payload = {
        'connected': bool(daemon and daemon.is_connected),
        'daemon_id': str(daemon.id) if daemon else None,
        'hardware_id': daemon.hardware_id if daemon else None,
        'last_heartbeat': daemon.last_heartbeat.isoformat() if daemon and daemon.last_heartbeat else None,
        'camera': None,
        'mount': None,
        'focuser': None,
        'guiding': {},
        'ai': {},
    }
    if snapshot:
        payload.update({
            'camera': snapshot.camera_state or None,
            'mount': snapshot.mount_state or None,
            'focuser': snapshot.focuser_state or None,
            'guiding': snapshot.guiding_state,
            'ai': snapshot.ai_metrics,
            'updated_at': snapshot.updated_at.isoformat(),
        })
    return payload


def send_relay_command(daemon, action, payload=None, wait_seconds=COMMAND_WAIT_SECONDS):
    if not daemon or not daemon.is_connected:
        return {
            'status_code': status.HTTP_503_SERVICE_UNAVAILABLE,
            'body': {
                'type': 'COMMAND_RESULT',
                'ok': False,
                'error': 'No connected daemon is available for relay commands.',
            },
        }

    channel_layer = get_channel_layer()
    if channel_layer is None:
        return {
            'status_code': status.HTTP_503_SERVICE_UNAVAILABLE,
            'body': {
                'type': 'COMMAND_RESULT',
                'ok': False,
                'error': 'Django Channels layer is not configured.',
            },
        }

    command_id = str(uuid.uuid4())
    cache.delete(command_result_key(command_id))
    message = {
        'type': 'COMMAND',
        'command_id': command_id,
        'action': action,
        'payload': payload or {},
    }
    async_to_sync(channel_layer.group_send)(
        f'daemon_{daemon.id}',
        {
            'type': 'command_relay',
            'data': message,
        },
    )

    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    while time.monotonic() < deadline:
        result = cache.get(command_result_key(command_id))
        if result is not None:
            result.setdefault('daemon_id', str(daemon.id))
            return {
                'status_code': status.HTTP_200_OK if result.get('ok') else status.HTTP_502_BAD_GATEWAY,
                'body': result,
            }
        time.sleep(COMMAND_POLL_SECONDS)

    cache.set(command_result_key(command_id), {'pending': True, 'command_id': command_id}, COMMAND_RESULT_TTL_SECONDS)
    return {
        'status_code': status.HTTP_202_ACCEPTED,
        'body': {
            'type': 'COMMAND_RESULT',
            'command_id': command_id,
            'daemon_id': str(daemon.id),
            'ok': None,
            'pending': True,
        },
    }


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

    @action(detail=False, methods=['get'], url_path='daemons')
    def daemons(self, request):
        """
        GET /api/equipment/daemons/
        Lists registered daemons for browser-side relay selection.
        """
        daemons = DaemonConnection.objects.all().order_by('-is_connected', '-last_heartbeat', '-created_at')
        return Response([
            {
                'id': str(daemon.id),
                'hardware_id': daemon.hardware_id,
                'daemon_version': daemon.daemon_version,
                'site_name': daemon.site_name,
                'is_connected': daemon.is_connected,
                'last_heartbeat': daemon.last_heartbeat.isoformat() if daemon.last_heartbeat else None,
            }
            for daemon in daemons
        ])
    
    @action(detail=False, methods=['get'], url_path='connected')
    def connected(self, request):
        """
        GET /api/equipment/connected/
        Returns currently connected equipment for dashboard display.
        This is the main endpoint the frontend will call.
        """
        daemon = active_daemon(request.query_params.get('daemon_id'))
        if daemon and daemon.is_connected:
            return Response(status_to_equipment(snapshot_to_status(daemon)))

        # Local development fallback when Django and daemon share a host.
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
        daemon = active_daemon(request.query_params.get('daemon_id'))
        if daemon and daemon.is_connected:
            return Response(snapshot_to_status(daemon))

        # Local development fallback when the reverse tunnel is not connected.
        daemon_status = fetch_daemon_status()
        if daemon_status is None:
            return Response(
                {'status': 'offline', 'detail': 'No connected daemon tunnel or localhost daemon is reachable.'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response(daemon_status)

    @action(detail=False, methods=['post'], url_path='camera/start-stream')
    def start_camera_stream(self, request):
        """
        POST /api/equipment/camera/start-stream/
        Browser-facing command relay to the local daemon.
        """
        daemon = active_daemon(request.data.get('daemon_id'))
        if daemon and daemon.is_connected:
            result = send_relay_command(daemon, 'camera.start_stream', request.data.get('payload', {}))
            body = result['body']
            if body.get('ok') and isinstance(body.get('payload'), dict):
                return Response(body['payload'], status=result['status_code'])
            return Response(body, status=result['status_code'])

        return self._local_post_fallback('camera/stream/start', {})

    @action(detail=False, methods=['post'], url_path='camera/snapshot')
    def camera_snapshot(self, request):
        """
        POST /api/equipment/camera/snapshot/
        Requests the daemon's latest low-rate preview frame through the relay.
        """
        daemon = active_daemon(request.data.get('daemon_id'))
        if daemon and daemon.is_connected:
            result = send_relay_command(daemon, 'camera.snapshot', request.data.get('payload', {}))
            return Response(result['body'], status=result['status_code'])
        return Response(
            {'status': 'offline', 'detail': 'No connected daemon tunnel is available for snapshots.'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @action(detail=False, methods=['post'], url_path='command')
    def command(self, request):
        """
        POST /api/equipment/command/
        Generic relay command endpoint.
        """
        action_name = request.data.get('action')
        if not action_name:
            return Response(
                {'status': 'error', 'detail': 'action is required.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        daemon = active_daemon(request.data.get('daemon_id'))
        result = send_relay_command(
            daemon,
            action_name,
            request.data.get('payload', {}),
            request.data.get('wait_seconds', COMMAND_WAIT_SECONDS),
        )
        return Response(result['body'], status=result['status_code'])

    def _local_post_fallback(self, path, payload):
        try:
            return Response(post_daemon(path, payload))
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
