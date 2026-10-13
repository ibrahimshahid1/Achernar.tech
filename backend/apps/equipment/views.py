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
from apps.realtime.state import daemon_is_live, snapshot_payload


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


def get_daemon(path, timeout=5):
    with urllib.request.urlopen(f'{daemon_api_base()}/{path.lstrip("/")}', timeout=timeout) as response:
        body = response.read().decode('utf-8')
        return json.loads(body) if body else {}


# The daemon publishes its own command registry at GET /api/v1/commands,
# generated from the same declaration its dispatcher matches on. Django
# reads that rather than keeping a copy.
#
# There used to be a hand-written whitelist here. It was missing five
# commands the daemon handled, and it disagreed with the daemon on what
# `camera.abort` meant. A copy that nothing forces to agree will drift; the
# fix is to stop copying, not to test the copy harder.
COMMAND_REGISTRY_TTL_SECONDS = int(os.getenv('ACHERNAR_COMMAND_REGISTRY_TTL', '60'))
_REGISTRY_CACHE = {'base': None, 'fetched_at': 0.0, 'commands': None}

# Transport quirks that are Django's business, not the contract's: the
# daemon takes the stream format as a query parameter, and a capture needs
# a read timeout long enough for the exposure itself.
_QUERY_PARAM_COMMANDS = {'camera.start_stream': 'format_index'}


def daemon_command_registry(force=False):
    """Return {action: {method, path, description}} from the live daemon.

    None when the daemon cannot be reached — in which case the command was
    going to fail anyway, and saying so is better than guessing from a
    stale local list.
    """
    base = daemon_api_base()
    now = time.monotonic()
    fresh = (
        _REGISTRY_CACHE['commands'] is not None
        and _REGISTRY_CACHE['base'] == base
        and now - _REGISTRY_CACHE['fetched_at'] < COMMAND_REGISTRY_TTL_SECONDS
    )
    if fresh and not force:
        return _REGISTRY_CACHE['commands']

    try:
        published = get_daemon('commands', timeout=2)
    except (OSError, urllib.error.HTTPError, ValueError):
        return None

    commands = {
        entry['name']: entry
        for entry in published
        if isinstance(entry, dict) and entry.get('name') and entry.get('path')
    }
    _REGISTRY_CACHE.update({'base': base, 'fetched_at': now, 'commands': commands})
    return commands


def command_result_key(command_id):
    return f"achernar:command_result:{command_id}"


def frame_preview_key(daemon_id):
    return f"achernar:frame_preview:{daemon_id}"


def active_daemon(daemon_id=None):
    qs = DaemonConnection.objects.all().order_by('-last_heartbeat', '-connected_at', '-created_at')
    if daemon_id:
        daemon = qs.filter(id=daemon_id).first()
        return daemon if daemon_is_live(daemon) else None
    for daemon in qs.filter(is_connected=True):
        if daemon_is_live(daemon):
            return daemon
    return None


def snapshot_to_status(daemon, snapshot=None):
    snapshot = snapshot or TelemetrySnapshot.objects.filter(daemon=daemon).first()
    payload = {
        'daemon_id': str(daemon.id) if daemon else None,
        'hardware_id': daemon.hardware_id if daemon else None,
        'last_heartbeat': daemon.last_heartbeat.isoformat() if daemon and daemon.last_heartbeat else None,
    }
    payload.update(snapshot_payload(daemon, snapshot))
    return payload


def send_relay_command(daemon, action, payload=None, wait_seconds=COMMAND_WAIT_SECONDS):
    if not daemon_is_live(daemon):
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
                'is_connected': daemon_is_live(daemon),
                'last_heartbeat': daemon.last_heartbeat.isoformat() if daemon.last_heartbeat else None,
            }
            for daemon in daemons
        ])
    
    @action(detail=False, methods=['get'], url_path='connected')
    def connected(self, request):
        """
        GET /api/equipment/connected/

        Returns the daemon's own status payload verbatim, plus a `source`
        saying where it came from. Django no longer reshapes it into UI cards:
        `status_to_equipment()` was a second copy of `cardsFromDaemon()` in the
        frontend's useEquipment.ts — presentation logic in Python, on the far
        side of a network hop, where the frontend could not see it and no test
        covered it. The two drifted on how an unmeasured value renders, which
        is the one thing that had to agree.

        `source` is one of:
          relay     — a live daemon over the WebSocket tunnel
          localhost — a daemon reachable directly from this host
          cache     — nothing reachable; rows sync last recorded, whose
                      `connected` is derived from `last_seen`, never trusted
          none      — nothing reachable and nothing recorded
        """
        daemon = active_daemon(request.query_params.get('daemon_id'))
        if daemon and daemon.is_connected:
            return Response({'source': 'relay', 'status': snapshot_to_status(daemon), 'cached': []})

        # Local development fallback when Django and daemon share a host.
        daemon_status = fetch_daemon_status()
        if daemon_status is not None:
            return Response({'source': 'localhost', 'status': daemon_status, 'cached': []})

        # No daemon is reachable by either route.
        #
        # This used to return whatever sat in ConnectedEquipment, verbatim and
        # unqualified — which meant four hand-created fixture rows ("ASI6200MM
        # Pro", "AP 1100GTO", ...) came back with connected: true and nothing
        # marking them as fake, six months after they were written. A plausible
        # value standing in for an unknown one.
        #
        # The rows are still worth returning: they say what hardware this site
        # has. Their connection state is not. The serializer derives `connected`
        # from `last_seen` and ships `stale` and `last_seen` alongside, so a
        # consumer can tell "connected" from "last seen in February".
        equipment = self.get_queryset().order_by('equipment_type', 'name')
        serializer = self.get_serializer(
            equipment, many=True,
            context={**self.get_serializer_context(), 'live_source_available': False},
        )
        return Response({
            'source': 'cache' if serializer.data else 'none',
            'status': None,
            'cached': serializer.data,
        })

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

        payload = request.data.get('payload', {})
        fmt = payload.get('format_index')
        path = f'camera/stream/start?format_index={fmt}' if fmt is not None else 'camera/stream/start'
        return self._local_post_fallback(path, {})

    @action(detail=False, methods=['post'], url_path='camera/snapshot')
    def camera_snapshot(self, request):
        """
        POST /api/equipment/camera/snapshot/
        Requests the daemon's latest low-rate preview frame through the relay.
        """
        daemon = active_daemon(request.data.get('daemon_id'))
        if daemon and daemon.is_connected:
            result = send_relay_command(daemon, 'camera.snapshot', request.data.get('payload', {}))
            body = result['body']
            if body.get('ok') and isinstance(body.get('payload'), dict):
                return Response(body['payload'], status=result['status_code'])
            return Response(body, status=result['status_code'])
        return self._local_post_fallback('camera/snapshot', {})

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
        if daemon and daemon.is_connected:
            result = send_relay_command(
                daemon,
                action_name,
                request.data.get('payload', {}),
                request.data.get('wait_seconds', COMMAND_WAIT_SECONDS),
            )
            return Response(result['body'], status=result['status_code'])

        registry = daemon_command_registry()
        if registry is None:
            # We could not even ask what this daemon supports.
            return Response(
                {
                    'type': 'COMMAND_RESULT',
                    'ok': False,
                    'error': 'No daemon is reachable, by relay or on localhost.',
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        if action_name in registry:
            return self._local_command_fallback(
                action_name,
                registry[action_name],
                request.data.get('payload', {}),
            )

        # An unsupported verb and an absent daemon are different failures. This
        # used to fall through to send_relay_command() with daemon=None, which
        # answered "No connected daemon is available for relay commands." —
        # naming the wrong cause, while the daemon was up and answering.
        return Response(
            {
                'type': 'COMMAND_RESULT',
                'ok': False,
                'error': f'Unsupported action: {action_name}',
                'supported_actions': sorted(registry),
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

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

    def _local_command_fallback(self, action_name, route, payload):
        """Translate one relay action into the daemon's own HTTP route.

        `route` is the daemon's published entry for this action, so the method
        and path are whatever this build actually serves. Only the transport
        quirks below are Django's own.
        """
        payload = payload or {}
        method = (route.get('method') or 'POST').upper()
        path = route['path']

        try:
            if method == 'GET':
                body = get_daemon(path)
            elif action_name in _QUERY_PARAM_COMMANDS:
                # The daemon takes this argument as a query parameter, not a body.
                key = _QUERY_PARAM_COMMANDS[action_name]
                value = payload.get(key)
                body = post_daemon(
                    f'{path}?{key}={value}' if value is not None else path,
                    {},
                )
            elif action_name == 'camera.capture':
                # Give the request room for the exposure itself plus overhead.
                exposure = float(payload.get('exposure', 1))
                body = post_daemon(path, payload, timeout=max(10, min(140, int(exposure) + 10)))
            else:
                # Commands that take no arguments ignore the body, so there is
                # nothing to gain from tracking which ones those are.
                body = post_daemon(path, payload)
            return Response({
                'type': 'COMMAND_RESULT',
                'ok': True,
                'pending': False,
                'payload': body,
            })
        except urllib.error.HTTPError as exc:
            return Response(
                {
                    'type': 'COMMAND_RESULT',
                    'ok': False,
                    'error': exc.reason,
                },
                status=exc.code,
            )
        except (OSError, ValueError):
            return Response(
                {
                    'type': 'COMMAND_RESULT',
                    'ok': False,
                    'error': 'Local daemon is not reachable from Django.',
                },
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
