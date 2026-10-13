"""
Django Channels WebSocket Consumers

Handles WebSocket connections for:
1. Daemon → Cloud persistent connection (IoT relay)
2. Web Client → Cloud telemetry subscription

PDF Section 16.2.2 - The Control Plane (Command & Telemetry)
"""
import json
import logging
import os
from urllib.parse import parse_qs
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.core.cache import cache
from django.db import transaction
from django.utils import timezone
from .state import HARDWARE_FIELDS, connected_hardware, snapshot_payload

logger = logging.getLogger(__name__)

DEFAULT_RELAY_DEV_TOKEN = "achernar-dev-token"
COMMAND_RESULT_TTL_SECONDS = 60


def expected_relay_token():
    return os.getenv("ACHERNAR_RELAY_DEV_TOKEN", DEFAULT_RELAY_DEV_TOKEN)


def command_result_key(command_id):
    return f"achernar:command_result:{command_id}"


def frame_preview_key(daemon_id):
    return f"achernar:frame_preview:{daemon_id}"


def clear_snapshot(snapshot):
    for field in HARDWARE_FIELDS.values():
        setattr(snapshot, field, {})
    snapshot.guiding_state = {}
    snapshot.ai_metrics = {}
    snapshot.device_health = {}
    snapshot.save()


def clear_cached_telemetry(daemon_id):
    cache.delete_many([frame_preview_key(daemon_id), f'telemetry:{daemon_id}'])


class DaemonConsumer(AsyncWebsocketConsumer):
    """
    Handles persistent WebSocket connection from Rust daemon.
    
    The daemon initiates an outbound connection that stays open,
    creating a reverse tunnel for commands.
    """
    
    async def connect(self):
        """Accept daemon connection and register in database"""
        self.daemon_id = self.scope['url_route']['kwargs'].get('daemon_id')
        
        if not self.daemon_id:
            await self.close()
            return

        query = parse_qs(self.scope.get("query_string", b"").decode("utf-8"))
        token = (query.get("token") or query.get("dev_token") or [""])[0]
        if token != expected_relay_token():
            logger.warning("Rejected daemon gateway connection with invalid token")
            await self.close(code=4401)
            return

        if not await self.daemon_exists():
            logger.warning(f"Rejected unknown daemon gateway connection: {self.daemon_id}")
            await self.close(code=4404)
            return
        
        # Add to daemon group for broadcasting
        self.group_name = f'daemon_{self.daemon_id}'
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        
        await self.accept()
        
        # Mark daemon as connected in database
        payload = await self.mark_daemon_connected()
        if payload is not None:
            await self.broadcast_telemetry(payload)
        
        logger.info(f"Daemon {self.daemon_id} connected")
    
    async def disconnect(self, close_code):
        """Handle daemon disconnection"""
        if hasattr(self, 'group_name'):
            await self.channel_layer.group_discard(self.group_name, self.channel_name)
        payload = await self.mark_daemon_disconnected()
        if payload is not None:
            await self.broadcast_telemetry(payload)
        logger.info(f"Daemon {self.daemon_id} disconnected with code {close_code}")
    
    async def receive(self, text_data):
        """
        Receive telemetry updates from daemon.
        
        Expected message format:
        {
            "type": "TELEMETRY_UPDATE",
            "timestamp": 1234567890,
            "payload": {
                "mount": {...},
                "camera": {...},
                "guiding": {...},
                "ai": {...}
            }
        }
        """
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'TELEMETRY_UPDATE':
                # Update heartbeat
                await self.update_daemon_heartbeat()
                
                # Store telemetry snapshot
                payload = await self.store_telemetry_snapshot(data['payload'])
                if payload is None:
                    return
                data['payload'] = payload
                
                # Broadcast to all connected web clients for this daemon
                await self.channel_layer.group_send(
                    f'daemon_{self.daemon_id}_clients',
                    {
                        'type': 'telemetry_broadcast',
                        'data': data
                    }
                )

            elif message_type == 'HELLO':
                await self.update_daemon_metadata(data)
                await self.update_daemon_heartbeat()
                await self.send(text_data=json.dumps({
                    "type": "HELLO_ACK",
                    "timestamp": timezone.now().isoformat(),
                    "daemon_id": self.daemon_id,
                }))

            elif message_type == 'COMMAND_RESULT':
                command_id = data.get("command_id")
                if command_id:
                    await self.store_command_result(command_id, data)
                    await self.channel_layer.group_send(
                        f'daemon_{self.daemon_id}_clients',
                        {
                            'type': 'command_result',
                            'data': data
                        }
                    )

            elif message_type == 'FRAME_PREVIEW':
                if not await self.store_frame_preview(data):
                    return
                await self.channel_layer.group_send(
                    f'daemon_{self.daemon_id}_clients',
                    {
                        'type': 'frame_preview',
                        'data': data
                    }
                )
            
            elif message_type == 'HEARTBEAT':
                await self.update_daemon_heartbeat()
        
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from daemon {self.daemon_id}")
        except Exception as e:
            logger.error(f"Error processing daemon message: {e}")
    
    async def command_relay(self, event):
        """
        Relay command from web client to daemon.
        
        Called when a web client sends a command that needs to reach the daemon.
        """
        await self.send(text_data=json.dumps(event['data']))

    async def broadcast_telemetry(self, payload):
        await self.channel_layer.group_send(
            f'daemon_{self.daemon_id}_clients',
            {
                'type': 'telemetry_broadcast',
                'data': {
                    'type': 'TELEMETRY_UPDATE',
                    'timestamp': int(timezone.now().timestamp()),
                    'payload': payload,
                },
            },
        )

    @database_sync_to_async
    def daemon_exists(self):
        from .models import DaemonConnection
        return DaemonConnection.objects.filter(id=self.daemon_id).exists()
    
    @database_sync_to_async
    def mark_daemon_connected(self):
        """Update daemon connection status in database"""
        from .models import DaemonConnection, TelemetrySnapshot
        try:
            with transaction.atomic():
                daemon = DaemonConnection.objects.select_for_update().get(id=self.daemon_id)
                daemon.mark_connected(self.channel_name)
                # A new gateway must supply its own first hardware snapshot.
                TelemetrySnapshot.objects.filter(daemon=daemon).delete()
                clear_cached_telemetry(self.daemon_id)
                return snapshot_payload(daemon)
        except DaemonConnection.DoesNotExist:
            logger.warning(f"Daemon {self.daemon_id} not found in database")

    @database_sync_to_async
    def update_daemon_metadata(self, data):
        """Update daemon version/hardware metadata from HELLO."""
        from .models import DaemonConnection
        try:
            daemon = DaemonConnection.objects.get(id=self.daemon_id)
            daemon.daemon_version = data.get("daemon_version", daemon.daemon_version)
            daemon.hardware_id = data.get("hardware_id", daemon.hardware_id)
            daemon.site_name = data.get("site_name", daemon.site_name)
            daemon.save(update_fields=["daemon_version", "hardware_id", "site_name", "updated_at"])
        except DaemonConnection.DoesNotExist:
            pass
    
    @database_sync_to_async
    def mark_daemon_disconnected(self):
        """Update daemon disconnection status"""
        from .models import DaemonConnection, TelemetrySnapshot
        try:
            with transaction.atomic():
                daemon = DaemonConnection.objects.select_for_update().get(id=self.daemon_id)
                # An old socket can close after its replacement has connected.
                # Only the socket that owns the current session may clear it.
                if daemon.gateway_channel_name != self.channel_name:
                    return None
                daemon.gateway_channel_name = ''
                daemon.mark_disconnected()
                snapshot, _ = TelemetrySnapshot.objects.get_or_create(daemon=daemon)
                clear_snapshot(snapshot)
                clear_cached_telemetry(self.daemon_id)
                return snapshot_payload(daemon, snapshot)
        except DaemonConnection.DoesNotExist:
            pass
    
    @database_sync_to_async
    def update_daemon_heartbeat(self):
        """Update last heartbeat timestamp"""
        from .models import DaemonConnection
        DaemonConnection.objects.filter(
            id=self.daemon_id, gateway_channel_name=self.channel_name, is_connected=True,
        ).update(last_heartbeat=timezone.now())
    
    @database_sync_to_async
    def store_telemetry_snapshot(self, payload):
        """Store latest telemetry in database (Digital Twin)"""
        from .models import DaemonConnection, TelemetrySnapshot
        try:
            with transaction.atomic():
                daemon = DaemonConnection.objects.select_for_update().get(id=self.daemon_id)
                if not daemon.is_connected or daemon.gateway_channel_name != self.channel_name:
                    return None
                snapshot, _ = TelemetrySnapshot.objects.get_or_create(daemon=daemon)
                previous_camera = snapshot.camera_state
                # TELEMETRY_UPDATE is a complete snapshot. Missing, null or
                # explicitly disconnected devices remove their old identity.
                for name, field in HARDWARE_FIELDS.items():
                    setattr(snapshot, field, connected_hardware(payload.get(name)) or {})
                snapshot.guiding_state = payload.get('guiding') or {}
                snapshot.ai_metrics = payload.get('ai') or {}
                snapshot.device_health = payload.get('device_health') or {}
                snapshot.save()
                cache.delete(f'telemetry:{self.daemon_id}')
                if (
                    not snapshot.camera_state
                    or previous_camera.get('connection_id') != snapshot.camera_state.get('connection_id')
                ):
                    cache.delete(frame_preview_key(self.daemon_id))
                return snapshot_payload(daemon, snapshot)
        
        except DaemonConnection.DoesNotExist:
            pass

    @database_sync_to_async
    def store_command_result(self, command_id, data):
        cache.set(command_result_key(command_id), data, COMMAND_RESULT_TTL_SECONDS)

    @database_sync_to_async
    def store_frame_preview(self, data):
        from .models import DaemonConnection, TelemetrySnapshot
        with transaction.atomic():
            daemon = DaemonConnection.objects.select_for_update().filter(
                id=self.daemon_id, gateway_channel_name=self.channel_name, is_connected=True,
            ).first()
            snapshot = TelemetrySnapshot.objects.filter(daemon=daemon).first() if daemon else None
            if not snapshot_payload(daemon, snapshot)['camera']:
                return False
            cache.set(frame_preview_key(self.daemon_id), data, 15)
            return True


class TelemetryConsumer(AsyncWebsocketConsumer):
    """
    Handles WebSocket connection from web clients for real-time telemetry.
    
    Clients subscribe to telemetry from a specific daemon.
    """
    
    async def connect(self):
        """Accept client connection and subscribe to daemon telemetry"""
        # TODO: Authenticate user
        self.daemon_id = self.scope['url_route']['kwargs'].get('daemon_id')
        
        if not self.daemon_id:
            await self.close()
            return
        
        # Add to client group for this daemon
        self.group_name = f'daemon_{self.daemon_id}_clients'
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        
        await self.accept()
        
        # Send latest telemetry snapshot immediately
        telemetry_data = await self.get_latest_telemetry()
        if telemetry_data:
            await self.send(text_data=json.dumps(telemetry_data))
        
        logger.info(f"Client connected to daemon {self.daemon_id} telemetry")
    
    async def disconnect(self, close_code):
        """Handle client disconnection"""
        await self.channel_layer.group_discard(self.group_name, self.channel_name)
        logger.info(f"Client disconnected from daemon {self.daemon_id}")
    
    async def receive(self, text_data):
        """
        Receive commands from web client to send to daemon.
        
        Expected format:
        {
            "type": "COMMAND",
            "action": "camera/capture",
            "payload": {...}
        }
        """
        try:
            data = json.loads(text_data)
            
            if data.get('type') == 'COMMAND':
                # Relay command to daemon
                await self.channel_layer.group_send(
                    f'daemon_{self.daemon_id}',
                    {
                        'type': 'command_relay',
                        'data': data
                    }
                )
        
        except json.JSONDecodeError:
            logger.error("Invalid JSON from web client")
        except Exception as e:
            logger.error(f"Error processing client message: {e}")
    
    async def telemetry_broadcast(self, event):
        """
        Receive telemetry from daemon and send to web client.
        
        Called when daemon sends telemetry update.
        """
        await self.send(text_data=json.dumps(event['data']))

    async def command_result(self, event):
        await self.send(text_data=json.dumps(event['data']))

    async def frame_preview(self, event):
        await self.send(text_data=json.dumps(event['data']))
    
    @database_sync_to_async
    def get_latest_telemetry(self):
        """Return latest telemetry snapshot for a newly connected client."""
        from .models import DaemonConnection, TelemetrySnapshot
        try:
            daemon = DaemonConnection.objects.get(id=self.daemon_id)
            snapshot = TelemetrySnapshot.objects.filter(daemon=daemon).first()
            
            if daemon:
                telemetry_data = {
                    'type': 'TELEMETRY_UPDATE',
                    'timestamp': int(timezone.now().timestamp()),
                    'payload': snapshot_payload(daemon, snapshot),
                }
                return telemetry_data
        
        except DaemonConnection.DoesNotExist:
            pass
        return None
