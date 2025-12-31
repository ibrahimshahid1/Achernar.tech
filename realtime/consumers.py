"""
Django Channels WebSocket Consumers

Handles WebSocket connections for:
1. Daemon → Cloud persistent connection (IoT relay)
2. Web Client → Cloud telemetry subscription

PDF Section 16.2.2 - The Control Plane (Command & Telemetry)
"""
import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.utils import timezone

logger = logging.getLogger(__name__)


class DaemonConsumer(AsyncWebsocketConsumer):
    """
    Handles persistent WebSocket connection from Rust daemon.
    
    The daemon initiates an outbound connection that stays open,
    creating a reverse tunnel for commands.
    """
    
    async def connect(self):
        """Accept daemon connection and register in database"""
        # TODO: Authenticate daemon via JWT token
        self.daemon_id = self.scope['url_route']['kwargs'].get('daemon_id')
        
        if not self.daemon_id:
            await self.close()
            return
        
        # Add to daemon group for broadcasting
        self.group_name = f'daemon_{self.daemon_id}'
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        
        await self.accept()
        
        # Mark daemon as connected in database
        await self.mark_daemon_connected()
        
        logger.info(f"Daemon {self.daemon_id} connected")
    
    async def disconnect(self, close_code):
        """Handle daemon disconnection"""
        await self.channel_layer.group_discard(self.group_name, self.channel_name)
        await self.mark_daemon_disconnected()
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
                await self.store_telemetry_snapshot(data['payload'])
                
                # Broadcast to all connected web clients for this daemon
                await self.channel_layer.group_send(
                    f'daemon_{self.daemon_id}_clients',
                    {
                        'type': 'telemetry_broadcast',
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
    
    @database_sync_to_async
    def mark_daemon_connected(self):
        """Update daemon connection status in database"""
        from .models import DaemonConnection
        try:
            daemon = DaemonConnection.objects.get(id=self.daemon_id)
            daemon.mark_connected()
        except DaemonConnection.DoesNotExist:
            logger.warning(f"Daemon {self.daemon_id} not found in database")
    
    @database_sync_to_async
    def mark_daemon_disconnected(self):
        """Update daemon disconnection status"""
        from .models import DaemonConnection
        try:
            daemon = DaemonConnection.objects.get(id=self.daemon_id)
            daemon.mark_disconnected()
        except DaemonConnection.DoesNotExist:
            pass
    
    @database_sync_to_async
    def update_daemon_heartbeat(self):
        """Update last heartbeat timestamp"""
        from .models import DaemonConnection
        try:
            daemon = DaemonConnection.objects.get(id=self.daemon_id)
            daemon.update_heartbeat()
        except DaemonConnection.DoesNotExist:
            pass
    
    @database_sync_to_async
    def store_telemetry_snapshot(self, payload):
        """Store latest telemetry in database (Digital Twin)"""
        from .models import DaemonConnection, TelemetrySnapshot
        try:
            daemon = DaemonConnection.objects.get(id=self.daemon_id)
            snapshot, created = TelemetrySnapshot.objects.get_or_create(daemon=daemon)
            
            # Update individual state components
            snapshot.mount_state = payload.get('mount', {})
            snapshot.camera_state = payload.get('camera', {})
            snapshot.guiding_state = payload.get('guiding', {})
            snapshot.ai_metrics = payload.get('ai', {})
            snapshot.save()
        
        except DaemonConnection.DoesNotExist:
            pass


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
        await self.send_latest_telemetry()
        
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
    
    @database_sync_to_async
    def send_latest_telemetry(self):
        """Send latest telemetry snapshot to newly connected client"""
        from .models import DaemonConnection, TelemetrySnapshot
        try:
            daemon = DaemonConnection.objects.get(id=self.daemon_id)
            snapshot = TelemetrySnapshot.objects.filter(daemon=daemon).first()
            
            if snapshot:
                telemetry_data = {
                    'type': 'TELEMETRY_UPDATE',
                    'timestamp': int(snapshot.updated_at.timestamp()),
                    'payload': {
                        'mount': snapshot.mount_state,
                        'camera': snapshot.camera_state,
                        'guiding': snapshot.guiding_state,
                        'ai': snapshot.ai_metrics,
                    }
                }
                # Use async send
                import asyncio
                asyncio.create_task(
                    self.send(text_data=json.dumps(telemetry_data))
                )
        
        except DaemonConnection.DoesNotExist:
            pass
