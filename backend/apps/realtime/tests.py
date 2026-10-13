import json
from datetime import timedelta
from unittest.mock import AsyncMock

from asgiref.sync import async_to_sync
from django.core.cache import cache
from django.test import TestCase, TransactionTestCase
from django.utils import timezone

from apps.equipment.views import snapshot_to_status
from apps.realtime.consumers import DaemonConsumer, TelemetryConsumer, frame_preview_key
from apps.realtime.models import DaemonConnection, TelemetrySnapshot
from apps.realtime.relay import get_daemon_telemetry, is_daemon_healthy
from apps.realtime.state import HARDWARE_FIELDS, snapshot_payload


def make_daemon(**overrides):
    values = {
        'hardware_id': 'test-rig',
        'is_connected': True,
        'last_heartbeat': timezone.now(),
        'gateway_channel_name': 'gateway.current',
    }
    values.update(overrides)
    return DaemonConnection.objects.create(**values)


def hardware_payload():
    return {
        name: {'connected': True, 'name': name, 'connection_id': f'{name}-session'}
        for name in HARDWARE_FIELDS
    }


def make_snapshot(daemon):
    return TelemetrySnapshot.objects.create(
        daemon=daemon,
        **{field: hardware_payload()[name] for name, field in HARDWARE_FIELDS.items()},
        guiding_state={'rms': 0.7},
        ai_metrics={'star_count': 12},
    )


class SnapshotFreshnessTests(TestCase):
    def setUp(self):
        cache.clear()
        self.daemon = make_daemon()
        self.snapshot = make_snapshot(self.daemon)

    def assert_no_live_devices(self, payload):
        for name in HARDWARE_FIELDS:
            self.assertIsNone(payload[name], name)
        self.assertFalse(payload['connected'])

    def test_fresh_snapshot_includes_all_four_hardware_types(self):
        payload = snapshot_to_status(self.daemon, self.snapshot)
        for name in HARDWARE_FIELDS:
            self.assertEqual(payload[name]['connection_id'], f'{name}-session')
        self.assertTrue(payload['connected'])
        self.assertTrue(payload['daemon_connected'])
        self.assertFalse(payload['telemetry_stale'])

    def test_dead_tunnel_masks_even_a_recent_hardware_snapshot(self):
        self.daemon.is_connected = False
        payload = snapshot_payload(self.daemon, self.snapshot)
        self.assert_no_live_devices(payload)
        self.assertFalse(payload['daemon_connected'])
        self.assertTrue(payload['telemetry_stale'])
        self.assertEqual(payload['guiding'], {})
        self.assertEqual(payload['ai'], {})

    def test_expired_heartbeat_masks_all_hardware(self):
        self.daemon.last_heartbeat = timezone.now() - timedelta(minutes=2)
        payload = snapshot_payload(self.daemon, self.snapshot)
        self.assert_no_live_devices(payload)
        self.assertFalse(payload['daemon_connected'])

    def test_heartbeat_alone_cannot_keep_old_hardware_live(self):
        self.snapshot.updated_at = timezone.now() - timedelta(minutes=2)
        payload = snapshot_payload(self.daemon, self.snapshot)
        self.assert_no_live_devices(payload)
        self.assertTrue(payload['daemon_connected'])
        self.assertTrue(payload['telemetry_stale'])

    def test_name_without_explicit_connection_does_not_count(self):
        for field in HARDWARE_FIELDS.values():
            setattr(self.snapshot, field, {'name': 'remembered device'})
        self.assert_no_live_devices(snapshot_payload(self.daemon, self.snapshot))

    def test_old_secondary_cache_cannot_resurrect_disconnected_devices(self):
        cache.set(f'telemetry:{self.daemon.id}', hardware_payload(), 30)
        self.daemon.is_connected = False
        self.daemon.save()
        self.assert_no_live_devices(get_daemon_telemetry(str(self.daemon.id)))

    def test_health_and_status_share_the_same_heartbeat_expiry(self):
        self.daemon.last_heartbeat = timezone.now() - timedelta(seconds=45)
        self.daemon.save()
        self.assertFalse(is_daemon_healthy(str(self.daemon.id)))
        self.assert_no_live_devices(snapshot_payload(self.daemon, self.snapshot))


class RelayLifecycleTests(TransactionTestCase):
    def setUp(self):
        cache.clear()
        self.daemon = make_daemon()
        self.snapshot = make_snapshot(self.daemon)
        self.consumer = DaemonConsumer()
        self.consumer.daemon_id = str(self.daemon.id)
        self.consumer.channel_name = 'gateway.current'
        self.consumer.group_name = f'daemon_{self.daemon.id}'
        self.consumer.channel_layer = AsyncMock()

    def test_device_disconnect_clears_identity_and_preview_cache(self):
        cache.set(frame_preview_key(self.daemon.id), {'jpeg': 'old'})
        cache.set(f'telemetry:{self.daemon.id}', hardware_payload())
        health = {'camera': {'connected': False, 'last_error': 'USB disconnected'}}
        payload = async_to_sync(self.consumer.store_telemetry_snapshot)({
            'camera': None,
            'mount': {'name': 'old mount', 'connected': False},
            'focuser': None,
            'filter_wheel': None,
            'device_health': health,
        })
        self.snapshot.refresh_from_db()
        for field in HARDWARE_FIELDS.values():
            self.assertEqual(getattr(self.snapshot, field), {})
        self.assertFalse(payload['connected'])
        self.assertTrue(payload['daemon_connected'])
        self.assertEqual(payload['device_health'], health)
        self.assertIsNone(cache.get(frame_preview_key(self.daemon.id)))
        self.assertIsNone(cache.get(f'telemetry:{self.daemon.id}'))

    def test_full_snapshot_removes_devices_missing_from_new_payload(self):
        payload = async_to_sync(self.consumer.store_telemetry_snapshot)({
            'filter_wheel': hardware_payload()['filter_wheel'],
        })
        self.assertIsNone(payload['camera'])
        self.assertIsNone(payload['mount'])
        self.assertIsNone(payload['focuser'])
        self.assertTrue(payload['filter_wheel']['connected'])

    def test_gateway_close_clears_every_device_and_broadcasts_disconnect(self):
        cache.set(frame_preview_key(self.daemon.id), {'jpeg': 'old'})
        async_to_sync(self.consumer.disconnect)(1006)
        self.daemon.refresh_from_db()
        self.snapshot.refresh_from_db()
        self.assertFalse(self.daemon.is_connected)
        for field in HARDWARE_FIELDS.values():
            self.assertEqual(getattr(self.snapshot, field), {})
        event = self.consumer.channel_layer.group_send.call_args.args[1]
        self.assertEqual(event['type'], 'telemetry_broadcast')
        self.assertFalse(event['data']['payload']['daemon_connected'])
        self.assertFalse(event['data']['payload']['connected'])
        self.assertIsNone(cache.get(frame_preview_key(self.daemon.id)))

    def test_old_socket_close_cannot_disconnect_replacement_gateway(self):
        self.consumer.channel_name = 'gateway.old'
        async_to_sync(self.consumer.disconnect)(1006)
        self.daemon.refresh_from_db()
        self.snapshot.refresh_from_db()
        self.assertTrue(self.daemon.is_connected)
        self.assertTrue(self.snapshot.camera_state['connected'])
        self.consumer.channel_layer.group_send.assert_not_called()

    def test_reconnect_waits_for_new_telemetry_instead_of_replaying_old_devices(self):
        self.consumer.channel_name = 'gateway.replacement'
        payload = async_to_sync(self.consumer.mark_daemon_connected)()
        self.daemon.refresh_from_db()
        self.assertEqual(self.daemon.gateway_channel_name, 'gateway.replacement')
        self.assertTrue(payload['daemon_connected'])
        self.assertFalse(payload['connected'])
        self.assertTrue(payload['telemetry_stale'])
        self.assertFalse(TelemetrySnapshot.objects.filter(daemon=self.daemon).exists())

    def test_late_old_gateway_telemetry_is_ignored(self):
        self.consumer.channel_name = 'gateway.old'
        self.assertIsNone(async_to_sync(self.consumer.store_telemetry_snapshot)({}))
        self.snapshot.refresh_from_db()
        self.assertTrue(self.snapshot.camera_state['connected'])

    def test_camera_replacement_clears_previous_preview(self):
        cache.set(frame_preview_key(self.daemon.id), {'jpeg': 'old camera'})
        payload = hardware_payload()
        payload['camera']['connection_id'] = 'new-camera-session'
        async_to_sync(self.consumer.store_telemetry_snapshot)(payload)
        self.assertIsNone(cache.get(frame_preview_key(self.daemon.id)))

    def test_late_preview_after_camera_disconnect_is_rejected(self):
        async_to_sync(self.consumer.store_telemetry_snapshot)({})
        self.assertFalse(async_to_sync(self.consumer.store_frame_preview)({'jpeg': 'late'}))
        self.assertIsNone(cache.get(frame_preview_key(self.daemon.id)))

    def test_new_browser_does_not_receive_a_stale_saved_camera(self):
        TelemetrySnapshot.objects.filter(pk=self.snapshot.pk).update(
            updated_at=timezone.now() - timedelta(minutes=2),
        )
        consumer = TelemetryConsumer()
        consumer.daemon_id = str(self.daemon.id)
        event = async_to_sync(consumer.get_latest_telemetry)()
        self.assertEqual(event['type'], 'TELEMETRY_UPDATE')
        self.assertTrue(event['payload']['telemetry_stale'])
        for name in HARDWARE_FIELDS:
            self.assertIsNone(event['payload'][name])

    def test_broadcast_sends_canonical_disconnected_snapshot(self):
        async_to_sync(self.consumer.receive)(json.dumps({
            'type': 'TELEMETRY_UPDATE', 'payload': dict.fromkeys(HARDWARE_FIELDS),
        }))
        event = self.consumer.channel_layer.group_send.call_args.args[1]
        self.assertFalse(event['data']['payload']['connected'])
        for name in HARDWARE_FIELDS:
            self.assertIsNone(event['data']['payload'][name])
