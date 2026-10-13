"""Contract tests for the daemon command relay.

The daemon's HTTP API is the only real contract in this system, and it now
publishes its own command registry at `GET /api/v1/commands`, generated from
the same declaration its dispatcher matches on.

Django used to keep a hand-written whitelist. It was missing five commands the
daemon handled, and it disagreed with the daemon on what `camera.abort` meant.
These tests cover what replaced it: that Django reads the published registry,
routes by it, caches it sanely, and fails honestly when it cannot be reached.

The registry's own invariants (unique names, relative paths, valid methods)
are tested in Rust, next to the declaration, in relay_commands.rs.
"""

from unittest import mock

from django.test import TestCase

from apps.equipment import views


def registry_payload():
    """A realistic slice of what GET /api/v1/commands returns."""
    return [
        {'name': 'equipment.status', 'method': 'GET', 'path': 'status',
         'description': 'Connection state and current readings.'},
        {'name': 'camera.capture', 'method': 'POST', 'path': 'camera/capture',
         'description': 'Take one exposure and write a FITS frame.'},
        {'name': 'camera.start_stream', 'method': 'POST', 'path': 'camera/stream/start',
         'description': 'Start the live view.'},
        {'name': 'camera.abort', 'method': 'POST', 'path': 'camera/abort',
         'description': 'Stop the live view. Idempotent.'},
        {'name': 'focuser.position', 'method': 'GET', 'path': 'focuser/position',
         'description': 'Current focuser position, or null.'},
    ]


class CommandRegistryTests(TestCase):
    def setUp(self):
        # The cache is module state; a stale entry would leak between tests.
        views._REGISTRY_CACHE.update({'base': None, 'fetched_at': 0.0, 'commands': None})

    def test_registry_is_keyed_by_action_name(self):
        with mock.patch.object(views, 'get_daemon', return_value=registry_payload()):
            registry = views.daemon_command_registry(force=True)

        self.assertIn('camera.capture', registry)
        self.assertEqual(registry['camera.capture']['path'], 'camera/capture')
        self.assertEqual(registry['equipment.status']['method'], 'GET')

    def test_unreachable_daemon_yields_none_not_a_guess(self):
        """A stale local list would be worse than admitting we cannot ask."""
        with mock.patch.object(views, 'get_daemon', side_effect=OSError('refused')):
            self.assertIsNone(views.daemon_command_registry(force=True))

    def test_registry_is_cached_between_calls(self):
        with mock.patch.object(views, 'get_daemon', return_value=registry_payload()) as fetch:
            views.daemon_command_registry(force=True)
            views.daemon_command_registry()
            views.daemon_command_registry()

        self.assertEqual(fetch.call_count, 1, 'registry refetched inside its TTL')

    def test_force_bypasses_the_cache(self):
        with mock.patch.object(views, 'get_daemon', return_value=registry_payload()) as fetch:
            views.daemon_command_registry(force=True)
            views.daemon_command_registry(force=True)

        self.assertEqual(fetch.call_count, 2)

    def test_cache_is_invalidated_when_the_daemon_address_changes(self):
        """Pointing at a different rig must not reuse the first rig's registry."""
        with mock.patch.object(views, 'get_daemon', return_value=registry_payload()) as fetch:
            with mock.patch.object(views, 'daemon_api_base', return_value='http://rig-a/api/v1'):
                views.daemon_command_registry(force=True)
            with mock.patch.object(views, 'daemon_api_base', return_value='http://rig-b/api/v1'):
                views.daemon_command_registry()

        self.assertEqual(fetch.call_count, 2)

    def test_malformed_entries_are_dropped_rather_than_half_trusted(self):
        payload = registry_payload() + [
            {'name': 'broken.no_path', 'method': 'POST'},
            {'method': 'GET', 'path': 'orphan'},
            'not a dict',
        ]
        with mock.patch.object(views, 'get_daemon', return_value=payload):
            registry = views.daemon_command_registry(force=True)

        self.assertNotIn('broken.no_path', registry)
        self.assertEqual(len(registry), len(registry_payload()))


# ─────────────────────────────────────────────────────────────────────────
#  Null-state honesty, Django's share of it
#
#  Card construction moved to the frontend when status_to_equipment() was
#  deleted, and its null-rendering tests moved with it — see
#  achernar-frontend/src/hooks/useEquipment.test.ts. What remains here is the
#  part Django still owns: never reporting stale cached hardware as connected.
# ─────────────────────────────────────────────────────────────────────────

from datetime import timedelta
from unittest import mock

from django.utils import timezone
from rest_framework.test import APITestCase

from apps.equipment.models import ConnectedEquipment
from apps.equipment.serializers import ConnectedEquipmentSerializer


def make_equipment(**overrides):
    """Create a row, then force `last_seen` past the auto_now default."""
    last_seen = overrides.pop('last_seen', None)
    defaults = {
        'equipment_id': 'camera_1',
        'equipment_type': 'camera',
        'name': 'ASI6200MM Pro',
        'site': 'Backyard Obs',
        'connected': True,
        'status': 'nominal',
        'driver': 'zwo_asi',
    }
    defaults.update(overrides)
    row = ConnectedEquipment.objects.create(**defaults)
    if last_seen is not None:
        # auto_now overwrites on save(), so go around the ORM's back.
        ConnectedEquipment.objects.filter(pk=row.pk).update(last_seen=last_seen)
        row.refresh_from_db()
    return row


class EquipmentFreshnessTests(TestCase):
    def test_stale_row_is_not_reported_connected(self):
        """The exact bug: a six-month-old row claiming connected: true."""
        row = make_equipment(last_seen=timezone.now() - timedelta(days=180))
        data = ConnectedEquipmentSerializer(row).data

        self.assertFalse(data['connected'], 'a six-month-old row is not connected')
        self.assertTrue(data['stale'])
        self.assertEqual(data['status'], 'offline')

    def test_stale_row_still_reports_when_it_was_last_seen(self):
        """Deriving connection state is only honest if staleness is visible."""
        seen = timezone.now() - timedelta(days=180)
        row = make_equipment(last_seen=seen)
        data = ConnectedEquipmentSerializer(row).data

        self.assertIsNotNone(data['last_seen'], 'last_seen must reach the client')
        self.assertEqual(data['meta']['last_seen'], data['last_seen'])

    def test_fresh_row_is_reported_connected(self):
        row = make_equipment(last_seen=timezone.now())
        data = ConnectedEquipmentSerializer(row).data

        self.assertTrue(data['connected'])
        self.assertFalse(data['stale'])
        self.assertEqual(data['status'], 'nominal')

    def test_fresh_row_that_reported_itself_disconnected_stays_disconnected(self):
        """Freshness can only take connection away, never grant it."""
        row = make_equipment(connected=False, last_seen=timezone.now())
        data = ConnectedEquipmentSerializer(row).data

        self.assertFalse(data['connected'])


class CommandErrorTests(APITestCase):
    def setUp(self):
        views._REGISTRY_CACHE.update({'base': None, 'fetched_at': 0.0, 'commands': None})

    def test_unknown_action_returns_400_naming_the_action(self):
        """An unsupported verb and an absent daemon are different failures."""
        with mock.patch.object(views, 'get_daemon', return_value=registry_payload()):
            response = self.client.post(
                '/api/equipment/command/',
                {'action': 'focus.run', 'payload': {}},
                format='json',
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn('focus.run', response.data['error'])
        self.assertIn('camera.capture', response.data['supported_actions'])

    def test_supported_actions_come_from_the_daemon_not_a_local_list(self):
        """Whatever this build declares is what Django reports."""
        payload = [{'name': 'only.this', 'method': 'GET', 'path': 'thing',
                    'description': 'The one command this build has.'}]
        with mock.patch.object(views, 'get_daemon', return_value=payload):
            response = self.client.post(
                '/api/equipment/command/',
                {'action': 'camera.capture'},
                format='json',
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data['supported_actions'], ['only.this'])

    def test_unreachable_daemon_returns_503_not_400(self):
        """Cannot-ask and does-not-exist must not report as the same thing."""
        with mock.patch.object(views, 'get_daemon', side_effect=OSError('refused')):
            response = self.client.post(
                '/api/equipment/command/',
                {'action': 'camera.capture'},
                format='json',
            )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn('supported_actions', response.data)

    def test_known_action_is_routed_to_the_published_path(self):
        with mock.patch.object(views, 'get_daemon', return_value=registry_payload()), \
             mock.patch.object(views, 'post_daemon', return_value={'status': 'ok'}) as post:
            response = self.client.post(
                '/api/equipment/command/',
                {'action': 'camera.abort'},
                format='json',
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.call_args.args[0], 'camera/abort')

    def test_get_commands_are_not_posted(self):
        registry = registry_payload()
        with mock.patch.object(views, 'get_daemon', side_effect=[registry, {'position': 1200}]) as get, \
             mock.patch.object(views, 'post_daemon') as post:
            response = self.client.post(
                '/api/equipment/command/',
                {'action': 'focuser.position'},
                format='json',
            )

        self.assertEqual(response.status_code, 200)
        post.assert_not_called()
        self.assertEqual(get.call_args.args[0], 'focuser/position')

    def test_query_parameter_commands_keep_their_quirk(self):
        with mock.patch.object(views, 'get_daemon', return_value=registry_payload()), \
             mock.patch.object(views, 'post_daemon', return_value={}) as post:
            self.client.post(
                '/api/equipment/command/',
                {'action': 'camera.start_stream', 'payload': {'format_index': 3}},
                format='json',
            )

        self.assertEqual(post.call_args.args[0], 'camera/stream/start?format_index=3')

    def test_missing_action_still_returns_400(self):
        response = self.client.post('/api/equipment/command/', {}, format='json')
        self.assertEqual(response.status_code, 400)


class ConnectedEndpointTests(APITestCase):
    """The endpoint says where its data came from, and never invents hardware."""

    def test_no_daemon_and_no_rows_says_so_explicitly(self):
        with mock.patch('apps.equipment.views.fetch_daemon_status', return_value=None):
            response = self.client.get('/api/equipment/connected/')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['source'], 'none')
        self.assertIsNone(response.data['status'])
        self.assertEqual(response.data['cached'], [])

    def test_no_daemon_never_reports_stale_hardware_as_connected(self):
        make_equipment(last_seen=timezone.now() - timedelta(days=180))

        with mock.patch('apps.equipment.views.fetch_daemon_status', return_value=None):
            response = self.client.get('/api/equipment/connected/')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['source'], 'cache')
        self.assertIsNone(response.data['status'], 'cached rows are not a live reading')

        cached = response.data['cached']
        self.assertEqual(len(cached), 1)
        self.assertEqual(cached[0]['name'], 'ASI6200MM Pro')
        self.assertFalse(cached[0]['connected'], 'stale hardware reported as connected')
        self.assertTrue(cached[0]['stale'])

    def test_unreachable_daemon_does_not_promote_even_recent_cached_devices(self):
        make_equipment(last_seen=timezone.now())
        with mock.patch('apps.equipment.views.fetch_daemon_status', return_value=None):
            response = self.client.get('/api/equipment/connected/')
        self.assertEqual(response.data['source'], 'cache')
        self.assertFalse(response.data['cached'][0]['connected'])
        self.assertFalse(response.data['cached'][0]['meta']['connected'])
        self.assertEqual(response.data['cached'][0]['status'], 'offline')

    def test_a_reachable_daemon_is_relayed_verbatim(self):
        """Django passes the daemon's payload through; it does not reshape it."""
        payload = {
            'connected': True,
            'camera': {'name': 'DSLR', 'connected': True, 'temperature': None},
            'mount': None,
            'focuser': None,
        }
        with mock.patch('apps.equipment.views.fetch_daemon_status', return_value=payload):
            response = self.client.get('/api/equipment/connected/')

        self.assertEqual(response.data['source'], 'localhost')
        self.assertEqual(response.data['status'], payload)
        # The null survives the hop. It used to arrive as "0.0 C".
        self.assertIsNone(response.data['status']['camera']['temperature'])
