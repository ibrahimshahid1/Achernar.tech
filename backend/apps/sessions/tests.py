"""Background provenance on frames: migration reversibility and contract.

Runs only against Django's disposable test database (in-memory SQLite with
USE_SQLITE=1); it never touches a live database.
"""

from django.db import IntegrityError, connection, transaction
from django.db.migrations.executor import MigrationExecutor
from django.test import TestCase, TransactionTestCase
from django.utils import timezone

from .models import Frame, ImagingSession
from .serializers import FrameSerializer

APP = 'achernar_sessions'


class BackgroundMigrationTests(TransactionTestCase):
    """Forward and backward over 0002 with a pre-contract row."""

    def migrate(self, target):
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate([(APP, target)])
        return executor.loader.project_state([(APP, target)]).apps

    def tearDown(self):
        self.migrate('0002_frame_background_provenance')

    def test_legacy_row_survives_both_directions_unchanged(self):
        apps = self.migrate('0001_initial')
        Session = apps.get_model(APP, 'ImagingSession')
        OldFrame = apps.get_model(APP, 'Frame')
        session = Session.objects.create(daemon_id='s1', started_at=timezone.now())
        # A historical stored 0.0: ambiguous, must be preserved, not rewritten.
        OldFrame.objects.create(
            daemon_id='f1', session=session, exposure_s=10.0, background_adu=0.0,
            captured_at=timezone.now(),
        )

        apps = self.migrate('0002_frame_background_provenance')
        frame = apps.get_model(APP, 'Frame').objects.get(daemon_id='f1')
        self.assertEqual(frame.background_adu, 0.0)
        self.assertIsNone(frame.background_schema_version)
        self.assertEqual(frame.background_reasons, [])
        self.assertEqual(Frame.objects.get(daemon_id='f1').background_provenance, 'legacy_unverified')

        apps = self.migrate('0001_initial')
        frame = apps.get_model(APP, 'Frame').objects.get(daemon_id='f1')
        self.assertEqual(frame.background_adu, 0.0)


class BackgroundContractTests(TestCase):
    def setUp(self):
        self.session = ImagingSession.objects.create(daemon_id='s2', started_at=timezone.now())

    def frame(self, daemon_id, **fields):
        return Frame(
            daemon_id=daemon_id, session=self.session, exposure_s=5.0,
            captured_at=timezone.now(), **fields,
        )

    def test_versioned_unavailable_row_cannot_carry_a_level(self):
        with self.assertRaises(IntegrityError), transaction.atomic():
            self.frame(
                'bad', background_schema_version=1, background_level_status='unavailable',
                background_adu=0.0,
            ).save()

    def test_versioned_unresolved_noise_cannot_carry_a_sigma(self):
        with self.assertRaises(IntegrityError), transaction.atomic():
            self.frame(
                'bad-sigma', background_schema_version=1, background_level_status='measured',
                background_adu=100.0, background_noise_status='unresolved', background_sigma=0.0,
            ).save()

    def test_serializer_reports_provenance_and_nulls(self):
        measured = self.frame(
            'ok', background_schema_version=1, background_method='sep_spatial_mesh',
            background_level_status='measured', background_noise_status='unresolved',
            background_adu=812.0, background_sigma=None, background_units='adu',
            background_reasons=['zero_empirical_scatter'],
        )
        measured.save()
        absent = self.frame('none')
        absent.save()
        data = FrameSerializer(measured).data
        self.assertEqual(data['background_provenance'], 'current')
        self.assertIsNone(data['background_sigma'])
        self.assertEqual(data['background_reasons'], ['zero_empirical_scatter'])
        self.assertEqual(FrameSerializer(absent).data['background_provenance'], 'absent')
        self.assertIsNone(FrameSerializer(absent).data['background_adu'])
