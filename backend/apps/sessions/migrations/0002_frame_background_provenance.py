"""Background provenance on frames (daemon BackgroundResult schema 1).

Additive and reversible: existing rows keep their background_adu and get
NULL provenance, which reads as legacy-unverified. Reversing drops only the
new columns and constraints.
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('achernar_sessions', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='frame',
            name='background_sigma',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='frame',
            name='background_schema_version',
            field=models.PositiveSmallIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='frame',
            name='background_method',
            field=models.CharField(blank=True, help_text='legacy_global_clip | sep_spatial_mesh', max_length=40, null=True),
        ),
        migrations.AddField(
            model_name='frame',
            name='background_config_sha256',
            field=models.CharField(blank=True, max_length=64, null=True),
        ),
        migrations.AddField(
            model_name='frame',
            name='background_level_status',
            field=models.CharField(blank=True, choices=[('measured', 'Measured'), ('unavailable', 'Unavailable')], max_length=20, null=True),
        ),
        migrations.AddField(
            model_name='frame',
            name='background_noise_status',
            field=models.CharField(blank=True, choices=[('measured', 'Measured'), ('unresolved', 'Unresolved'), ('unavailable', 'Unavailable')], max_length=20, null=True),
        ),
        migrations.AddField(
            model_name='frame',
            name='background_noise_source',
            field=models.CharField(blank=True, max_length=40, null=True),
        ),
        migrations.AddField(
            model_name='frame',
            name='background_units',
            field=models.CharField(blank=True, help_text='adu | adu_proxy | unverified', max_length=20, null=True),
        ),
        migrations.AddField(
            model_name='frame',
            name='background_reasons',
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddConstraint(
            model_name='frame',
            constraint=models.CheckConstraint(check=models.Q(('background_schema_version__isnull', True), ('background_adu__isnull', True), ('background_level_status', 'measured'), _connector='OR'), name='frame_background_level_only_when_measured'),
        ),
        migrations.AddConstraint(
            model_name='frame',
            constraint=models.CheckConstraint(check=models.Q(('background_schema_version__isnull', True), ('background_sigma__isnull', True), ('background_noise_status', 'measured'), _connector='OR'), name='frame_background_sigma_only_when_measured'),
        ),
    ]
