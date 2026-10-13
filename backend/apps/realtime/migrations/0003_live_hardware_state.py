from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [('realtime', '0002_dev_relay_tunnel')]

    operations = [
        migrations.AddField(
            model_name='daemonconnection',
            name='gateway_channel_name',
            field=models.CharField(blank=True, max_length=255),
        ),
        migrations.AddField(
            model_name='telemetrysnapshot',
            name='filter_wheel_state',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name='telemetrysnapshot',
            name='device_health',
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
