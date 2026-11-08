from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [('realtime', '0003_live_hardware_state')]

    operations = [
        migrations.AddField(
            model_name='telemetrysnapshot',
            name='status_blocks',
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
