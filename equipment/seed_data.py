"""
Seed equipment data for testing.
Run with: python manage.py shell < equipment/seed_data.py
Or import and run from Django shell.
"""

from equipment.models import ConnectedEquipment

# Clear existing data
ConnectedEquipment.objects.all().delete()

# Create seed equipment matching the original frontend seed
equipment_data = [
    {
        'equipment_id': 'eq-mount-001',
        'equipment_type': 'mount',
        'name': 'AP 1100GTO',
        'site': 'La Palma',
        'status': 'nominal',
        'connected': True,
        'driver': 'indi',
        'tags': ['tracking', 'goto', 'pier-west'],
        'meta': {'tracking': True, 'model': 'Astro-Physics'},
    },
    {
        'equipment_id': 'eq-cam-001',
        'equipment_type': 'camera',
        'name': 'ASI6200MM Pro',
        'site': 'La Palma',
        'status': 'nominal',
        'connected': True,
        'driver': 'zwo_asi',
        'tags': ['mono', 'cooling', 'usb3'],
        'meta': {'temp': '-10.1°C', 'mode': 'exposing'},
    },
    {
        'equipment_id': 'eq-focus-001',
        'equipment_type': 'focuser',
        'name': 'Moonlite CHL',
        'site': 'Siding Spring',
        'status': 'warning',
        'connected': True,
        'driver': 'indi',
        'tags': ['temp-drift', 'backlash'],
        'meta': {'drift': 'ΔT>3°C', 'steps': 12840},
    },
    {
        'equipment_id': 'eq-fw-001',
        'equipment_type': 'filter-wheel',
        'name': 'EFW 7x2"',
        'site': 'Atacama',
        'status': 'nominal',
        'connected': True,
        'driver': 'indi',
        'tags': ['Ha', 'OIII', 'SII'],
        'meta': {'slot': 3, 'filter': 'Ha'},
    },
    {
        'equipment_id': 'eq-dome-001',
        'equipment_type': 'dome',
        'name': 'Robodome v2',
        'site': 'Atacama',
        'status': 'offline',
        'connected': False,
        'driver': 'indi',
        'tags': ['shutter', 'az-sync'],
        'meta': {'reason': 'link down'},
    },
]

# Create equipment entries
created_count = 0
for data in equipment_data:
    equipment = ConnectedEquipment.objects.create(**data)
    print(f"Created: {equipment}")
    created_count += 1

print(f"\n✅ Created {created_count} equipment entries")
print(f"Total equipment in database: {ConnectedEquipment.objects.count()}")
