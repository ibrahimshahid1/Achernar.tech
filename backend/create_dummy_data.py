from apps.equipment.models import ConnectedEquipment, EquipmentType, ConnectionStatus

def create_dummy_data():
    print("Creating dummy equipment data...")
    ConnectedEquipment.objects.all().delete()
    
    ConnectedEquipment.objects.create(
        equipment_id="mount_1",
        equipment_type=EquipmentType.MOUNT,
        name="AP 1100GTO",
        connected=True,
        site="Backyard Obs",
        status=ConnectionStatus.NOMINAL,
        tags=["tracking", "guided"],
        driver="indi_ap_gto",
        meta={"ra": "14h 22m", "dec": "-32d 05m"}
    )

    ConnectedEquipment.objects.create(
        equipment_id="camera_1",
        equipment_type=EquipmentType.CAMERA,
        name="ASI6200MM Pro",
        connected=True,
        site="Backyard Obs",
        status=ConnectionStatus.NOMINAL,
        tags=["cooled", "mono"],
        driver="zwo_asi",
        meta={"temp": -10.0, "gain": 100}
    )

    ConnectedEquipment.objects.create(
        equipment_id="focuser_1",
        equipment_type=EquipmentType.FOCUSER,
        name="MoonLite NiteCrawler",
        connected=True,
        site="Backyard Obs",
        status=ConnectionStatus.NOMINAL,
        tags=["rotator"],
        driver="moonlite_nc",
        meta={"pos": 24500}
    )
    
    ConnectedEquipment.objects.create(
        equipment_id="filter_wheel_1",
        equipment_type=EquipmentType.FILTER_WHEEL,
        name="ZWO EFW 7x50",
        connected=True,
        site="Backyard Obs",
        status=ConnectionStatus.NOMINAL,
        tags=["narrowband"],
        driver="zwo_efw",
        meta={"current": "Ha"}
    )

    print("Dummy data created successfully.")

if __name__ == "__main__":
    create_dummy_data()
