import React, { useEffect, useState } from 'react';

function EquipmentCard() {
  const [equipment, setEquipment] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/equipment/')
      .then((res) => res.json())
      .then((data) => setEquipment(data))
      .catch(() =>
        setEquipment({
          mount: 'Unknown mount',
          camera: 'Unknown camera',
          filter_wheel: 'Unknown',
        })
      );
  }, []);

  if (!equipment) {
    return <div className="card">Loading equipment...</div>;
  }

  return (
    <div className="card">
      <h3>Your Equipment</h3>
      <ul>
        <li>Mount: {equipment.mount}</li>
        <li>Camera: {equipment.camera}</li>
        <li>Filter Wheel: {equipment.filter_wheel}</li>
      </ul>
    </div>
  );
}

export default EquipmentCard;
