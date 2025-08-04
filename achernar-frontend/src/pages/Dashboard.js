import React from 'react';
import UserCard from '../components/UserCard';
import EquipmentCard from '../components/EquipmentCard';

function Dashboard() {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
        gap: '1rem',
      }}
    >
      <UserCard />
      <EquipmentCard />
    </div>
  );
}

export default Dashboard;
