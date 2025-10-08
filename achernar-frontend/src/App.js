import React, { useEffect, useState } from 'react';
import Dashboard from './pages/Dashboard';

function App() {
  const [apiMessage, setApiMessage] = useState('');

  useEffect(() => {
    fetch('http://localhost:8000/api/hello/')
      .then(res => res.json())
      .then(data => setApiMessage(data.message));
  }, []);

  return (
    <div style={{ background: '#18181b', color: '#fff', minHeight: '100vh', padding: '2rem' }}>
      <h1>Achernar Dashboard (MVP)</h1>
      <div style={{ margin: '2rem 0' }}>
        <strong>Django API says:</strong> {apiMessage}
      </div>
      <Dashboard />
    </div>
  );
}

export default App;
