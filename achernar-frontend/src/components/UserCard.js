import React, { useEffect, useState } from 'react';

function UserCard() {
  const [user, setUser] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/user/')
      .then((res) => res.json())
      .then((data) => setUser(data))
      .catch(() => setUser({ username: 'anonymous', email: 'unknown' }));
  }, []);

  if (!user) {
    return <div className="card">Loading user...</div>;
  }

  return (
    <div className="card">
      <h2>{user.username}</h2>
      <p>{user.email}</p>
    </div>
  );
}

export default UserCard;
