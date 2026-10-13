import { Navigate, Route, Routes } from "react-router-dom";
import TopNav from "./components/TopNav";
import Dashboard from "./pages/Dashboard";
import RunRoom from "./pages/RunRoom";

export default function App() {
  return (
    <div className="app-root">
      <header className="top-nav-wrapper">
        <TopNav />
      </header>

      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/run" element={<RunRoom />} />
        <Route path="/equipment" element={<Navigate to="/dashboard" replace />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </div>
  );
}
