import { Navigate, Route, Routes } from "react-router-dom";
import TopNav from "./components/TopNav";
import Dashboard from "./pages/Dashboard";
import Equipment from "./pages/Equipment";
import Queue from "./pages/Queue";
import DataLab from "./pages/DataLab";
import Logs from "./pages/Logs";

export default function App() {
  return (
    <div className="app-root">
      <header className="top-nav-wrapper">
        <TopNav />
      </header>

      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/equipment" element={<Equipment />} />
        <Route path="/queue" element={<Queue />} />
        <Route path="/data-lab" element={<DataLab />} />
        <Route path="/logs" element={<Logs />} />
      </Routes>
    </div>
  );
}
