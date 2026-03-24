import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Upload from './pages/Upload'
import Jobs from './pages/Jobs'
import JobDetail from './pages/JobDetail'
import Batch from './pages/Batch'
import BendInspection from './pages/BendInspection'
import PartCatalog from './pages/PartCatalog'
import LiveScan from './pages/LiveScan'
import BenderViewPage from './pages/BenderViewPage'
import Login from './pages/Login'
import { useAuth } from './hooks/useAuth'

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth()

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
      </div>
    )
  }

  return isAuthenticated ? <>{children}</> : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      {/* Bender's View — standalone full-screen, no sidebar */}
      <Route path="/bender-view/:jobId" element={
        <PrivateRoute><BenderViewPage /></PrivateRoute>
      } />
      <Route
        path="/"
        element={
          <PrivateRoute>
            <Layout />
          </PrivateRoute>
        }
      >
        <Route index element={<Dashboard />} />
        <Route path="upload" element={<Upload />} />
        <Route path="jobs" element={<Jobs />} />
        <Route path="jobs/:jobId" element={<JobDetail />} />
        <Route path="batch" element={<Batch />} />
        <Route path="bend-inspection" element={<BendInspection />} />
        <Route path="parts" element={<PartCatalog />} />
        <Route path="live-scan" element={<LiveScan />} />
      </Route>
    </Routes>
  )
}
