import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { lazy, Suspense } from 'react'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Upload from './pages/Upload'
import Jobs from './pages/Jobs'
import JobDetail from './pages/JobDetail'
import Batch from './pages/Batch'
import BendInspection from './pages/BendInspection'
import PartCatalog from './pages/PartCatalog'
import VendorSetupPage from './features/part-catalog/pages/VendorSetupPage'
import LiveScan from './pages/LiveScan'
import ManualAssistant from './pages/ManualAssistant'
import ShermanChat from './pages/ShermanChat'
import BenderViewPage from './pages/BenderViewPage'
import BendLabeling from './pages/BendLabeling'
import Login from './pages/Login'
import { useAuth } from './hooks/useAuth'
import { LegacyEngineRoute } from './hooks/useLineFeatures'

// Production line (C11 integration). Loaded on demand so three.js and the
// onboarding wizard stay out of the main chunk.
const StationPage = lazy(() => import('./features/station'))
const ProductionPartsPage = lazy(() => import('./features/part-onboarding').then((m) => ({ default: m.ProductionPartsPage })))
const NewProductionPartPage = lazy(() => import('./features/part-onboarding').then((m) => ({ default: m.NewProductionPartPage })))
const OnboardingWizardPage = lazy(() => import('./features/part-onboarding').then((m) => ({ default: m.OnboardingWizardPage })))

function RouteFallback() {
  return (
    <div className="min-h-[50vh] flex items-center justify-center" role="status" aria-label="Loading">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
    </div>
  )
}

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth()
  const location = useLocation()

  // Only the local, CAD-derived corpus presentation route is viewable without
  // a running catalog backend. Production builds never take this branch.
  if (import.meta.env.DEV && /^\/parts\/vendor-setup\/corpus-\d{8}$/.test(location.pathname)) {
    return <>{children}</>
  }

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
  const labelingOnly = import.meta.env.VITE_LABELING_ONLY === 'true'

  if (labelingOnly) {
    return (
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/bend-labeling" element={
          <PrivateRoute><BendLabeling /></PrivateRoute>
        } />
        <Route path="*" element={<Navigate to="/bend-labeling" replace />} />
      </Routes>
    )
  }

  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/sherman-chat" element={<ShermanChat />} />
      <Route path="/assistant" element={<ShermanChat />} />
      {/* Human training tool: standalone full-screen surface, no QC application chrome. */}
      <Route path="/bend-labeling" element={
        <PrivateRoute><BendLabeling /></PrivateRoute>
      } />
      {/* Bender's View — standalone full-screen, no sidebar */}
      <Route path="/bender-view/:jobId" element={
        <PrivateRoute><BenderViewPage /></PrivateRoute>
      } />
      {/* Press-brake station kiosk (C08): standalone, outside Layout and PrivateRoute; its own device-token gate. */}
      <Route path="/station/:stationId" element={
        <Suspense fallback={<RouteFallback />}><StationPage /></Suspense>
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
        <Route path="upload" element={<LegacyEngineRoute><Upload /></LegacyEngineRoute>} />
        <Route path="jobs" element={<Jobs />} />
        <Route path="jobs/:jobId" element={<JobDetail />} />
        <Route path="batch" element={<LegacyEngineRoute><Batch /></LegacyEngineRoute>} />
        <Route path="bend-inspection" element={<LegacyEngineRoute><BendInspection /></LegacyEngineRoute>} />
        <Route path="parts" element={<PartCatalog />} />
        <Route path="parts/vendor-setup" element={<VendorSetupPage />} />
        <Route path="parts/vendor-setup/:partId" element={<VendorSetupPage />} />
        <Route path="live-scan" element={<LiveScan />} />
        <Route path="assistant-dev" element={<ManualAssistant />} />
        {/* Production onboarding (C09) */}
        <Route path="production/parts" element={<Suspense fallback={<RouteFallback />}><ProductionPartsPage /></Suspense>} />
        <Route path="production/parts/new" element={<Suspense fallback={<RouteFallback />}><NewProductionPartPage /></Suspense>} />
        <Route path="production/parts/:partId/onboard" element={<Suspense fallback={<RouteFallback />}><OnboardingWizardPage /></Suspense>} />
      </Route>
    </Routes>
  )
}
