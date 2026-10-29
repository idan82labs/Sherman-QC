import React from 'react'
import ReactDOM from 'react-dom/client'
import { HashRouter, Navigate, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import VendorSetupPage from './features/part-catalog/pages/VendorSetupPage'
import './vendorDemoStaticApi'
import './vendorDemoWebMcp'
import './index.css'

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, refetchOnWindowFocus: false } } })

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <HashRouter>
        <Routes>
          <Route path="/parts/vendor-setup/:partId" element={<VendorSetupPage />} />
          <Route path="*" element={<Navigate to="/parts/vendor-setup/corpus-37361005" replace />} />
        </Routes>
      </HashRouter>
    </QueryClientProvider>
  </React.StrictMode>,
)
