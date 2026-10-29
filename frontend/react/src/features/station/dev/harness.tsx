/**
 * Development harness for the station screen (not part of the app bundle).
 *
 * Served by `scripts/check-station-layout.mjs` through a Vite dev server at
 * `/src/features/station/dev/harness.html?station=<id>`; that script answers
 * every `/api/line/**` request with deterministic mocks built from
 * `contracts/production/examples`. Run it with `--serve` to keep the mock
 * server open for manual work. The app's global stylesheet is loaded so the
 * page renders exactly as it will inside the app shell. `historyIdleMs` and
 * `offlineStaleMs` / `stallAfterMs` query parameters shorten the page's timers for the check.
 */
import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

import '../../../index.css'
import StationPage from '../index'

const params = new URLSearchParams(window.location.search)
const stationId = params.get('station') ?? 'cell-1'
// Shorter timers for the browser check (production defaults: 60 s / 10 s).
const msParam = (name: string): number | undefined => {
  const v = Number(params.get(name))
  return Number.isFinite(v) && v > 0 ? v : undefined
}
const historyIdleMs = msParam('historyIdleMs')
const offlineStaleMs = msParam('offlineStaleMs')
const stallAfterMs = msParam('stallAfterMs')
const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <StationPage stationId={stationId} historyIdleMs={historyIdleMs} offlineStaleMs={offlineStaleMs} stallAfterMs={stallAfterMs} />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
)
