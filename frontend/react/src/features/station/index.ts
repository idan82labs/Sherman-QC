/**
 * Press-brake station screen (C08). Ready-to-mount route component.
 *
 * Mounting (C11 owns App.tsx): register the route OUTSIDE the app Layout and
 * OUTSIDE PrivateRoute, because a kiosk authenticates with a station device
 * token (the page shows its own token gate) and needs the full screen:
 *
 *   import StationPage from './features/station'
 *   <Route path="/station/:stationId" element={<StationPage />} />
 *
 * Optional props: `stationId` (overrides the route parameter),
 * `historyIdleMs` (history view returns to live after this long without
 * interaction, default 60 s) and `offlineStaleMs` (verdict marked possibly
 * outdated after the feed has been unreachable this long, default 10 s).
 * A history view also returns to live as soon as a newer part completes.
 * Layer toggles are reloaded when `stationId` changes, so no `key` is needed.
 *
 * Requirements: a react-query `QueryClientProvider` and a react-router
 * context above it (both already in `main.tsx`). The page talks only to
 * `/api/line` (C07): `GET /stations/{id}/latest` (ETag), `POST /stream-token`
 * + `GET /events/stream` (optional SSE), `GET /jobs/{id}/result` and
 * `GET /jobs/{id}/artifacts/{name}`. Station ids match `^[a-z0-9-]{1,32}$`.
 *
 * Browser storage (per-viewer conveniences, all try/catch):
 * `sherman.station.token.<stationId>` (device token), `token` (user JWT,
 * read only), `sherman.station.lang`, `sherman.station.layers.v2.<stationId>`.
 */
export { default, HISTORY_IDLE_MS, OFFLINE_STALE_MS } from './pages/StationPage'
export type { StationPageProps } from './pages/StationPage'
export { validateResult, checkEnvelope, deriveSummary, deriveStageDecision, toRows, CONTRACT_KEYS } from './model/stationView'
