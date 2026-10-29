import type { ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Navigate } from 'react-router-dom'
import api from '../services/api'

/** `GET /api/line/features`: which UI areas apply on this server (no auth, flags only). */
export interface LineFeatures {
  line_enabled: boolean
  /** False while the production line is enabled (SHERMAN_LEGACY_ENGINE=0): the route
   *  policy refuses the legacy engine uploads, so their pages are hidden. */
  legacy_engine_enabled: boolean
}

/** Paths of the legacy analysis pages whose engine calls the line refuses. */
export const LEGACY_ENGINE_PATHS: ReadonlySet<string> = new Set(['/upload', '/batch', '/bend-inspection'])

/**
 * `legacyEngine`: true / false once known, null while loading. A server without the
 * endpoint (older legacy app) answers with an error: the legacy pages stay.
 */
export function useLineFeatures(): { legacyEngine: boolean | null } {
  const q = useQuery({
    queryKey: ['line-features'],
    queryFn: async () => (await api.get<LineFeatures>('/line/features')).data,
    staleTime: 60_000,
    retry: 1,
  })
  if (q.data) return { legacyEngine: q.data.legacy_engine_enabled !== false }
  return { legacyEngine: q.isError ? true : null }
}

/** Route guard for the legacy analysis pages: redirects home while the line is enabled. */
export function LegacyEngineRoute({ children }: { children: ReactNode }) {
  const { legacyEngine } = useLineFeatures()
  if (legacyEngine === null) return null
  if (!legacyEngine) return <Navigate to="/" replace />
  return <>{children}</>
}
