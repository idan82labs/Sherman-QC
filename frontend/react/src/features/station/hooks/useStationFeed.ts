/**
 * Live station feed: ETag polling with an optional SSE accelerator.
 *
 * - Polls `GET /api/line/stations/{id}/latest` every 2 s with
 *   `If-None-Match`; a 304 keeps the previous feed (no re-render churn).
 * - When the SSE stream is available (`POST /stream-token`, then
 *   `EventSource(/events/stream?station_id&token)`), any line event triggers
 *   an immediate refetch and polling relaxes to a 10 s safety net. Any SSE
 *   error closes the stream and falls back to 2 s polling; the stream is
 *   retried with back-off.
 * - `connection` is `live` (stream open and feed fresh), `polling` (feed
 *   fresh, no stream) or `offline` (the last fetch failed).
 * - A 401/403 stops polling and is returned as `authError` so the page can
 *   show the token gate; nothing is retried with a rejected token.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'

import { getFeed, mintStreamToken, streamUrl, StationApiError } from '../api/stationApi'
import type { StationFeed } from '../model/types'

export type Connection = 'live' | 'polling' | 'offline'

export const POLL_MS = 2000
export const LIVE_SAFETY_POLL_MS = 10_000
const SSE_RETRY_MIN_MS = 15_000
const SSE_RETRY_MAX_MS = 120_000
// The server ends every stream after a fixed lifetime (~5 min). A stream that was healthy for at
// least SSE_HEALTHY_MS reopens after SSE_REOPEN_MS instead of the error backoff, so the station does
// not drop to "Polling" for 15 s every few minutes.
const SSE_REOPEN_MS = 1_000
const SSE_HEALTHY_MS = 30_000

/** Event kinds that can change what the station shows (LineEventV1.kind). */
const STATION_EVENTS = [
  'SCAN_RECEIVED',
  'SCAN_QUARANTINED',
  'JOB_QUEUED',
  'JOB_STARTED',
  'JOB_PROGRESS',
  'JOB_COMPLETED',
  'JOB_FAILED',
  'JOB_CANCELLED',
  'JOB_RECOVERED',
  'QUARANTINE_ASSIGNED',
  'QUARANTINE_DISMISSED',
] as const

export interface StationFeedState {
  feed: StationFeed | null
  connection: Connection
  /** ms epoch of the last successful fetch (200 or 304). */
  lastUpdated: number | null
  /** server_time minus local time, ms (for elapsed-time display). */
  serverOffsetMs: number
  error: Error | null
  authError: StationApiError | null
  refetch: () => void
}

export function useStationFeed(
  stationId: string,
  options: { enabled: boolean; tokenVersion: number; sse?: boolean },
): StationFeedState {
  const { enabled, tokenVersion } = options
  const sseWanted = options.sse ?? true
  // State is scoped to (station, token): a new token starts from a clean slate.
  const scope = `${stationId}|${tokenVersion}`
  const etagRef = useRef<{ scope: string; etag: string | null; feed: StationFeed | null }>({ scope, etag: null, feed: null })
  const [serverOffsetMs, setServerOffsetMs] = useState(0)
  const [stream, setStream] = useState<{ scope: string; open: boolean }>({ scope, open: false })
  const [auth, setAuth] = useState<{ scope: string; error: StationApiError | null }>({ scope, error: null })
  const authError = auth.scope === scope ? auth.error : null
  const sseActive = enabled && !authError && sseWanted && typeof window !== 'undefined' && typeof window.EventSource !== 'undefined'
  const streamOpen = sseActive && stream.scope === scope && stream.open

  const query = useQuery({
    queryKey: ['station-feed', stationId, tokenVersion],
    enabled: enabled && !authError,
    queryFn: async ({ signal }) => {
      const cache = etagRef.current.scope === scope ? etagRef.current : { scope, etag: null, feed: null }
      etagRef.current = cache
      try {
        let res = await getFeed(stationId, cache.feed ? cache.etag : null, signal)
        if (res.status === 'not_modified' && !cache.feed) res = await getFeed(stationId, null, signal)
        if (res.status === 'ok') {
          cache.etag = res.etag
          cache.feed = res.feed
          const server = Date.parse(res.feed.server_time)
          if (Number.isFinite(server)) setServerOffsetMs(server - Date.now())
        }
        return cache.feed as StationFeed
      } catch (err) {
        if (err instanceof StationApiError && err.isAuth) setAuth({ scope, error: err })
        throw err
      }
    },
    refetchInterval: streamOpen ? LIVE_SAFETY_POLL_MS : POLL_MS,
    refetchIntervalInBackground: true,
    refetchOnWindowFocus: false,
    staleTime: 0,
    retry: false,
  })

  const { refetch } = query
  const refetchNow = useCallback(() => {
    void refetch()
  }, [refetch])

  // Optional SSE accelerator.
  useEffect(() => {
    if (!sseActive) return
    let cancelled = false
    let source: EventSource | null = null
    let retryTimer: number | undefined
    let delay = SSE_RETRY_MIN_MS
    const controller = new AbortController()

    const schedule = () => {
      if (cancelled) return
      retryTimer = window.setTimeout(connect, delay)
      delay = Math.min(SSE_RETRY_MAX_MS, delay * 2)
    }
    const onEvent = () => {
      void refetch()
    }
    async function connect() {
      if (cancelled) return
      try {
        const { token } = await mintStreamToken(stationId, controller.signal)
        if (cancelled) return
        source = new EventSource(streamUrl(stationId, token))
        let openedAt = 0
        source.onopen = () => {
          delay = SSE_RETRY_MIN_MS
          openedAt = Date.now()
          setStream({ scope, open: true })
          // Catch up on anything that happened while no stream was open (events are not replayed).
          void refetch()
        }
        source.onmessage = onEvent
        for (const kind of STATION_EVENTS) source.addEventListener(kind, onEvent)
        source.onerror = () => {
          source?.close()
          source = null
          setStream({ scope, open: false })
          if (openedAt && Date.now() - openedAt >= SSE_HEALTHY_MS) {
            // A healthy stream ended (server lifetime): reopen promptly and catch up on the feed.
            if (!cancelled) retryTimer = window.setTimeout(connect, SSE_REOPEN_MS)
            void refetch()
          } else {
            schedule()
          }
        }
      } catch (err) {
        if (cancelled) return
        setStream({ scope, open: false })
        if (err instanceof StationApiError && err.isAuth) return // polling surfaces the auth problem
        schedule()
      }
    }
    void connect()
    return () => {
      cancelled = true
      controller.abort()
      window.clearTimeout(retryTimer)
      source?.close()
      setStream({ scope, open: false })
    }
  }, [sseActive, scope, stationId, refetch])

  const failed = query.isError && !authError
  const connection: Connection = failed ? 'offline' : streamOpen ? 'live' : 'polling'
  return {
    feed: query.data ?? null,
    connection,
    lastUpdated: query.dataUpdatedAt || null,
    serverOffsetMs,
    error: query.error ?? null,
    authError,
    refetch: refetchNow,
  }
}
