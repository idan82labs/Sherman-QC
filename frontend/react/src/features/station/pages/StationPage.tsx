/**
 * 16:9 press-brake station screen (kiosk), route `/station/:stationId`.
 *
 * A fixed 100vw x 100vh grid with no page scroll, designed for 1920x1080
 * and for benders reading it from the brake (station v2):
 * - top banner: Sherman logo, the part, "8 / 12 bends done" with one pip
 *   per bend, one big part verdict (OK / OK so far / CHECK / FIX / Rescan),
 *   then station, connection, clock and language;
 * - the 3D part on the reading-start side (about 61 % of the width): every
 *   bend has a tag with its studio name at its actual bend line; tapping a
 *   bend (tag, list row or pip) focuses it and opens a callout card on a
 *   leader line (expected -> actual, flange first, verdict, what to do);
 *   the heatmap is an optional layer, off by default;
 * - the bend list on the other side: name, verdict chip, flange A / B and
 *   angle deltas with tolerance bars; not-yet-formed bends greyed;
 * - a footer with the live job status, recent scans and the PDF download.
 *
 * Data: `useStationFeed` (ETag polling, optional SSE). Every result is
 * re-validated with the contract rules before it is shown, and cross-checked
 * against its envelope (job id, station, job summary state); a result that
 * fails is never rendered as a verdict. History jobs open read-only and the
 * live feed keeps running underneath; the screen returns to live by itself
 * as soon as a newer part completes, or after `historyIdleMs` (60 s) with no
 * touch, click or key. While a new scan is being measured, or once the feed
 * has been unreachable for `offlineStaleMs` (10 s), the verdict on screen is
 * dimmed and a notice says what it is (`data-stale=active|offline`). Hebrew
 * switches the whole screen to right-to-left (the grid mirrors; numbers stay
 * left-to-right).
 *
 * Held-back scans: when the hotfolder quarantines a scan the footer ribbon
 * offers "Choose part", which opens the part picker (`PartPicker`); the
 * held list is read from `GET /quarantine/held` and refreshed with the feed.
 * A job whose part was assigned by an operator says so in the footer
 * (`GET /jobs/{id}` `assignment`).
 *
 * Re-rendering: the 1 Hz clocks live in small self-ticking components
 * (header clock, elapsed time in the ribbon); the viewer and the table are
 * memoised, so an idle kiosk does not re-render the 3D scene every second.
 */
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Download, FlipHorizontal2, Wifi, WifiOff, Radio, Loader2, Unplug, UserCheck } from 'lucide-react'

import '../station.css'
import {
  artifactUrl,
  fetchArtifact,
  getHeldScans,
  getJobAssignment,
  getJobResult,
  getZeroshotOverlay,
  getChiralityEyeCheck,
  readToken,
  saveBlob,
  saveDeviceToken,
  StationApiError,
} from '../api/stationApi'
import { stageStatusText, t, validityText } from '../copy'
import BrandLogo from '../../../components/BrandLogo'
import { BendList } from '../components/BendList'
import { HeatmapLegend } from '../components/HeatmapLegend'
import { HistoryStrip } from '../components/HistoryStrip'
import { LayerToggles } from '../components/LayerToggles'
import { OverlayPanel } from '../components/OverlayPanel'
import { PartPicker } from '../components/PartPicker'
import { StationAuthGate, type AuthGateReason } from '../components/StationAuthGate'
import { StationViewer, type StationViewerHandle } from '../components/StationViewer'
import { StatusRibbon } from '../components/StatusRibbon'
import { StageBox, VerdictBanner } from '../components/TopBanner'
import { useStationFeed } from '../hooks/useStationFeed'
import { useViewerAssets, type ViewerAssetSources } from '../hooks/useViewerAssets'
import { loadLayers, saveLayers, type LayerState } from '../model/layers'
import { checkEnvelope, isIncompletePart, stageStrip, summaryBanner, toRows, validateResult, worstOrder } from '../model/stationView'
import type { Lang, LineJobSummary, ShopfloorInspectionResult } from '../model/types'
import { DISPLAY_TOKENS } from '../model/vocabulary'
import { effectiveSummaryState } from '../model/zeroshotOverlay'

const STATION_ID_RE = /^[a-z0-9-]{1,32}$/
const LANG_KEY = 'sherman.station.lang'
/** A history view returns to live after this long without interaction. */
export const HISTORY_IDLE_MS = 60_000
/** The verdict is marked possibly outdated once the feed is unreachable this long. */
export const OFFLINE_STALE_MS = 10_000

function loadLang(): Lang {
  try {
    return window.localStorage.getItem(LANG_KEY) === 'he' ? 'he' : 'en'
  } catch {
    return 'en'
  }
}

function saveLang(lang: Lang) {
  try {
    window.localStorage.setItem(LANG_KEY, lang)
  } catch {
    // storage blocked: language resets on reload
  }
}

function hhmmss(ms: number | null): string {
  if (ms == null) return '—'
  return new Date(ms).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

/** Self-ticking header clock (the only per-second render on an idle screen). */
const HeaderClock = memo(function HeaderClock() {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [])
  return <span className="st-clock st-num" dir="ltr">{hhmmss(now)}</span>
})

/** True once wall-clock time has reached `deadline` (null: never). */
function useDeadlinePassed(deadline: number | null): boolean {
  const [firedFor, setFiredFor] = useState<number | null>(null)
  useEffect(() => {
    if (deadline == null) return
    const id = window.setTimeout(() => setFiredFor(deadline), Math.max(0, deadline - Date.now()))
    return () => window.clearTimeout(id)
  }, [deadline])
  return deadline != null && firedFor === deadline
}

interface HistoryView {
  job: LineJobSummary
  /** latest_completed job id when history was opened; a newer one ends the view. */
  latestAtOpen: string | null
}

interface Shown {
  key: string
  jobId: string
  job: LineJobSummary | null
  result: ShopfloorInspectionResult
  sources: ViewerAssetSources
  pdfUrl: string | null
  live: boolean
}

function liveSources(jobId: string, artifacts: Record<string, string>): ViewerAssetSources {
  return {
    jobId,
    stageMesh: artifacts['stage_mesh.ply'] ?? null,
    finalMesh: artifacts['final_mesh.ply'] ?? null,
    scan: artifacts['viewer_scan.ply'] ?? null,
    deviations: artifacts['viewer_deviations.f32'] ?? null,
  }
}

function historySources(jobId: string, result: ShopfloorInspectionResult): ViewerAssetSources {
  const va = result.viewer_assets
  return {
    jobId,
    stageMesh: va.stage_mesh ? artifactUrl(jobId, 'stage_mesh.ply') : null,
    finalMesh: va.final_mesh ? artifactUrl(jobId, 'final_mesh.ply') : null,
    scan: va.registered_scan ? artifactUrl(jobId, 'viewer_scan.ply') : null,
    deviations: va.deviations ? artifactUrl(jobId, 'viewer_deviations.f32') : null,
  }
}

export interface StationPageProps {
  /** Overrides the `:stationId` route parameter (used by the dev harness). */
  stationId?: string
  /** History view returns to live after this long without interaction (default 60 s). */
  historyIdleMs?: number
  /** Feed unreachable this long marks the verdict as possibly outdated (default 10 s). */
  offlineStaleMs?: number
  /** A running job without progress this long is flagged as possibly stalled (default 2 min). */
  stallAfterMs?: number
}

export default function StationPage({
  stationId: stationProp,
  historyIdleMs = HISTORY_IDLE_MS,
  offlineStaleMs = OFFLINE_STALE_MS,
  stallAfterMs,
}: StationPageProps) {
  const params = useParams<{ stationId: string }>()
  const stationId = (stationProp ?? params.stationId ?? '').trim()
  const validStation = STATION_ID_RE.test(stationId)

  const [lang, setLang] = useState<Lang>(loadLang)
  const toggleLang = useCallback(() => {
    setLang(prev => {
      const next: Lang = prev === 'en' ? 'he' : 'en'
      saveLang(next)
      return next
    })
  }, [])

  const [tokenVersion, setTokenVersion] = useState(0)
  // Read on every render (cheap): a saved or cleared token takes effect at once.
  const token = validStation ? readToken(stationId) : null
  const feedState = useStationFeed(stationId, { enabled: validStation && !!token, tokenVersion })
  const { feed, connection, lastUpdated, serverOffsetMs, authError, error } = feedState

  // Layer toggles are per station: reloaded when the station changes.
  const [layerState, setLayerState] = useState<{ stationId: string; layers: LayerState }>(() => ({
    stationId,
    layers: loadLayers(stationId),
  }))
  if (layerState.stationId !== stationId) setLayerState({ stationId, layers: loadLayers(stationId) })
  const layers = layerState.layers
  const setLayers = useCallback(
    (next: LayerState) => {
      setLayerState({ stationId, layers: next })
      saveLayers(stationId, next)
    },
    [stationId],
  )

  // --- live and history results ---------------------------------------
  const latest = feed?.latest_completed ?? null
  const latestId = latest?.job.job_id ?? null
  const liveValidation = useMemo(() => {
    if (!latest) return null
    const v = validateResult(latest.result)
    if (!v.ok) return v
    const envelope = checkEnvelope(v.result, {
      jobId: latest.job.job_id,
      stationId,
      summaryState: latest.job.summary_state,
      feedStationId: feed?.station_id ?? null,
    })
    return envelope.length ? { ok: false as const, result: null, errors: envelope } : v
  }, [latest, stationId, feed?.station_id])

  const [history, setHistory] = useState<HistoryView | null>(null)
  // A newer completed part always takes the screen back to live.
  if (history && history.latestAtOpen !== latestId) setHistory(null)
  const historyJob = history?.job ?? null
  const openHistory = useCallback(
    (job: LineJobSummary) => setHistory(job.job_id === latestId ? null : { job, latestAtOpen: latestId }),
    [latestId],
  )
  const backToLive = useCallback(() => setHistory(null), [])

  // ... and so does leaving it untouched for historyIdleMs.
  const lastInteraction = useRef(0)
  useEffect(() => {
    const mark = () => {
      lastInteraction.current = Date.now()
    }
    const events = ['pointerdown', 'keydown', 'wheel', 'touchstart'] as const
    for (const e of events) window.addEventListener(e, mark, { capture: true, passive: true })
    return () => {
      for (const e of events) window.removeEventListener(e, mark, { capture: true })
    }
  }, [])
  useEffect(() => {
    if (!history) return
    lastInteraction.current = Math.max(lastInteraction.current, Date.now())
    const id = window.setInterval(() => {
      if (Date.now() - lastInteraction.current >= historyIdleMs) setHistory(null)
    }, 500)
    return () => window.clearInterval(id)
  }, [history, historyIdleMs])

  const historyQuery = useQuery({
    queryKey: ['station-job-result', stationId, historyJob?.job_id ?? null, tokenVersion],
    enabled: !!historyJob && !!token,
    staleTime: Infinity,
    retry: 1,
    refetchOnWindowFocus: false,
    queryFn: ({ signal }) => getJobResult(stationId, (historyJob as LineJobSummary).job_id, signal),
  })

  const historyEnvelopeErrors = useMemo(
    () => (historyJob && historyQuery.data ? checkEnvelope(historyQuery.data, { jobId: historyJob.job_id, stationId, summaryState: historyJob.summary_state }) : []),
    [historyJob, historyQuery.data, stationId],
  )

  const shown: Shown | null = useMemo(() => {
    if (historyJob && historyQuery.data) {
      if (historyEnvelopeErrors.length) return null
      const result = historyQuery.data
      return {
        key: `history:${historyJob.job_id}`,
        jobId: historyJob.job_id,
        job: historyJob,
        result,
        sources: historySources(historyJob.job_id, result),
        pdfUrl: result.artifacts.pdf ? artifactUrl(historyJob.job_id, 'report.pdf') : null,
        live: false,
      }
    }
    if (!historyJob && latest && liveValidation?.ok) {
      return {
        key: `live:${latest.job.job_id}`,
        jobId: latest.job.job_id,
        job: latest.job,
        result: liveValidation.result,
        sources: liveSources(latest.job.job_id, latest.artifacts),
        pdfUrl: latest.artifacts['report.pdf'] ?? null,
        live: true,
      }
    }
    return null
  }, [historyJob, historyQuery.data, historyEnvelopeErrors, latest, liveValidation])

  // A new arrival (any job after the first one this screen saw) is highlighted.
  const [firstSeenId, setFirstSeenId] = useState<string | null>(null)
  if (latestId && firstSeenId === null) setFirstSeenId(latestId)
  const [announcedId, setAnnouncedId] = useState<string | null>(null)
  const isNewArrival =
    !!shown?.live && !!latestId && firstSeenId !== null && latestId !== firstSeenId && announcedId !== latestId
  // The arrival highlight runs once (3 s), not again when returning from history.
  useEffect(() => {
    if (!isNewArrival) return
    const id = window.setTimeout(() => setAnnouncedId(latestId), 3000)
    return () => window.clearTimeout(id)
  }, [isNewArrival, latestId])

  // Zero-shot overlay of the job on screen (display / REVIEW only; absent = E3 alone).
  const overlayQuery = useQuery({
    queryKey: ['station-zs-overlay', stationId, shown?.jobId ?? null, shown?.result.scan.sha256 ?? null, tokenVersion],
    enabled: !!shown && !!token,
    staleTime: Infinity,
    retry: 1,
    refetchOnWindowFocus: false,
    queryFn: ({ signal }) => getZeroshotOverlay(stationId, (shown as Shown).jobId, (shown as Shown).result, signal),
  })
  const overlay = shown && overlayQuery.data ? overlayQuery.data : null

  // Left/right-sensitive part (display only): an eye-check reminder until the engine checks it.
  const chiralityQuery = useQuery({
    queryKey: ['station-chirality', stationId, shown?.result.recipe.recipe_id ?? null, tokenVersion],
    enabled: !!shown && !!token,
    staleTime: 5 * 60_000,
    retry: false,
    refetchOnWindowFocus: false,
    queryFn: ({ signal }) => getChiralityEyeCheck(stationId, (shown as Shown).result.recipe.recipe_id, signal),
  })
  const eyeCheck = !!shown && chiralityQuery.data === true

  const result = shown?.result ?? null
  const rows = useMemo(() => (result ? toRows(result, lang) : []), [result, lang])
  const banner = useMemo(() => {
    if (!result) return null
    const b = summaryBanner(result, lang, { validity: c => validityText(c, lang), stage: c => stageStatusText(c, lang) })
    // The overlay may only lower a pass-like verdict to REVIEW (an unexpected bend); it never makes one.
    if (!overlay || !overlay.verdict_lowered) return b
    const tok = DISPLAY_TOKENS[overlay.effective_summary_state]
    return { ...b, state: overlay.effective_summary_state, tone: tok.tone, color: tok.color, title: tok[lang], details: [...b.details, t('zsReasonUnexpected', lang)] }
  }, [result, lang, overlay])
  const bendLabels = useMemo(() => Object.fromEntries(rows.map(r => [r.bendUid, r.label])), [rows])
  const strip = useMemo(() => (result ? stageStrip(result) : null), [result])
  const incomplete = result ? isIncompletePart(result) : false
  const assets = useViewerAssets(stationId, result, shown?.sources ?? null, incomplete)

  // Selection and worst-first cycling are scoped to the result on screen.
  const viewKey = shown?.key ?? 'none'
  const [selection, setSelection] = useState<{ key: string; uid: string | null }>({ key: 'none', uid: null })
  const selectedUid = selection.key === viewKey ? selection.uid : null
  const [worstCursor, setWorstCursor] = useState<{ key: string; index: number }>({ key: 'none', index: 0 })
  const worstIndex = worstCursor.key === viewKey ? worstCursor.index : 0
  const viewerRef = useRef<StationViewerHandle>(null)

  const selectBend = useCallback(
    (uid: string) => {
      setSelection({ key: viewKey, uid })
      viewerRef.current?.focusBend(uid)
    },
    [viewKey],
  )
  const order = useMemo(() => (result ? worstOrder(result) : []), [result])
  const focusWorst = useCallback(() => {
    if (!order.length) return
    const uid = order[worstIndex % order.length]
    setWorstCursor({ key: viewKey, index: worstIndex + 1 })
    selectBend(uid)
  }, [order, worstIndex, viewKey, selectBend])
  const clearSelection = useCallback(() => setSelection({ key: viewKey, uid: null }), [viewKey])
  const fitAll = useCallback(() => {
    setSelection({ key: viewKey, uid: null })
    viewerRef.current?.fitAll()
  }, [viewKey])

  // --- staleness of the verdict on screen --------------------------------
  const activeJob = feed?.active?.job ?? null

  // --- held-back scans (part picker) -------------------------------------
  // Re-read whenever the feed's issue or active job changes, and every 15 s.
  const queryClient = useQueryClient()
  const heldQuery = useQuery({
    queryKey: ['station-held', stationId, tokenVersion, feed?.latest_issue?.at ?? null, activeJob?.job_id ?? null],
    enabled: validStation && !!token && !!feed,
    refetchInterval: 15_000,
    retry: 1,
    queryFn: ({ signal }) => getHeldScans(stationId, signal),
  })
  const heldItems = heldQuery.data?.items ?? []
  const [pickerOpen, setPickerOpen] = useState(false)
  const openPicker = useCallback(() => setPickerOpen(true), [])
  const closePicker = useCallback(() => setPickerOpen(false), [])
  const refreshHeld = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: ['station-held', stationId] })
  }, [queryClient, stationId])

  // Was the part of the job on screen assigned by an operator? (display only)
  const assignmentQuery = useQuery({
    queryKey: ['station-job-assignment', stationId, shown?.jobId ?? null, tokenVersion],
    enabled: !!shown && !!token,
    staleTime: Infinity,
    retry: 1,
    refetchOnWindowFocus: false,
    queryFn: ({ signal }) => getJobAssignment(stationId, (shown as Shown).jobId, signal),
  })
  const assignedByOperator = !!shown && assignmentQuery.data?.by_operator === true
  const offlineStale = useDeadlinePassed(connection === 'offline' && lastUpdated != null ? lastUpdated + offlineStaleMs : null)
  const stale: 'offline' | 'active' | null = !shown?.live ? null : offlineStale ? 'offline' : activeJob ? 'active' : null

  // History chips: the job on screen (and a refused latest) use the validated verdict.
  const chipOverrides = useMemo(() => {
    const out: Record<string, string | null> = {}
    if (latest) out[latest.job.job_id] = liveValidation?.ok ? liveValidation.result.summary.state : null
    if (shown) out[shown.jobId] = effectiveSummaryState(shown.result, overlay)
    return out
  }, [latest, liveValidation, shown, overlay])

  // --- PDF download --------------------------------------------------------
  const [pdfState, setPdfState] = useState<'idle' | 'busy' | 'failed'>('idle')
  const downloadPdf = useCallback(async () => {
    if (!shown?.pdfUrl) return
    setPdfState('busy')
    try {
      const art = await fetchArtifact(stationId, shown.pdfUrl)
      const serial = shown.result.context.serial ?? shown.jobId.slice(5, 13)
      saveBlob(art.data, art.fileName ?? `${shown.result.sku}_${serial}.pdf`, 'application/pdf')
      setPdfState('idle')
    } catch {
      setPdfState('failed')
    }
  }, [shown, stationId])

  // --- gates and empty states ------------------------------------------------
  const dir = lang === 'he' ? 'rtl' : 'ltr'
  if (!validStation) {
    return (
      <div className="st-root st-root-message" dir={dir} lang={lang}>
        <p>{t('station', lang)}: <span dir="ltr">{stationId || '—'}</span></p>
      </div>
    )
  }
  const gateReason: AuthGateReason | null = !token
    ? 'missing'
    : authError
      ? authError.status === 403 && authError.code !== 'TOKEN_INVALID' ? 'forbidden' : 'rejected'
      : null
  if (gateReason) {
    return (
      <div className="st-root st-root-gate" dir={dir} lang={lang}>
        <StationAuthGate
          lang={lang}
          stationId={stationId}
          reason={gateReason}
          onToggleLang={toggleLang}
          onSubmit={tok => {
            saveDeviceToken(stationId, tok)
            setTokenVersion(v => v + 1)
          }}
        />
      </div>
    )
  }

  const notConfigured = error instanceof StationApiError && error.status === 503
  // Part identity: from the result on screen, else from the job summary (never from a refused document).
  const identity = shown
    ? { sku: shown.result.sku, partName: shown.result.part_name, serial: shown.result.context.serial, version: shown.result.recipe.version }
    : latest
      ? { sku: latest.job.sku, partName: latest.job.part_name, serial: latest.job.serial, version: latest.job.recipe_version }
      : null
  const ConnIcon = connection === 'live' ? Radio : connection === 'polling' ? Wifi : WifiOff
  const shownSerial = shown ? (shown.result.context.serial ?? shown.jobId.slice(5, 13)) : ''
  const activeSerial = activeJob ? (activeJob.serial ?? activeJob.scan_file_name ?? activeJob.job_id.slice(5, 13)) : ''

  let centerMessage: string | null = null
  if (!feed) centerMessage = notConfigured ? t('notConfigured', lang) : error ? t('feedError', lang) : t('connecting', lang)
  else if (historyJob && (historyQuery.isError || historyEnvelopeErrors.length > 0)) centerMessage = t('resultRejected', lang)
  else if (historyJob && !historyQuery.data) centerMessage = t('loadingResult', lang)
  else if (!latest && !historyJob) centerMessage = t('waitingFirst', lang)
  else if (!historyJob && liveValidation && !liveValidation.ok) centerMessage = t('resultRejected', lang)

  return (
    <div className="st-root" dir={dir} lang={lang} data-testid="station-root" data-connection={connection} data-stale={stale ?? undefined}>
      <header className="st-header">
        <div className="st-brand">
          <BrandLogo height={46} alt={t('logoAlt', lang)} />
        </div>
        <div className="st-ident">
          <span className="st-sku" dir="ltr">{identity?.sku ?? '—'}</span>
          <span className="st-partname" dir="auto" title={identity?.partName ?? undefined}>{identity?.partName ?? ''}</span>
          <dl className="st-meta">
            <div>
              <dt>{t('serial', lang)}</dt>
              <dd dir="ltr">{identity?.serial ?? '—'}</dd>
            </div>
            <div>
              <dt>{t('recipe', lang)}</dt>
              <dd dir="ltr">{identity ? `v${identity.version}` : '—'}</dd>
            </div>
            <div>
              <dt>{t('job', lang)}</dt>
              <dd dir="ltr" className="st-jobid" data-testid="station-job-id" data-job-id={shown?.jobId ?? ''} title={shown?.jobId ?? ''}>
                {shown ? shown.jobId.slice(5, 13) : '—'}
              </dd>
            </div>
          </dl>
        </div>
        {shown && result && banner && strip ? (
          <div className={`st-header-result${stale ? ' is-stale' : ''}`}>
            <StageBox lang={lang} strip={strip} result={result} selectedUid={selectedUid} />
            <VerdictBanner key={`b-${shown.key}`} lang={lang} banner={banner} rows={rows} result={result} isNew={isNewArrival && !stale} />
          </div>
        ) : (
          <div className="st-header-result" aria-hidden />
        )}
        <div className="st-header-end">
          <button type="button" className="st-lang" onClick={toggleLang} aria-label={t('langToggleLabel', lang)} data-testid="lang-toggle">
            {t('langToggle', lang)}
          </button>
          <div className="st-header-status">
            <HeaderClock />
            <span className={`st-conn st-conn-${connection}`} data-testid="connection" title={`${t('updated', lang)} ${hhmmss(lastUpdated)}`}>
              <ConnIcon size={20} aria-hidden />
              <span>{t(connection === 'live' ? 'connLive' : connection === 'polling' ? 'connPolling' : 'connOffline', lang)}</span>
            </span>
            <span className="st-station">
              {t('station', lang)} <strong dir="ltr">{stationId}</strong>
            </span>
          </div>
        </div>
      </header>

      <main className="st-stage-area">
        {shown && result ? (
          <>
            <StationViewer
              ref={viewerRef}
              lang={lang}
              viewKey={viewKey}
              rows={rows}
              assets={assets}
              layers={layers}
              heatmapBand={result.viewer_assets.heatmap_band_mm}
              selectedUid={selectedUid}
              onSelectBend={selectBend}
              onClearSelection={clearSelection}
            />
            <div className="st-viewer-top">
              {eyeCheck && (
                <div className="st-flag st-flag-chiral" role="status" data-testid="chirality-flag" data-viewer-reserved>
                  <FlipHorizontal2 size={28} aria-hidden />
                  <span className="st-flag-chiral-text">
                    <strong>{t('chiralityCheck', lang)}</strong>
                    <small>{t('chiralityCheckDetail', lang)}</small>
                  </span>
                </div>
              )}
              {!selectedUid && (
                <div className="st-flag st-flag-hint" data-testid="tap-hint" data-viewer-reserved>
                  <span>{t('tapBendHint', lang)}</span>
                </div>
              )}
              {layers.cad && !assets.stageMesh && assets.finalMesh && (
                <div className="st-flag st-flag-note" data-testid="cad-fallback-flag" data-viewer-reserved>
                  <span>{t('cadFinishedFallback', lang)}</span>
                </div>
              )}
              {!shown.live && (
                <div className="st-flag st-flag-history" data-testid="history-flag" data-viewer-reserved>
                  <span>{t('viewingEarlier', lang)}</span>
                  <button type="button" onClick={backToLive}>{t('backToLive', lang)}</button>
                </div>
              )}
              {!shown.live && latestId && latestId !== shown.jobId && <div className="st-flag" data-viewer-reserved>{t('newerAvailable', lang)}</div>}
            </div>
            <HeatmapLegend
              lang={lang}
              band={result.viewer_assets.heatmap_band_mm}
              issue={assets.heatmapIssue}
              loading={assets.heatmapLoading}
              active={layers.heatmap && layers.scan}
            />
            <LayerToggles
              lang={lang}
              layers={layers}
              onChange={setLayers}
              disabled={{
                // The ghost needs a stage model to contrast with; without one the finished part is the CAD layer.
                ghost: !incomplete || !shown.sources.stageMesh,
                scan: !shown.sources.scan,
                heatmap: !!assets.heatmapIssue || !layers.scan || !shown.sources.scan,
              }}
              onWorst={focusWorst}
              worstLabel={worstIndex === 0 ? t('focusWorst', lang) : t('focusNext', lang)}
              worstDisabled={!order.length}
              onFit={fitAll}
            />
          </>
        ) : (
          <div className="st-empty" data-testid="station-empty">
            <p>{centerMessage}</p>
            {historyJob && (
              <button type="button" className="st-lang" onClick={backToLive}>
                {t('backToLive', lang)}
              </button>
            )}
          </div>
        )}
      </main>

      <aside className={`st-panel${stale ? ' is-stale' : ''}`} data-testid="station-panel" data-stale={stale ?? undefined}>
        {shown && result ? (
          <>
            {stale && (
              <div className={`st-stale st-stale-${stale}`} data-testid="stale-notice" data-stale={stale} role="status">
                {stale === 'active' ? <Loader2 className="st-spin" size={30} aria-hidden /> : <Unplug size={30} aria-hidden />}
                <div className="st-stale-text">
                  {stale === 'active' ? (
                    <>
                      <strong>
                        {t('staleMeasuring', lang)} <bdi dir="ltr">{activeSerial}</bdi>…
                      </strong>
                      <span>
                        {t('stalePrevious', lang)} <bdi dir="ltr">{shownSerial}</bdi>
                      </span>
                    </>
                  ) : (
                    <>
                      <strong>
                        {t('staleOffline', lang)} <bdi className="st-num" dir="ltr">{hhmmss(lastUpdated)}</bdi>
                      </strong>
                      <span>
                        {t('staleOutdated', lang)} (<bdi dir="ltr">{shownSerial}</bdi>)
                      </span>
                    </>
                  )}
                </div>
              </div>
            )}
            <div className="st-panel-result">
              <BendList
                lang={lang}
                rows={rows}
                generalMeasurements={result.general_measurements}
                selectedUid={selectedUid}
                onSelect={selectBend}
                inspectionInvalid={result.inspection_validity.state !== 'VALID'}
                footer={overlay ? <OverlayPanel lang={lang} overlay={overlay} labels={bendLabels} /> : null}
              />
            </div>
          </>
        ) : (
          <div className="st-panel-empty" aria-hidden />
        )}
      </aside>

      <footer className="st-footer">
        <StatusRibbon
          lang={lang}
          active={activeJob}
          issue={feed?.latest_issue ?? null}
          shownFinishedAt={latest?.job.finished_at ?? null}
          serverOffsetMs={serverOffsetMs}
          stallAfterMs={stallAfterMs}
          heldCount={heldItems.length}
          heldResolvable={heldItems.some(i => i.resolvable)}
          onChoosePart={openPicker}
        />
        <HistoryStrip
          lang={lang}
          jobs={feed?.recent ?? []}
          shownJobId={shown?.jobId ?? null}
          verifiedStates={chipOverrides}
          onOpen={openHistory}
        />
        {assignedByOperator && (
          <span className="st-assigned-badge" data-testid="assigned-by-operator">
            <UserCheck size={20} aria-hidden />
            <span>{t('assignedByOperator', lang)}</span>
          </span>
        )}
        <span className="st-assist-badge" data-testid="operator-assist">{t('operatorAssist', lang)}</span>
        <button
          type="button"
          className="st-pdf"
          onClick={() => void downloadPdf()}
          disabled={!shown?.pdfUrl || pdfState === 'busy'}
          data-testid="download-pdf"
          title={!shown?.pdfUrl ? t('pdfNotReady', lang) : undefined}
        >
          <Download size={24} aria-hidden />
          <span>
            {pdfState === 'busy' ? t('downloading', lang) : pdfState === 'failed' ? t('pdfFailed', lang) : shown?.pdfUrl ? t('downloadPdf', lang) : t('pdfNotReady', lang)}
          </span>
        </button>
      </footer>
      {pickerOpen && (
        <PartPicker
          lang={lang}
          stationId={stationId}
          items={heldItems}
          canOperate={heldQuery.data?.canOperate === true}
          onClose={closePicker}
          onResolved={refreshHeld}
        />
      )}
    </div>
  )
}
