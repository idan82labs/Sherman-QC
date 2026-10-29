import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  Box,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleDot,
  Crosshair,
  FileScan,
  Loader2,
  Save,
  Search,
  Trash2,
} from 'lucide-react'
import clsx from 'clsx'

import { getErrorMessage } from '../../../services/api'
import { useAuth } from '../../../hooks/useAuth'
import { bendLabelingApi } from '../api'
import { addBend, removeBend, replaceEndpoint } from '../labelState'
import type {
  CameraPreset,
  EndpointEdit,
  HumanBendLabels,
  LabelStatus,
  PlyAsset,
  Point3,
} from '../types'

const BendPointCloudViewer = lazy(() => import('../components/BendPointCloudViewer'))
const EMPTY_ASSETS: PlyAsset[] = []

const SOURCE_LABELS: Record<PlyAsset['source'], string> = {
  'source-assets': 'Verified source',
  data: 'Corpus intake',
  uploads: 'Uploads',
  live_scans: 'Live scans',
  output: 'Derived output',
}

const STATUS_LABELS: Record<LabelStatus, string> = {
  not_started: 'Not started',
  in_progress: 'In progress',
  complete: 'Complete',
  needs_review: 'Needs review',
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatPoint(point: Point3) {
  return point.map((value) => value.toFixed(3)).join(', ')
}

function statusClass(status: LabelStatus) {
  return clsx(
    status === 'complete' && 'text-emerald-300',
    status === 'in_progress' && 'text-amber-300',
    status === 'needs_review' && 'text-red-300',
    status === 'not_started' && 'text-slate-400',
  )
}

export default function BendLabelingPage() {
  const queryClient = useQueryClient()
  const { user } = useAuth()
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [sourceFilter, setSourceFilter] = useState<'queue' | 'all' | PlyAsset['source']>('queue')
  const [statusFilter, setStatusFilter] = useState<'all' | LabelStatus>('all')
  const [draftPoint, setDraftPoint] = useState<Point3 | null>(null)
  const [endpointEdit, setEndpointEdit] = useState<EndpointEdit>(null)
  const [selectedBendId, setSelectedBendId] = useState<string | null>(null)
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>('iso')
  const [pointSize, setPointSize] = useState(2)
  const [workingLabels, setWorkingLabels] = useState<HumanBendLabels | null>(null)
  const [dirty, setDirty] = useState(false)
  const [viewerLoading, setViewerLoading] = useState(false)
  const [viewerError, setViewerError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [savedFlash, setSavedFlash] = useState(false)
  const hydratedRevisionRef = useRef<string | null>(null)

  const assetsQuery = useQuery({
    queryKey: ['bend-labeling-assets'],
    queryFn: bendLabelingApi.listAssets,
    staleTime: 15_000,
  })
  const assets = assetsQuery.data?.assets ?? EMPTY_ASSETS
  const filteredAssets = useMemo(() => {
    const needle = search.trim().toLowerCase()
    return assets.filter((asset) => {
      const sourceMatches = sourceFilter === 'all'
        || (sourceFilter === 'queue'
          ? asset.source !== 'output' && asset.asset_kind !== 'reference_mesh'
          : asset.source === sourceFilter)
      const statusMatches = statusFilter === 'all' || asset.label_status === statusFilter
      const searchMatches = !needle
        || asset.name.toLowerCase().includes(needle)
        || asset.relative_path.toLowerCase().includes(needle)
      return sourceMatches && statusMatches && searchMatches
    })
  }, [assets, search, sourceFilter, statusFilter])

  const activeAssetId = selectedAssetId ?? filteredAssets[0]?.asset_id ?? null
  const selectedAsset = assets.find((asset) => asset.asset_id === activeAssetId) ?? null
  const labelsQuery = useQuery({
    queryKey: ['bend-labeling-labels', activeAssetId],
    queryFn: () => bendLabelingApi.labels(activeAssetId!),
    enabled: Boolean(activeAssetId),
    staleTime: 0,
  })

  useEffect(() => {
    if (!labelsQuery.data) return
    const hydrationKey = `${labelsQuery.data.asset_id}:${labelsQuery.data.revision}`
    if (hydratedRevisionRef.current === hydrationKey) return
    hydratedRevisionRef.current = hydrationKey
    const fallbackName = user?.username || ''
    const storedName = localStorage.getItem('bend-labeling-annotator') || fallbackName
    setWorkingLabels({
      ...labelsQuery.data,
      annotator: labelsQuery.data.annotator || storedName,
    })
    setDraftPoint(null)
    setEndpointEdit(null)
    setSelectedBendId(labelsQuery.data.bends[0]?.bend_id ?? null)
    setDirty(false)
    setActionError(null)
    setSavedFlash(false)
  }, [labelsQuery.data, user?.username])

  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => {
      if (!dirty) return
      event.preventDefault()
      event.returnValue = ''
    }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [dirty])

  const saveMutation = useMutation({
    mutationFn: (labels: HumanBendLabels) => bendLabelingApi.save(labels.asset_id, labels),
    onSuccess: (saved) => {
      hydratedRevisionRef.current = `${saved.asset_id}:${saved.revision}`
      setWorkingLabels(saved)
      setDirty(false)
      setSavedFlash(true)
      setActionError(null)
      queryClient.setQueryData(['bend-labeling-labels', saved.asset_id], saved)
      queryClient.invalidateQueries({ queryKey: ['bend-labeling-assets'] })
    },
    onError: (error) => setActionError(getErrorMessage(error)),
  })

  const updateLabels = useCallback((updater: (labels: HumanBendLabels) => HumanBendLabels) => {
    setWorkingLabels((current) => current ? updater(current) : current)
    setDirty(true)
    setSavedFlash(false)
    setActionError(null)
  }, [])

  const handlePick = useCallback((point: Point3) => {
    if (!workingLabels) return
    if (endpointEdit) {
      updateLabels((labels) => replaceEndpoint(labels, endpointEdit.bendId, endpointEdit.endpoint, point))
      setSelectedBendId(endpointEdit.bendId)
      setEndpointEdit(null)
      return
    }
    if (!draftPoint) {
      setDraftPoint(point)
      return
    }
    try {
      const next = addBend(workingLabels, draftPoint, point)
      const added = next.bends[next.bends.length - 1]
      setWorkingLabels(next)
      setSelectedBendId(added.bend_id)
      setDirty(true)
      setSavedFlash(false)
      setActionError(null)
      setDraftPoint(null)
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Could not add bend line.')
    }
  }, [draftPoint, endpointEdit, updateLabels, workingLabels])

  const selectAsset = useCallback((assetId: string) => {
    if (assetId === activeAssetId) return
    if (dirty && !window.confirm('Discard unsaved changes and open another PLY?')) return
    setSelectedAssetId(assetId)
    hydratedRevisionRef.current = null
    setWorkingLabels(null)
    setDraftPoint(null)
    setEndpointEdit(null)
    setViewerError(null)
  }, [activeAssetId, dirty])

  const selectRelative = useCallback((direction: -1 | 1) => {
    if (!activeAssetId || !filteredAssets.length) return
    const index = filteredAssets.findIndex((asset) => asset.asset_id === activeAssetId)
    const nextIndex = Math.min(filteredAssets.length - 1, Math.max(0, index + direction))
    selectAsset(filteredAssets[nextIndex].asset_id)
  }, [activeAssetId, filteredAssets, selectAsset])

  const save = useCallback((complete = false) => {
    if (!workingLabels) return
    if (!workingLabels.annotator.trim()) {
      setActionError('Enter the annotator name before saving.')
      return
    }
    if (complete && !workingLabels.bends.length && !workingLabels.no_bends_present) {
      setActionError('Add at least one bend, or explicitly mark this part as having no bends.')
      return
    }
    localStorage.setItem('bend-labeling-annotator', workingLabels.annotator.trim())
    saveMutation.mutate({
      ...workingLabels,
      annotator: workingLabels.annotator.trim(),
      review_status: complete ? 'complete' : workingLabels.review_status,
    })
  }, [saveMutation, workingLabels])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      const typing = target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA'
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
        event.preventDefault()
        save(false)
      } else if (!typing && event.key === 'Escape') {
        setDraftPoint(null)
        setEndpointEdit(null)
      } else if (!typing && (event.key === 'Delete' || event.key === 'Backspace') && selectedBendId) {
        event.preventDefault()
        updateLabels((labels) => removeBend(labels, selectedBendId))
        setSelectedBendId(null)
      } else if (!typing && event.key === 'ArrowRight') {
        selectRelative(1)
      } else if (!typing && event.key === 'ArrowLeft') {
        selectRelative(-1)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [save, selectRelative, selectedBendId, updateLabels])

  const currentIndex = activeAssetId
    ? filteredAssets.findIndex((asset) => asset.asset_id === activeAssetId)
    : -1
  const completedCount = assets.filter((asset) => asset.label_status === 'complete').length

  return (
    <div className="bend-labeling-root min-h-[100dvh] bg-[#0b1016] text-slate-200">
      <header className="flex min-h-16 flex-wrap items-center gap-4 border-b border-slate-700/80 bg-[#0e151d] px-4 py-2 lg:flex-nowrap lg:px-5">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md border border-amber-400/40 bg-amber-400/10 text-amber-300">
            <Crosshair className="h-5 w-5" aria-hidden="true" />
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-base font-semibold tracking-tight text-slate-50">Sherman Bend Label Studio</h1>
            <p className="text-xs text-slate-400">Human ground truth in source PLY coordinates</p>
          </div>
        </div>
        <div className="ml-auto flex items-center gap-4 font-mono text-xs text-slate-400">
          <span><strong className="text-slate-100">{assetsQuery.data?.inventory_total ?? 0}</strong> PLY files</span>
          <span><strong className="text-emerald-300">{completedCount}</strong> complete</span>
          <span className={dirty ? 'text-amber-300' : 'text-slate-500'}>{dirty ? 'Unsaved changes' : 'Saved state'}</span>
        </div>
      </header>

      <main className="grid min-h-[calc(100dvh-4rem)] grid-cols-1 xl:grid-cols-[320px_minmax(520px,1fr)_360px]">
        <aside className="border-b border-slate-700/80 bg-[#0e151d] xl:border-b-0 xl:border-r" aria-label="PLY queue">
          <div className="space-y-3 border-b border-slate-700/80 p-3">
            <label className="block">
              <span className="sr-only">Search PLY files</span>
              <span className="relative block">
                <Search className="pointer-events-none absolute left-3 top-3 h-4 w-4 text-slate-500" aria-hidden="true" />
                <input
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Part number or filename"
                  className="h-10 w-full rounded-md border border-slate-600 bg-[#111b25] pl-9 pr-3 text-sm text-slate-100 outline-none placeholder:text-slate-500 focus:border-amber-400 focus:ring-2 focus:ring-amber-400/20"
                />
              </span>
            </label>
            <div className="grid grid-cols-2 gap-2">
              <label>
                <span className="sr-only">Asset source</span>
                <select value={sourceFilter} onChange={(event) => setSourceFilter(event.target.value as typeof sourceFilter)} className="h-10 w-full rounded-md border border-slate-600 bg-[#111b25] px-2 text-xs text-slate-200 focus:border-amber-400 focus:outline-none">
                  <option value="queue">Human queue</option>
                  <option value="all">All sources</option>
                  {Object.entries(SOURCE_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                </select>
              </label>
              <label>
                <span className="sr-only">Label status</span>
                <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as typeof statusFilter)} className="h-10 w-full rounded-md border border-slate-600 bg-[#111b25] px-2 text-xs text-slate-200 focus:border-amber-400 focus:outline-none">
                  <option value="all">All states</option>
                  {Object.entries(STATUS_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                </select>
              </label>
            </div>
            <p className="font-mono text-[11px] text-slate-500">{filteredAssets.length} visible / {assets.length} indexed</p>
          </div>

          <div className="max-h-[360px] overflow-y-auto xl:max-h-[calc(100dvh-13rem)]">
            {assetsQuery.isLoading && <div className="p-5 text-sm text-slate-400">Indexing PLY files...</div>}
            {assetsQuery.isError && <div className="m-3 rounded-md border border-red-400/40 bg-red-400/10 p-3 text-sm text-red-200">{getErrorMessage(assetsQuery.error)}</div>}
            {!assetsQuery.isLoading && !filteredAssets.length && <div className="p-5 text-sm text-slate-400">No PLY files match these filters.</div>}
            {filteredAssets.map((asset) => (
              <button
                key={asset.asset_id}
                type="button"
                onClick={() => selectAsset(asset.asset_id)}
                className={clsx(
                  'block w-full border-b border-slate-800 px-3 py-3 text-left transition-colors [content-visibility:auto] [contain-intrinsic-size:72px] focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-amber-400',
                  activeAssetId === asset.asset_id ? 'bg-amber-400/10' : 'hover:bg-slate-800/45',
                )}
              >
                <span className="flex items-start gap-3">
                  {asset.asset_kind === 'reference_mesh' ? <Box className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" /> : <FileScan className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" />}
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium text-slate-100">{asset.name}</span>
                    <span className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-[10px] text-slate-500">
                      <span>{SOURCE_LABELS[asset.source]}</span>
                      <span>{formatBytes(asset.size_bytes)}</span>
                      {asset.vertex_count != null && <span>{asset.vertex_count.toLocaleString()} pts</span>}
                    </span>
                    <span className={clsx('mt-1 block text-[11px]', statusClass(asset.label_status))}>
                      {STATUS_LABELS[asset.label_status]}{asset.bend_count ? `, ${asset.bend_count} bends` : ''}
                    </span>
                  </span>
                </span>
              </button>
            ))}
          </div>
        </aside>

        <section className="min-w-0 bg-[#111821]" aria-label="3D bend annotation viewer">
          <div className="flex min-h-12 flex-wrap items-center gap-2 border-b border-slate-700/80 bg-[#111821] px-3 py-2">
            <div className="mr-auto min-w-0">
              <p className="truncate text-sm font-medium text-slate-100">{selectedAsset?.name ?? 'Select a PLY file'}</p>
              {selectedAsset && <p className="truncate font-mono text-[10px] text-slate-500">{selectedAsset.source}/{selectedAsset.relative_path}</p>}
            </div>
            <div className="flex items-center rounded-md border border-slate-600 bg-[#0e151d] p-0.5" aria-label="Camera views">
              {(['iso', 'front', 'right', 'top'] as CameraPreset[]).map((preset) => (
                <button key={preset} type="button" onClick={() => setCameraPreset(preset)} className={clsx('h-9 min-w-11 rounded px-2 text-xs capitalize focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400', cameraPreset === preset ? 'bg-slate-600 text-white' : 'text-slate-400 hover:text-slate-100')}>
                  {preset}
                </button>
              ))}
            </div>
            <label className="flex h-10 items-center gap-2 rounded-md border border-slate-600 bg-[#0e151d] px-2 text-xs text-slate-400">
              Point size
              <input type="range" min="1" max="5" step="0.5" value={pointSize} onChange={(event) => setPointSize(Number(event.target.value))} className="w-20 accent-amber-400" />
              <span className="w-4 font-mono text-slate-200">{pointSize}</span>
            </label>
          </div>

          <div className="relative h-[58dvh] min-h-[460px] xl:h-[calc(100dvh-7rem)]">
            {selectedAsset && workingLabels && (
              <Suspense fallback={<div className="flex h-full items-center justify-center text-sm text-slate-400">Loading 3D tools...</div>}>
                <BendPointCloudViewer
                  assetUrl={bendLabelingApi.assetUrl(selectedAsset.asset_id)}
                  faceCount={selectedAsset.face_count}
                  bends={workingLabels.bends}
                  draftPoint={draftPoint}
                  selectedBendId={selectedBendId}
                  cameraPreset={cameraPreset}
                  pointSize={pointSize}
                  pickingEnabled
                  onPick={handlePick}
                  onSelectBend={setSelectedBendId}
                  onLoadingChange={setViewerLoading}
                  onError={setViewerError}
                />
              </Suspense>
            )}
            {(labelsQuery.isLoading || viewerLoading) && (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-[#111821]/85">
                <div className="flex items-center gap-3 text-sm text-slate-300"><Loader2 className="h-5 w-5 animate-spin text-amber-300" /> Loading point cloud</div>
              </div>
            )}
            {(labelsQuery.isError || viewerError) && (
              <div className="absolute inset-0 flex items-center justify-center p-6">
                <div role="alert" className="max-w-md rounded-md border border-red-400/40 bg-red-400/10 p-4 text-sm text-red-100">{viewerError || getErrorMessage(labelsQuery.error)}</div>
              </div>
            )}
            {!selectedAsset && !assetsQuery.isLoading && (
              <div className="flex h-full items-center justify-center text-slate-400">Choose a PLY file from the queue.</div>
            )}
            {draftPoint && (
              <div className="absolute left-3 top-3 max-w-sm rounded-md border border-orange-400/50 bg-[#151b22]/95 px-3 py-2 text-xs text-orange-200">
                Start fixed at [{formatPoint(draftPoint)}]. Click the bend endpoint, or press Esc to cancel.
              </div>
            )}
            {endpointEdit && (
              <div className="absolute left-3 top-3 rounded-md border border-amber-400/50 bg-[#151b22]/95 px-3 py-2 text-xs text-amber-100">
                Click a new {endpointEdit.endpoint} point for {endpointEdit.bendId}. Press Esc to cancel.
              </div>
            )}
            <div className="pointer-events-none absolute bottom-3 left-3 rounded-md border border-slate-600 bg-[#0e151d]/95 px-3 py-2 text-[11px] text-slate-300">
              <span className="font-medium text-amber-200">Mark:</span> click start, click end
              <span className="mx-2 text-slate-600">|</span>
              <span>Rotate: left drag</span>
              <span className="mx-2 text-slate-600">|</span>
              <span>Pan: right drag</span>
              <span className="mx-2 text-slate-600">|</span>
              <span>Zoom: wheel</span>
            </div>
          </div>

          <div className="flex min-h-12 items-center justify-between border-t border-slate-700/80 bg-[#0e151d] px-3">
            <button type="button" onClick={() => selectRelative(-1)} disabled={currentIndex <= 0} className="flex h-10 items-center gap-1 rounded-md px-3 text-sm text-slate-300 hover:bg-slate-800 disabled:opacity-30"><ChevronLeft className="h-4 w-4" /> Previous</button>
            <span className="font-mono text-xs text-slate-500">{currentIndex >= 0 ? currentIndex + 1 : 0} / {filteredAssets.length}</span>
            <button type="button" onClick={() => selectRelative(1)} disabled={currentIndex < 0 || currentIndex >= filteredAssets.length - 1} className="flex h-10 items-center gap-1 rounded-md px-3 text-sm text-slate-300 hover:bg-slate-800 disabled:opacity-30">Next <ChevronRight className="h-4 w-4" /></button>
          </div>
        </section>

        <aside className="border-t border-slate-700/80 bg-[#0e151d] xl:border-l xl:border-t-0" aria-label="Bend label controls">
          <div className="border-b border-slate-700/80 p-4">
            <label htmlFor="annotator" className="mb-1.5 block text-xs font-medium text-slate-300">Annotator</label>
            <input
              id="annotator"
              autoComplete="name"
              value={workingLabels?.annotator ?? ''}
              onChange={(event) => updateLabels((labels) => ({ ...labels, annotator: event.target.value }))}
              placeholder="Full name"
              disabled={!workingLabels}
              className="h-10 w-full rounded-md border border-slate-600 bg-[#111b25] px-3 text-sm text-slate-100 outline-none placeholder:text-slate-500 focus:border-amber-400 focus:ring-2 focus:ring-amber-400/20 disabled:opacity-50"
            />
          </div>

          <div className="border-b border-slate-700/80 p-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-100">Bend lines</h2>
              <span className="font-mono text-xs text-slate-400">{workingLabels?.bends.length ?? 0}</span>
            </div>
            <p className="mt-1 text-xs leading-5 text-slate-400">Place each line along the center of the physical fold. Inspect from at least two views.</p>
            <button type="button" onClick={() => { setDraftPoint(null); setEndpointEdit(null) }} className="mt-3 flex h-10 w-full items-center justify-center gap-2 rounded-md border border-amber-400/50 bg-amber-400/10 text-sm font-medium text-amber-100 hover:bg-amber-400/15 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400">
              <Crosshair className="h-4 w-4" /> {draftPoint ? 'Restart current line' : 'Add bend line'}
            </button>
          </div>

          <div className="max-h-[420px] overflow-y-auto xl:max-h-[calc(100dvh-31rem)]">
            {workingLabels?.bends.map((bend) => (
              <div key={bend.bend_id} className={clsx('border-b border-slate-800 p-3', selectedBendId === bend.bend_id && 'bg-amber-400/5')}>
                <button type="button" onClick={() => setSelectedBendId(bend.bend_id)} className="flex h-9 w-full items-center justify-between rounded px-1 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400">
                  <span className="flex items-center gap-2 text-sm font-medium text-slate-100"><CircleDot className="h-4 w-4 text-amber-300" /> {bend.bend_id}</span>
                  <span className="text-[11px] text-slate-500">{bend.confidence}</span>
                </button>
                <div className="mt-1 space-y-1 font-mono text-[10px] text-slate-500">
                  <p className="truncate">S [{formatPoint(bend.line_start)}]</p>
                  <p className="truncate">E [{formatPoint(bend.line_end)}]</p>
                </div>
                <div className="mt-2 grid grid-cols-[1fr_1fr_40px] gap-2">
                  <button type="button" onClick={() => { setEndpointEdit({ bendId: bend.bend_id, endpoint: 'start' }); setDraftPoint(null) }} className="h-9 rounded-md border border-slate-600 text-xs text-slate-300 hover:border-amber-400 hover:text-amber-200">Move start</button>
                  <button type="button" onClick={() => { setEndpointEdit({ bendId: bend.bend_id, endpoint: 'end' }); setDraftPoint(null) }} className="h-9 rounded-md border border-slate-600 text-xs text-slate-300 hover:border-amber-400 hover:text-amber-200">Move end</button>
                  <button type="button" onClick={() => { updateLabels((labels) => removeBend(labels, bend.bend_id)); setSelectedBendId(null) }} aria-label={`Delete ${bend.bend_id}`} className="flex h-9 items-center justify-center rounded-md border border-slate-600 text-slate-400 hover:border-red-400 hover:text-red-300"><Trash2 className="h-4 w-4" /></button>
                </div>
              </div>
            ))}
            {workingLabels && !workingLabels.bends.length && (
              <div className="p-5 text-center text-xs leading-5 text-slate-500">No bend lines yet. Rotate the part, then click the start and end of each real bend.</div>
            )}
          </div>

          <div className="space-y-3 border-t border-slate-700/80 p-4">
            <label className="flex min-h-11 cursor-pointer items-center gap-3 rounded-md border border-slate-600 px-3 py-2 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={workingLabels?.no_bends_present ?? false}
                disabled={!workingLabels || Boolean(workingLabels.bends.length)}
                onChange={(event) => updateLabels((labels) => ({ ...labels, no_bends_present: event.target.checked, review_status: 'in_progress' }))}
                className="h-4 w-4 accent-amber-400"
              />
              No physical bends are present
            </label>
            <label className="block">
              <span className="mb-1.5 block text-xs font-medium text-slate-300">Part notes</span>
              <textarea value={workingLabels?.notes ?? ''} onChange={(event) => updateLabels((labels) => ({ ...labels, notes: event.target.value }))} disabled={!workingLabels} rows={2} className="w-full resize-none rounded-md border border-slate-600 bg-[#111b25] px-3 py-2 text-sm text-slate-100 outline-none placeholder:text-slate-500 focus:border-amber-400 focus:ring-2 focus:ring-amber-400/20" placeholder="Ambiguities, occlusion, or review notes" />
            </label>

            {actionError && <div role="alert" className="flex gap-2 rounded-md border border-red-400/40 bg-red-400/10 p-3 text-xs leading-5 text-red-100"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> {actionError}</div>}
            {savedFlash && !dirty && <div aria-live="polite" className="flex items-center gap-2 text-xs text-emerald-300"><Check className="h-4 w-4" /> Revision {workingLabels?.revision} saved</div>}

            <div className="grid grid-cols-2 gap-2">
              <button type="button" onClick={() => save(false)} disabled={!workingLabels || saveMutation.isPending} className="flex h-11 items-center justify-center gap-2 rounded-md border border-slate-500 bg-slate-700 text-sm font-medium text-white hover:bg-slate-600 disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400">
                <Save className="h-4 w-4" /> Save draft
              </button>
              <button type="button" onClick={() => save(true)} disabled={!workingLabels || saveMutation.isPending} className="flex h-11 items-center justify-center gap-2 rounded-md bg-amber-400 text-sm font-semibold text-[#17120a] hover:bg-amber-300 disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-200">
                <Check className="h-4 w-4" /> Complete
              </button>
            </div>
            <p className="text-center font-mono text-[10px] text-slate-500">Cmd/Ctrl+S saves. Esc cancels. Arrow keys change part.</p>
          </div>
        </aside>
      </main>
    </div>
  )
}
