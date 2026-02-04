import { useState, useEffect, useCallback, lazy, Suspense } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Radio,
  CheckCircle,
  XCircle,
  AlertTriangle,
  RotateCcw,
  Target,
  Loader2,
  Folder,
  FolderOpen,
  Settings,
  Play,
  Square,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'
import clsx from 'clsx'
import { liveScanApi } from '../services/api'
import type {
  LiveScanSession,
  RecognitionCandidate,
  ScanData,
  GapCluster,
} from '../types'

// Lazy load the 3D viewer for better initial page load
const CoverageViewer = lazy(() => import('../components/CoverageViewer'))

// Session state colors
const stateColors: Record<string, string> = {
  idle: 'bg-gray-500',
  identifying: 'bg-yellow-500',
  scanning: 'bg-blue-500',
  analyzing: 'bg-purple-500',
  complete: 'bg-green-500',
  abandoned: 'bg-red-500',
  error: 'bg-red-500',
}

// Session state labels
const stateLabels: Record<string, string> = {
  idle: 'Waiting for Scan',
  identifying: 'Identifying Part...',
  scanning: 'Scanning',
  analyzing: 'Analyzing...',
  complete: 'Complete',
  abandoned: 'Abandoned',
  error: 'Error',
}

// Coverage bar component
function CoverageBar({ percent }: { percent: number }) {
  const getColor = () => {
    if (percent >= 95) return 'bg-green-500'
    if (percent >= 80) return 'bg-blue-500'
    if (percent >= 50) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="text-dark-300">Coverage</span>
        <span className="font-medium">{percent.toFixed(1)}%</span>
      </div>
      <div className="h-3 bg-dark-700 rounded-full overflow-hidden">
        <div
          className={clsx('h-full transition-all duration-300', getColor())}
          style={{ width: `${Math.min(100, percent)}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-dark-400">
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
    </div>
  )
}

// Recognition candidate card
function CandidateCard({
  candidate,
  isSelected,
  onSelect,
}: {
  candidate: RecognitionCandidate
  isSelected: boolean
  onSelect: () => void
}) {
  return (
    <button
      onClick={onSelect}
      className={clsx(
        'w-full text-left p-3 rounded-lg border transition-all',
        isSelected
          ? 'border-primary-500 bg-primary-500/10'
          : 'border-dark-600 hover:border-dark-500 bg-dark-800'
      )}
    >
      <div className="flex items-center justify-between mb-1">
        <span className="font-medium">{candidate.part_number}</span>
        <span
          className={clsx(
            'text-sm px-2 py-0.5 rounded',
            candidate.similarity >= 0.95
              ? 'bg-green-500/20 text-green-400'
              : candidate.similarity >= 0.85
              ? 'bg-yellow-500/20 text-yellow-400'
              : 'bg-red-500/20 text-red-400'
          )}
        >
          {(candidate.similarity * 100).toFixed(1)}%
        </span>
      </div>
      {candidate.part_name && (
        <p className="text-sm text-dark-400 truncate">{candidate.part_name}</p>
      )}
      {candidate.warning && (
        <p className="text-xs text-yellow-500 mt-1">{candidate.warning}</p>
      )}
    </button>
  )
}

// Gap guidance component
function GapGuidance({ gaps }: { gaps: GapCluster[] }) {
  if (!gaps || gaps.length === 0) return null

  return (
    <div className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
      <div className="flex items-center gap-2 text-yellow-500 mb-2">
        <Target className="w-4 h-4" />
        <span className="font-medium">Scan These Areas</span>
      </div>
      <ul className="space-y-1 text-sm text-dark-300">
        {gaps.slice(0, 3).map((gap, i) => (
          <li key={i} className="flex items-center gap-2">
            <span className="w-2 h-2 bg-yellow-500 rounded-full" />
            {gap.location_hint} ({gap.diameter_mm.toFixed(0)}mm gap)
          </li>
        ))}
      </ul>
    </div>
  )
}

export default function LiveScan() {
  const [selectedCandidate, setSelectedCandidate] = useState<string | null>(null)
  const [session, setSession] = useState<LiveScanSession | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [watchPathInput, setWatchPathInput] = useState('')

  const queryClient = useQueryClient()

  // Fetch live scan config (watch folder)
  const { data: config } = useQuery({
    queryKey: ['live-scan-config'],
    queryFn: liveScanApi.getConfig,
    staleTime: 30000,
  })

  // Fetch live scan status
  const { data: status, refetch: refetchStatus } = useQuery({
    queryKey: ['live-scan-status'],
    queryFn: liveScanApi.getStatus,
    refetchInterval: 5000,
  })

  // Update watch path when config loads
  useEffect(() => {
    if (config?.watch_path && !watchPathInput) {
      setWatchPathInput(config.watch_path)
    }
  }, [config?.watch_path])

  // Mutation to update watch path
  const updateWatchPath = useMutation({
    mutationFn: (path: string) => liveScanApi.setWatchPath(path),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['live-scan-config'] })
      queryClient.invalidateQueries({ queryKey: ['live-scan-status'] })
    },
  })

  // Mutation to start watcher
  const startWatcher = useMutation({
    mutationFn: () => liveScanApi.startManager(watchPathInput),
    onSuccess: () => {
      refetchStatus()
    },
  })

  // Mutation to stop watcher
  const stopWatcher = useMutation({
    mutationFn: () => liveScanApi.stopManager(),
    onSuccess: () => {
      refetchStatus()
    },
  })

  const handleSaveWatchPath = () => {
    if (watchPathInput && watchPathInput !== config?.watch_path) {
      updateWatchPath.mutate(watchPathInput)
    }
  }

  // SSE connection for real-time updates
  useEffect(() => {
    let eventSource: EventSource | null = null
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null

    const connect = () => {
      eventSource = new EventSource('/api/live-scan/session/stream')

      eventSource.onopen = () => {
        setIsConnected(true)
        console.log('SSE connected')
      }

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          setSession(data as LiveScanSession | null)
        } catch (err) {
          console.error('Failed to parse SSE data:', err)
        }
      }

      eventSource.onerror = () => {
        setIsConnected(false)
        eventSource?.close()

        // Reconnect after 2 seconds
        reconnectTimeout = setTimeout(connect, 2000)
      }
    }

    connect()

    return () => {
      eventSource?.close()
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout)
      }
    }
  }, [])

  // Fetch recognition status
  const { data: recognitionStatus } = useQuery({
    queryKey: ['recognition-status'],
    queryFn: liveScanApi.getRecognitionStatus,
    refetchInterval: 30000,
  })

  // Auto-select top candidate when recognition completes
  useEffect(() => {
    if (session?.recognition?.candidates?.length && !selectedCandidate) {
      const topCandidate = session.recognition.candidates[0]
      if (topCandidate.similarity >= 0.85) {
        setSelectedCandidate(topCandidate.part_id)
      }
    }
  }, [session?.recognition?.candidates, selectedCandidate])

  // Confirm part selection
  const handleConfirmPart = useCallback(async () => {
    if (!selectedCandidate || !session) return

    const candidate = session.recognition?.candidates?.find(
      (c: RecognitionCandidate) => c.part_id === selectedCandidate
    )
    if (!candidate) return

    try {
      await liveScanApi.confirmPart(session.id, candidate.part_id, candidate.part_number)
      // SSE will push the updated session automatically
    } catch (error) {
      console.error('Failed to confirm part:', error)
    }
  }, [selectedCandidate, session])

  // Complete scan
  const handleCompleteScan = useCallback(async () => {
    if (!session) return

    try {
      await liveScanApi.completeScan(session.id)
      // SSE will push the updated session automatically
    } catch (error) {
      console.error('Failed to complete scan:', error)
    }
  }, [session])

  // Cancel session
  const handleCancel = useCallback(async () => {
    if (!session) return

    try {
      await liveScanApi.cancelSession(session.id)
      setSelectedCandidate(null)
    } catch (error) {
      console.error('Failed to cancel session:', error)
    }
  }, [session])

  // Reset session to start fresh
  const handleReset = useCallback(async () => {
    try {
      await liveScanApi.resetSession()
      setSelectedCandidate(null)
    } catch (error) {
      console.error('Failed to reset session:', error)
    }
  }, [])

  const state = session?.state || 'idle'
  const coverage = session?.coverage_percent || 0
  const candidates: RecognitionCandidate[] = session?.recognition?.candidates || []
  const gaps: GapCluster[] = session?.gap_clusters || []

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold">Live Scan</h1>
          <p className="text-dark-400">Real-time part scanning and recognition</p>
        </div>

        <div className="flex items-center gap-4">
          {/* Watcher status */}
          <div className="flex items-center gap-2 px-3 py-1.5 bg-dark-800 rounded-full">
            <div
              className={clsx(
                'w-2 h-2 rounded-full',
                status?.running ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'
              )}
            />
            <span className="text-sm text-dark-300">
              {status?.running ? 'Watching' : 'Stopped'}
            </span>
          </div>

          {/* Connection status */}
          <div className="flex items-center gap-2 px-3 py-1.5 bg-dark-800 rounded-full">
            <div
              className={clsx(
                'w-2 h-2 rounded-full',
                isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
              )}
            />
            <span className="text-sm text-dark-300">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          {/* Recognition index status */}
          {recognitionStatus && (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-dark-800 rounded-lg">
              <Folder className="w-4 h-4 text-primary-500" />
              <span className="text-sm">
                {recognitionStatus.index_count} parts indexed
              </span>
            </div>
          )}

          {/* Settings toggle */}
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={clsx(
              'flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors',
              showSettings ? 'bg-primary-500 text-white' : 'bg-dark-800 hover:bg-dark-700'
            )}
          >
            <Settings className="w-4 h-4" />
            <span className="text-sm">Settings</span>
            {showSettings ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="card p-4 mb-4">
          <h3 className="font-medium mb-3 flex items-center gap-2">
            <FolderOpen className="w-4 h-4 text-primary-500" />
            Watch Folder Configuration
          </h3>

          <div className="flex items-end gap-4">
            <div className="flex-1">
              <label className="block text-sm text-dark-400 mb-1">
                VXelements Export Folder
              </label>
              <input
                type="text"
                value={watchPathInput}
                onChange={(e) => setWatchPathInput(e.target.value)}
                placeholder="/path/to/vxelements/exports"
                className="w-full px-3 py-2 bg-dark-800 border border-dark-600 rounded-lg focus:border-primary-500 focus:outline-none"
              />
              <p className="text-xs text-dark-500 mt-1">
                Set this to the folder where VXelements auto-exports scan files (.ply, .stl)
              </p>
            </div>

            <button
              onClick={handleSaveWatchPath}
              disabled={!watchPathInput || watchPathInput === config?.watch_path || updateWatchPath.isPending}
              className="btn btn-secondary flex items-center gap-2"
            >
              {updateWatchPath.isPending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <CheckCircle className="w-4 h-4" />
              )}
              Save Path
            </button>

            {status?.running ? (
              <button
                onClick={() => stopWatcher.mutate()}
                disabled={stopWatcher.isPending}
                className="btn bg-red-500 hover:bg-red-600 text-white flex items-center gap-2"
              >
                {stopWatcher.isPending ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Square className="w-4 h-4" />
                )}
                Stop Watching
              </button>
            ) : (
              <button
                onClick={() => startWatcher.mutate()}
                disabled={startWatcher.isPending || !watchPathInput}
                className="btn btn-primary flex items-center gap-2"
              >
                {startWatcher.isPending ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
                Start Watching
              </button>
            )}
          </div>

          {updateWatchPath.isSuccess && (
            <div className="mt-3 p-2 bg-green-500/10 border border-green-500/30 rounded text-sm text-green-400">
              Watch path updated successfully. Click "Start Watching" to begin.
            </div>
          )}

          {(updateWatchPath.isError || startWatcher.isError) && (
            <div className="mt-3 p-2 bg-red-500/10 border border-red-500/30 rounded text-sm text-red-400">
              Error: {(updateWatchPath.error as Error)?.message || (startWatcher.error as Error)?.message}
            </div>
          )}
        </div>
      )}

      <div className="flex-1 grid grid-cols-12 gap-6 min-h-0">
        {/* Left Panel - Status and Controls */}
        <div className="col-span-3 flex flex-col gap-4 overflow-y-auto">
          {/* Session Status Card */}
          <div className="card p-4">
            <div className="flex items-center gap-3 mb-4">
              <div className={clsx('w-3 h-3 rounded-full', stateColors[state])} />
              <span className="font-medium">{stateLabels[state]}</span>
              {(state === 'identifying' || state === 'analyzing') && (
                <Loader2 className="w-4 h-4 animate-spin text-primary-500 ml-auto" />
              )}
            </div>

            {session?.part_number && (
              <div className="p-3 bg-dark-800 rounded-lg mb-4">
                <div className="text-sm text-dark-400 mb-1">Current Part</div>
                <div className="font-medium text-lg">{session.part_number}</div>
              </div>
            )}

            {/* Scan count */}
            <div className="flex items-center justify-between text-sm">
              <span className="text-dark-400">Scans Received</span>
              <span className="font-medium">{session?.scans?.length || 0}</span>
            </div>

            {/* Total points */}
            <div className="flex items-center justify-between text-sm mt-2">
              <span className="text-dark-400">Total Points</span>
              <span className="font-medium">
                {session?.total_points?.toLocaleString() || 0}
              </span>
            </div>
          </div>

          {/* Coverage Card */}
          <div className="card p-4">
            <CoverageBar percent={coverage} />

            {gaps.length > 0 && (
              <div className="mt-4">
                <GapGuidance gaps={gaps} />
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div className="card p-4 space-y-3">
            {state === 'scanning' && (
              <>
                <button
                  onClick={handleCompleteScan}
                  disabled={coverage < 50}
                  className="btn btn-primary w-full flex items-center justify-center gap-2"
                >
                  <CheckCircle className="w-5 h-5" />
                  Complete Scan
                </button>
                <button
                  onClick={handleCancel}
                  className="btn btn-secondary w-full flex items-center justify-center gap-2"
                >
                  <XCircle className="w-5 h-5" />
                  Cancel
                </button>
              </>
            )}

            {state === 'idle' && (
              <div className="text-center py-4">
                <Radio className="w-12 h-12 text-dark-600 mx-auto mb-2" />
                <p className="text-dark-400 text-sm">
                  Waiting for scan files from VXelements...
                </p>
              </div>
            )}

            {state === 'complete' && (
              <div className="text-center py-4">
                <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-2" />
                <p className="text-dark-400 text-sm">
                  Scan complete! Analysis results available.
                </p>
                <button
                  onClick={handleReset}
                  className="btn btn-secondary mt-3 flex items-center justify-center gap-2 mx-auto"
                >
                  <RotateCcw className="w-4 h-4" />
                  Start New Scan
                </button>
              </div>
            )}

            {state === 'error' && session?.error_message && (
              <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                <div className="flex items-center gap-2 text-red-500 mb-1">
                  <AlertTriangle className="w-4 h-4" />
                  <span className="font-medium">Error</span>
                </div>
                <p className="text-sm text-red-400">{session.error_message}</p>
              </div>
            )}
          </div>
        </div>

        {/* Center - 3D Viewer */}
        <div className="col-span-6 flex flex-col min-h-0">
          <div className="flex-1 card overflow-hidden">
            <Suspense
              fallback={
                <div className="w-full h-full flex items-center justify-center bg-dark-900">
                  <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
                </div>
              }
            >
              <CoverageViewer
                coverage={coverage}
                gaps={gaps}
                totalPoints={session?.total_points || 0}
                state={state}
              />
            </Suspense>
          </div>
        </div>

        {/* Right Panel - Recognition Results */}
        <div className="col-span-3 flex flex-col gap-4 overflow-y-auto">
          <div className="card p-4">
            <h3 className="font-medium mb-3 flex items-center gap-2">
              <Target className="w-4 h-4 text-primary-500" />
              Part Recognition
            </h3>

            {state === 'idle' && (
              <div className="text-center py-6 text-dark-400 text-sm">
                Start scanning to identify the part
              </div>
            )}

            {state === 'identifying' && (
              <div className="text-center py-6">
                <Loader2 className="w-8 h-8 animate-spin text-primary-500 mx-auto mb-2" />
                <p className="text-dark-400 text-sm">Analyzing scan...</p>
              </div>
            )}

            {candidates.length > 0 && (
              <div className="space-y-2">
                {candidates.map((candidate) => (
                  <CandidateCard
                    key={candidate.part_id}
                    candidate={candidate}
                    isSelected={selectedCandidate === candidate.part_id}
                    onSelect={() => setSelectedCandidate(candidate.part_id)}
                  />
                ))}

                {state === 'identifying' && (
                  <button
                    onClick={handleConfirmPart}
                    disabled={!selectedCandidate}
                    className="btn btn-primary w-full mt-4"
                  >
                    Confirm Selection
                  </button>
                )}
              </div>
            )}

            {session?.recognition?.is_confident && (
              <div className="mt-3 p-2 bg-green-500/10 border border-green-500/30 rounded text-center">
                <span className="text-sm text-green-400">
                  Auto-confirmed with high confidence
                </span>
              </div>
            )}
          </div>

          {/* Recent Scans */}
          {session?.scans && session.scans.length > 0 && (
            <div className="card p-4">
              <h3 className="font-medium mb-3">Recent Scans</h3>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {session.scans.slice(-5).reverse().map((scan: ScanData, i: number) => (
                  <div
                    key={i}
                    className="flex items-center justify-between text-sm p-2 bg-dark-800 rounded"
                  >
                    <span className="truncate">{scan.filename}</span>
                    <span className="text-dark-400">
                      {scan.points_count?.toLocaleString() || '?'} pts
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
