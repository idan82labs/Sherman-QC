import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Upload,
  RefreshCw,
  CheckCircle,
  XCircle,
  Clock,
  Trash2,
  Play,
  ChevronDown,
  ChevronRight,
  Activity,
  Download,
  Image as ImageIcon,
} from 'lucide-react'
import clsx from 'clsx'
import {
  bendInspectionApi,
  getErrorMessage,
  type BendInspectionJob,
  type BendMatch,
  type ScanQualitySummary,
} from '../services/api'
import type { BendAnnotation } from '../components/ThreeViewer'

const ThreeViewer = lazy(() => import('../components/ThreeViewer'))

export default function BendInspection() {
  const [cadFile, setCadFile] = useState<File | null>(null)
  const [scanFile, setScanFile] = useState<File | null>(null)
  const [partId, setPartId] = useState('PART_001')
  const [partName, setPartName] = useState('')
  const [toleranceAngle, setToleranceAngle] = useState(1.0)
  const [toleranceRadius, setToleranceRadius] = useState(0.5)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [expandedJobs, setExpandedJobs] = useState<Set<string>>(new Set())

  const queryClient = useQueryClient()

  // List bend inspection jobs
  const { data: jobsData, isLoading: jobsLoading, refetch: refetchJobs } = useQuery({
    queryKey: ['bend-inspection-jobs'],
    queryFn: () => bendInspectionApi.list({ limit: 20 }),
    refetchInterval: 5000, // Refresh every 5 seconds
  })

  // Get selected job details
  const { data: selectedJob } = useQuery({
    queryKey: ['bend-inspection-job', selectedJobId],
    queryFn: () => bendInspectionApi.get(selectedJobId!),
    enabled: !!selectedJobId,
    refetchInterval: (query) => {
      const data = query.state.data
      return data?.status === 'running' ? 2000 : false
    },
  })

  // Start analysis mutation
  const analyzeMutation = useMutation({
    mutationFn: (formData: FormData) => bendInspectionApi.analyze(formData, setUploadProgress),
    onSuccess: (data) => {
      setSelectedJobId(data.job_id)
      setCadFile(null)
      setScanFile(null)
      setUploadProgress(0)
      queryClient.invalidateQueries({ queryKey: ['bend-inspection-jobs'] })
    },
  })

  // Delete job mutation
  const deleteMutation = useMutation({
    mutationFn: bendInspectionApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['bend-inspection-jobs'] })
      if (selectedJobId && selectedJobId === deleteMutation.variables) {
        setSelectedJobId(null)
      }
    },
  })

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault()

    if (!cadFile || !scanFile) {
      alert('Please select both CAD and scan files')
      return
    }

    const formData = new FormData()
    formData.append('cad_file', cadFile)
    formData.append('scan_file', scanFile)
    formData.append('part_id', partId)
    formData.append('part_name', partName || partId)
    formData.append('default_tolerance_angle', toleranceAngle.toString())
    formData.append('default_tolerance_radius', toleranceRadius.toString())

    analyzeMutation.mutate(formData)
  }, [cadFile, scanFile, partId, partName, toleranceAngle, toleranceRadius, analyzeMutation])

  const toggleJobExpanded = (jobId: string) => {
    setExpandedJobs((prev) => {
      const next = new Set(prev)
      if (next.has(jobId)) {
        next.delete(jobId)
      } else {
        next.add(jobId)
      }
      return next
    })
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-100">Progressive Bend Inspection</h1>
          <p className="text-dark-400 mt-1">
            Inspect partial or complete bends against CAD specifications
          </p>
        </div>
        <button
          onClick={() => refetchJobs()}
          className="btn btn-secondary flex items-center"
        >
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Form */}
        <div className="lg:col-span-1">
          <div className="card p-6">
            <h2 className="font-semibold text-dark-100 mb-4 flex items-center">
              <Upload className="w-5 h-5 mr-2 text-primary-400" />
              New Inspection
            </h2>

            <form onSubmit={handleSubmit} className="space-y-4">
              {/* CAD File */}
              <div>
                <label className="block text-sm font-medium text-dark-300 mb-1">
                  CAD Reference File *
                </label>
                <div className="relative">
                  <input
                    type="file"
                    accept=".stl,.ply,.obj,.step,.stp"
                    onChange={(e) => setCadFile(e.target.files?.[0] || null)}
                    className="w-full text-sm text-dark-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-medium file:bg-primary-500/20 file:text-primary-400 hover:file:bg-primary-500/30 cursor-pointer"
                  />
                </div>
                {cadFile && (
                  <p className="text-xs text-dark-500 mt-1">{cadFile.name}</p>
                )}
              </div>

              {/* Scan File */}
              <div>
                <label className="block text-sm font-medium text-dark-300 mb-1">
                  Scan Point Cloud *
                </label>
                <div className="relative">
                  <input
                    type="file"
                    accept=".stl,.ply,.obj,.pcd"
                    onChange={(e) => setScanFile(e.target.files?.[0] || null)}
                    className="w-full text-sm text-dark-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-medium file:bg-secondary-500/20 file:text-secondary-400 hover:file:bg-secondary-500/30 cursor-pointer"
                  />
                </div>
                {scanFile && (
                  <p className="text-xs text-dark-500 mt-1">{scanFile.name}</p>
                )}
              </div>

              {/* Part Info */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-dark-300 mb-1">
                    Part ID
                  </label>
                  <input
                    type="text"
                    value={partId}
                    onChange={(e) => setPartId(e.target.value)}
                    className="input w-full"
                    placeholder="PART_001"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-dark-300 mb-1">
                    Part Name
                  </label>
                  <input
                    type="text"
                    value={partName}
                    onChange={(e) => setPartName(e.target.value)}
                    className="input w-full"
                    placeholder="Optional"
                  />
                </div>
              </div>

              {/* Tolerances */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-dark-300 mb-1">
                    Angle Tol. (°)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="10"
                    value={toleranceAngle}
                    onChange={(e) => setToleranceAngle(parseFloat(e.target.value))}
                    className="input w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-dark-300 mb-1">
                    Radius Tol. (mm)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="5"
                    value={toleranceRadius}
                    onChange={(e) => setToleranceRadius(parseFloat(e.target.value))}
                    className="input w-full"
                  />
                </div>
              </div>

              {/* Progress */}
              {analyzeMutation.isPending && uploadProgress > 0 && (
                <div className="mt-4">
                  <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-primary-500 to-secondary-500 transition-all"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                  <p className="text-xs text-dark-400 mt-1">Uploading... {uploadProgress}%</p>
                </div>
              )}

              {/* Error */}
              {analyzeMutation.isError && (
                <div className="p-3 bg-red-500/10 border border-red-500/20 rounded text-sm text-red-400">
                  {getErrorMessage(analyzeMutation.error)}
                </div>
              )}

              {/* Submit */}
              <button
                type="submit"
                disabled={!cadFile || !scanFile || analyzeMutation.isPending}
                className="btn btn-primary w-full flex items-center justify-center"
              >
                {analyzeMutation.isPending ? (
                  <>
                    <Clock className="w-4 h-4 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    Start Inspection
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Info Box */}
          <div className="card p-4 mt-4 bg-amber-500/5 border-amber-500/20">
            <h3 className="text-sm font-medium text-amber-400 mb-2">Progressive Inspection</h3>
            <p className="text-xs text-dark-400">
              This tool compares a partial scan (after some bends) against the complete CAD model.
              It identifies which bends have been made and verifies they are within tolerance.
            </p>
            <ul className="text-xs text-dark-400 mt-2 space-y-1">
              <li>• Works at any stage of bending (2/11 bends, 5/11 bends, etc.)</li>
              <li>• Detects bend angles and matches to CAD specifications</li>
              <li>• No need for global alignment - bends are matched by angle</li>
            </ul>
          </div>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-4">
          {/* Selected Job Details */}
          {selectedJob && (
            <div className="card p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <StatusIcon status={selectedJob.status} />
                  <div className="ml-3">
                    <h3 className="font-semibold text-dark-100">{selectedJob.part_id}</h3>
                    <p className="text-sm text-dark-400">{selectedJob.part_name}</p>
                    {(selectedJob.seed_case?.scan_file || selectedJob.seed_case?.state) && (
                      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                        {selectedJob.seed_case?.state && (
                          <span className="rounded-full border border-primary-500/30 bg-primary-500/10 px-2 py-0.5 text-primary-300">
                            {selectedJob.seed_case.state === 'partial' ? 'Partial scan' : 'Full scan'}
                          </span>
                        )}
                        {selectedJob.seed_case?.scan_file && (
                          <span className="text-dark-500">
                            Source scan: {selectedJob.seed_case.scan_file}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => setSelectedJobId(null)}
                  className="text-dark-400 hover:text-dark-200"
                >
                  <XCircle className="w-5 h-5" />
                </button>
              </div>

              {/* Progress bar for running jobs */}
              {selectedJob.status === 'running' && (
                <div className="mb-4">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-dark-400">{selectedJob.stage || 'Processing...'}</span>
                    <span className="text-dark-300">{selectedJob.progress}%</span>
                  </div>
                  <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-primary-500 to-secondary-500 transition-all"
                      style={{ width: `${selectedJob.progress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Error */}
              {selectedJob.status === 'failed' && selectedJob.error && (
                <div className="p-3 bg-red-500/10 border border-red-500/20 rounded text-sm text-red-400 mb-4">
                  {selectedJob.error}
                </div>
              )}

              {/* Results */}
              {selectedJob.status === 'completed' && selectedJob.report && (
                <BendInspectionResults
                  report={selectedJob.report}
                  jobId={selectedJob.job_id}
                  referenceMeshUrl={selectedJob.artifacts?.reference_mesh_url}
                  artifacts={selectedJob.artifacts}
                />
              )}
            </div>
          )}

          {/* Job List */}
          <div className="card p-6">
            <h2 className="font-semibold text-dark-100 mb-4 flex items-center">
              <Activity className="w-5 h-5 mr-2 text-primary-400" />
              Recent Inspections
            </h2>

            {jobsLoading ? (
              <div className="animate-pulse space-y-3">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="h-16 bg-dark-700 rounded" />
                ))}
              </div>
            ) : jobsData?.jobs && jobsData.jobs.length > 0 ? (
              <div className="space-y-2">
                {jobsData.jobs.map((job) => (
                  <JobListItem
                    key={job.job_id}
                    job={job}
                    isSelected={selectedJobId === job.job_id}
                    isExpanded={expandedJobs.has(job.job_id)}
                    onSelect={() => setSelectedJobId(job.job_id)}
                    onToggleExpand={() => toggleJobExpanded(job.job_id)}
                    onDelete={() => {
                      if (confirm('Delete this inspection?')) {
                        deleteMutation.mutate(job.job_id)
                      }
                    }}
                  />
                ))}
              </div>
            ) : (
              <p className="text-dark-400 text-center py-8">
                No inspections yet. Start a new inspection above.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// Status icon component
function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'completed':
      return <CheckCircle className="w-6 h-6 text-emerald-400" />
    case 'failed':
      return <XCircle className="w-6 h-6 text-red-400" />
    case 'running':
      return <Clock className="w-6 h-6 text-amber-400 animate-pulse" />
    default:
      return <Clock className="w-6 h-6 text-dark-400" />
  }
}

// Job list item component
function JobListItem({
  job,
  isSelected,
  isExpanded,
  onSelect,
  onToggleExpand,
  onDelete,
}: {
  job: BendInspectionJob
  isSelected: boolean
  isExpanded: boolean
  onSelect: () => void
  onToggleExpand: () => void
  onDelete: () => void
}) {
  const summary = job.report?.summary
  const completedBends = summary ? (summary.completed_bends ?? summary.detected) : null
  const expectedBends = summary ? (summary.expected_bends ?? summary.total_bends) : null
  const structuredCompletedBends = summary?.structured_completed_bends ?? null

  return (
    <div
      className={clsx(
        'border rounded-lg transition-all',
        isSelected
          ? 'border-primary-500/50 bg-primary-500/5'
          : 'border-dark-700 hover:border-dark-600'
      )}
    >
      <div
        className="flex items-center p-3 cursor-pointer"
        onClick={onSelect}
      >
        <button
          onClick={(e) => {
            e.stopPropagation()
            onToggleExpand()
          }}
          className="mr-2 text-dark-400 hover:text-dark-200"
        >
          {isExpanded ? (
            <ChevronDown className="w-4 h-4" />
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
        </button>

        <StatusIcon status={job.status} />

        <div className="ml-3 flex-1 min-w-0">
          <p className="font-medium text-dark-100 truncate">{job.part_id}</p>
          <div className="mt-1 flex flex-wrap items-center gap-2">
            {job.seed_case?.state && (
              <span className="rounded-full border border-dark-600 bg-dark-800 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-primary-300">
                {job.seed_case.state}
              </span>
            )}
            {job.seed_case?.scan_file && (
              <p className="text-xs text-dark-400 truncate max-w-[240px]">
                {job.seed_case.scan_file}
              </p>
            )}
          </div>
          <p className="text-xs text-dark-500 mt-1">
            {new Date(job.created_at).toLocaleString()}
          </p>
        </div>

        {job.status === 'completed' && summary && completedBends !== null && expectedBends !== null && (
          <div className="text-right mr-4">
            <p className="text-sm font-medium text-dark-100">
              {(structuredCompletedBends ?? completedBends)}/{expectedBends}
            </p>
            <p className="text-xs text-dark-400">
              {structuredCompletedBends != null ? 'model/expected' : 'completed/expected'}
            </p>
            {structuredCompletedBends != null && structuredCompletedBends !== completedBends && (
              <p className="text-[11px] text-dark-500">evidence {completedBends}</p>
            )}
          </div>
        )}

        {job.status === 'running' && (
          <div className="text-right mr-4">
            <p className="text-sm font-medium text-amber-400">{job.progress}%</p>
            <p className="text-xs text-dark-400">processing</p>
          </div>
        )}

        <button
          onClick={(e) => {
            e.stopPropagation()
            onDelete()
          }}
          className="p-2 text-dark-400 hover:text-red-400 transition-colors"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {/* Expanded details */}
      {isExpanded && job.status === 'completed' && job.report && (
        <div className="px-4 pb-4 border-t border-dark-700 mt-2 pt-3">
          <BendInspectionResults
            report={job.report}
            compact
            jobId={job.job_id}
            referenceMeshUrl={job.artifacts?.reference_mesh_url}
            artifacts={job.artifacts}
          />
        </div>
      )}
    </div>
  )
}

// Bend inspection results component
function BendInspectionResults({
  report,
  jobId,
  referenceMeshUrl,
  artifacts,
  compact = false,
}: {
  report: BendInspectionJob['report']
  jobId?: string
  referenceMeshUrl?: string
  artifacts?: BendInspectionJob['artifacts']
  compact?: boolean
}) {
  if (!report) return null

  const { summary, matches } = report
  const completedBends = summary.completed_bends ?? summary.detected
  const structuredCompletedBends = summary.structured_completed_bends
  const displayedCompletedBends = structuredCompletedBends ?? completedBends
  const completedInSpec = summary.completed_in_spec ?? summary.passed
  const completedOutOfSpec = summary.completed_out_of_spec ?? (completedBends - completedInSpec)
  const remainingBends = summary.remaining_bends ?? Math.max(0, summary.total_bends - completedBends)
  const structuredRemainingBends = summary.structured_remaining_bends
  const scanQuality = report.scan_quality
  const preferredFocusedBendId = useMemo(() => {
    const issueMatches = matches.filter((match) => match.status === 'FAIL' || match.status === 'WARNING')
    if (issueMatches.length > 1) {
      // Multi-issue jobs should open in overview mode so all problematic bends
      // stay visible at once. Camera focus starts only after an explicit user selection.
      return null
    }
    return issueMatches[0]?.bend_id
      ?? matches.find((match) => match.status === 'PASS')?.bend_id
      ?? matches.find((match) => match.status !== 'NOT_DETECTED')?.bend_id
      ?? report.operator_actions?.[0]?.bend_id
      ?? matches[0]?.bend_id
      ?? null
  }, [matches, report.operator_actions])
  const [focusedBendId, setFocusedBendId] = useState<string | null>(preferredFocusedBendId)

  useEffect(() => {
    setFocusedBendId(preferredFocusedBendId)
  }, [preferredFocusedBendId])

  return (
    <div className={clsx('space-y-4', compact && 'space-y-2')}>
      {/* Operator brief */}
      {!!report.operator_actions?.length && (
        <div className="bg-dark-700/40 border border-dark-600 rounded-lg p-3">
          <p className="text-xs text-dark-400 uppercase tracking-wide">Action queue</p>
          <ul className="mt-2 space-y-1 text-xs text-dark-300">
            {report.operator_actions.slice(0, compact ? 2 : 4).map((a, idx) => (
              <li key={`${a.bend_id}-${idx}`}>
                <span className="text-dark-100 font-medium">{a.bend_id}</span>
                {' - '}
                {a.action}
              </li>
            ))}
          </ul>
        </div>
      )}

      {!compact && structuredCompletedBends != null && (
        <div className="rounded-lg border border-primary-500/30 bg-primary-500/8 p-3">
          <div className="flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-xs text-primary-300 uppercase tracking-wide">Structured Count Model</p>
              <p className="mt-1 text-sm text-dark-100">
                Posterior median estimates <span className="font-semibold">{structuredCompletedBends}</span> completed bends
                {summary.expected_bends != null ? ` out of ${summary.expected_bends}` : ''}.
              </p>
              <p className="text-xs text-dark-400 mt-1">
                Direct bend evidence currently supports {completedBends} explicit bend matches.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <FocusedMetricCard
                label="Model Count"
                value={`${structuredCompletedBends}`}
                tone="neutral"
              />
              <FocusedMetricCard
                label="Direct Evidence"
                value={`${completedBends}`}
                tone="neutral"
              />
              <FocusedMetricCard
                label="Delta"
                value={formatSignedMetric(summary.structured_count_delta_vs_match, 0)}
                tone="neutral"
              />
              <FocusedMetricCard
                label="Mean / MAP"
                value={`${summary.structured_count_mean?.toFixed?.(1) ?? '-'} / ${summary.structured_count_map ?? '-'}`}
                tone="neutral"
              />
            </div>
          </div>
        </div>
      )}

      {scanQuality && (
        <ScanQualityQueue quality={scanQuality} compact={compact} />
      )}

      {!compact && !!jobId && (
        <BendCadOverlay3D
          jobId={jobId}
          matches={matches}
          referenceMeshUrl={referenceMeshUrl}
          focusedBendId={focusedBendId}
        />
      )}

      {!compact && !!artifacts && (
        <BendArtifactsPanel artifacts={artifacts} />
      )}

      <BendProgressStrip
        matches={matches}
        compact={compact}
        focusedBendId={focusedBendId}
        onFocusBend={setFocusedBendId}
      />
      <BendGeometryCue
        matches={matches}
        compact={compact}
        focusedBendId={focusedBendId}
      />

      {/* Summary */}
      <div className={clsx(
        'grid gap-3',
        compact ? 'grid-cols-4' : 'grid-cols-2 md:grid-cols-7'
      )}>
        <SummaryCard
          label="Completed"
          value={`${displayedCompletedBends}/${summary.expected_bends ?? summary.total_bends}`}
          subValue={`${summary.progress_pct.toFixed(0)}%`}
          compact={compact}
        />
        <SummaryCard
          label="In Spec"
          value={completedInSpec.toString()}
          color="emerald"
          compact={compact}
        />
        <SummaryCard
          label="Out of Spec"
          value={completedOutOfSpec.toString()}
          color="red"
          compact={compact}
        />
        <SummaryCard
          label="Remaining"
          value={(structuredRemainingBends ?? remainingBends).toString()}
          color={(structuredRemainingBends ?? remainingBends) === 0 ? 'emerald' : undefined}
          compact={compact}
        />
        {!compact && structuredCompletedBends != null && (
          <SummaryCard
            label="Evidence Count"
            value={completedBends.toString()}
            subValue="explicit bends"
            compact={compact}
          />
        )}
        <SummaryCard
          label="Failed"
          value={summary.failed.toString()}
          color="red"
          compact={compact}
        />
        <SummaryCard
          label="Warnings"
          value={summary.warnings.toString()}
          color="amber"
          compact={compact}
        />
        {!compact && (
          <SummaryCard
            label="Rolled/Folded"
            value={`${summary.rolled_bends ?? 0}/${summary.folded_bends ?? 0}`}
            compact={compact}
          />
        )}
      </div>

      {/* Bend table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-dark-500 border-b border-dark-700">
              <th className={clsx('pb-2', compact ? 'pr-2' : 'pr-4')}>Bend</th>
              <th className={clsx('pb-2', compact ? 'pr-2' : 'pr-4')}>Form</th>
              <th className={clsx('pb-2', compact ? 'pr-2' : 'pr-4')}>Angle (deg)</th>
              <th className={clsx('pb-2', compact ? 'pr-2' : 'pr-4')}>Radius (mm)</th>
              <th className={clsx('pb-2', compact ? 'pr-2' : 'pr-4')}>Edge/Arc (mm)</th>
              <th className="pb-2">Status</th>
              {!compact && <th className="pb-2 pl-3">Action</th>}
            </tr>
          </thead>
          <tbody>
            {matches.map((match) => (
              <BendRow
                key={match.bend_id}
                match={match}
                compact={compact}
                focused={focusedBendId === match.bend_id}
                onFocus={() => setFocusedBendId(match.bend_id)}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function BendProgressStrip({
  matches,
  compact,
  focusedBendId,
  onFocusBend,
}: {
  matches: BendMatch[]
  compact?: boolean
  focusedBendId?: string | null
  onFocusBend?: (bendId: string) => void
}) {
  const statusClasses: Record<string, string> = {
    PASS: 'bg-emerald-500/25 border-emerald-500/50 text-emerald-200',
    FAIL: 'bg-red-500/25 border-red-500/50 text-red-200',
    WARNING: 'bg-amber-500/25 border-amber-500/50 text-amber-200',
    NOT_DETECTED: 'bg-dark-700 border-dark-500 text-dark-300',
  }

  return (
    <div className="rounded-lg border border-dark-600 bg-dark-800/40 p-3">
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs text-dark-400 uppercase tracking-wide">Bend Progress Map</p>
        {!compact && (
          <p className="text-xs text-dark-500">CAD bend IDs colored by inspection status</p>
        )}
      </div>
      <div className={clsx('grid gap-1', compact ? 'grid-cols-6' : 'grid-cols-8')}>
        {matches.map((match) => (
          <button
            key={match.bend_id}
            className={clsx(
              'rounded border px-1.5 py-1 text-[11px] text-center font-medium truncate transition-all',
              statusClasses[match.status] || statusClasses.NOT_DETECTED,
              focusedBendId === match.bend_id && 'ring-2 ring-primary-400/80 border-primary-300'
            )}
            title={`${match.bend_id} - ${match.status}`}
            onClick={() => onFocusBend?.(match.bend_id)}
            type="button"
          >
            {match.bend_id}
          </button>
        ))}
      </div>
    </div>
  )
}

type AxisView = 'XY' | 'XZ' | 'YZ'

const VIEW_AXES: Record<AxisView, [number, number]> = {
  XY: [0, 1],
  XZ: [0, 2],
  YZ: [1, 2],
}

const STATUS_STROKE: Record<string, string> = {
  PASS: '#34d399',
  FAIL: '#f87171',
  WARNING: '#fbbf24',
  NOT_DETECTED: '#64748b',
}

function parseLinePoint(raw?: [number, number, number] | number[] | null): [number, number, number] | null {
  if (!Array.isArray(raw) || raw.length < 3) return null
  const x = Number(raw[0])
  const y = Number(raw[1])
  const z = Number(raw[2])
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null
  return [x, y, z]
}

function midpoint3D(
  a: [number, number, number],
  b: [number, number, number],
): [number, number, number] {
  return [
    (a[0] + b[0]) / 2,
    (a[1] + b[1]) / 2,
    (a[2] + b[2]) / 2,
  ]
}

function toViewerStatus(status: BendMatch['status']): BendAnnotation['status'] {
  switch (status) {
    case 'PASS':
      return 'pass'
    case 'FAIL':
      return 'fail'
    case 'WARNING':
      return 'warning'
    default:
      return 'pending'
  }
}

function formatSignedMetric(value: number | null | undefined, digits: number, suffix = '') {
  if (value == null || !Number.isFinite(value)) return '-'
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}${suffix}`
}

function BendCadOverlay3D({
  jobId,
  matches,
  referenceMeshUrl,
  focusedBendId,
}: {
  jobId: string
  matches: BendMatch[]
  referenceMeshUrl?: string
  focusedBendId?: string | null
}) {
  const focusPoint = useMemo<[number, number, number] | null>(() => {
    const focused = matches.find((match) => match.bend_id === focusedBendId)
    if (!focused) return null
    return parseLinePoint(focused.callout_anchor)
      ?? midpointFromMatch(focused)
      ?? null
  }, [focusedBendId, matches])
  const focusedMatch = useMemo(
    () => matches.find((match) => match.bend_id === focusedBendId) ?? null,
    [focusedBendId, matches],
  )
  const issueMatches = useMemo(
    () => matches.filter((match) => match.status === 'FAIL' || match.status === 'WARNING'),
    [matches],
  )

  const bendAnnotations = useMemo<BendAnnotation[]>(() => {
    const overlays: BendAnnotation[] = []
    for (const match of matches) {
      const lineStart = parseLinePoint(match.display_cad_line_start ?? match.cad_line_start)
      const lineEnd = parseLinePoint(match.display_cad_line_end ?? match.cad_line_end)
      if (!lineStart || !lineEnd) continue
      const detectedLineStart = parseLinePoint(match.display_detected_line_start ?? match.detected_line_start)
      const detectedLineEnd = parseLinePoint(match.display_detected_line_end ?? match.detected_line_end)
      const calloutAnchor = parseLinePoint(match.callout_anchor) ?? midpoint3D(lineStart, lineEnd)
      overlays.push({
        id: match.bend_id,
        label: match.bend_id,
        position: midpoint3D(lineStart, lineEnd),
        calloutAnchor,
        lineStart,
        lineEnd,
        detectedLineStart: detectedLineStart ?? undefined,
        detectedLineEnd: detectedLineEnd ?? undefined,
        expectedAngle: match.target_angle,
        measuredAngle: match.measured_angle,
        deviation: match.angle_deviation,
        status: toViewerStatus(match.status),
        active: focusedBendId === match.bend_id,
        displayGeometryCanonical: !!(match.display_cad_line_start || match.display_detected_line_start),
        radiusDeviation: match.radius_deviation,
        lineCenterDeviationMm: match.line_center_deviation_mm,
        toleranceAngle: match.tolerance_angle,
        toleranceRadius: match.tolerance_radius,
      })
    }
    return overlays
  }, [focusedBendId, matches])
  const completed = matches.filter((match) => match.status !== 'NOT_DETECTED').length
  const inSpec = matches.filter((match) => match.status === 'PASS').length
  const remaining = Math.max(0, matches.length - completed)

  if (!bendAnnotations.length) return null

  const referenceUrl = referenceMeshUrl || `/api/reference-mesh/${jobId}.ply`
  const alignedScanUrl = `/api/aligned-scan/${jobId}.ply`

  return (
    <div className="rounded-lg border border-dark-600 bg-dark-800/40 p-3">
      <div className="flex items-center justify-between mb-3">
        <div>
          <p className="text-xs text-dark-400 uppercase tracking-wide">3D CAD Bend Overlay</p>
          <p className="text-xs text-dark-500 mt-1">
            Aligned scan over original CAD with bend lines colored by inspection status
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-dark-400">
          <LegendPill color="bg-primary-500" label={`${completed}/${matches.length} complete`} />
          <LegendPill color="bg-emerald-500" label={`${inSpec} in spec`} />
          <LegendPill color="bg-slate-500" label={`${remaining} remaining`} />
          <LegendPill color="bg-emerald-500" label="In spec" />
          <LegendPill color="bg-amber-500" label="Warning" />
          <LegendPill color="bg-red-500" label="Out of spec" />
          <LegendPill color="bg-slate-500" label="Not detected" />
        </div>
      </div>

      <div className="mb-3 rounded-xl border border-dark-600 bg-dark-900/75 px-4 py-3">
        {focusedMatch ? (
          <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[11px] uppercase tracking-[0.18em] text-dark-400">Selected bend</span>
                <span className={clsx(
                  'rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em]',
                  focusedMatch.status === 'PASS' && 'border-emerald-400/50 bg-emerald-500/12 text-emerald-200',
                  focusedMatch.status === 'WARNING' && 'border-amber-400/50 bg-amber-500/12 text-amber-200',
                  focusedMatch.status === 'FAIL' && 'border-red-400/50 bg-red-500/12 text-red-200',
                  focusedMatch.status === 'NOT_DETECTED' && 'border-dark-500 bg-dark-700 text-dark-300',
                )}>
                  {focusedMatch.status.replace('_', ' ')}
                </span>
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <div className="text-xl font-semibold text-dark-100">{focusedMatch.bend_id}</div>
                <div className="text-sm text-dark-400">
                  {focusedMatch.bend_form === 'ROLLED' ? 'Rolled bend' : 'Folded bend'}
                </div>
                {focusedMatch.action_item && (
                  <div className="text-sm text-dark-300">
                    {focusedMatch.action_item}
                  </div>
                )}
              </div>
              <p className="mt-2 text-xs text-dark-500">
                Viewer rotates to the selected bend’s inspection angle. Continue rotating manually to verify edge profile and detected overlay against the CAD reference.
              </p>
            </div>
            <div className="flex flex-wrap gap-2 lg:max-w-[520px] lg:justify-end">
              <FocusedMetricCard
                label="Expected"
                value={`${focusedMatch.target_angle.toFixed(1)}°`}
                tone="neutral"
              />
              <FocusedMetricCard
                label="Measured"
                value={focusedMatch.measured_angle != null ? `${focusedMatch.measured_angle.toFixed(1)}°` : '-'}
                tone="neutral"
              />
              <FocusedMetricCard
                label="Δ Angle"
                value={formatSignedMetric(focusedMatch.angle_deviation, 1, '°')}
                tone={focusedMatch.status}
              />
              <FocusedMetricCard
                label="Δ Radius"
                value={formatSignedMetric(focusedMatch.radius_deviation, 2, 'mm')}
                tone={focusedMatch.status}
              />
              <FocusedMetricCard
                label="Δ Center"
                value={formatSignedMetric(focusedMatch.line_center_deviation_mm, 2, 'mm')}
                tone={focusedMatch.status}
              />
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-1 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-[11px] uppercase tracking-[0.18em] text-dark-400">Overview mode</p>
              <p className="mt-1 text-sm text-dark-200">
                {issueMatches.length
                  ? `${issueMatches.length} bends need attention. Select a bend chip below to isolate its diff and inspection angle.`
                  : 'No out-of-tolerance bends are active. Select any bend below to inspect its measured vs expected geometry.'}
              </p>
            </div>
            <div className="rounded-lg border border-dark-600 bg-dark-800/70 px-3 py-2 text-xs text-dark-300">
              The viewer keeps all warning/fail bends visible at once.
            </div>
          </div>
        )}
      </div>

      <Suspense fallback={
        <div className="h-[560px] rounded-lg bg-dark-900 flex items-center justify-center">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary-500" />
        </div>
      }>
        <ThreeViewer
          modelUrl={alignedScanUrl}
          referenceUrl={referenceUrl}
          bendAnnotations={bendAnnotations}
          focusPoint={focusPoint}
          focusAnnotationId={focusedBendId}
          className="h-[560px]"
        />
      </Suspense>
    </div>
  )
}

function FocusedMetricCard({
  label,
  value,
  tone,
}: {
  label: string
  value: string
  tone: BendMatch['status'] | 'neutral'
}) {
  const toneClasses: Record<typeof tone, string> = {
    neutral: 'border-dark-600 bg-dark-800/80 text-dark-100',
    PASS: 'border-emerald-400/35 bg-emerald-500/10 text-emerald-100',
    WARNING: 'border-amber-400/35 bg-amber-500/10 text-amber-100',
    FAIL: 'border-red-400/35 bg-red-500/10 text-red-100',
    NOT_DETECTED: 'border-dark-500 bg-dark-800/80 text-dark-300',
  }

  return (
    <div className={clsx('w-[120px] rounded-lg border px-3 py-2', toneClasses[tone])}>
      <div className="text-[10px] uppercase tracking-[0.16em] text-dark-400">{label}</div>
      <div className="mt-1 text-sm font-semibold">{value}</div>
    </div>
  )
}

function midpointFromMatch(match: BendMatch): [number, number, number] | null {
  const a = parseLinePoint(match.display_detected_line_start ?? match.detected_line_start)
  const b = parseLinePoint(match.display_detected_line_end ?? match.detected_line_end)
  if (a && b) return midpoint3D(a, b)
  const c = parseLinePoint(match.display_cad_line_start ?? match.cad_line_start)
  const d = parseLinePoint(match.display_cad_line_end ?? match.cad_line_end)
  if (c && d) return midpoint3D(c, d)
  return null
}

function BendArtifactsPanel({
  artifacts,
}: {
  artifacts?: BendInspectionJob['artifacts']
}) {
  if (!artifacts?.bend_overlay_overview_url && !artifacts?.bend_report_pdf_url) return null

  return (
    <div className="rounded-lg border border-dark-600 bg-dark-800/40 p-3">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <p className="text-xs text-dark-400 uppercase tracking-wide">Report Artifacts</p>
          <p className="text-xs text-dark-500 mt-1">
            Backend-generated overview and bend-inspection PDF aligned with the current overlay
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {artifacts.bend_report_pdf_url && (
            <a
              href={artifacts.bend_report_pdf_url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-lg border border-primary-500/40 bg-primary-500/10 px-3 py-2 text-xs font-medium text-primary-300 hover:bg-primary-500/15"
            >
              <Download className="w-3.5 h-3.5" />
              Download PDF
            </a>
          )}
        </div>
      </div>

      {artifacts.bend_overlay_overview_url && (
        <a href={artifacts.bend_overlay_overview_url} target="_blank" rel="noreferrer" className="block">
          <div className="overflow-hidden rounded-lg border border-dark-600 bg-dark-900/70">
            <img
              src={artifacts.bend_overlay_overview_url}
              alt="Bend overlay overview"
              className="w-full h-auto object-cover"
            />
          </div>
        </a>
      )}

      <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-dark-400">
        {artifacts.bend_overlay_issue_urls?.length ? (
          <span className="inline-flex items-center gap-1">
            <ImageIcon className="w-3.5 h-3.5" />
            {artifacts.bend_overlay_issue_urls.length} issue snapshot{artifacts.bend_overlay_issue_urls.length === 1 ? '' : 's'}
          </span>
        ) : (
          <span>No issue snapshots for this run</span>
        )}
      </div>
    </div>
  )
}

function LegendPill({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-dark-600 px-2 py-1">
      <span className={`h-2 w-2 rounded-full ${color}`} />
      {label}
    </span>
  )
}

function projectToView(
  p: [number, number, number],
  view: AxisView,
): [number, number] {
  const [a, b] = VIEW_AXES[view]
  return [p[a], p[b]]
}

function normalize2D(
  p: [number, number],
  bounds: { minX: number; maxX: number; minY: number; maxY: number },
  width: number,
  height: number,
  padding: number,
): [number, number] {
  const sx = Math.max(1e-6, bounds.maxX - bounds.minX)
  const sy = Math.max(1e-6, bounds.maxY - bounds.minY)
  const nx = (p[0] - bounds.minX) / sx
  const ny = (p[1] - bounds.minY) / sy
  const x = padding + nx * (width - padding * 2)
  const y = height - padding - ny * (height - padding * 2)
  return [x, y]
}

function viewBounds(matches: BendMatch[], view: AxisView) {
  const pts: Array<[number, number]> = []
  for (const m of matches) {
    const c0 = parseLinePoint(m.display_cad_line_start ?? m.cad_line_start)
    const c1 = parseLinePoint(m.display_cad_line_end ?? m.cad_line_end)
    const s0 = parseLinePoint(m.display_detected_line_start ?? m.detected_line_start)
    const s1 = parseLinePoint(m.display_detected_line_end ?? m.detected_line_end)
    if (c0) pts.push(projectToView(c0, view))
    if (c1) pts.push(projectToView(c1, view))
    if (s0) pts.push(projectToView(s0, view))
    if (s1) pts.push(projectToView(s1, view))
  }
  if (pts.length === 0) {
    return { minX: -1, maxX: 1, minY: -1, maxY: 1 }
  }
  const xs = pts.map((p) => p[0])
  const ys = pts.map((p) => p[1])
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const dx = Math.max(1.0, maxX - minX)
  const dy = Math.max(1.0, maxY - minY)
  return {
    minX: minX - dx * 0.06,
    maxX: maxX + dx * 0.06,
    minY: minY - dy * 0.06,
    maxY: maxY + dy * 0.06,
  }
}

function BendGeometryCue({
  matches,
  compact,
  focusedBendId,
}: {
  matches: BendMatch[]
  compact?: boolean
  focusedBendId?: string | null
}) {
  if (!matches.length) return null
  const hasCadGeometry = matches.some((m) => {
    const c0 = parseLinePoint(m.display_cad_line_start ?? m.cad_line_start)
    const c1 = parseLinePoint(m.display_cad_line_end ?? m.cad_line_end)
    return !!c0 && !!c1
  })
  const views: AxisView[] = compact ? ['XY'] : ['XY', 'XZ', 'YZ']
  const width = compact ? 360 : 320
  const height = compact ? 190 : 180
  const padding = 14

  return (
    <div className="rounded-lg border border-dark-600 bg-dark-800/40 p-3">
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs text-dark-400 uppercase tracking-wide">Bend Geometry Cue</p>
        {!compact && (
          <p className="text-xs text-dark-500">
            CAD bend IDs anchored to geometry, with detected overlays
          </p>
        )}
      </div>

      {!hasCadGeometry ? (
        <div className="rounded border border-dark-600 bg-dark-900/60 p-3 text-xs text-dark-400">
          Bend geometry coordinates are unavailable for this older report. Run a new bend inspection to render labeled CAD bend overlays.
        </div>
      ) : (
      <div className={clsx('grid gap-2', compact ? 'grid-cols-1' : 'grid-cols-3')}>
        {views.map((view) => {
          const bounds = viewBounds(matches, view)
          return (
            <div key={view} className="rounded border border-dark-600 bg-dark-900/60 p-2">
              <div className="text-[11px] text-dark-400 mb-1">{view} projection</div>
              <svg
                viewBox={`0 0 ${width} ${height}`}
                className="w-full h-auto rounded bg-[#0d1728]"
                role="img"
                aria-label={`Bend geometry ${view}`}
              >
                {matches.map((m) => {
                  const c0 = parseLinePoint(m.display_cad_line_start ?? m.cad_line_start)
                  const c1 = parseLinePoint(m.display_cad_line_end ?? m.cad_line_end)
                  if (!c0 || !c1) return null
                  const [x1, y1] = normalize2D(projectToView(c0, view), bounds, width, height, padding)
                  const [x2, y2] = normalize2D(projectToView(c1, view), bounds, width, height, padding)
                  const stroke = STATUS_STROKE[m.status] || STATUS_STROKE.NOT_DETECTED
                  const midX = (x1 + x2) / 2
                  const midY = (y1 + y2) / 2
                  const isFocused = focusedBendId === m.bend_id

                  const d0 = parseLinePoint(m.display_detected_line_start ?? m.detected_line_start)
                  const d1 = parseLinePoint(m.display_detected_line_end ?? m.detected_line_end)
                  const hasDetected = !!d0 && !!d1

                  return (
                    <g key={`${view}-${m.bend_id}`}>
                      <line
                        x1={x1}
                        y1={y1}
                        x2={x2}
                        y2={y2}
                        stroke={stroke}
                        strokeWidth={isFocused ? 3.6 : 2.2}
                        opacity={isFocused ? 1 : 0.8}
                      />
                      {hasDetected && d0 && d1 && (
                        <line
                          x1={normalize2D(projectToView(d0, view), bounds, width, height, padding)[0]}
                          y1={normalize2D(projectToView(d0, view), bounds, width, height, padding)[1]}
                          x2={normalize2D(projectToView(d1, view), bounds, width, height, padding)[0]}
                          y2={normalize2D(projectToView(d1, view), bounds, width, height, padding)[1]}
                          stroke="#38bdf8"
                          strokeWidth={isFocused ? 2.2 : 1.3}
                          strokeDasharray="4,3"
                          opacity={isFocused ? 1 : 0.9}
                        />
                      )}
                      <text
                        x={midX + 3}
                        y={midY - 3}
                        fill={isFocused ? '#ffffff' : '#e5e7eb'}
                        fontSize={isFocused ? '10' : '9'}
                        fontWeight={isFocused ? '700' : '600'}
                      >
                        {m.bend_id}
                      </text>
                    </g>
                  )
                })}
              </svg>
            </div>
          )
        })}
      </div>
      )}

      <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-dark-400">
        <span className="inline-flex items-center gap-1">
          <span className="inline-block w-3 h-0.5 bg-emerald-400" />
          In spec
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="inline-block w-3 h-0.5 bg-amber-400" />
          Warning
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="inline-block w-3 h-0.5 bg-red-400" />
          Out of spec
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="inline-block w-3 h-0.5 bg-slate-400" />
          Not detected
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="inline-block w-3 h-0.5 bg-sky-400 border-b border-dashed border-sky-300" />
          Detected overlay
        </span>
      </div>
    </div>
  )
}

function ScanQualityQueue({
  quality,
  compact,
}: {
  quality: ScanQualitySummary
  compact?: boolean
}) {
  const status = quality.status || 'UNKNOWN'
  const palette = {
    GOOD: {
      badge: 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40',
      panel: 'bg-emerald-500/5 border-emerald-500/30',
    },
    FAIR: {
      badge: 'bg-amber-500/20 text-amber-300 border border-amber-500/40',
      panel: 'bg-amber-500/5 border-amber-500/30',
    },
    POOR: {
      badge: 'bg-red-500/20 text-red-300 border border-red-500/40',
      panel: 'bg-red-500/5 border-red-500/30',
    },
  }[status] ?? {
    badge: 'bg-dark-600 text-dark-200 border border-dark-500',
    panel: 'bg-dark-700/40 border-dark-600',
  }

  const coverage = quality.coverage_pct != null ? `${quality.coverage_pct.toFixed(0)}%` : '-'
  const density = quality.density_pts_per_cm2 != null ? `${quality.density_pts_per_cm2.toFixed(1)}` : '-'
  const spacing = quality.median_spacing_mm != null ? `${quality.median_spacing_mm.toFixed(2)} mm` : '-'
  const score = quality.score != null ? `${quality.score.toFixed(0)}` : '-'

  return (
    <div className={clsx('rounded-lg border p-3', palette.panel)}>
      <div className="flex flex-wrap items-center gap-2 justify-between">
        <div className="flex items-center gap-2">
          <span className={clsx('px-2 py-1 rounded text-xs font-semibold', palette.badge)}>
            {quality.queue_label || `Scan Quality: ${status}`}
          </span>
          {!compact && (
            <span className="text-xs text-dark-300">Score {score}/100</span>
          )}
        </div>
        <div className="text-xs text-dark-300">
          Cov {coverage} | Density {density} pts/cm^2 | Spacing {spacing}
        </div>
      </div>
      {!compact && quality.notes && quality.notes.length > 0 && (
        <div className="mt-2 text-xs text-dark-300 space-y-1">
          {quality.notes.slice(0, 2).map((note, idx) => (
            <div key={`${idx}-${note}`}>- {note}</div>
          ))}
        </div>
      )}
    </div>
  )
}

function SummaryCard({
  label,
  value,
  subValue,
  color,
  compact,
}: {
  label: string
  value: string
  subValue?: string
  color?: 'emerald' | 'red' | 'amber'
  compact?: boolean
}) {
  const colorClasses = {
    emerald: 'text-emerald-400',
    red: 'text-red-400',
    amber: 'text-amber-400',
  }

  return (
    <div className={clsx(
      'bg-dark-700/50 rounded-lg',
      compact ? 'p-2' : 'p-3'
    )}>
      <p className="text-xs text-dark-500">{label}</p>
      <p className={clsx(
        'font-semibold',
        compact ? 'text-base' : 'text-lg',
        color ? colorClasses[color] : 'text-dark-100'
      )}>
        {value}
        {subValue && (
          <span className="text-dark-400 font-normal text-xs ml-1">
            ({subValue})
          </span>
        )}
      </p>
    </div>
  )
}

function BendRow({
  match,
  compact,
  focused,
  onFocus,
}: {
  match: BendMatch
  compact?: boolean
  focused?: boolean
  onFocus?: () => void
}) {
  const statusColors = {
    PASS: 'bg-emerald-500/20 text-emerald-400',
    FAIL: 'bg-red-500/20 text-red-400',
    WARNING: 'bg-amber-500/20 text-amber-400',
    NOT_DETECTED: 'bg-dark-600 text-dark-400',
  }

  return (
    <tr
      className={clsx(
        'border-b border-dark-700/50 cursor-pointer transition-colors',
        focused && 'bg-primary-500/8',
      )}
      onClick={onFocus}
    >
      <td className={clsx('py-2 text-dark-200', compact ? 'pr-2' : 'pr-4')}>
        <span className={clsx(focused && 'text-primary-300 font-semibold')}>
          {match.bend_id}
        </span>
      </td>
      <td className={clsx('py-2 text-dark-300', compact ? 'pr-2' : 'pr-4')}>
        {match.bend_form || '-'}
      </td>
      <td className={clsx('py-2 text-dark-300', compact ? 'pr-2' : 'pr-4')}>
        <div className="space-y-0.5">
          <div className="text-dark-200">Exp {match.target_angle.toFixed(1)}</div>
          <div>Act {match.measured_angle !== null ? match.measured_angle.toFixed(1) : '-'}</div>
          <div className={clsx(
            (match.angle_deviation !== null && Math.abs(match.angle_deviation) > match.tolerance_angle)
              ? 'text-red-400'
              : 'text-emerald-400'
          )}>
            Δ {match.angle_deviation !== null ? `${match.angle_deviation >= 0 ? '+' : ''}${match.angle_deviation.toFixed(1)}` : '-'}
          </div>
        </div>
      </td>
      <td className={clsx('py-2 text-dark-300', compact ? 'pr-2' : 'pr-4')}>
        <div className="space-y-0.5">
          <div>Exp {match.target_radius.toFixed(2)}</div>
          <div>Act {match.measured_radius !== null ? match.measured_radius.toFixed(2) : '-'}</div>
          <div className={clsx(
            (match.radius_deviation !== null && Math.abs(match.radius_deviation) > match.tolerance_radius)
              ? 'text-red-400'
              : 'text-emerald-400'
          )}>
            Δ {match.radius_deviation !== null ? `${match.radius_deviation >= 0 ? '+' : ''}${match.radius_deviation.toFixed(2)}` : '-'}
          </div>
        </div>
      </td>
      <td className={clsx('py-2 text-dark-300', compact ? 'pr-2' : 'pr-4')}>
        <div className="space-y-0.5">
          <div>
            Len Exp {match.expected_line_length_mm !== null && match.expected_line_length_mm !== undefined
              ? `${match.expected_line_length_mm.toFixed(2)}`
              : '-'}
          </div>
          <div>
            Len Act {match.actual_line_length_mm !== null && match.actual_line_length_mm !== undefined
              ? `${match.actual_line_length_mm.toFixed(2)}`
              : '-'}
          </div>
          <div className={clsx(
            (match.line_length_deviation_mm !== null && match.line_length_deviation_mm !== undefined && Math.abs(match.line_length_deviation_mm) > match.tolerance_radius)
              ? 'text-red-400'
              : 'text-emerald-400'
          )}>
            Len Δ {match.line_length_deviation_mm !== null && match.line_length_deviation_mm !== undefined
              ? `${match.line_length_deviation_mm >= 0 ? '+' : ''}${match.line_length_deviation_mm.toFixed(2)}`
              : '-'}
          </div>
          <div>
            Arc Exp {match.expected_arc_length_mm !== null && match.expected_arc_length_mm !== undefined
              ? `${match.expected_arc_length_mm.toFixed(2)}`
              : '-'}
          </div>
          <div>
            Arc Act {match.actual_arc_length_mm !== null && match.actual_arc_length_mm !== undefined
              ? `${match.actual_arc_length_mm.toFixed(2)}`
              : '-'}
          </div>
          <div className={clsx(
            (match.arc_length_deviation_mm !== null && match.arc_length_deviation_mm !== undefined && Math.abs(match.arc_length_deviation_mm) > match.tolerance_radius)
              ? 'text-red-400'
              : 'text-emerald-400'
          )}>
            Arc Δ {match.arc_length_deviation_mm !== null && match.arc_length_deviation_mm !== undefined
              ? `${match.arc_length_deviation_mm >= 0 ? '+' : ''}${match.arc_length_deviation_mm.toFixed(2)}`
              : '-'}
          </div>
          <div className={clsx(
            (match.line_center_deviation_mm !== null && match.line_center_deviation_mm !== undefined && Math.abs(match.line_center_deviation_mm) > match.tolerance_radius)
              ? 'text-red-400'
              : 'text-emerald-400'
          )}>
            Pos Δ {match.line_center_deviation_mm !== null && match.line_center_deviation_mm !== undefined
              ? `${match.line_center_deviation_mm >= 0 ? '+' : ''}${match.line_center_deviation_mm.toFixed(2)}`
              : '-'}
          </div>
        </div>
      </td>
      <td className="py-2">
        <span className={clsx(
          'px-2 py-0.5 rounded text-xs font-medium',
          statusColors[match.status]
        )}>
          {match.status === 'NOT_DETECTED' ? 'Not yet' : match.status}
        </span>
      </td>
      {!compact && (
        <td className="py-2 pl-3 text-dark-300 max-w-[300px]">
          <div className="text-xs">
            {match.action_item || '-'}
          </div>
          {match.issue_location && match.issue_location !== 'none' && (
            <div className="text-[11px] text-dark-500 mt-1">
              Location: {match.issue_location}
            </div>
          )}
        </td>
      )}
    </tr>
  )
}
