import { useState, useCallback } from 'react'
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
} from 'lucide-react'
import clsx from 'clsx'
import { bendInspectionApi, getErrorMessage, type BendInspectionJob, type BendMatch } from '../services/api'

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
                <BendInspectionResults report={selectedJob.report} />
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
          <p className="text-xs text-dark-400">
            {new Date(job.created_at).toLocaleString()}
          </p>
        </div>

        {job.status === 'completed' && job.report && (
          <div className="text-right mr-4">
            <p className="text-sm font-medium text-dark-100">
              {job.report.summary.detected}/{job.report.summary.total_bends}
            </p>
            <p className="text-xs text-dark-400">bends detected</p>
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
          <BendInspectionResults report={job.report} compact />
        </div>
      )}
    </div>
  )
}

// Bend inspection results component
function BendInspectionResults({
  report,
  compact = false,
}: {
  report: BendInspectionJob['report']
  compact?: boolean
}) {
  if (!report) return null

  const { summary, matches } = report

  return (
    <div className={clsx('space-y-4', compact && 'space-y-2')}>
      {/* Summary */}
      <div className={clsx(
        'grid gap-3',
        compact ? 'grid-cols-4' : 'grid-cols-2 md:grid-cols-4'
      )}>
        <SummaryCard
          label="Progress"
          value={`${summary.detected}/${summary.total_bends}`}
          subValue={`${summary.progress_pct.toFixed(0)}%`}
          compact={compact}
        />
        <SummaryCard
          label="Passed"
          value={summary.passed.toString()}
          color="emerald"
          compact={compact}
        />
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
      </div>

      {/* Bend table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-dark-500 border-b border-dark-700">
              <th className={clsx('pb-2', compact ? 'pr-2' : 'pr-4')}>Bend</th>
              <th className={clsx('pb-2', compact ? 'pr-2' : 'pr-4')}>Target</th>
              <th className={clsx('pb-2', compact ? 'pr-2' : 'pr-4')}>Measured</th>
              <th className={clsx('pb-2', compact ? 'pr-2' : 'pr-4')}>Deviation</th>
              <th className="pb-2">Status</th>
            </tr>
          </thead>
          <tbody>
            {matches.map((match) => (
              <BendRow key={match.bend_id} match={match} compact={compact} />
            ))}
          </tbody>
        </table>
      </div>
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

function BendRow({ match, compact }: { match: BendMatch; compact?: boolean }) {
  const statusColors = {
    PASS: 'bg-emerald-500/20 text-emerald-400',
    FAIL: 'bg-red-500/20 text-red-400',
    WARNING: 'bg-amber-500/20 text-amber-400',
    NOT_DETECTED: 'bg-dark-600 text-dark-400',
  }

  return (
    <tr className="border-b border-dark-700/50">
      <td className={clsx('py-2 text-dark-200', compact ? 'pr-2' : 'pr-4')}>
        {match.bend_id}
      </td>
      <td className={clsx('py-2 text-dark-300', compact ? 'pr-2' : 'pr-4')}>
        {match.target_angle.toFixed(1)}°
      </td>
      <td className={clsx('py-2 text-dark-300', compact ? 'pr-2' : 'pr-4')}>
        {match.measured_angle !== null ? `${match.measured_angle.toFixed(1)}°` : '-'}
      </td>
      <td className={clsx('py-2', compact ? 'pr-2' : 'pr-4')}>
        {match.angle_deviation !== null ? (
          <span className={clsx(
            Math.abs(match.angle_deviation) > match.tolerance_angle
              ? 'text-red-400'
              : 'text-emerald-400'
          )}>
            {match.angle_deviation >= 0 ? '+' : ''}{match.angle_deviation.toFixed(1)}°
          </span>
        ) : '-'}
      </td>
      <td className="py-2">
        <span className={clsx(
          'px-2 py-0.5 rounded text-xs font-medium',
          statusColors[match.status]
        )}>
          {match.status === 'NOT_DETECTED' ? 'Not yet' : match.status}
        </span>
      </td>
    </tr>
  )
}
