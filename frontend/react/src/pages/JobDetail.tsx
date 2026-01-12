import { useParams, useNavigate, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  ArrowLeft,
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  Trash2,
  RefreshCw,
  FileText,
  Ruler,
  Target,
  Zap,
  Box,
  type LucideIcon,
} from 'lucide-react'
import { jobsApi } from '../services/api'
import clsx from 'clsx'
import { useEffect, useState, lazy, Suspense } from 'react'

// Lazy load the 3D viewer for better performance
const ThreeViewer = lazy(() => import('../components/ThreeViewer'))

export default function JobDetail() {
  const { jobId } = useParams<{ jobId: string }>()
  const navigate = useNavigate()
  const [progress, setProgress] = useState(0)
  const [currentStep, setCurrentStep] = useState('')

  const { data: job, isLoading, refetch } = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => jobsApi.get(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const data = query.state.data
      return data?.status === 'running' || data?.status === 'pending' ? 2000 : false
    },
  })

  const { data: result } = useQuery({
    queryKey: ['job-result', jobId],
    queryFn: () => jobsApi.getResult(jobId!),
    enabled: !!jobId && job?.status === 'completed',
  })

  // Poll progress for running jobs
  useEffect(() => {
    if (job?.status !== 'running') return

    const pollProgress = async () => {
      try {
        const progressData = await jobsApi.getProgress(jobId!)
        setProgress(progressData.progress)
        setCurrentStep(progressData.current_step)
      } catch {
        // Ignore errors
      }
    }

    pollProgress()
    const interval = setInterval(pollProgress, 1000)
    return () => clearInterval(interval)
  }, [jobId, job?.status])

  const handleDelete = async () => {
    if (confirm('Are you sure you want to delete this job?')) {
      await jobsApi.delete(jobId!)
      navigate('/jobs')
    }
  }

  if (isLoading) {
    return (
      <div className="animate-pulse space-y-6">
        <div className="h-8 bg-dark-700 rounded w-1/4" />
        <div className="h-64 bg-dark-700 rounded-lg" />
        <div className="h-96 bg-dark-700 rounded-lg" />
      </div>
    )
  }

  if (!job) {
    return (
      <div className="text-center py-12">
        <AlertTriangle className="w-16 h-16 text-dark-500 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-dark-100 mb-2">Job Not Found</h2>
        <p className="text-dark-400 mb-4">The requested job could not be found.</p>
        <Link to="/jobs" className="btn btn-primary">
          Back to Jobs
        </Link>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <button
            onClick={() => navigate('/jobs')}
            className="mr-4 p-2 hover:bg-dark-700 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5 text-dark-400" />
          </button>
          <div>
            <h1 className="text-2xl font-bold text-dark-100">{job.part_id}</h1>
            <p className="text-dark-400">{job.part_name}</p>
          </div>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => refetch()}
            className="btn btn-secondary flex items-center"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </button>
          <button
            onClick={handleDelete}
            className="btn btn-secondary text-red-400 hover:bg-red-500/20 flex items-center"
          >
            <Trash2 className="w-4 h-4 mr-2" />
            Delete
          </button>
        </div>
      </div>

      {/* Status and progress */}
      {(job.status === 'running' || job.status === 'pending') && (
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <Clock className="w-5 h-5 text-amber-400 mr-2 animate-pulse" />
              <span className="font-medium text-dark-100">
                {job.status === 'pending' ? 'Waiting to start...' : 'Analysis in progress...'}
              </span>
            </div>
            <span className="text-dark-400">{Math.round(progress)}%</span>
          </div>
          <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary-500 to-secondary-500 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          {currentStep && (
            <p className="mt-3 text-sm text-dark-400">{currentStep}</p>
          )}
        </div>
      )}

      {/* Job info cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <InfoCard
          icon={job.status === 'completed' ? CheckCircle : job.status === 'failed' ? XCircle : Clock}
          iconColor={
            job.status === 'completed'
              ? 'text-emerald-400'
              : job.status === 'failed'
              ? 'text-red-400'
              : 'text-amber-400'
          }
          label="Status"
          value={job.status}
        />
        <InfoCard
          icon={FileText}
          iconColor="text-primary-400"
          label="Material"
          value={job.material.replace('_', ' ')}
        />
        <InfoCard
          icon={Ruler}
          iconColor="text-secondary-400"
          label="Tolerance"
          value={`+/- ${job.tolerance} mm`}
        />
        <InfoCard
          icon={Target}
          iconColor={
            job.quality_score !== null
              ? job.quality_score >= 90
                ? 'text-emerald-400'
                : job.quality_score >= 70
                ? 'text-amber-400'
                : 'text-red-400'
              : 'text-dark-400'
          }
          label="Quality Score"
          value={job.quality_score !== null ? `${job.quality_score.toFixed(1)}%` : '-'}
        />
      </div>

      {/* Error display */}
      {job.status === 'failed' && job.error && (
        <div className="card p-6 border-red-500/20 bg-red-500/5">
          <div className="flex items-start">
            <XCircle className="w-5 h-5 text-red-400 mr-3 mt-0.5 flex-shrink-0" />
            <div>
              <h3 className="font-semibold text-red-400 mb-2">Analysis Failed</h3>
              <p className="text-dark-300">{job.error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {job.status === 'completed' && result && (
        <>
          {/* Summary */}
          <div className="card p-6">
            <h3 className="font-semibold text-dark-100 mb-4">Analysis Summary</h3>
            <div className="prose prose-invert max-w-none">
              <p className="text-dark-300 whitespace-pre-wrap">
                {result.qc_result?.summary || 'No summary available'}
              </p>
            </div>
          </div>

          {/* Defects */}
          {result.qc_result?.defects && result.qc_result.defects.length > 0 && (
            <div className="card p-6">
              <h3 className="font-semibold text-dark-100 mb-4">
                Detected Defects ({result.qc_result.defects.length})
              </h3>
              <div className="space-y-3">
                {result.qc_result.defects.map((defect: any, index: number) => (
                  <div
                    key={index}
                    className={clsx(
                      'p-4 rounded-lg border',
                      defect.severity === 'critical'
                        ? 'bg-red-500/10 border-red-500/20'
                        : defect.severity === 'major'
                        ? 'bg-amber-500/10 border-amber-500/20'
                        : 'bg-dark-700/50 border-dark-600'
                    )}
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <p className="font-medium text-dark-100">{defect.type}</p>
                        <p className="text-sm text-dark-400 mt-1">{defect.description}</p>
                      </div>
                      <span
                        className={clsx(
                          'px-2.5 py-1 rounded-full text-xs font-medium',
                          defect.severity === 'critical'
                            ? 'bg-red-500/20 text-red-400'
                            : defect.severity === 'major'
                            ? 'bg-amber-500/20 text-amber-400'
                            : 'bg-dark-600 text-dark-300'
                        )}
                      >
                        {defect.severity}
                      </span>
                    </div>
                    {defect.location && (
                      <p className="text-xs text-dark-500 mt-2">
                        Location: {defect.location}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* GD&T Results */}
          {result.qc_result?.gdt_results && Object.keys(result.qc_result.gdt_results).length > 0 && (
            <div className="card p-6">
              <h3 className="font-semibold text-dark-100 mb-4">GD&T Measurements</h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-dark-700/50">
                    <tr>
                      <th className="text-left px-4 py-2 text-xs font-medium text-dark-400 uppercase">
                        Characteristic
                      </th>
                      <th className="text-left px-4 py-2 text-xs font-medium text-dark-400 uppercase">
                        Value
                      </th>
                      <th className="text-left px-4 py-2 text-xs font-medium text-dark-400 uppercase">
                        Tolerance
                      </th>
                      <th className="text-left px-4 py-2 text-xs font-medium text-dark-400 uppercase">
                        Status
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-dark-700">
                    {Object.entries(result.qc_result.gdt_results).map(
                      ([key, value]: [string, any]) => (
                        <tr key={key}>
                          <td className="px-4 py-3 text-dark-100 capitalize">
                            {key.replace(/_/g, ' ')}
                          </td>
                          <td className="px-4 py-3 text-dark-300">
                            {typeof value === 'number' ? value.toFixed(4) : String(value)}
                          </td>
                          <td className="px-4 py-3 text-dark-400">
                            +/- {job.tolerance} mm
                          </td>
                          <td className="px-4 py-3">
                            <span
                              className={clsx(
                                'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium',
                                typeof value === 'number' && value <= job.tolerance
                                  ? 'bg-emerald-500/20 text-emerald-400'
                                  : 'bg-red-500/20 text-red-400'
                              )}
                            >
                              {typeof value === 'number' && value <= job.tolerance
                                ? 'Pass'
                                : 'Fail'}
                            </span>
                          </td>
                        </tr>
                      )
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* 3D Viewer */}
          {(job.scan_path || job.reference_path) && (
            <div className="card p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-dark-100">3D Model Viewer</h3>
                <Box className="w-5 h-5 text-dark-400" />
              </div>
              <Suspense
                fallback={
                  <div className="h-[400px] bg-dark-900 rounded-lg flex items-center justify-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
                  </div>
                }
              >
                <ThreeViewer
                  modelUrl={job.scan_path ? `/api/files/${job.scan_path}` : undefined}
                  referenceUrl={job.reference_path ? `/api/files/${job.reference_path}` : undefined}
                  tolerance={job.tolerance}
                  className="h-[400px]"
                />
              </Suspense>
            </div>
          )}

          {/* Heatmap */}
          {result.heatmap_path && (
            <div className="card p-6">
              <h3 className="font-semibold text-dark-100 mb-4">Deviation Heatmap</h3>
              <div className="bg-dark-700/50 rounded-lg overflow-hidden">
                <img
                  src={`/api/files/${result.heatmap_path}`}
                  alt="Deviation heatmap"
                  className="w-full h-auto"
                />
              </div>
            </div>
          )}

          {/* Recommendations */}
          {result.qc_result?.recommendations && result.qc_result.recommendations.length > 0 && (
            <div className="card p-6">
              <h3 className="font-semibold text-dark-100 mb-4">Recommendations</h3>
              <ul className="space-y-2">
                {result.qc_result.recommendations.map((rec: string, index: number) => (
                  <li key={index} className="flex items-start">
                    <Zap className="w-4 h-4 text-primary-400 mr-2 mt-1 flex-shrink-0" />
                    <span className="text-dark-300">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}

      {/* Metadata */}
      <div className="card p-6">
        <h3 className="font-semibold text-dark-100 mb-4">Job Information</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-dark-500">Job ID</p>
            <p className="text-dark-300 font-mono">{job.job_id}</p>
          </div>
          <div>
            <p className="text-dark-500">Created</p>
            <p className="text-dark-300">{new Date(job.created_at).toLocaleString()}</p>
          </div>
          {job.completed_at && (
            <div>
              <p className="text-dark-500">Completed</p>
              <p className="text-dark-300">{new Date(job.completed_at).toLocaleString()}</p>
            </div>
          )}
          <div>
            <p className="text-dark-500">Duration</p>
            <p className="text-dark-300">
              {job.completed_at
                ? `${Math.round(
                    (new Date(job.completed_at).getTime() - new Date(job.created_at).getTime()) /
                      1000
                  )}s`
                : '-'}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

function InfoCard({
  icon: Icon,
  iconColor,
  label,
  value,
}: {
  icon: LucideIcon
  iconColor: string
  label: string
  value: string
}) {
  return (
    <div className="card p-4">
      <div className="flex items-center">
        <div className="w-10 h-10 bg-dark-700 rounded-lg flex items-center justify-center mr-3">
          <Icon className={clsx('w-5 h-5', iconColor)} />
        </div>
        <div>
          <p className="text-xs text-dark-500">{label}</p>
          <p className="font-medium text-dark-100 capitalize">{value}</p>
        </div>
      </div>
    </div>
  )
}
