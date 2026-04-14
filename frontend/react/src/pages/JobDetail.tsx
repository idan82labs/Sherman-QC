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
  BarChart3,
  Download,
  Layers,
  FileSpreadsheet,
  Brain,
  type LucideIcon,
} from 'lucide-react'
import { jobsApi } from '../services/api'
import clsx from 'clsx'
import { useEffect, useState, lazy, Suspense, useMemo } from 'react'
import { BendResult, Correlation2D3D, PipelineStatus, DimensionAnalysisResult } from '../types'
import type { BendAnnotation, DeviationStats } from '../components/ThreeViewer'

// Lazy load the 3D viewer for better performance
const ThreeViewer = lazy(() => import('../components/ThreeViewer'))

// Lazy load enhanced analysis components
const BendAnalysisPanel = lazy(() => import('../components/BendAnalysisPanel'))
const FeatureCorrelationView = lazy(() => import('../components/FeatureCorrelationView'))
const ModelPipelineStatus = lazy(() => import('../components/ModelPipelineStatus'))
const DrawingAnalysisPanel = lazy(() => import('../components/DrawingAnalysisPanel'))
const DimensionAnalysisPanel = lazy(() => import('../components/DimensionAnalysisPanel'))

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

  // Get the result - it can come from job.result or from the separate result endpoint
  const { data: resultData } = useQuery({
    queryKey: ['job-result', jobId],
    queryFn: () => jobsApi.getResult(jobId!),
    enabled: !!jobId && job?.status === 'completed',
  })

  // Normalize result data - handle both embedded and separate result structures
  const qcResult = job?.result || resultData?.result || null

  // Poll progress for running jobs
  useEffect(() => {
    if (job?.status !== 'running') return

    const pollProgress = async () => {
      try {
        const progressData = await jobsApi.getProgress(jobId!)
        setProgress(progressData.progress || 0)
        setCurrentStep(progressData.current_step || '')
      } catch {
        // Ignore polling errors
      }
    }

    pollProgress()
    const interval = setInterval(pollProgress, 1000)
    return () => clearInterval(interval)
  }, [jobId, job?.status])

  const handleDelete = async () => {
    if (confirm('Are you sure you want to delete this job?')) {
      try {
        await jobsApi.delete(jobId!)
        navigate('/jobs')
      } catch (err) {
        console.error('Failed to delete job:', err)
      }
    }
  }

  const handleExportPdf = async () => {
    try {
      const response = await fetch(`/api/download/${jobId}/pdf`)
      if (!response.ok) throw new Error('Failed to download PDF')

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `QC_Report_${job?.part_id || jobId}.pdf`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err) {
      console.error('Failed to export PDF:', err)
      alert('Failed to export PDF report')
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

  // Safe accessors for job fields
  const qualityScore = job.quality_score ?? qcResult?.quality_score ?? null
  const material = job.material || 'Unknown'
  const tolerance = job.tolerance ?? 0.1
  const operatorBrief = buildOperatorBrief(qcResult)

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
            <h1 className="text-2xl font-bold text-dark-100">{job.part_id || 'Unknown Part'}</h1>
            <p className="text-dark-400">{job.part_name || ''}</p>
          </div>
        </div>
        <div className="flex gap-2">
          {job.status === 'completed' && (
            <button
              onClick={handleExportPdf}
              className="btn btn-primary flex items-center"
            >
              <Download className="w-4 h-4 mr-2" />
              Export PDF
            </button>
          )}
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
          value={job.status || 'unknown'}
        />
        <InfoCard
          icon={FileText}
          iconColor="text-primary-400"
          label="Material"
          value={material.replace(/_/g, ' ')}
        />
        <InfoCard
          icon={Ruler}
          iconColor="text-secondary-400"
          label="Tolerance"
          value={`+/- ${tolerance} mm`}
        />
        <InfoCard
          icon={Target}
          iconColor={
            qualityScore != null
              ? qualityScore >= 90
                ? 'text-emerald-400'
                : qualityScore >= 70
                ? 'text-amber-400'
                : 'text-red-400'
              : 'text-dark-400'
          }
          label="Quality Score"
          value={qualityScore != null ? `${qualityScore.toFixed(1)}%` : '-'}
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
      {job.status === 'completed' && qcResult && (
        <>
          {/* Summary */}
          <div className="card p-6">
            <h3 className="font-semibold text-dark-100 mb-4">Operator Brief</h3>
            <div className="space-y-2">
              <p className="text-dark-200">{operatorBrief.headline}</p>
              {!!operatorBrief.actions.length && (
                <ul className="text-sm text-dark-300 space-y-1">
                  {operatorBrief.actions.map((line, idx) => (
                    <li key={idx}>{line}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          {/* Statistics */}
          {qcResult.statistics && (
            <div className="card p-6">
              <h3 className="font-semibold text-dark-100 mb-4 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2" />
                Deviation Statistics
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatItem
                  label="Mean Deviation"
                  value={formatNumber(qcResult.statistics.mean_deviation_mm ?? qcResult.statistics.mean_deviation, 'mm')}
                />
                <StatItem
                  label="Max Deviation"
                  value={formatNumber(qcResult.statistics.max_deviation_mm ?? qcResult.statistics.max_deviation, 'mm')}
                />
                <StatItem
                  label="Min Deviation"
                  value={formatNumber(qcResult.statistics.min_deviation_mm ?? qcResult.statistics.min_deviation, 'mm')}
                />
                <StatItem
                  label="Std Deviation"
                  value={formatNumber(qcResult.statistics.std_deviation_mm ?? qcResult.statistics.std_deviation, 'mm')}
                />
                <StatItem
                  label="Total Points"
                  value={qcResult.statistics.total_points?.toLocaleString() ?? '-'}
                />
                <StatItem
                  label="Points In Tolerance"
                  value={qcResult.statistics.points_in_tolerance?.toLocaleString() ?? '-'}
                />
                <StatItem
                  label="Pass Rate"
                  value={formatNumber(qcResult.statistics.pass_rate ?? qcResult.statistics.conformance_percentage, '%')}
                />
                <StatItem
                  label="Overall Result"
                  value={qcResult.overall_result || '-'}
                  highlight={qcResult.overall_result === 'PASS' ? 'green' : qcResult.overall_result === 'FAIL' ? 'red' : undefined}
                />
                <StatItem
                  label="Release Decision"
                  value={qcResult.release_decision || 'LEGACY_ONLY'}
                  highlight={qcResult.release_decision === 'AUTO_PASS' ? 'green' : qcResult.release_decision === 'AUTO_FAIL' ? 'red' : undefined}
                />
              </div>
            </div>
          )}

          {/* Regional Analysis */}
          {Array.isArray(qcResult.regions) && qcResult.regions.length > 0 && (
            <div className="card p-6">
              <h3 className="font-semibold text-dark-100 mb-4 flex items-center">
                <Layers className="w-5 h-5 mr-2 text-primary-400" />
                Regional Analysis
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-dark-500 border-b border-dark-700">
                      <th className="pb-3 pr-4">Region</th>
                      <th className="pb-3 pr-4">Points</th>
                      <th className="pb-3 pr-4">Mean Dev.</th>
                      <th className="pb-3 pr-4">Max Dev.</th>
                      <th className="pb-3 pr-4">Pass Rate</th>
                      <th className="pb-3">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {qcResult.regions.map((region: { name: string; point_count: number; mean_deviation_mm: number; max_deviation_mm: number; pass_rate: number; status: string }, index: number) => (
                      <tr key={index} className="border-b border-dark-700/50">
                        <td className="py-3 pr-4 text-dark-200">{region.name}</td>
                        <td className="py-3 pr-4 text-dark-300">{region.point_count?.toLocaleString()}</td>
                        <td className="py-3 pr-4 text-dark-300">{region.mean_deviation_mm?.toFixed(4)}mm</td>
                        <td className="py-3 pr-4 text-dark-300">{region.max_deviation_mm?.toFixed(4)}mm</td>
                        <td className="py-3 pr-4 text-dark-300">{region.pass_rate?.toFixed(1)}%</td>
                        <td className="py-3">
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            region.status === 'OK' ? 'bg-green-500/20 text-green-400' :
                            region.status === 'ATTENTION' ? 'bg-red-500/20 text-red-400' :
                            'bg-yellow-500/20 text-yellow-400'
                          }`}>
                            {region.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Recommendations */}
          {Array.isArray(qcResult.recommendations) && qcResult.recommendations.length > 0 && (
            <div className="card p-6">
              <h3 className="font-semibold text-dark-100 mb-4">Recommendations</h3>
              <ul className="space-y-3">
                {qcResult.recommendations.map((rec: string | { priority?: string; action?: string; expected_improvement?: string }, index: number) => {
                  // Handle both string and object recommendations
                  const isObject = typeof rec === 'object' && rec !== null
                  const action = isObject ? rec.action : rec
                  const priority = isObject ? rec.priority : null
                  const improvement = isObject ? rec.expected_improvement : null

                  return (
                    <li key={index} className="flex items-start">
                      <Zap className={`w-4 h-4 mr-2 mt-1 flex-shrink-0 ${
                        priority === 'HIGH' ? 'text-red-400' :
                        priority === 'MEDIUM' ? 'text-yellow-400' :
                        'text-primary-400'
                      }`} />
                      <div className="flex-1">
                        {priority && (
                          <span className={`text-xs font-medium px-2 py-0.5 rounded mr-2 ${
                            priority === 'HIGH' ? 'bg-red-500/20 text-red-300' :
                            priority === 'MEDIUM' ? 'bg-yellow-500/20 text-yellow-300' :
                            'bg-green-500/20 text-green-300'
                          }`}>
                            {priority}
                          </span>
                        )}
                        <span className="text-dark-300">{action}</span>
                        {improvement && (
                          <p className="text-dark-500 text-sm mt-1">Expected: {improvement}</p>
                        )}
                      </div>
                    </li>
                  )
                })}
              </ul>
            </div>
          )}

          {/* Dimension Analysis Section (Non-AI) */}
          {qcResult.dimension_analysis && (
            <DimensionAnalysisSection dimensionAnalysis={qcResult.dimension_analysis} />
          )}

          {/* Enhanced Analysis Sections (AI-Powered) */}
          <EnhancedAnalysisSections qcResult={qcResult} tolerance={tolerance} />

          {/* 3D Viewer - only show for supported formats (STL, PLY) */}
          <ThreeViewerSection job={job} tolerance={tolerance} qcResult={qcResult} />
        </>
      )}

      {/* Metadata */}
      <div className="card p-6">
        <h3 className="font-semibold text-dark-100 mb-4">Job Information</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-dark-500">Job ID</p>
            <p className="text-dark-300 font-mono">{job.job_id || jobId}</p>
          </div>
          <div>
            <p className="text-dark-500">Created</p>
            <p className="text-dark-300">{formatDate(job.created_at)}</p>
          </div>
          {job.completed_at && (
            <div>
              <p className="text-dark-500">Completed</p>
              <p className="text-dark-300">{formatDate(job.completed_at)}</p>
            </div>
          )}
          <div>
            <p className="text-dark-500">Duration</p>
            <p className="text-dark-300">{formatDuration(job.created_at, job.completed_at)}</p>
          </div>
        </div>
      </div>
    </div>
  )
}

// Helper components
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

function StatItem({
  label,
  value,
  highlight
}: {
  label: string
  value: string
  highlight?: 'green' | 'red'
}) {
  return (
    <div className="bg-dark-700/50 rounded-lg p-3">
      <p className="text-xs text-dark-500 mb-1">{label}</p>
      <p className={clsx(
        'font-medium',
        highlight === 'green' ? 'text-emerald-400' :
        highlight === 'red' ? 'text-red-400' :
        'text-dark-100'
      )}>
        {value}
      </p>
    </div>
  )
}

function DimensionAnalysisSection({ dimensionAnalysis }: { dimensionAnalysis: DimensionAnalysisResult }) {
  return (
    <div className="card p-6 border-emerald-500/20">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <FileSpreadsheet className="w-5 h-5 text-emerald-400 mr-2" />
          <h3 className="font-semibold text-dark-100">Dimension Analysis</h3>
          <span className="ml-3 text-xs bg-emerald-500/20 text-emerald-400 px-2 py-1 rounded">
            Specification-Based
          </span>
        </div>
        <div className="text-xs text-dark-500">
          Non-AI comparison from XLSX specifications
        </div>
      </div>
      <Suspense fallback={
        <div className="animate-pulse h-48 bg-dark-700 rounded" />
      }>
        <DimensionAnalysisPanel dimensionAnalysis={dimensionAnalysis} />
      </Suspense>
    </div>
  )
}

function EnhancedAnalysisSections({ qcResult, tolerance }: { qcResult: any; tolerance: number }) {
  // Check if enhanced analysis data exists
  const bendResults: BendResult[] = qcResult?.bend_results || []
  const correlation: Correlation2D3D | null = qcResult?.correlation_2d_3d || null
  const pipelineStatus: PipelineStatus | null = qcResult?.pipeline_status || null
  const drawingExtraction = qcResult?.drawing_extraction || qcResult?.enhanced_analysis?.drawing_extraction || null
  const aiDetailedAnalysis = qcResult?.ai_detailed_analysis || ''
  const drawingContext = qcResult?.drawing_context || ''

  // Check if there's any content to show
  const hasDrawingAnalysis = drawingExtraction || aiDetailedAnalysis || drawingContext
  const hasEnhancedData = bendResults.length > 0 || correlation || pipelineStatus || hasDrawingAnalysis

  // Don't render if no enhanced data
  if (!hasEnhancedData) {
    return null
  }

  return (
    <Suspense fallback={
      <div className="card p-6 animate-pulse">
        <div className="h-6 bg-dark-700 rounded w-1/4 mb-4" />
        <div className="h-48 bg-dark-700 rounded" />
      </div>
    }>
      {/* Pipeline Status */}
      {pipelineStatus && (
        <div className="mb-6">
          <ModelPipelineStatus status={pipelineStatus} />
        </div>
      )}

      {/* AI Section Header */}
      {hasEnhancedData && (
        <div className="flex items-center mb-4">
          <Brain className="w-5 h-5 text-amber-400 mr-2" />
          <h3 className="font-semibold text-dark-100">AI-Powered Analysis</h3>
          <span className="ml-3 text-xs bg-amber-500/20 text-amber-400 px-2 py-1 rounded">
            Multi-Model AI
          </span>
        </div>
      )}

      {/* AI Drawing Analysis */}
      {hasDrawingAnalysis && (
        <div className="mb-6">
          <DrawingAnalysisPanel
            drawingExtraction={drawingExtraction}
            aiDetailedAnalysis={aiDetailedAnalysis}
            drawingContext={drawingContext}
          />
        </div>
      )}

      {/* Bend Analysis (AI-detected) */}
      {bendResults.length > 0 && (
        <div className="card p-6 mb-6 border-amber-500/10">
          <div className="flex items-center mb-4">
            <Layers className="w-5 h-5 text-primary-400 mr-2" />
            <h3 className="font-semibold text-dark-100">AI Bend Detection</h3>
            <span className="ml-2 text-xs text-dark-400">
              {bendResults.length} bend{bendResults.length !== 1 ? 's' : ''} detected
            </span>
          </div>
          <BendAnalysisPanel bends={bendResults} tolerance={tolerance} />
        </div>
      )}

      {/* 2D-3D Correlation */}
      {correlation && (
        <div className="mb-6">
          <FeatureCorrelationView correlation={correlation} />
        </div>
      )}
    </Suspense>
  )
}

function ThreeViewerSection({ job, tolerance, qcResult }: { job: any; tolerance: number; qcResult?: any }) {
  const [deviations, setDeviations] = useState<number[] | undefined>(undefined)
  const [deviationsLoading, setDeviationsLoading] = useState(false)
  const [deviationStats, setDeviationStats] = useState<DeviationStats | undefined>(undefined)

  const supportedScanFormats = ['.stl', '.ply']
  const cadFormats = ['.step', '.stp', '.iges', '.igs']
  const scanPath = job.scan_path || ''
  const refPath = job.reference_path || ''

  const scanSupported = supportedScanFormats.some(ext => scanPath.toLowerCase().endsWith(ext))
  const refIsCad = cadFormats.some(ext => refPath.toLowerCase().endsWith(ext))
  const refSupported = supportedScanFormats.some(ext => refPath.toLowerCase().endsWith(ext)) || refIsCad

  // Extract bend annotations from dimension analysis
  const bendAnnotations: BendAnnotation[] = useMemo(() => {
    if (!qcResult?.dimension_analysis?.bend_analysis?.matches) return []

    return qcResult.dimension_analysis.bend_analysis.matches
      .filter((match: any) => match.cad?.position)
      .map((match: any) => ({
        id: match.dim_id,
        position: match.cad.position as [number, number, number],
        expectedAngle: match.expected,
        measuredAngle: match.scan?.angle ?? null,
        deviation: match.scan?.deviation ?? null,
        status: match.status as 'pass' | 'fail' | 'warning' | 'pending',
      }))
  }, [qcResult])

  // Fetch deviations from API for heatmap
  useEffect(() => {
    if (job.status !== 'completed' || !job.job_id) return

    const fetchDeviations = async () => {
      setDeviationsLoading(true)
      try {
        const response = await fetch(`/api/deviations/${job.job_id}`)
        if (response.ok) {
          const data = await response.json()
          setDeviations(data.deviations)

          // Compute stats from deviations
          if (data.deviations && data.deviations.length > 0) {
            const devs = data.deviations as number[]
            const min = Math.min(...devs)
            const max = Math.max(...devs)
            const mean = devs.reduce((a: number, b: number) => a + b, 0) / devs.length
            const minIdx = devs.indexOf(min)
            const maxIdx = devs.indexOf(max)
            setDeviationStats({ min, max, mean, minIdx, maxIdx })
          }
        }
      } catch (error) {
        console.log('Deviations not available:', error)
      } finally {
        setDeviationsLoading(false)
      }
    }

    fetchDeviations()
  }, [job.job_id, job.status])

  if (!scanSupported) {
    return null
  }

  // Extract job_id and filename from full path
  const extractRelativePath = (fullPath: string): string | null => {
    if (!fullPath) return null
    const parts = fullPath.split('/')
    const uploadsIndex = parts.findIndex(p => p === 'uploads')
    if (uploadsIndex >= 0 && parts.length > uploadsIndex + 2) {
      return `${parts[uploadsIndex + 1]}/${parts.slice(uploadsIndex + 2).join('/')}`
    }
    const filename = parts[parts.length - 1]
    if (job.job_id && filename) {
      return `${job.job_id}/${filename}`
    }
    return null
  }

  const scanRelPath = extractRelativePath(scanPath)

  // Build URLs for the 3D models
  // For completed jobs, use ALIGNED scan (after ICP alignment) so it overlays with CAD
  // For in-progress jobs, fall back to original uploaded file
  let scanUrl: string | undefined
  if (job.status === 'completed' && job.job_id) {
    // Use aligned scan from output directory (properly aligned to CAD)
    scanUrl = `/api/aligned-scan/${job.job_id}.ply`
  } else if (scanSupported && scanRelPath) {
    // Fall back to original uploaded file
    scanUrl = `/api/files/${scanRelPath}`
  }

  // For CAD files, use the converted mesh from output directory
  // For native mesh files (STL/PLY), use the uploaded file
  let refUrl: string | undefined
  if (refIsCad && job.job_id) {
    // Use converted reference mesh from output directory
    // Include .ply extension so ThreeViewer recognizes the format
    refUrl = `/api/reference-mesh/${job.job_id}.ply`
  } else if (refSupported) {
    const refRelPath = extractRelativePath(refPath)
    refUrl = refRelPath ? `/api/files/${refRelPath}` : undefined
  }

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-dark-100">3D Model Viewer</h3>
        <div className="flex items-center gap-2">
          {deviationsLoading && (
            <span className="text-xs text-dark-400">Loading heatmap...</span>
          )}
          <Box className="w-5 h-5 text-dark-400" />
        </div>
      </div>
      <Suspense fallback={
        <div className="h-[400px] bg-dark-900 rounded-lg flex items-center justify-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
        </div>
      }>
        <ThreeViewer
          modelUrl={scanUrl}
          referenceUrl={refUrl}
          deviations={deviations}
          tolerance={tolerance}
          className="h-[400px]"
          bendAnnotations={bendAnnotations}
          deviationStats={deviationStats}
        />
      </Suspense>
    </div>
  )
}

// Utility functions
function formatNumber(value: number | null | undefined, suffix: string): string {
  if (value == null || isNaN(value)) return '-'
  return `${value.toFixed(4)} ${suffix}`
}

function formatDate(dateStr: string | null | undefined): string {
  if (!dateStr) return '-'
  try {
    return new Date(dateStr).toLocaleString()
  } catch {
    return '-'
  }
}

function formatDuration(startStr: string | null | undefined, endStr: string | null | undefined): string {
  if (!startStr || !endStr) return '-'
  try {
    const start = new Date(startStr).getTime()
    const end = new Date(endStr).getTime()
    if (isNaN(start) || isNaN(end)) return '-'
    const seconds = Math.round((end - start) / 1000)
    return `${seconds}s`
  } catch {
    return '-'
  }
}

function buildOperatorBrief(qcResult: any): { headline: string; actions: string[] } {
  if (!qcResult) {
    return {
      headline: 'Analysis complete.',
      actions: [],
    }
  }

  const overall = qcResult.overall_result || 'REVIEW'
  const releaseDecision = qcResult.release_decision || (overall === 'PASS' ? 'AUTO_PASS' : overall === 'FAIL' ? 'AUTO_FAIL' : 'HOLD')
  const stats = qcResult.statistics || {}
  const passRate = typeof stats.pass_rate === 'number'
    ? stats.pass_rate
    : stats.conformance_percentage
  const maxDeviation = typeof stats.max_deviation_mm === 'number'
    ? stats.max_deviation_mm
    : stats.max_deviation
  const headlineParts = [
    `Result: ${overall}`,
    `Release: ${releaseDecision}`,
    passRate != null ? `Pass rate ${Number(passRate).toFixed(1)}%` : null,
    maxDeviation != null ? `Max deviation ${Number(maxDeviation).toFixed(3)} mm` : null,
  ].filter(Boolean)

  const actions: string[] = []
  const recs = Array.isArray(qcResult.recommendations) ? qcResult.recommendations : []
  for (const rec of recs) {
    if (typeof rec === 'string' && rec.trim()) {
      actions.push(rec.trim())
    } else if (rec && typeof rec === 'object' && typeof rec.action === 'string' && rec.action.trim()) {
      actions.push(rec.action.trim())
    }
    if (actions.length >= 4) break
  }

  if (!actions.length) {
    const rootCauses = Array.isArray(qcResult.root_causes) ? qcResult.root_causes : []
    for (const rc of rootCauses) {
      if (rc && typeof rc === 'object' && typeof rc.description === 'string' && rc.description.trim()) {
        actions.push(`Investigate: ${rc.description.trim()}`)
      }
      if (actions.length >= 3) break
    }
  }

  return {
    headline: headlineParts.join(' | ') || 'Analysis complete.',
    actions,
  }
}
