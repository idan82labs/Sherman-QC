import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { bendInspectionApi } from '../services/api'
import BenderView from '../components/BenderView'
import { ArrowLeft } from 'lucide-react'

export default function BenderViewPage() {
  const { jobId } = useParams<{ jobId: string }>()

  const { data: job, isLoading, error } = useQuery({
    queryKey: ['bend-inspection-job', jobId],
    queryFn: () => bendInspectionApi.get(jobId!),
    enabled: !!jobId,
    refetchInterval: false,
  })

  if (isLoading) {
    return (
      <div className="min-h-screen bg-dark-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary-500" />
      </div>
    )
  }

  if (error || !job) {
    return (
      <div className="min-h-screen bg-dark-900 flex items-center justify-center text-dark-400">
        Job not found or failed to load.
      </div>
    )
  }

  const matches = job.report?.matches ?? []

  return (
    <div className="min-h-screen bg-dark-900 p-6">
      {/* Header */}
      <div className="max-w-[1400px] mx-auto mb-4 flex items-center gap-4 print:hidden">
        <Link
          to="/bend-inspection"
          className="flex items-center gap-1.5 text-dark-400 hover:text-dark-200 transition-colors text-sm"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Inspection
        </Link>
        <div className="h-4 w-px bg-dark-700" />
        <div>
          <span className="text-dark-200 font-semibold">{job.part_id}</span>
          {job.part_name && (
            <span className="text-dark-500 ml-2 text-sm">{job.part_name}</span>
          )}
        </div>
      </div>

      {/* Bender View */}
      <div className="max-w-[1400px] mx-auto">
        <BenderView
          artifacts={job.artifacts}
          matches={matches}
        />
      </div>
    </div>
  )
}
