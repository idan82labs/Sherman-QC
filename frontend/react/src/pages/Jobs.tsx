import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Search,
  Filter,
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Trash2,
} from 'lucide-react'
import { jobsApi } from '../services/api'
import clsx from 'clsx'

const ITEMS_PER_PAGE = 10

export default function Jobs() {
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('')
  const [page, setPage] = useState(0)

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['jobs', statusFilter, page],
    queryFn: () =>
      jobsApi.list({
        status: statusFilter || undefined,
        limit: ITEMS_PER_PAGE,
        offset: page * ITEMS_PER_PAGE,
      }),
    refetchInterval: 10000,
  })

  const filteredJobs = data?.jobs?.filter(
    (job) =>
      job.part_id.toLowerCase().includes(search.toLowerCase()) ||
      job.part_name.toLowerCase().includes(search.toLowerCase())
  )

  const totalPages = Math.ceil((data?.count || 0) / ITEMS_PER_PAGE)

  const handleDelete = async (jobId: string, e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (confirm('Are you sure you want to delete this job?')) {
      await jobsApi.delete(jobId)
      refetch()
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-100">Jobs</h1>
          <p className="text-dark-400 mt-1">View and manage analysis jobs</p>
        </div>
        <Link to="/upload" className="btn btn-primary">
          New Analysis
        </Link>
      </div>

      {/* Filters */}
      <div className="card p-4">
        <div className="flex flex-col sm:flex-row gap-4">
          {/* Search */}
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-500" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by part ID or name..."
              className="input pl-10 w-full"
            />
          </div>

          {/* Status filter */}
          <div className="flex items-center gap-2">
            <Filter className="w-5 h-5 text-dark-500" />
            <select
              value={statusFilter}
              onChange={(e) => {
                setStatusFilter(e.target.value)
                setPage(0)
              }}
              className="input"
            >
              <option value="">All Status</option>
              <option value="pending">Pending</option>
              <option value="running">Running</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
            </select>
          </div>
        </div>
      </div>

      {/* Jobs table */}
      <div className="card overflow-hidden">
        {isLoading ? (
          <div className="p-8">
            <div className="animate-pulse space-y-4">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-16 bg-dark-700 rounded-lg" />
              ))}
            </div>
          </div>
        ) : filteredJobs?.length === 0 ? (
          <div className="p-8 text-center text-dark-400">
            <AlertTriangle className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No jobs found</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-dark-700/50">
                <tr>
                  <th className="text-left px-6 py-3 text-xs font-medium text-dark-400 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="text-left px-6 py-3 text-xs font-medium text-dark-400 uppercase tracking-wider">
                    Part ID
                  </th>
                  <th className="text-left px-6 py-3 text-xs font-medium text-dark-400 uppercase tracking-wider">
                    Part Name
                  </th>
                  <th className="text-left px-6 py-3 text-xs font-medium text-dark-400 uppercase tracking-wider">
                    Material
                  </th>
                  <th className="text-left px-6 py-3 text-xs font-medium text-dark-400 uppercase tracking-wider">
                    Quality Score
                  </th>
                  <th className="text-left px-6 py-3 text-xs font-medium text-dark-400 uppercase tracking-wider">
                    Created
                  </th>
                  <th className="text-right px-6 py-3 text-xs font-medium text-dark-400 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-700">
                {filteredJobs?.map((job) => (
                  <tr
                    key={job.job_id}
                    className="hover:bg-dark-700/30 transition-colors cursor-pointer"
                  >
                    <td className="px-6 py-4">
                      <Link to={`/jobs/${job.job_id}`} className="block">
                        <StatusBadge status={job.status} />
                      </Link>
                    </td>
                    <td className="px-6 py-4">
                      <Link
                        to={`/jobs/${job.job_id}`}
                        className="font-medium text-dark-100 hover:text-primary-400"
                      >
                        {job.part_id}
                      </Link>
                    </td>
                    <td className="px-6 py-4 text-dark-300">
                      <Link to={`/jobs/${job.job_id}`} className="block">
                        {job.part_name}
                      </Link>
                    </td>
                    <td className="px-6 py-4 text-dark-300 capitalize">
                      <Link to={`/jobs/${job.job_id}`} className="block">
                        {job.material.replace('_', ' ')}
                      </Link>
                    </td>
                    <td className="px-6 py-4">
                      <Link to={`/jobs/${job.job_id}`} className="block">
                        {job.quality_score !== null ? (
                          <QualityBadge score={job.quality_score} />
                        ) : (
                          <span className="text-dark-500">-</span>
                        )}
                      </Link>
                    </td>
                    <td className="px-6 py-4 text-dark-400 text-sm">
                      <Link to={`/jobs/${job.job_id}`} className="block">
                        {new Date(job.created_at).toLocaleString()}
                      </Link>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <button
                        onClick={(e) => handleDelete(job.job_id, e)}
                        className="p-2 text-dark-400 hover:text-red-400 hover:bg-dark-700 rounded-lg transition-colors"
                        title="Delete job"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-6 py-4 border-t border-dark-700">
            <p className="text-sm text-dark-400">
              Showing {page * ITEMS_PER_PAGE + 1} to{' '}
              {Math.min((page + 1) * ITEMS_PER_PAGE, data?.count || 0)} of{' '}
              {data?.count} jobs
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="btn btn-secondary p-2 disabled:opacity-50"
              >
                <ChevronLeft className="w-5 h-5" />
              </button>
              <button
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="btn btn-secondary p-2 disabled:opacity-50"
              >
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const config = {
    completed: {
      icon: CheckCircle,
      className: 'bg-emerald-500/20 text-emerald-400',
    },
    failed: {
      icon: XCircle,
      className: 'bg-red-500/20 text-red-400',
    },
    running: {
      icon: Clock,
      className: 'bg-amber-500/20 text-amber-400',
    },
    pending: {
      icon: Clock,
      className: 'bg-dark-600 text-dark-400',
    },
  }[status] || {
    icon: AlertTriangle,
    className: 'bg-dark-600 text-dark-400',
  }

  const Icon = config.icon

  return (
    <span
      className={clsx(
        'inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium',
        config.className
      )}
    >
      <Icon className="w-3 h-3 mr-1" />
      {status}
    </span>
  )
}

function QualityBadge({ score }: { score: number }) {
  const getColor = () => {
    if (score >= 90) return 'text-emerald-400'
    if (score >= 70) return 'text-amber-400'
    return 'text-red-400'
  }

  return (
    <span className={clsx('font-medium', getColor())}>
      {score.toFixed(1)}%
    </span>
  )
}
