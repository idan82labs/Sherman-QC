import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  Activity,
  CheckCircle,
  XCircle,
  Clock,
  HardDrive,
  TrendingUp,
  AlertTriangle,
  Plus,
  type LucideIcon,
} from 'lucide-react'
import { statsApi, jobsApi } from '../services/api'
import clsx from 'clsx'

export default function Dashboard() {
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: statsApi.get,
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const { data: recentJobs, isLoading: jobsLoading } = useQuery({
    queryKey: ['jobs', 'recent'],
    queryFn: () => jobsApi.list({ limit: 5 }),
    refetchInterval: 10000, // Refresh every 10 seconds
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-100">Dashboard</h1>
          <p className="text-dark-400 mt-1">Quality control overview</p>
        </div>
        <Link to="/upload" className="btn btn-primary flex items-center">
          <Plus className="w-5 h-5 mr-2" />
          New Analysis
        </Link>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Jobs"
          value={stats?.total_jobs || 0}
          icon={Activity}
          color="blue"
          isLoading={statsLoading}
        />
        <StatCard
          title="Completed"
          value={stats?.completed || 0}
          icon={CheckCircle}
          color="green"
          isLoading={statsLoading}
        />
        <StatCard
          title="Failed"
          value={stats?.failed || 0}
          icon={XCircle}
          color="red"
          isLoading={statsLoading}
        />
        <StatCard
          title="Running"
          value={stats?.running || 0}
          icon={Clock}
          color="yellow"
          isLoading={statsLoading}
        />
      </div>

      {/* Storage and quick actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Storage usage */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-dark-100">Storage Usage</h3>
            <HardDrive className="w-5 h-5 text-dark-400" />
          </div>
          {statsLoading ? (
            <div className="animate-pulse space-y-3">
              <div className="h-4 bg-dark-700 rounded w-1/2" />
              <div className="h-2 bg-dark-700 rounded" />
            </div>
          ) : (
            <>
              <p className="text-3xl font-bold text-dark-100">
                {stats?.storage?.total_mb.toFixed(1)} MB
              </p>
              <div className="mt-4 space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-dark-400">Uploads</span>
                  <span className="text-dark-200">
                    {stats?.storage?.uploads_mb.toFixed(1)} MB
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-dark-400">Output</span>
                  <span className="text-dark-200">
                    {stats?.storage?.output_mb.toFixed(1)} MB
                  </span>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Quick actions */}
        <div className="card p-6">
          <h3 className="font-semibold text-dark-100 mb-4">Quick Actions</h3>
          <div className="space-y-3">
            <Link
              to="/upload"
              className="flex items-center p-3 bg-dark-700/50 rounded-lg hover:bg-dark-700 transition-colors"
            >
              <div className="w-10 h-10 bg-primary-500/20 rounded-lg flex items-center justify-center mr-3">
                <Plus className="w-5 h-5 text-primary-400" />
              </div>
              <div>
                <p className="font-medium text-dark-100">New Analysis</p>
                <p className="text-sm text-dark-400">Upload files for QC</p>
              </div>
            </Link>
            <Link
              to="/batch"
              className="flex items-center p-3 bg-dark-700/50 rounded-lg hover:bg-dark-700 transition-colors"
            >
              <div className="w-10 h-10 bg-secondary-500/20 rounded-lg flex items-center justify-center mr-3">
                <TrendingUp className="w-5 h-5 text-secondary-400" />
              </div>
              <div>
                <p className="font-medium text-dark-100">Batch Processing</p>
                <p className="text-sm text-dark-400">Analyze multiple parts</p>
              </div>
            </Link>
          </div>
        </div>

        {/* System status */}
        <div className="card p-6">
          <h3 className="font-semibold text-dark-100 mb-4">System Status</h3>
          <div className="space-y-3">
            <StatusItem label="API Server" status="online" />
            <StatusItem label="Database" status="online" />
            <StatusItem
              label="AI Engine"
              status={stats?.running ? 'busy' : 'online'}
            />
          </div>
        </div>
      </div>

      {/* Recent jobs */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-dark-100">Recent Jobs</h3>
          <Link
            to="/jobs"
            className="text-sm text-primary-400 hover:text-primary-300"
          >
            View all
          </Link>
        </div>

        {jobsLoading ? (
          <div className="animate-pulse space-y-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-16 bg-dark-700 rounded-lg" />
            ))}
          </div>
        ) : recentJobs?.jobs?.length === 0 ? (
          <div className="text-center py-8 text-dark-400">
            <Activity className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No jobs yet</p>
            <Link
              to="/upload"
              className="text-primary-400 hover:text-primary-300 mt-2 inline-block"
            >
              Start your first analysis
            </Link>
          </div>
        ) : (
          <div className="space-y-3">
            {recentJobs?.jobs?.map((job) => (
              <Link
                key={job.job_id}
                to={`/jobs/${job.job_id}`}
                className="flex items-center justify-between p-4 bg-dark-700/50 rounded-lg hover:bg-dark-700 transition-colors"
              >
                <div className="flex items-center">
                  <JobStatusIcon status={job.status} />
                  <div className="ml-3">
                    <p className="font-medium text-dark-100">{job.part_id}</p>
                    <p className="text-sm text-dark-400">{job.part_name}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm text-dark-200">{job.material}</p>
                  <p className="text-xs text-dark-400">
                    {new Date(job.created_at).toLocaleDateString()}
                  </p>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// Helper components
function StatCard({
  title,
  value,
  icon: Icon,
  color,
  isLoading,
}: {
  title: string
  value: number
  icon: LucideIcon
  color: 'blue' | 'green' | 'red' | 'yellow'
  isLoading: boolean
}) {
  const colorClasses = {
    blue: 'bg-blue-500/20 text-blue-400',
    green: 'bg-emerald-500/20 text-emerald-400',
    red: 'bg-red-500/20 text-red-400',
    yellow: 'bg-amber-500/20 text-amber-400',
  }

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-dark-400">{title}</p>
          {isLoading ? (
            <div className="h-8 w-16 bg-dark-700 rounded animate-pulse mt-1" />
          ) : (
            <p className="text-3xl font-bold text-dark-100 mt-1">{value}</p>
          )}
        </div>
        <div className={clsx('w-12 h-12 rounded-xl flex items-center justify-center', colorClasses[color])}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </div>
  )
}

function StatusItem({
  label,
  status,
}: {
  label: string
  status: 'online' | 'offline' | 'busy'
}) {
  const statusClasses = {
    online: 'bg-emerald-500',
    offline: 'bg-red-500',
    busy: 'bg-amber-500',
  }

  return (
    <div className="flex items-center justify-between">
      <span className="text-dark-300">{label}</span>
      <div className="flex items-center">
        <div className={clsx('w-2 h-2 rounded-full mr-2', statusClasses[status])} />
        <span className="text-sm text-dark-400 capitalize">{status}</span>
      </div>
    </div>
  )
}

function JobStatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'completed':
      return (
        <div className="w-10 h-10 bg-emerald-500/20 rounded-lg flex items-center justify-center">
          <CheckCircle className="w-5 h-5 text-emerald-400" />
        </div>
      )
    case 'failed':
      return (
        <div className="w-10 h-10 bg-red-500/20 rounded-lg flex items-center justify-center">
          <XCircle className="w-5 h-5 text-red-400" />
        </div>
      )
    case 'running':
      return (
        <div className="w-10 h-10 bg-amber-500/20 rounded-lg flex items-center justify-center">
          <Clock className="w-5 h-5 text-amber-400 animate-pulse" />
        </div>
      )
    default:
      return (
        <div className="w-10 h-10 bg-dark-600 rounded-lg flex items-center justify-center">
          <AlertTriangle className="w-5 h-5 text-dark-400" />
        </div>
      )
  }
}
