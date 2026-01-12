import { useState, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Upload,
  Layers,
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  Trash2,
  ChevronDown,
  ChevronUp,
  FileText,
  X,
  Play,
} from 'lucide-react'
import { batchApi } from '../services/api'
import clsx from 'clsx'

interface UploadFile {
  file: File
  partId: string
  partName: string
}

export default function Batch() {
  const [isUploadMode, setIsUploadMode] = useState(false)
  const [files, setFiles] = useState<UploadFile[]>([])
  const [batchName, setBatchName] = useState('')
  const [material, setMaterial] = useState('aluminum')
  const [tolerance, setTolerance] = useState('0.1')
  const [expandedBatch, setExpandedBatch] = useState<string | null>(null)
  const [error, setError] = useState('')

  const queryClient = useQueryClient()

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['batches'],
    queryFn: () => batchApi.list(),
    refetchInterval: 5000,
  })

  const uploadMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      return batchApi.startAnalysis(formData)
    },
    onSuccess: () => {
      setIsUploadMode(false)
      setFiles([])
      setBatchName('')
      queryClient.invalidateQueries({ queryKey: ['batches'] })
    },
    onError: (err: any) => {
      setError(err.response?.data?.detail || 'Failed to start batch analysis')
    },
  })

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const droppedFiles = Array.from(e.dataTransfer.files)
    const newFiles: UploadFile[] = droppedFiles.map((file, index) => ({
      file,
      partId: `PART-${String(files.length + index + 1).padStart(3, '0')}`,
      partName: file.name.replace(/\.[^/.]+$/, ''),
    }))
    setFiles((prev) => [...prev, ...newFiles])
  }, [files.length])

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFiles = Array.from(e.target.files || [])
      const newFiles: UploadFile[] = selectedFiles.map((file, index) => ({
        file,
        partId: `PART-${String(files.length + index + 1).padStart(3, '0')}`,
        partName: file.name.replace(/\.[^/.]+$/, ''),
      }))
      setFiles((prev) => [...prev, ...newFiles])
    },
    [files.length]
  )

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const updateFile = (index: number, updates: Partial<UploadFile>) => {
    setFiles((prev) =>
      prev.map((f, i) => (i === index ? { ...f, ...updates } : f))
    )
  }

  const handleSubmit = async () => {
    setError('')

    if (files.length === 0) {
      setError('Please upload at least one file')
      return
    }

    if (!batchName.trim()) {
      setError('Please enter a batch name')
      return
    }

    const formData = new FormData()
    files.forEach((f) => {
      formData.append('scans', f.file)
      formData.append(`part_ids`, f.partId)
      formData.append(`part_names`, f.partName)
    })
    formData.append('name', batchName.trim())
    formData.append('material', material)
    formData.append('tolerance', tolerance)

    uploadMutation.mutate(formData)
  }

  const handleDelete = async (batchId: string) => {
    if (confirm('Are you sure you want to delete this batch?')) {
      await batchApi.delete(batchId)
      refetch()
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-100">Batch Processing</h1>
          <p className="text-dark-400 mt-1">Analyze multiple parts at once</p>
        </div>
        {!isUploadMode && (
          <button
            onClick={() => setIsUploadMode(true)}
            className="btn btn-primary flex items-center"
          >
            <Layers className="w-5 h-5 mr-2" />
            New Batch
          </button>
        )}
      </div>

      {/* Upload mode */}
      {isUploadMode && (
        <div className="card p-6 space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-dark-100">Create New Batch</h3>
            <button
              onClick={() => {
                setIsUploadMode(false)
                setFiles([])
                setError('')
              }}
              className="p-2 hover:bg-dark-700 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-dark-400" />
            </button>
          </div>

          {error && (
            <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center text-red-400">
              <AlertTriangle className="w-5 h-5 mr-3 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Batch details */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="label">Batch Name *</label>
              <input
                type="text"
                value={batchName}
                onChange={(e) => setBatchName(e.target.value)}
                className="input"
                placeholder="e.g., Production Run 001"
              />
            </div>
            <div>
              <label className="label">Material</label>
              <select
                value={material}
                onChange={(e) => setMaterial(e.target.value)}
                className="input"
              >
                <option value="aluminum">Aluminum</option>
                <option value="steel">Steel</option>
                <option value="stainless_steel">Stainless Steel</option>
                <option value="titanium">Titanium</option>
                <option value="brass">Brass</option>
                <option value="copper">Copper</option>
                <option value="plastic">Plastic</option>
              </select>
            </div>
            <div>
              <label className="label">Tolerance (mm)</label>
              <select
                value={tolerance}
                onChange={(e) => setTolerance(e.target.value)}
                className="input"
              >
                <option value="0.01">+/- 0.01 mm</option>
                <option value="0.05">+/- 0.05 mm</option>
                <option value="0.1">+/- 0.1 mm</option>
                <option value="0.25">+/- 0.25 mm</option>
                <option value="0.5">+/- 0.5 mm</option>
              </select>
            </div>
          </div>

          {/* File drop zone */}
          <div
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            className="border-2 border-dashed border-dark-600 rounded-lg p-8 text-center hover:border-primary-500/50 transition-colors"
          >
            <input
              type="file"
              id="batch-files"
              accept=".pdf,.png,.jpg,.jpeg"
              multiple
              onChange={handleFileSelect}
              className="hidden"
            />
            <label htmlFor="batch-files" className="cursor-pointer">
              <Upload className="w-12 h-12 text-dark-500 mx-auto mb-4" />
              <p className="text-dark-300 mb-2">
                Drag & drop files or <span className="text-primary-400">browse</span>
              </p>
              <p className="text-sm text-dark-500">
                Upload multiple scan files (PDF, PNG, JPG)
              </p>
            </label>
          </div>

          {/* File list */}
          {files.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-sm text-dark-400">{files.length} files selected</p>
                <button
                  onClick={() => setFiles([])}
                  className="text-sm text-red-400 hover:text-red-300"
                >
                  Clear all
                </button>
              </div>
              <div className="max-h-64 overflow-y-auto space-y-2">
                {files.map((f, index) => (
                  <div
                    key={index}
                    className="flex items-center gap-4 p-3 bg-dark-700/50 rounded-lg"
                  >
                    <FileText className="w-5 h-5 text-dark-400 flex-shrink-0" />
                    <div className="flex-1 grid grid-cols-2 gap-2">
                      <input
                        type="text"
                        value={f.partId}
                        onChange={(e) => updateFile(index, { partId: e.target.value })}
                        className="input text-sm py-1"
                        placeholder="Part ID"
                      />
                      <input
                        type="text"
                        value={f.partName}
                        onChange={(e) => updateFile(index, { partName: e.target.value })}
                        className="input text-sm py-1"
                        placeholder="Part Name"
                      />
                    </div>
                    <span className="text-xs text-dark-500 w-16 text-right">
                      {(f.file.size / 1024 / 1024).toFixed(1)} MB
                    </span>
                    <button
                      onClick={() => removeFile(index)}
                      className="p-1 hover:bg-dark-600 rounded transition-colors"
                    >
                      <X className="w-4 h-4 text-dark-400" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Submit */}
          <div className="flex justify-end gap-4">
            <button
              onClick={() => {
                setIsUploadMode(false)
                setFiles([])
              }}
              className="btn btn-secondary"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              disabled={uploadMutation.isPending || files.length === 0}
              className={clsx(
                'btn btn-primary flex items-center',
                (uploadMutation.isPending || files.length === 0) &&
                  'opacity-50 cursor-not-allowed'
              )}
            >
              {uploadMutation.isPending ? (
                <>
                  <svg
                    className="animate-spin -ml-1 mr-2 h-5 w-5"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  Starting...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5 mr-2" />
                  Start Batch Analysis
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Batch list */}
      <div className="card">
        <div className="p-4 border-b border-dark-700">
          <h3 className="font-semibold text-dark-100">Recent Batches</h3>
        </div>

        {isLoading ? (
          <div className="p-8">
            <div className="animate-pulse space-y-4">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="h-20 bg-dark-700 rounded-lg" />
              ))}
            </div>
          </div>
        ) : data?.batches?.length === 0 ? (
          <div className="p-8 text-center text-dark-400">
            <Layers className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No batches yet</p>
            <button
              onClick={() => setIsUploadMode(true)}
              className="text-primary-400 hover:text-primary-300 mt-2"
            >
              Create your first batch
            </button>
          </div>
        ) : (
          <div className="divide-y divide-dark-700">
            {data?.batches?.map((batch) => (
              <BatchItem
                key={batch.batch_id}
                batch={batch}
                isExpanded={expandedBatch === batch.batch_id}
                onToggle={() =>
                  setExpandedBatch(
                    expandedBatch === batch.batch_id ? null : batch.batch_id
                  )
                }
                onDelete={() => handleDelete(batch.batch_id)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function BatchItem({
  batch,
  isExpanded,
  onToggle,
  onDelete,
}: {
  batch: any
  isExpanded: boolean
  onToggle: () => void
  onDelete: () => void
}) {
  const statusMap: Record<string, { icon: typeof CheckCircle; className: string }> = {
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
  }
  const statusConfig = statusMap[batch.status] || {
    icon: AlertTriangle,
    className: 'bg-dark-600 text-dark-400',
  }

  const StatusIcon = statusConfig.icon

  return (
    <div>
      <div
        className="p-4 hover:bg-dark-700/30 cursor-pointer transition-colors"
        onClick={onToggle}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <span
              className={clsx(
                'inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium mr-4',
                statusConfig.className
              )}
            >
              <StatusIcon className="w-3 h-3 mr-1" />
              {batch.status}
            </span>
            <div>
              <p className="font-medium text-dark-100">{batch.name}</p>
              <p className="text-sm text-dark-400">
                {batch.total_parts} parts | {batch.material}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {batch.status === 'running' && (
              <div className="w-32">
                <div className="flex justify-between text-xs text-dark-400 mb-1">
                  <span>Progress</span>
                  <span>{Math.round(batch.progress || 0)}%</span>
                </div>
                <div className="h-1.5 bg-dark-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary-500 transition-all duration-300"
                    style={{ width: `${batch.progress || 0}%` }}
                  />
                </div>
              </div>
            )}
            <div className="text-right text-sm text-dark-400">
              {new Date(batch.created_at).toLocaleDateString()}
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation()
                onDelete()
              }}
              className="p-2 text-dark-400 hover:text-red-400 hover:bg-dark-700 rounded-lg transition-colors"
            >
              <Trash2 className="w-4 h-4" />
            </button>
            {isExpanded ? (
              <ChevronUp className="w-5 h-5 text-dark-400" />
            ) : (
              <ChevronDown className="w-5 h-5 text-dark-400" />
            )}
          </div>
        </div>
      </div>

      {/* Expanded details */}
      {isExpanded && (
        <div className="px-4 pb-4 bg-dark-700/20">
          <div className="grid grid-cols-4 gap-4 text-sm mb-4">
            <div>
              <p className="text-dark-500">Completed</p>
              <p className="text-dark-200">{batch.completed_parts || 0}</p>
            </div>
            <div>
              <p className="text-dark-500">Failed</p>
              <p className="text-dark-200">{batch.failed_parts || 0}</p>
            </div>
            <div>
              <p className="text-dark-500">Tolerance</p>
              <p className="text-dark-200">+/- {batch.tolerance} mm</p>
            </div>
            <div>
              <p className="text-dark-500">Avg Score</p>
              <p className="text-dark-200">
                {batch.avg_quality_score
                  ? `${batch.avg_quality_score.toFixed(1)}%`
                  : '-'}
              </p>
            </div>
          </div>

          {/* Parts list */}
          {batch.parts && Object.keys(batch.parts).length > 0 && (
            <div className="space-y-2">
              {Object.values(batch.parts).map((part: any) => (
                <Link
                  key={part.job_id}
                  to={`/jobs/${part.job_id}`}
                  className="flex items-center justify-between p-3 bg-dark-700/50 rounded-lg hover:bg-dark-700 transition-colors"
                >
                  <div className="flex items-center">
                    <PartStatusIcon status={part.status} />
                    <div className="ml-3">
                      <p className="font-medium text-dark-100">{part.part_id}</p>
                      <p className="text-xs text-dark-400">{part.part_name}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    {part.quality_score !== null && (
                      <span
                        className={clsx(
                          'font-medium',
                          part.quality_score >= 90
                            ? 'text-emerald-400'
                            : part.quality_score >= 70
                            ? 'text-amber-400'
                            : 'text-red-400'
                        )}
                      >
                        {part.quality_score.toFixed(1)}%
                      </span>
                    )}
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function PartStatusIcon({ status }: { status: string }) {
  const statusMap: Record<string, { icon: typeof CheckCircle; className: string }> = {
    completed: { icon: CheckCircle, className: 'text-emerald-400' },
    failed: { icon: XCircle, className: 'text-red-400' },
    running: { icon: Clock, className: 'text-amber-400 animate-pulse' },
    pending: { icon: Clock, className: 'text-dark-400' },
  }
  const config = statusMap[status] || { icon: AlertTriangle, className: 'text-dark-400' }

  const Icon = config.icon
  return <Icon className={clsx('w-5 h-5', config.className)} />
}
