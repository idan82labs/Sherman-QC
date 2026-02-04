import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Search,
  Filter,
  Plus,
  Upload,
  FileText,
  CheckCircle,
  AlertCircle,
  Clock,
  ChevronLeft,
  ChevronRight,
  Trash2,
  Edit2,
  X,
  FileUp,
  Database,
} from 'lucide-react'
import { partCatalogApi, getErrorMessage } from '../services/api'
import type { Part, PartCreateRequest, PartUpdateRequest } from '../types'
import clsx from 'clsx'

const ITEMS_PER_PAGE = 15

type PartStatus = 'all' | 'ready' | 'needs_cad' | 'needs_embedding'

export default function PartCatalog() {
  const queryClient = useQueryClient()
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState<PartStatus>('all')
  const [page, setPage] = useState(0)
  const [showAddModal, setShowAddModal] = useState(false)
  const [showEditModal, setShowEditModal] = useState<Part | null>(null)
  const [showUploadModal, setShowUploadModal] = useState<Part | null>(null)
  const [showImportModal, setShowImportModal] = useState(false)

  // Fetch parts
  const { data, isLoading } = useQuery({
    queryKey: ['parts', search, statusFilter, page],
    queryFn: () =>
      partCatalogApi.list({
        search: search || undefined,
        has_cad: statusFilter === 'needs_cad' ? false : statusFilter === 'ready' ? true : undefined,
        has_embedding: statusFilter === 'needs_embedding' ? false : statusFilter === 'ready' ? true : undefined,
        limit: ITEMS_PER_PAGE,
        offset: page * ITEMS_PER_PAGE,
      }),
    refetchInterval: 30000,
  })

  // Fetch stats
  const { data: stats } = useQuery({
    queryKey: ['parts-stats'],
    queryFn: () => partCatalogApi.stats(),
  })

  const totalPages = Math.ceil((data?.total || 0) / ITEMS_PER_PAGE)

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (partId: string) => partCatalogApi.delete(partId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['parts'] })
      queryClient.invalidateQueries({ queryKey: ['parts-stats'] })
    },
  })

  const handleDelete = async (part: Part, e: React.MouseEvent) => {
    e.stopPropagation()
    if (confirm(`Delete part ${part.part_number}? This will also delete CAD files and embeddings.`)) {
      deleteMutation.mutate(part.id)
    }
  }

  const getPartStatus = (part: Part): 'ready' | 'needs_cad' | 'needs_embedding' => {
    if (!part.cad_file_path) return 'needs_cad'
    if (!part.has_embedding) return 'needs_embedding'
    return 'ready'
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-100">Part Catalog</h1>
          <p className="text-dark-400 mt-1">Manage parts, CAD files, and bend specifications</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setShowImportModal(true)}
            className="btn btn-secondary flex items-center"
          >
            <FileUp className="w-4 h-4 mr-2" />
            Import CSV
          </button>
          <button
            onClick={() => setShowAddModal(true)}
            className="btn btn-primary flex items-center"
          >
            <Plus className="w-4 h-4 mr-2" />
            Add Part
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          label="Total Parts"
          value={stats?.total || 0}
          icon={Database}
          color="text-primary-400"
        />
        <StatCard
          label="Ready"
          value={stats?.ready || 0}
          icon={CheckCircle}
          color="text-emerald-400"
        />
        <StatCard
          label="Needs CAD"
          value={stats?.needs_cad || 0}
          icon={AlertCircle}
          color="text-amber-400"
        />
        <StatCard
          label="Needs Embedding"
          value={stats?.needs_embedding || 0}
          icon={Clock}
          color="text-blue-400"
        />
      </div>

      {/* Filters */}
      <div className="card p-4">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-500" />
            <input
              type="text"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value)
                setPage(0)
              }}
              placeholder="Search by part number or name..."
              className="input pl-10 w-full"
            />
          </div>

          <div className="flex items-center gap-2">
            <Filter className="w-5 h-5 text-dark-500" />
            <select
              value={statusFilter}
              onChange={(e) => {
                setStatusFilter(e.target.value as PartStatus)
                setPage(0)
              }}
              className="input"
            >
              <option value="all">All Status</option>
              <option value="ready">Ready</option>
              <option value="needs_cad">Needs CAD</option>
              <option value="needs_embedding">Needs Embedding</option>
            </select>
          </div>
        </div>
      </div>

      {/* Parts Table */}
      <div className="card overflow-hidden">
        {isLoading ? (
          <div className="p-8">
            <div className="animate-pulse space-y-4">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-16 bg-dark-700 rounded-lg" />
              ))}
            </div>
          </div>
        ) : data?.parts?.length === 0 ? (
          <div className="p-8 text-center text-dark-400">
            <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No parts found</p>
            <button
              onClick={() => setShowAddModal(true)}
              className="btn btn-primary mt-4"
            >
              Add First Part
            </button>
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
                    Part Number
                  </th>
                  <th className="text-left px-6 py-3 text-xs font-medium text-dark-400 uppercase tracking-wider">
                    Name
                  </th>
                  <th className="text-left px-6 py-3 text-xs font-medium text-dark-400 uppercase tracking-wider">
                    Customer
                  </th>
                  <th className="text-left px-6 py-3 text-xs font-medium text-dark-400 uppercase tracking-wider">
                    Material
                  </th>
                  <th className="text-left px-6 py-3 text-xs font-medium text-dark-400 uppercase tracking-wider">
                    Bends
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
                {data?.parts?.map((part) => (
                  <tr
                    key={part.id}
                    className="hover:bg-dark-700/30 transition-colors"
                  >
                    <td className="px-6 py-4">
                      <StatusBadge status={getPartStatus(part)} />
                    </td>
                    <td className="px-6 py-4 font-medium text-dark-100">
                      {part.part_number}
                    </td>
                    <td className="px-6 py-4 text-dark-300">
                      {part.part_name}
                    </td>
                    <td className="px-6 py-4 text-dark-300">
                      {part.customer || '-'}
                    </td>
                    <td className="px-6 py-4 text-dark-300">
                      {part.material || '-'}
                    </td>
                    <td className="px-6 py-4 text-dark-300">
                      {part.bend_count ?? '-'}
                    </td>
                    <td className="px-6 py-4 text-dark-400 text-sm">
                      {new Date(part.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center justify-end gap-1">
                        {!part.cad_file_path && (
                          <button
                            onClick={() => setShowUploadModal(part)}
                            className="p-2 text-dark-400 hover:text-primary-400 hover:bg-dark-700 rounded-lg transition-colors"
                            title="Upload CAD"
                          >
                            <Upload className="w-4 h-4" />
                          </button>
                        )}
                        <button
                          onClick={() => setShowEditModal(part)}
                          className="p-2 text-dark-400 hover:text-primary-400 hover:bg-dark-700 rounded-lg transition-colors"
                          title="Edit"
                        >
                          <Edit2 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={(e) => handleDelete(part, e)}
                          className="p-2 text-dark-400 hover:text-red-400 hover:bg-dark-700 rounded-lg transition-colors"
                          title="Delete"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
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
              {Math.min((page + 1) * ITEMS_PER_PAGE, data?.total || 0)} of{' '}
              {data?.total} parts
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

      {/* Modals */}
      {showAddModal && (
        <AddPartModal
          onClose={() => setShowAddModal(false)}
          onSuccess={() => {
            setShowAddModal(false)
            queryClient.invalidateQueries({ queryKey: ['parts'] })
            queryClient.invalidateQueries({ queryKey: ['parts-stats'] })
          }}
        />
      )}

      {showEditModal && (
        <EditPartModal
          part={showEditModal}
          onClose={() => setShowEditModal(null)}
          onSuccess={() => {
            setShowEditModal(null)
            queryClient.invalidateQueries({ queryKey: ['parts'] })
          }}
        />
      )}

      {showUploadModal && (
        <UploadCadModal
          part={showUploadModal}
          onClose={() => setShowUploadModal(null)}
          onSuccess={() => {
            setShowUploadModal(null)
            queryClient.invalidateQueries({ queryKey: ['parts'] })
            queryClient.invalidateQueries({ queryKey: ['parts-stats'] })
          }}
        />
      )}

      {showImportModal && (
        <ImportCsvModal
          onClose={() => setShowImportModal(false)}
          onSuccess={() => {
            setShowImportModal(false)
            queryClient.invalidateQueries({ queryKey: ['parts'] })
            queryClient.invalidateQueries({ queryKey: ['parts-stats'] })
          }}
        />
      )}
    </div>
  )
}

// Status Badge Component
function StatusBadge({ status }: { status: 'ready' | 'needs_cad' | 'needs_embedding' }) {
  const config = {
    ready: {
      icon: CheckCircle,
      label: 'Ready',
      className: 'bg-emerald-500/20 text-emerald-400',
    },
    needs_cad: {
      icon: AlertCircle,
      label: 'Needs CAD',
      className: 'bg-amber-500/20 text-amber-400',
    },
    needs_embedding: {
      icon: Clock,
      label: 'Pending',
      className: 'bg-blue-500/20 text-blue-400',
    },
  }[status]

  const Icon = config.icon

  return (
    <span
      className={clsx(
        'inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium',
        config.className
      )}
    >
      <Icon className="w-3 h-3 mr-1" />
      {config.label}
    </span>
  )
}

// Stat Card Component
function StatCard({
  label,
  value,
  icon: Icon,
  color,
}: {
  label: string
  value: number
  icon: React.ElementType
  color: string
}) {
  return (
    <div className="card p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-dark-400">{label}</p>
          <p className="text-2xl font-bold text-dark-100">{value}</p>
        </div>
        <Icon className={clsx('w-8 h-8', color)} />
      </div>
    </div>
  )
}

// Add Part Modal
function AddPartModal({
  onClose,
  onSuccess,
}: {
  onClose: () => void
  onSuccess: () => void
}) {
  const [formData, setFormData] = useState<PartCreateRequest>({
    part_number: '',
    part_name: '',
    customer: '',
    material: '',
    default_tolerance_angle: 1.0,
    default_tolerance_linear: 0.5,
    notes: '',
  })
  const [error, setError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setIsSubmitting(true)

    try {
      await partCatalogApi.create(formData)
      onSuccess()
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <Modal title="Add New Part" onClose={onClose}>
      <form onSubmit={handleSubmit} className="space-y-4">
        {error && (
          <div className="p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 text-sm">
            {error}
          </div>
        )}

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-dark-300 mb-1">
              Part Number *
            </label>
            <input
              type="text"
              value={formData.part_number}
              onChange={(e) => setFormData({ ...formData, part_number: e.target.value })}
              className="input w-full"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-dark-300 mb-1">
              Part Name *
            </label>
            <input
              type="text"
              value={formData.part_name}
              onChange={(e) => setFormData({ ...formData, part_name: e.target.value })}
              className="input w-full"
              required
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-dark-300 mb-1">
              Customer
            </label>
            <input
              type="text"
              value={formData.customer}
              onChange={(e) => setFormData({ ...formData, customer: e.target.value })}
              className="input w-full"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-dark-300 mb-1">
              Material
            </label>
            <input
              type="text"
              value={formData.material}
              onChange={(e) => setFormData({ ...formData, material: e.target.value })}
              className="input w-full"
              placeholder="e.g., Steel 1.5mm"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-dark-300 mb-1">
              Angle Tolerance (deg)
            </label>
            <input
              type="number"
              step="0.1"
              value={formData.default_tolerance_angle}
              onChange={(e) =>
                setFormData({ ...formData, default_tolerance_angle: parseFloat(e.target.value) })
              }
              className="input w-full"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-dark-300 mb-1">
              Linear Tolerance (mm)
            </label>
            <input
              type="number"
              step="0.1"
              value={formData.default_tolerance_linear}
              onChange={(e) =>
                setFormData({ ...formData, default_tolerance_linear: parseFloat(e.target.value) })
              }
              className="input w-full"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-dark-300 mb-1">Notes</label>
          <textarea
            value={formData.notes}
            onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
            className="input w-full h-20 resize-none"
          />
        </div>

        <div className="flex justify-end gap-2 pt-4">
          <button type="button" onClick={onClose} className="btn btn-secondary">
            Cancel
          </button>
          <button type="submit" disabled={isSubmitting} className="btn btn-primary">
            {isSubmitting ? 'Creating...' : 'Create Part'}
          </button>
        </div>
      </form>
    </Modal>
  )
}

// Edit Part Modal
function EditPartModal({
  part,
  onClose,
  onSuccess,
}: {
  part: Part
  onClose: () => void
  onSuccess: () => void
}) {
  const [formData, setFormData] = useState<PartUpdateRequest>({
    part_name: part.part_name,
    customer: part.customer || '',
    material: part.material || '',
    default_tolerance_angle: part.default_tolerance_angle,
    default_tolerance_linear: part.default_tolerance_linear,
    notes: part.notes || '',
  })
  const [error, setError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setIsSubmitting(true)

    try {
      await partCatalogApi.update(part.id, formData)
      onSuccess()
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <Modal title={`Edit ${part.part_number}`} onClose={onClose}>
      <form onSubmit={handleSubmit} className="space-y-4">
        {error && (
          <div className="p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 text-sm">
            {error}
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-dark-300 mb-1">Part Name</label>
          <input
            type="text"
            value={formData.part_name}
            onChange={(e) => setFormData({ ...formData, part_name: e.target.value })}
            className="input w-full"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-dark-300 mb-1">Customer</label>
            <input
              type="text"
              value={formData.customer}
              onChange={(e) => setFormData({ ...formData, customer: e.target.value })}
              className="input w-full"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-dark-300 mb-1">Material</label>
            <input
              type="text"
              value={formData.material}
              onChange={(e) => setFormData({ ...formData, material: e.target.value })}
              className="input w-full"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-dark-300 mb-1">
              Angle Tolerance (deg)
            </label>
            <input
              type="number"
              step="0.1"
              value={formData.default_tolerance_angle}
              onChange={(e) =>
                setFormData({ ...formData, default_tolerance_angle: parseFloat(e.target.value) })
              }
              className="input w-full"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-dark-300 mb-1">
              Linear Tolerance (mm)
            </label>
            <input
              type="number"
              step="0.1"
              value={formData.default_tolerance_linear}
              onChange={(e) =>
                setFormData({ ...formData, default_tolerance_linear: parseFloat(e.target.value) })
              }
              className="input w-full"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-dark-300 mb-1">Notes</label>
          <textarea
            value={formData.notes}
            onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
            className="input w-full h-20 resize-none"
          />
        </div>

        <div className="flex justify-end gap-2 pt-4">
          <button type="button" onClick={onClose} className="btn btn-secondary">
            Cancel
          </button>
          <button type="submit" disabled={isSubmitting} className="btn btn-primary">
            {isSubmitting ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </form>
    </Modal>
  )
}

// Upload CAD Modal
function UploadCadModal({
  part,
  onClose,
  onSuccess,
}: {
  part: Part
  onClose: () => void
  onSuccess: () => void
}) {
  const [file, setFile] = useState<File | null>(null)
  const [analyzeBends, setAnalyzeBends] = useState(true)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')
  const [isUploading, setIsUploading] = useState(false)

  const handleUpload = async () => {
    if (!file) return

    setError('')
    setIsUploading(true)
    setProgress(0)

    try {
      await partCatalogApi.uploadCad(part.id, file, analyzeBends, setProgress)
      onSuccess()
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <Modal title={`Upload CAD for ${part.part_number}`} onClose={onClose}>
      <div className="space-y-4">
        {error && (
          <div className="p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 text-sm">
            {error}
          </div>
        )}

        <div
          className={clsx(
            'border-2 border-dashed rounded-lg p-8 text-center transition-colors',
            file ? 'border-primary-500 bg-primary-500/10' : 'border-dark-600 hover:border-dark-500'
          )}
        >
          {file ? (
            <div>
              <FileText className="w-12 h-12 mx-auto text-primary-400 mb-2" />
              <p className="text-dark-100 font-medium">{file.name}</p>
              <p className="text-dark-400 text-sm">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
              <button
                onClick={() => setFile(null)}
                className="text-primary-400 text-sm mt-2 hover:underline"
              >
                Change file
              </button>
            </div>
          ) : (
            <label className="cursor-pointer">
              <Upload className="w-12 h-12 mx-auto text-dark-500 mb-2" />
              <p className="text-dark-300">Drop CAD file here or click to browse</p>
              <p className="text-dark-500 text-sm mt-1">Supports STL, STEP, PLY</p>
              <input
                type="file"
                accept=".stl,.step,.stp,.ply"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="hidden"
              />
            </label>
          )}
        </div>

        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={analyzeBends}
            onChange={(e) => setAnalyzeBends(e.target.checked)}
            className="w-4 h-4 rounded border-dark-600 bg-dark-700 text-primary-500"
          />
          <span className="text-dark-300 text-sm">Automatically detect bends from CAD</span>
        </label>

        {isUploading && (
          <div className="space-y-2">
            <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary-500 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-dark-400 text-sm text-center">{progress}% uploaded</p>
          </div>
        )}

        <div className="flex justify-end gap-2 pt-4">
          <button type="button" onClick={onClose} className="btn btn-secondary">
            Cancel
          </button>
          <button
            onClick={handleUpload}
            disabled={!file || isUploading}
            className="btn btn-primary"
          >
            {isUploading ? 'Uploading...' : 'Upload CAD'}
          </button>
        </div>
      </div>
    </Modal>
  )
}

// Import CSV Modal
function ImportCsvModal({
  onClose,
  onSuccess,
}: {
  onClose: () => void
  onSuccess: () => void
}) {
  const [file, setFile] = useState<File | null>(null)
  const [error, setError] = useState('')
  const [result, setResult] = useState<{ imported: number; errors: string[] } | null>(null)
  const [isImporting, setIsImporting] = useState(false)

  const handleImport = async () => {
    if (!file) return

    setError('')
    setResult(null)
    setIsImporting(true)

    try {
      const res = await partCatalogApi.importCsv(file)
      setResult({ imported: res.imported, errors: res.errors })
      if (res.imported > 0 && res.errors.length === 0) {
        setTimeout(onSuccess, 1500)
      }
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setIsImporting(false)
    }
  }

  return (
    <Modal title="Import Parts from CSV" onClose={onClose}>
      <div className="space-y-4">
        {error && (
          <div className="p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 text-sm">
            {error}
          </div>
        )}

        {result && (
          <div
            className={clsx(
              'p-3 rounded-lg text-sm',
              result.errors.length > 0
                ? 'bg-amber-500/20 border border-amber-500/50 text-amber-400'
                : 'bg-emerald-500/20 border border-emerald-500/50 text-emerald-400'
            )}
          >
            <p className="font-medium">
              Imported {result.imported} parts
              {result.errors.length > 0 && ` with ${result.errors.length} errors`}
            </p>
            {result.errors.length > 0 && (
              <ul className="mt-2 list-disc list-inside">
                {result.errors.slice(0, 5).map((err, i) => (
                  <li key={i}>{err}</li>
                ))}
                {result.errors.length > 5 && (
                  <li>...and {result.errors.length - 5} more errors</li>
                )}
              </ul>
            )}
          </div>
        )}

        <div
          className={clsx(
            'border-2 border-dashed rounded-lg p-8 text-center transition-colors',
            file ? 'border-primary-500 bg-primary-500/10' : 'border-dark-600 hover:border-dark-500'
          )}
        >
          {file ? (
            <div>
              <FileText className="w-12 h-12 mx-auto text-primary-400 mb-2" />
              <p className="text-dark-100 font-medium">{file.name}</p>
              <button
                onClick={() => setFile(null)}
                className="text-primary-400 text-sm mt-2 hover:underline"
              >
                Change file
              </button>
            </div>
          ) : (
            <label className="cursor-pointer">
              <FileUp className="w-12 h-12 mx-auto text-dark-500 mb-2" />
              <p className="text-dark-300">Drop CSV file here or click to browse</p>
              <p className="text-dark-500 text-sm mt-1">
                Required columns: part_number, part_name
              </p>
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="hidden"
              />
            </label>
          )}
        </div>

        <div className="text-dark-400 text-sm">
          <p className="font-medium mb-1">CSV Format:</p>
          <code className="block bg-dark-700 p-2 rounded text-xs">
            part_number,part_name,customer,material,default_tolerance_angle
          </code>
        </div>

        <div className="flex justify-end gap-2 pt-4">
          <button type="button" onClick={onClose} className="btn btn-secondary">
            {result ? 'Close' : 'Cancel'}
          </button>
          {!result && (
            <button
              onClick={handleImport}
              disabled={!file || isImporting}
              className="btn btn-primary"
            >
              {isImporting ? 'Importing...' : 'Import'}
            </button>
          )}
        </div>
      </div>
    </Modal>
  )
}

// Modal Component
function Modal({
  title,
  onClose,
  children,
}: {
  title: string
  onClose: () => void
  children: React.ReactNode
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="relative bg-dark-800 rounded-xl border border-dark-700 w-full max-w-lg max-h-[90vh] overflow-auto">
        <div className="flex items-center justify-between p-4 border-b border-dark-700">
          <h2 className="text-lg font-semibold text-dark-100">{title}</h2>
          <button
            onClick={onClose}
            className="p-1 text-dark-400 hover:text-dark-100 rounded-lg hover:bg-dark-700"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="p-4">{children}</div>
      </div>
    </div>
  )
}
