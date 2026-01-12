import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload as UploadIcon, X, FileText, AlertCircle, CheckCircle } from 'lucide-react'
import { useMutation } from '@tanstack/react-query'
import { jobsApi } from '../services/api'
import clsx from 'clsx'

interface UploadFile {
  file: File
  type: 'scan' | 'cad'
  preview?: string
}

export default function Upload() {
  const navigate = useNavigate()
  const [scanFile, setScanFile] = useState<UploadFile | null>(null)
  const [cadFile, setCadFile] = useState<UploadFile | null>(null)
  const [partId, setPartId] = useState('')
  const [partName, setPartName] = useState('')
  const [material, setMaterial] = useState('aluminum')
  const [tolerance, setTolerance] = useState('0.1')
  const [error, setError] = useState('')

  const uploadMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      return jobsApi.startAnalysis(formData)
    },
    onSuccess: (data) => {
      navigate(`/jobs/${data.job_id}`)
    },
    onError: (err: any) => {
      setError(err.response?.data?.detail || 'Failed to start analysis')
    },
  })

  const handleDrop = useCallback(
    (type: 'scan' | 'cad') => (e: React.DragEvent) => {
      e.preventDefault()
      const file = e.dataTransfer.files[0]
      if (file) {
        const uploadFile: UploadFile = { file, type }
        if (type === 'scan') {
          setScanFile(uploadFile)
        } else {
          setCadFile(uploadFile)
        }
      }
    },
    []
  )

  const handleFileSelect = useCallback(
    (type: 'scan' | 'cad') => (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) {
        const uploadFile: UploadFile = { file, type }
        if (type === 'scan') {
          setScanFile(uploadFile)
        } else {
          setCadFile(uploadFile)
        }
      }
    },
    []
  )

  const removeFile = (type: 'scan' | 'cad') => {
    if (type === 'scan') {
      setScanFile(null)
    } else {
      setCadFile(null)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (!scanFile) {
      setError('Please upload a scan file (PDF or image)')
      return
    }

    if (!partId.trim()) {
      setError('Please enter a part ID')
      return
    }

    const formData = new FormData()
    formData.append('scan', scanFile.file)
    if (cadFile) {
      formData.append('cad', cadFile.file)
    }
    formData.append('part_id', partId.trim())
    formData.append('part_name', partName.trim() || partId.trim())
    formData.append('material', material)
    formData.append('tolerance', tolerance)

    uploadMutation.mutate(formData)
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-dark-100">New Analysis</h1>
        <p className="text-dark-400 mt-1">Upload scan and CAD files for quality control analysis</p>
      </div>

      {error && (
        <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center text-red-400">
          <AlertCircle className="w-5 h-5 mr-3 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* File uploads */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Scan file */}
          <div className="card p-6">
            <h3 className="font-semibold text-dark-100 mb-4">Scan File *</h3>
            {scanFile ? (
              <div className="border border-dark-600 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className="w-10 h-10 bg-primary-500/20 rounded-lg flex items-center justify-center mr-3">
                      <FileText className="w-5 h-5 text-primary-400" />
                    </div>
                    <div>
                      <p className="font-medium text-dark-100 truncate max-w-[180px]">
                        {scanFile.file.name}
                      </p>
                      <p className="text-sm text-dark-400">
                        {(scanFile.file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => removeFile('scan')}
                    className="p-2 hover:bg-dark-700 rounded-lg transition-colors"
                  >
                    <X className="w-5 h-5 text-dark-400" />
                  </button>
                </div>
              </div>
            ) : (
              <div
                onDrop={handleDrop('scan')}
                onDragOver={(e) => e.preventDefault()}
                className="border-2 border-dashed border-dark-600 rounded-lg p-8 text-center hover:border-primary-500/50 transition-colors cursor-pointer"
              >
                <input
                  type="file"
                  id="scan-file"
                  accept=".pdf,.png,.jpg,.jpeg"
                  onChange={handleFileSelect('scan')}
                  className="hidden"
                />
                <label htmlFor="scan-file" className="cursor-pointer">
                  <UploadIcon className="w-12 h-12 text-dark-500 mx-auto mb-4" />
                  <p className="text-dark-300 mb-2">
                    Drag & drop or <span className="text-primary-400">browse</span>
                  </p>
                  <p className="text-sm text-dark-500">PDF, PNG, JPG (max 50MB)</p>
                </label>
              </div>
            )}
          </div>

          {/* CAD file */}
          <div className="card p-6">
            <h3 className="font-semibold text-dark-100 mb-4">CAD Reference (Optional)</h3>
            {cadFile ? (
              <div className="border border-dark-600 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className="w-10 h-10 bg-secondary-500/20 rounded-lg flex items-center justify-center mr-3">
                      <FileText className="w-5 h-5 text-secondary-400" />
                    </div>
                    <div>
                      <p className="font-medium text-dark-100 truncate max-w-[180px]">
                        {cadFile.file.name}
                      </p>
                      <p className="text-sm text-dark-400">
                        {(cadFile.file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => removeFile('cad')}
                    className="p-2 hover:bg-dark-700 rounded-lg transition-colors"
                  >
                    <X className="w-5 h-5 text-dark-400" />
                  </button>
                </div>
              </div>
            ) : (
              <div
                onDrop={handleDrop('cad')}
                onDragOver={(e) => e.preventDefault()}
                className="border-2 border-dashed border-dark-600 rounded-lg p-8 text-center hover:border-secondary-500/50 transition-colors cursor-pointer"
              >
                <input
                  type="file"
                  id="cad-file"
                  accept=".pdf,.dxf,.step,.stp"
                  onChange={handleFileSelect('cad')}
                  className="hidden"
                />
                <label htmlFor="cad-file" className="cursor-pointer">
                  <UploadIcon className="w-12 h-12 text-dark-500 mx-auto mb-4" />
                  <p className="text-dark-300 mb-2">
                    Drag & drop or <span className="text-secondary-400">browse</span>
                  </p>
                  <p className="text-sm text-dark-500">PDF, DXF, STEP (max 50MB)</p>
                </label>
              </div>
            )}
          </div>
        </div>

        {/* Part details */}
        <div className="card p-6">
          <h3 className="font-semibold text-dark-100 mb-4">Part Details</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label htmlFor="part-id" className="label">
                Part ID *
              </label>
              <input
                id="part-id"
                type="text"
                value={partId}
                onChange={(e) => setPartId(e.target.value)}
                className="input"
                placeholder="e.g., PART-001"
                required
              />
            </div>
            <div>
              <label htmlFor="part-name" className="label">
                Part Name
              </label>
              <input
                id="part-name"
                type="text"
                value={partName}
                onChange={(e) => setPartName(e.target.value)}
                className="input"
                placeholder="e.g., Bracket Assembly"
              />
            </div>
            <div>
              <label htmlFor="material" className="label">
                Material
              </label>
              <select
                id="material"
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
                <option value="other">Other</option>
              </select>
            </div>
            <div>
              <label htmlFor="tolerance" className="label">
                Tolerance (mm)
              </label>
              <select
                id="tolerance"
                value={tolerance}
                onChange={(e) => setTolerance(e.target.value)}
                className="input"
              >
                <option value="0.01">+/- 0.01 mm (Very Tight)</option>
                <option value="0.05">+/- 0.05 mm (Tight)</option>
                <option value="0.1">+/- 0.1 mm (Standard)</option>
                <option value="0.25">+/- 0.25 mm (Loose)</option>
                <option value="0.5">+/- 0.5 mm (Very Loose)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Submit */}
        <div className="flex justify-end gap-4">
          <button
            type="button"
            onClick={() => navigate('/')}
            className="btn btn-secondary"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={uploadMutation.isPending}
            className={clsx(
              'btn btn-primary flex items-center',
              uploadMutation.isPending && 'opacity-50 cursor-not-allowed'
            )}
          >
            {uploadMutation.isPending ? (
              <>
                <svg
                  className="animate-spin -ml-1 mr-2 h-5 w-5"
                  xmlns="http://www.w3.org/2000/svg"
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
                Starting Analysis...
              </>
            ) : (
              <>
                <CheckCircle className="w-5 h-5 mr-2" />
                Start Analysis
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  )
}
