import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload as UploadIcon, X, AlertCircle, CheckCircle, Box, FileImage, Scan, FileSpreadsheet, Database, ChevronDown } from 'lucide-react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { jobsApi, getErrorMessage, partsApi } from '../services/api'
import { useNotification } from '../contexts/NotificationContext'
import clsx from 'clsx'

interface UploadFile {
  file: File
  type: 'scan' | 'cad' | 'drawing' | 'dimension'
  preview?: string
}

interface CatalogPart {
  id: string
  part_number: string
  part_name?: string | null
  name?: string | null
  customer?: string | null
  has_bend_specs: boolean
}

export default function Upload() {
  const navigate = useNavigate()
  const { success, error: showError } = useNotification()
  const [scanFile, setScanFile] = useState<UploadFile | null>(null)
  const [cadFile, setCadFile] = useState<UploadFile | null>(null)
  const [drawingFile, setDrawingFile] = useState<UploadFile | null>(null)
  const [dimensionFile, setDimensionFile] = useState<UploadFile | null>(null)
  const [partId, setPartId] = useState('')
  const [partName, setPartName] = useState('')
  const [material, setMaterial] = useState('aluminum')
  const [tolerance, setTolerance] = useState('0.1')
  const [error, setError] = useState('')
  const [uploadProgress, setUploadProgress] = useState(0)

  // Catalog picker state
  const [useCatalog, setUseCatalog] = useState(false)
  const [selectedCatalogPart, setSelectedCatalogPart] = useState<CatalogPart | null>(null)
  const [catalogDropdownOpen, setCatalogDropdownOpen] = useState(false)
  const getCatalogPartName = (part: CatalogPart): string =>
    part.part_name || part.name || part.part_number

  // Fetch parts with CAD from catalog
  const { data: catalogParts } = useQuery({
    queryKey: ['parts-with-cad'],
    queryFn: partsApi.getPartsWithCad,
    enabled: useCatalog,
  })

  const uploadMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      setUploadProgress(0)
      return jobsApi.startAnalysis(formData, (progress) => {
        setUploadProgress(progress)
      })
    },
    onSuccess: (data) => {
      success('Analysis started successfully!')
      navigate(`/jobs/${data.job_id}`)
    },
    onError: (err: unknown) => {
      const message = getErrorMessage(err)
      setError(message)
      showError(message)
      setUploadProgress(0)
    },
  })

  const handleDrop = useCallback(
    (type: 'scan' | 'cad' | 'drawing' | 'dimension') => (e: React.DragEvent) => {
      e.preventDefault()
      const file = e.dataTransfer.files[0]
      if (file) {
        const uploadFile: UploadFile = { file, type }
        if (type === 'scan') {
          setScanFile(uploadFile)
        } else if (type === 'cad') {
          setCadFile(uploadFile)
        } else if (type === 'dimension') {
          setDimensionFile(uploadFile)
        } else {
          setDrawingFile(uploadFile)
        }
      }
    },
    []
  )

  const handleFileSelect = useCallback(
    (type: 'scan' | 'cad' | 'drawing' | 'dimension') => (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) {
        const uploadFile: UploadFile = { file, type }
        if (type === 'scan') {
          setScanFile(uploadFile)
        } else if (type === 'cad') {
          setCadFile(uploadFile)
        } else if (type === 'dimension') {
          setDimensionFile(uploadFile)
        } else {
          setDrawingFile(uploadFile)
        }
      }
    },
    []
  )

  const removeFile = (type: 'scan' | 'cad' | 'drawing' | 'dimension') => {
    if (type === 'scan') {
      setScanFile(null)
    } else if (type === 'cad') {
      setCadFile(null)
    } else if (type === 'dimension') {
      setDimensionFile(null)
    } else {
      setDrawingFile(null)
    }
  }

  // Mutation for catalog-based analysis
  const catalogAnalysisMutation = useMutation({
    mutationFn: async ({ partId, scanFile }: { partId: string; scanFile: File }) => {
      return partsApi.analyzeFromCatalog(partId, scanFile)
    },
    onSuccess: (data) => {
      success('Analysis started successfully!')
      navigate(`/bend-inspection/${data.job_id}`)
    },
    onError: (err: unknown) => {
      const message = getErrorMessage(err)
      setError(message)
      showError(message)
    },
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (!scanFile) {
      setError('Please upload a scan file (STL, PLY, PCD, or OBJ)')
      return
    }

    // If using catalog, use catalog-based analysis
    if (useCatalog) {
      if (!selectedCatalogPart) {
        setError('Please select a part from the catalog')
        return
      }
      catalogAnalysisMutation.mutate({
        partId: selectedCatalogPart.id,
        scanFile: scanFile.file,
      })
      return
    }

    // Otherwise, use traditional upload-based analysis
    if (!cadFile) {
      setError('Please upload a CAD reference file (STL, STEP, or IGES)')
      return
    }

    if (!partId.trim()) {
      setError('Please enter a part ID')
      return
    }

    const formData = new FormData()
    formData.append('scan_file', scanFile.file)
    formData.append('reference_file', cadFile.file)
    if (drawingFile) {
      formData.append('drawing_file', drawingFile.file)
    }
    if (dimensionFile) {
      formData.append('dimension_file', dimensionFile.file)
    }
    formData.append('part_id', partId.trim())
    formData.append('part_name', partName.trim() || partId.trim())
    formData.append('material', material)
    formData.append('tolerance', tolerance)

    uploadMutation.mutate(formData)
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-dark-100">New Scan Analysis</h1>
        <p className="text-dark-400 mt-1">Upload 3D scan and CAD reference files for quality control analysis</p>
      </div>

      {error && (
        <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center text-red-400">
          <AlertCircle className="w-5 h-5 mr-3 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Primary file uploads - Scan and CAD */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Scan file - Point Cloud */}
          <div className="card p-6">
            <div className="flex items-center mb-4">
              <Scan className="w-5 h-5 text-primary-400 mr-2" />
              <h3 className="font-semibold text-dark-100">3D Scan File *</h3>
            </div>
            <p className="text-sm text-dark-400 mb-4">
              Point cloud from LiDAR or 3D scanner (the actual manufactured part)
            </p>
            {scanFile ? (
              <div className="border border-dark-600 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className="w-10 h-10 bg-primary-500/20 rounded-lg flex items-center justify-center mr-3">
                      <Scan className="w-5 h-5 text-primary-400" />
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
                  accept=".ply,.stl,.pcd,.obj,.xyz,.pts"
                  onChange={handleFileSelect('scan')}
                  className="hidden"
                />
                <label htmlFor="scan-file" className="cursor-pointer">
                  <UploadIcon className="w-12 h-12 text-dark-500 mx-auto mb-4" />
                  <p className="text-dark-300 mb-2">
                    Drag & drop or <span className="text-primary-400">browse</span>
                  </p>
                  <p className="text-sm text-dark-500">STL, PLY, PCD, OBJ, XYZ (max 100MB)</p>
                </label>
              </div>
            )}
          </div>

          {/* CAD file - Reference geometry */}
          <div className="card p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <Box className="w-5 h-5 text-secondary-400 mr-2" />
                <h3 className="font-semibold text-dark-100">CAD Reference *</h3>
              </div>

              {/* Toggle between upload and catalog */}
              <div className="flex items-center gap-2 bg-dark-800 rounded-lg p-1">
                <button
                  type="button"
                  onClick={() => {
                    setUseCatalog(false)
                    setSelectedCatalogPart(null)
                  }}
                  className={clsx(
                    'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                    !useCatalog
                      ? 'bg-secondary-500 text-white'
                      : 'text-dark-400 hover:text-dark-200'
                  )}
                >
                  <UploadIcon className="w-4 h-4 inline mr-1" />
                  Upload
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setUseCatalog(true)
                    setCadFile(null)
                  }}
                  className={clsx(
                    'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                    useCatalog
                      ? 'bg-primary-500 text-white'
                      : 'text-dark-400 hover:text-dark-200'
                  )}
                >
                  <Database className="w-4 h-4 inline mr-1" />
                  Catalog
                </button>
              </div>
            </div>

            <p className="text-sm text-dark-400 mb-4">
              {useCatalog
                ? 'Select a part from the catalog (CAD already uploaded)'
                : 'Nominal geometry from CAD (the designed/intended part)'}
            </p>

            {useCatalog ? (
              /* Catalog Picker */
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setCatalogDropdownOpen(!catalogDropdownOpen)}
                  className="w-full flex items-center justify-between px-4 py-3 bg-dark-800 border border-dark-600 rounded-lg hover:border-primary-500 transition-colors"
                >
                  {selectedCatalogPart ? (
                    <div className="flex items-center">
                      <Database className="w-5 h-5 text-primary-400 mr-3" />
                      <div className="text-left">
                        <p className="font-medium text-dark-100">{selectedCatalogPart.part_number}</p>
                        <p className="text-sm text-dark-400">{getCatalogPartName(selectedCatalogPart)}</p>
                      </div>
                    </div>
                  ) : (
                    <span className="text-dark-400">Select a part from catalog...</span>
                  )}
                  <ChevronDown className={clsx('w-5 h-5 text-dark-400 transition-transform', catalogDropdownOpen && 'rotate-180')} />
                </button>

                {catalogDropdownOpen && (
                  <div className="absolute z-10 w-full mt-2 bg-dark-800 border border-dark-600 rounded-lg shadow-xl max-h-64 overflow-y-auto">
                    {catalogParts?.parts?.length > 0 ? (
                      catalogParts.parts.map((part: CatalogPart) => (
                        <button
                          key={part.id}
                          type="button"
                          onClick={() => {
                            setSelectedCatalogPart(part)
                            setPartId(part.part_number)
                            setPartName(getCatalogPartName(part))
                            setCatalogDropdownOpen(false)
                          }}
                          className={clsx(
                            'w-full flex items-center px-4 py-3 hover:bg-dark-700 transition-colors text-left',
                            selectedCatalogPart?.id === part.id && 'bg-primary-500/10'
                          )}
                        >
                          <div className="flex-1">
                            <p className="font-medium text-dark-100">{part.part_number}</p>
                            <p className="text-sm text-dark-400">
                              {getCatalogPartName(part)} {part.customer && `• ${part.customer}`}
                            </p>
                          </div>
                          {part.has_bend_specs && (
                            <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded">
                              Bend Specs
                            </span>
                          )}
                        </button>
                      ))
                    ) : (
                      <div className="px-4 py-6 text-center text-dark-400">
                        <Database className="w-8 h-8 mx-auto mb-2 opacity-50" />
                        <p>No parts with CAD files found</p>
                        <p className="text-sm">Upload CAD files in Part Catalog first</p>
                      </div>
                    )}
                  </div>
                )}

                {selectedCatalogPart && (
                  <div className="mt-3 p-3 bg-primary-500/10 border border-primary-500/30 rounded-lg">
                    <div className="flex items-center text-primary-400 text-sm">
                      <CheckCircle className="w-4 h-4 mr-2" />
                      Using CAD from catalog: <span className="font-medium ml-1">{selectedCatalogPart.part_number}</span>
                    </div>
                  </div>
                )}
              </div>
            ) : cadFile ? (
              <div className="border border-dark-600 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className="w-10 h-10 bg-secondary-500/20 rounded-lg flex items-center justify-center mr-3">
                      <Box className="w-5 h-5 text-secondary-400" />
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
                  accept=".stl,.step,.stp,.iges,.igs,.obj"
                  onChange={handleFileSelect('cad')}
                  className="hidden"
                />
                <label htmlFor="cad-file" className="cursor-pointer">
                  <UploadIcon className="w-12 h-12 text-dark-500 mx-auto mb-4" />
                  <p className="text-dark-300 mb-2">
                    Drag & drop or <span className="text-secondary-400">browse</span>
                  </p>
                  <p className="text-sm text-dark-500">STL, STEP, IGES, OBJ (max 100MB)</p>
                </label>
              </div>
            )}
          </div>
        </div>

        {/* Technical Drawing - Optional */}
        <div className="card p-6">
          <div className="flex items-center mb-4">
            <FileImage className="w-5 h-5 text-amber-400 mr-2" />
            <h3 className="font-semibold text-dark-100">Technical Drawing</h3>
            <span className="ml-2 text-xs bg-dark-700 text-dark-400 px-2 py-1 rounded">Optional</span>
          </div>
          <p className="text-sm text-dark-400 mb-4">
            2D engineering drawing with GD&T callouts, dimensions, and tolerances.
            <span className="text-amber-400"> AI will use this to provide context-aware analysis</span> with accurate
            feature names, callout references, and manufacturing specifications.
          </p>
          {drawingFile ? (
            <div className="border border-dark-600 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="w-10 h-10 bg-amber-500/20 rounded-lg flex items-center justify-center mr-3">
                    <FileImage className="w-5 h-5 text-amber-400" />
                  </div>
                  <div>
                    <p className="font-medium text-dark-100 truncate max-w-[300px]">
                      {drawingFile.file.name}
                    </p>
                    <p className="text-sm text-dark-400">
                      {(drawingFile.file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => removeFile('drawing')}
                  className="p-2 hover:bg-dark-700 rounded-lg transition-colors"
                >
                  <X className="w-5 h-5 text-dark-400" />
                </button>
              </div>
            </div>
          ) : (
            <div
              onDrop={handleDrop('drawing')}
              onDragOver={(e) => e.preventDefault()}
              className="border-2 border-dashed border-dark-600 rounded-lg p-6 text-center hover:border-amber-500/50 transition-colors cursor-pointer"
            >
              <input
                type="file"
                id="drawing-file"
                accept=".pdf,.png,.jpg,.jpeg,.tiff,.tif"
                onChange={handleFileSelect('drawing')}
                className="hidden"
              />
              <label htmlFor="drawing-file" className="cursor-pointer flex items-center justify-center">
                <FileImage className="w-8 h-8 text-dark-500 mr-4" />
                <div className="text-left">
                  <p className="text-dark-300">
                    Drag & drop or <span className="text-amber-400">browse</span>
                  </p>
                  <p className="text-sm text-dark-500">PDF, PNG, JPG, TIFF (max 50MB)</p>
                </div>
              </label>
            </div>
          )}
          <div className="mt-3 p-3 bg-amber-500/5 border border-amber-500/20 rounded-lg">
            <p className="text-xs text-amber-400/80">
              <strong>Pro tip:</strong> Including the technical drawing enables AI to correlate deviations
              to specific callouts (e.g., "Feature B is 0.08mm out of position per callout 3.2") and
              provide manufacturing-specific recommendations based on GD&T requirements.
            </p>
          </div>
        </div>

        {/* Dimension Specifications - Optional XLSX */}
        <div className="card p-6">
          <div className="flex items-center mb-4">
            <FileSpreadsheet className="w-5 h-5 text-emerald-400 mr-2" />
            <h3 className="font-semibold text-dark-100">Dimension Specifications</h3>
            <span className="ml-2 text-xs bg-dark-700 text-dark-400 px-2 py-1 rounded">Optional</span>
          </div>
          <p className="text-sm text-dark-400 mb-4">
            Excel file with dimension specifications (bend angles, linear dimensions, tolerances).
            <span className="text-emerald-400"> Enables precise bend-by-bend comparison</span> showing
            expected vs actual values with pass/fail status.
          </p>
          {dimensionFile ? (
            <div className="border border-dark-600 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="w-10 h-10 bg-emerald-500/20 rounded-lg flex items-center justify-center mr-3">
                    <FileSpreadsheet className="w-5 h-5 text-emerald-400" />
                  </div>
                  <div>
                    <p className="font-medium text-dark-100 truncate max-w-[300px]">
                      {dimensionFile.file.name}
                    </p>
                    <p className="text-sm text-dark-400">
                      {(dimensionFile.file.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => removeFile('dimension')}
                  className="p-2 hover:bg-dark-700 rounded-lg transition-colors"
                >
                  <X className="w-5 h-5 text-dark-400" />
                </button>
              </div>
            </div>
          ) : (
            <div
              onDrop={handleDrop('dimension')}
              onDragOver={(e) => e.preventDefault()}
              className="border-2 border-dashed border-dark-600 rounded-lg p-6 text-center hover:border-emerald-500/50 transition-colors cursor-pointer"
            >
              <input
                type="file"
                id="dimension-file"
                accept=".xlsx,.xls"
                onChange={handleFileSelect('dimension')}
                className="hidden"
              />
              <label htmlFor="dimension-file" className="cursor-pointer flex items-center justify-center">
                <FileSpreadsheet className="w-8 h-8 text-dark-500 mr-4" />
                <div className="text-left">
                  <p className="text-dark-300">
                    Drag & drop or <span className="text-emerald-400">browse</span>
                  </p>
                  <p className="text-sm text-dark-500">Excel files (.xlsx, .xls)</p>
                </div>
              </label>
            </div>
          )}
          <div className="mt-3 p-3 bg-emerald-500/5 border border-emerald-500/20 rounded-lg">
            <p className="text-xs text-emerald-400/80">
              <strong>Non-AI Analysis:</strong> Dimension specifications provide deterministic comparison
              without AI interpretation. Results show exact deviations from spec for each dimension,
              ideal for compliance documentation and traceable QC records.
            </p>
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
                <optgroup label="Aluminum Alloys">
                  <option value="Al-6061-T6">Al 6061-T6</option>
                  <option value="Al-7075-T6">Al 7075-T6</option>
                  <option value="Al-5052-H32">Al 5052-H32</option>
                  <option value="Al-2024-T3">Al 2024-T3</option>
                </optgroup>
                <optgroup label="Steel">
                  <option value="Steel-1018">Steel 1018 (Mild)</option>
                  <option value="Steel-4140">Steel 4140</option>
                  <option value="Steel-A36">Steel A36</option>
                </optgroup>
                <optgroup label="Stainless Steel">
                  <option value="SS-304">SS 304</option>
                  <option value="SS-316">SS 316</option>
                  <option value="SS-17-4PH">SS 17-4 PH</option>
                </optgroup>
                <optgroup label="Other Metals">
                  <option value="Titanium-6Al4V">Titanium 6Al-4V</option>
                  <option value="Brass-C360">Brass C360</option>
                  <option value="Copper-C110">Copper C110</option>
                  <option value="Inconel-718">Inconel 718</option>
                </optgroup>
                <optgroup label="Plastics">
                  <option value="Delrin">Delrin (POM)</option>
                  <option value="PEEK">PEEK</option>
                  <option value="Nylon">Nylon</option>
                  <option value="UHMW">UHMW-PE</option>
                </optgroup>
                <option value="other">Other</option>
              </select>
            </div>
            <div>
              <label htmlFor="tolerance" className="label">
                Default Tolerance (mm)
              </label>
              <select
                id="tolerance"
                value={tolerance}
                onChange={(e) => setTolerance(e.target.value)}
                className="input"
              >
                <option value="0.005">+/- 0.005 mm (Precision)</option>
                <option value="0.01">+/- 0.01 mm (Very Tight)</option>
                <option value="0.025">+/- 0.025 mm (Tight)</option>
                <option value="0.05">+/- 0.05 mm (Standard - Machined)</option>
                <option value="0.1">+/- 0.1 mm (General)</option>
                <option value="0.25">+/- 0.25 mm (Loose)</option>
                <option value="0.5">+/- 0.5 mm (Very Loose)</option>
              </select>
              <p className="text-xs text-dark-500 mt-1">
                Used when not specified in drawing. GD&T from drawing takes precedence.
              </p>
            </div>
          </div>
        </div>

        {/* Upload Progress */}
        {uploadMutation.isPending && uploadProgress > 0 && (
          <div className="card p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-dark-300">Uploading files...</span>
              <span className="text-sm font-medium text-primary-400">{uploadProgress}%</span>
            </div>
            <div className="w-full bg-dark-700 rounded-full h-2">
              <div
                className="bg-primary-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* Submit */}
        <div className="flex justify-end gap-4">
          <button
            type="button"
            onClick={() => navigate('/')}
            className="btn btn-secondary"
            disabled={uploadMutation.isPending || catalogAnalysisMutation.isPending}
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={uploadMutation.isPending || catalogAnalysisMutation.isPending}
            className={clsx(
              'btn btn-primary flex items-center',
              (uploadMutation.isPending || catalogAnalysisMutation.isPending) && 'opacity-50 cursor-not-allowed'
            )}
          >
            {uploadMutation.isPending || catalogAnalysisMutation.isPending ? (
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
                {catalogAnalysisMutation.isPending
                  ? 'Analyzing...'
                  : uploadProgress < 100
                  ? `Uploading ${uploadProgress}%...`
                  : 'Starting Analysis...'}
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
