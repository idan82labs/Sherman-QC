import axios, { AxiosError, AxiosProgressEvent } from 'axios'
import type {
  AuthResponse,
  LoginRequest,
  Job,
  JobProgress,
  JobsListResponse,
  BatchJob,
  BatchSummary,
  BatchListResponse,
  SystemStats,
  User,
  APIError,
  Part,
  PartBendSpec,
  PartCatalogStats,
  PartsListResponse,
  PartCreateRequest,
  PartUpdateRequest,
  BendSpecCreateRequest,
} from '../types'

// Re-export APIError for backwards compatibility
export type { APIError }

// Helper to extract error message from API responses
// Handles both new standardized format and legacy format
export function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<APIError | { detail?: string; message?: string }>
    const data = axiosError.response?.data

    // New standardized format
    if (data && 'error' in data && typeof data.error === 'string') {
      return data.error
    }

    // Legacy formats
    if (data && 'detail' in data) {
      return data.detail || axiosError.message
    }
    if (data && 'message' in data) {
      return data.message || axiosError.message
    }

    return axiosError.message || 'An unexpected error occurred'
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'An unexpected error occurred'
}

// Extract error code from standardized error response
export function getErrorCode(error: unknown): string | null {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<APIError>
    return axiosError.response?.data?.code || null
  }
  return null
}

const api = axios.create({
  baseURL: '/api',
  timeout: 30000, // 30 second timeout
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Handle 401 responses
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// Auth API
export const authApi = {
  login: async (data: LoginRequest): Promise<AuthResponse> => {
    const response = await api.post('/auth/login', data)
    return response.data
  },

  getMe: async (): Promise<User> => {
    const response = await api.get('/auth/me')
    return response.data
  },

  changePassword: async (currentPassword: string, newPassword: string) => {
    const response = await api.post('/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    })
    return response.data
  },
}

// Jobs API
export const jobsApi = {
  list: async (params?: {
    status?: string
    part_id?: string
    limit?: number
    offset?: number
  }): Promise<JobsListResponse> => {
    const response = await api.get('/jobs', { params })
    return response.data
  },

  get: async (jobId: string): Promise<Job> => {
    const response = await api.get(`/jobs/${jobId}`)
    return response.data
  },

  delete: async (jobId: string) => {
    const response = await api.delete(`/jobs/${jobId}`)
    return response.data
  },

  getProgress: async (jobId: string): Promise<JobProgress> => {
    const response = await api.get(`/progress/${jobId}`)
    return response.data
  },

  getResult: async (jobId: string) => {
    const response = await api.get(`/result/${jobId}`)
    return response.data
  },

  startAnalysis: async (
    formData: FormData,
    onUploadProgress?: (progress: number) => void
  ): Promise<{ job_id: string }> => {
    const response = await api.post('/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120000, // 2 minute timeout for file uploads
      onUploadProgress: (event: AxiosProgressEvent) => {
        if (onUploadProgress && event.total) {
          const progress = Math.round((event.loaded / event.total) * 100)
          onUploadProgress(progress)
        }
      },
    })
    return response.data
  },
}

// Batch API
export const batchApi = {
  list: async (params?: {
    status?: string
    limit?: number
    offset?: number
  }): Promise<BatchListResponse> => {
    const response = await api.get('/batch', { params })
    return response.data
  },

  get: async (batchId: string): Promise<BatchJob> => {
    const response = await api.get(`/batch/${batchId}`)
    return response.data
  },

  getSummary: async (batchId: string): Promise<BatchSummary> => {
    const response = await api.get(`/batch/${batchId}/summary`)
    return response.data
  },

  delete: async (batchId: string) => {
    const response = await api.delete(`/batch/${batchId}`)
    return response.data
  },

  startAnalysis: async (formData: FormData): Promise<{ batch_id: string }> => {
    const response = await api.post('/batch/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  },
}

// Stats API
export const statsApi = {
  get: async (): Promise<SystemStats> => {
    const response = await api.get('/stats')
    return response.data
  },
}

// Health API
export const healthApi = {
  check: async () => {
    const response = await api.get('/health')
    return response.data
  },
}

// Bend Inspection API
export interface BendInspectionJob {
  job_id: string
  status: string
  progress: number
  stage?: string
  part_id: string
  part_name: string
  created_at: string
  completed_at?: string
  processing_time_ms?: number
  seed_case?: {
    state?: 'full' | 'partial' | string
    cad_file?: string
    scan_file?: string
  }
  report?: BendInspectionReport
  table?: string
  artifacts?: {
    reference_mesh_url?: string
    bend_overlay_overview_url?: string
    bend_overlay_issue_urls?: string[]
    bend_overlay_manifest_url?: string
    bend_report_pdf_url?: string
  }
  error?: string
}

export interface BendMatch {
  bend_id: string
  bend_form?: 'FOLDED' | 'ROLLED' | string
  cad_line_start?: [number, number, number] | null
  cad_line_end?: [number, number, number] | null
  detected_line_start?: [number, number, number] | null
  detected_line_end?: [number, number, number] | null
  display_cad_line_start?: [number, number, number] | null
  display_cad_line_end?: [number, number, number] | null
  display_detected_line_start?: [number, number, number] | null
  display_detected_line_end?: [number, number, number] | null
  callout_anchor?: [number, number, number] | null
  target_angle: number
  measured_angle: number | null
  angle_deviation: number | null
  target_radius: number
  measured_radius: number | null
  radius_deviation: number | null
  expected_line_length_mm?: number | null
  actual_line_length_mm?: number | null
  line_length_deviation_mm?: number | null
  line_start_deviation_mm?: number | null
  line_end_deviation_mm?: number | null
  line_center_deviation_mm?: number | null
  expected_arc_length_mm?: number | null
  actual_arc_length_mm?: number | null
  arc_length_deviation_mm?: number | null
  issue_location?: string
  action_item?: string
  status: 'PASS' | 'FAIL' | 'WARNING' | 'NOT_DETECTED'
  confidence: number
  tolerance_angle: number
  tolerance_radius: number
  physical_completion_state?: 'FORMED' | 'NOT_FORMED' | 'UNKNOWN' | string
  completion_state?: 'FORMED' | 'UNFORMED' | 'UNKNOWN' | string
  metrology_state?: 'IN_TOL' | 'OUT_OF_TOL' | 'UNMEASURABLE' | string
  positional_state?: 'ON_POSITION' | 'MISLOCATED' | 'UNKNOWN_POSITION' | string
  correspondence_state?: 'CONFIDENT' | 'AMBIGUOUS' | 'UNRESOLVED' | string
  correspondence_confidence?: number
  correspondence_margin?: number
  correspondence_runner_up_gap?: number | null
  correspondence_top_candidate_selected?: boolean | null
  correspondence_candidate_reused?: boolean | null
  correspondence_candidate_count?: number
  correspondence_source?: string
  observability_state_internal?: 'SUFFICIENT' | 'PARTIAL' | 'INSUFFICIENT' | string
  completion_observable?: boolean
  metrology_observable?: boolean
  position_observable?: boolean
  observability_state?: 'OBSERVED_FORMED' | 'OBSERVED_NOT_FORMED' | 'PARTIALLY_OBSERVED' | 'UNOBSERVED' | string
  observability_confidence?: number
  visibility_score?: number
  visibility_score_source?: string
  surface_visibility_ratio?: number
  local_support_score?: number
  side_balance_score?: number
  assignment_source?: 'GLOBAL_DETECTION' | 'CAD_LOCAL_NEIGHBORHOOD' | 'SEQUENCE_CONTINUITY' | 'NONE' | string
  assignment_confidence?: number
  assignment_candidate_id?: string | null
  assignment_candidate_kind?: 'MEASUREMENT' | 'GLOBAL_DETECTION' | 'NULL' | 'NONE' | string
  assignment_candidate_score?: number
  assignment_null_score?: number
  assignment_candidate_count?: number
  measurement_context?: Record<string, unknown>
  position_evidence?: {
    status?: 'SUPPORTED' | 'CONTRADICTED' | 'INCONCLUSIVE' | string
    source?: 'HEATMAP' | 'GEOMETRIC_PROXY' | 'BOTH' | string
    trusted_frame?: boolean
    reason?: string
  }
  correspondence_ready?: boolean
  datum_ready?: boolean
  metrology_claim_eligible?: boolean
  position_claim_eligible?: boolean
  claim_gate_reasons?: string[]
  primary_hold_cause?: string | null
  blocker_attribution?: 'engine_gap' | 'scan_limited' | 'process_or_policy' | string | null
  blocker_subtype?: string | null
  engine_recoverable?: boolean
  scan_reacquisition_recommended?: boolean
  metrology_failure_driver?: 'angle_only' | 'radius_only' | 'arc_only' | 'angle_radius_arc' | string | null
}

export interface ScanQualitySummary {
  status: 'GOOD' | 'FAIR' | 'POOR' | string
  queue_label?: string
  queue_color?: 'green' | 'yellow' | 'red' | string
  score?: number
  coverage_pct?: number
  overlap_iou_pct?: number
  density_pts_per_cm2?: number
  median_spacing_mm?: number
  p90_spacing_mm?: number
  bend_evidence_pct?: number
  in_spec_ratio_pct?: number
  point_count?: number
  notes?: string[]
}

export interface BendInspectionReport {
  part_id: string
  summary: {
    total_bends: number
    detected: number
    passed: number
    failed: number
    warnings: number
    progress_pct: number
    completed_bends?: number
    completed_in_spec?: number
    completed_out_of_spec?: number
    remaining_bends?: number
    is_complete?: boolean
    expected_bends?: number
    expected_progress_pct?: number
    overdetected_vs_expected?: number
    rolled_bends?: number
    folded_bends?: number
    critical_actions?: number
    scan_quality_status?: string
    scan_coverage_pct?: number
    scan_density_pts_per_cm2?: number
    match_evidence_completed_bends?: number
    match_evidence_remaining_bends?: number
    structured_completed_bends?: number
    structured_remaining_bends?: number
    structured_count_source?: string
    structured_count_mean?: number
    structured_count_map?: number
    structured_count_delta_vs_match?: number
    formed_bends?: number
    unformed_bends?: number
    unknown_completion_bends?: number
    in_tolerance_bends?: number
    out_of_tolerance_bends?: number
    unmeasurable_bends?: number
    on_position_bends?: number
    mislocated_bends?: number
    unknown_position_bends?: number
    overall_result?: string
    release_decision?: 'AUTO_PASS' | 'HOLD' | 'AUTO_FAIL' | string
    release_blocked_by?: string[]
    release_hold_reasons?: string[]
    claim_gate_reasons?: string[]
    blocker_attribution_breakdown?: Record<string, number>
    engine_gap_bends?: number
    scan_limited_bends?: number
    process_or_policy_bends?: number
    engine_gap_position_unknown_bends?: number
    scan_limited_position_unknown_bends?: number
    process_or_policy_out_of_tol_bends?: number
    engine_recoverable_bends?: number
    scan_limited_due_to_observability?: number
    scan_limited_due_to_datum?: number
    scan_limited_due_to_correspondence?: number
    trusted_alignment_for_release?: boolean
    trusted_position_evidence?: boolean
    correspondence_confident_bends?: number
    correspondence_ambiguous_bends?: number
    correspondence_unresolved_bends?: number
    observability_sufficient_bends?: number
    observability_partial_bends?: number
    observability_insufficient_bends?: number
  }
  operator_brief?: {
    headline?: string
    actions?: Array<{
      bend_id: string
      status: 'PASS' | 'FAIL' | 'WARNING' | 'NOT_DETECTED' | string
      location?: string
      action?: string
    }>
  }
  operator_actions?: Array<{
    bend_id: string
    status: 'PASS' | 'FAIL' | 'WARNING' | 'NOT_DETECTED' | string
    location?: string
    action?: string
  }>
  scan_quality?: ScanQualitySummary
  structured_context?: {
    count_posterior?: {
      count_distribution?: number[]
      median_completed_bends?: number
      map_completed_bends?: number
      mean_completed_bends?: number
    }
    confusable_group_count?: number
    joint_confusable_components?: string[][]
    joint_exclusivity_rejections?: number
  }
  matches: BendMatch[]
  unmatched_detections: unknown[]
  processing_time_ms: number
  overall_result?: string
  release_decision?: 'AUTO_PASS' | 'HOLD' | 'AUTO_FAIL' | string
}

export interface BendOverlayManifestEntry {
  bend_id: string
  status: 'PASS' | 'FAIL' | 'WARNING' | 'NOT_DETECTED' | string
  cad_line?: {
    raw_start?: [number, number, number] | null
    raw_end?: [number, number, number] | null
    display_start?: [number, number, number] | null
    display_end?: [number, number, number] | null
  }
  detected_line?: {
    raw_start?: [number, number, number] | null
    raw_end?: [number, number, number] | null
    display_start?: [number, number, number] | null
    display_end?: [number, number, number] | null
  }
  delta?: {
    angle_deg?: number | null
    radius_mm?: number | null
    center_mm?: number | null
    tolerance_angle_deg?: number | null
    tolerance_radius_mm?: number | null
  }
  anchor?: [number, number, number] | null
  include_in_overview?: boolean
  issue_tile?: string | null
  action_item?: string
  issue_location?: string
}

export interface BendOverlayManifest {
  generated_at: string
  overview_image: string
  issue_tiles: Array<{
    bend_id: string
    status: string
    image: string
    title?: string
  }>
  bends: BendOverlayManifestEntry[]
}

export interface BendSpecification {
  bend_id: string
  target_angle: number
  target_radius: number
  bend_line_length: number
  tolerance_angle: number
  tolerance_radius: number
}

export const bendInspectionApi = {
  // Start a new bend inspection analysis
  analyze: async (
    formData: FormData,
    onUploadProgress?: (progress: number) => void
  ): Promise<{ job_id: string; status: string; message: string }> => {
    const response = await api.post('/bend-inspection/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120000,
      onUploadProgress: (event) => {
        if (onUploadProgress && event.total) {
          const progress = Math.round((event.loaded / event.total) * 100)
          onUploadProgress(progress)
        }
      },
    })
    return response.data
  },

  // Get bend inspection result
  get: async (jobId: string): Promise<BendInspectionJob> => {
    const response = await api.get(`/bend-inspection/${jobId}`)
    return response.data
  },

  // Get ASCII table view of results
  getTable: async (jobId: string): Promise<{ table: string }> => {
    const response = await api.get(`/bend-inspection/${jobId}/table`)
    return response.data
  },

  getOverlayManifest: async (jobId: string): Promise<BendOverlayManifest> => {
    const response = await api.get(`/bend-inspection/${jobId}/overlay/manifest.json`)
    return response.data
  },

  // Extract CAD bends only (for preview)
  extractCadBends: async (
    formData: FormData
  ): Promise<{ part_id: string; total_bends: number; bends: BendSpecification[]; message: string }> => {
    const response = await api.post('/bend-inspection/extract-cad', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 60000,
    })
    return response.data
  },

  // List bend inspection jobs
  list: async (params?: {
    status?: string
    limit?: number
  }): Promise<{ jobs: BendInspectionJob[]; total: number }> => {
    const response = await api.get('/bend-inspection/jobs', { params })
    return response.data
  },

  // Delete bend inspection job
  delete: async (jobId: string) => {
    const response = await api.delete(`/bend-inspection/${jobId}`)
    return response.data
  },
}

// Part Catalog API
export const partCatalogApi = {
  // List parts with filters
  list: async (params?: {
    search?: string
    customer?: string
    has_cad?: boolean
    has_embedding?: boolean
    status?: string
    limit?: number
    offset?: number
  }): Promise<PartsListResponse> => {
    const response = await api.get('/parts', { params })
    return response.data
  },

  // Get catalog stats
  stats: async (): Promise<PartCatalogStats> => {
    const response = await api.get('/parts/stats')
    return response.data
  },

  // Get part by ID
  get: async (partId: string): Promise<{ part: Part; bend_specs: PartBendSpec[] }> => {
    const response = await api.get(`/parts/${partId}`)
    return response.data
  },

  // Get part by part number
  getByNumber: async (partNumber: string): Promise<{ part: Part; bend_specs: PartBendSpec[] }> => {
    const response = await api.get(`/parts/by-number/${encodeURIComponent(partNumber)}`)
    return response.data
  },

  // Create part
  create: async (data: PartCreateRequest): Promise<{ part: Part; message: string }> => {
    const response = await api.post('/parts', data)
    return response.data
  },

  // Update part
  update: async (partId: string, data: PartUpdateRequest): Promise<{ part: Part; message: string }> => {
    const response = await api.put(`/parts/${partId}`, data)
    return response.data
  },

  // Delete part
  delete: async (partId: string): Promise<{ deleted: boolean; message: string }> => {
    const response = await api.delete(`/parts/${partId}`)
    return response.data
  },

  // Upload CAD file
  uploadCad: async (
    partId: string,
    file: File,
    analyzeBends: boolean = true,
    onProgress?: (progress: number) => void
  ): Promise<{ part: Part; bends_detected: number; message: string }> => {
    const formData = new FormData()
    formData.append('cad_file', file)
    formData.append('analyze_bends', String(analyzeBends))

    const response = await api.post(`/parts/${partId}/cad`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120000,
      onUploadProgress: (event) => {
        if (onProgress && event.total) {
          onProgress(Math.round((event.loaded / event.total) * 100))
        }
      },
    })
    return response.data
  },

  // Get bend specs
  getBendSpecs: async (partId: string): Promise<{ bend_specs: PartBendSpec[] }> => {
    const response = await api.get(`/parts/${partId}/bend-specs`)
    return response.data
  },

  // Add bend spec
  addBendSpec: async (
    partId: string,
    data: BendSpecCreateRequest
  ): Promise<{ bend_spec: PartBendSpec; message: string }> => {
    const response = await api.post(`/parts/${partId}/bend-specs`, data)
    return response.data
  },

  // Import from CSV
  importCsv: async (file: File): Promise<{ imported: number; errors: string[]; message: string }> => {
    const formData = new FormData()
    formData.append('csv_file', file)

    const response = await api.post('/parts/import-csv', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  },
}

// Live Scan API
export const liveScanApi = {
  // Get current session
  getSession: async () => {
    const response = await api.get('/live-scan/session')
    return response.data
  },

  // Get recognition system status
  getRecognitionStatus: async () => {
    const response = await api.get('/recognize/status')
    return response.data
  },

  // Confirm part selection
  confirmPart: async (sessionId: string, partId: string, partNumber: string) => {
    const response = await api.post(`/live-scan/session/${sessionId}/confirm`, {
      part_id: partId,
      part_number: partNumber,
    })
    return response.data
  },

  // Complete scan
  completeScan: async (sessionId: string) => {
    const response = await api.post(`/live-scan/session/${sessionId}/complete`)
    return response.data
  },

  // Cancel session
  cancelSession: async (sessionId: string) => {
    const response = await api.post(`/live-scan/session/${sessionId}/cancel`)
    return response.data
  },

  // Reset session (clear to start fresh)
  resetSession: async () => {
    const response = await api.post('/live-scan/session/reset')
    return response.data
  },

  // Start session manager (admin)
  startManager: async (watchPath: string) => {
    const response = await api.post('/live-scan/start', { watch_path: watchPath })
    return response.data
  },

  // Stop session manager (admin)
  stopManager: async () => {
    const response = await api.post('/live-scan/stop')
    return response.data
  },

  // Get live scan configuration
  getConfig: async () => {
    const response = await api.get('/live-scan/config')
    return response.data
  },

  // Update watch folder path
  setWatchPath: async (watchPath: string) => {
    const response = await api.post('/live-scan/config', { watch_path: watchPath })
    return response.data
  },

  // Get live scan status
  getStatus: async () => {
    const response = await api.get('/live-scan/status')
    return response.data
  },

  // Get sampled points for current live session (for 3D rendering)
  getSessionPoints: async (params?: {
    max_points?: number
    max_scans?: number
  }): Promise<{
    session_id: string | null
    scan_count: number
    point_count: number
    points: Array<[number, number, number]>
    warning?: string
  }> => {
    const response = await api.get('/live-scan/session/points', { params })
    return response.data
  },

  // List available built-in demo scans
  getDemoOptions: async (): Promise<{
    default_part_number: string
    options: Array<{
      part_number: string
      part_name?: string
      cad_available: boolean
      scan_available: boolean
    }>
  }> => {
    const response = await api.get('/live-scan/demo/options')
    return response.data
  },

  // Load a functional demo LiDAR scan into live session flow
  loadTestScan: async (payload?: {
    part_number?: string
    point_count?: number
    noise_sigma_mm?: number
  }) => {
    const response = await api.post('/live-scan/demo/load', payload || {})
    return response.data
  },
}

// Parts with CAD API (for analysis picker)
export const partsApi = {
  // Get parts that have CAD files for analysis
  getPartsWithCad: async () => {
    const response = await api.get('/parts/with-cad')
    return response.data
  },

  // Analyze scan using part from catalog
  analyzeFromCatalog: async (partId: string, scanFile: File) => {
    const formData = new FormData()
    formData.append('part_id', partId.toString())
    formData.append('scan_file', scanFile)
    const response = await api.post('/bend-inspection/analyze-from-catalog', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  },
}

export default api
