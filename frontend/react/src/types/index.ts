// User types
export interface User {
  id: number
  username: string
  email: string
  role: 'admin' | 'operator' | 'viewer'
  is_active: boolean
  created_at: string
  last_login: string | null
}

// Auth types
export interface LoginRequest {
  username: string
  password: string
}

export interface AuthResponse {
  access_token: string
  token_type: string
  user: User
}

// Job types
export interface Job {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  stage: string
  message: string
  error: string | null
  part_id: string
  part_name: string
  material: string
  tolerance: number
  quality_score: number | null
  reference_path: string
  scan_path: string
  drawing_path: string | null
  report_path: string | null
  pdf_path: string | null
  result: QCResult | null
  created_at: string
  updated_at: string
  completed_at: string | null
}

export interface JobProgress {
  job_id: string
  status: string
  progress: number
  stage: string
  message: string
  current_step: string
  error: string | null
}

// QC Result types
export interface QCResult {
  overall_result: 'PASS' | 'FAIL' | 'REVIEW'
  quality_score: number
  part_id: string
  part_name: string
  material: string
  tolerance: number
  statistics: DeviationStatistics
  regions: RegionAnalysis[]
  ai_analysis: AIAnalysis | null
  heatmaps: {
    deviation?: string
    regions?: string
    distance?: string
  }
}

export interface DeviationStatistics {
  mean_deviation: number
  max_deviation: number
  min_deviation: number
  std_deviation: number
  points_in_tolerance: number
  total_points: number
  conformance_percentage: number
}

export interface RegionAnalysis {
  name: string
  status: 'OK' | 'WARN' | 'FAIL'
  mean_deviation: number
  max_deviation: number
  point_count: number
  notes: string | null
}

export interface AIAnalysis {
  summary: string
  critical_findings: string[]
  root_causes: RootCause[]
  recommendations: string[]
}

export interface RootCause {
  category: string
  description: string
  confidence: number
}

// Batch types
export interface BatchJob {
  batch_id: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  total_parts: number
  completed_parts: number
  failed_parts: number
  current_part: string
  material: string
  tolerance: number
  avg_quality_score: number | null
  created_at: string
  completed_at: string | null
  parts: Record<string, BatchPart>
}

export interface BatchPart {
  part_index: number
  job_id: string
  part_id: string
  part_name: string
  status: string
  progress: number
  result: string | null
  quality_score: number | null
  error: string | null
}

export interface BatchSummary {
  batch_id: string
  name: string
  material: string
  tolerance: number
  summary: {
    total_parts: number
    completed: number
    processing_errors: number
    qc_passed: number
    qc_failed: number
    pass_rate: number
  }
  quality_scores: {
    average: number | null
    min: number | null
    max: number | null
  }
  created_at: string
  completed_at: string
  duration_seconds: number
}

// Stats types
export interface SystemStats {
  total_jobs: number
  completed: number
  failed: number
  running: number
  pending: number
  storage: {
    uploads_mb: number
    output_mb: number
    total_mb: number
  }
}

// GD&T types
export interface GDTResult {
  gdt_type: string
  tolerance: number
  measured_value: number
  conformance: boolean
  details: Record<string, number | string>
}

// SPC types
export interface SPCCapability {
  cp: number
  cpk: number
  pp: number
  ppk: number
  mean: number
  std_dev: number
  sample_size: number
  rating: string
  ppm_below: number
  ppm_above: number
  ppm_total: number
}
