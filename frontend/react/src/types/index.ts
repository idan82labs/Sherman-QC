// API Response types
export interface APIError {
  error: string
  code: string
  details?: Record<string, unknown>
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  count: number
  offset: number
  limit: number
  has_more: boolean
}

export interface JobsListResponse {
  jobs: Job[]
  total: number
  count: number
  offset: number
  limit: number
  has_more: boolean
}

export interface BatchListResponse {
  batches: BatchJob[]
  total: number
  count: number
  offset: number
  limit: number
  has_more: boolean
}

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
  overall_result: 'PASS' | 'FAIL' | 'WARNING' | 'REVIEW'
  release_decision?: 'AUTO_PASS' | 'HOLD' | 'AUTO_FAIL' | string
  release_blocked_by?: string[]
  release_hold_reasons?: string[]
  claim_gate_reasons?: string[]
  blocker_attribution_breakdown?: Record<string, number>
  engine_gap_bends?: number
  scan_limited_bends?: number
  process_or_policy_bends?: number
  engine_recoverable_bends?: number
  scan_limited_due_to_observability?: number
  scan_limited_due_to_datum?: number
  scan_limited_due_to_correspondence?: number
  trusted_alignment_for_release?: boolean
  trusted_position_evidence?: boolean
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

export interface Recommendation {
  priority?: 'HIGH' | 'MEDIUM' | 'LOW'
  action: string
  expected_improvement?: string
}

export interface AIAnalysis {
  summary: string
  critical_findings: string[]
  root_causes: RootCause[]
  recommendations: (string | Recommendation)[]
}

export interface RootCause {
  category: string
  description: string
  confidence: number
}

export interface Defect {
  type: string
  location: string
  severity: 'minor' | 'moderate' | 'severe' | 'critical' | 'major'
  deviation_mm: number
  material_related: boolean
  description: string
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

// Enhanced Analysis Types (Bend Detection + Multi-Model)

export interface BendFeature {
  bend_id: number
  bend_name: string
  angle_degrees: number
  radius_mm: number
  bend_line_start: [number, number, number]
  bend_line_end: [number, number, number]
  bend_apex: [number, number, number]
  region_point_count: number
  adjacent_surfaces: number[]
  detection_confidence: number
  bend_direction: string
}

export interface BendResult {
  bend: BendFeature
  nominal_angle: number | null
  angle_deviation_deg: number
  springback_indicator: number
  over_bend_indicator: number
  apex_deviation_mm: number
  radius_deviation_mm: number
  twist_angle_deg: number
  status: 'pass' | 'warning' | 'fail'
  recommendations: string[]
  root_causes: string[]
}

export interface BendDetectionResult {
  bends: BendFeature[]
  total_bends_detected: number
  detection_method: string
  processing_time_ms: number
  warnings: string[]
}

export interface CorrelationMapping {
  mapping_id: string
  drawing_element_id: string
  drawing_element_type: string
  feature_3d_id: string
  feature_3d_type: string
  confidence: number
  nominal_value: number | null
  measured_value: number | null
  deviation: number | null
  is_in_tolerance: boolean | null
  notes: string
}

export interface CriticalDeviation {
  deviation_id: string
  feature_id: string
  feature_type: string
  location: [number, number, number]
  deviation_mm: number
  severity: 'critical' | 'major' | 'minor'
  affected_dimensions: string[]
  affected_gdt: string[]
  affected_bends: number[]
  description: string
}

export interface Correlation2D3D {
  correlation_id: string
  correlation_timestamp: string
  mappings: CorrelationMapping[]
  critical_deviations: CriticalDeviation[]
  unmatched_drawing_elements: string[]
  unmatched_3d_features: string[]
  overall_correlation_score: number
  model_used: string
}

export interface EnhancedRootCause {
  cause_id: string
  category: 'tooling' | 'material' | 'process' | 'design' | 'measurement'
  description: string
  severity: 'critical' | 'major' | 'minor'
  confidence: number
  affected_features: string[]
  affected_bends: number[]
  evidence: string[]
  recommendations: string[]
  estimated_impact: string | null
  priority: number
}

export interface PipelineStageStatus {
  stage: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
  model_used: string | null
  fallback_used: boolean
  start_time: string | null
  end_time: string | null
  duration_ms: number | null
  error_message: string | null
}

export interface PipelineStatus {
  job_id: string
  current_stage: string
  stages: Record<string, PipelineStageStatus>
  overall_progress: number
  started_at: string | null
  completed_at: string | null
  total_duration_ms: number | null
}

export interface EnhancedAnalysisResult {
  job_id: string
  analysis_timestamp: string
  drawing_extraction: DrawingExtraction | null
  point_cloud_features: PointCloudFeatures | null
  correlation_2d_3d: Correlation2D3D | null
  bend_results: BendResult[]
  enhanced_root_causes: EnhancedRootCause[]
  pipeline_status: PipelineStatus | null
  summary: {
    total_bends_detected: number
    bends_in_tolerance: number
    bends_out_of_tolerance: number
    critical_issues_count: number
  }
  metadata: {
    models_used: string[]
    fallbacks_triggered: string[]
    processing_time_ms: number
  }
}

export interface DrawingExtraction {
  drawing_id: string
  extraction_timestamp: string
  dimensions: DrawingDimension[]
  gdt_callouts: GDTCallout[]
  bends: DrawingBend[]
  material: string | null
  thickness: number | null
  part_number: string | null
  revision: string | null
  notes: string[]
  extraction_confidence: number
  model_used: string
}

export interface DrawingDimension {
  id: string
  type: string
  nominal: number
  tolerance_plus: number
  tolerance_minus: number
  unit: string
  feature_ref: string | null
  location_hint: string | null
}

export interface GDTCallout {
  id: string
  type: string
  tolerance: number
  unit: string
  feature_ref: string | null
  datums: string[]
  modifier: string | null
}

export interface DrawingBend {
  id: string
  sequence: number
  angle_nominal: number
  angle_tolerance: number
  radius: number
  direction: string
  bend_line_location: string | null
}

export interface PointCloudFeatures {
  scan_id: string
  detection_timestamp: string
  bends: BendFeature[]
  surfaces: Feature3D[]
  edges: Feature3D[]
  other_features: Feature3D[]
  total_points: number
  processed_points: number
  detection_confidence: number
  model_used: string
}

export interface Feature3D {
  feature_id: string
  feature_type: string
  location: [number, number, number]
  normal: [number, number, number] | null
  parameters: Record<string, unknown>
  point_count: number
  deviation_stats: Record<string, number>
  detection_confidence: number
}

// Dimension Analysis Types (XLSX-based comparison)

export interface DimensionComparison {
  dim_id: number
  type: string
  description: string
  expected: number
  reference_source: 'XLSX' | 'CAD' | 'N/A'
  xlsx_value: number | null
  cad_value: number | null
  scan_value: number | null
  unit: string
  tolerance: string
  xlsx_vs_cad_deviation: number | null
  scan_vs_cad_deviation: number | null
  scan_deviation: number | null
  scan_deviation_percent: number | null
  status: 'pass' | 'fail' | 'warning' | 'not_measured' | 'pending'
}

export interface BendMatchAnalysis {
  dim_id: number
  type: string
  expected: number
  tolerance: string
  cad: {
    angle: number | null
    radius: number | null
    position: [number, number, number] | null
    deviation: number | null
    matched: boolean
  }
  scan: {
    angle: number | null
    radius: number | null
    deviation: number | null
    measured: boolean
  }
  status: 'pass' | 'fail' | 'warning' | 'pending'
}

export interface DimensionAnalysisSummary {
  total_dimensions: number
  measured: number
  passed: number
  failed: number
  warnings: number
  pass_rate: number
}

export interface BendAnalysisSummary {
  total_bends: number
  passed: number
  failed: number
}

export interface DimensionAnalysisResult {
  metadata: {
    part_id: string
    part_name: string
    timestamp: string
  }
  summary: DimensionAnalysisSummary
  bend_summary: BendAnalysisSummary
  dimensions: DimensionComparison[]
  bend_analysis: {
    success: boolean
    summary: {
      total_bends: number
      matched: number
      passed: number
      failed: number
      warnings: number
    }
    bend_radius_spec: number | null
    matches: BendMatchAnalysis[]
    unmatched_xlsx: any[]
    unmatched_cad_count: number
    warnings: string[]
    error: string | null
  } | null
  failed_dimensions: DimensionComparison[]
  worst_deviations: DimensionComparison[]
}

// Part Catalog Types
export interface Part {
  id: string
  part_number: string
  part_name: string
  customer: string | null
  revision: string | null
  cad_file_path: string | null
  cad_file_hash: string | null
  cad_uploaded_at: string | null
  material: string | null
  default_tolerance_angle: number
  default_tolerance_linear: number
  bend_count: number | null
  has_embedding: boolean
  embedding_version: string | null
  embedding_computed_at: string | null
  created_at: string
  updated_at: string
  erp_import_id: string | null
  notes: string | null
}

export interface PartBendSpec {
  id: string
  part_id: string
  bend_id: string
  target_angle: number
  target_radius: number | null
  tolerance_angle: number | null
  tolerance_radius: number | null
  bend_line_start: [number, number, number] | null
  bend_line_end: [number, number, number] | null
  notes: string | null
}

export interface PartCatalogStats {
  total: number
  with_cad: number
  with_embedding: number
  ready: number
  needs_cad: number
  needs_embedding: number
}

export interface PartsListResponse {
  parts: Part[]
  total: number
  counts: PartCatalogStats
}

export interface PartCreateRequest {
  part_number: string
  part_name: string
  customer?: string
  revision?: string
  material?: string
  default_tolerance_angle?: number
  default_tolerance_linear?: number
  notes?: string
}

export interface PartUpdateRequest {
  part_name?: string
  customer?: string
  revision?: string
  material?: string
  default_tolerance_angle?: number
  default_tolerance_linear?: number
  notes?: string
}

export interface BendSpecCreateRequest {
  bend_id: string
  target_angle: number
  target_radius?: number
  tolerance_angle?: number
  tolerance_radius?: number
  notes?: string
}

// Live Scan Types
export interface ScanData {
  filename: string
  size_bytes: number
  received_at: string
  points_count: number | null
}

export interface RecognitionCandidate {
  part_id: string
  part_number: string
  part_name: string | null
  customer: string | null
  similarity: number
  match_confidence?: number
  distance: number
  has_cad: boolean
  warning?: string
}

export interface RecognitionResult {
  candidates: RecognitionCandidate[]
  top_match: string | null
  top_similarity: number
  processing_time_ms: number
  is_confident: boolean
}

export interface GapCluster {
  center: [number, number, number]
  size_voxels: number
  size_mm3: number
  diameter_mm: number
  location_hint: string
}

export interface LiveScanSession {
  id: string
  state: 'idle' | 'identifying' | 'scanning' | 'analyzing' | 'complete' | 'abandoned' | 'error'
  created_at: string
  updated_at: string
  part_id: string | null
  part_number: string | null
  recognition: RecognitionResult | null
  scans: ScanData[]
  total_points: number
  coverage_percent: number
  analysis_job_id: string | null
  error_message: string | null
  gap_clusters: GapCluster[]
}

export interface RecognitionStatus {
  index_count: number
  catalog_stats: {
    total: number
    with_cad: number
    with_embedding: number
    ready: number
    needs_cad: number
    needs_embedding: number
  }
  embedding_version: string
  status: 'ready' | 'no_parts'
}
