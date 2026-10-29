/**
 * TypeScript mirrors of the production contracts the station screen consumes.
 *
 * Source of truth: `contracts/production/inspection_result.py`,
 * `line_job.py`, `characteristics.py`, `states.py` (C01). These are
 * structural mirrors only; runtime validation lives in `stationView.ts`
 * (`validateResult`), which re-applies the Python honesty rules so a
 * malformed or tampered document is refused instead of rendered.
 *
 * Frame: every coordinate is in `cad_model_mm` (the recipe STEP frame, mm).
 */

export type Vec3 = [number, number, number]

export type CharacteristicKind =
  | 'FLANGE_LENGTH'
  | 'INCLUDED_ANGLE'
  | 'BEND_LINE_POSITION'
  | 'BACKGAUGE_LENGTH'
  | 'GENERAL_MEASUREMENT'

export type FlangeSide = 'A' | 'B'
export type MeasurementState = 'MEASURED' | 'NOT_MEASURED' | 'NOT_APPLICABLE'
export type Decision = 'PASS' | 'FAIL' | 'REVIEW' | 'UNKNOWN' | 'NOT_APPLICABLE'
export type GuardBandDecision = 'PASS' | 'FAIL' | 'REVIEW'

export const UNKNOWN_REASONS = [
  'OCCLUDED',
  'EDGE_NOT_SEEN',
  'LOW_DENSITY',
  'AMBIGUOUS_SEGMENTATION',
  'REGISTRATION_UNTRUSTED',
  'AMBIGUOUS_POSE',
  'TIMEOUT',
  'UNSUPPORTED_FEATURE',
  'INVALID_INSPECTION',
  'BEND_STATE_UNKNOWN',
  'ADJACENT_BEND_NOT_MEASURED',
  'NOMINAL_UNAVAILABLE',
  'NO_TOLERANCE',
] as const
export type UnknownReason = (typeof UNKNOWN_REASONS)[number]

export const NOT_APPLICABLE_REASONS = [
  'BEND_NOT_YET_FORMED',
  'NOT_INSPECTED_BY_RECIPE',
  'NOT_APPLICABLE_AT_STAGE',
] as const
export type NotApplicableReason = (typeof NOT_APPLICABLE_REASONS)[number]

export type BoundaryKind = 'FREE_EDGE' | 'ADJACENT_BEND'
export type ExpectedSource =
  | 'COMPILED_NOMINAL'
  | 'STAGE_MODEL_NOMINAL'
  | 'RECIPE_TARGET'
  | 'CAD_DISTANCE'
  | 'NOMINAL_LOCUS_REFERENCE'

export const BEND_STATES = ['FORMED', 'NOT_YET_FORMED', 'PARTIALLY_FORMED', 'UNEXPECTED_FORMED', 'UNKNOWN'] as const
export type BendState = (typeof BEND_STATES)[number]
export const FORMED_LIKE_STATES: ReadonlySet<BendState> = new Set(['FORMED', 'PARTIALLY_FORMED', 'UNEXPECTED_FORMED'])

export type ValidityState =
  | 'VALID'
  | 'INSUFFICIENT_CAPTURE'
  | 'UNTRUSTED_REGISTRATION'
  | 'UNIT_MISMATCH_SUSPECTED'
  | 'UNSUPPORTED_INPUT'
  | 'WRONG_PART_SUSPECTED'
  | 'MIRRORED_PART_SUSPECTED'
  | 'TIMEOUT'

export const SUMMARY_STATES = [
  'ALL_MEASURED_IN_TOLERANCE',
  'STAGE_IN_TOLERANCE',
  'OUT_OF_TOLERANCE_FOUND',
  'REVIEW_REQUIRED',
  'INCOMPLETE',
  'INVALID_INSPECTION',
] as const
export type SummaryState = (typeof SUMMARY_STATES)[number]

export const STAGE_STATUSES = [
  'NOT_EVALUATED',
  'AMBIGUOUS_POSE',
  'UNDETERMINED',
  'PARTIAL_BEND_FOUND',
  'OUT_OF_ORDER_SUSPECTED',
  'STAGE_MISMATCH',
  'FINAL_INCOMPLETE',
  'CONSISTENT',
] as const
export type StageStatus = (typeof STAGE_STATUSES)[number]
export type StageDecisionValue = 'PASS' | 'REVIEW' | 'UNKNOWN'
export type InspectionStage = 'FINAL' | 'IN_PROCESS'

export interface ToleranceRef {
  plus: number
  minus: number
  origin: 'RECIPE_DEFAULT' | 'RECIPE_BEND_OVERRIDE' | 'RECIPE_MEASUREMENT_OVERRIDE'
  standard: string
}

export interface StationStats {
  count: number
  median: number
  min: number
  max: number
}

export interface CharacteristicEvidence {
  method: string
  support_points?: number | null
  coverage_ratio?: number | null
  fit_rms_mm?: number | null
  notes?: string[]
}

/** Mirror of `CharacteristicResultV1` (the survey's "QuantityResultV1"). */
export interface CharacteristicResult {
  characteristic_id: string
  kind: CharacteristicKind
  bend_uid: string | null
  side: FlangeSide | null
  stage_index: number | null
  measurement_id: string | null
  unit: 'mm' | 'deg'
  required: boolean
  state: MeasurementState
  decision: Decision
  reason: string | null
  expected: number | null
  expected_source: ExpectedSource | null
  actual: number | null
  deviation: number | null
  uncertainty: number | null
  tolerance: ToleranceRef | null
  boundary_kind: BoundaryKind | null
  adjacent_bend_uid: string | null
  stations: StationStats | null
  evidence: CharacteristicEvidence | null
}

export interface BendStateEvidence {
  nominal_included_deg: number
  measured_included_deg: number | null
  formed_fraction: number | null
  parent_support_points?: number | null
  child_support_points?: number | null
  coverage_ratio?: number | null
  prior_formed_probability?: number | null
}

export interface BendDisplay {
  nominal_polyline_mm: Vec3[]
  measured_polyline_mm: Vec3[] | null
  label_anchor_mm: Vec3
  side_a_anchor_mm: Vec3 | null
  side_b_anchor_mm: Vec3 | null
}

export interface BendRow {
  bend_uid: string
  label: string
  sequence_index: number
  machine_move_index: number | null
  state: BendState
  state_reasons: string[]
  state_evidence: BendStateEvidence
  /** FLANGE A, FLANGE B, ANGLE, POSITION, then optional BACKGAUGE. */
  characteristics: CharacteristicResult[]
  display: BendDisplay
  diagnostics: Record<string, number>
  notes: string[]
}

export interface GeneralMeasurementRow {
  measurement_id: string
  label: string
  characteristic: CharacteristicResult
  nominal_start_mm: Vec3
  nominal_end_mm: Vec3
  measured_start_mm: Vec3 | null
  measured_end_mm: Vec3 | null
}

export interface UnplannedCrease {
  crease_id: string
  reason: 'EXTRA_BEND_SUSPECTED' | 'UNPLANNED_CREASE'
  polyline_mm: Vec3[]
  included_deg: number | null
  note: string
}

export interface Registration {
  method: string
  frame: 'cad_model_mm'
  raw_to_frame: number[][] | null
  unit_factor: number
  datum_fitness: number | null
  rmse_mm: number | null
  pose_candidates_considered: number | null
  pose_margin: number | null
  pose_ambiguous: boolean
  /** Contract amendment ENGINE_IO-A2: tied symmetric twins; only readings identical under every twin are published. */
  identity_ambiguous: boolean
  in_plane_degenerate: boolean
  abstained: boolean
}

export interface InspectionValidity {
  state: ValidityState
  reasons: string[]
  metrology_qualified: false
  registration: Registration
  coverage: {
    surface_coverage_ratio: number | null
    datum_coverage_ratio: number | null
    scan_point_count: number | null
  }
}

export interface StageDecision {
  inspection_stage: InspectionStage
  declared_stage: number | null
  declared_stage_source: 'SIDECAR' | 'OPERATOR' | 'API' | null
  bends_total: number
  inferred_stage: number | null
  formed_bend_uids: string[]
  not_yet_formed_bend_uids: string[]
  unknown_bend_uids: string[]
  status: StageStatus
  flags: StageStatus[]
  decision: StageDecisionValue
  part_complete: boolean | null
  stage_model_id: string | null
  detail: string
}

export interface WorstCharacteristic {
  characteristic_id: string
  bend_uid: string | null
  kind: CharacteristicKind
  deviation: number
  uncertainty: number
  decision: Decision
  exceedance_ratio: number | null
}

export interface Summary {
  state: SummaryState
  bends_total: number
  bends_formed: number
  bends_not_yet_formed: number
  bends_unknown: number
  characteristics_required: number
  required_pass: number
  required_fail: number
  required_review: number
  required_unknown: number
  unplanned_creases: number
  worst: WorstCharacteristic | null
  reasons: string[]
}

export interface ViewerAssets {
  frame: 'cad_model_mm'
  registered_scan: string | null
  point_count: number | null
  deviations: string | null
  deviation_encoding: 'float32_le_nan_missing'
  deviation_sign: 'SIGNED_ALONG_STAGE_MODEL_OUTWARD_NORMAL'
  deviation_reference_stage: string | null
  heatmap_band_mm: number
  heatmap_unavailable_reason: string | null
  segment_labels: string | null
  stage_mesh: string | null
  final_mesh: string | null
  bend_geometry: string | null
}

export interface ResultContext {
  job_id: string | null
  station_id: string | null
  scan_file_name: string | null
  serial: string | null
  scan_timestamp: string | null
  received_at: string | null
  /** Studio (CAD) label per bend_uid; display only (optional, may be absent or null). */
  bend_display_labels?: Record<string, string> | null
}

/** Mirror of `ShopfloorInspectionResultV1`. */
export interface ShopfloorInspectionResult {
  schema: 'sherman_shopfloor_inspection_result_v1'
  created_at: string
  context: ResultContext
  sku: string
  part_name: string | null
  recipe: { recipe_id: string; version: number; recipe_sha256: string; content_sha256: string }
  scan: { sha256: string; format: string; vertex_count: number; face_count: number }
  engine: {
    engine_name: string
    engine_version: string
    config_sha256: string | null
    bundle_id: string | null
    compiled_part_sha256: string
    algorithm_versions: Record<string, string>
    duration_s: number | null
    peak_rss_bytes: number | null
  }
  authority: { mode: 'OPERATOR_ASSIST'; production_disposition: 'NOT_AUTHORIZED'; note: string }
  inspection_validity: InspectionValidity
  stage_decision: StageDecision
  summary: Summary
  bends: BendRow[]
  general_measurements: GeneralMeasurementRow[]
  unplanned_creases: UnplannedCrease[]
  viewer_assets: ViewerAssets
  artifacts: { pdf: string | null; overview_png: string | null }
  warnings: string[]
}

export type ExecutionStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
export type JobStage = 'queued' | 'preparing' | 'engine' | 'postprocess' | 'done'

/** Mirror of `LineJobSummaryV1`. */
export interface LineJobSummary {
  job_id: string
  kind: 'inspect' | 'recipe_compile'
  station_id: string | null
  sku: string
  part_name: string | null
  recipe_id: string
  recipe_version: number
  scan_file_name: string | null
  serial: string | null
  execution_status: ExecutionStatus
  stage: JobStage
  progress: number
  created_at: string
  started_at: string | null
  finished_at: string | null
  failure_code: string | null
  summary_state: SummaryState | null
  inspection_validity: ValidityState | null
  stage_status?: StageStatus | null
}

export interface StationIssue {
  kind: 'QUARANTINE' | 'JOB_FAILED'
  at: string
  file_name: string | null
  reason_code: string
  message: string
}

/** Mirror of `StationFeedV1`. */
export interface StationFeed {
  schema: 'sherman_station_feed_v1'
  station_id: string
  etag: string
  server_time: string
  latest_completed: {
    job: LineJobSummary
    result: ShopfloorInspectionResult
    /** served artifact name -> URL (/api/line/jobs/{job_id}/artifacts/{name}) */
    artifacts: Record<string, string>
  } | null
  active: { job: LineJobSummary } | null
  latest_issue: StationIssue | null
  recent: LineJobSummary[]
}

/** Served artifact names (`contracts.production.layout.ARTIFACT_ALLOWLIST`). */
export const ARTIFACT_NAMES = [
  'viewer_scan.ply',
  'viewer_deviations.f32',
  'stage_mesh.ply',
  'final_mesh.ply',
  'bend_geometry.json',
  'report.pdf',
  'overview.png',
] as const
export type ArtifactName = (typeof ARTIFACT_NAMES)[number]

export type Lang = 'en' | 'he'
