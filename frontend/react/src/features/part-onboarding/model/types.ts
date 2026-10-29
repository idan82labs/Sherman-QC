/**
 * Wire types for the production onboarding wizard (C09).
 *
 * These mirror the C01 contracts (`contracts/production/part_recipe.py`,
 * `compiled_part.py`) and the C05 `/api/production` routes as implemented in
 * `apps/api/routes/production_recipes.py` (see `api/onboardingApi.ts` for the
 * endpoint list). Raw C05 bodies are never used directly: `model/adapters.ts`
 * validates and normalises every response into these shapes, and anything
 * malformed becomes null instead of a crash. The Python models are normative;
 * when they disagree with this file, this file is the bug.
 *
 * Conventions (ENGINE_IO §1): lengths in mm, angles in degrees, every angle is
 * the INCLUDED angle (180 = flat), flange length is OUTSIDE_MOLD_LINE.
 * Nothing in here is a scan measurement.
 */

export type Point3 = [number, number, number]

// ---------------------------------------------------------------- tolerances

/** Bilateral band [-minus, +plus]. Both >= 0 and plus + minus > 0 (common.Tolerance). */
export interface ToleranceV1 {
  plus: number
  minus: number
}

export const TOLERANCE_SOURCES = [
  'DRAWING', 'ISO_2768_F', 'ISO_2768_M', 'ISO_2768_C', 'SHOP_STANDARD', 'CUSTOMER_SPEC',
] as const
export type ToleranceSource = typeof TOLERANCE_SOURCES[number]

export interface DefaultTolerancesV1 {
  angle_deg: ToleranceV1
  flange_mm: ToleranceV1
  bend_line_position_mm?: ToleranceV1 | null
  backgauge_mm?: ToleranceV1 | null
  general_mm?: ToleranceV1 | null
  /** Half-width of the heatmap colour band; not a conformance tolerance. */
  surface_profile_mm: number
  source: ToleranceSource
  note: string
}

export type FlangeSideRequirement = 'REQUIRED' | 'NOT_INSPECTED'

export interface FlangeInspectionV1 {
  side_a: FlangeSideRequirement
  side_b: FlangeSideRequirement
}

/** part_recipe.BendOverrideV1, keyed by Vendor Setup draft feature_id. */
export interface BendOverrideV1 {
  angle_deg?: ToleranceV1 | null
  flange_a_mm?: ToleranceV1 | null
  flange_b_mm?: ToleranceV1 | null
  bend_line_position_mm?: ToleranceV1 | null
  backgauge_mm?: ToleranceV1 | null
  flange_inspection?: FlangeInspectionV1 | null
  /** Manual marks only: included angle, 0 < x < 180. */
  manual_included_angle_deg?: number | null
}

export interface MeasurementOverrideV1 {
  tolerance?: ToleranceV1 | null
  required?: boolean | null
}

export type InspectionStage = 'FINAL' | 'IN_PROCESS'

/** part_recipe.ReleaseRequestV1 (also persisted as the per-part release draft). */
export interface ReleaseRequestV1 {
  expected_draft_revision: number
  expected_release_draft_revision?: number | null
  default_tolerances?: DefaultTolerancesV1 | null
  bend_overrides: Record<string, BendOverrideV1>
  measurement_overrides: Record<string, MeasurementOverrideV1>
  thickness_mm?: number | null
  inspection_stage: InspectionStage
  confirmation?: 'OPERATOR_CONFIRMED' | 'DEV_AUTO_CONFIRMED'
  release_note: string
}

export interface ReleaseDraftResponse {
  /** Optimistic revision of the persisted release draft (0 when none was saved). */
  revision: number
  request: ReleaseRequestV1 | null
  updated_at?: string | null
}

// ---------------------------------------------------------------- parts / onboarding

export const ONBOARDING_STATES = [
  'NEEDS_CAD', 'NEEDS_DRAWING', 'NEEDS_BEND_ORDER', 'NEEDS_TOLERANCES',
  'READY_TO_RELEASE', 'RELEASED', 'RELEASED_STALE',
] as const
export type OnboardingState = typeof ONBOARDING_STATES[number]

export interface ActiveRecipeRef {
  recipe_id: string
  version: number
  recipe_sha256: string
  content_sha256?: string
  /** Absent in the cheap GET /parts summary. */
  released_at?: string | null
  released_by?: { user_id: string; username: string; role: string } | null
  /** Onboarding only: the sources the active recipe was released from. */
  source?: { cad_sha256: string; drawing_sha256: string; draft_revision: number } | null
}

export interface ProductionPartRow {
  part_id: string
  part_number: string
  sku: string | null
  sku_display?: string | null
  part_name: string
  revision: string | null
  material?: string | null
  onboarding_state: OnboardingState
  active_recipe?: ActiveRecipeRef | null
}

export interface ProductionPartsResponse {
  parts: ProductionPartRow[]
}

export interface CreateProductionPartRequest {
  part_number: string
  part_name: string
  revision?: string
  material?: string
  customer?: string
  /** C05 create_part also stores `notes`; it has no thickness (that lives in the release draft). */
  notes?: string
}

/**
 * The part fields the wizard reads: the legacy catalog Part (Vendor Setup) or the
 * C05 `_part_summary` (onboarding / parts list), which carries `has_cad`,
 * `cad_format` and `has_drawing` instead of the file paths.
 */
export interface OnboardingPart {
  id: string
  part_number: string
  part_name: string
  revision: string | null
  material: string | null
  customer?: string | null
  cad_file_path?: string | null
  cad_file_hash: string | null
  cad_uploaded_at: string | null
  /** C05 summary: lower-case extension of the catalog CAD (e.g. 'step'), or null. */
  cad_format?: string | null
  has_cad?: boolean
  has_drawing?: boolean
  drawing_file_path?: string | null
  drawing_file_hash: string | null
  drawing_uploaded_at?: string | null
  bend_count?: number | null
  thickness_mm?: number | null
}

/** One row of the C05 dry-run checklist (`onboarding.checklist`). */
export interface ServerChecklistItem {
  key: 'sku' | 'cad' | 'drawing' | 'bend_order' | 'flat' | 'compile' | 'tolerances' | 'other' | string
  ok: boolean
  codes: string[]
  detail: string[]
}

/** C05 release-service `_compile_report`: what E1 said about the last compile. */
export interface CompileReportV1 {
  status?: 'COMPILED' | 'BLOCKED' | 'FAILED' | string
  compiled_part_sha256?: string | null
  compiler_version?: string | null
  blocker_codes?: string[]
  warning_codes?: string[]
  issues: Array<{ code: string; severity: 'BLOCKER' | 'WARNING'; bend_uid?: string | null; bend_label?: string | null; graph_bend_id?: string | null; detail?: string | null }>
  unmarked_graph_bends: Array<{ graph_bend_id: string; sweep_deg: number; inner_radius_mm: number }>
  detail?: string | null
  consistency_problems?: string[]
}

/** `onboarding.last_compile`: the last release attempt's compile, and whether it used the current CAD + draft. */
export interface LastCompileV1 {
  status: 'COMPILED' | 'BLOCKED' | 'FAILED' | 'PENDING' | string
  codes: string[]
  at: string | null
  recipe_sha256?: string | null
  applies_to_current_draft: boolean
  report: CompileReportV1 | null
}

export interface OnboardingResponse {
  part: OnboardingPart
  sku: string | null
  /** C05 dry-run checklist; the wizard merges these codes into its own step states. */
  checklist: ServerChecklistItem[]
  blockers: string[]
  messages?: Record<string, string> | null
  ready_to_release: boolean
  /** A release now would reproduce the active recipe (nothing to release). */
  release_would_be_unchanged?: boolean
  active_recipe?: ActiveRecipeRef | null
  onboarding_state: OnboardingState
  last_compile?: LastCompileV1 | null
  release_draft_revision?: number
  links?: { vendor_setup?: string; wizard?: string } | null
}

// ---------------------------------------------------------------- vendor setup (read-only mirror)

export interface VendorMarkLite {
  feature_id: string
  label: string
  source: 'cad_candidate' | 'manual'
  status: 'active' | 'removed'
  review_status?: 'pending' | 'confirmed'
  line_start: Point3 | null
  line_end: Point3 | null
  target_angle?: number | null
  target_radius?: number | null
}

export interface VendorMeasurementLite {
  id: string
  label: string
  kind: 'bend_pair' | 'points'
  bend_ids: string[]
  cad_distance: number
  target_mm: number | null
}

export interface VendorDraftLite {
  cad_file_hash: string
  candidate_generation: string
  draft_revision?: number
  is_stale?: boolean
  marks: VendorMarkLite[]
  sequence: string[]
  machine_moves?: string[][]
  general_measurements?: VendorMeasurementLite[]
  operator_sequence_confirmed: boolean
}

export interface VendorSetupLite {
  part: OnboardingPart & { preview_mesh_path?: string | null }
  candidates: VendorMarkLite[]
  draft: VendorDraftLite | null
  candidate_generation: string
}

// ---------------------------------------------------------------- flat intake (C05 PartFlatRecord, normalised)

export type FlatFormat = 'DXF' | 'STEP'
/** Recipe FlatBindingV1.state (what a release records). */
export type FlatBindingState = 'UNBOUND' | 'AUTO_PROPOSED' | 'OPERATOR_CONFIRMED'
/** C05 `part_flats.state`: UPLOADED (parsed, nothing decided), PARSE_FAILED, PROPOSED, CONFIRMED, DECLINED (kept unbound). */
export const FLAT_RECORD_STATES = ['UPLOADED', 'PARSE_FAILED', 'PROPOSED', 'CONFIRMED', 'DECLINED'] as const
export type FlatRecordState = typeof FLAT_RECORD_STATES[number]

/** E1 flat.FlatBendLine.to_json(): ids L1..Ln (DXF) or I1..In (flat STEP). */
export interface FlatBendLineV1 {
  line_id: string
  start_mm: number[]
  end_mm: number[]
  source: 'LAYER' | 'LINETYPE' | 'ANNOTATION' | 'INFERRED' | string
  layer?: string | null
  linetype?: string | null
  direction?: string | null
  angle_deg?: number | null
  annotation?: string | null
}

/**
 * Parse result. C05 stores `{status, summary: ParsedFlat.summary(), geometry: {outer, inner, ...}}`
 * or `{status: PARSE_FAILED, reason, detail}`; the adapter flattens it into this shape.
 */
export interface FlatParseV1 {
  status: 'PARSED' | 'PARSE_FAILED'
  code?: string | null
  message?: string | null
  format?: FlatFormat | string
  units?: string
  scale_to_mm?: number
  outer_points?: number
  inner_loops?: number
  outer_area_mm2?: number
  thickness_mm?: number | null
  notes?: string[]
  bend_lines: FlatBendLineV1[]
  /** Outer cut loop in flat mm (2D), decimated for display by C05. */
  outline_mm?: Array<[number, number]> | null
  inner_loops_mm?: Array<Array<[number, number]>> | null
  decimated?: boolean
}

export interface FlatFitChoiceV1 {
  equivalent_fits: number
  narrowed_by: string[]
  decided_by: string
  ambiguous: boolean
  mirror_determined: boolean
  alternative_maps: Array<Record<string, string>>
}

/** E1 flat_binding.FlatBindingProposal.to_json(). */
export interface FlatBindingProposalV1 {
  method: 'GRAPH_MATCH_RIGID_FIT' | 'MANUAL' | string
  bend_line_map: Record<string, string>
  mirrored: boolean
  confidence: number
  needs_operator_review: boolean
  k_factor: number
  k_uncertainty: number
  k_flat_fit: number | null
  k_plausible_band: [number, number] | null
  k_clamped: boolean
  outline_rms_mm: number
  runner_up_rms_mm: number | null
  fit_choice?: FlatFitChoiceV1 | null
  line_offsets_mm?: Record<string, number | null>
  developed_max_abs_delta_mm: number | null
  developed_detail?: string | null
}

/** The last proposal attempt C05 stored (E1 propose output plus the basis it was made against). */
export interface FlatProposalRecordV1 {
  status: 'PROPOSED' | 'RECONCILIATION_BLOCKED' | 'PARSE_FAILED' | 'CAD_READ_FAILED' | string
  proposal: FlatBindingProposalV1 | null
  reconciliation: { mark_not_in_flange_graph?: string[]; unmarked_bend?: number } | null
  /** flat_intake.binding_basis_sha256 of the CAD + bend identities the proposal used. */
  basis_sha256: string | null
  cad_sha256: string | null
  /** bend_uid -> mark label at proposal time. */
  bend_labels: Record<string, string>
  proposed_at: string | null
  reason?: string | null
  detail?: string | null
}

export interface ReleasedByV1 { user_id: string; username: string; role: string }

/** The operator decision C05 stored: a confirmed map, or "keep unbound". */
export interface FlatDecisionV1 {
  state: 'OPERATOR_CONFIRMED' | 'UNBOUND'
  /** flat line id -> bend_uid; empty when UNBOUND. A partial map is valid. */
  bend_line_map: Record<string, string>
  mirrored: boolean | null
  method: 'GRAPH_MATCH_RIGID_FIT' | 'MANUAL' | null
  basis_sha256: string | null
  cad_sha256: string | null
  confirmed_by?: ReleasedByV1 | null
  confirmed_at?: string | null
  decided_by?: ReleasedByV1 | null
  decided_at?: string | null
}

/** A part's flat as the wizard uses it (adapted from C05 `PartFlatRecord`, `model/adapters.ts`). */
export interface FlatIntakeV1 {
  part_id: string
  /** Optimistic revision: every C05 write takes it as expected_revision. */
  revision: number
  state: FlatRecordState
  flat: {
    sha256: string
    original_filename: string | null
    format: FlatFormat
    size_bytes?: number | null
    uploaded_at: string | null
  }
  parse: FlatParseV1
  proposal: FlatProposalRecordV1 | null
  binding: FlatDecisionV1 | null
}

/** PUT /parts/{id}/flat/binding (C05 set_flat_binding). */
export interface FlatBindingDecisionRequest {
  expected_revision: number
  state: 'OPERATOR_CONFIRMED' | 'UNBOUND'
  bend_line_map?: Record<string, string>
  /** null lets C05 take the proposal's mirror (it never stays null when a proposal exists). */
  mirrored?: boolean | null
}

// ---------------------------------------------------------------- compile (E1 CompiledPartV1 subset)

export type CharacteristicKind =
  'FLANGE_LENGTH' | 'INCLUDED_ANGLE' | 'BEND_LINE_POSITION' | 'BACKGAUGE_LENGTH' | 'GENERAL_MEASUREMENT'

export interface ToleranceRefV1 extends ToleranceV1 {
  origin: 'RECIPE_DEFAULT' | 'RECIPE_BEND_OVERRIDE' | 'RECIPE_MEASUREMENT_OVERRIDE'
  standard: ToleranceSource
}

export interface CharacteristicNominalV1 {
  characteristic_id: string
  kind: CharacteristicKind
  bend_uid?: string | null
  side?: 'A' | 'B' | null
  stage_index?: number | null
  measurement_id?: string | null
  unit: 'mm' | 'deg'
  nominal?: number | null
  unavailable_reason?: 'NOMINAL_UNAVAILABLE' | 'UNSUPPORTED_FEATURE' | null
  convention: string
  boundary_kind?: 'FREE_EDGE' | 'ADJACENT_BEND' | null
  adjacent_bend_uid?: string | null
  tolerance?: ToleranceRefV1 | null
  recipe_requirement: 'REQUIRED' | 'NOT_INSPECTED' | 'OPTIONAL'
}

export interface CompiledBendLite {
  bend_uid: string
  sequence_index: number
  label: string
  machine_move_index?: number | null
  graph_bend_id: string
  inner_radius_mm: number
  thickness_mm: number
  sweep_deg: number
  nominal_included_deg: number
  direction?: 'UP' | 'DOWN' | null
  bend_length_mm: number
  supported?: boolean
  unsupported_feature?: string | null
  /** C05 expectations document only: the recipe's own included angle and where it came from. */
  recipe_included_deg?: number | null
  recipe_angle_source?: string | null
}

export interface CompileIssueV1 {
  code: string
  severity: 'BLOCKER' | 'WARNING'
  bend_uid?: string | null
  graph_bend_id?: string | null
  detail?: string
}

export interface CompiledPartLite {
  schema?: string
  status: 'COMPILED' | 'BLOCKED'
  compiled_part_sha256?: string
  thickness?: { t_mm: number; source: string; spread_mm?: number } | null
  k_factor?: { value: number; source: 'BOUND_FLAT' | 'SHOP_DEFAULT'; uncertainty: number } | null
  bends: CompiledBendLite[]
  unmarked_graph_bends?: Array<{ graph_bend_id: string; sweep_deg: number; inner_radius_mm: number }>
  characteristics: CharacteristicNominalV1[]
  flat?: {
    provided: boolean
    parse_status: string
    binding_state?: FlatBindingState | null
    developed_length_check?: { status: 'OK' | 'MISMATCH' | 'NOT_EVALUATED'; threshold_mm?: number; max_abs_delta_mm?: number | null; detail?: string }
  }
  issues: CompileIssueV1[]
}

/** POST/GET /parts/{id}/compile-preview: a dry-run E1 compile of the current draft. */
export interface CompilePreviewResponse {
  status: 'PENDING' | 'COMPILED' | 'BLOCKED' | 'FAILED'
  job_id?: string | null
  /** Inputs the preview was computed from, so the wizard can tell a stale preview. */
  draft_revision?: number | null
  release_draft_revision?: number | null
  flat_sha256?: string | null
  compiled_part?: CompiledPartLite | null
  /** Recipe dry-run blockers (RecipeBuildError codes) that prevented the compile. */
  recipe_codes?: string[]
  error?: { code: string; message?: string } | null
}

/** C05 `expectations_document` (schema sherman_recipe_expectations_v1), stored at release. */
export interface RecipeExpectationsDocV1 {
  schema: 'sherman_recipe_expectations_v1'
  recipe_id: string
  recipe_sha256: string
  compiled_part_sha256: string
  algo_version: string
  thickness?: CompiledPartLite['thickness']
  k_factor?: CompiledPartLite['k_factor']
  bends: CompiledBendLite[]
  characteristics: CharacteristicNominalV1[]
  warnings?: Array<{ code: string; bend_uid?: string | null; detail?: string }>
}

export interface ExpectationsResponse {
  recipe_id?: string
  status: 'PENDING' | 'READY' | 'FAILED'
  algo_version?: string | null
  job_id?: string | null
  expectations?: RecipeExpectationsDocV1 | CompiledPartLite | { compiled_part?: CompiledPartLite | null } | null
  error?: { code: string; message?: string } | null
}

export interface ReleaseResultRecipe {
  recipe_id: string
  recipe_version: number
  recipe_sha256: string
  content_sha256?: string
  released_at?: string
  supersedes_version?: number | null
}

export interface ReleaseResponse {
  status: 'released' | 'unchanged'
  recipe: ReleaseResultRecipe
  /** E1 warnings of the release compile ("CODE: detail"). */
  warnings: string[]
}

/** Normalised HTTP error from any /api/production route. */
export interface ProductionApiError {
  status: number
  codes: string[]
  messages: Record<string, string>
  message: string
  /** C05 `detail.details`: code -> specifics (bend labels, file names). */
  details: Record<string, string[]>
  /** C05 release refusal: the E1 compile report, when the compile ran. */
  compile: CompileReportV1 | null
  /** True when the body carried C05's structured `{detail: {codes}}`; false for a bare framework 404/405. */
  structured: boolean
}
