/**
 * Pure view model of the station screen.
 *
 * - `validateResult` re-applies the honesty rules of the Python contract
 *   (`contracts/production/inspection_result.py`) to a result received over
 *   the network. A document that fails is never rendered as a verdict.
 * - `deriveStageDecision` / `deriveSummary` mirror the Python pure
 *   derivations exactly (same precedence, same counts, same worst pick).
 * - `toRows`, `formatQuantity`, `conformanceStyle`, `summaryBanner`,
 *   `worstFocus` / `worstOrder` and `stageStrip` turn a validated result into
 *   display data. UNKNOWN is never given a pass/fail colour, and a nominal is
 *   never shown as an actual.
 *
 * No React, no DOM: `scripts/check-station-view-model.mjs` imports this file
 * directly (TypeScript transpile) and asserts it against the C01 examples.
 */
import {
  BEND_STATES,
  FORMED_LIKE_STATES,
  NOT_APPLICABLE_REASONS,
  STAGE_STATUSES,
  SUMMARY_STATES,
  UNKNOWN_REASONS,
  type BendRow,
  type BendState,
  type CharacteristicKind,
  type CharacteristicResult,
  type Decision,
  type GeneralMeasurementRow,
  type GuardBandDecision,
  type Lang,
  type ShopfloorInspectionResult,
  type StageDecisionValue,
  type StageStatus,
  type Summary,
  type SummaryState,
  type ToleranceRef,
  type Vec3,
  type WorstCharacteristic,
} from './types'
import { DISPLAY_TOKENS, type Tone } from './vocabulary'
import { reasonText, t } from '../copy'

// ---------------------------------------------------------------------------
// Contract constants and pure rules (mirrors of characteristics.py)
// ---------------------------------------------------------------------------

export const SHOPFLOOR_RESULT_SCHEMA = 'sherman_shopfloor_inspection_result_v1'
export const AUTHORITY_MODE = 'OPERATOR_ASSIST'
export const PRODUCTION_DISPOSITION = 'NOT_AUTHORIZED'
export const GUARD_BAND_EPS = 1e-9
const BEND_UID_RE = /^bu_[0-9a-f]{20}$/
const STAGE_MODEL_ID_RE = /^S[0-9]{1,4}(-[a-z0-9]{1,16})?$/

/** Fixed per-bend order: flange A, flange B, angle, position (+ optional backgauge). */
export const BEND_CHARACTERISTIC_ORDER: ReadonlyArray<[CharacteristicKind, 'A' | 'B' | null]> = [
  ['FLANGE_LENGTH', 'A'],
  ['FLANGE_LENGTH', 'B'],
  ['INCLUDED_ANGLE', null],
  ['BEND_LINE_POSITION', null],
]

const NA_REASONS_BY_KIND: Record<CharacteristicKind, ReadonlySet<string>> = {
  // NOT_APPLICABLE_AT_STAGE on a flange: contract amendment ENGINE_IO-A1 (adjacent bend NOT_YET_FORMED)
  FLANGE_LENGTH: new Set(['BEND_NOT_YET_FORMED', 'NOT_INSPECTED_BY_RECIPE', 'NOT_APPLICABLE_AT_STAGE']),
  INCLUDED_ANGLE: new Set(['BEND_NOT_YET_FORMED']),
  BEND_LINE_POSITION: new Set(['BEND_NOT_YET_FORMED']),
  BACKGAUGE_LENGTH: new Set(['BEND_NOT_YET_FORMED', 'NOT_APPLICABLE_AT_STAGE']),
  GENERAL_MEASUREMENT: new Set(['NOT_INSPECTED_BY_RECIPE', 'NOT_APPLICABLE_AT_STAGE']),
}
const UNKNOWN_REASON_SET: ReadonlySet<string> = new Set(UNKNOWN_REASONS)
const NA_REASON_SET: ReadonlySet<string> = new Set(NOT_APPLICABLE_REASONS)

export function round6(x: number): number {
  const r = Math.round(x * 1e6) / 1e6
  return Object.is(r, -0) ? 0 : r
}

export function unitForKind(kind: CharacteristicKind): 'mm' | 'deg' {
  return kind === 'INCLUDED_ANGLE' ? 'deg' : 'mm'
}

export function makeCharacteristicId(c: Pick<CharacteristicResult, 'kind' | 'bend_uid' | 'side' | 'stage_index' | 'measurement_id'>): string | null {
  if (c.kind === 'GENERAL_MEASUREMENT') {
    if (!c.measurement_id || c.bend_uid != null || c.side != null || c.stage_index != null) return null
    return `gm/${c.measurement_id}`
  }
  if (!c.bend_uid || c.measurement_id != null) return null
  if (c.kind === 'FLANGE_LENGTH') {
    if ((c.side !== 'A' && c.side !== 'B') || c.stage_index != null) return null
    return `${c.bend_uid}/FLANGE_LENGTH/${c.side}`
  }
  if (c.kind === 'BACKGAUGE_LENGTH') {
    if (c.stage_index == null || c.stage_index < 0) return null
    return `${c.bend_uid}/BACKGAUGE_LENGTH/S${c.stage_index}`
  }
  if (c.stage_index != null) return null
  return `${c.bend_uid}/${c.kind}`
}

/** Guard-band rule (characteristics.guard_band_decision). */
export function guardBandDecision(deviation: number, uncertainty: number, plus: number, minus: number): GuardBandDecision {
  if (!(uncertainty > 0)) throw new RangeError('uncertainty must be strictly positive')
  if (plus < 0 || minus < 0 || !(plus + minus > 0)) throw new RangeError('invalid tolerance band')
  if (deviation + uncertainty <= plus + GUARD_BAND_EPS && deviation - uncertainty >= -minus - GUARD_BAND_EPS) return 'PASS'
  if (deviation - uncertainty > plus + GUARD_BAND_EPS || deviation + uncertainty < -minus - GUARD_BAND_EPS) return 'FAIL'
  return 'REVIEW'
}

/** `(|d| + U) / tol_side`; null when the limit on the deviation's side is zero. */
export function exceedanceRatio(deviation: number, uncertainty: number, plus: number, minus: number): number | null {
  const side = deviation >= 0 ? plus : minus
  if (side <= 0) return null
  return (Math.abs(deviation) + uncertainty) / side
}

// ---------------------------------------------------------------------------
// Derivations (mirrors of derive_stage_decision / derive_summary)
// ---------------------------------------------------------------------------

// inspection_result._STAGE_PRECEDENCE: REVIEW evidence outranks an unknown bend
// elsewhere (as REVIEW outranks UNKNOWN for characteristics), so UNDETERMINED
// never hides it.
const STAGE_PRECEDENCE: ReadonlyArray<[StageStatus, StageDecisionValue]> = [
  ['NOT_EVALUATED', 'UNKNOWN'],
  ['AMBIGUOUS_POSE', 'UNKNOWN'],
  ['PARTIAL_BEND_FOUND', 'REVIEW'],
  ['OUT_OF_ORDER_SUSPECTED', 'REVIEW'],
  ['STAGE_MISMATCH', 'REVIEW'],
  ['FINAL_INCOMPLETE', 'REVIEW'],
  ['UNDETERMINED', 'UNKNOWN'],
  ['CONSISTENT', 'PASS'],
]

export interface DerivedStageDecision {
  bends_total: number
  inferred_stage: number | null
  formed_bend_uids: string[]
  not_yet_formed_bend_uids: string[]
  unknown_bend_uids: string[]
  status: StageStatus
  flags: StageStatus[]
  decision: StageDecisionValue
  part_complete: boolean | null
}

type StageInputRow = Pick<BendRow, 'bend_uid' | 'sequence_index' | 'machine_move_index' | 'state'>

export function deriveStageDecision(args: {
  inspectionStage: string
  declaredStage: number | null
  bends: ReadonlyArray<StageInputRow>
  poseAmbiguous: boolean
  validityState?: string
}): DerivedStageDecision {
  const validityState = args.validityState ?? 'VALID'
  const rows = [...args.bends].sort((a, b) => a.sequence_index - b.sequence_index)
  const formed = rows.filter(r => FORMED_LIKE_STATES.has(r.state)).map(r => r.bend_uid)
  const notFormed = rows.filter(r => r.state === 'NOT_YET_FORMED').map(r => r.bend_uid)
  const unknown = rows.filter(r => r.state === 'UNKNOWN').map(r => r.bend_uid)

  const firstOfMove = new Map<number, number>()
  for (const r of rows) {
    if (r.machine_move_index != null && !firstOfMove.has(r.machine_move_index)) firstOfMove.set(r.machine_move_index, r.sequence_index)
  }
  const rank = (r: StageInputRow) => (r.machine_move_index != null ? (firstOfMove.get(r.machine_move_index) as number) : r.sequence_index)
  const formedRanks = rows.filter(r => FORMED_LIKE_STATES.has(r.state)).map(rank)
  const unformedRanks = rows.filter(r => r.state === 'NOT_YET_FORMED').map(rank)
  let orderConsistent = !(formedRanks.length && unformedRanks.length && Math.min(...unformedRanks) < Math.max(...formedRanks))
  if (rows.some(r => r.state === 'UNEXPECTED_FORMED')) orderConsistent = false

  const inferred = unknown.length ? null : formed.length
  const flags: StageStatus[] = []
  if (validityState !== 'VALID') flags.push('NOT_EVALUATED')
  if (args.poseAmbiguous) flags.push('AMBIGUOUS_POSE')
  if (unknown.length) flags.push('UNDETERMINED')
  if (rows.some(r => r.state === 'PARTIALLY_FORMED')) flags.push('PARTIAL_BEND_FOUND')
  if (!orderConsistent) flags.push('OUT_OF_ORDER_SUSPECTED')
  if (args.declaredStage != null && inferred != null && args.declaredStage !== inferred) flags.push('STAGE_MISMATCH')
  if (args.inspectionStage === 'FINAL' && notFormed.length) flags.push('FINAL_INCOMPLETE')
  if (!flags.length) flags.push('CONSISTENT')
  const order = STAGE_PRECEDENCE.map(([name]) => name)
  flags.sort((a, b) => order.indexOf(a) - order.indexOf(b))
  const status = flags[0]
  const decision = (STAGE_PRECEDENCE.find(([name]) => name === status) as [StageStatus, StageDecisionValue])[1]
  const partComplete =
    unknown.length || validityState !== 'VALID'
      ? null
      : !notFormed.length && !rows.some(r => r.state === 'PARTIALLY_FORMED')
  return {
    bends_total: rows.length,
    inferred_stage: inferred,
    formed_bend_uids: formed,
    not_yet_formed_bend_uids: notFormed,
    unknown_bend_uids: unknown,
    status,
    flags,
    decision,
    part_complete: partComplete,
  }
}

/** All characteristics in display order: bends by sequence, then general measurements. */
export function iterCharacteristics(
  bends: ReadonlyArray<BendRow>,
  generalMeasurements: ReadonlyArray<GeneralMeasurementRow> = [],
): CharacteristicResult[] {
  const out: CharacteristicResult[] = []
  for (const row of bends) out.push(...row.characteristics)
  for (const gm of generalMeasurements) out.push(gm.characteristic)
  return out
}

const COMPLETE_BEND_STATES: ReadonlySet<BendState> = new Set(['FORMED', 'UNEXPECTED_FORMED'])

/** Pure part summary; same precedence and fields as `derive_summary`. */
export function deriveSummary(args: {
  validityState: string
  bends: ReadonlyArray<BendRow>
  generalMeasurements?: ReadonlyArray<GeneralMeasurementRow>
  stageDecision?: string
  stageStatus?: string
  unplannedCreases?: number
}): Summary {
  const stageDecision = args.stageDecision ?? 'PASS'
  const stageStatus = args.stageStatus ?? 'CONSISTENT'
  const creases = args.unplannedCreases ?? 0
  const required = iterCharacteristics(args.bends, args.generalMeasurements ?? []).filter(c => c.required)
  const count = (d: Decision) => required.filter(c => c.decision === d).length
  const counts = { PASS: count('PASS'), FAIL: count('FAIL'), REVIEW: count('REVIEW'), UNKNOWN: count('UNKNOWN') }
  const reasons: string[] = []
  let state: SummaryState
  if (args.validityState !== 'VALID') {
    state = 'INVALID_INSPECTION'
    reasons.push(`VALIDITY:${args.validityState}`)
  } else if (counts.FAIL) {
    state = 'OUT_OF_TOLERANCE_FOUND'
  } else if (counts.REVIEW || stageDecision === 'REVIEW' || creases) {
    state = 'REVIEW_REQUIRED'
  } else if (counts.UNKNOWN || stageDecision === 'UNKNOWN') {
    state = 'INCOMPLETE'
  } else if (!required.length) {
    state = 'INCOMPLETE'
    reasons.push('NO_REQUIRED_MEASURED')
  } else if (!args.bends.every(r => COMPLETE_BEND_STATES.has(r.state))) {
    state = 'STAGE_IN_TOLERANCE'
    reasons.push('PART_NOT_COMPLETE')
  } else {
    state = 'ALL_MEASURED_IN_TOLERANCE'
  }
  if (args.validityState === 'VALID') {
    if (counts.FAIL) reasons.push('REQUIRED_FAIL')
    if (counts.REVIEW) reasons.push('REQUIRED_REVIEW')
    if (counts.UNKNOWN) reasons.push('REQUIRED_UNKNOWN')
    if (stageDecision !== 'PASS') reasons.push(`STAGE:${stageStatus}`)
    if (creases) reasons.push('UNPLANNED_CREASES')
    // Optional characteristics never change the state, but a red or amber row is
    // never shown under an unexplained banner.
    const optional = iterCharacteristics(args.bends, args.generalMeasurements ?? []).filter(c => !c.required)
    if (optional.some(c => c.decision === 'FAIL')) reasons.push('OPTIONAL_FAIL')
    if (optional.some(c => c.decision === 'REVIEW')) reasons.push('OPTIONAL_REVIEW')
  }

  let worst: WorstCharacteristic | null = null
  let worstKey: number | null = null
  if (args.validityState === 'VALID') {
    for (const c of required) {
      if (c.state !== 'MEASURED' || c.deviation == null || c.tolerance == null || c.uncertainty == null) continue
      const ratio = exceedanceRatio(c.deviation, c.uncertainty, c.tolerance.plus, c.tolerance.minus)
      const key = ratio === null ? Infinity : ratio
      if (worstKey === null || key > worstKey) {
        worstKey = key
        worst = {
          characteristic_id: c.characteristic_id,
          bend_uid: c.bend_uid,
          kind: c.kind,
          deviation: c.deviation,
          uncertainty: c.uncertainty,
          decision: c.decision,
          exceedance_ratio: ratio === null ? null : round6(ratio),
        }
      }
    }
  }
  return {
    state,
    bends_total: args.bends.length,
    bends_formed: args.bends.filter(r => FORMED_LIKE_STATES.has(r.state)).length,
    bends_not_yet_formed: args.bends.filter(r => r.state === 'NOT_YET_FORMED').length,
    bends_unknown: args.bends.filter(r => r.state === 'UNKNOWN').length,
    characteristics_required: required.length,
    required_pass: counts.PASS,
    required_fail: counts.FAIL,
    required_review: counts.REVIEW,
    required_unknown: counts.UNKNOWN,
    unplanned_creases: creases,
    worst,
    reasons,
  }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

export type ValidationResult =
  | { ok: true; result: ShopfloorInspectionResult; errors: [] }
  | { ok: false; result: null; errors: string[] }

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
const close = (a: number | null | undefined, b: number | null | undefined, tol = 2e-6) =>
  a == null || b == null ? a == null && b == null : Math.abs(a - b) <= tol

/**
 * Field names of every contract model a result contains (pydantic
 * `extra="forbid"` at every level). `check-station-view-model.mjs` asserts
 * these lists equal the Python models' `model_fields`.
 */
export const CONTRACT_KEYS = {
  ShopfloorInspectionResultV1: ['schema', 'created_at', 'context', 'sku', 'part_name', 'recipe', 'scan', 'engine', 'authority', 'inspection_validity', 'stage_decision', 'summary', 'bends', 'general_measurements', 'unplanned_creases', 'viewer_assets', 'artifacts', 'warnings'],
  AuthorityV1: ['mode', 'production_disposition', 'note'],
  InspectionValidityV1: ['state', 'reasons', 'metrology_qualified', 'registration', 'coverage'],
  RegistrationV1: ['method', 'frame', 'raw_to_frame', 'unit_factor', 'datum_fitness', 'rmse_mm', 'pose_candidates_considered', 'pose_margin', 'pose_ambiguous', 'identity_ambiguous', 'in_plane_degenerate', 'abstained'],
  CoverageV1: ['surface_coverage_ratio', 'datum_coverage_ratio', 'scan_point_count'],
  StageDecisionV1: ['inspection_stage', 'declared_stage', 'declared_stage_source', 'bends_total', 'inferred_stage', 'formed_bend_uids', 'not_yet_formed_bend_uids', 'unknown_bend_uids', 'status', 'flags', 'decision', 'part_complete', 'stage_model_id', 'detail'],
  SummaryV1: ['state', 'bends_total', 'bends_formed', 'bends_not_yet_formed', 'bends_unknown', 'characteristics_required', 'required_pass', 'required_fail', 'required_review', 'required_unknown', 'unplanned_creases', 'worst', 'reasons'],
  WorstCharacteristicV1: ['characteristic_id', 'bend_uid', 'kind', 'deviation', 'uncertainty', 'decision', 'exceedance_ratio'],
  ViewerAssetsV1: ['frame', 'registered_scan', 'point_count', 'deviations', 'deviation_encoding', 'deviation_sign', 'deviation_reference_stage', 'heatmap_band_mm', 'heatmap_unavailable_reason', 'segment_labels', 'stage_mesh', 'final_mesh', 'bend_geometry'],
  ArtifactsV1: ['pdf', 'overview_png'],
  EngineInfoV1: ['engine_name', 'engine_version', 'config_sha256', 'bundle_id', 'compiled_part_sha256', 'algorithm_versions', 'duration_s', 'peak_rss_bytes'],
  ScanInfoV1: ['sha256', 'format', 'vertex_count', 'face_count'],
  ResultContextV1: ['job_id', 'station_id', 'scan_file_name', 'serial', 'scan_timestamp', 'received_at', 'bend_display_labels'],
  RecipeRefV1: ['recipe_id', 'version', 'recipe_sha256', 'content_sha256'],
  BendRowV1: ['bend_uid', 'label', 'sequence_index', 'machine_move_index', 'state', 'state_reasons', 'state_evidence', 'characteristics', 'display', 'diagnostics', 'notes'],
  BendDisplayV1: ['nominal_polyline_mm', 'measured_polyline_mm', 'label_anchor_mm', 'side_a_anchor_mm', 'side_b_anchor_mm'],
  BendStateEvidenceV1: ['nominal_included_deg', 'measured_included_deg', 'formed_fraction', 'parent_support_points', 'child_support_points', 'coverage_ratio', 'prior_formed_probability'],
  CharacteristicResultV1: ['characteristic_id', 'kind', 'bend_uid', 'side', 'stage_index', 'measurement_id', 'unit', 'required', 'state', 'decision', 'reason', 'expected', 'expected_source', 'actual', 'deviation', 'uncertainty', 'tolerance', 'boundary_kind', 'adjacent_bend_uid', 'stations', 'evidence'],
  CharacteristicEvidenceV1: ['method', 'support_points', 'coverage_ratio', 'fit_rms_mm', 'notes'],
  StationStatsV1: ['count', 'median', 'min', 'max'],
  ToleranceRef: ['plus', 'minus', 'origin', 'standard'],
  GeneralMeasurementRowV1: ['measurement_id', 'label', 'characteristic', 'nominal_start_mm', 'nominal_end_mm', 'measured_start_mm', 'measured_end_mm'],
  UnplannedCreaseV1: ['crease_id', 'reason', 'polyline_mm', 'included_deg', 'note'],
} as const satisfies Record<string, readonly string[]>

export type ContractModelName = keyof typeof CONTRACT_KEYS
const CONTRACT_KEY_SETS = Object.fromEntries(
  Object.entries(CONTRACT_KEYS).map(([k, v]) => [k, new Set<string>(v)]),
) as unknown as Record<ContractModelName, ReadonlySet<string>>

/** Refuse unknown fields (the contract forbids extras at every level). */
function checkKeys(obj: unknown, model: ContractModelName, where: string, errors: string[]): void {
  if (!isObj(obj)) return
  const allowed = CONTRACT_KEY_SETS[model]
  const extra = Object.keys(obj).filter(k => !allowed.has(k))
  if (extra.length) errors.push(`${where}: unknown field(s) ${extra.join(', ')}`)
}

/** A defaulted literal: absent is the default, present must equal it. */
function checkLiteral(obj: Record<string, unknown>, key: string, value: unknown, where: string, errors: string[]): void {
  if (key in obj && obj[key] !== value) errors.push(`${where}.${key} must be ${JSON.stringify(value)}`)
}

export const FRAME_CAD_MODEL = 'cad_model_mm'
export const DEVIATION_ENCODING = 'float32_le_nan_missing'
export const DEVIATION_SIGN = 'SIGNED_ALONG_STAGE_MODEL_OUTWARD_NORMAL'
export const ENGINE_NAME = 'sherman_production_engine'
export const UNIT_FACTORS: readonly number[] = [1, 25.4, 1000]
/** inspection_result.RIGID_TOLERANCE */
export const RIGID_TOLERANCE = 2e-5
const DECLARED_STAGE_SOURCES: ReadonlySet<string> = new Set(['SIDECAR', 'OPERATOR', 'API'])
const BOUNDARY_KIND_KINDS: ReadonlySet<string> = new Set(['FLANGE_LENGTH', 'BACKGAUGE_LENGTH'])

/**
 * inspection_result.check_rigid_transform: 4x4 row-major, last row
 * [0,0,0,1], R^T R = I within RIGID_TOLERANCE, det(R) > 0 (a mirror is
 * MIRRORED_PART_SUSPECTED and publishes no transform).
 */
export function rigidTransformError(m: unknown): string | null {
  if (!Array.isArray(m) || m.length !== 4 || m.some(r => !Array.isArray(r) || r.length !== 4 || r.some(v => !isNum(v)))) {
    return 'raw_to_frame must be a 4x4 matrix of finite numbers'
  }
  const rows = m as number[][]
  if (rows[3][0] !== 0 || rows[3][1] !== 0 || rows[3][2] !== 0 || rows[3][3] !== 1) return 'raw_to_frame last row must be [0, 0, 0, 1]'
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const dot = rows[0][i] * rows[0][j] + rows[1][i] * rows[1][j] + rows[2][i] * rows[2][j]
      if (Math.abs(dot - (i === j ? 1 : 0)) > RIGID_TOLERANCE) return 'raw_to_frame must be rigid (no scale or shear)'
    }
  }
  const r = rows
  const det =
    r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) -
    r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) +
    r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
  if (!(det > 0)) return 'raw_to_frame must be a proper rotation (a mirror is MIRRORED_PART_SUSPECTED)'
  return null
}

function checkTolerance(tol: unknown, where: string, errors: string[]): tol is ToleranceRef {
  if (tol == null) return false
  if (!isObj(tol) || !isNum(tol.plus) || !isNum(tol.minus) || tol.plus < 0 || tol.minus < 0 || !(tol.plus + tol.minus > 0)) {
    errors.push(`${where}: invalid tolerance band`)
    return false
  }
  return true
}

function validateCharacteristic(c: unknown, where: string, errors: string[]): c is CharacteristicResult {
  if (!isObj(c)) {
    errors.push(`${where}: not an object`)
    return false
  }
  const ch = c as unknown as CharacteristicResult
  const start = errors.length
  const expectedId = makeCharacteristicId(ch)
  if (expectedId === null || ch.characteristic_id !== expectedId) errors.push(`${where}: characteristic_id does not match its identity`)
  if (ch.unit !== unitForKind(ch.kind)) errors.push(`${where}: wrong unit for ${ch.kind}`)
  if (ch.side != null && ch.kind !== 'FLANGE_LENGTH' && ch.kind !== 'BACKGAUGE_LENGTH') errors.push(`${where}: only flange/backgauge carry a side`)
  checkKeys(c, 'CharacteristicResultV1', where, errors)
  checkKeys(ch.tolerance, 'ToleranceRef', `${where}.tolerance`, errors)
  checkKeys(ch.evidence, 'CharacteristicEvidenceV1', `${where}.evidence`, errors)
  if (ch.boundary_kind != null && !BOUNDARY_KIND_KINDS.has(ch.kind)) errors.push(`${where}: boundary_kind applies to flange/backgauge lengths only`)
  if (ch.adjacent_bend_uid != null && ch.boundary_kind !== 'ADJACENT_BEND') errors.push(`${where}: adjacent_bend_uid requires boundary_kind ADJACENT_BEND`)
  if (ch.stations != null) {
    const st = ch.stations as unknown
    checkKeys(st, 'StationStatsV1', `${where}.stations`, errors)
    if (!isObj(st) || !Number.isInteger(st.count) || (st.count as number) < 1 || !isNum(st.min) || !isNum(st.median) || !isNum(st.max) || !(st.min <= st.median && st.median <= st.max)) {
      errors.push(`${where}: station stats need count >= 1 and min <= median <= max`)
    }
  }
  if ((ch.expected == null) !== (ch.expected_source == null)) errors.push(`${where}: expected and expected_source go together`)
  if (typeof ch.required !== 'boolean') errors.push(`${where}: required must be boolean`)
  const hasTol = checkTolerance(ch.tolerance, where, errors)
  if (ch.state === 'MEASURED') {
    if (!isNum(ch.actual) || !isNum(ch.uncertainty) || !(ch.uncertainty > 0)) {
      errors.push(`${where}: MEASURED requires an actual and a strictly positive uncertainty`)
    } else if (ch.expected == null) {
      if (ch.deviation != null) errors.push(`${where}: deviation requires an expected value`)
      if (ch.decision !== 'UNKNOWN' || ch.reason !== 'NOMINAL_UNAVAILABLE') errors.push(`${where}: measured without nominal must be UNKNOWN (NOMINAL_UNAVAILABLE)`)
    } else {
      if (!isNum(ch.deviation) || !close(ch.deviation, ch.actual - ch.expected)) errors.push(`${where}: deviation must equal actual - expected`)
      if (hasTol && isNum(ch.deviation)) {
        const tol = ch.tolerance as ToleranceRef
        const derived = guardBandDecision(ch.deviation, ch.uncertainty, tol.plus, tol.minus)
        if (ch.decision !== derived) errors.push(`${where}: decision ${ch.decision} contradicts the guard band (${derived})`)
        if (ch.reason != null) errors.push(`${where}: a guard-band decision carries no reason`)
      } else if (!hasTol && (ch.decision !== 'UNKNOWN' || ch.reason !== 'NO_TOLERANCE')) {
        errors.push(`${where}: measured without tolerance must be UNKNOWN (NO_TOLERANCE)`)
      }
    }
  } else if (ch.state === 'NOT_MEASURED' || ch.state === 'NOT_APPLICABLE') {
    if ([ch.actual, ch.deviation, ch.uncertainty, ch.stations].some(v => v != null)) {
      errors.push(`${where}: ${ch.state} must not carry an actual, deviation, uncertainty or stations`)
    }
    if (ch.state === 'NOT_MEASURED') {
      if (ch.decision !== 'UNKNOWN') errors.push(`${where}: NOT_MEASURED decides UNKNOWN`)
      if (!ch.reason || !UNKNOWN_REASON_SET.has(ch.reason)) errors.push(`${where}: NOT_MEASURED needs an UNKNOWN reason`)
    } else {
      if (ch.decision !== 'NOT_APPLICABLE') errors.push(`${where}: NOT_APPLICABLE decides NOT_APPLICABLE`)
      const allowed = NA_REASONS_BY_KIND[ch.kind]
      if (!allowed || !ch.reason || !allowed.has(ch.reason) || !NA_REASON_SET.has(ch.reason)) errors.push(`${where}: invalid NOT_APPLICABLE reason`)
      if (ch.required) errors.push(`${where}: a NOT_APPLICABLE characteristic is never required`)
    }
  } else {
    errors.push(`${where}: unknown measurement state`)
  }
  if (ch.state !== 'NOT_APPLICABLE') {
    if (ch.kind === 'FLANGE_LENGTH' || ch.kind === 'INCLUDED_ANGLE') {
      if (!ch.required || !hasTol) errors.push(`${where}: an applicable ${ch.kind} is required and toleranced`)
    } else if (ch.kind === 'BEND_LINE_POSITION' || ch.kind === 'BACKGAUGE_LENGTH') {
      if (ch.required !== hasTol) errors.push(`${where}: ${ch.kind} is required exactly when a tolerance exists`)
    } else if (ch.required && !hasTol) {
      errors.push(`${where}: a required GENERAL_MEASUREMENT is toleranced`)
    }
  }
  return errors.length === start
}

function validateBendRow(row: unknown, where: string, errors: string[]): void {
  if (!isObj(row)) {
    errors.push(`${where}: not an object`)
    return
  }
  const r = row as unknown as BendRow
  checkKeys(row, 'BendRowV1', where, errors)
  checkKeys(r.state_evidence, 'BendStateEvidenceV1', `${where}.state_evidence`, errors)
  checkKeys(r.display, 'BendDisplayV1', `${where}.display`, errors)
  if (typeof r.bend_uid !== 'string' || !BEND_UID_RE.test(r.bend_uid)) errors.push(`${where}: invalid bend_uid`)
  if (typeof r.label !== 'string' || r.label.length < 1 || r.label.length > 32) errors.push(`${where}: label must be 1-32 characters`)
  if (r.machine_move_index != null && !(Number.isInteger(r.machine_move_index) && r.machine_move_index >= 1)) errors.push(`${where}: machine_move_index must be >= 1`)
  if (!(BEND_STATES as readonly string[]).includes(r.state)) errors.push(`${where}: invalid bend state`)
  const chars = Array.isArray(r.characteristics) ? r.characteristics : []
  if (chars.length !== 4 && chars.length !== 5) errors.push(`${where}: a bend row has 4 characteristics plus an optional backgauge`)
  chars.forEach((c, i) => validateCharacteristic(c, `${where}.characteristics[${i}]`, errors))
  BEND_CHARACTERISTIC_ORDER.forEach(([kind, side], i) => {
    const c = chars[i]
    if (c && (c.kind !== kind || (c.side ?? null) !== side)) errors.push(`${where}: characteristics must follow FLANGE A, FLANGE B, ANGLE, POSITION`)
  })
  if (chars.length === 5 && chars[4]?.kind !== 'BACKGAUGE_LENGTH') errors.push(`${where}: the fifth characteristic must be BACKGAUGE_LENGTH`)
  if (chars.some(c => c?.bend_uid !== r.bend_uid)) errors.push(`${where}: characteristics must belong to their bend row`)
  for (const c of chars) {
    if (c?.reason === 'BEND_NOT_YET_FORMED' && r.state !== 'NOT_YET_FORMED') errors.push(`${where}: BEND_NOT_YET_FORMED only on a NOT_YET_FORMED bend`)
  }
  if (r.state === 'NOT_YET_FORMED') {
    if (chars.some(c => c?.state !== 'NOT_APPLICABLE')) errors.push(`${where}: a NOT_YET_FORMED bend has only NOT_APPLICABLE characteristics`)
    if (chars.slice(2, 4).some(c => c?.reason !== 'BEND_NOT_YET_FORMED')) errors.push(`${where}: angle/position of a NOT_YET_FORMED bend are BEND_NOT_YET_FORMED`)
  } else if (r.state === 'UNKNOWN') {
    const reasons = Array.isArray(r.state_reasons) ? r.state_reasons : []
    if (!reasons.length || reasons.some(x => !UNKNOWN_REASON_SET.has(x))) errors.push(`${where}: an UNKNOWN bend needs UNKNOWN reasons`)
    if (chars.some(c => c?.state === 'MEASURED')) errors.push(`${where}: an UNKNOWN bend has no MEASURED characteristic`)
  } else if (r.state === 'PARTIALLY_FORMED') {
    if (chars[2]?.decision === 'PASS') errors.push(`${where}: a PARTIALLY_FORMED bend cannot have a PASS angle`)
  }
  const display = r.display
  if (!isObj(display) || !Array.isArray(display.nominal_polyline_mm) || display.nominal_polyline_mm.length < 2 || !Array.isArray(display.label_anchor_mm)) {
    errors.push(`${where}: invalid display geometry`)
  } else if ((chars[3]?.state === 'MEASURED') !== (display.measured_polyline_mm != null)) {
    errors.push(`${where}: measured_polyline_mm exists exactly when BEND_LINE_POSITION is MEASURED`)
  }
}

function sameSummary(a: Summary, b: Summary): string[] {
  const diffs: string[] = []
  const keys: (keyof Summary)[] = [
    'state', 'bends_total', 'bends_formed', 'bends_not_yet_formed', 'bends_unknown', 'characteristics_required',
    'required_pass', 'required_fail', 'required_review', 'required_unknown', 'unplanned_creases',
  ]
  for (const k of keys) if (a[k] !== b[k]) diffs.push(`summary.${k}`)
  if (JSON.stringify(a.reasons ?? []) !== JSON.stringify(b.reasons)) diffs.push('summary.reasons')
  const wa = a.worst
  const wb = b.worst
  if ((wa == null) !== (wb == null)) diffs.push('summary.worst')
  else if (wa && wb) {
    if (wa.characteristic_id !== wb.characteristic_id || wa.decision !== wb.decision || wa.kind !== wb.kind || (wa.bend_uid ?? null) !== wb.bend_uid) diffs.push('summary.worst')
    if (!close(wa.deviation, wb.deviation) || !close(wa.uncertainty, wb.uncertainty) || !close(wa.exceedance_ratio, wb.exceedance_ratio)) diffs.push('summary.worst values')
  }
  return diffs
}

/**
 * Validate a raw result document with the contract's honesty rules.
 * Returns the typed result only when every rule holds.
 */
export function validateResult(raw: unknown): ValidationResult {
  const errors: string[] = []
  if (!isObj(raw)) return { ok: false, result: null, errors: ['result is not an object'] }
  const r = raw as unknown as ShopfloorInspectionResult
  if (r.schema !== SHOPFLOOR_RESULT_SCHEMA) errors.push('schema tag mismatch')
  const authority = (r as { authority?: unknown }).authority
  if (!isObj(authority) || authority.mode !== AUTHORITY_MODE || authority.production_disposition !== PRODUCTION_DISPOSITION) {
    errors.push('authority literals altered (must be OPERATOR_ASSIST / NOT_AUTHORIZED)')
  }
  checkKeys(raw, 'ShopfloorInspectionResultV1', 'result', errors)
  checkKeys(authority, 'AuthorityV1', 'authority', errors)
  for (const [key, model] of [
    ['recipe', 'RecipeRefV1'],
    ['scan', 'ScanInfoV1'],
    ['engine', 'EngineInfoV1'],
    ['context', 'ResultContextV1'],
    ['artifacts', 'ArtifactsV1'],
  ] as const) {
    const sub = (raw as Record<string, unknown>)[key]
    // context / artifacts have defaults in the model, but serialised results
    // always carry them and the screen reads them, so all five are required.
    if (!isObj(sub)) errors.push(`${key} must be an object`)
    checkKeys(sub, model, key, errors)
  }
  const engine = (raw as Record<string, unknown>).engine
  if (isObj(engine)) {
    checkLiteral(engine, 'engine_name', ENGINE_NAME, 'engine', errors)
    if (typeof engine.engine_version !== 'string' || !engine.engine_version) errors.push('engine.engine_version is required')
  }
  if (!Array.isArray(r.bends)) return { ok: false, result: null, errors: [...errors, 'bends missing'] }
  const gms = Array.isArray(r.general_measurements) ? r.general_measurements : []
  const creases = Array.isArray(r.unplanned_creases) ? r.unplanned_creases : []
  if (!isObj(r.recipe) || r.recipe.recipe_id !== `${r.sku}@v${r.recipe.version}`) errors.push("recipe.recipe_id must be '<sku>@v<version>'")

  r.bends.forEach((row, i) => validateBendRow(row, `bends[${i}]`, errors))
  gms.forEach((gm, i) => {
    checkKeys(gm, 'GeneralMeasurementRowV1', `general_measurements[${i}]`, errors)
    if (!isObj(gm) || !validateCharacteristic(gm.characteristic, `general_measurements[${i}].characteristic`, errors)) return
    const c = gm.characteristic
    if (c.kind !== 'GENERAL_MEASUREMENT' || c.measurement_id !== gm.measurement_id) errors.push(`general_measurements[${i}]: must wrap its GENERAL_MEASUREMENT`)
    const measured = gm.measured_start_mm != null && gm.measured_end_mm != null
    if ((gm.measured_start_mm == null) !== (gm.measured_end_mm == null) || measured !== (c.state === 'MEASURED')) {
      errors.push(`general_measurements[${i}]: measured anchors exist exactly when MEASURED`)
    }
  })
  creases.forEach((c, i) => checkKeys(c, 'UnplannedCreaseV1', `unplanned_creases[${i}]`, errors))
  const indices = r.bends.map(b => b?.sequence_index)
  if (indices.some((v, i) => v !== i + 1)) errors.push('bend rows must be sorted by a contiguous sequence_index 1..N')
  const uids = r.bends.map(b => b?.bend_uid)
  if (new Set(uids).size !== uids.length) errors.push('bend rows must not repeat a bend_uid')
  for (const row of r.bends) {
    for (const c of row?.characteristics ?? []) {
      if (c?.adjacent_bend_uid != null && !uids.includes(c.adjacent_bend_uid)) errors.push('adjacent_bend_uid must name a bend of this result')
    }
  }
  if (errors.length) return { ok: false, result: null, errors }

  const validity = r.inspection_validity
  if (!isObj(validity) || !isObj(validity.registration) || !isObj(r.stage_decision) || !isObj(r.summary) || !isObj(r.viewer_assets)) {
    return { ok: false, result: null, errors: ['validity, stage_decision, summary or viewer_assets missing'] }
  }
  const reg = validity.registration
  const va = r.viewer_assets
  const sd = r.stage_decision
  checkKeys(validity, 'InspectionValidityV1', 'inspection_validity', errors)
  checkKeys(reg, 'RegistrationV1', 'registration', errors)
  checkKeys(validity.coverage, 'CoverageV1', 'inspection_validity.coverage', errors)
  checkKeys(va, 'ViewerAssetsV1', 'viewer_assets', errors)
  checkKeys(sd, 'StageDecisionV1', 'stage_decision', errors)
  checkKeys(r.summary, 'SummaryV1', 'summary', errors)
  checkKeys(r.summary.worst, 'WorstCharacteristicV1', 'summary.worst', errors)
  const validityRec = validity as unknown as Record<string, unknown>
  const regRec = reg as unknown as Record<string, unknown>
  const vaRec = va as unknown as Record<string, unknown>
  // Defaulted literals: the legend's sign convention and the 3D frame depend on these.
  checkLiteral(validityRec, 'metrology_qualified', false, 'inspection_validity', errors)
  checkLiteral(regRec, 'frame', FRAME_CAD_MODEL, 'registration', errors)
  checkLiteral(vaRec, 'frame', FRAME_CAD_MODEL, 'viewer_assets', errors)
  checkLiteral(vaRec, 'deviation_encoding', DEVIATION_ENCODING, 'viewer_assets', errors)
  checkLiteral(vaRec, 'deviation_sign', DEVIATION_SIGN, 'viewer_assets', errors)
  if ('unit_factor' in regRec && !UNIT_FACTORS.includes(regRec.unit_factor as number)) errors.push('registration.unit_factor must be 1, 25.4 or 1000')
  if (reg.raw_to_frame != null) {
    const rigid = rigidTransformError(reg.raw_to_frame)
    if (rigid) errors.push(rigid)
  }
  if ((sd.declared_stage == null) !== (sd.declared_stage_source == null)) errors.push('declared_stage and declared_stage_source go together')
  if (sd.declared_stage != null) {
    if (!Number.isInteger(sd.declared_stage) || sd.declared_stage < 0) errors.push('declared_stage must be a non-negative integer')
    else if (sd.declared_stage > sd.bends_total) errors.push('declared_stage cannot exceed bends_total')
    if (!DECLARED_STAGE_SOURCES.has(sd.declared_stage_source as string)) errors.push('invalid declared_stage_source')
  }
  if (validity.state !== 'VALID' && !(validity.reasons ?? []).length) errors.push('a non-VALID inspection must state reasons')
  if (validity.state === 'VALID' && (reg.abstained || reg.raw_to_frame == null)) errors.push('a VALID inspection needs a non-abstained registration with a transform')
  if (reg.abstained && reg.raw_to_frame != null) errors.push('an abstained registration publishes no transform')
  if (!(isNum(va.heatmap_band_mm) && va.heatmap_band_mm > 0)) errors.push('viewer_assets.heatmap_band_mm must be positive')
  if (va.deviations != null) {
    if (va.registered_scan == null || va.point_count == null || va.deviation_reference_stage == null) errors.push('deviations need registered_scan, point_count and a reference stage')
    if (va.heatmap_unavailable_reason != null) errors.push('a published heatmap has no unavailable reason')
  } else if (!va.heatmap_unavailable_reason) {
    errors.push('without deviations, heatmap_unavailable_reason is required')
  }
  if (sd.stage_model_id != null && !STAGE_MODEL_ID_RE.test(sd.stage_model_id)) errors.push('invalid stage_model_id')
  if (!(STAGE_STATUSES as readonly string[]).includes(sd.status)) errors.push('invalid stage status')
  if (!(SUMMARY_STATES as readonly string[]).includes(r.summary.state)) errors.push('invalid summary state')

  const all = iterCharacteristics(r.bends, gms)
  if (all.some(c => c.reason === 'NOT_APPLICABLE_AT_STAGE') && !r.bends.some(b => b.state === 'NOT_YET_FORMED')) {
    errors.push('NOT_APPLICABLE_AT_STAGE needs a NOT_YET_FORMED bend')
  }
  {
    const stateOf = new Map(r.bends.map(b => [b.bend_uid, b.state]))
    for (const b of r.bends) {
      for (const c of b.characteristics) {
        if (c.kind === 'FLANGE_LENGTH' && c.reason === 'NOT_APPLICABLE_AT_STAGE'
          && (c.adjacent_bend_uid == null || stateOf.get(c.adjacent_bend_uid) !== 'NOT_YET_FORMED')) {
          errors.push('a FLANGE_LENGTH is NOT_APPLICABLE_AT_STAGE only when its adjacent bend is NOT_YET_FORMED')
        }
      }
    }
  }
  if (validity.state !== 'VALID') {
    if (all.some(c => c.state === 'MEASURED')) errors.push('an invalid inspection has no MEASURED characteristic')
    if (r.bends.some(b => b.state !== 'UNKNOWN')) errors.push('an invalid inspection cannot claim a bend state')
    if (va.deviations != null) errors.push('an invalid inspection publishes no heatmap')
    if (sd.stage_model_id != null) errors.push('an invalid inspection decodes no stage model')
  }
  if (va.deviations != null && (sd.stage_model_id == null || va.deviation_reference_stage !== sd.stage_model_id)) {
    errors.push('the heatmap reference stage must equal stage_decision.stage_model_id')
  }
  if (reg.identity_ambiguous && !reg.pose_ambiguous) errors.push('identity_ambiguous needs pose_ambiguous')
  if (reg.pose_ambiguous && !reg.identity_ambiguous) {
    if (all.some(c => c.state === 'MEASURED')) errors.push('an ambiguous pose measures nothing')
    for (const b of r.bends) {
      if (b.state === 'NOT_YET_FORMED') continue
      if (b.state !== 'UNKNOWN' || !b.state_reasons.includes('AMBIGUOUS_POSE')) errors.push('an ambiguous pose leaves bends UNKNOWN (AMBIGUOUS_POSE) or NOT_YET_FORMED')
    }
    if (va.deviations != null) errors.push('an ambiguous pose publishes no heatmap')
  }
  if (reg.identity_ambiguous) {
    // amendment ENGINE_IO-A2: only readings identical under every tied twin; never a position, heatmap or stage model
    for (const b of r.bends) {
      if (b.state === 'UNKNOWN' && !b.state_reasons.includes('AMBIGUOUS_POSE')) errors.push('an identity-ambiguous result leaves unknown bends UNKNOWN (AMBIGUOUS_POSE)')
      if (b.characteristics[3]?.state === 'MEASURED') errors.push('an identity-ambiguous result measures no bend-line position')
    }
    if (va.deviations != null) errors.push('an ambiguous pose publishes no heatmap')
    if (sd.stage_model_id != null) errors.push('an identity-ambiguous result decodes no stage model')
  }
  if (reg.in_plane_degenerate) {
    for (const b of r.bends) {
      const pos = b.characteristics[3]
      if (pos.state === 'MEASURED') errors.push('a degenerate in-plane pose cannot measure bend-line position')
      else if (b.state !== 'NOT_YET_FORMED' && b.state !== 'UNKNOWN' && (pos.state !== 'NOT_MEASURED' || pos.reason !== 'REGISTRATION_UNTRUSTED')) {
        errors.push('a degenerate in-plane pose reports bend-line position as UNKNOWN(REGISTRATION_UNTRUSTED)')
      }
    }
    if (va.deviations != null) errors.push('a degenerate in-plane pose publishes no heatmap')
  }

  const derivedStage = deriveStageDecision({
    inspectionStage: sd.inspection_stage,
    declaredStage: sd.declared_stage,
    bends: r.bends,
    poseAmbiguous: !!reg.pose_ambiguous,
    validityState: validity.state,
  })
  for (const [key, value] of Object.entries(derivedStage)) {
    const actual = (sd as unknown as Record<string, unknown>)[key]
    if (JSON.stringify(actual ?? null) !== JSON.stringify(value)) errors.push(`stage_decision.${key} must equal the derived value`)
  }
  if (sd.stage_model_id != null && sd.inferred_stage != null) {
    const k = Number(sd.stage_model_id.slice(1).split('-')[0])
    if (k !== sd.inferred_stage) errors.push('stage_model_id must name a stage with inferred_stage formed bends')
    if (!sd.flags.includes('OUT_OF_ORDER_SUSPECTED') && sd.stage_model_id !== `S${sd.inferred_stage}`) errors.push('an order-consistent decode uses S<inferred_stage>')
  }
  const derivedSummary = deriveSummary({
    validityState: validity.state,
    bends: r.bends,
    generalMeasurements: gms,
    stageDecision: sd.decision,
    stageStatus: sd.status,
    unplannedCreases: creases.length,
  })
  errors.push(...sameSummary(r.summary, derivedSummary).map(k => `${k} must equal derive_summary (expected ${derivedSummary.state})`))

  if (errors.length) return { ok: false, result: null, errors }
  return { ok: true, result: r, errors: [] }
}

/**
 * Cross-check a validated result against the envelope it arrived in (the
 * feed's job summary, or the history job that was opened). A result whose
 * own context names another job or station, or whose summary differs from
 * the job summary, is refused: the screen must never show another job's
 * verdict under this station's header. Null context fields are not compared.
 */
export function checkEnvelope(
  result: ShopfloorInspectionResult,
  envelope: { jobId: string; stationId: string; summaryState?: string | null; feedStationId?: string | null },
): string[] {
  const errors: string[] = []
  const ctx = result.context
  if (ctx.job_id != null && ctx.job_id !== envelope.jobId) errors.push(`result.context.job_id ${ctx.job_id} is not job ${envelope.jobId}`)
  if (ctx.station_id != null && ctx.station_id !== envelope.stationId) errors.push(`result.context.station_id ${ctx.station_id} is not station ${envelope.stationId}`)
  if (envelope.feedStationId != null && envelope.feedStationId !== envelope.stationId) errors.push(`feed is for station ${envelope.feedStationId}, not ${envelope.stationId}`)
  if (envelope.summaryState != null && envelope.summaryState !== result.summary.state) {
    errors.push(`job summary_state ${envelope.summaryState} differs from result summary ${result.summary.state}`)
  }
  return errors
}

// ---------------------------------------------------------------------------
// Formatting and display styles
// ---------------------------------------------------------------------------

export const MINUS = '−'

/**
 * Number of decimals for a quantity: 0.1 mm / 0.1 deg, refined to 0.01 when
 * the tolerance is tight enough that 0.1 would be more than a fifth of it.
 */
export function quantityDecimals(tolerance: Pick<ToleranceRef, 'plus' | 'minus'> | null | undefined): number {
  if (!tolerance) return 1
  const sides = [tolerance.plus, tolerance.minus].filter(v => v > 0)
  const narrow = sides.length ? Math.min(...sides) : 0
  return narrow > 0 && narrow / 5 < 0.1 - 1e-12 ? 2 : 1
}

/**
 * Format a value. `signed` gives an explicit sign on a non-zero value (used
 * for Δ); zero after rounding prints without a sign. Uses a true minus sign.
 */
export function formatQuantity(
  value: number | null | undefined,
  unit: 'mm' | 'deg',
  opts: { signed?: boolean; decimals?: number; withUnit?: boolean } = {},
): string {
  if (value == null || !Number.isFinite(value)) return '—'
  const decimals = opts.decimals ?? 1
  const factor = 10 ** decimals
  const rounded = Math.round(Math.abs(value) * factor) / factor
  const body = rounded.toFixed(decimals)
  let sign = ''
  if (rounded !== 0) sign = value < 0 ? MINUS : opts.signed ? '+' : ''
  // Angles carry a degree sign by default; mm values only when asked.
  const suffix = opts.withUnit === false ? '' : unit === 'deg' ? '°' : opts.withUnit ? ' mm' : ''
  return `${sign}${body}${suffix}`
}

export function formatTolerance(tol: Pick<ToleranceRef, 'plus' | 'minus'> | null | undefined, unit: 'mm' | 'deg', decimals?: number): string {
  if (!tol) return '—'
  const d = decimals ?? quantityDecimals(tol)
  const u = unit === 'deg' ? '°' : ''
  if (Math.abs(tol.plus - tol.minus) < 1e-9) return `±${tol.plus.toFixed(d)}${u}`
  return `+${tol.plus.toFixed(d)}/${MINUS}${tol.minus.toFixed(d)}${u}`
}

export interface ConformanceStyle {
  tone: Tone
  color: string
  /** Short status label (decision name, "Not measured" or "—"). */
  label: string
  /** Reason copy for UNKNOWN / NOT_APPLICABLE, else empty. */
  reason: string
  /** True only for a MEASURED value with a guard-band decision. */
  decided: boolean
}

/**
 * Display style of one characteristic. Only a MEASURED PASS/FAIL/REVIEW gets
 * a decision colour; NOT_MEASURED is neutral grey "Not measured" with the
 * reason; NOT_APPLICABLE is "—" (muted); a MEASURED value without nominal or
 * tolerance is "Undetermined" grey with the reason.
 */
export function conformanceStyle(c: CharacteristicResult, lang: Lang): ConformanceStyle {
  if (c.state === 'MEASURED' && (c.decision === 'PASS' || c.decision === 'FAIL' || c.decision === 'REVIEW')) {
    const tok = DISPLAY_TOKENS[c.decision]
    return { tone: tok.tone, color: tok.color, label: tok[lang], reason: '', decided: true }
  }
  if (c.state === 'MEASURED') {
    const tok = DISPLAY_TOKENS.UNKNOWN
    return { tone: 'unknown', color: tok.color, label: tok[lang], reason: c.reason ? reasonText(c.reason, lang) : '', decided: false }
  }
  if (c.state === 'NOT_MEASURED') {
    return {
      tone: 'unknown',
      color: DISPLAY_TOKENS.UNKNOWN.color,
      label: t('notMeasured', lang),
      reason: c.reason ? reasonText(c.reason, lang) : '',
      decided: false,
    }
  }
  return {
    tone: 'inactive',
    color: DISPLAY_TOKENS.NOT_APPLICABLE.color,
    label: '—',
    reason: c.reason ? reasonText(c.reason, lang) : '',
    decided: false,
  }
}

export interface CellView {
  characteristic: CharacteristicResult
  style: ConformanceStyle
  decimals: number
  expectedText: string
  /** Empty-dash unless MEASURED: a nominal is never shown as actual. */
  actualText: string
  deltaText: string
  tolText: string
}

export function cellView(c: CharacteristicResult, lang: Lang): CellView {
  const decimals = quantityDecimals(c.tolerance)
  const measured = c.state === 'MEASURED'
  return {
    characteristic: c,
    style: conformanceStyle(c, lang),
    decimals,
    expectedText: formatQuantity(c.expected, c.unit, { decimals }),
    actualText: measured ? formatQuantity(c.actual, c.unit, { decimals }) : '—',
    deltaText: measured && c.deviation != null ? formatQuantity(c.deviation, c.unit, { decimals, signed: true }) : '—',
    tolText: formatTolerance(c.tolerance, c.unit, decimals),
  }
}

const DECISION_SEVERITY: Record<Decision, number> = { FAIL: 4, REVIEW: 3, UNKNOWN: 2, PASS: 1, NOT_APPLICABLE: 0 }

export interface BendViewRow {
  bend: BendRow
  bendUid: string
  shortUid: string
  label: string
  sequenceIndex: number
  machineMoveIndex: number | null
  /** Position in a run of consecutive bends that share a machine move (מהלך). */
  moveRun: { first: boolean; last: boolean; size: number } | null
  state: BendState
  stateLabel: string
  flangeA: CellView
  flangeB: CellView
  angle: CellView
  position: CellView
  backgauge: CellView | null
  /** Worst decision over the row's required characteristics. */
  worstDecision: Decision
  tone: Tone
  color: string
  /** Needs the operator's attention (FAIL, REVIEW, required UNKNOWN, odd state). */
  attention: boolean
}

/** `bu_12380e8...` -> `1238`. */
export function shortUid(uid: string): string {
  return uid.startsWith('bu_') ? uid.slice(3, 7) : uid.slice(0, 4)
}

export function rowTone(row: Pick<BendRow, 'state' | 'characteristics'>): { worst: Decision; tone: Tone; color: string } {
  let worst: Decision = 'NOT_APPLICABLE'
  for (const c of row.characteristics) {
    const d: Decision = c.required ? c.decision : c.state === 'MEASURED' && c.decision !== 'UNKNOWN' ? c.decision : 'NOT_APPLICABLE'
    if (DECISION_SEVERITY[d] > DECISION_SEVERITY[worst]) worst = d
  }
  if (row.state === 'NOT_YET_FORMED') return { worst: 'NOT_APPLICABLE', tone: 'inactive', color: DISPLAY_TOKENS.NOT_YET_FORMED.color }
  if (row.state === 'UNKNOWN') return { worst: worst === 'NOT_APPLICABLE' ? 'UNKNOWN' : worst, tone: 'unknown', color: DISPLAY_TOKENS.UNKNOWN.color }
  if ((row.state === 'PARTIALLY_FORMED' || row.state === 'UNEXPECTED_FORMED') && DECISION_SEVERITY[worst] < DECISION_SEVERITY.REVIEW) {
    return { worst, tone: 'review', color: DISPLAY_TOKENS.REVIEW.color }
  }
  if (worst === 'NOT_APPLICABLE') return { worst, tone: 'inactive', color: DISPLAY_TOKENS.NOT_APPLICABLE.color }
  const tok = DISPLAY_TOKENS[worst]
  return { worst, tone: tok.tone, color: tok.color }
}

/** Per-bend rows in sequence order, flange A / flange B / angle first. */
/**
 * The bend's name on screen: the studio (CAD) label carried in
 * `context.bend_display_labels`, else the recipe's sequence label. The PDF uses
 * the same rule (`reporting/labels.py` `bend_display_label`).
 */
export function bendDisplayLabel(result: ShopfloorInspectionResult, bend: BendRow): string {
  const label = result.context?.bend_display_labels?.[bend.bend_uid]
  return typeof label === 'string' && label.length > 0 ? label : bend.label
}

export function toRows(result: ShopfloorInspectionResult, lang: Lang): BendViewRow[] {
  const bends = [...result.bends].sort((a, b) => a.sequence_index - b.sequence_index)
  return bends.map((bend, i) => {
    const [fa, fb, ang, pos, bg] = bend.characteristics
    const move = bend.machine_move_index
    let moveRun: BendViewRow['moveRun'] = null
    if (move != null) {
      const prevSame = i > 0 && bends[i - 1].machine_move_index === move
      const nextSame = i < bends.length - 1 && bends[i + 1].machine_move_index === move
      let size = 0
      for (const b of bends) if (b.machine_move_index === move) size++
      moveRun = { first: !prevSame, last: !nextSame, size }
    }
    const tone = rowTone(bend)
    const attention =
      tone.worst === 'FAIL' || tone.worst === 'REVIEW' || tone.worst === 'UNKNOWN' ||
      bend.state === 'UNKNOWN' || bend.state === 'PARTIALLY_FORMED' || bend.state === 'UNEXPECTED_FORMED'
    return {
      bend,
      bendUid: bend.bend_uid,
      shortUid: shortUid(bend.bend_uid),
      label: bendDisplayLabel(result, bend),
      sequenceIndex: bend.sequence_index,
      machineMoveIndex: move,
      moveRun,
      state: bend.state,
      stateLabel: DISPLAY_TOKENS[bend.state === 'UNKNOWN' ? 'UNKNOWN' : bend.state][lang],
      flangeA: cellView(fa, lang),
      flangeB: cellView(fb, lang),
      angle: cellView(ang, lang),
      position: cellView(pos, lang),
      backgauge: bg ? cellView(bg, lang) : null,
      worstDecision: tone.worst,
      tone: tone.tone,
      color: tone.color,
      attention,
    }
  })
}

export interface BannerView {
  state: SummaryState
  tone: Tone
  color: string
  title: string
  required: number
  pass: number
  fail: number
  review: number
  unknown: number
  /** Short human reason lines (validity, stage) for non-green states. */
  details: string[]
}

export function summaryBanner(result: ShopfloorInspectionResult, lang: Lang, texts: {
  validity: (code: string) => string
  stage: (code: string) => string
}): BannerView {
  const s = result.summary
  const tok = DISPLAY_TOKENS[s.state]
  const details: string[] = []
  if (s.state === 'INVALID_INSPECTION') {
    details.push(texts.validity(result.inspection_validity.state))
  } else if (result.stage_decision.decision !== 'PASS') {
    details.push(texts.stage(result.stage_decision.status))
  }
  if (s.unplanned_creases) details.push(t('unplannedCreases', lang))
  return {
    state: s.state,
    tone: tok.tone,
    color: tok.color,
    title: tok[lang],
    required: s.characteristics_required,
    pass: s.required_pass,
    fail: s.required_fail,
    review: s.required_review,
    unknown: s.required_unknown,
    details,
  }
}

/**
 * Bends ordered for the "worst first" jump: FAIL, then REVIEW, then required
 * UNKNOWN, then the rest by exceedance ratio (largest first). NOT_YET_FORMED
 * bends are excluded (an unformed bend is never a defect).
 */
export function worstOrder(result: ShopfloorInspectionResult): string[] {
  const scored = result.bends
    .filter(b => b.state !== 'NOT_YET_FORMED')
    .map(b => {
      let severity = 0
      let ratio = -1
      for (const c of b.characteristics) {
        if (!c.required && c.state !== 'MEASURED') continue
        const d = c.required ? c.decision : 'PASS'
        severity = Math.max(severity, d === 'FAIL' ? 3 : d === 'REVIEW' ? 2 : d === 'UNKNOWN' ? 1 : 0)
        if (c.state === 'MEASURED' && c.deviation != null && c.uncertainty != null && c.tolerance) {
          const r = exceedanceRatio(c.deviation, c.uncertainty, c.tolerance.plus, c.tolerance.minus)
          ratio = Math.max(ratio, r === null ? Infinity : r)
        }
      }
      if (b.state === 'UNKNOWN') severity = Math.max(severity, 1)
      if (b.state === 'PARTIALLY_FORMED' || b.state === 'UNEXPECTED_FORMED') severity = Math.max(severity, 2)
      return { uid: b.bend_uid, seq: b.sequence_index, severity, ratio }
    })
  scored.sort((a, b) => b.severity - a.severity || b.ratio - a.ratio || a.seq - b.seq)
  return scored.map(s => s.uid)
}

/**
 * The bend to focus for "worst first": the most severe bend of `worstOrder`
 * (a FAIL outranks a REVIEW with a larger ratio). Null when nothing is formed.
 */
export function worstFocus(result: ShopfloorInspectionResult): string | null {
  return worstOrder(result)[0] ?? null
}

export interface StageTile {
  bendUid: string
  label: string
  sequenceIndex: number
  machineMoveIndex: number | null
  state: BendState
  tone: Tone
  color: string
}

export interface StageStripView {
  inspectionStage: 'FINAL' | 'IN_PROCESS'
  inferredStage: number | null
  total: number
  status: StageStatus
  decision: StageDecisionValue
  partComplete: boolean | null
  tiles: StageTile[]
}

export function stageStrip(result: ShopfloorInspectionResult): StageStripView {
  const sd = result.stage_decision
  const tiles = [...result.bends]
    .sort((a, b) => a.sequence_index - b.sequence_index)
    .map(b => {
      const tone = rowTone(b)
      return {
        bendUid: b.bend_uid,
        label: bendDisplayLabel(result, b),
        sequenceIndex: b.sequence_index,
        machineMoveIndex: b.machine_move_index,
        state: b.state,
        tone: tone.tone,
        color: tone.color,
      }
    })
  return {
    inspectionStage: sd.inspection_stage,
    inferredStage: sd.inferred_stage,
    total: sd.bends_total,
    status: sd.status,
    decision: sd.decision,
    partComplete: sd.part_complete,
    tiles,
  }
}

/** Whether the result is a mid-sequence (incomplete part) scan. */
export function isIncompletePart(result: ShopfloorInspectionResult): boolean {
  return result.bends.some(b => b.state === 'NOT_YET_FORMED')
}

// ---------------------------------------------------------------------------
// Bender view (station v2): verdict words, tolerance bars, tag anchors
// ---------------------------------------------------------------------------
//
// Display-only mappings of what the engine already decided. Nothing here
// re-decides a characteristic: a verdict is the row tone (`rowTone`, the
// worst required decision), and a tolerance bar only places the stored
// deviation, uncertainty and tolerance on a scale.

/**
 * The bender's word for a bend (plan mapping: PASS = OK, REVIEW = CHECK,
 * FAIL = FIX). UNSURE = a required value the scan could not decide (grey);
 * NOT_BENT = NOT_YET_FORMED (grey, never a defect); NONE = nothing required.
 */
export type BendVerdict = 'OK' | 'CHECK' | 'FIX' | 'UNSURE' | 'NOT_BENT' | 'NONE'

export function bendVerdict(row: Pick<BendViewRow, 'tone' | 'state'>): BendVerdict {
  if (row.state === 'NOT_YET_FORMED') return 'NOT_BENT'
  switch (row.tone) {
    case 'fail': return 'FIX'
    case 'review': return 'CHECK'
    case 'pass': return 'OK'
    case 'unknown': return 'UNSURE'
    default: return 'NONE'
  }
}

/** Verdict word of one characteristic (its own decision, MEASURED or not). */
export function characteristicVerdict(c: Pick<CharacteristicResult, 'state' | 'decision'>): BendVerdict {
  if (c.state === 'NOT_APPLICABLE') return 'NONE'
  if (c.state !== 'MEASURED') return 'UNSURE'
  return c.decision === 'FAIL' ? 'FIX' : c.decision === 'REVIEW' ? 'CHECK' : c.decision === 'PASS' ? 'OK' : 'UNSURE'
}

/** Part verdict for the top banner, from the engine's summary state. */
export type PartVerdict = 'OK' | 'OK_SO_FAR' | 'CHECK' | 'FIX' | 'UNSURE' | 'RESCAN'

export function partVerdict(state: SummaryState): PartVerdict {
  switch (state) {
    case 'ALL_MEASURED_IN_TOLERANCE': return 'OK'
    case 'STAGE_IN_TOLERANCE': return 'OK_SO_FAR'
    case 'OUT_OF_TOLERANCE_FOUND': return 'FIX'
    case 'REVIEW_REQUIRED': return 'CHECK'
    case 'INCOMPLETE': return 'UNSURE'
    default: return 'RESCAN'
  }
}

/** How many bends carry each verdict (bends, not characteristics). */
export function verdictCounts(rows: Pick<BendViewRow, 'tone' | 'state'>[]): Record<BendVerdict, number> {
  const out: Record<BendVerdict, number> = { OK: 0, CHECK: 0, FIX: 0, UNSURE: 0, NOT_BENT: 0, NONE: 0 }
  for (const r of rows) out[bendVerdict(r)] += 1
  return out
}

/**
 * Where a measured value sits within its tolerance, for a small horizontal
 * bar. Percentages (0..100, left to right, minus on the left) of a scale that
 * spans twice the wider tolerance side each way: the tolerance zone is the
 * middle, the marker is the stored deviation, the whisker its stored
 * uncertainty. `over` is -1 / +1 when the deviation is beyond the scale end
 * (the marker is pinned there). Null unless MEASURED with a deviation and a
 * tolerance.
 */
export interface ToleranceBar {
  zoneFrom: number
  zoneTo: number
  zero: number
  at: number
  lo: number
  hi: number
  over: -1 | 0 | 1
}

export function toleranceBar(c: Pick<CharacteristicResult, 'state' | 'deviation' | 'uncertainty' | 'tolerance'>): ToleranceBar | null {
  if (c.state !== 'MEASURED' || c.deviation == null || !Number.isFinite(c.deviation) || !c.tolerance) return null
  const span = 2 * Math.max(c.tolerance.plus, c.tolerance.minus)
  if (!(span > 0)) return null
  const pct = (v: number) => Math.min(100, Math.max(0, ((v + span) / (2 * span)) * 100))
  const dev = c.deviation
  const u = c.uncertainty != null && Number.isFinite(c.uncertainty) ? Math.abs(c.uncertainty) : 0
  return {
    zoneFrom: pct(-c.tolerance.minus),
    zoneTo: pct(c.tolerance.plus),
    zero: 50,
    at: pct(dev),
    lo: pct(dev - u),
    hi: pct(dev + u),
    over: dev > span ? 1 : dev < -span ? -1 : 0,
  }
}

/**
 * Plain direction of an out-of-tolerance value (FAIL only; a REVIEW may be
 * caused by uncertainty alone, so it gets no direction). Included angle below
 * nominal = bent too far; flange longer than nominal = too long.
 */
export type Direction = 'OVERBENT' | 'UNDERBENT' | 'TOO_LONG' | 'TOO_SHORT'

export function failDirection(c: Pick<CharacteristicResult, 'kind' | 'state' | 'decision' | 'deviation'>): Direction | null {
  if (c.state !== 'MEASURED' || c.decision !== 'FAIL' || c.deviation == null || c.deviation === 0) return null
  if (c.kind === 'INCLUDED_ANGLE') return c.deviation < 0 ? 'OVERBENT' : 'UNDERBENT'
  if (c.kind === 'FLANGE_LENGTH' || c.kind === 'BACKGAUGE_LENGTH') return c.deviation > 0 ? 'TOO_LONG' : 'TOO_SHORT'
  return null
}

const VERDICT_RANK: Record<BendVerdict, number> = { FIX: 4, CHECK: 3, UNSURE: 2, OK: 1, NOT_BENT: 0, NONE: 0 }

function cellRank(cell: CellView | null): number {
  if (!cell) return 0
  const c = cell.characteristic
  if (!c.required && c.state !== 'MEASURED') return 0
  return VERDICT_RANK[characteristicVerdict(c)]
}

/**
 * The bend-line position or backgauge cell when it, and not a flange or the
 * angle, sets the row's verdict (so the list can say why the bend is flagged).
 */
export function extraConcern(row: BendViewRow): CellView | null {
  if (row.state === 'NOT_YET_FORMED') return null
  const main = Math.max(cellRank(row.flangeA), cellRank(row.flangeB), cellRank(row.angle))
  let best: CellView | null = null
  for (const cell of [row.position, row.backgauge]) {
    if (cell && cellRank(cell) > main && cellRank(cell) >= VERDICT_RANK.CHECK && cellRank(cell) > cellRank(best)) best = cell
  }
  return best
}

/**
 * Where the bend's 3D tag and callout sit: the middle (by length) of the
 * measured bend line when the engine measured one, else the engine's label
 * anchor. Frame coordinates (`cad_model_mm`).
 */
export function bendTagAnchor(bend: Pick<BendRow, 'display'>): Vec3 {
  const line = bend.display.measured_polyline_mm
  if (line && line.length >= 2) {
    const seg: number[] = []
    let total = 0
    for (let i = 1; i < line.length; i++) {
      const [a, b] = [line[i - 1], line[i]]
      const d = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2])
      seg.push(d)
      total += d
    }
    if (total > 0) {
      let walk = total / 2
      for (let i = 0; i < seg.length; i++) {
        if (walk <= seg[i] || i === seg.length - 1) {
          const k = seg[i] > 0 ? Math.min(1, walk / seg[i]) : 0
          const [a, b] = [line[i], line[i + 1]]
          return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k]
        }
        walk -= seg[i]
      }
    }
  }
  return bend.display.label_anchor_mm
}
