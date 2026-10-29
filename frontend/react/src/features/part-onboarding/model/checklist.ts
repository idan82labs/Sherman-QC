/**
 * Pure onboarding model: SKU preview, tolerance validation, Vendor Setup draft
 * summary, release-request building and the wizard's step states.
 *
 * No React, no network: `scripts/check-onboarding-model.mjs` transpiles this file
 * and asserts its behaviour (including the failure paths).
 *
 * Honesty rules carried here:
 * - The server's `blockers` list is the release authority. The UI is ready only
 *   when `ready_to_release` is true AND `blockers` is empty; a disagreement is
 *   itself a blocker (READY_FLAG_DISAGREES), never resolved in favour of release.
 * - A step is `done` only on positive evidence. Missing data is `todo`, never done.
 * - The server's dry-run checklist (`onboarding.checklist`) is merged into the
 *   step states: a code the server reports keeps the step open even when the
 *   local view looks complete.
 */
import type {
  BendOverrideV1, CompilePreviewResponse, DefaultTolerancesV1, FlatIntakeV1, LastCompileV1, OnboardingPart, OnboardingResponse,
  ReleaseDraftResponse, ReleaseRequestV1, ToleranceV1, VendorDraftLite, VendorSetupLite,
} from './types'
import { flatBindingFreshness, proposalIsCurrent, contextFromSetup } from './flat'

// ---------------------------------------------------------------- SKU (CONTRACTS §1, canonical_sku.py)

export const SKU_PATTERN = /^[A-Z0-9][A-Z0-9._-]{0,63}$/

export type SkuPreview =
  | { ok: true; sku: string }
  | { ok: false; code: 'SKU_INVALID'; reason: 'EMPTY' | 'TOO_LONG' | 'FIRST_CHARACTER' | 'CHARACTERS'; sku: string }

/** Same rule as `domains.production.recipe.canonical_sku`: NFKC, strip, whitespace runs -> '-', ASCII upper. */
export function canonicalSkuPreview(raw: string): SkuPreview {
  const text = String(raw ?? '').normalize('NFKC').trim().replace(/\s+/g, '-').replace(/[a-z]/g, c => c.toUpperCase())
  if (SKU_PATTERN.test(text)) return { ok: true, sku: text }
  const reason = !text ? 'EMPTY' : text.length > 64 ? 'TOO_LONG' : !/^[A-Z0-9]/.test(text) ? 'FIRST_CHARACTER' : 'CHARACTERS'
  return { ok: false, code: 'SKU_INVALID', reason, sku: text }
}

// ---------------------------------------------------------------- tolerances

export type ToleranceProblem = 'NOT_FINITE' | 'NEGATIVE' | 'ZERO_WIDTH' | 'TOO_LARGE'

/**
 * A tolerance band is valid when both sides are finite and >= 0 and the band has
 * positive width (common.Tolerance). One side may be 0 (a unilateral band);
 * both 0 is rejected. `max` caps either side (e.g. < 90 deg for angles).
 */
export function validateTolerance(tol: Partial<ToleranceV1> | null | undefined, max = Infinity): ToleranceProblem[] {
  if (!tol) return ['NOT_FINITE']
  const plus = tol.plus, minus = tol.minus
  if (typeof plus !== 'number' || typeof minus !== 'number' || !Number.isFinite(plus) || !Number.isFinite(minus)) return ['NOT_FINITE']
  const problems: ToleranceProblem[] = []
  if (plus < 0 || minus < 0) problems.push('NEGATIVE')
  if (!(plus + minus > 0)) problems.push('ZERO_WIDTH')
  if (plus >= max || minus >= max) problems.push('TOO_LARGE')
  return problems
}

/** Parse two text fields into a band; empty text is NOT_FINITE (never silently 0). */
export function parseTolerance(plusText: string, minusText: string, max = Infinity): { value: ToleranceV1 | null; problems: ToleranceProblem[] } {
  const parse = (text: string) => (text.trim() === '' ? Number.NaN : Number(text.trim().replace(',', '.')))
  const value = { plus: parse(plusText), minus: parse(minusText) }
  const problems = validateTolerance(value, max)
  return { value: problems.length ? null : value, problems }
}

export const TOLERANCE_LIMITS = { angle_deg: 90, length_mm: 1000 } as const

export type DefaultToleranceField = 'angle_deg' | 'flange_mm' | 'bend_line_position_mm' | 'backgauge_mm' | 'general_mm' | 'surface_profile_mm' | 'source'

export function validateDefaultTolerances(d: DefaultTolerancesV1 | null | undefined): Partial<Record<DefaultToleranceField, string[]>> {
  if (!d) return { angle_deg: ['NOT_FINITE'], flange_mm: ['NOT_FINITE'] }
  const out: Partial<Record<DefaultToleranceField, string[]>> = {}
  const put = (field: DefaultToleranceField, problems: string[]) => { if (problems.length) out[field] = problems }
  put('angle_deg', validateTolerance(d.angle_deg, TOLERANCE_LIMITS.angle_deg))
  put('flange_mm', validateTolerance(d.flange_mm, TOLERANCE_LIMITS.length_mm))
  for (const optional of ['bend_line_position_mm', 'backgauge_mm', 'general_mm'] as const) {
    const band = d[optional]
    if (band != null) put(optional, validateTolerance(band, TOLERANCE_LIMITS.length_mm))
  }
  if (!(typeof d.surface_profile_mm === 'number' && Number.isFinite(d.surface_profile_mm) && d.surface_profile_mm > 0)) {
    put('surface_profile_mm', ['NOT_FINITE'])
  }
  if (!d.source) put('source', ['MISSING'])
  return out
}

/** Included angle for a manual mark: strictly between 0 and 180 (180 would be flat, no bend). */
export function validIncludedAngle(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 && value < 180
}

// ---------------------------------------------------------------- CAD

const STEP_EXT = /\.(step|stp)$/i
const STEP_FORMATS = new Set(['step', 'stp'])

/**
 * Production recipes require a STEP CAD (part_recipe.CadRef). C05 summaries carry
 * `cad_format` (the lower-case extension); the legacy catalog part carries the path.
 */
export function isStepCad(part: Pick<OnboardingPart, 'cad_file_path' | 'cad_file_hash' | 'cad_format'> | null | undefined): boolean {
  if (!part?.cad_file_hash) return false
  if (typeof part.cad_format === 'string' && part.cad_format) return STEP_FORMATS.has(part.cad_format.toLowerCase())
  return !!part.cad_file_path && STEP_EXT.test(part.cad_file_path)
}

/**
 * The part as the wizard shows it: the C05 onboarding summary (authoritative for
 * hashes and state) over the legacy catalog part (which still has file paths).
 */
export function mergedPart(snapshot: Pick<WizardSnapshot, 'onboarding' | 'vendorSetup'>): OnboardingPart | null {
  const server = snapshot.onboarding?.part ?? null
  const legacy = snapshot.vendorSetup?.part ?? null
  if (!server) return legacy
  if (!legacy || legacy.id !== server.id) return server
  const out: OnboardingPart = { ...legacy }
  for (const [key, value] of Object.entries(server)) if (value !== undefined) (out as unknown as Record<string, unknown>)[key] = value
  // A path is shown only when it belongs to the same file the server reports.
  if (legacy.cad_file_hash !== server.cad_file_hash) out.cad_file_path = null
  if (legacy.drawing_file_hash !== server.drawing_file_hash) out.drawing_file_path = null
  return out
}

/**
 * Included angle from a catalog candidate. STEP candidates store the native sweep
 * (`included = 180 - sweep`, ENGINE_IO §1); returns null when it is not a valid bend.
 */
export function includedFromCandidateSweep(targetAngle: number | null | undefined): number | null {
  if (typeof targetAngle !== 'number' || !Number.isFinite(targetAngle)) return null
  const included = 180 - targetAngle
  return included > 0 && included < 180 ? included : null
}

// ---------------------------------------------------------------- Vendor Setup summary

export interface OrderedBend {
  feature_id: string
  label: string
  step: number
  source: 'cad_candidate' | 'manual'
  confirmed: boolean
  hasLine: boolean
  /** 1-based machine-move (מהלך) group, or null when the bend is in no group. */
  moveIndex: number | null
  /** Included angle derived from the CAD candidate sweep (STEP), null for manual marks. */
  cadIncludedDeg: number | null
}

export interface VendorSummary {
  hasDraft: boolean
  draftRevision: number
  stale: boolean
  sequenceConfirmed: boolean
  activeCount: number
  confirmedCount: number
  removedCount: number
  ordered: OrderedBend[]
  machineMoves: string[][]
  manualActiveIds: string[]
  measurementIds: string[]
  candidateCount: number
}

/** Only ordered, contiguous, disjoint groups count (vendor_setup_service.py rule); anything else is dropped. */
export function validMachineMoves(sequence: string[], moves: string[][] | undefined | null): string[][] {
  if (!moves?.length) return []
  let cursor = 0
  for (const move of moves) {
    if (!Array.isArray(move) || !move.length) return []
    const start = sequence.indexOf(move[0], cursor)
    if (start < 0 || start + move.length > sequence.length) return []
    if (move.some((id, index) => sequence[start + index] !== id)) return []
    cursor = start + move.length
  }
  return moves
}

export function summarizeVendorSetup(setup: VendorSetupLite | null | undefined): VendorSummary {
  const draft = setup?.draft ?? null
  const marks = new Map((draft?.marks ?? []).map(mark => [mark.feature_id, mark]))
  const sequence = (draft?.sequence ?? []).filter(id => marks.get(id)?.status === 'active')
  const moves = validMachineMoves(sequence, draft?.machine_moves)
  const moveOf = new Map<string, number>()
  moves.forEach((move, index) => move.forEach(id => moveOf.set(id, index + 1)))
  const ordered: OrderedBend[] = sequence.map((id, index) => {
    const mark = marks.get(id)!
    return {
      feature_id: id, label: mark.label, step: index + 1, source: mark.source,
      confirmed: mark.review_status === 'confirmed', hasLine: !!(mark.line_start && mark.line_end),
      moveIndex: moveOf.get(id) ?? null,
      cadIncludedDeg: mark.source === 'cad_candidate' ? includedFromCandidateSweep(mark.target_angle) : null,
    }
  })
  return {
    hasDraft: !!draft,
    draftRevision: draft?.draft_revision ?? 0,
    stale: !!draft?.is_stale,
    sequenceConfirmed: draft?.operator_sequence_confirmed === true,
    activeCount: ordered.length,
    confirmedCount: ordered.filter(row => row.confirmed).length,
    removedCount: (draft?.marks ?? []).filter(mark => mark.status === 'removed').length,
    ordered,
    machineMoves: moves,
    manualActiveIds: ordered.filter(row => row.source === 'manual').map(row => row.feature_id),
    measurementIds: (draft?.general_measurements ?? []).map(item => item.id),
    candidateCount: setup?.candidates?.length ?? 0,
  }
}

/**
 * How a stale saved bend setup can be repaired (catalog `get_vendor_setup_draft`
 * staleness: CAD bytes, CAD upload time, part revision, drawing hash, candidate
 * generation). When only the drawing changed, the marks still describe the same
 * CAD, so re-saving the same draft against the new drawing is safe
 * (`DRAWING_ONLY`). Anything else needs the bends reviewed again (`REVIEW`).
 */
export type StaleDraftRepair = 'NONE' | 'DRAWING_ONLY' | 'REVIEW'

export function staleDraftRepair(setup: VendorSetupLite | null | undefined): StaleDraftRepair {
  const draft = setup?.draft as (VendorDraftLite & { cad_uploaded_at?: string | null; part_revision?: string | null; drawing_file_hash?: string | null }) | null | undefined
  if (!setup || !draft || !draft.is_stale) return 'NONE'
  const part = setup.part
  const sameCad = draft.cad_file_hash === part.cad_file_hash && (draft.cad_uploaded_at ?? null) === (part.cad_uploaded_at ?? null)
    && (draft.part_revision ?? null) === (part.revision ?? null) && draft.candidate_generation === setup.candidate_generation
  return sameCad && (draft.drawing_file_hash ?? null) !== (part.drawing_file_hash ?? null) ? 'DRAWING_ONLY' : 'REVIEW'
}

/** Codes that keep the bends & order step from being done (mirrors the builder's draft gate). */
export function bendStepCodes(summary: VendorSummary): string[] {
  const codes: string[] = []
  if (!summary.hasDraft) return ['DRAFT_MISSING']
  if (summary.stale) codes.push('DRAFT_STALE')
  if (summary.activeCount === 0) codes.push('NO_ACTIVE_BENDS')
  if (summary.ordered.some(row => !row.hasLine)) codes.push('MARK_WITHOUT_LINE')
  if (summary.confirmedCount < summary.activeCount) codes.push('UNCONFIRMED_ACTIVE_MARKS')
  if (!summary.sequenceConfirmed) codes.push('SEQUENCE_NOT_CONFIRMED')
  return codes
}

// ---------------------------------------------------------------- release request

export const DEFAULT_TOLERANCE_SEED: DefaultTolerancesV1 = {
  angle_deg: { plus: 1, minus: 1 },
  flange_mm: { plus: 0.5, minus: 0.5 },
  bend_line_position_mm: null,
  backgauge_mm: null,
  general_mm: null,
  surface_profile_mm: 0.5,
  source: 'DRAWING',
  note: '',
}

export function emptyReleaseRequest(draftRevision: number, releaseDraftRevision: number | null): ReleaseRequestV1 {
  return {
    expected_draft_revision: draftRevision,
    expected_release_draft_revision: releaseDraftRevision,
    default_tolerances: null,
    bend_overrides: {},
    measurement_overrides: {},
    thickness_mm: null,
    inspection_stage: 'FINAL',
    release_note: '',
  }
}

function prunedOverride(override: BendOverrideV1): BendOverrideV1 | null {
  const out: BendOverrideV1 = {}
  for (const [key, value] of Object.entries(override) as Array<[keyof BendOverrideV1, unknown]>) {
    if (value === null || value === undefined) continue
    if (key === 'flange_inspection') {
      const fi = value as { side_a: string; side_b: string }
      if (fi.side_a === 'REQUIRED' && fi.side_b === 'REQUIRED') continue
    }
    ;(out as Record<string, unknown>)[key] = value
  }
  return Object.keys(out).length ? out : null
}

/**
 * Normalise the editable request into the exact ReleaseRequestV1 JSON (extra=forbid):
 * drops empty overrides, overrides for bends no longer active, and the default
 * (both sides REQUIRED) flange inspection; pins the draft revisions.
 */
export function buildReleaseRequest(
  request: ReleaseRequestV1, summary: VendorSummary, releaseDraftRevision: number | null,
): ReleaseRequestV1 {
  const active = new Set(summary.ordered.map(row => row.feature_id))
  const manual = new Set(summary.manualActiveIds)
  const bend_overrides: Record<string, BendOverrideV1> = {}
  for (const [fid, override] of Object.entries(request.bend_overrides || {})) {
    if (!active.has(fid)) continue
    const clean = prunedOverride(manual.has(fid) ? override : { ...override, manual_included_angle_deg: null })
    if (clean) bend_overrides[fid] = clean
  }
  const measurements = new Set(summary.measurementIds)
  const measurement_overrides: ReleaseRequestV1['measurement_overrides'] = {}
  for (const [id, override] of Object.entries(request.measurement_overrides || {})) {
    if (!measurements.has(id)) continue
    const clean: Record<string, unknown> = {}
    if (override.tolerance) clean.tolerance = override.tolerance
    if (typeof override.required === 'boolean') clean.required = override.required
    if (Object.keys(clean).length) measurement_overrides[id] = clean
  }
  const out: ReleaseRequestV1 = {
    expected_draft_revision: summary.draftRevision,
    expected_release_draft_revision: releaseDraftRevision,
    default_tolerances: request.default_tolerances ?? null,
    bend_overrides,
    measurement_overrides,
    thickness_mm: typeof request.thickness_mm === 'number' && request.thickness_mm > 0 ? request.thickness_mm : null,
    inspection_stage: request.inspection_stage === 'IN_PROCESS' ? 'IN_PROCESS' : 'FINAL',
    release_note: (request.release_note || '').slice(0, 2000),
  }
  if (request.confirmation) out.confirmation = request.confirmation
  return out
}

/** Local pre-checks before PUT release-draft; the server re-validates and is the authority. */
export function releaseRequestCodes(request: ReleaseRequestV1 | null | undefined, summary: VendorSummary): string[] {
  const codes = new Set<string>()
  if (!request || !request.default_tolerances) codes.add('TOLERANCES_MISSING')
  else if (Object.keys(validateDefaultTolerances(request.default_tolerances)).length) codes.add('TOLERANCE_INVALID')
  const active = new Set(summary.ordered.map(row => row.feature_id))
  for (const [fid, override] of Object.entries(request?.bend_overrides || {})) {
    if (!active.has(fid)) { codes.add('OVERRIDE_UNKNOWN_BEND'); continue }
    for (const key of ['angle_deg', 'flange_a_mm', 'flange_b_mm', 'bend_line_position_mm', 'backgauge_mm'] as const) {
      const band = override[key]
      if (band != null && validateTolerance(band, key === 'angle_deg' ? TOLERANCE_LIMITS.angle_deg : TOLERANCE_LIMITS.length_mm).length) codes.add('TOLERANCE_INVALID')
    }
  }
  for (const fid of summary.manualActiveIds) {
    if (!validIncludedAngle(request?.bend_overrides?.[fid]?.manual_included_angle_deg)) codes.add('MANUAL_MARK_MISSING_NOMINAL_ANGLE')
  }
  const measurementIds = new Set(summary.measurementIds)
  for (const [id, override] of Object.entries(request?.measurement_overrides || {})) {
    if (!measurementIds.has(id)) codes.add('OVERRIDE_UNKNOWN_MEASUREMENT')
    else if (override.tolerance && validateTolerance(override.tolerance, TOLERANCE_LIMITS.length_mm).length) codes.add('TOLERANCE_INVALID')
  }
  if (request?.thickness_mm != null && !(Number.isFinite(request.thickness_mm) && request.thickness_mm > 0)) codes.add('TOLERANCE_INVALID')
  return [...codes].sort()
}

// ---------------------------------------------------------------- readiness

export interface ReleaseReadiness {
  ready: boolean
  codes: string[]
  /** Nothing to release: the inputs reproduce the active recipe. */
  unchanged: boolean
}

/** The server is the authority: ready only when it says so AND it lists no blocker. */
export function releaseReadiness(onboarding: OnboardingResponse | null | undefined): ReleaseReadiness {
  if (!onboarding) return { ready: false, codes: [], unchanged: false }
  const codes = [...new Set((onboarding.blockers || []).filter(code => typeof code === 'string'))]
  if (onboarding.ready_to_release === true && codes.length === 0) return { ready: true, codes, unchanged: false }
  if (codes.length === 0 && onboarding.active_recipe) {
    // A released part with nothing new to release is not a disagreement; there is simply nothing to do.
    if (onboarding.release_would_be_unchanged === true) return { ready: false, codes, unchanged: true }
    if (onboarding.onboarding_state === 'RELEASED') return { ready: false, codes, unchanged: false }
  }
  if (codes.length === 0) return { ready: false, codes: ['READY_FLAG_DISAGREES'], unchanged: false }
  return { ready: false, codes, unchanged: false }
}

/** Codes of one server checklist row (`sku`, `cad`, `drawing`, `bend_order`, `flat`, `compile`, `tolerances`, `other`). */
export function serverStepCodes(onboarding: OnboardingResponse | null | undefined, key: string): string[] {
  const item = onboarding?.checklist?.find(row => row.key === key)
  return item && !item.ok ? item.codes : []
}

// ---------------------------------------------------------------- steps

export const WIZARD_STEPS = ['part', 'cad', 'drawing', 'flat', 'bends', 'tolerances', 'review', 'release'] as const
export type WizardStepKey = typeof WIZARD_STEPS[number]
export type StepStatus = 'done' | 'todo' | 'attention' | 'blocked' | 'optional' | 'working'

export interface StepState {
  key: WizardStepKey
  status: StepStatus
  codes: string[]
}

export interface WizardSnapshot {
  onboarding?: OnboardingResponse | null
  vendorSetup?: VendorSetupLite | null
  flat?: FlatIntakeV1 | null
  releaseDraft?: ReleaseDraftResponse | null
  /** Pre-release compile preview; `unavailable` when the server has no such route (C05 today). */
  compilePreview?: { kind: 'preview'; preview: CompilePreviewResponse } | { kind: 'none' } | { kind: 'unavailable' } | null
}

export function isStepKey(value: string | null | undefined): value is WizardStepKey {
  return !!value && (WIZARD_STEPS as readonly string[]).includes(value)
}

/**
 * Whether a compile preview was computed from the current draft, release draft and
 * flat. Provenance must be present and equal: a preview that does not say what it
 * was computed from is stale, never current (`flat_sha256: null` means "no flat").
 */
export function previewIsCurrent(preview: CompilePreviewResponse | null | undefined, snapshot: WizardSnapshot): boolean {
  if (!preview || preview.status === 'PENDING' || !snapshot.vendorSetup?.draft || !snapshot.releaseDraft) return false
  const summary = summarizeVendorSetup(snapshot.vendorSetup)
  if (typeof preview.draft_revision !== 'number' || preview.draft_revision !== summary.draftRevision) return false
  if (typeof preview.release_draft_revision !== 'number' || preview.release_draft_revision !== snapshot.releaseDraft.revision) return false
  if (preview.flat_sha256 === undefined) return false
  return (preview.flat_sha256 ?? null) === (snapshot.flat?.flat.sha256 ?? null)
}

/** The last release attempt's compile, when it was made from the current CAD + draft and did not compile. */
export function blockingLastCompile(onboarding: OnboardingResponse | null | undefined): LastCompileV1 | null {
  const last = onboarding?.last_compile
  return last && last.applies_to_current_draft && last.status !== 'COMPILED' ? last : null
}

const uniq = (codes: string[]) => [...new Set(codes)]

export function deriveSteps(snapshot: WizardSnapshot): StepState[] {
  const onboarding = snapshot.onboarding
  const part = mergedPart(snapshot)
  const summary = summarizeVendorSetup(snapshot.vendorSetup)
  const steps: StepState[] = []

  // 1. Part details
  const sku = part ? canonicalSkuPreview(part.part_number) : null
  const skuCodes = serverStepCodes(onboarding, 'sku')
  steps.push(!part ? { key: 'part', status: 'todo', codes: [] }
    : !sku?.ok || skuCodes.length ? { key: 'part', status: 'blocked', codes: uniq([...(!sku?.ok ? ['SKU_INVALID'] : []), ...skuCodes]) }
      : !part.part_name?.trim() ? { key: 'part', status: 'attention', codes: [] }
        : { key: 'part', status: 'done', codes: [] })

  // 2. Folded CAD (STEP only)
  const cadServer = serverStepCodes(onboarding, 'cad')
  steps.push(!part?.cad_file_hash ? { key: 'cad', status: 'todo', codes: uniq(['CAD_MISSING', ...cadServer]) }
    : !isStepCad(part) ? { key: 'cad', status: 'blocked', codes: uniq(['CAD_NOT_STEP', ...cadServer]) }
      : cadServer.length ? { key: 'cad', status: 'blocked', codes: cadServer }
        : snapshot.vendorSetup && summary.candidateCount === 0 && summary.activeCount === 0
          ? { key: 'cad', status: 'attention', codes: ['CAD_NO_CANDIDATES'] }
          : { key: 'cad', status: 'done', codes: [] })

  // 3. Drawing (required by the recipe)
  const drawingServer = serverStepCodes(onboarding, 'drawing')
  steps.push(!part?.drawing_file_hash ? { key: 'drawing', status: 'todo', codes: uniq(['DRAWING_MISSING', ...drawingServer]) }
    : drawingServer.length ? { key: 'drawing', status: 'blocked', codes: drawingServer }
      : { key: 'drawing', status: 'done', codes: [] })

  // 4. Unfolded part (optional, but once uploaded it must parse and be decided: confirmed or kept unbound)
  const flat = snapshot.flat
  const flatServer = serverStepCodes(onboarding, 'flat')
  if (!flat) steps.push(flatServer.length ? { key: 'flat', status: 'blocked', codes: flatServer } : { key: 'flat', status: 'optional', codes: [] })
  else if (flat.parse.status !== 'PARSED') steps.push({ key: 'flat', status: 'blocked', codes: uniq(['FLAT_PARSE_FAILED', ...flatServer]) })
  else {
    const freshness = flatBindingFreshness(flat, snapshot.vendorSetup)
    const local: string[] = freshness === 'FRESH' || freshness === 'DECLINED' || freshness === 'UNVERIFIED' ? []
      : freshness === 'UNCONFIRMED'
        ? [flat.proposal?.status === 'RECONCILIATION_BLOCKED' ? 'FLAT_RECONCILIATION_BLOCKED'
          : flat.state === 'PROPOSED' && !proposalIsCurrent(flat, contextFromSetup(snapshot.vendorSetup)) ? 'FLAT_BINDING_STALE' : 'FLAT_BINDING_UNCONFIRMED']
        : ['FLAT_BINDING_STALE']
    const codes = uniq([...local, ...flatServer])
    if (codes.length) steps.push({ key: 'flat', status: 'attention', codes })
    else if (freshness === 'UNVERIFIED') steps.push({ key: 'flat', status: 'working', codes: [] })
    else steps.push({ key: 'flat', status: 'done', codes: [] })
  }

  // 5. Bends & order (Vendor Setup)
  const bendCodes = uniq([...bendStepCodes(summary), ...(snapshot.vendorSetup ? serverStepCodes(onboarding, 'bend_order') : [])])
  steps.push(!snapshot.vendorSetup ? { key: 'bends', status: 'todo', codes: [] }
    : bendCodes.length === 0 ? { key: 'bends', status: 'done', codes: [] }
      : { key: 'bends', status: bendCodes.includes('DRAFT_STALE') ? 'blocked' : 'attention', codes: bendCodes })

  // 6. Tolerances (persisted release draft)
  const request = snapshot.releaseDraft?.request ?? null
  const tolCodes = uniq([...releaseRequestCodes(request, summary), ...(request ? serverStepCodes(onboarding, 'tolerances') : [])])
  steps.push(!request ? { key: 'tolerances', status: 'todo', codes: ['TOLERANCES_MISSING'] }
    : tolCodes.length ? { key: 'tolerances', status: 'attention', codes: tolCodes }
      : { key: 'tolerances', status: 'done', codes: [] })

  // 7. Review expectations. A blocking compile of the current inputs always wins;
  // otherwise the compile preview (when this server has one) decides.
  const lastBlocking = blockingLastCompile(onboarding)
  const compileServer = serverStepCodes(onboarding, 'compile')
  const state = snapshot.compilePreview
  const preview = state?.kind === 'preview' ? state.preview : null
  if (lastBlocking || compileServer.length) {
    const issueCodes = lastBlocking?.report?.issues.filter(issue => issue.severity === 'BLOCKER').map(issue => issue.code) ?? []
    steps.push({ key: 'review', status: 'blocked', codes: uniq([...compileServer, ...(lastBlocking?.codes ?? []), ...issueCodes]) })
  } else if (state?.kind === 'unavailable') {
    // No pre-release compile on this server: E1 runs at release. Nothing was reviewed, so this is never `done`.
    steps.push({ key: 'review', status: 'optional', codes: ['COMPILE_PREVIEW_UNAVAILABLE'] })
  } else if (!preview) steps.push({ key: 'review', status: 'todo', codes: ['COMPILE_PREVIEW_MISSING'] })
  else if (preview.status === 'PENDING') steps.push({ key: 'review', status: 'working', codes: [] })
  else if (!previewIsCurrent(preview, snapshot)) steps.push({ key: 'review', status: 'attention', codes: ['COMPILE_PREVIEW_STALE'] })
  else if (preview.status === 'COMPILED') steps.push({ key: 'review', status: 'done', codes: [] })
  else {
    const blockerCodes = preview.compiled_part?.issues?.filter(issue => issue.severity === 'BLOCKER').map(issue => issue.code) ?? []
    const codes = [...new Set([...(preview.recipe_codes ?? []), ...blockerCodes, ...(preview.error ? [preview.error.code] : [])])]
    steps.push({ key: 'review', status: 'blocked', codes: codes.length ? codes : ['COMPILE_BLOCKED'] })
  }

  // 8. Release
  const readiness = releaseReadiness(onboarding)
  const onboardingState = onboarding?.onboarding_state
  steps.push(onboardingState === 'RELEASED' && onboarding?.active_recipe ? { key: 'release', status: 'done', codes: [] }
    : onboardingState === 'RELEASED_STALE' ? { key: 'release', status: readiness.unchanged ? 'done' : 'attention', codes: readiness.codes }
      : readiness.ready ? { key: 'release', status: 'todo', codes: [] }
        : { key: 'release', status: onboarding ? 'blocked' : 'todo', codes: readiness.codes })
  return steps
}

/**
 * The step to suggest next: the step right after `current` when it is an untouched optional step
 * (optional is offered, never silently skipped - the footer "Next" opens it too), else the first
 * open step after `current`, else the first open one before it.
 */
export function nextOpenStep(steps: StepState[], current: WizardStepKey): WizardStepKey | null {
  const open = (step: StepState) => step.status !== 'done' && step.status !== 'optional'
  const at = steps.findIndex(step => step.key === current)
  const following = at >= 0 ? steps[at + 1] : undefined
  if (following && following.status === 'optional') return following.key
  return (steps.slice(at + 1).find(open) ?? steps.slice(0, at).find(open))?.key ?? null
}

/** First step that still needs the operator (used to resume the wizard). */
export function firstOpenStep(steps: StepState[]): WizardStepKey {
  return steps.find(step => step.status !== 'done' && step.status !== 'optional')?.key ?? 'release'
}
