/**
 * Pure view model for "Review expectations": the E1 CompiledPart nominals grouped
 * per bend in sequence order, plus the part-level reconciliation picture.
 *
 * - The compiled flange-graph angle is authoritative (ENGINE_IO §2). When the
 *   recipe angle differs by more than ANGLE_DISAGREEMENT_DEG both are shown.
 * - A characteristic with `unavailable_reason` has no nominal. It is shown as
 *   unavailable with advice, never as a number. This holds for included angles
 *   too: the bend's own `nominal_included_deg` is used only when the compile
 *   produced no INCLUDED_ANGLE characteristic at all.
 */
import type { CharacteristicNominalV1, CompiledPartLite, CompileIssueV1, ExpectationsResponse } from './types'
import { compileSeverity } from './codes'
import { compiledFromExpectationsDoc } from './adapters'

export const ANGLE_DISAGREEMENT_DEG = 0.5

export interface SideExpectation {
  side: 'A' | 'B'
  characteristic: CharacteristicNominalV1 | null
  nominal: number | null
  unavailable: 'NOMINAL_UNAVAILABLE' | 'UNSUPPORTED_FEATURE' | 'MISSING' | null
  boundaryKind: 'FREE_EDGE' | 'ADJACENT_BEND' | null
  adjacentLabel: string | null
  requirement: 'REQUIRED' | 'NOT_INSPECTED' | 'OPTIONAL' | null
}

export interface BendExpectation {
  bend_uid: string
  label: string
  sequence_index: number
  machine_move_index: number | null
  graph_bend_id: string
  included: CharacteristicNominalV1 | null
  includedDeg: number | null
  /** Set when the compile withheld the angle nominal (unsupported feature, or not derivable). */
  includedUnavailable: 'NOMINAL_UNAVAILABLE' | 'UNSUPPORTED_FEATURE' | null
  recipeIncludedDeg: number | null
  angleDisagrees: boolean
  innerRadiusMm: number
  sides: [SideExpectation, SideExpectation]
  bendLine: CharacteristicNominalV1 | null
  backgauge: CharacteristicNominalV1[]
  supported: boolean
  unsupportedFeature: string | null
  issues: CompileIssueV1[]
}

export interface ExpectationsView {
  status: 'COMPILED' | 'BLOCKED'
  bends: BendExpectation[]
  generalMeasurements: CharacteristicNominalV1[]
  blockers: CompileIssueV1[]
  warnings: CompileIssueV1[]
  unmarkedGraphBends: Array<{ graph_bend_id: string; sweep_deg: number; inner_radius_mm: number }>
  kFactor: CompiledPartLite['k_factor'] | null
  thickness: CompiledPartLite['thickness'] | null
  flatCheck: NonNullable<CompiledPartLite['flat']>['developed_length_check'] | null
  unavailableSides: number
}

/**
 * Accept the C05 expectations document (`sherman_recipe_expectations_v1`), a
 * CompiledPart directly, or `{compiled_part}`. Anything else is null.
 */
export function compiledFrom(payload: ExpectationsResponse['expectations'] | CompiledPartLite | null | undefined): CompiledPartLite | null {
  if (!payload || typeof payload !== 'object') return null
  if ((payload as { schema?: unknown }).schema === 'sherman_recipe_expectations_v1') return compiledFromExpectationsDoc(payload)
  const candidate = 'status' in payload && 'bends' in payload ? payload as CompiledPartLite
    : (payload as { compiled_part?: CompiledPartLite | null }).compiled_part ?? null
  if (!candidate || !Array.isArray(candidate.bends) || !Array.isArray(candidate.characteristics)) return null
  return { ...candidate, issues: Array.isArray(candidate.issues) ? candidate.issues : [] }
}

function sideOf(chars: CharacteristicNominalV1[], side: 'A' | 'B', labelOf: Map<string, string>): SideExpectation {
  const c = chars.find(item => item.kind === 'FLANGE_LENGTH' && item.side === side) ?? null
  const nominal = c && c.unavailable_reason == null && typeof c.nominal === 'number' && Number.isFinite(c.nominal) ? c.nominal : null
  return {
    side, characteristic: c, nominal,
    unavailable: !c ? 'MISSING' : nominal == null ? (c.unavailable_reason ?? 'NOMINAL_UNAVAILABLE') : null,
    boundaryKind: c?.boundary_kind ?? null,
    adjacentLabel: c?.adjacent_bend_uid ? labelOf.get(c.adjacent_bend_uid) ?? null : null,
    requirement: c?.recipe_requirement ?? null,
  }
}

/**
 * Group compiled nominals per bend. `recipeIncludedBySequence` gives the recipe's
 * included angle per sequence index (from the CAD candidate sweep or the manual
 * entry), so a convention disagreement can show both values.
 */
export function buildExpectations(compiled: CompiledPartLite, recipeIncludedBySequence: Map<number, number | null> = new Map()): ExpectationsView {
  const labelOf = new Map(compiled.bends.map(bend => [bend.bend_uid, bend.label]))
  const issues = (compiled.issues || []).map(issue => ({ ...issue, severity: issue.severity ?? compileSeverity(issue.code) }))
  const bends = [...compiled.bends].sort((a, b) => a.sequence_index - b.sequence_index).map(bend => {
    const chars = compiled.characteristics.filter(item => item.bend_uid === bend.bend_uid)
    const included = chars.find(item => item.kind === 'INCLUDED_ANGLE') ?? null
    const includedUnavailable = included && (included.unavailable_reason != null || !(typeof included.nominal === 'number' && Number.isFinite(included.nominal)))
      ? (included.unavailable_reason ?? 'NOMINAL_UNAVAILABLE') : null
    const fallback = typeof bend.nominal_included_deg === 'number' && Number.isFinite(bend.nominal_included_deg) ? bend.nominal_included_deg : null
    const includedDeg = included ? (includedUnavailable ? null : included.nominal as number) : fallback
    // The recipe's own angle: from the expectations document when present, else what the wizard derived.
    const recipe = typeof bend.recipe_included_deg === 'number' && Number.isFinite(bend.recipe_included_deg)
      ? bend.recipe_included_deg : recipeIncludedBySequence.get(bend.sequence_index) ?? null
    const bendIssues = issues.filter(issue => issue.bend_uid === bend.bend_uid)
    const disagreeIssue = bendIssues.some(issue => issue.code === 'ANGLE_CONVENTION_DISAGREES_WITH_RECIPE')
    const numeric = includedDeg != null && recipe != null && Math.abs(includedDeg - recipe) > ANGLE_DISAGREEMENT_DEG
    return {
      bend_uid: bend.bend_uid, label: bend.label, sequence_index: bend.sequence_index,
      machine_move_index: bend.machine_move_index ?? null, graph_bend_id: bend.graph_bend_id,
      included, includedDeg, includedUnavailable, recipeIncludedDeg: recipe, angleDisagrees: disagreeIssue || numeric,
      innerRadiusMm: bend.inner_radius_mm,
      sides: [sideOf(chars, 'A', labelOf), sideOf(chars, 'B', labelOf)] as [SideExpectation, SideExpectation],
      bendLine: chars.find(item => item.kind === 'BEND_LINE_POSITION') ?? null,
      backgauge: chars.filter(item => item.kind === 'BACKGAUGE_LENGTH').sort((a, b) => (a.stage_index ?? 0) - (b.stage_index ?? 0)),
      supported: bend.supported !== false,
      unsupportedFeature: bend.unsupported_feature ?? null,
      issues: bendIssues,
    }
  })
  const unavailableSides = bends.reduce((count, bend) => count + bend.sides.filter(side =>
    side.unavailable && side.requirement !== 'NOT_INSPECTED').length, 0)
  return {
    status: compiled.status,
    bends,
    generalMeasurements: compiled.characteristics.filter(item => item.kind === 'GENERAL_MEASUREMENT'),
    blockers: issues.filter(issue => issue.severity === 'BLOCKER'),
    warnings: issues.filter(issue => issue.severity === 'WARNING'),
    unmarkedGraphBends: compiled.unmarked_graph_bends ?? [],
    kFactor: compiled.k_factor ?? null,
    thickness: compiled.thickness ?? null,
    flatCheck: compiled.flat?.developed_length_check ?? null,
    unavailableSides,
  }
}

/** Format a nominal for display; null stays null (never 0). */
export function formatNominal(value: number | null | undefined, unit: 'mm' | 'deg', digits = unit === 'deg' ? 1 : 2): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null
  return unit === 'deg' ? `${value.toFixed(digits)}°` : value.toFixed(digits)
}

export function formatBand(tol: { plus: number; minus: number } | null | undefined, unit: 'mm' | 'deg'): string | null {
  if (!tol) return null
  const digits = unit === 'deg' ? 2 : 3
  const trim = (v: number) => String(Number(v.toFixed(digits)))
  const suffix = unit === 'deg' ? '°' : ''
  return tol.plus === tol.minus ? `±${trim(tol.plus)}${suffix}` : `+${trim(tol.plus)}/−${trim(tol.minus)}${suffix}`
}
