/**
 * Text-field state for the tolerance editor and its exact conversion to and from
 * ReleaseRequestV1. Pure, so the check script can exercise it.
 *
 * - Required bands (angle, flange length) must be filled; an empty field is an
 *   error, never a silent 0.
 * - Optional bands (bend-line position, backgauge, general) and every per-bend
 *   override are "not set" when both fields are empty, and an error when only
 *   one side is filled or the band is invalid.
 * - No tolerance is pre-filled: a part with no saved draft starts with empty
 *   required bands. The one pre-filled field is the heatmap colour band
 *   (`SURFACE_BAND_DISPLAY_DEFAULT_MM`), which only scales the deviation colours
 *   and is never a conformance tolerance; the editor labels it as a default.
 */
import type {
  BendOverrideV1, DefaultTolerancesV1, FlangeSideRequirement, InspectionStage, ReleaseRequestV1, ToleranceSource, ToleranceV1,
} from './types'
import { TOLERANCE_LIMITS, validateTolerance, type ToleranceProblem, type VendorSummary } from './checklist'

/** Pre-filled heatmap colour band (± mm). Display only: it never decides PASS or FAIL. */
export const SURFACE_BAND_DISPLAY_DEFAULT_MM = 0.5

export interface BandText { plus: string; minus: string; symmetric: boolean }

export interface BendRowForm {
  angle: BandText
  flangeA: BandText
  flangeB: BandText
  sideA: FlangeSideRequirement
  sideB: FlangeSideRequirement
  manualAngle: string
}

export interface MeasurementRowForm { tol: BandText; required: boolean | null }

export interface ToleranceForm {
  angle: BandText
  flange: BandText
  bendLine: BandText
  backgauge: BandText
  general: BandText
  surface: string
  source: ToleranceSource | ''
  note: string
  stage: InspectionStage
  thickness: string
  releaseNote: string
  bends: Record<string, BendRowForm>
  measurements: Record<string, MeasurementRowForm>
}

const EMPTY: BandText = { plus: '', minus: '', symmetric: true }

function num(value: number): string {
  return Number.isFinite(value) ? String(Number(value.toFixed(6))) : ''
}

export function bandText(tol: ToleranceV1 | null | undefined): BandText {
  if (!tol) return { ...EMPTY }
  return { plus: num(tol.plus), minus: num(tol.minus), symmetric: tol.plus === tol.minus }
}

export function formFromRequest(request: ReleaseRequestV1 | null | undefined, summary: VendorSummary): ToleranceForm {
  const d = request?.default_tolerances ?? null
  const bends: Record<string, BendRowForm> = {}
  for (const row of summary.ordered) {
    const o: BendOverrideV1 = request?.bend_overrides?.[row.feature_id] ?? {}
    bends[row.feature_id] = {
      angle: bandText(o.angle_deg), flangeA: bandText(o.flange_a_mm), flangeB: bandText(o.flange_b_mm),
      sideA: o.flange_inspection?.side_a ?? 'REQUIRED', sideB: o.flange_inspection?.side_b ?? 'REQUIRED',
      manualAngle: o.manual_included_angle_deg != null ? num(o.manual_included_angle_deg) : '',
    }
  }
  const measurements: Record<string, MeasurementRowForm> = {}
  for (const id of summary.measurementIds) {
    const o = request?.measurement_overrides?.[id]
    measurements[id] = { tol: bandText(o?.tolerance), required: typeof o?.required === 'boolean' ? o.required : null }
  }
  return {
    angle: bandText(d?.angle_deg), flange: bandText(d?.flange_mm), bendLine: bandText(d?.bend_line_position_mm),
    backgauge: bandText(d?.backgauge_mm), general: bandText(d?.general_mm),
    surface: d ? num(d.surface_profile_mm) : String(SURFACE_BAND_DISPLAY_DEFAULT_MM), source: d?.source ?? '', note: d?.note ?? '',
    stage: request?.inspection_stage === 'IN_PROCESS' ? 'IN_PROCESS' : 'FINAL',
    thickness: request?.thickness_mm != null ? num(request.thickness_mm) : '',
    releaseNote: request?.release_note ?? '',
    bends, measurements,
  }
}

function parseNumber(text: string): number {
  const trimmed = text.trim().replace(',', '.')
  return trimmed === '' ? Number.NaN : Number(trimmed)
}

export type BandResult = { value: ToleranceV1 | null; problems: ToleranceProblem[]; empty: boolean }

/** Read a band; `symmetric` copies plus into minus. */
export function readBand(band: BandText, max: number): BandResult {
  const minusText = band.symmetric ? band.plus : band.minus
  const empty = band.plus.trim() === '' && minusText.trim() === ''
  if (empty) return { value: null, problems: [], empty }
  const value = { plus: parseNumber(band.plus), minus: parseNumber(minusText) }
  const problems = validateTolerance(value, max)
  return { value: problems.length ? null : value, problems, empty }
}

export interface FormResult {
  request: ReleaseRequestV1
  /** field path -> problem codes; empty when the form is valid. */
  errors: Record<string, string[]>
}

export function requestFromForm(form: ToleranceForm, summary: VendorSummary, releaseDraftRevision: number | null): FormResult {
  const errors: Record<string, string[]> = {}
  const need = (path: string, band: BandText, max: number): ToleranceV1 | null => {
    const r = readBand(band, max)
    if (r.empty) { errors[path] = ['NOT_FINITE']; return null }
    if (r.problems.length) errors[path] = r.problems
    return r.value
  }
  const maybe = (path: string, band: BandText, max: number): ToleranceV1 | null => {
    const r = readBand(band, max)
    if (r.problems.length) errors[path] = r.problems
    return r.value
  }
  const angle = need('angle', form.angle, TOLERANCE_LIMITS.angle_deg)
  const flange = need('flange', form.flange, TOLERANCE_LIMITS.length_mm)
  const bendLine = maybe('bendLine', form.bendLine, TOLERANCE_LIMITS.length_mm)
  const backgauge = maybe('backgauge', form.backgauge, TOLERANCE_LIMITS.length_mm)
  const general = maybe('general', form.general, TOLERANCE_LIMITS.length_mm)
  const surface = parseNumber(form.surface)
  if (!(Number.isFinite(surface) && surface > 0)) errors.surface = ['NOT_FINITE']
  if (!form.source) errors.source = ['MISSING']
  let thickness: number | null = null
  if (form.thickness.trim()) {
    const t = parseNumber(form.thickness)
    if (Number.isFinite(t) && t > 0) thickness = t
    else errors.thickness = ['NOT_FINITE']
  }
  const defaults: DefaultTolerancesV1 | null = angle && flange && form.source && Number.isFinite(surface) && surface > 0 ? {
    angle_deg: angle, flange_mm: flange, bend_line_position_mm: bendLine, backgauge_mm: backgauge, general_mm: general,
    surface_profile_mm: surface, source: form.source, note: form.note.slice(0, 500),
  } : null

  const manual = new Set(summary.manualActiveIds)
  const bend_overrides: Record<string, BendOverrideV1> = {}
  for (const row of summary.ordered) {
    const f = form.bends[row.feature_id]
    if (!f) {
      if (manual.has(row.feature_id)) errors[`bends.${row.feature_id}.manualAngle`] = ['NOT_FINITE']
      continue
    }
    const o: BendOverrideV1 = {}
    const a = maybe(`bends.${row.feature_id}.angle`, f.angle, TOLERANCE_LIMITS.angle_deg); if (a) o.angle_deg = a
    const fa = maybe(`bends.${row.feature_id}.flangeA`, f.flangeA, TOLERANCE_LIMITS.length_mm); if (fa) o.flange_a_mm = fa
    const fb = maybe(`bends.${row.feature_id}.flangeB`, f.flangeB, TOLERANCE_LIMITS.length_mm); if (fb) o.flange_b_mm = fb
    if (f.sideA !== 'REQUIRED' || f.sideB !== 'REQUIRED') o.flange_inspection = { side_a: f.sideA, side_b: f.sideB }
    if (manual.has(row.feature_id)) {
      const v = parseNumber(f.manualAngle)
      if (Number.isFinite(v) && v > 0 && v < 180) o.manual_included_angle_deg = v
      else errors[`bends.${row.feature_id}.manualAngle`] = ['NOT_FINITE']
    }
    if (Object.keys(o).length) bend_overrides[row.feature_id] = o
  }
  const measurement_overrides: ReleaseRequestV1['measurement_overrides'] = {}
  for (const id of summary.measurementIds) {
    const f = form.measurements[id]
    if (!f) continue
    const tol = maybe(`measurements.${id}`, f.tol, TOLERANCE_LIMITS.length_mm)
    const o: { tolerance?: ToleranceV1; required?: boolean } = {}
    if (tol) o.tolerance = tol
    if (typeof f.required === 'boolean') o.required = f.required
    if (Object.keys(o).length) measurement_overrides[id] = o
  }
  return {
    request: {
      expected_draft_revision: summary.draftRevision,
      expected_release_draft_revision: releaseDraftRevision,
      default_tolerances: defaults,
      bend_overrides, measurement_overrides,
      thickness_mm: thickness,
      inspection_stage: form.stage,
      release_note: form.releaseNote.slice(0, 2000),
    },
    errors,
  }
}
