/**
 * Left/right (chirality) map of a released part, display only.
 *
 * Mirror of `sherman_chirality_status_v0` (`apps/production_line/recipes/chirality.py`), served by
 * `GET /api/production/parts/{id}/chirality` (and, read-only, `/recipes/{recipe_id}/chirality`). The
 * summary is `chirality_summary_v0` from the research map: the part class of the final stage and the
 * features that tell the part from its mirror image, in the CAD model frame (mm). Nothing here is an
 * inspection result; the engine does not read the map yet.
 */
import type { OnbCopy } from '../copy'

export type ChiralityPartClass = 'ACHIRAL' | 'CHIRAL_GROSS' | 'CHIRAL_BY_FEATURES'
export type ChiralityFeatureType = 'HOLE' | 'CUTOUT' | 'NOTCH_OR_TAB' | 'OPENING' | 'FORM' | 'UNCLASSIFIED'
export type ChiralityState = 'READY' | 'PENDING' | 'FAILED' | 'NOT_COMPUTED'
export type Point3 = [number, number, number]

export interface ChiralityFeature {
  feature: number
  type: ChiralityFeatureType
  /** Where the opening is: on this part ("proper"), only on the mirror image ("mirror"), or n/a. */
  opening_in: 'proper' | 'mirror' | null
  size_mm: number[] | null
  panel_id: string | null
  centre_final_mm: Point3 | null
  max_deviation_mm: number | null
  verifiable: boolean
}

export interface ChiralitySummary {
  part_class: ChiralityPartClass
  left_right_sensitive: boolean
  stage_classes: Record<string, string>
  features: ChiralityFeature[]
  verifiable: boolean
  map_sha256: string | null
}

export interface ChiralityStatus {
  status: ChiralityState
  reason: string | null
  recipe_id: string | null
  compiled_part_sha256: string | null
  summary: ChiralitySummary | null
}

const CLASSES: readonly string[] = ['ACHIRAL', 'CHIRAL_GROSS', 'CHIRAL_BY_FEATURES']
const TYPES: readonly string[] = ['HOLE', 'CUTOUT', 'NOTCH_OR_TAB', 'OPENING', 'FORM', 'UNCLASSIFIED']
const STATES: readonly string[] = ['READY', 'PENDING', 'FAILED', 'NOT_COMPUTED']

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const str = (v: unknown): string | null => (typeof v === 'string' ? v : null)

function parsePoint(v: unknown): Point3 | null {
  if (!Array.isArray(v) || v.length !== 3) return null
  const p = v.map(num)
  return p.every(x => x !== null) ? (p as Point3) : null
}

function parseFeature(v: unknown): ChiralityFeature | null {
  if (!v || typeof v !== 'object') return null
  const f = v as Record<string, unknown>
  const size = Array.isArray(f.size_mm) ? f.size_mm.map(num).filter((x): x is number => x !== null) : null
  return {
    feature: num(f.feature) ?? 0,
    type: (TYPES.includes(String(f.type)) ? f.type : 'UNCLASSIFIED') as ChiralityFeatureType,
    opening_in: f.opening_in === 'proper' || f.opening_in === 'mirror' ? f.opening_in : null,
    size_mm: size && size.length ? size : null,
    panel_id: str(f.panel_id),
    centre_final_mm: parsePoint(f.centre_final_mm),
    max_deviation_mm: num(f.max_deviation_mm),
    verifiable: f.verifiable === true,
  }
}

function parseSummary(v: unknown): ChiralitySummary | null {
  if (!v || typeof v !== 'object') return null
  const s = v as Record<string, unknown>
  if (!CLASSES.includes(String(s.part_class))) return null
  const partClass = s.part_class as ChiralityPartClass
  return {
    part_class: partClass,
    // Derived from the class, never trusted separately (the two must not disagree on screen).
    left_right_sensitive: partClass !== 'ACHIRAL',
    stage_classes: s.stage_classes && typeof s.stage_classes === 'object' ? (s.stage_classes as Record<string, string>) : {},
    features: Array.isArray(s.features) ? s.features.map(parseFeature).filter((f): f is ChiralityFeature => f !== null) : [],
    verifiable: s.verifiable === true,
    map_sha256: str(s.map_sha256),
  }
}

/** The status document, or null when the body is not one. A READY status without a valid summary is FAILED. */
export function parseChiralityStatus(body: unknown): ChiralityStatus | null {
  if (!body || typeof body !== 'object') return null
  const b = body as Record<string, unknown>
  if (b.schema !== 'sherman_chirality_status_v0' || !STATES.includes(String(b.status))) return null
  const summary = parseSummary(b.summary)
  const status = b.status === 'READY' && !summary ? 'FAILED' : (b.status as ChiralityState)
  return {
    status,
    reason: status === 'FAILED' && b.status === 'READY' ? 'CHIRALITY_OUTPUT_INVALID' : str(b.reason),
    recipe_id: str(b.recipe_id),
    compiled_part_sha256: str(b.compiled_part_sha256),
    summary: status === 'READY' ? summary : null,
  }
}

export function isLeftRightSensitive(status: ChiralityStatus | null | undefined): boolean {
  return !!status && status.status === 'READY' && !!status.summary?.left_right_sensitive
}

/** Station badge rule: only parts told apart by features (a gross mirror does not register as the part). */
export function needsEyeCheck(status: ChiralityStatus | null | undefined): boolean {
  return !!status && status.status === 'READY' && status.summary?.part_class === 'CHIRAL_BY_FEATURES'
}

function fmt(n: number): string {
  return Math.abs(n - Math.round(n)) < 0.05 ? String(Math.round(n)) : n.toFixed(1)
}

/** "Cutout 28 × 15 mm · flange P2 · on this part" in the operator's language. */
export function featureText(f: ChiralityFeature, t: OnbCopy): string {
  const parts = [t[`chiralityType${f.type}` as const]]
  // The size is isolated left-to-right (U+2066 ... U+2069) so "28 × 15" keeps its order in Hebrew.
  if (f.size_mm) parts[0] += ` \u2066${f.size_mm.map(fmt).join(' × ')}\u2069 ${t.chiralityMm}`
  if (f.panel_id) parts.push(t.chiralityFlange.replace('{p}', f.panel_id))
  if (f.opening_in) parts.push(f.opening_in === 'proper' ? t.chiralityOnPart : t.chiralityOnMirror)
  return parts.join(' · ')
}

/** Viewer markers (CAD model frame, mm) for the features with a position, numbered as in the list. */
export function featureMarkers(status: ChiralityStatus | null | undefined, t: OnbCopy): { id: string; point: Point3; label: string }[] {
  if (!isLeftRightSensitive(status)) return []
  return status!.summary!.features.flatMap((f, i) => (f.centre_final_mm
    ? [{ id: `chirality:${f.feature}`, point: f.centre_final_mm, label: t.chiralityMarker.replace('{n}', String(i + 1)) }]
    : []))
}
