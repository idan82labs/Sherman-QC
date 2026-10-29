/**
 * Pure helpers for the uploaded unfolded part (flat DXF / flat STEP).
 *
 * - Binding freshness, by the same rule C05 applies at release
 *   (`flat_intake.release_flat_ref`): an operator-confirmed binding counts only
 *   while the CAD sha256 and the set of bend identities it was confirmed against
 *   (`binding_basis_sha256`) are the current ones. Anything else is stale and is
 *   never shown as the unfolded view. A confirmed map may bind only some bends
 *   (FlatBindingV1 allows it); those bends are simply not drawn.
 * - Binding-map validation for the operator's confirm/edit action.
 * - 2D geometry for the SVG view, and an adapter that turns a bound flat into
 *   the `unfold_preview` shape Vendor Setup's viewer already draws.
 *
 * Flat coordinates are the file's own 2D frame, scaled to mm by the E1 parser.
 * They are not the CAD frame and are never compared numerically with CAD lines.
 * Every function here tolerates malformed input and never throws.
 */
import { bindingBasisSha256, bindingMarks, type BindingMark, type BindingMarkInput } from './bendIdentity'
import type { FlatBendLineV1, FlatIntakeV1, Point3, VendorSetupLite } from './types'

export type FlatFreshness = 'FRESH' | 'UNCONFIRMED' | 'DECLINED' | 'UNVERIFIED' | 'STALE_CAD' | 'MARKS_CHANGED' | 'LINES_CHANGED'

export interface FlatBindingContext {
  cadSha256: string | null | undefined
  /** The draft whose marks are checked (saved, or the one being edited in Vendor Setup). */
  draft: { marks?: readonly BindingMarkInput[] | null; sequence?: readonly string[] | null } | null | undefined
}

export function contextFromSetup(setup: VendorSetupLite | null | undefined): FlatBindingContext | null {
  if (!setup?.draft || !setup.part) return null
  return { cadSha256: setup.part.cad_file_hash, draft: setup.draft }
}

/** Marks with their bend_uid for a context (empty when it has no CAD hash or draft). */
export function contextMarks(ctx: FlatBindingContext | null | undefined): BindingMark[] {
  try { return ctx?.cadSha256 ? bindingMarks(ctx.draft, ctx.cadSha256) : [] } catch { return [] }
}

/** Freshness of the confirmed binding against an explicit draft context (C05 release rule). */
export function bindingFreshnessFor(intake: FlatIntakeV1 | null | undefined, ctx: FlatBindingContext | null): FlatFreshness {
  try {
    if (!intake || intake.parse?.status !== 'PARSED') return 'UNCONFIRMED'
    if (intake.state === 'DECLINED' && intake.binding?.state === 'UNBOUND') return 'DECLINED'
    const binding = intake.binding
    if (intake.state !== 'CONFIRMED' || binding?.state !== 'OPERATOR_CONFIRMED') return 'UNCONFIRMED'
    if (!ctx) return 'UNVERIFIED'
    if (!ctx.cadSha256 || binding.cad_sha256 !== ctx.cadSha256) return 'STALE_CAD'
    const marks = contextMarks(ctx)
    if (!marks.length || !binding.basis_sha256 || binding.basis_sha256 !== bindingBasisSha256(ctx.cadSha256, marks)) return 'MARKS_CHANGED'
    const lineIds = new Set((intake.parse.bend_lines ?? []).map(line => line.line_id))
    const uids = new Set(marks.map(mark => mark.bend_uid))
    const entries = Object.entries(binding.bend_line_map ?? {})
    if (!entries.length || entries.some(([lineId, uid]) => !lineIds.has(lineId) || !uids.has(uid))) return 'LINES_CHANGED'
    return 'FRESH'
  } catch {
    return 'UNCONFIRMED'
  }
}

export function flatBindingFreshness(intake: FlatIntakeV1 | null | undefined, setup: VendorSetupLite | null | undefined): FlatFreshness {
  return bindingFreshnessFor(intake, contextFromSetup(setup))
}

/** feature_id -> bound flat bend line, only when the binding is FRESH for `ctx`. */
export function boundFlatLinesByFeature(intake: FlatIntakeV1 | null | undefined, ctx: FlatBindingContext | null): Map<string, FlatBendLineV1> | null {
  if (!intake || bindingFreshnessFor(intake, ctx) !== 'FRESH' || !intake.binding) return null
  const lineById = new Map(intake.parse.bend_lines.map(line => [line.line_id, line]))
  const fidByUid = new Map(contextMarks(ctx).map(mark => [mark.bend_uid, mark.feature_id]))
  const out = new Map<string, FlatBendLineV1>()
  for (const [lineId, uid] of Object.entries(intake.binding.bend_line_map)) {
    const fid = fidByUid.get(uid), line = lineById.get(lineId)
    if (fid && line) out.set(fid, line)
  }
  return out
}

/** Marks (feature ids) a FRESH binding leaves without a flat line; empty when not fresh. */
export function unboundMarks(intake: FlatIntakeV1 | null | undefined, ctx: FlatBindingContext | null): string[] {
  const lines = boundFlatLinesByFeature(intake, ctx)
  if (!lines) return []
  return contextMarks(ctx).filter(mark => !lines.has(mark.feature_id)).map(mark => mark.feature_id)
}

/** Whether the stored proposal was made against the current CAD + marks (else it must be proposed again). */
export function proposalIsCurrent(intake: FlatIntakeV1 | null | undefined, ctx: FlatBindingContext | null): boolean {
  try {
    const record = intake?.proposal
    if (!record || record.status !== 'PROPOSED' || !record.proposal || !ctx?.cadSha256) return false
    const marks = contextMarks(ctx)
    return !!marks.length && record.cad_sha256 === ctx.cadSha256 && record.basis_sha256 === bindingBasisSha256(ctx.cadSha256, marks)
  } catch {
    return false
  }
}

// ---------------------------------------------------------------- binding map validation

export type BindingMapProblem = 'EMPTY' | 'UNKNOWN_LINE' | 'UNKNOWN_BEND' | 'DUPLICATE_BEND' | 'MIRROR_UNDECIDED'

/** A confirmable map binds known flat line ids to distinct, known bend_uids (FlatBindingV1 / C05 set_binding rule). */
export function validateBindingMap(map: Record<string, string>, lineIds: string[], bendUids: string[]): BindingMapProblem[] {
  const problems = new Set<BindingMapProblem>()
  const entries = Object.entries(map || {}).filter(([, uid]) => !!uid)
  if (!entries.length) problems.add('EMPTY')
  const lines = new Set(lineIds), uids = new Set(bendUids), seen = new Set<string>()
  for (const [lineId, uid] of entries) {
    if (!lines.has(lineId)) problems.add('UNKNOWN_LINE')
    if (!uids.has(uid)) problems.add('UNKNOWN_BEND')
    if (seen.has(uid)) problems.add('DUPLICATE_BEND')
    seen.add(uid)
  }
  return [...problems]
}

/** Drop unbound rows ('' = "no bend") so the request carries only real assignments. */
export function compactBindingMap(map: Record<string, string>): Record<string, string> {
  return Object.fromEntries(Object.entries(map || {}).filter(([, uid]) => !!uid).sort(([a], [b]) => a.localeCompare(b, 'en', { numeric: true })))
}

export function sameBindingMap(a: Record<string, string>, b: Record<string, string>): boolean {
  const ca = compactBindingMap(a), cb = compactBindingMap(b)
  const ka = Object.keys(ca), kb = Object.keys(cb)
  return ka.length === kb.length && ka.every(key => ca[key] === cb[key])
}

// ---------------------------------------------------------------- mirror decision

/** Operator's answer to "which side of the flat is up": as the proposal fitted it, or the mirror image. */
export type MirrorChoice = 'AS_PROPOSED' | 'FLIPPED' | null

/**
 * The `mirrored` flag to send with a confirmation.
 *
 * E1 lists alternative maps exactly when the mirror is not determined, so an
 * alternative (or an edited row) may belong to the opposite mirror. The compile
 * then checks the confirmed map against the fit of the confirmed mirror, and
 * C05 replaces a null flag with the proposal's. So when the mirror is not
 * determined and the map is not the proposal's, the operator must say which
 * side is up (`needsChoice`) before confirming; the answer is sent explicitly.
 */
export function mirrorForConfirm(
  proposal: { mirrored: boolean; bend_line_map: Record<string, string>; fit_choice?: { mirror_determined?: boolean } | null } | null | undefined,
  map: Record<string, string>, choice: MirrorChoice,
): { mirrored: boolean | null; needsChoice: boolean; undetermined: boolean } {
  if (!proposal) return { mirrored: choice === null ? null : choice === 'FLIPPED', needsChoice: false, undetermined: true }
  const undetermined = proposal.fit_choice?.mirror_determined !== true
  if (!undetermined) return { mirrored: proposal.mirrored, needsChoice: false, undetermined }
  if (choice === 'AS_PROPOSED') return { mirrored: proposal.mirrored, needsChoice: false, undetermined }
  if (choice === 'FLIPPED') return { mirrored: !proposal.mirrored, needsChoice: false, undetermined }
  if (sameBindingMap(map, proposal.bend_line_map)) return { mirrored: proposal.mirrored, needsChoice: false, undetermined }
  return { mirrored: null, needsChoice: true, undetermined }
}

// ---------------------------------------------------------------- 2D geometry

export type Vec2 = [number, number]

export interface FlatGeometry {
  outline: Vec2[] | null
  holes: Vec2[][]
  lines: Array<{ id: string; a: Vec2; b: Vec2; line: FlatBendLineV1 }>
  bbox: { minX: number; minY: number; maxX: number; maxY: number } | null
}

function vec2(value: unknown): Vec2 | null {
  if (!Array.isArray(value) || value.length < 2) return null
  const x = Number(value[0]), y = Number(value[1])
  return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null
}

function polygon(value: unknown): Vec2[] | null {
  if (!Array.isArray(value) || value.length < 3) return null
  const out: Vec2[] = []
  for (const point of value) {
    const p = vec2(point)
    if (!p) return null
    out.push(p)
  }
  return out
}

/** Validated 2D geometry; malformed entries are dropped, never repaired. */
export function flatGeometry(intake: FlatIntakeV1 | null | undefined): FlatGeometry {
  const parse = intake?.parse
  const outline = polygon(parse?.outline_mm)
  const holes = (parse?.inner_loops_mm || []).map(polygon).filter((loop): loop is Vec2[] => !!loop)
  const lines = (parse?.bend_lines || []).flatMap(line => {
    const a = vec2(line.start_mm), b = vec2(line.end_mm)
    return a && b && (a[0] !== b[0] || a[1] !== b[1]) ? [{ id: line.line_id, a, b, line }] : []
  })
  const points: Vec2[] = [...(outline || []), ...holes.flat(), ...lines.flatMap(line => [line.a, line.b])]
  if (!points.length) return { outline, holes, lines, bbox: null }
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const [x, y] of points) { minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y) }
  if (maxX - minX <= 0 && maxY - minY <= 0) return { outline, holes, lines, bbox: null }
  return { outline, holes, lines, bbox: { minX, minY, maxX, maxY } }
}

/** Uniform fit of the flat bbox into a w x h viewport (y up in mm, y down on screen). */
export function fitProjector(bbox: NonNullable<FlatGeometry['bbox']>, width: number, height: number, pad: number) {
  const spanX = Math.max(bbox.maxX - bbox.minX, 1e-9), spanY = Math.max(bbox.maxY - bbox.minY, 1e-9)
  const scale = Math.min((width - 2 * pad) / spanX, (height - 2 * pad) / spanY)
  const offsetX = (width - spanX * scale) / 2, offsetY = (height - spanY * scale) / 2
  return {
    scale,
    project: ([x, y]: Vec2): Vec2 => [offsetX + (x - bbox.minX) * scale, height - offsetY - (y - bbox.minY) * scale],
  }
}

// ---------------------------------------------------------------- Vendor Setup unfolded-view adapter

/** Structural mirror of `types/index.ts` VendorUnfoldPreview (kept local so this module stays pure). */
export interface FlatUnfoldPreview {
  status: 'AVAILABLE'
  reason: null
  model: 'SELECTED_CAD_SKIN_DEVELOPMENT'
  units: 'assumed_mm'
  units_verified: false
  manufacturing_ready: false
  source_sha256: string
  vertices: number[]
  triangles: number[]
  bend_lines: Array<{ feature_id: string; line_start: Point3; line_end: Point3; bend_direction: 'up' | 'down' | 'unknown'; sweep_deg: number }>
  flange_surfaces: Array<{ feature_id: string; vertices: number[]; triangles: number[] }>
  diagnostics: Record<string, unknown>
}

/** `(contour, holes) -> triangles as index triples into [...contour, ...holes.flat()]` (e.g. three's ShapeUtils.triangulateShape). */
export type Triangulator = (contour: Vec2[], holes: Vec2[][]) => number[][]

/**
 * Turn a FRESH operator-confirmed flat into the unfold-preview shape (z = 0 plane,
 * flat mm, re-centred). Returns null when the binding is not fresh, the outline is
 * missing, or triangulation fails, so the caller keeps its existing view.
 * `diagnostics.source = 'UPLOADED_FLAT'` marks the provenance; callers must not
 * describe it as a CAD development.
 */
export function boundFlatUnfoldPreview(
  intake: FlatIntakeV1 | null | undefined, ctx: FlatBindingContext | null, triangulate: Triangulator,
): FlatUnfoldPreview | null {
  try {
    return buildFlatUnfoldPreview(intake, ctx, triangulate)
  } catch {
    return null
  }
}

function buildFlatUnfoldPreview(
  intake: FlatIntakeV1 | null | undefined, ctx: FlatBindingContext | null, triangulate: Triangulator,
): FlatUnfoldPreview | null {
  const byFeature = boundFlatLinesByFeature(intake, ctx)
  if (!intake || !byFeature || !byFeature.size || !ctx?.cadSha256) return null
  const geometry = flatGeometry(intake)
  if (!geometry.outline || !geometry.bbox) return null
  const cx = (geometry.bbox.minX + geometry.bbox.maxX) / 2, cy = (geometry.bbox.minY + geometry.bbox.maxY) / 2
  const shift = ([x, y]: Vec2): Vec2 => [x - cx, y - cy]
  const contour = geometry.outline.map(shift)
  const holes = geometry.holes.map(loop => loop.map(shift))
  let faces: number[][]
  try { faces = triangulate(contour, holes) } catch { return null }
  const all = [...contour, ...holes.flat()]
  if (!Array.isArray(faces) || !faces.length || faces.some(face => face.length !== 3 || face.some(i => !Number.isInteger(i) || i < 0 || i >= all.length))) return null
  const direction = (value: string | null | undefined): 'up' | 'down' | 'unknown' => {
    const text = (value || '').toUpperCase()
    return text === 'UP' ? 'up' : text === 'DOWN' ? 'down' : 'unknown'
  }
  const bend_lines: FlatUnfoldPreview['bend_lines'] = []
  for (const [feature_id, line] of byFeature) {
    const a = shift(vec2(line.start_mm)!), b = shift(vec2(line.end_mm)!)
    // The flat carries no trusted sweep (annotations are free text); 0 means "not stated".
    bend_lines.push({ feature_id, line_start: [a[0], a[1], 0], line_end: [b[0], b[1], 0], bend_direction: direction(line.direction),
      sweep_deg: 0 })
  }
  return {
    status: 'AVAILABLE', reason: null, model: 'SELECTED_CAD_SKIN_DEVELOPMENT', units: 'assumed_mm', units_verified: false,
    manufacturing_ready: false, source_sha256: ctx.cadSha256,
    vertices: all.flatMap(([x, y]) => [x, y, 0]), triangles: faces.flat(), bend_lines, flange_surfaces: [],
    diagnostics: { source: 'UPLOADED_FLAT', flat_sha256: intake.flat.sha256, flat_format: intake.flat.format,
      binding_state: intake.binding?.state ?? null, confirmed_at: intake.binding?.confirmed_at ?? null },
  }
}

// ---------------------------------------------------------------- Vendor Setup draft guard

/** The Vendor Setup response fields the bridge checks (structural, so the page's own types fit). */
export interface BridgeSetupData {
  part: { cad_file_hash?: string | null; cad_uploaded_at?: string | null; revision?: string | null }
  candidate_generation?: string | null
  /** The saved draft as the server returned it; a stale one blocks the flat view until a fresh draft is saved. */
  draft?: { is_stale?: boolean } | null
}

/** The draft being edited in Vendor Setup (structural subset of VendorSetupDraft). */
export interface BridgeDraft {
  marks: readonly BindingMarkInput[]
  sequence?: readonly string[]
  is_stale?: boolean
  cad_file_hash?: string | null
  cad_uploaded_at?: string | null
  candidate_generation?: string | null
  part_revision?: string | null
}

/**
 * Whether the edited draft belongs to exactly this CAD upload, candidate
 * generation and part revision, and is not stale: the same guards Vendor Setup
 * applies before it shows its own CAD development (`canMapUnfoldDraft`). While
 * the saved server draft is stale (the page is editing a reset draft built from
 * the current candidates), the uploaded flat is not used either.
 */
export function draftMatchesSetup(data: BridgeSetupData | null | undefined, draft: BridgeDraft | null | undefined): boolean {
  if (!data?.part || !draft || draft.is_stale || data.draft?.is_stale) return false
  const cad = data.part.cad_file_hash
  if (typeof cad !== 'string' || !/^[0-9a-f]{64}$/.test(cad) || draft.cad_file_hash !== cad) return false
  if ((draft.cad_uploaded_at ?? null) !== (data.part.cad_uploaded_at ?? null)) return false
  if (!data.candidate_generation || draft.candidate_generation !== data.candidate_generation) return false
  return (draft.part_revision ?? null) === (data.part.revision ?? null)
}

// ---------------------------------------------------------------- wizard return context

/** Only an in-app onboarding route may be used as the "return to wizard" target. */
export function safeWizardReturn(raw: string | null | undefined, partId: string): string | null {
  if (!raw || !partId) return null
  const expectedPrefix = `/production/parts/${encodeURIComponent(partId)}/onboard`
  if (!raw.startsWith('/') || raw.startsWith('//') || raw.includes('\\')) return null
  const [path, query = ''] = raw.split('?', 2)
  if (path !== expectedPrefix) return null
  if (query && !/^step=[a-z]+$/.test(query)) return null
  return raw
}

export function wizardReturnHref(partId: string, step = 'bends'): string {
  return `/production/parts/${encodeURIComponent(partId)}/onboard?step=${step}`
}

/** Vendor Setup link that carries the return context. */
export function vendorSetupHref(partId: string, step = 'bends'): string {
  return `/parts/vendor-setup/${encodeURIComponent(partId)}?return=${encodeURIComponent(wizardReturnHref(partId, step))}`
}
