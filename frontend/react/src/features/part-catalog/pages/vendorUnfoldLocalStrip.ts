/** Pure V5 local finite-strip add-on; common mesh, lineage and oracle admission
 * remain the caller's responsibility. Browser hashes are checked declarations,
 * not independently observed runtime source identities. */
const TOLERANCE = 1e-5
const MODEL = 'NATIVE_TESSELLATED_FLANGE_FINITE_DEVELOPED_STRIP_WITH_LOCAL_SEAM_SIDE'
const LOCAL_SOURCE = '7268fede9a8817b86a18f749d6fb682088a709a5aae4fc0c31b6c00066c14fad'
const V4_COMPONENT = '01ff5236499273d28e36a75ebeb6e34c8172de2a0ee1d6b99cfb416de27d59dc'
const WORK_KEYS = ['triangles', 'projected_vertices', 'clip_edge_tests', 'seam_edge_tests'] as const
const CAPS = { triangles: 1_000_000, projected_vertices: 3_000_000, clip_edge_tests: 20_000_000, seam_edge_tests: 3_000_000 }
const OBLIGATIONS = ['finite_width_clipping_checked', 'finite_width_native_seam_span_checked',
  'whole_unchanged_mesh_overlap_checked', 'seam_closure_checked', 'native_surface_ownership_checked',
  'cross_face_overlap_checked', 'native_surface_coverage_checked', 'excluded_caps_and_hole_walls_checked']
const LIMITATIONS = ['LOCAL_FINITE_STRIP_GUARD_IS_NATIVE_TESSELLATION_ONLY',
  'LOCAL_STRIP_WIDTH_IS_SELECTED_NATIVE_RADIUS_TIMES_SWEEP', 'LOCAL_NATIVE_SEAM_BOUNDARY_AND_MATERIAL_SIDE_REQUIRED',
  'WHOLE_UNCHANGED_FACE_OVERLAP_REMAINS_REQUIRED', 'TANGENT_SEAM_MUST_COVER_NATIVE_FINITE_AXIAL_SPAN']
const EXTERNAL_GATES = ['native_seam_identity_and_same_radius_sweep_strip_width',
  'native_ownership_and_complete_skin_material_coverage', 'proper_chart_transform_and_native_seam_closure',
  'whole_unchanged_developed_mesh_overlap']
const ROW_KEYS = ['feature_id', 'cylinder_source_face_index', 'parent_source_face_index',
  'child_source_face_index', 'parent', 'child']
const CERT_KEYS = ['status', 'mechanism_version', 'model', 'side', 'geometry_mutated', 'manufacturing_ready',
  'exact_binary_clipping', 'tolerance_mm', 'axial_bounds', 'developed_strip_width_mm',
  'certified_strip_intrusion_bounds', 'charged_uncertainty_mm', 'intersecting_axial_triangles',
  'retained_remote_side_triangles', 'native_seam_boundary_interval_count', 'native_seam_local_material_side_checked',
  'whole_supplied_mesh_finite_strip_checked', 'work', 'shared_work', 'required_external_gates', 'radius_mm', 'sweep_radians']
type Row = Record<string, unknown>
type Work = Record<(typeof WORK_KEYS)[number], number>
type Dyadic = { n: bigint; exponent: number }
type Side = 'parent' | 'child'
type Certificate = { work: Work; shared: Work; bounds: number[]; radius: number; sweep: number; width: number }
type Event = { ident: string; side: Side; local: Work; shared: Work; parent: number; child: number }

function record(value: unknown): value is Row { return value !== null && typeof value === 'object' && !Array.isArray(value) }
function closed(value: unknown, keys: readonly string[]): value is Row {
  return record(value) && Object.keys(value).length === keys.length && keys.every(key => key in value)
}
function finite(value: unknown): value is number { return typeof value === 'number' && Number.isFinite(value) }
function integer(value: unknown): value is number { return typeof value === 'number' && Number.isSafeInteger(value) }
function feature(value: unknown): value is string { return typeof value === 'string' && value.length > 0 && value.length <= 100 && value.trim().length > 0 }
function point(value: unknown): value is number[] {
  return Array.isArray(value) && value.length === 3 && value.every(v => finite(v) && Math.abs(v) <= 1e6)
}

const ieee = new DataView(new ArrayBuffer(8))
function bits(value: number): bigint { ieee.setFloat64(0, value); return ieee.getBigUint64(0) }
function dyadic(value: number): Dyadic {
  const raw = bits(value), encoded = Number((raw >> 52n) & 0x7ffn)
  const mantissa = raw & ((1n << 52n) - 1n)
  const magnitude = encoded === 0 ? mantissa : mantissa + (1n << 52n)
  return { n: raw >> 63n ? -magnitude : magnitude, exponent: encoded === 0 ? -1074 : encoded - 1075 }
}
function ulp(value: number): Dyadic {
  const encoded = Number((bits(value) >> 52n) & 0x7ffn)
  return { n: 1n, exponent: encoded === 0 ? -1074 : encoded - 1075 }
}
function negate(value: Dyadic): Dyadic { return { n: -value.n, exponent: value.exponent } }
function multiply(a: Dyadic, b: Dyadic): Dyadic { return { n: a.n * b.n, exponent: a.exponent + b.exponent } }
function compare(a: Dyadic[], b: Dyadic[]): number {
  const exponent = Math.min(...a.map(v => v.exponent), ...b.map(v => v.exponent))
  const sum = (values: Dyadic[]) => values.reduce((n, v) => n + (v.n << BigInt(v.exponent - exponent)), 0n)
  const av = sum(a), bv = sum(b)
  return av < bv ? -1 : av > bv ? 1 : 0
}
function oneUlp(value: number, expected: number): boolean {
  return finite(expected) && compare([dyadic(value), dyadic(-expected)], [ulp(expected)]) <= 0
    && compare([dyadic(expected), dyadic(-value)], [ulp(expected)]) <= 0
}
function nextUp(value: number): number {
  if (value === 0) return Number.MIN_VALUE
  ieee.setBigUint64(0, value > 0 ? bits(value) + 1n : bits(value) - 1n)
  return ieee.getFloat64(0)
}
const TOL = dyadic(TOLERANCE)
const FRAME_ERROR = multiply(TOL, dyadic(64 * Number.EPSILON))
const MAX_UNCERTAINTY = nextUp(9 * TOLERANCE / 64)

function indices(value: unknown, count: number, nativeCount: number): Set<number> | null {
  if (!Array.isArray(value) || value.length !== count || !value.every(v => integer(v) && v >= 0 && v < nativeCount)) return null
  return new Set(value).size === count ? new Set(value) : null
}
function work(value: unknown, local = false): Work | null {
  if (!closed(value, WORK_KEYS) || !WORK_KEYS.every(key => integer(value[key]) && value[key] >= 0 && value[key] <= CAPS[key])) return null
  const w = value as Work
  if (local && !(w.triangles >= 1 && w.triangles <= 200_000 && w.projected_vertices >= 3 && w.projected_vertices <= 600_000
      && w.clip_edge_tests >= 3 * w.triangles && w.clip_edge_tests <= 18 * w.triangles && w.seam_edge_tests === 3 * w.triangles)) return null
  return w
}
function certificate(value: unknown, side: Side): Certificate | null {
  if (!closed(value, CERT_KEYS) || value.status !== 'CERTIFIED' || value.mechanism_version !== 1 || value.model !== MODEL
      || value.side !== side || value.geometry_mutated !== false || value.manufacturing_ready !== false
      || value.exact_binary_clipping !== true || value.native_seam_local_material_side_checked !== true
      || value.whole_supplied_mesh_finite_strip_checked !== true || value.tolerance_mm !== TOLERANCE) return null
  const bounds = value.axial_bounds
  if (!Array.isArray(bounds) || bounds.length !== 2 || !finite(bounds[0]) || !finite(bounds[1])) return null
  const [low, high] = bounds
  if (compare([dyadic(high), dyadic(-low)], [{ n: 4n * TOL.n, exponent: TOL.exponent }]) <= 0
      || compare([ulp(low)], [{ n: TOL.n, exponent: TOL.exponent - 6 }]) > 0
      || compare([ulp(high)], [{ n: TOL.n, exponent: TOL.exponent - 6 }]) > 0) return null
  const radius = value.radius_mm, sweep = value.sweep_radians, width = value.developed_strip_width_mm
  if (!finite(radius) || !finite(sweep) || !finite(width) || radius <= TOLERANCE || radius > 1e6
      || sweep <= TOLERANCE || sweep > 175 * (Math.PI / 180) + TOLERANCE
      || compare([dyadic(width)], [{ n: 4n * TOL.n, exponent: TOL.exponent }]) <= 0
      || compare([ulp(width)], [{ n: TOL.n, exponent: TOL.exponent - 6 }]) > 0 || !oneUlp(width, radius * sweep)) return null
  const uncertainty = value.charged_uncertainty_mm
  if (!finite(uncertainty) || uncertainty < 0 || uncertainty > MAX_UNCERTAINTY) return null
  const intrusion = value.certified_strip_intrusion_bounds
  if (!Array.isArray(intrusion) || intrusion.length !== 2 || !finite(intrusion[0]) || !finite(intrusion[1])) return null
  const [lower, upper] = intrusion, u = dyadic(uncertainty)
  if (!(0 < lower && lower < upper)
      || compare([dyadic(lower)], [TOL, negate(FRAME_ERROR), negate(u)]) < 0
      || compare([dyadic(lower)], [TOL, negate(u), ulp(uncertainty), ulp(lower)]) > 0
      || compare([dyadic(upper)], [dyadic(width), TOL, u]) > 0
      || compare([dyadic(upper)], [dyadic(width), TOL, u, negate(ulp(uncertainty)), negate(ulp(upper))]) < 0) return null
  const local = work(value.work, true), shared = work(value.shared_work)
  const intersecting = value.intersecting_axial_triangles, remote = value.retained_remote_side_triangles, intervals = value.native_seam_boundary_interval_count
  if (!local || !shared || !integer(intersecting) || intersecting < 1 || intersecting > local.triangles
      || !integer(remote) || remote < 0 || remote > intersecting || !integer(intervals) || intervals < 1 || intervals > local.seam_edge_tests) return null
  const gates = value.required_external_gates
  if (!Array.isArray(gates) || gates.length !== EXTERNAL_GATES.length || !gates.every(v => typeof v === 'string')
      || new Set(gates).size !== EXTERNAL_GATES.length || !EXTERNAL_GATES.every(gate => gates.includes(gate))) return null
  return { work: local, shared, bounds, radius, sweep, width }
}

/** Requires actual exported bend lines to bind their sweep to the same native
 * cylinder used for both side guards. JSON integers are checked by value. */
export function validUnfoldLocalStripDiagnostics(diag: unknown, candidates: unknown, bendLines: unknown): boolean {
  try {
    if (!record(diag) || !Array.isArray(candidates) || candidates.length < 1 || candidates.length > 32) return false
    const ids = new Set<string>()
    for (const row of candidates) {
      if (!record(row) || !feature(row.feature_id) || ids.has(row.feature_id) || !point(row.line_start) || !point(row.line_end)
          || Math.hypot(...row.line_start.map((v, i) => v - (row.line_end as number[])[i])) <= 1e-9) return false
      ids.add(row.feature_id)
    }
    if (diag.mechanism_version !== 5 || diag.finite_width_model !== MODEL || diag.local_strip_source_sha256 !== LOCAL_SOURCE
        || diag.v4_component_source_sha256 !== V4_COMPONENT || diag.candidate_count !== ids.size || !OBLIGATIONS.every(k => diag[k] === true)) return false
    const limitations = diag.limitations
    if (!Array.isArray(limitations) || limitations.length > 64 || !limitations.every(v => typeof v === 'string' && v.length > 0 && v.length <= 256)
        || !LIMITATIONS.every(v => limitations.includes(v))) return false
    const count = diag.native_face_count
    if (!integer(count) || count < 2 * (2 * ids.size + 1) || count > 1024) return false
    const selected = indices(diag.selected_source_face_indices, 2 * ids.size + 1, count)
    const cylinders = indices(diag.selected_cylinder_face_indices, ids.size, count)
    if (!selected || !cylinders || ![...cylinders].every(v => selected.has(v))) return false
    const planes = new Set([...selected].filter(v => !cylinders.has(v))), root = diag.root_source_face_index
    if (!integer(root) || !planes.has(root) || diag.selected_skin_rule !== 'LARGEST_PLANAR_ROOT_OF_COMPLETE_TREE_SKIN') return false
    if (!Array.isArray(bendLines) || bendLines.length !== ids.size) return false
    const sweeps = new Map<string, number>()
    for (const line of bendLines) {
      if (!record(line) || !feature(line.feature_id) || !ids.has(line.feature_id) || sweeps.has(line.feature_id) || !finite(line.sweep_deg)) return false
      sweeps.set(line.feature_id, line.sweep_deg)
    }
    if (!Array.isArray(diag.finite_width_proofs) || diag.finite_width_proofs.length !== ids.size) return false
    const features = new Set<string>(), used = new Set<number>(), children = new Set<number>(), events: Event[] = []
    for (const row of diag.finite_width_proofs) {
      if (!closed(row, ROW_KEYS) || !feature(row.feature_id) || !ids.has(row.feature_id) || features.has(row.feature_id)
          || !integer(row.cylinder_source_face_index) || !cylinders.has(row.cylinder_source_face_index) || used.has(row.cylinder_source_face_index)
          || !integer(row.parent_source_face_index) || !integer(row.child_source_face_index)
          || !planes.has(row.parent_source_face_index) || !planes.has(row.child_source_face_index)
          || row.parent_source_face_index === row.child_source_face_index || row.child_source_face_index === root || children.has(row.child_source_face_index)) return false
      const a = certificate(row.parent, 'parent'), b = certificate(row.child, 'child')
      if (!a || !b || a.bounds[0] !== b.bounds[0] || a.bounds[1] !== b.bounds[1] || a.radius !== b.radius
          || a.sweep !== b.sweep || a.width !== b.width || !oneUlp(sweeps.get(row.feature_id)!, a.sweep * (180 / Math.PI))) return false
      features.add(row.feature_id); used.add(row.cylinder_source_face_index); children.add(row.child_source_face_index)
      for (const [side, cert] of [['parent', a], ['child', b]] as const) {
        events.push({ ident: row.feature_id, side, local: cert.work, shared: cert.shared, parent: row.parent_source_face_index, child: row.child_source_face_index })
      }
    }
    if (features.size !== ids.size || used.size !== cylinders.size || children.size !== planes.size - 1) return false
    const total = work(diag.finite_width_work)
    if (!total) return false
    events.sort((a, b) => a.shared.triangles - b.shared.triangles)
    let previous: Work = { triangles: 0, projected_vertices: 0, clip_edge_tests: 0, seam_edge_tests: 0 }
    const developed = new Set([root])
    for (let i = 0; i < events.length; i++) {
      const e = events[i]
      if (!WORK_KEYS.every(k => previous[k] + e.local[k] === e.shared[k])) return false
      if (i % 2 === 0) {
        if (e.side !== 'parent' || !developed.has(e.parent) || developed.has(e.child)) return false
      } else {
        const prior = events[i - 1]
        if (e.side !== 'child' || e.ident !== prior.ident || e.parent !== prior.parent || e.child !== prior.child) return false
        developed.add(e.child)
      }
      previous = e.shared
    }
    return WORK_KEYS.every(k => previous[k] === total[k]) && developed.size === planes.size
  } catch {
    return false
  }
}
