/** Pure V4 diagnostic ADD-ON, not standalone geometry/AVAILABLE admission.
 * Mirrors the reviewed Python finite-width ledger contract. Source hashes are
 * checked declarations; the browser cannot independently verify runtime pins.
 */
const GEOM_TOL = 1e-5
const MODEL = 'NATIVE_TESSELLATION_FINITE_AXIAL_SLAB'
const V4_SOURCE = '01ff5236499273d28e36a75ebeb6e34c8172de2a0ee1d6b99cfb416de27d59dc'
const LEGACY_SOURCE = '8412778cee16d97eb70454cdc76904be939125c9fcacf4159d18c49e150f8f07'
const WIDTH_SOURCE = '11d8cc1b912797a9a14b15b0c64d900566f07eaa536bacd28428142ca55615a2'
const WORK_KEYS = ['triangles', 'projected_vertices', 'clip_edge_tests'] as const
const WORK_CAPS = { triangles: 1_000_000, projected_vertices: 3_000_000, clip_edge_tests: 12_000_000 }
const EXTERNAL_GATES = ['native_ownership_and_complete_skin_coverage',
  'native_seam_identity_and_chart_closure', 'whole_unchanged_developed_mesh_overlap']
const LIMITATIONS = ['FINITE_WIDTH_GUARD_IS_NATIVE_TESSELLATION_ONLY',
  'WHOLE_UNCHANGED_FACE_OVERLAP_REMAINS_REQUIRED', 'TANGENT_SEAM_MUST_COVER_NATIVE_FINITE_AXIAL_SPAN']
const OBLIGATIONS = ['finite_width_clipping_checked', 'finite_width_native_seam_span_checked',
  'whole_unchanged_mesh_overlap_checked', 'seam_closure_checked', 'native_surface_ownership_checked',
  'cross_face_overlap_checked', 'native_surface_coverage_checked', 'excluded_caps_and_hole_walls_checked']
const ROW_KEYS = ['feature_id', 'cylinder_source_face_index', 'parent_source_face_index',
  'child_source_face_index', 'parent', 'child']
const CERTIFICATE_KEYS = ['status', 'model', 'side', 'geometry_mutated', 'manufacturing_ready',
  'exact_binary_clipping', 'tolerance_mm', 'axial_bounds', 'expanded_axial_bounds', 'intersecting_triangles',
  'max_oriented_side_distance_mm', 'charged_uncertainty_mm', 'work', 'shared_work', 'required_external_gates']
type RecordValue = Record<string, unknown>
type Work = Record<(typeof WORK_KEYS)[number], number>
type Dyadic = { n: bigint; exponent: number }
type Side = 'parent' | 'child'
type Event = { feature: string; side: Side; work: Work; shared: Work; parent: number; child: number }

function record(value: unknown): value is RecordValue {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}
function closed(value: unknown, keys: readonly string[]): value is RecordValue {
  if (!record(value)) return false
  const actual = Object.keys(value)
  return actual.length === keys.length && keys.every(key => actual.includes(key))
}
function finite(value: unknown): value is number { return typeof value === 'number' && Number.isFinite(value) }
function integer(value: unknown): value is number { return typeof value === 'number' && Number.isSafeInteger(value) }
function featureId(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0 && value.length <= 100 && value.trim().length > 0
}
function point(value: unknown): value is number[] {
  if (!Array.isArray(value) || value.length !== 3) return false
  for (let index = 0; index < 3; index++) if (!finite(value[index]) || Math.abs(value[index]) > 1e6) return false
  return true
}

const ieee = new DataView(new ArrayBuffer(8))
function bits(value: number): bigint { ieee.setFloat64(0, value); return ieee.getBigUint64(0) }
function dyadic(value: number): Dyadic {
  const raw = bits(value), encodedExponent = Number((raw >> 52n) & 0x7ffn)
  const mantissa = raw & ((1n << 52n) - 1n)
  const magnitude = encodedExponent === 0 ? mantissa : mantissa + (1n << 52n)
  return { n: raw >> 63n ? -magnitude : magnitude,
    exponent: encodedExponent === 0 ? -1074 : encodedExponent - 1075 }
}
function ulpDyadic(value: number): Dyadic {
  const encodedExponent = Number((bits(value) >> 52n) & 0x7ffn)
  return { n: 1n, exponent: encodedExponent === 0 ? -1074 : encodedExponent - 1075 }
}
function exactCompare(left: Dyadic[], right: Dyadic[]): number {
  const exponent = Math.min(...left.map(value => value.exponent), ...right.map(value => value.exponent))
  const sum = (values: Dyadic[]) => values.reduce((total, value) => total + (value.n << BigInt(value.exponent - exponent)), 0n)
  const a = sum(left), b = sum(right)
  return a < b ? -1 : a > b ? 1 : 0
}
function nextUp(value: number): number {
  if (value === 0) return Number.MIN_VALUE
  const raw = bits(value)
  ieee.setBigUint64(0, value > 0 ? raw + 1n : raw - 1n)
  return ieee.getFloat64(0)
}
const TOL = dyadic(GEOM_TOL)
const MAX_UNCERTAINTY = nextUp((GEOM_TOL / 8 + GEOM_TOL / 64) / (1 - 64 * Number.EPSILON))

function indices(value: unknown, count: number, nativeCount: number): Set<number> | null {
  if (!Array.isArray(value) || value.length !== count) return null
  const result = new Set<number>()
  for (const index of value) {
    if (!integer(index) || index < 0 || index >= nativeCount || result.has(index)) return null
    result.add(index)
  }
  return result
}
function work(value: unknown, local = false): Work | null {
  if (!closed(value, WORK_KEYS)) return null
  for (const key of WORK_KEYS) if (!integer(value[key]) || value[key] < 0 || value[key] > WORK_CAPS[key]) return null
  const result = value as Work
  if (local && (result.triangles < 1 || result.triangles > 200_000 || result.projected_vertices < 3
      || result.projected_vertices > 600_000 || result.clip_edge_tests < 3 * result.triangles
      || result.clip_edge_tests > 7 * result.triangles)) return null
  return result
}
function certificate(value: unknown, side: Side): { work: Work; shared: Work; bounds: number[] } | null {
  if (!closed(value, CERTIFICATE_KEYS) || value.status !== 'CERTIFIED' || value.model !== MODEL || value.side !== side
      || value.geometry_mutated !== false || value.manufacturing_ready !== false || value.exact_binary_clipping !== true
      || value.tolerance_mm !== GEOM_TOL) return null
  const bounds = value.axial_bounds, expanded = value.expanded_axial_bounds
  if (!Array.isArray(bounds) || bounds.length !== 2 || !finite(bounds[0]) || !finite(bounds[1])
      || !Array.isArray(expanded) || expanded.length !== 2 || !finite(expanded[0]) || !finite(expanded[1])) return null
  const low = bounds[0], high = bounds[1]
  if (exactCompare([dyadic(high), dyadic(-low)], [{ n: 4n * TOL.n, exponent: TOL.exponent }]) <= 0
      || exactCompare([ulpDyadic(low)], [{ n: TOL.n, exponent: TOL.exponent - 6 }]) > 0
      || exactCompare([ulpDyadic(high)], [{ n: TOL.n, exponent: TOL.exponent - 6 }]) > 0
      // Both Python's float(Fraction(low) +/- Fraction(tol)) and these binary64
      // operations are round-to-nearest IEEE results. No expansion epsilon.
      || expanded[0] !== low - GEOM_TOL || expanded[1] !== high + GEOM_TOL) return null
  const maximum = value.max_oriented_side_distance_mm, uncertainty = value.charged_uncertainty_mm
  if (!finite(maximum) || !finite(uncertainty) || uncertainty < 0 || uncertainty > MAX_UNCERTAINTY || maximum > GEOM_TOL) return null
  // Compare exact binary values. Each helper report is rounded UP by at most
  // its one serialized ULP; this is not a geometry tolerance or new epsilon.
  if (exactCompare([dyadic(maximum), dyadic(uncertainty)], [TOL, ulpDyadic(maximum), ulpDyadic(uncertainty)]) > 0) return null
  const local = work(value.work, true), cumulative = work(value.shared_work)
  if (!local || !cumulative || !integer(value.intersecting_triangles) || value.intersecting_triangles < 1
      || value.intersecting_triangles > local.triangles) return null
  const gates = value.required_external_gates
  if (!Array.isArray(gates) || gates.length !== EXTERNAL_GATES.length || !gates.every(gate => typeof gate === 'string')
      || new Set(gates).size !== EXTERNAL_GATES.length || !EXTERNAL_GATES.every(gate => gates.includes(gate))) return null
  return { work: local, shared: cumulative, bounds }
}

/** Validate only V4 declarations; the caller MUST retain all common body,
 * Double/Float32 marker, source/candidate, coverage and false-approval gates.
 * JSON loses integer lexical forms (4 vs 4.0); check bounded integer values.
 */
export function validUnfoldFiniteWidthDiagnostics(diag: unknown, candidates: unknown): boolean {
  try {
    if (!record(diag) || !Array.isArray(candidates) || candidates.length < 1 || candidates.length > 32) return false
    const ids = new Set<string>()
    for (const row of candidates) {
      if (!record(row) || !featureId(row.feature_id) || ids.has(row.feature_id) || !point(row.line_start) || !point(row.line_end)
          || Math.hypot(row.line_start[0] - row.line_end[0], row.line_start[1] - row.line_end[1], row.line_start[2] - row.line_end[2]) <= 1e-9) return false
      ids.add(row.feature_id)
    }
    if (diag.mechanism_version !== 4 || diag.transport_version !== 3 || diag.finite_width_model !== MODEL
        || diag.v4_engine_source_sha256 !== V4_SOURCE || diag.legacy_engine_source_sha256 !== LEGACY_SOURCE
        || diag.finite_width_source_sha256 !== WIDTH_SOURCE || diag.candidate_count !== ids.size
        || !OBLIGATIONS.every(key => diag[key] === true)) return false
    for (const key of ['transitive_runtime_closure_verified', 'hostile_upload_sandbox_verified']) {
      if (key in diag && diag[key] !== false) return false
    }
    const limitations = diag.limitations
    if (!Array.isArray(limitations) || limitations.length > 64 || !limitations.every(value => typeof value === 'string' && value.length > 0 && value.length <= 256)
        || !LIMITATIONS.every(value => limitations.includes(value))) return false
    const nativeCount = diag.native_face_count
    if (!integer(nativeCount) || nativeCount < 2 * (2 * ids.size + 1) || nativeCount > 1024) return false
    const selected = indices(diag.selected_source_face_indices, 2 * ids.size + 1, nativeCount)
    const cylinders = indices(diag.selected_cylinder_face_indices, ids.size, nativeCount)
    if (!selected || !cylinders || ![...cylinders].every(index => selected.has(index))) return false
    const planes = new Set([...selected].filter(index => !cylinders.has(index))), root = diag.root_source_face_index
    if (!integer(root) || !planes.has(root) || diag.selected_skin_rule !== 'LARGEST_PLANAR_ROOT_OF_COMPLETE_TREE_SKIN'
        || !Array.isArray(diag.finite_width_proofs) || diag.finite_width_proofs.length !== ids.size) return false
    const features = new Set<string>(), usedCylinders = new Set<number>(), children = new Set<number>(), events: Event[] = []
    for (const row of diag.finite_width_proofs) {
      if (!closed(row, ROW_KEYS) || !featureId(row.feature_id) || !ids.has(row.feature_id) || features.has(row.feature_id)
          || !integer(row.cylinder_source_face_index) || !cylinders.has(row.cylinder_source_face_index) || usedCylinders.has(row.cylinder_source_face_index)
          || !integer(row.parent_source_face_index) || !integer(row.child_source_face_index)
          || !planes.has(row.parent_source_face_index) || !planes.has(row.child_source_face_index)
          || row.parent_source_face_index === row.child_source_face_index || row.child_source_face_index === root || children.has(row.child_source_face_index)) return false
      const parent = certificate(row.parent, 'parent'), child = certificate(row.child, 'child')
      if (!parent || !child || parent.bounds[0] !== child.bounds[0] || parent.bounds[1] !== child.bounds[1]) return false
      features.add(row.feature_id); usedCylinders.add(row.cylinder_source_face_index); children.add(row.child_source_face_index)
      for (const [side, cert] of [['parent', parent], ['child', child]] as const) {
        events.push({ feature: row.feature_id, side, work: cert.work, shared: cert.shared,
          parent: row.parent_source_face_index, child: row.child_source_face_index })
      }
    }
    if (features.size !== ids.size || usedCylinders.size !== cylinders.size || children.size !== planes.size - 1) return false
    const global = work(diag.finite_width_work)
    if (!global) return false
    // Feature-sorted rows are not chronological. Positive cumulative triangle
    // counts reconstruct the sole shared budget, including parent/child order.
    events.sort((a, b) => a.shared.triangles - b.shared.triangles)
    let previous: Work = { triangles: 0, projected_vertices: 0, clip_edge_tests: 0 }
    const developed = new Set([root])
    for (let index = 0; index < events.length; index++) {
      const event = events[index]
      if (!WORK_KEYS.every(key => previous[key] + event.work[key] === event.shared[key])) return false
      if (index % 2 === 0) {
        if (event.side !== 'parent' || !developed.has(event.parent) || developed.has(event.child)) return false
      } else {
        const prior = events[index - 1]
        if (event.side !== 'child' || event.feature !== prior.feature || event.parent !== prior.parent || event.child !== prior.child) return false
        developed.add(event.child)
      }
      previous = event.shared
    }
    return WORK_KEYS.every(key => previous[key] === global[key]) && developed.size === planes.size
  } catch {
    return false
  }
}
