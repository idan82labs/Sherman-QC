import type { VendorUnfoldPreview } from '../../../types'
import type { ViewerSnapshot } from '../../../shared/3d/ThreeViewer'
import type { ManualBendSegment } from '../../../shared/3d/bend-annotations/types'
import { validUnfoldFiniteWidthDiagnostics } from './vendorUnfoldFiniteWidth'
import { validUnfoldLocalStripDiagnostics } from './vendorUnfoldLocalStrip'

// Linear/angular allowances retain the frozen V3 producer's authority. Other
// caps bound the viewer boundary; none are manufacturing or verified-unit proof.
const GEOM_TOL = 1e-5
const ANGULAR_TOL_RADIANS = 1e-5
const MAX_COORDINATE = 1e6
const MIN_SEGMENT_LENGTH = 1e-9
const MAX_COORDINATES = 1_800_000
const MAX_INDICES = 600_000
const MAX_TOTAL_COORDINATES = 2 * MAX_COORDINATES
const MAX_TOTAL_INDICES = 2 * MAX_INDICES
// One fixed shared work cap for both Double and actual WebGL Float32 charts.
const MAX_MARKER_TRIANGLE_TESTS = 2 * 32 * (MAX_INDICES / 3)
const ENVELOPE_KEYS = ['status', 'reason', 'model', 'units', 'units_verified', 'manufacturing_ready',
  'source_sha256', 'vertices', 'triangles', 'bend_lines', 'flange_surfaces', 'diagnostics']
const APPROVAL_KEYS = new Set(['units_verified', 'manufacturing_ready', 'release_allowed',
  'cut_release_allowed', 'physical_calibration_verified'])
const LIMITATIONS = [
  'VISUAL_ONLY_SELECTED_CAD_SKIN', 'NOT_NEUTRAL_AXIS_OR_K_FACTOR_BLANK',
  'UNITS_ASSUMED_NOT_VERIFIED', 'PLANAR_FLANGES_AND_PARTIAL_CYLINDERS_ONLY',
  'TREE_TOPOLOGY_ONLY', 'MAXIMUM_BEND_SWEEP_175_DEGREES', 'SINGLE_VALID_SOLID_ONLY',
  'BOUNDED_ANALYTICAL_AND_RATIONAL_SPLINE_NATIVE_TRIMS',
  'NONPLANAR_BOUNDARY_WALLS_REQUIRE_EXACT_AFFINE_EXTRUSION_PROOF',
  'CIRCULAR_RAIL_SET_FALLBACK_REQUIRES_COMPLETE_CYLINDRICAL_HOLE_WALL',
  'COPLANAR_SKIN_FACE_SPLITS_UNSUPPORTED',
]
type RecordValue = Record<string, unknown>
type TriangleXY = { xy: number[]; sign: number; minX: number; maxX: number; minY: number; maxY: number }
type Mesh = { vertices: number[]; triangles: number[] }

function record(value: unknown): value is RecordValue {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}
function coordinate(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && Math.abs(value) <= MAX_COORDINATE
}
function point(value: unknown, flat = false): value is number[] {
  if (!Array.isArray(value) || value.length !== 3) return false
  for (let index = 0; index < 3; index++) if (!coordinate(value[index])) return false
  return !flat || Math.abs(value[2]) <= GEOM_TOL
}
function featureId(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0 && value.length <= 100 && value.trim().length > 0
}
function hash(value: unknown): value is string {
  return typeof value === 'string' && /^[0-9a-f]{64}$/.test(value)
}
function cross(ax: number, ay: number, bx: number, by: number): number { return ax * by - ay * bx }
function segmentLength(start: number[], end: number[]): number {
  return Math.hypot(start[0] - end[0], start[1] - end[1], start[2] - end[2])
}
function safeReceipt(value: unknown): boolean {
  const seen = new WeakSet<object>()
  let remaining = 5_000_000
  const visit = (item: unknown, depth: number): boolean => {
    if (--remaining < 0 || depth > 16) return false
    if (item === null || typeof item === 'boolean') return true
    if (typeof item === 'number') return Number.isFinite(item)
    if (typeof item === 'string') return item.length <= 4096
    if (typeof item !== 'object' || seen.has(item)) return false
    seen.add(item)
    if (Array.isArray(item)) {
      for (let index = 0; index < item.length; index++) if (!visit(item[index], depth + 1)) return false
      return true
    }
    const keys = Object.keys(item)
    if (keys.length > 4096) return false
    for (const key of keys) {
      const child = (item as RecordValue)[key]
      if (APPROVAL_KEYS.has(key) && child !== false) return false
      if (!visit(child, depth + 1)) return false
    }
    return true
  }
  return visit(value, 0)
}
function validMesh(vertices: unknown, triangles: unknown): Mesh | null {
  if (!Array.isArray(vertices) || vertices.length < 9 || vertices.length > MAX_COORDINATES || vertices.length % 3 !== 0
      || !Array.isArray(triangles) || triangles.length < 3 || triangles.length > MAX_INDICES || triangles.length % 3 !== 0) return null
  // Indexed loops deliberately reject sparse arrays; Array.every skips holes.
  for (let index = 0; index < vertices.length; index++) {
    if (!coordinate(vertices[index]) || (index % 3 === 2 && Math.abs(vertices[index]) > GEOM_TOL)) return null
  }
  for (let index = 0; index < triangles.length; index++) {
    if (!Number.isInteger(triangles[index]) || triangles[index] < 0 || triangles[index] >= vertices.length / 3) return null
  }
  const floatVertices = new Float32Array(vertices)
  for (let index = 0; index < triangles.length; index += 3) {
    const a = triangles[index] * 3, b = triangles[index + 1] * 3, c = triangles[index + 2] * 3
    for (const values of [vertices, floatVertices]) {
      const area2 = cross(values[b] - values[a], values[b + 1] - values[a + 1], values[c] - values[a], values[c + 1] - values[a + 1])
      if (!Number.isFinite(area2) || Math.abs(area2) <= 1e-12) return null
    }
  }
  return { vertices, triangles }
}
function faceIndices(value: unknown, count: number, nativeCount: number): Set<number> | null {
  if (!Array.isArray(value) || value.length !== count) return null
  const indices = new Set<number>()
  for (const index of value) {
    if (!Number.isInteger(index) || index < 0 || index >= nativeCount || indices.has(index)) return null
    indices.add(index)
  }
  return indices
}
// V5 composes its separately reviewed finite-strip certificates with native
// chart lineage, exact overlays and coherent winding on the original payload.
function validV5Preview(preview: RecordValue, candidates: unknown): boolean {
  const d = preview.diagnostics
  if (!record(d) || d.mechanism_version !== 5 || d.transport_version !== 4
      || d.cache_profile !== 'V5_CONTROLLED_TRANSPORT_4'
      || !hash(d.v5_engine_source_sha256) || d.face_index_space !== 'STITCHED_NATIVE_CHART'
      || d.exported_triangle_winding !== 'POSITIVE_Z' || d.orientation_changes_vertex_geometry !== false
      || !validUnfoldLocalStripDiagnostics(d, candidates, preview.bend_lines)) return false
  const topo = d.chart_topology
  if (!record(topo) || topo.model !== 'NATIVE_EDGE_CONNECTED_COPLANAR_FLANGE_SUPERNODES'
      || topo.source_vertices_preserved !== true || topo.source_triangles_preserved !== true || topo.geometry_fitted !== false
      || !Number.isSafeInteger(d.original_native_face_count) || Number(d.original_native_face_count) < Number(d.native_face_count)
      || Number(d.original_native_face_count) > 1024 || topo.native_face_count !== d.original_native_face_count
      || topo.chart_face_count !== d.native_face_count || !Array.isArray(topo.charts)
      || topo.charts.length !== d.native_face_count || !Array.isArray(d.selected_source_face_indices)
      || !Array.isArray(d.selected_cylinder_face_indices) || !Array.isArray(d.developed_native_face_slices)
      || !Array.isArray(preview.vertices) || !Array.isArray(preview.triangles) || !Array.isArray(preview.flange_surfaces)
      || !Array.isArray(d.finite_width_proofs)) return false
  const selected = new Set(d.selected_source_face_indices), cylinders = new Set(d.selected_cylinder_face_indices)
  const sources = new Set<number>(), nativeSelected: number[] = [], expected: RecordValue[] = []
  const spans = new Map<number, { v: number; t: number; vc: number; tc: number }>()
  let v = 0, t = 0
  // Validate all native-to-chart ownership, including unselected faces.
  for (let i = 0; i < topo.charts.length; i++) {
    const chart = topo.charts[i]
    if (!record(chart) || chart.chart_face_index !== i || !['plane', 'cylinder', 'extruded_boundary'].includes(String(chart.kind))
        || !Array.isArray(chart.source_face_indices) || !Array.isArray(chart.members)
        || chart.source_face_indices.length < 1 || chart.source_face_indices.length !== chart.members.length
        || (chart.kind !== 'plane' && chart.members.length !== 1)) return false
    let vc = 0, tc = 0, last = -1
    for (let j = 0; j < chart.members.length; j++) {
      const member = chart.members[j], id = chart.source_face_indices[j]
      if (!Number.isSafeInteger(id) || id <= last || id < 0 || id >= Number(d.original_native_face_count) || sources.has(id)
          || !record(member) || Object.keys(member).length !== 5 || member.source_face_index !== id
          || member.vertex_start !== vc || member.triangle_start !== tc || !Number.isSafeInteger(member.vertex_count)
          || !Number.isSafeInteger(member.triangle_count) || Number(member.vertex_count) < 3 || Number(member.triangle_count) < 1) return false
      sources.add(id); last = id
      vc += Number(member.vertex_count); tc += Number(member.triangle_count)
    }
    if (selected.has(i) && (cylinders.has(i) ? chart.kind !== 'cylinder' : chart.kind !== 'plane')) return false
  }
  if (sources.size !== d.original_native_face_count) return false
  for (const i of d.selected_source_face_indices) {
    const chart = topo.charts[i] as RecordValue
    const members = chart.members as RecordValue[]
    const vc = members.reduce((n, m) => n + Number(m.vertex_count), 0), tc = members.reduce((n, m) => n + Number(m.triangle_count), 0)
    spans.set(i, { v, t, vc, tc })
    for (const m of members) {
      nativeSelected.push(Number(m.source_face_index))
      expected.push({ source_face_index: m.source_face_index, chart_face_index: i,
        body_vertex_start: v + Number(m.vertex_start), body_vertex_count: m.vertex_count,
        body_triangle_start: t + Number(m.triangle_start), body_triangle_count: m.triangle_count })
    }
    v += vc; t += tc
  }
  if (JSON.stringify(d.selected_chart_face_indices) !== JSON.stringify(d.selected_source_face_indices)
      || JSON.stringify(d.selected_native_face_indices) !== JSON.stringify(nativeSelected.sort((a, b) => a - b))
      || d.developed_native_face_slices.length !== expected.length || v * 3 !== preview.vertices.length || t * 3 !== preview.triangles.length
      || !Number.isSafeInteger(d.reoriented_triangle_count) || Number(d.reoriented_triangle_count) < 0 || Number(d.reoriented_triangle_count) > t) return false
  for (let i = 0; i < expected.length; i++) {
    const row = d.developed_native_face_slices[i], want = expected[i]
    if (!record(row) || Object.keys(row).length !== 6 || !Object.keys(want).every(key => row[key] === want[key])) return false
    const low = Number(row.body_vertex_start), high = low + Number(row.body_vertex_count)
    for (let k = Number(row.body_triangle_start) * 3; k < (Number(row.body_triangle_start) + Number(row.body_triangle_count)) * 3; k++) {
      if (preview.triangles[k] < low || preview.triangles[k] >= high) return false
    }
  }
  for (let k = 0; k < preview.triangles.length; k += 3) {
    const a = preview.triangles[k] * 3, b = preview.triangles[k + 1] * 3, c = preview.triangles[k + 2] * 3
    if (cross(preview.vertices[b] - preview.vertices[a], preview.vertices[b + 1] - preview.vertices[a + 1],
      preview.vertices[c] - preview.vertices[a], preview.vertices[c + 1] - preview.vertices[a + 1]) <= 1e-12) return false
  }
  for (const overlay of preview.flange_surfaces) {
    if (!record(overlay)) return false
    const proof = d.finite_width_proofs.find(row => record(row) && row.feature_id === overlay.feature_id)
    if (!record(proof)) return false
    const span = spans.get(Number(proof.child_source_face_index))
    if (!span || JSON.stringify(overlay.vertices) !== JSON.stringify(preview.vertices.slice(span.v * 3, (span.v + span.vc) * 3))
        || JSON.stringify(overlay.triangles) !== JSON.stringify(preview.triangles.slice(span.t * 3, (span.t + span.tc) * 3).map(n => n - span.v))) return false
  }
  // Receipt-only gate: browser does not independently read native CAD.
  const oracle = d.geometry_oracle
  return record(oracle) && oracle.status === 'VERIFIED_UNDER_DECLARED_GEOMETRY_MODEL'
    && oracle.verification_model === 'DISCRETE_NATIVE_METRIC_CHART' && oracle.continuous_area_check === 'NOT_ESTABLISHED'
}
function xyKey(x: number, y: number): string { return `${x},${y}` }
function triangleKey(ids: number[]): string { return ids.sort((a, b) => a - b).join(',') }
function triangleCoordinateKey(xy: number[]): string {
  const points = [[xy[0], xy[1]], [xy[2], xy[3]], [xy[4], xy[5]]]
  // Sort whole numeric XY pairs AFTER conversion. Double ordering can change
  // when distinct coordinates round together; flattened/string sorting is not
  // an unordered coordinate-pair identity.
  points.sort((a, b) => a[0] - b[0] || a[1] - b[1])
  return JSON.stringify(points)
}
function triangleXY(xy: number[]): TriangleXY {
  return { xy, sign: Math.sign(cross(xy[2] - xy[0], xy[3] - xy[1], xy[4] - xy[0], xy[5] - xy[1])),
    minX: Math.min(xy[0], xy[2], xy[4]), maxX: Math.max(xy[0], xy[2], xy[4]),
    minY: Math.min(xy[1], xy[3], xy[5]), maxY: Math.max(xy[1], xy[3], xy[5]) }
}

/** Clip the complete marker against the union of body triangles, not samples. */
function markerSupported(start: number[], end: number[], triangles: TriangleXY[], budget: { remaining: number }): boolean {
  const dx = end[0] - start[0], dy = end[1] - start[1], length = Math.hypot(dx, dy)
  const minX = Math.min(start[0], end[0]), maxX = Math.max(start[0], end[0])
  const minY = Math.min(start[1], end[1]), maxY = Math.max(start[1], end[1])
  const intervals: [number, number][] = []
  for (const triangle of triangles) {
    if (--budget.remaining < 0) return false
    if (maxX < triangle.minX || minX > triangle.maxX || maxY < triangle.minY || minY > triangle.maxY) continue
    let low = 0, high = 1
    for (let side = 0; side < 3 && low <= high; side++) {
      const next = (side + 1) % 3, x = triangle.xy[side * 2], y = triangle.xy[side * 2 + 1]
      const ex = triangle.xy[next * 2] - x, ey = triangle.xy[next * 2 + 1] - y
      const atStart = triangle.sign * cross(ex, ey, start[0] - x, start[1] - y)
      const slope = triangle.sign * cross(ex, ey, dx, dy)
      if (!Number.isFinite(atStart) || !Number.isFinite(slope)) return false
      if (slope === 0) { if (atStart < 0) { low = 1; high = 0 } }
      else if (slope > 0) low = Math.max(low, -atStart / slope)
      else high = Math.min(high, -atStart / slope)
    }
    if (low <= high) intervals.push([low, high])
  }
  intervals.sort((a, b) => a[0] - b[0] || a[1] - b[1])
  if (!intervals.some(([low, high]) => high > low)) return false
  let coveredEnd = 0, uncovered = 0
  for (const [low, high] of intervals) {
    uncovered += Math.max(0, low - coveredEnd)
    coveredEnd = Math.max(coveredEnd, high)
  }
  uncovered += Math.max(0, 1 - coveredEnd)
  // One physical error budget for ALL gaps. Triangles are never inflated.
  return Number.isFinite(uncovered) && uncovered * length <= GEOM_TOL
}

/** Validate V3/V4 declarations and mesh relationships, not a cryptographic
 * candidate binding: released receipts contain no candidate-content digest. */
export function validatedUnfoldPreview(data: unknown): VendorUnfoldPreview | null {
  try {
    if (!record(data) || !record(data.part) || !hash(data.part.cad_file_hash) || !record(data.unfold_preview)) return null
    const preview = data.unfold_preview, sourceHash = data.part.cad_file_hash
    const envelopeKeys = Object.keys(preview)
    if (envelopeKeys.length !== ENVELOPE_KEYS.length || !ENVELOPE_KEYS.every(key => envelopeKeys.includes(key)) || !safeReceipt(preview)) return null
    if (preview.status !== 'AVAILABLE' || preview.reason !== null || preview.model !== 'SELECTED_CAD_SKIN_DEVELOPMENT'
        || preview.units !== 'assumed_mm' || preview.units_verified !== false || preview.manufacturing_ready !== false
        || preview.source_sha256 !== sourceHash || ('release_allowed' in preview && preview.release_allowed !== false)
        || ('cut_release_allowed' in preview && preview.cut_release_allowed !== false)
        || !Array.isArray(data.candidates) || data.candidates.length < 1 || data.candidates.length > 32) return null
    const ids = new Set<string>(), candidatePoints = new Map<string, { start: number[]; end: number[] }>()
    for (const candidate of data.candidates) {
      if (!record(candidate) || candidate.source !== 'cad_candidate' || !featureId(candidate.feature_id) || ids.has(candidate.feature_id)
          || !point(candidate.line_start) || !point(candidate.line_end)
          || segmentLength(candidate.line_start, candidate.line_end) <= MIN_SEGMENT_LENGTH
          || ('source_sha256' in candidate && candidate.source_sha256 !== sourceHash)) return null
      ids.add(candidate.feature_id)
      candidatePoints.set(candidate.feature_id, { start: candidate.line_start, end: candidate.line_end })
    }
    const diagnostics = preview.diagnostics
    const limitations = record(diagnostics) ? diagnostics.limitations : null
    if (!record(diagnostics) || (diagnostics.mechanism_version !== 3 && diagnostics.mechanism_version !== 4 && diagnostics.mechanism_version !== 5) || diagnostics.candidate_count !== ids.size
        || diagnostics.selected_skin_rule !== 'LARGEST_PLANAR_ROOT_OF_COMPLETE_TREE_SKIN'
        || diagnostics.direction_status !== 'UNKNOWN_NATIVE_NORMAL_SIGN_NOT_AUTHORITY'
        || !['seam_closure_checked', 'native_surface_ownership_checked', 'cross_face_overlap_checked',
          'native_surface_coverage_checked', 'excluded_caps_and_hole_walls_checked'].every(key => diagnostics[key] === true)
        || !Array.isArray(limitations) || limitations.length > 64
        || !limitations.every(value => typeof value === 'string')
        || !LIMITATIONS.filter(value => diagnostics.mechanism_version !== 5 || value !== 'COPLANAR_SKIN_FACE_SPLITS_UNSUPPORTED')
          .every(value => limitations.includes(value))) return null
    // No version relabelling. V3 retains its existing path; V4 additionally
    // requires reviewed finite-width proofs and transport/runtime declarations.
    if (diagnostics.mechanism_version === 4 && !validUnfoldFiniteWidthDiagnostics(diagnostics, data.candidates)) return null
    if (diagnostics.mechanism_version === 5 && !validV5Preview(preview, data.candidates)) return null
    if ('candidate_binding' in diagnostics) {
      if (!Array.isArray(diagnostics.candidate_binding) || diagnostics.candidate_binding.length !== ids.size) return null
      const bindingIds = new Set<string>()
      for (const row of diagnostics.candidate_binding) {
        if (!record(row) || !featureId(row.feature_id) || bindingIds.has(row.feature_id) || !point(row.line_start) || !point(row.line_end)) return null
        const candidate = candidatePoints.get(row.feature_id)
        if (!candidate || !row.line_start.every((value, index) => value === candidate.start[index])
            || !row.line_end.every((value, index) => value === candidate.end[index])) return null
        bindingIds.add(row.feature_id)
      }
    }
    // Python's canonical JSON number representation need not match JavaScript's.
    // Check a supplied digest's syntax, but bind via exact component equality.
    if ('candidate_input_sha256' in diagnostics && (!hash(diagnostics.candidate_input_sha256) || !('candidate_binding' in diagnostics))) return null
    const nativeCount = diagnostics.native_face_count
    if (typeof nativeCount !== 'number' || !Number.isInteger(nativeCount) || nativeCount < 2 * (2 * ids.size + 1) || nativeCount > 1024) return null
    const selected = faceIndices(diagnostics.selected_source_face_indices, 2 * ids.size + 1, nativeCount)
    const cylinders = faceIndices(diagnostics.selected_cylinder_face_indices, ids.size, nativeCount)
    if (!selected || !cylinders || ![...cylinders].every(index => selected.has(index))
        || typeof diagnostics.root_source_face_index !== 'number' || !selected.has(diagnostics.root_source_face_index)
        || cylinders.has(diagnostics.root_source_face_index)) return null
    const body = validMesh(preview.vertices, preview.triangles)
    if (!body || body.triangles.length / 3 < selected.size || !Array.isArray(preview.bend_lines) || preview.bend_lines.length !== ids.size
        || !Array.isArray(preview.flange_surfaces) || preview.flange_surfaces.length !== ids.size) return null
    const pointIds = new Map<string, number>(), bodyTriangleKeys = new Set<string>(), floatBodyTriangleKeys = new Set<string>()
    const bodyTriangles: TriangleXY[] = [], floatBodyTriangles: TriangleXY[] = []
    for (let index = 0; index < body.vertices.length; index += 3) {
      const key = xyKey(body.vertices[index], body.vertices[index + 1])
      if (!pointIds.has(key)) pointIds.set(key, pointIds.size)
    }
    for (let index = 0; index < body.triangles.length; index += 3) {
      const xy: number[] = [], triangleIds: number[] = []
      for (let side = 0; side < 3; side++) {
        const vertex = body.triangles[index + side] * 3, x = body.vertices[vertex], y = body.vertices[vertex + 1]
        xy.push(x, y); triangleIds.push(pointIds.get(xyKey(x, y))!)
      }
      const key = triangleKey(triangleIds)
      if (bodyTriangleKeys.has(key)) return null
      bodyTriangleKeys.add(key)
      const floatXY = xy.map(value => Math.fround(value)), floatKey = triangleCoordinateKey(floatXY)
      if (floatBodyTriangleKeys.has(floatKey)) return null
      floatBodyTriangleKeys.add(floatKey)
      bodyTriangles.push(triangleXY(xy))
      floatBodyTriangles.push(triangleXY(floatXY))
    }
    const lineIds = new Set<string>(), markerBudget = { remaining: MAX_MARKER_TRIANGLE_TESTS }
    for (const line of preview.bend_lines) {
      if (!record(line) || !featureId(line.feature_id) || !ids.has(line.feature_id) || lineIds.has(line.feature_id)
          || !point(line.line_start, true) || !point(line.line_end, true) || line.bend_direction !== 'unknown'
          || typeof line.sweep_deg !== 'number' || !Number.isFinite(line.sweep_deg) || line.sweep_deg <= 0
          || line.sweep_deg > 175 + ANGULAR_TOL_RADIANS * 180 / Math.PI
          || Math.hypot(line.line_start[0] - line.line_end[0], line.line_start[1] - line.line_end[1]) <= MIN_SEGMENT_LENGTH
          || Math.hypot(Math.fround(line.line_start[0]) - Math.fround(line.line_end[0]), Math.fround(line.line_start[1]) - Math.fround(line.line_end[1])) <= MIN_SEGMENT_LENGTH
          || !markerSupported(line.line_start, line.line_end, bodyTriangles, markerBudget)
          // Noncollapsed Float32 triangles alone do not guarantee a supported
          // marker: independent rounding can move it outside a slanted edge.
          // Both charts must pass separately with the SAME 1e-5 allowance.
          || !markerSupported(line.line_start.map(value => Math.fround(value)), line.line_end.map(value => Math.fround(value)), floatBodyTriangles, markerBudget)) return null
      lineIds.add(line.feature_id)
    }
    const overlayIds = new Set<string>(), overlayTriangleKeys = new Set<string>()
    let coordinateCount = body.vertices.length, indexCount = body.triangles.length
    for (const surface of preview.flange_surfaces) {
      if (!record(surface) || !featureId(surface.feature_id) || !ids.has(surface.feature_id) || overlayIds.has(surface.feature_id)) return null
      const mesh = validMesh(surface.vertices, surface.triangles)
      if (!mesh || mesh.triangles.length >= body.triangles.length) return null
      coordinateCount += mesh.vertices.length; indexCount += mesh.triangles.length
      if (coordinateCount > MAX_TOTAL_COORDINATES || indexCount > MAX_TOTAL_INDICES) return null
      for (let index = 0; index < mesh.triangles.length; index += 3) {
        const triangleIds: number[] = []
        for (let side = 0; side < 3; side++) {
          const vertex = mesh.triangles[index + side] * 3
          // XY is exact; V3 overlays can retain <= GEOM_TOL residual Z before
          // the main body is projected onto Z=0. No coordinate snapping.
          const id = pointIds.get(xyKey(mesh.vertices[vertex], mesh.vertices[vertex + 1]))
          if (id === undefined) return null
          triangleIds.push(id)
        }
        const key = triangleKey(triangleIds)
        if (!bodyTriangleKeys.has(key) || overlayTriangleKeys.has(key)) return null
        overlayTriangleKeys.add(key)
      }
      overlayIds.add(surface.feature_id)
    }
    // A child flange cannot consume the root or bend strips. This necessary
    // count check is not a substitute for the producer's native face ledger.
    if (bodyTriangleKeys.size - overlayTriangleKeys.size < cylinders.size + 1) return null
    return preview as unknown as VendorUnfoldPreview
  } catch {
    // Even hostile/non-JSON object getters must not crash the operator screen.
    return null
  }
}

/** Deterministic top-down technical view of the supplied flat CAD-skin mesh. */
export function captureUnfoldPreview(preview: VendorUnfoldPreview, segments: ManualBendSegment[], colors: Map<string, string>): ViewerSnapshot {
  const canvas = document.createElement('canvas')
  canvas.width = 2200; canvas.height = 1500
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Flat view capture unavailable')
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (let index = 0; index < preview.vertices.length; index += 3) {
    minX = Math.min(minX, preview.vertices[index]); maxX = Math.max(maxX, preview.vertices[index])
    minY = Math.min(minY, preview.vertices[index + 1]); maxY = Math.max(maxY, preview.vertices[index + 1])
  }
  const scale = Math.min((canvas.width - 180) / (maxX - minX), (canvas.height - 180) / (maxY - minY))
  if (!Number.isFinite(scale) || scale <= 0) throw new Error('Invalid flat view bounds')
  const offsetX = (canvas.width - (maxX - minX) * scale) / 2, offsetY = (canvas.height - (maxY - minY) * scale) / 2
  const project = (point: number[]) => [offsetX + (point[0] - minX) * scale, canvas.height - offsetY - (point[1] - minY) * scale]
  const drawTriangles = (vertices: number[], triangles: number[], color: string, alpha = 1) => {
    ctx.fillStyle = color; ctx.globalAlpha = alpha; ctx.beginPath()
    for (let index = 0; index < triangles.length; index += 3) {
      const points = triangles.slice(index, index + 3).map(id => project(vertices.slice(id * 3, id * 3 + 3)))
      ctx.moveTo(points[0][0], points[0][1]); ctx.lineTo(points[1][0], points[1][1]); ctx.lineTo(points[2][0], points[2][1]); ctx.closePath()
    }
    ctx.fill(); ctx.globalAlpha = 1
  }
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, canvas.width, canvas.height)
  drawTriangles(preview.vertices, preview.triangles, '#dde2e7')
  for (const surface of preview.flange_surfaces) {
    const color = colors.get(surface.feature_id)
    if (color) drawTriangles(surface.vertices, surface.triangles, color, 0.24)
  }
  const edges = new Map<string, { points: number[][]; count: number }>()
  const pointKey = (point: number[]) => `${Math.round(point[0] * 1e5)},${Math.round(point[1] * 1e5)}`
  for (let index = 0; index < preview.triangles.length; index += 3) {
    const points = preview.triangles.slice(index, index + 3).map(id => preview.vertices.slice(id * 3, id * 3 + 3))
    for (let side = 0; side < 3; side++) {
      const a = points[side], b = points[(side + 1) % 3], key = [pointKey(a), pointKey(b)].sort().join('|')
      const edge = edges.get(key)
      if (edge) edge.count++
      else edges.set(key, { points: [a, b], count: 1 })
    }
  }
  ctx.strokeStyle = '#334155'; ctx.lineWidth = 3; ctx.beginPath()
  for (const edge of edges.values()) if (edge.count === 1) {
    const a = project(edge.points[0]), b = project(edge.points[1])
    ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1])
  }
  ctx.stroke()
  const labels = segments.map(segment => {
    const a = project(segment.lineStart), b = project(segment.lineEnd)
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1])
    ctx.strokeStyle = '#15212d'; ctx.lineWidth = 11; ctx.stroke()
    ctx.strokeStyle = segment.color || '#334155'; ctx.lineWidth = 7; ctx.stroke()
    return { x: (a[0] + b[0]) / 2, y: (a[1] + b[1]) / 2, featureId: segment.id,
      text: [segment.accessibleLabel, segment.label && `#${segment.label}`, segment.groupLabel].filter(Boolean).join(' · '), color: segment.color || '#334155' }
  })
  return { dataUrl: canvas.toDataURL('image/png'), width: canvas.width, height: canvas.height, labels }
}
