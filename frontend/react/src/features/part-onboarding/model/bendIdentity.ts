/**
 * Browser mirror of the recipe bend identity and the C05 flat-binding basis.
 *
 * - `bendUid` is `domains.production.recipe.bend_identity.bend_uid`:
 *   `"bu_" + sha256("<cad_sha256>|<source>|<a>;<b>|<radius>")[:20]`, endpoints
 *   rounded to 1e-3 mm exactly as Python's `round(x, 3)` (round-half-even on the
 *   binary value) and `-0.000` folded to `0.000`, sorted numerically.
 * - `bindingMarks` is `apps.production_line.recipes.flat_intake.binding_marks`:
 *   the active, located marks in bending order that get a uid.
 * - `bindingBasisSha256` is `flat_intake.binding_basis_sha256`, the identity a
 *   confirmed flat binding is checked against at release (FLAT_BINDING_STALE).
 *
 * C05 stores flat bindings by bend_uid only, so the wizard and Vendor Setup need
 * these to map flat lines back to the marks they are editing and to tell a
 * current binding from a stale one. `scripts/check-onboarding-model.mjs` checks
 * parity with the Python functions, rounding ties included. A disagreement can
 * only make a binding look stale (the safe direction), never fresh.
 *
 * Pure and synchronous: `crypto.subtle` is unavailable on plain-http shop-floor
 * hosts, so SHA-256 is implemented here.
 */
import type { Point3 } from './types'

// ---------------------------------------------------------------- SHA-256 (FIPS 180-4)

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
])

/** Lower-case hex SHA-256 of the UTF-8 bytes of `text`. */
export function sha256Hex(text: string): string {
  const data = new TextEncoder().encode(text)
  const bitLength = data.length * 8
  const padded = new Uint8Array(((data.length + 9 + 63) >> 6) << 6)
  padded.set(data)
  padded[data.length] = 0x80
  const view = new DataView(padded.buffer)
  view.setUint32(padded.length - 8, Math.floor(bitLength / 0x100000000))
  view.setUint32(padded.length - 4, bitLength >>> 0)
  const h = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19])
  const w = new Uint32Array(64)
  const rotr = (x: number, n: number) => (x >>> n) | (x << (32 - n))
  for (let offset = 0; offset < padded.length; offset += 64) {
    for (let i = 0; i < 16; i++) w[i] = view.getUint32(offset + i * 4)
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3)
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10)
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0
    }
    let a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7]
    for (let i = 0; i < 64; i++) {
      const t1 = (k + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i]) >>> 0
      const t2 = ((rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) >>> 0
      k = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0
    }
    h[0] = (h[0] + a) >>> 0; h[1] = (h[1] + b) >>> 0; h[2] = (h[2] + c) >>> 0; h[3] = (h[3] + d) >>> 0
    h[4] = (h[4] + e) >>> 0; h[5] = (h[5] + f) >>> 0; h[6] = (h[6] + g) >>> 0; h[7] = (h[7] + k) >>> 0
  }
  return Array.from(h, word => word.toString(16).padStart(8, '0')).join('')
}

// ---------------------------------------------------------------- Python-compatible rounding

/**
 * `f"{round(value, 3):.3f}"` with `-0.000` folded to `0.000` (bend_identity._q).
 *
 * `toFixed` is exact but breaks an exact tie away from zero; Python's `round`
 * breaks it to even. A double is an exact tie at 3 decimals only when its
 * fraction is an odd multiple of 1/16 (0.0625, 0.1875, ...), so those are
 * resolved by hand.
 */
export function pyRound3Text(value: number): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error('bend geometry must be finite numbers')
  let text = value.toFixed(3)
  const sixteenths = Math.abs(value) * 16
  if (Number.isInteger(sixteenths) && sixteenths % 2 === 1 && Number.isSafeInteger(sixteenths * 125)) {
    // |value| * 1000 = sixteenths * 62.5 = n + 0.5 exactly: pick the even neighbour.
    const below = Math.floor(sixteenths * 62.5)
    const n = below % 2 === 0 ? below : below + 1
    const magnitude = `${Math.floor(n / 1000)}.${String(n % 1000).padStart(3, '0')}`
    text = value < 0 ? `-${magnitude}` : magnitude
  }
  return Number(text) === 0 ? '0.000' : text
}

// ---------------------------------------------------------------- bend identity

export const BEND_UID_SOURCES = ['cad_candidate', 'manual'] as const
const HEX64 = /^[0-9a-f]{64}$/

function pointText(point: readonly number[]): [string, string, string] {
  if (!Array.isArray(point) || point.length !== 3) throw new Error('bend line endpoints must be 3D points')
  return [pyRound3Text(point[0]), pyRound3Text(point[1]), pyRound3Text(point[2])]
}

function compareRounded(a: [string, string, string], b: [string, string, string]): number {
  for (let i = 0; i < 3; i++) {
    const d = Number(a[i]) - Number(b[i])
    if (d !== 0) return d < 0 ? -1 : 1
  }
  return 0
}

/** `bend_identity.bend_uid`; throws on the same inputs Python refuses. */
export function bendUid(cadSha256: string, source: string, lineStart: readonly number[], lineEnd: readonly number[], radius: number | null): string {
  if (typeof cadSha256 !== 'string' || !HEX64.test(cadSha256)) throw new Error('cad_sha256 must be 64 lowercase hex characters')
  if (!(BEND_UID_SOURCES as readonly string[]).includes(source)) throw new Error('unknown bend source')
  const [a, b] = [pointText(lineStart), pointText(lineEnd)].sort(compareRounded)
  if (a.join() === b.join()) throw new Error('bend line endpoints coincide at 1e-3 mm')
  const radiusText = radius == null ? 'none' : pyRound3Text(radius)
  return `bu_${sha256Hex(`${cadSha256}|${source}|${a.join(',')};${b.join(',')}|${radiusText}`).slice(0, 20)}`
}

// ---------------------------------------------------------------- binding marks / basis (C05 flat_intake)

export interface BindingMarkInput {
  feature_id: string
  label?: string | null
  source?: string
  status?: string
  line_start?: unknown
  line_end?: unknown
  target_radius?: unknown
}

export interface BindingMark {
  bend_uid: string
  feature_id: string
  label: string
  line_start: Point3
  line_end: Point3
}

const DECIMAL_TEXT = /^\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*$/

/** `flat_intake._parse_point`: a finite 3-vector, from an array or its JSON text. */
function parsePoint(raw: unknown): Point3 | null {
  let value: unknown = raw
  if (raw == null) return null
  if (typeof raw === 'string') {
    try { value = JSON.parse(raw) } catch { return null }
  }
  if (!Array.isArray(value) || value.length !== 3) return null
  const out: number[] = []
  for (const item of value) {
    // Python float(): numbers, booleans and decimal text; never hex or empty text.
    const n = typeof item === 'number' ? item : typeof item === 'boolean' ? Number(item)
      : typeof item === 'string' && DECIMAL_TEXT.test(item) ? Number(item) : Number.NaN
    if (!Number.isFinite(n)) return null
    out.push(n)
  }
  return out as Point3
}

/** Active, located marks in bending order with their recipe bend_uid (flat_intake.binding_marks). */
export function bindingMarks(draft: { marks?: readonly BindingMarkInput[] | null; sequence?: readonly string[] | null } | null | undefined,
  cadSha256: string | null | undefined): BindingMark[] {
  if (!draft || !cadSha256) return []
  const byId = new Map<string, BindingMarkInput>()
  for (const mark of draft.marks ?? []) if (mark && typeof mark === 'object') byId.set(mark.feature_id, mark)
  const out: BindingMark[] = []
  for (const fid of draft.sequence ?? []) {
    const mark = byId.get(fid)
    if (!mark || mark.status !== 'active') continue
    const start = parsePoint(mark.line_start), end = parsePoint(mark.line_end)
    if (!start || !end || start.every((v, i) => v === end[i])) continue
    const r = mark.target_radius
    const radius = typeof r === 'number' && Number.isFinite(r) ? r : null
    let uid: string
    try { uid = bendUid(cadSha256, String(mark.source ?? ''), start, end, radius) } catch { continue }
    out.push({ bend_uid: uid, feature_id: fid, label: String(mark.label || fid), line_start: start, line_end: end })
  }
  return out
}

/** `flat_intake.binding_basis_sha256`: the CAD bytes plus the set of bend identities. */
export function bindingBasisSha256(cadSha256: string, marks: ReadonlyArray<{ bend_uid: string }>): string {
  const uids = [...new Set(marks.map(mark => String(mark.bend_uid)))].sort()
  return sha256Hex(JSON.stringify({ bend_uids: uids, cad_sha256: cadSha256 }))
}
