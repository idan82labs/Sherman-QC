/**
 * Fetch-based client for the station screen (`/api/line`, C07).
 *
 * Auth: a Bearer token on every request, never in a URL. The token is the
 * station device token (`localStorage['sherman.station.token.<stationId>']`,
 * preferred on a kiosk) or the signed-in user's JWT (`localStorage['token']`).
 * The only token that ever appears in a URL is the short-lived (300 s),
 * station-bound SSE stream token, because `EventSource` cannot send headers.
 *
 * Artifacts are fetched only from same-origin paths under `/api/line/`
 * (`fetchArtifact` refuses anything else), so a result document can never
 * make the kiosk send its token to another host.
 *
 * Errors carry the HTTP status and the stable `X-Sherman-Auth-Code` header.
 */
import { validateResult } from '../model/stationView'
import { validateOverlay, type ZeroshotOverlay } from '../model/zeroshotOverlay'
import { needsEyeCheck } from '../model/chirality'
import { ARTIFACT_NAMES, type ArtifactName, type ShopfloorInspectionResult, type StationFeed } from '../model/types'
import { parseHeld, type AssignBody, type DismissReason, type HeldResponse, type ReleasedPart } from '../model/partPicker'

export const LINE_API = '/api/line'
const USER_TOKEN_KEY = 'token'
const DEVICE_TOKEN_PREFIX = 'sherman.station.token.'
export const DEVICE_TOKEN_RE = /^sdt_[A-Za-z0-9_-]{32,128}$/
const JOB_ID_RE = /^line_[0-9a-f]{32}$/

export class StationApiError extends Error {
  readonly status: number
  readonly code: string | null
  constructor(message: string, status: number, code: string | null = null) {
    super(message)
    this.name = 'StationApiError'
    this.status = status
    this.code = code
  }
  get isAuth(): boolean {
    return this.status === 401 || this.status === 403
  }
}

export type TokenSource = 'device' | 'user'
export interface StationToken {
  token: string
  source: TokenSource
}

function storageGet(key: string): string | null {
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

function storageSet(key: string, value: string | null): void {
  try {
    if (value === null) window.localStorage.removeItem(key)
    else window.localStorage.setItem(key, value)
  } catch {
    // Storage blocked (private mode / kiosk policy): the token lives for this page only.
  }
}

const sessionTokens = new Map<string, string>()

export function deviceTokenKey(stationId: string): string {
  return `${DEVICE_TOKEN_PREFIX}${stationId}`
}

/** The token to use for a station: device token first, then the user's JWT. */
export function readToken(stationId: string): StationToken | null {
  const device = storageGet(deviceTokenKey(stationId)) ?? sessionTokens.get(stationId) ?? null
  if (device) return { token: device, source: 'device' }
  const user = storageGet(USER_TOKEN_KEY)
  if (user) return { token: user, source: 'user' }
  return null
}

export function saveDeviceToken(stationId: string, token: string): void {
  if (!DEVICE_TOKEN_RE.test(token)) throw new Error('not a station device token')
  sessionTokens.set(stationId, token)
  storageSet(deviceTokenKey(stationId), token)
}

export function clearDeviceToken(stationId: string): void {
  sessionTokens.delete(stationId)
  storageSet(deviceTokenKey(stationId), null)
}

function authHeaders(stationId: string, extra: Record<string, string> = {}): Record<string, string> {
  const tok = readToken(stationId)
  return tok ? { ...extra, Authorization: `Bearer ${tok.token}` } : { ...extra }
}

async function failure(res: Response, what: string): Promise<StationApiError> {
  let code = res.headers.get('X-Sherman-Auth-Code')
  let detail = ''
  try {
    const body = await res.json()
    if (body && typeof body.detail === 'string') detail = body.detail
    // Line API refusals: {"detail": {"code", "message"}} (+ X-Sherman-Error-Code).
    else if (body && body.detail && typeof body.detail === 'object') {
      if (typeof body.detail.message === 'string') detail = body.detail.message
      if (!code && typeof body.detail.code === 'string') code = body.detail.code
    }
  } catch {
    // non-JSON error body
  }
  if (!code) code = res.headers.get('X-Sherman-Error-Code')
  return new StationApiError(`${what} failed (${res.status})${detail ? `: ${detail}` : ''}`, res.status, code)
}

export type FeedResponse =
  | { status: 'ok'; feed: StationFeed; etag: string | null }
  | { status: 'not_modified'; etag: string | null }

/** GET /stations/{id}/latest with If-None-Match; a 304 keeps the previous feed. */
export async function getFeed(stationId: string, etag: string | null, signal?: AbortSignal): Promise<FeedResponse> {
  const headers = authHeaders(stationId, { Accept: 'application/json' })
  if (etag) headers['If-None-Match'] = etag
  const res = await fetch(`${LINE_API}/stations/${encodeURIComponent(stationId)}/latest`, {
    headers,
    signal,
    cache: 'no-store',
    credentials: 'same-origin',
  })
  if (res.status === 304) return { status: 'not_modified', etag }
  if (!res.ok) throw await failure(res, 'station feed')
  const feed = (await res.json()) as StationFeed
  if (!feed || feed.schema !== 'sherman_station_feed_v1') throw new StationApiError('station feed has the wrong schema', 502)
  return { status: 'ok', feed, etag: res.headers.get('ETag') ?? feed.etag ?? null }
}

/** GET /jobs/{id}/result, validated with the contract rules. */
export async function getJobResult(stationId: string, jobId: string, signal?: AbortSignal): Promise<ShopfloorInspectionResult> {
  if (!JOB_ID_RE.test(jobId)) throw new StationApiError('invalid job id', 400)
  const res = await fetch(`${LINE_API}/jobs/${jobId}/result`, {
    headers: authHeaders(stationId, { Accept: 'application/json' }),
    signal,
    credentials: 'same-origin',
  })
  if (!res.ok) throw await failure(res, 'job result')
  const v = validateResult(await res.json())
  if (!v.ok) throw new StationApiError(`job result rejected: ${v.errors[0]}`, 422, 'RESULT_INVALID')
  return v.result
}

/**
 * GET /jobs/{id}/zeroshot-overlay: the job's zero-shot overlay (display / REVIEW only), validated against
 * the result on screen. 404 (no overlay for this job) and any refused document give null: the screen then
 * shows E3's result alone.
 */
export async function getZeroshotOverlay(
  stationId: string,
  jobId: string,
  result: ShopfloorInspectionResult,
  signal?: AbortSignal,
): Promise<ZeroshotOverlay | null> {
  if (!JOB_ID_RE.test(jobId)) throw new StationApiError('invalid job id', 400)
  const res = await fetch(`${LINE_API}/jobs/${jobId}/zeroshot-overlay`, {
    headers: authHeaders(stationId, { Accept: 'application/json' }),
    signal,
    credentials: 'same-origin',
  })
  if (res.status === 404) return null
  if (!res.ok) throw await failure(res, 'zero-shot overlay')
  return validateOverlay(await res.json(), result)
}

/**
 * GET /api/production/recipes/{recipe_id}/chirality?station_id=: does the recipe on screen need the
 * left/right eye check (display only)? Read-only on the server (never starts a computation). Any
 * refusal or failure is false: the badge is a reminder, never a reason to hold the screen.
 */
export async function getChiralityEyeCheck(stationId: string, recipeId: string, signal?: AbortSignal): Promise<boolean> {
  const q = new URLSearchParams({ station_id: stationId })
  try {
    const res = await fetch(`/api/production/recipes/${encodeURIComponent(recipeId)}/chirality?${q.toString()}`, {
      headers: authHeaders(stationId, { Accept: 'application/json' }),
      signal,
      credentials: 'same-origin',
    })
    if (!res.ok) return false
    return needsEyeCheck(await res.json(), recipeId)
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err
    return false
  }
}

/** Artifact URL for a job (allowlisted names only). */
export function artifactUrl(jobId: string, name: ArtifactName): string {
  if (!JOB_ID_RE.test(jobId)) throw new StationApiError('invalid job id', 400)
  if (!(ARTIFACT_NAMES as readonly string[]).includes(name)) throw new StationApiError('artifact not allowlisted', 400)
  return `${LINE_API}/jobs/${jobId}/artifacts/${name}`
}

/**
 * Resolve an artifact URL and refuse anything that is not same-origin under
 * `/api/line/`. Returns the absolute URL string.
 */
export function checkArtifactUrl(url: string): string {
  let parsed: URL
  try {
    parsed = new URL(url, window.location.origin)
  } catch {
    throw new StationApiError('artifact URL is malformed', 400, 'ARTIFACT_URL_REFUSED')
  }
  if (parsed.origin !== window.location.origin) throw new StationApiError('artifact URL is not same-origin', 400, 'ARTIFACT_URL_REFUSED')
  if (!parsed.pathname.startsWith(`${LINE_API}/`) || parsed.pathname.includes('/../') || parsed.username || parsed.password) {
    throw new StationApiError('artifact URL is outside /api/line/', 400, 'ARTIFACT_URL_REFUSED')
  }
  return parsed.toString()
}

export interface ArtifactBytes {
  data: ArrayBuffer
  contentType: string | null
  fileName: string | null
}

/** Fetch an artifact with the Authorization header (same-origin /api/line/ only). */
export async function fetchArtifact(stationId: string, url: string, signal?: AbortSignal): Promise<ArtifactBytes> {
  const safe = checkArtifactUrl(url)
  const res = await fetch(safe, { headers: authHeaders(stationId), signal, credentials: 'same-origin' })
  if (!res.ok) throw await failure(res, 'artifact')
  const disposition = res.headers.get('Content-Disposition') ?? ''
  const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(disposition)
  let fileName: string | null = null
  if (m) {
    try {
      fileName = decodeURIComponent(m[1])
    } catch {
      fileName = m[1]
    }
  }
  return { data: await res.arrayBuffer(), contentType: res.headers.get('Content-Type'), fileName }
}

export interface StreamToken {
  token: string
  expires_at: string | null
}

/** POST /stream-token: a 300 s station-bound token for the SSE stream. */
export async function mintStreamToken(stationId: string, signal?: AbortSignal): Promise<StreamToken> {
  const res = await fetch(`${LINE_API}/stream-token`, {
    method: 'POST',
    headers: authHeaders(stationId, { 'Content-Type': 'application/json', Accept: 'application/json' }),
    body: JSON.stringify({ station_id: stationId }),
    signal,
    credentials: 'same-origin',
  })
  if (!res.ok) throw await failure(res, 'stream token')
  const body = await res.json()
  if (!body || typeof body.token !== 'string') throw new StationApiError('stream token response malformed', 502)
  return { token: body.token, expires_at: typeof body.expires_at === 'string' ? body.expires_at : null }
}

export function streamUrl(stationId: string, token: string): string {
  const q = new URLSearchParams({ station_id: stationId, token })
  return `${LINE_API}/events/stream?${q.toString()}`
}

/** Save bytes as a file through a temporary object URL. */
export function saveBlob(data: ArrayBuffer, fileName: string, contentType = 'application/octet-stream'): void {
  const blob = new Blob([data], { type: contentType })
  const href = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = href
  a.download = fileName
  a.rel = 'noopener'
  document.body.appendChild(a)
  a.click()
  a.remove()
  window.setTimeout(() => URL.revokeObjectURL(href), 10_000)
}

// --------------------------------------------------------------------------- part picker
//
// Held-back scans and the released-part search are read with the station's
// token (a device token may read). Assign / dismiss need `line.operate`: the
// signed-in operator's JWT, or an operator token obtained in the picker
// (`operatorLogin`) and kept in memory for that dialog only.

function withToken(stationId: string, operatorToken: string | null, extra: Record<string, string>): Record<string, string> {
  return operatorToken ? { ...extra, Authorization: `Bearer ${operatorToken}` } : authHeaders(stationId, extra)
}

/** GET /quarantine/held: the station's open held-back scans with suggested parts. */
export async function getHeldScans(stationId: string, signal?: AbortSignal): Promise<HeldResponse> {
  const q = new URLSearchParams({ station_id: stationId })
  const res = await fetch(`${LINE_API}/quarantine/held?${q.toString()}`, {
    headers: authHeaders(stationId, { Accept: 'application/json' }),
    signal,
    cache: 'no-store',
    credentials: 'same-origin',
  })
  if (!res.ok) throw await failure(res, 'held scans')
  return parseHeld(await res.json())
}

/** GET /released-parts: released parts matching `query` (SKU prefix / substring, part name). */
export async function searchReleasedParts(stationId: string, query: string, signal?: AbortSignal): Promise<ReleasedPart[]> {
  const q = new URLSearchParams({ station_id: stationId, limit: '50' })
  if (query.trim()) q.set('q', query.trim().slice(0, 64))
  const res = await fetch(`${LINE_API}/released-parts?${q.toString()}`, {
    headers: authHeaders(stationId, { Accept: 'application/json' }),
    signal,
    cache: 'no-store',
    credentials: 'same-origin',
  })
  if (!res.ok) throw await failure(res, 'released parts')
  const body = await res.json()
  return Array.isArray(body?.items) ? (body.items as ReleasedPart[]) : []
}

export interface AssignResult {
  scan_file_id: string
  job_id: string
  recipe_id: string
  recipe_version: number
  sku: string
  assigned_now: boolean
}

const SCAN_FILE_ID_RE = /^scan_[0-9a-f]{24}$/

/** POST /quarantine/{id}/assign (idempotent on the server: a double click returns the same job). */
export async function assignHeld(stationId: string, scanFileId: string, body: AssignBody, operatorToken: string | null): Promise<AssignResult> {
  if (!SCAN_FILE_ID_RE.test(scanFileId)) throw new StationApiError('invalid scan id', 400)
  const res = await fetch(`${LINE_API}/quarantine/${scanFileId}/assign`, {
    method: 'POST',
    headers: withToken(stationId, operatorToken, { 'Content-Type': 'application/json', Accept: 'application/json' }),
    body: JSON.stringify(body),
    credentials: 'same-origin',
  })
  if (!res.ok) throw await failure(res, 'assign part')
  return (await res.json()) as AssignResult
}

/** POST /quarantine/{id}/dismiss with the operator's reason (idempotent). */
export async function dismissHeld(stationId: string, scanFileId: string, reason: DismissReason, note: string, operatorToken: string | null): Promise<void> {
  if (!SCAN_FILE_ID_RE.test(scanFileId)) throw new StationApiError('invalid scan id', 400)
  const res = await fetch(`${LINE_API}/quarantine/${scanFileId}/dismiss`, {
    method: 'POST',
    headers: withToken(stationId, operatorToken, { 'Content-Type': 'application/json', Accept: 'application/json' }),
    body: JSON.stringify(note.trim() ? { reason, note: note.trim().slice(0, 500) } : { reason }),
    credentials: 'same-origin',
  })
  if (!res.ok) throw await failure(res, 'dismiss scan')
}

/** POST /api/auth/login: an operator's JWT for one picker session (kept in memory, never stored). */
export async function operatorLogin(username: string, password: string): Promise<{ token: string; role: string | null; name: string }> {
  const res = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify({ username, password }),
    credentials: 'same-origin',
  })
  if (!res.ok) throw await failure(res, 'operator sign-in')
  const body = await res.json()
  if (!body || typeof body.access_token !== 'string') throw new StationApiError('sign-in response malformed', 502)
  return { token: body.access_token, role: typeof body.user?.role === 'string' ? body.user.role : null, name: String(body.user?.username ?? username) }
}

export interface JobAssignment {
  by_operator: boolean
  assigned_at: string | null
  from_reason: string | null
}

/** GET /jobs/{id} -> its `assignment` block (null: matched automatically by the hotfolder). */
export async function getJobAssignment(stationId: string, jobId: string, signal?: AbortSignal): Promise<JobAssignment | null> {
  if (!JOB_ID_RE.test(jobId)) throw new StationApiError('invalid job id', 400)
  const res = await fetch(`${LINE_API}/jobs/${jobId}`, {
    headers: authHeaders(stationId, { Accept: 'application/json' }),
    signal,
    credentials: 'same-origin',
  })
  if (!res.ok) throw await failure(res, 'job')
  const body = await res.json()
  const a = body?.assignment
  return a && a.by_operator === true ? { by_operator: true, assigned_at: a.assigned_at ?? null, from_reason: a.from_reason ?? null } : null
}
