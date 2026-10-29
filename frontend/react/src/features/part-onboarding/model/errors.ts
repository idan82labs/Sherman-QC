/**
 * Normalise any /api/production error body into `{status, codes, messages}`.
 *
 * C05 errors are `4xx/5xx {detail: {codes: [...], messages: {code: text},
 * details?: {code: [text]}, compile?: <E1 compile report>}}`
 * (`production_recipe_service.RecipeApiError`, `release_service.ReleaseGateError`).
 * Older shapes (`detail` string, pydantic `detail` list, `{code}`) are tolerated.
 * A transport failure gets a code too, so the UI never shows an empty error.
 * `structured` tells a C05 refusal from a bare framework 404/405 (route absent).
 * Pure: takes a plain `{status, data}` pair, not an axios object.
 */
import { parseCompileReport } from './adapters'
import type { ProductionApiError } from './types'

const CODE_RE = /^[A-Z][A-Z0-9_]{2,63}$/

export function normaliseApiError(status: number | null | undefined, data: unknown, fallbackMessage = ''): ProductionApiError {
  const codes: string[] = []
  const messages: Record<string, string> = {}
  const details: Record<string, string[]> = {}
  let compile: ProductionApiError['compile'] = null
  let structured = false
  let message = fallbackMessage
  const addCode = (value: unknown) => { if (typeof value === 'string' && CODE_RE.test(value) && !codes.includes(value)) codes.push(value) }
  const body = (data && typeof data === 'object' ? data : {}) as Record<string, unknown>
  const detail = body.detail
  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    const d = detail as Record<string, unknown>
    if (Array.isArray(d.codes)) d.codes.forEach(addCode)
    addCode(d.code)
    structured = codes.length > 0
    if (d.details && typeof d.details === 'object' && !Array.isArray(d.details)) {
      for (const [key, items] of Object.entries(d.details as Record<string, unknown>)) {
        if (Array.isArray(items)) details[key] = items.filter((item): item is string => typeof item === 'string')
      }
    }
    compile = parseCompileReport(d.compile)
    if (d.messages && typeof d.messages === 'object') {
      for (const [key, text] of Object.entries(d.messages as Record<string, unknown>)) if (typeof text === 'string') messages[key] = text
    }
    if (typeof d.message === 'string') message = d.message
  } else if (typeof detail === 'string') {
    message = detail
    addCode(detail.split(':', 1)[0].trim())
  } else if (Array.isArray(detail)) {
    message = detail.map(item => (item && typeof item === 'object' && typeof (item as { msg?: unknown }).msg === 'string' ? (item as { msg: string }).msg : '')).filter(Boolean).join('; ')
    if (status === 422) addCode('DRAFT_INVALID')
  }
  if (Array.isArray(body.codes)) body.codes.forEach(addCode)
  addCode(body.code)
  if (typeof body.error === 'string' && !message) message = body.error
  const s = typeof status === 'number' ? status : 0
  if (!codes.length) {
    if (!s) codes.push('NETWORK_ERROR')
    else if (s === 503) codes.push('LINE_NOT_CONFIGURED')
    else if (s === 401 || s === 403) codes.push('FORBIDDEN')
    else if (s === 404) codes.push('NOT_FOUND')
    else if (s >= 500) codes.push('SERVER_ERROR')
  }
  return { status: s, codes, messages, message, details, compile, structured }
}

export function isProductionApiError(value: unknown): value is ProductionApiError {
  return !!value && typeof value === 'object' && Array.isArray((value as ProductionApiError).codes)
    && typeof (value as ProductionApiError).status === 'number'
}
