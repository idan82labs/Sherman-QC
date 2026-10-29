/**
 * HTTP client for the onboarding wizard. Uses the shared axios instance (Bearer
 * token, /api base). Every rejection is a `ProductionApiError`, and every
 * response goes through `model/adapters.ts`, so pages never see a raw body.
 *
 * Routes consumed, as implemented by C05 (`apps/api/routes/production_recipes.py`):
 *
 *   GET    /api/production/parts                                -> {parts: [{part, sku, onboarding_state, active_recipe}]}
 *   POST   /api/production/parts                                -> 201 {part, sku}; 409 SKU_COLLISION; 422 SKU_INVALID
 *   GET    /api/production/parts/{id}/onboarding                -> checklist, blockers, ready_to_release, last_compile, ...
 *   GET    /api/production/parts/{id}/release-draft             -> {revision, request}
 *   PUT    /api/production/parts/{id}/release-draft             body ReleaseRequestV1; 409 RELEASE_DRAFT_REVISION_CHANGED
 *   POST   /api/production/parts/{id}/recipe/release            body ReleaseRequestV1 -> 201 released | 200 unchanged
 *   GET    /api/production/recipes/{recipe_id}/expectations     -> {status, expectations: sherman_recipe_expectations_v1}
 *   GET    /api/production/parts/{id}/flat                      -> PartFlatRecord | 404 FLAT_NOT_FOUND
 *   POST   /api/production/parts/{id}/flat                      multipart "file" -> PartFlatRecord (parsed, UPLOADED)
 *   POST   /api/production/parts/{id}/flat/proposal?expected_revision=N -> PartFlatRecord (PROPOSED)
 *   PUT    /api/production/parts/{id}/flat/binding              {expected_revision, state, bend_line_map?, mirrored?}
 *   DELETE /api/production/parts/{id}/flat?expected_revision=N
 *   GET    /api/production/parts/{id}/chirality                 -> sherman_chirality_status_v0 (display only; schedules the map)
 *   POST   /api/production/parts/{id}/chirality/retry           -> the same, re-running a FAILED map
 *
 * Optional, not implemented by C05 today: `GET|POST /parts/{id}/compile-preview`
 * (a dry-run compile before release). A bare 404/405 from it means "not
 * available on this server" and the Review step says so; it never pretends a
 * preview ran.
 *
 * Legacy catalog routes reused unchanged: CAD and drawing upload, Vendor Setup
 * read (`services/api.ts` partCatalogApi).
 */
import axios from 'axios'
import { api, partCatalogApi } from '../../../services/api'
import { parseExpectations, parseFlatRecord, parseOnboarding, parsePartsList, parseReleaseResult } from '../model/adapters'
import { normaliseApiError } from '../model/errors'
import { staleDraftRepair } from '../model/checklist'
import { parseChiralityStatus, type ChiralityStatus } from '../model/chirality'
import type {
  CompilePreviewResponse, CreateProductionPartRequest, ExpectationsResponse, FlatBindingDecisionRequest, FlatIntakeV1,
  OnboardingResponse, ProductionApiError, ProductionPartRow, ReleaseDraftResponse, ReleaseRequestV1, ReleaseResponse,
  VendorSetupLite,
} from '../model/types'

const BASE = '/production'

function toApiError(error: unknown): ProductionApiError {
  if (axios.isAxiosError(error)) return normaliseApiError(error.response?.status ?? 0, error.response?.data, error.message)
  return normaliseApiError(0, null, error instanceof Error ? error.message : '')
}

function invalidBody(what: string): ProductionApiError {
  return normaliseApiError(502, { detail: { codes: ['RESPONSE_INVALID'], messages: { RESPONSE_INVALID: `unexpected ${what} response` } } })
}

async function call<T>(run: () => Promise<{ data: T; status: number }>): Promise<T> {
  try {
    return (await run()).data
  } catch (error) {
    throw toApiError(error)
  }
}

/** Like `call`, but a 404 means "nothing there yet" and resolves to null. */
async function optional<T>(run: () => Promise<{ data: T; status: number }>): Promise<T | null> {
  try {
    return (await run()).data
  } catch (error) {
    const normalised = toApiError(error)
    if (normalised.status === 404) return null
    throw normalised
  }
}

const enc = encodeURIComponent

/** Compile-preview availability: a structured answer, none yet, or no such route on this server. */
export type CompilePreviewState =
  | { kind: 'preview'; preview: CompilePreviewResponse }
  | { kind: 'none' }
  | { kind: 'unavailable' }

export const onboardingKeys = {
  parts: ['production', 'parts'] as const,
  onboarding: (partId: string) => ['production', 'onboarding', partId] as const,
  vendorSetup: (partId: string) => ['vendor-setup', partId] as const,
  releaseDraft: (partId: string) => ['production', 'release-draft', partId] as const,
  flat: (partId: string) => ['production', 'flat', partId] as const,
  compilePreview: (partId: string) => ['production', 'compile-preview', partId] as const,
  expectations: (recipeId: string) => ['production', 'expectations', recipeId] as const,
  chirality: (partId: string) => ['production', 'chirality', partId] as const,
}

export const onboardingApi = {
  listParts: async (): Promise<ProductionPartRow[]> => parsePartsList(await call(() => api.get<unknown>(`${BASE}/parts`))),

  createPart: async (body: CreateProductionPartRequest): Promise<{ part: { id: string; part_number: string }; sku: string }> => {
    const data = await call(() => api.post<{ part?: { id?: unknown; part_number?: unknown }; sku?: unknown }>(`${BASE}/parts`, body))
    if (typeof data?.part?.id !== 'string') throw invalidBody('create part')
    return { part: { id: data.part.id, part_number: String(data.part.part_number ?? '') }, sku: String(data.sku ?? '') }
  },

  getOnboarding: async (partId: string): Promise<OnboardingResponse> => {
    const parsed = parseOnboarding(await call(() => api.get<unknown>(`${BASE}/parts/${enc(partId)}/onboarding`)))
    if (!parsed) throw invalidBody('onboarding')
    return parsed
  },

  getVendorSetup: async (partId: string): Promise<VendorSetupLite> => {
    try {
      return await partCatalogApi.getVendorSetup(partId) as unknown as VendorSetupLite
    } catch (error) {
      throw toApiError(error)
    }
  },

  /**
   * One-click repair of a bend setup that went stale only because a new drawing
   * was attached (`staleDraftRepair` === 'DRAWING_ONLY'): re-read the setup and
   * save the same draft against the current drawing through the Vendor Setup
   * PUT route (which re-checks CAD, revision and candidates server-side). Returns
   * 'REVIEW' without writing when the CAD side changed too.
   */
  refreshStaleDraft: async (partId: string): Promise<'UPDATED' | 'NOT_STALE' | 'REVIEW'> => {
    try {
      const setup = await partCatalogApi.getVendorSetup(partId)
      const repair = staleDraftRepair(setup as unknown as VendorSetupLite)
      if (repair === 'NONE') return 'NOT_STALE'
      if (repair === 'REVIEW' || !setup.draft) return 'REVIEW'
      const { updated_at: savedAt, draft_revision: revision, is_stale: wasStale, ...rest } = setup.draft
      void savedAt; void wasStale // read-only server fields, never sent back
      await partCatalogApi.saveVendorSetup(partId, { ...rest, drawing_file_hash: setup.part.drawing_file_hash, expected_draft_revision: revision ?? 0 })
      return 'UPDATED'
    } catch (error) {
      throw toApiError(error)
    }
  },

  uploadCad: async (partId: string, file: File, onProgress?: (percent: number) => void) => {
    try {
      return await partCatalogApi.uploadCad(partId, file, true, onProgress)
    } catch (error) {
      throw toApiError(error)
    }
  },

  uploadDrawing: async (partId: string, file: File, onProgress?: (percent: number) => void) => {
    try {
      return await partCatalogApi.uploadDrawing(partId, file, onProgress)
    } catch (error) {
      throw toApiError(error)
    }
  },

  getReleaseDraft: async (partId: string): Promise<ReleaseDraftResponse> => {
    const data = await optional(() => api.get<ReleaseDraftResponse>(`${BASE}/parts/${enc(partId)}/release-draft`))
    return data && typeof data.revision === 'number' ? { revision: data.revision, request: data.request ?? null, updated_at: data.updated_at ?? null } : { revision: 0, request: null }
  },

  putReleaseDraft: (partId: string, body: ReleaseRequestV1) =>
    call(() => api.put<ReleaseDraftResponse>(`${BASE}/parts/${enc(partId)}/release-draft`, body)),

  release: async (partId: string, body: ReleaseRequestV1): Promise<ReleaseResponse & { httpStatus: number }> => {
    let response: { data: unknown; status: number }
    try {
      response = await api.post<unknown>(`${BASE}/parts/${enc(partId)}/recipe/release`, body, { timeout: 180_000 })
    } catch (error) {
      throw toApiError(error)
    }
    const parsed = parseReleaseResult(response.data, response.status)
    if (!parsed) throw invalidBody('release')
    return { ...parsed, httpStatus: response.status }
  },

  getExpectations: async (recipeId: string): Promise<ExpectationsResponse> => {
    const parsed = parseExpectations(await call(() => api.get<unknown>(`${BASE}/recipes/${enc(recipeId)}/expectations`)))
    if (!parsed) throw invalidBody('expectations')
    return parsed
  },

  /** The part's flat, or null when it has none (404) or the body is not a flat record. */
  getFlat: async (partId: string): Promise<FlatIntakeV1 | null> =>
    parseFlatRecord(await optional(() => api.get<unknown>(`${BASE}/parts/${enc(partId)}/flat`))),

  uploadFlat: async (partId: string, file: File, onProgress?: (percent: number) => void): Promise<FlatIntakeV1 | null> => {
    const form = new FormData()
    form.append('file', file)
    return parseFlatRecord(await call(() => api.post<unknown>(`${BASE}/parts/${enc(partId)}/flat`, form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 180_000,
      onUploadProgress: event => { if (onProgress && event.total) onProgress(Math.round((event.loaded / event.total) * 100)) },
    })))
  },

  proposeFlat: async (partId: string, expectedRevision: number): Promise<FlatIntakeV1 | null> =>
    parseFlatRecord(await call(() => api.post<unknown>(`${BASE}/parts/${enc(partId)}/flat/proposal`, null,
      { params: { expected_revision: expectedRevision }, timeout: 180_000 }))),

  decideFlatBinding: async (partId: string, body: FlatBindingDecisionRequest): Promise<FlatIntakeV1 | null> =>
    parseFlatRecord(await call(() => api.put<unknown>(`${BASE}/parts/${enc(partId)}/flat/binding`, body))),

  deleteFlat: (partId: string, expectedRevision: number) =>
    call(() => api.delete<unknown>(`${BASE}/parts/${enc(partId)}/flat`, { params: { expected_revision: expectedRevision } })),

  getCompilePreview: async (partId: string): Promise<CompilePreviewState> => {
    try {
      const response = await api.get<CompilePreviewResponse>(`${BASE}/parts/${enc(partId)}/compile-preview`)
      return response.data && typeof response.data.status === 'string' ? { kind: 'preview', preview: response.data } : { kind: 'unavailable' }
    } catch (error) {
      const normalised = toApiError(error)
      if ((normalised.status === 404 || normalised.status === 405) && !normalised.structured) return { kind: 'unavailable' }
      if (normalised.status === 404) return { kind: 'none' }
      throw normalised
    }
  },

  getChirality: async (partId: string): Promise<ChiralityStatus> => {
    const parsed = parseChiralityStatus(await call(() => api.get<unknown>(`${BASE}/parts/${enc(partId)}/chirality`)))
    if (!parsed) throw invalidBody('chirality')
    return parsed
  },

  retryChirality: async (partId: string): Promise<ChiralityStatus> => {
    const parsed = parseChiralityStatus(await call(() => api.post<unknown>(`${BASE}/parts/${enc(partId)}/chirality/retry`)))
    if (!parsed) throw invalidBody('chirality')
    return parsed
  },

  startCompilePreview: (partId: string, body: ReleaseRequestV1) =>
    call(() => api.post<CompilePreviewResponse>(`${BASE}/parts/${enc(partId)}/compile-preview`, body, { timeout: 180_000 })),
}
