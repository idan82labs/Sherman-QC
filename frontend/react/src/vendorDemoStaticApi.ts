/** Static, browser-local adapter for the tablet vendor demonstration only. */
import { api } from './services/api'
import type { Part, VendorSetupDraft, VendorSetupResponse } from './types'

interface ManifestRow {
  id: string
  part_number: string
  cad_sha256: string
  candidate_count: number
}

interface Fixture {
  part_number: string
  cad_sha256: string
  candidate_generation: string
  candidates: VendorSetupResponse['candidates']
  flange_surfaces: NonNullable<VendorSetupResponse['flange_surfaces']>
  flange_surface_status: string
  unfold_preview?: VendorSetupResponse['unfold_preview']
}

const manifestPromise: Promise<ManifestRow[]> = fetch('/vendor-corpus/manifest.json', { cache: 'no-store' })
  .then(response => {
    if (!response.ok) throw new Error('CAD demo manifest unavailable')
    return response.json()
  })
  .then(value => {
    if (value.schema_version !== 1 || !Array.isArray(value.parts) || value.parts.length < 5) throw new Error('CAD demo manifest invalid')
    return value.parts as ManifestRow[]
  })

const fixtures = new Map<string, Promise<Fixture>>()
const sessionDrafts = new Map<string, VendorSetupDraft>()
const storageKey = (id: string) => `sherman.vendorCorpusDemo.draft.${id}`

async function fixtureFor(id: string): Promise<{ row: ManifestRow; fixture: Fixture }> {
  const row = (await manifestPromise).find(item => item.id === id && /^corpus-\d{8}$/.test(item.id))
  if (!row) throw new Error('Part not found in this CAD demo')
  let request = fixtures.get(id)
  if (!request) {
    request = fetch(`/vendor-corpus/${row.part_number}.json`, { cache: 'no-store' }).then(async response => {
      if (!response.ok) throw new Error('CAD demo fixture unavailable')
      const value = await response.json() as Fixture
      if (value.part_number !== row.part_number || value.cad_sha256 !== row.cad_sha256 || value.candidates.length !== row.candidate_count) {
        throw new Error('CAD demo source binding mismatch')
      }
      return value
    })
    fixtures.set(id, request)
  }
  return { row, fixture: await request }
}

function part(row: ManifestRow): Part {
  return {
    id: row.id, part_number: row.part_number, part_name: 'Existing corpus CAD',
    customer: null, revision: 'source-CAD-demo', cad_file_path: '/static-CAD-demo',
    cad_file_hash: row.cad_sha256, cad_uploaded_at: '2026-09-24',
    preview_mesh_path: `/vendor-corpus/${row.part_number}.ply`,
    drawing_file_path: null, drawing_file_hash: null, drawing_uploaded_at: null,
    material: null, default_tolerance_angle: 1, default_tolerance_linear: 0.5,
    bend_count: row.candidate_count, has_embedding: false, embedding_version: null,
    embedding_computed_at: null, created_at: '', updated_at: '', erp_import_id: null, notes: null,
  }
}

function readDraft(id: string): VendorSetupDraft | null {
  if (sessionDrafts.has(id)) return sessionDrafts.get(id) || null
  try {
    const stored = sessionStorage.getItem(storageKey(id))
    if (stored) {
      const draft = JSON.parse(stored) as VendorSetupDraft
      sessionDrafts.set(id, draft)
      return draft
    }
  } catch { /* This tablet may block session storage. In-memory tagging still works. */ }
  return null
}

function saveDraft(id: string, draft: VendorSetupDraft): void {
  sessionDrafts.set(id, draft)
  try { sessionStorage.setItem(storageKey(id), JSON.stringify(draft)) } catch { /* Session remains in memory. */ }
}

function validGeneralMeasurements(draft: VendorSetupDraft): boolean {
  const rows = draft.general_measurements || []
  if (rows.length > 100 || new Set(rows.map(row => row.id)).size !== rows.length) return false
  return rows.every(row => {
    if (!row.id.trim() || row.id.length > 100 || !row.label.trim() || row.label.length > 100 || row.note.length > 500
        || [row.point_start, row.point_end].some(point => point.length !== 3 || !point.every(Number.isFinite))) return false
    const distance = Math.hypot(...row.point_start.map((value, index) => value - row.point_end[index]))
    if (!Number.isFinite(row.cad_distance) || distance <= 0 || !Number.isFinite(distance)
        || Math.abs(row.cad_distance - distance) > Math.max(1e-12, Math.abs(distance) * 1e-9)
        || row.target_mm != null && (!Number.isFinite(row.target_mm) || row.target_mm <= 0)) return false
    if (row.kind === 'points') return row.bend_ids.length === 0
    return row.kind === 'bend_pair' && row.bend_ids.length === 2 && new Set(row.bend_ids).size === 2
      && row.bend_ids.every(id => draft.marks.some(mark => mark.feature_id === id && mark.status === 'active' && mark.line_start && mark.line_end))
  })
}

api.defaults.adapter = async config => {
  const method = (config.method || 'get').toUpperCase()
  const path = new URL(config.url || '', window.location.origin).pathname
  let data: unknown
  if (path === '/parts' && method === 'GET') {
    const rows = await manifestPromise
    data = { parts: rows.map(part), total: rows.length, limit: rows.length, offset: 0 }
  } else {
    const match = /^\/parts\/(corpus-\d{8})\/vendor-setup$/.exec(path)
    if (!match) throw new Error('Only the five CAD demo parts are available on this site')
    const id = match[1]
    const { row, fixture } = await fixtureFor(id)
    if (method === 'GET') {
      data = { part: part(row), candidates: fixture.candidates, candidate_generation: fixture.candidate_generation,
        draft: readDraft(id), flange_surfaces: fixture.flange_surfaces,
        flange_surface_status: fixture.flange_surface_status, unfold_preview: fixture.unfold_preview }
    } else if (method === 'PUT') {
      const draft = JSON.parse(String(config.data)) as VendorSetupDraft
      const active = draft.marks.filter(mark => mark.status === 'active').map(mark => mark.feature_id)
      if (draft.cad_file_hash !== row.cad_sha256 || draft.candidate_generation !== fixture.candidate_generation
          || draft.sequence.length !== active.length || new Set(draft.sequence).size !== active.length
          || active.some(id => !draft.sequence.includes(id))
          || (draft.machine_moves?.length && (() => {
            let cursor = 0
            return draft.machine_moves!.some(move => {
              const start = draft.sequence.indexOf(move[0], cursor)
              if (!move.length || start < 0 || start + move.length > draft.sequence.length || draft.sequence.slice(start, start + move.length).some((id, index) => id !== move[index])) return true
              cursor = start + move.length
              return false
            })
          })())
          || !validGeneralMeasurements(draft)
          || (active.length > 0 && draft.operator_sequence_confirmed !== true)) {
        throw new Error('Invalid local demo draft or unconfirmed bend order')
      }
      const saved = { ...draft, draft_revision: (readDraft(id)?.draft_revision || 0) + 1, is_stale: false }
      saveDraft(id, saved)
      data = { draft: saved }
    } else throw new Error('Unsupported CAD demo action')
  }
  return { data, status: 200, statusText: 'OK', headers: {}, config }
}
