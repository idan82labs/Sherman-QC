/**
 * Station part picker model (held-back scans): pure helpers shared by the
 * picker dialog and its node check (`scripts/check-station-part-picker.mjs`).
 *
 * Shapes mirror `GET /api/line/quarantine/held` and `GET /api/line/released-parts`
 * (`apps/api/services/part_picker_service.py`). The picker never decides
 * anything: the server validates every pick exactly like the hotfolder's
 * automatic match, and E3 still runs every identity check on the job.
 */

export type InspectionStage = 'FINAL' | 'IN_PROCESS'
export type SuggestionSource = 'RESOLVER_CANDIDATE' | 'FILENAME_SKU' | 'FILENAME_FUZZY'
export const DISMISS_REASONS = ['RESCAN', 'NOT_A_PART', 'DUPLICATE', 'WRONG_STATION', 'OTHER'] as const
export type DismissReason = (typeof DISMISS_REASONS)[number]

export interface PartSuggestion {
  sku: string
  recipe_id: string
  source: SuggestionSource
  score: number
  recipe_version: number | null
  revision: string | null
  part_name: string | null
  inspection_stage: InspectionStage | null
  assignable: boolean | null
}

export interface HeldScan {
  scan_file_id: string
  station_id: string
  file_name: string
  received_at: string
  quarantined_at: string | null
  reason_code: string
  resolvable: boolean
  message: string
  outcome: string | null
  outcome_reason: string | null
  parsed: {
    status: string | null
    sku: string | null
    revision: string | null
    expected_revision: string | null
    revision_status: string | null
    serial: string | null
    scan_label: string | null
    candidates: string[] | null
  }
  stats: {
    size_bytes: number | null
    format: string | null
    vertex_count: number | null
    face_count: number | null
    bbox_mm: [number[], number[]] | null
  }
  suggestions: PartSuggestion[]
}

export interface RecipeVersion {
  recipe_id: string
  recipe_version: number
  status: 'active' | 'superseded' | 'retired'
  released_at: string | null
  selectable: boolean
}

export interface ReleasedPart {
  sku: string
  recipe_id: string
  recipe_version: number
  revision: string | null
  part_name: string | null
  inspection_stage: InspectionStage | null
  bends_total: number | null
  compile_status: string
  assignable: boolean
  not_ready_reason: string | null
  preview_url: string | null
  versions: RecipeVersion[]
}

export interface HeldResponse {
  items: HeldScan[]
  canOperate: boolean
}

const SCAN_ID_RE = /^scan_[0-9a-f]{24}$/

/** Keep only well-formed held items (a malformed row is dropped, never guessed). */
export function parseHeld(body: unknown): HeldResponse {
  const obj = (body ?? {}) as { items?: unknown; can_operate?: unknown }
  const raw = Array.isArray(obj.items) ? obj.items : []
  const items = raw.filter(
    (i): i is HeldScan =>
      !!i &&
      typeof i === 'object' &&
      SCAN_ID_RE.test(String((i as HeldScan).scan_file_id)) &&
      typeof (i as HeldScan).file_name === 'string' &&
      typeof (i as HeldScan).reason_code === 'string',
  )
  return {
    items: items.map(i => ({ ...i, suggestions: Array.isArray(i.suggestions) ? i.suggestions : [] })),
    canOperate: obj.can_operate === true,
  }
}

/** Upper-case alphanumerics only (the server's SKU match is case- and separator-insensitive). */
export function normalizeQuery(q: string): string {
  return (q.match(/[A-Za-z0-9]+/g) ?? []).join('').toUpperCase()
}

export interface PickRow {
  part: ReleasedPart
  suggestion: PartSuggestion | null
}

/**
 * Rows for the picker list: the held scan's suggestions first (in server
 * order), then the search results that are not already suggested. A
 * suggestion whose part is not in `parts` (search not loaded yet) is shown
 * from its own fields.
 */
export function pickerRows(suggestions: PartSuggestion[], parts: ReleasedPart[], query: string): PickRow[] {
  const bySku = new Map(parts.map(p => [p.sku, p]))
  const nq = normalizeQuery(query)
  const rows: PickRow[] = []
  const seen = new Set<string>()
  for (const s of suggestions) {
    if (seen.has(s.sku)) continue
    if (nq && !normalizeQuery(s.sku).includes(nq) && !(s.part_name ?? '').toLowerCase().includes(query.trim().toLowerCase())) continue
    const part = bySku.get(s.sku) ?? suggestionAsPart(s)
    seen.add(s.sku)
    rows.push({ part, suggestion: s })
  }
  for (const p of parts) {
    if (seen.has(p.sku)) continue
    seen.add(p.sku)
    rows.push({ part: p, suggestion: null })
  }
  return rows
}

function suggestionAsPart(s: PartSuggestion): ReleasedPart {
  return {
    sku: s.sku,
    recipe_id: s.recipe_id,
    recipe_version: s.recipe_version ?? 0,
    revision: s.revision,
    part_name: s.part_name,
    inspection_stage: s.inspection_stage,
    bends_total: null,
    compile_status: s.assignable ? 'COMPILED' : 'UNKNOWN',
    assignable: s.assignable === true,
    not_ready_reason: null,
    preview_url: null,
    versions: [{ recipe_id: s.recipe_id, recipe_version: s.recipe_version ?? 0, status: 'active', released_at: null, selectable: true }],
  }
}

/** The stage pre-selected for a part: the recipe's own inspection stage (FINAL when unknown). */
export function defaultStage(part: Pick<ReleasedPart, 'inspection_stage'>): InspectionStage {
  return part.inspection_stage === 'IN_PROCESS' ? 'IN_PROCESS' : 'FINAL'
}

export interface AssignBody {
  recipe_id: string
  declared_stage?: InspectionStage
  stage_index?: number
}

/**
 * The assign request for the confirmed choice. The exact recipe id is sent
 * (not the SKU), so a version released after the operator confirmed is
 * refused (RECIPE_NOT_ACTIVE) instead of silently used. The stage is sent
 * only when it differs from the recipe's own, or with a bend count.
 */
export function assignBody(recipeId: string, part: Pick<ReleasedPart, 'inspection_stage'>, stage: InspectionStage, stageIndex: number | null): AssignBody {
  const body: AssignBody = { recipe_id: recipeId }
  if (stageIndex != null) {
    body.declared_stage = stage
    body.stage_index = stageIndex
  } else if (stage !== defaultStage(part)) {
    body.declared_stage = stage
  }
  return body
}

/** A stage choice the server will accept (IN_PROCESS on a FINAL recipe needs the bend count). */
export function stageChoiceComplete(part: Pick<ReleasedPart, 'inspection_stage' | 'bends_total'>, stage: InspectionStage, stageIndex: number | null): boolean {
  if (stage === 'FINAL') return stageIndex == null || stageIndex === part.bends_total
  if (stageIndex == null) return part.inspection_stage === 'IN_PROCESS'
  return stageIndex >= 0 && (part.bends_total == null || stageIndex <= part.bends_total)
}

export function canDismiss(reason: DismissReason | null, note: string): boolean {
  if (!reason) return false
  return reason !== 'OTHER' || note.trim().length > 0
}

/** "1.2 MB" style size, LTR digits. */
export function formatBytes(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

/** Bounding-box extents "120 × 80 × 40 mm" from `[[min], [max]]`. */
export function formatExtents(bbox: [number[], number[]] | null | undefined): string {
  if (!bbox || bbox.length !== 2) return '—'
  const [lo, hi] = bbox
  return `${[0, 1, 2].map(k => Math.round((hi[k] ?? 0) - (lo[k] ?? 0))).join(' × ')} mm`
}
