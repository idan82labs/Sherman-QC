/**
 * Defensive adapters from the C05 `/api/production` bodies to the wizard's types.
 *
 * Every function takes `unknown`, never throws, and returns null (or drops the
 * row) for anything that does not have the fields the UI relies on. Pages and
 * the Vendor Setup bridge only ever see adapted values, so a server that
 * answers with an older or partial shape degrades to "nothing there" instead
 * of crashing a render.
 *
 * Shapes adapted (apps/api/services/production_recipe_service.py, C05):
 * - GET /parts rows `{part: _part_summary, sku, onboarding_state, active_recipe}`;
 * - GET /parts/{id}/onboarding (checklist, blockers, last_compile, flat, ...);
 * - flat routes: `PartFlatRecord.to_dict()` plus `review`
 *   (`{flat_sha256, revision, state, parse, proposal, binding}`);
 * - POST /recipe/release `{outcome, recipe_id, version, recipe_sha256, warnings}`;
 * - GET /recipes/{id}/expectations, whose `expectations` is the
 *   `sherman_recipe_expectations_v1` document (`release_service.expectations_document`).
 */
import type {
  ActiveRecipeRef, CharacteristicNominalV1, CompiledBendLite, CompiledPartLite, CompileReportV1, ExpectationsResponse,
  FlatBendLineV1, FlatDecisionV1, FlatFormat, FlatIntakeV1, FlatParseV1, FlatProposalRecordV1, FlatRecordState,
  LastCompileV1, OnboardingPart, OnboardingResponse, OnboardingState, ProductionPartRow, ReleaseResponse, ServerChecklistItem,
} from './types'
import { FLAT_RECORD_STATES, ONBOARDING_STATES } from './types'

type Obj = Record<string, unknown>

const isObj = (value: unknown): value is Obj => !!value && typeof value === 'object' && !Array.isArray(value)
const str = (value: unknown): string | null => (typeof value === 'string' ? value : null)
const num = (value: unknown): number | null => (typeof value === 'number' && Number.isFinite(value) ? value : null)
const bool = (value: unknown): boolean | null => (typeof value === 'boolean' ? value : null)
const strList = (value: unknown): string[] => (Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [])
const HEX64 = /^[0-9a-f]{64}$/

function strMap(value: unknown): Record<string, string> {
  if (!isObj(value)) return {}
  const out: Record<string, string> = {}
  for (const [key, item] of Object.entries(value)) if (typeof item === 'string') out[key] = item
  return out
}

// ---------------------------------------------------------------- parts

function partOf(raw: unknown): OnboardingPart | null {
  if (!isObj(raw)) return null
  const id = str(raw.id) ?? str(raw.part_id)
  const partNumber = str(raw.part_number)
  if (!id || partNumber == null) return null
  return {
    ...(raw as unknown as OnboardingPart),
    id,
    part_number: partNumber,
    part_name: str(raw.part_name) ?? '',
    revision: str(raw.revision),
    material: str(raw.material),
    customer: str(raw.customer),
    cad_file_path: str(raw.cad_file_path),
    cad_file_hash: str(raw.cad_file_hash),
    cad_uploaded_at: str(raw.cad_uploaded_at),
    cad_format: str(raw.cad_format),
    has_cad: bool(raw.has_cad) ?? undefined,
    has_drawing: bool(raw.has_drawing) ?? undefined,
    drawing_file_path: str(raw.drawing_file_path),
    drawing_file_hash: str(raw.drawing_file_hash),
    drawing_uploaded_at: str(raw.drawing_uploaded_at),
  }
}

function onboardingStateOf(value: unknown): OnboardingState | null {
  return typeof value === 'string' && (ONBOARDING_STATES as readonly string[]).includes(value) ? value as OnboardingState : null
}

function activeRecipeOf(raw: unknown): ActiveRecipeRef | null {
  if (!isObj(raw)) return null
  const id = str(raw.recipe_id), version = num(raw.version), sha = str(raw.recipe_sha256)
  if (!id || version == null || !sha) return null
  const source = isObj(raw.source) ? raw.source : null
  return {
    recipe_id: id, version, recipe_sha256: sha,
    content_sha256: str(raw.content_sha256) ?? undefined,
    released_at: str(raw.released_at),
    released_by: isObj(raw.released_by) ? raw.released_by as unknown as ActiveRecipeRef['released_by'] : null,
    source: source && str(source.cad_sha256) && str(source.drawing_sha256) && num(source.draft_revision) != null
      ? { cad_sha256: str(source.cad_sha256)!, drawing_sha256: str(source.drawing_sha256)!, draft_revision: num(source.draft_revision)! } : null,
  }
}

/** GET /parts body -> rows. A row without a part id or a known state is dropped (never an `undefined` link). */
export function parsePartsList(data: unknown): ProductionPartRow[] {
  const rows = isObj(data) && Array.isArray(data.parts) ? data.parts : []
  const out: ProductionPartRow[] = []
  for (const row of rows) {
    if (!isObj(row)) continue
    // C05 nests the catalog summary under `part`; a flat row is accepted too.
    const part = partOf(isObj(row.part) ? row.part : row)
    const state = onboardingStateOf(row.onboarding_state)
    if (!part || !state) continue
    out.push({
      part_id: part.id, part_number: part.part_number, sku: str(row.sku), sku_display: str(row.sku_display),
      part_name: part.part_name, revision: part.revision, material: part.material,
      onboarding_state: state, active_recipe: activeRecipeOf(row.active_recipe),
    })
  }
  return out
}

// ---------------------------------------------------------------- onboarding

function checklistOf(raw: unknown): ServerChecklistItem[] {
  if (!Array.isArray(raw)) return []
  return raw.flatMap(item => (isObj(item) && typeof item.key === 'string' && typeof item.ok === 'boolean'
    ? [{ key: item.key, ok: item.ok, codes: strList(item.codes), detail: strList(item.detail) }] : []))
}

/** E1 compile report from C05 (`_compile_report`); null when it has no issue list. */
export function parseCompileReport(raw: unknown): CompileReportV1 | null {
  if (!isObj(raw)) return null
  const issues = Array.isArray(raw.issues) ? raw.issues.flatMap(issue => {
    if (!isObj(issue) || typeof issue.code !== 'string') return []
    return [{
      code: issue.code, severity: issue.severity === 'WARNING' ? 'WARNING' as const : 'BLOCKER' as const,
      bend_uid: str(issue.bend_uid), bend_label: str(issue.bend_label), graph_bend_id: str(issue.graph_bend_id), detail: str(issue.detail),
    }]
  }) : []
  const unmarked = Array.isArray(raw.unmarked_graph_bends) ? raw.unmarked_graph_bends.flatMap(item => (isObj(item)
    && typeof item.graph_bend_id === 'string' && num(item.sweep_deg) != null
    ? [{ graph_bend_id: item.graph_bend_id, sweep_deg: num(item.sweep_deg)!, inner_radius_mm: num(item.inner_radius_mm) ?? Number.NaN }] : [])) : []
  return {
    status: str(raw.status) ?? undefined, compiled_part_sha256: str(raw.compiled_part_sha256), compiler_version: str(raw.compiler_version),
    blocker_codes: strList(raw.blocker_codes), warning_codes: strList(raw.warning_codes), issues, unmarked_graph_bends: unmarked,
    detail: str(raw.detail), consistency_problems: strList(raw.consistency_problems),
  }
}

function lastCompileOf(raw: unknown): LastCompileV1 | null {
  if (!isObj(raw) || typeof raw.status !== 'string') return null
  return {
    status: raw.status, codes: strList(raw.codes), at: str(raw.at), recipe_sha256: str(raw.recipe_sha256),
    applies_to_current_draft: raw.applies_to_current_draft === true, report: parseCompileReport(raw.report),
  }
}

/** GET /parts/{id}/onboarding -> OnboardingResponse, or null without a part and a known state. */
export function parseOnboarding(data: unknown): OnboardingResponse | null {
  if (!isObj(data)) return null
  const part = partOf(data.part)
  const state = onboardingStateOf(data.onboarding_state)
  if (!part || !state) return null
  return {
    part, sku: str(data.sku), checklist: checklistOf(data.checklist), blockers: strList(data.blockers),
    messages: isObj(data.messages) ? strMap(data.messages) : null,
    // Only a literal `true` counts: the release button must never open on a missing or odd flag.
    ready_to_release: data.ready_to_release === true,
    release_would_be_unchanged: data.release_would_be_unchanged === true,
    active_recipe: activeRecipeOf(data.active_recipe), onboarding_state: state, last_compile: lastCompileOf(data.last_compile),
    release_draft_revision: num(data.release_draft_revision) ?? undefined,
    links: isObj(data.links) ? strMap(data.links) : null,
  }
}

// ---------------------------------------------------------------- flat record

function bendLineOf(raw: unknown): FlatBendLineV1 | null {
  if (!isObj(raw) || typeof raw.line_id !== 'string' || !Array.isArray(raw.start_mm) || !Array.isArray(raw.end_mm)) return null
  return {
    line_id: raw.line_id, start_mm: raw.start_mm as number[], end_mm: raw.end_mm as number[], source: str(raw.source) ?? 'UNKNOWN',
    layer: str(raw.layer), linetype: str(raw.linetype), direction: str(raw.direction), angle_deg: num(raw.angle_deg), annotation: str(raw.annotation),
  }
}

function loopsOf(raw: unknown): Array<Array<[number, number]>> {
  return Array.isArray(raw) ? raw.filter(Array.isArray) as Array<Array<[number, number]>> : []
}

/**
 * C05 parse document -> FlatParseV1. `{status: PARSED, summary, geometry}` or
 * `{status: PARSE_FAILED, reason, detail}`; anything else is PARSE_FAILED with no
 * code, never PARSED.
 */
function parseOf(raw: unknown, recordState: FlatRecordState): FlatParseV1 {
  if (!isObj(raw) || raw.status !== 'PARSED' || recordState === 'PARSE_FAILED') {
    const doc = isObj(raw) ? raw : {}
    return { status: 'PARSE_FAILED', code: str(doc.reason) ?? str(doc.code), message: str(doc.detail) ?? str(doc.message), bend_lines: [] }
  }
  const summary = isObj(raw.summary) ? raw.summary : {}
  const geometry = isObj(raw.geometry) ? raw.geometry : {}
  const lines = (Array.isArray(summary.bend_lines) ? summary.bend_lines : Array.isArray(geometry.bend_lines) ? geometry.bend_lines : [])
    .map(bendLineOf).filter((line): line is FlatBendLineV1 => !!line)
  return {
    status: 'PARSED', format: str(summary.format) ?? undefined, units: str(summary.units) ?? undefined,
    scale_to_mm: num(summary.scale_to_mm) ?? undefined, outer_points: num(summary.outer_points) ?? undefined,
    inner_loops: num(summary.inner_loops) ?? undefined, outer_area_mm2: num(summary.outer_area_mm2) ?? undefined,
    thickness_mm: num(summary.thickness_mm), notes: strList(summary.notes), bend_lines: lines,
    outline_mm: Array.isArray(geometry.outer) ? geometry.outer as Array<[number, number]> : null,
    inner_loops_mm: loopsOf(geometry.inner), decimated: geometry.decimated === true,
  }
}

function proposalRecordOf(raw: unknown): FlatProposalRecordV1 | null {
  if (!isObj(raw) || typeof raw.status !== 'string') return null
  const inner = isObj(raw.proposal) && isObj(raw.proposal.bend_line_map) && typeof raw.proposal.mirrored === 'boolean' ? raw.proposal : null
  const rec = isObj(raw.reconciliation) ? raw.reconciliation : null
  return {
    status: raw.status,
    proposal: inner ? { ...(inner as unknown as NonNullable<FlatProposalRecordV1['proposal']>), bend_line_map: strMap(inner.bend_line_map),
      fit_choice: isObj(inner.fit_choice) ? {
        equivalent_fits: num(inner.fit_choice.equivalent_fits) ?? 0, narrowed_by: strList(inner.fit_choice.narrowed_by),
        decided_by: str(inner.fit_choice.decided_by) ?? '', ambiguous: inner.fit_choice.ambiguous === true,
        // Not stated means not determined: the UI then asks the operator.
        mirror_determined: inner.fit_choice.mirror_determined === true,
        alternative_maps: Array.isArray(inner.fit_choice.alternative_maps) ? inner.fit_choice.alternative_maps.filter(isObj).map(strMap) : [],
      } : null } : null,
    reconciliation: rec ? { mark_not_in_flange_graph: strList(rec.mark_not_in_flange_graph), unmarked_bend: num(rec.unmarked_bend) ?? undefined } : null,
    basis_sha256: str(raw.basis_sha256), cad_sha256: str(raw.cad_sha256), bend_labels: strMap(raw.bend_labels),
    proposed_at: str(raw.proposed_at), reason: str(raw.reason), detail: str(raw.detail),
  }
}

function decisionOf(raw: unknown): FlatDecisionV1 | null {
  if (!isObj(raw)) return null
  if (raw.state === 'UNBOUND') {
    return { state: 'UNBOUND', bend_line_map: {}, mirrored: null, method: null, basis_sha256: null, cad_sha256: null,
      decided_by: isObj(raw.decided_by) ? raw.decided_by as unknown as FlatDecisionV1['decided_by'] : null, decided_at: str(raw.decided_at) }
  }
  if (raw.state !== 'OPERATOR_CONFIRMED' || !isObj(raw.bend_line_map)) return null
  return {
    state: 'OPERATOR_CONFIRMED', bend_line_map: strMap(raw.bend_line_map), mirrored: bool(raw.mirrored),
    method: raw.method === 'GRAPH_MATCH_RIGID_FIT' || raw.method === 'MANUAL' ? raw.method : null,
    basis_sha256: str(raw.basis_sha256), cad_sha256: str(raw.cad_sha256),
    confirmed_by: isObj(raw.confirmed_by) ? raw.confirmed_by as unknown as FlatDecisionV1['confirmed_by'] : null, confirmed_at: str(raw.confirmed_at),
  }
}

/**
 * C05 flat record -> FlatIntakeV1, or null for anything that is not a flat record
 * (no sha256, no revision, unknown state). A CONFIRMED / DECLINED record whose
 * decision does not parse is downgraded to "nothing decided" (UPLOADED), so it
 * can never be shown as bound.
 */
export function parseFlatRecord(data: unknown): FlatIntakeV1 | null {
  if (!isObj(data)) return null
  const sha = str(data.flat_sha256)
  const revision = num(data.revision)
  const rawState = str(data.state)
  if (!sha || !HEX64.test(sha) || revision == null || !rawState || !(FLAT_RECORD_STATES as readonly string[]).includes(rawState)) return null
  let state = rawState as FlatRecordState
  const format: FlatFormat = data.format === 'STEP' ? 'STEP' : 'DXF'
  const binding = decisionOf(data.binding)
  if ((state === 'CONFIRMED' && binding?.state !== 'OPERATOR_CONFIRMED') || (state === 'DECLINED' && binding?.state !== 'UNBOUND')) state = 'UPLOADED'
  const parse = parseOf(data.parse, state)
  if (parse.status !== 'PARSED') state = 'PARSE_FAILED'
  return {
    part_id: str(data.part_id) ?? '', revision, state,
    flat: { sha256: sha, original_filename: str(data.original_filename), format, size_bytes: num(data.size_bytes), uploaded_at: str(data.uploaded_at) },
    parse, proposal: proposalRecordOf(data.proposal),
    binding: state === 'CONFIRMED' || state === 'DECLINED' ? binding : null,
  }
}

// ---------------------------------------------------------------- release

/** POST /recipe/release body -> ReleaseResponse; null when it names no recipe. */
export function parseReleaseResult(data: unknown, httpStatus: number): ReleaseResponse | null {
  if (!isObj(data)) return null
  // C05: top-level {outcome, recipe_id, version, recipe_sha256}; tolerate {status, recipe: {...}} too.
  const nested = isObj(data.recipe) ? data.recipe : {}
  const id = str(data.recipe_id) ?? str(nested.recipe_id)
  const version = num(data.version) ?? num(nested.recipe_version) ?? num(nested.version)
  const sha = str(data.recipe_sha256) ?? str(nested.recipe_sha256)
  if (!id || version == null || !sha) return null
  const outcome = str(data.outcome) ?? str(data.status)
  return {
    status: outcome === 'unchanged' || (outcome == null && httpStatus === 200) ? 'unchanged' : 'released',
    recipe: { recipe_id: id, recipe_version: version, recipe_sha256: sha, content_sha256: str(data.content_sha256) ?? str(nested.content_sha256) ?? undefined,
      supersedes_version: num(data.supersedes_version) ?? num(nested.supersedes_version) },
    warnings: strList(data.warnings),
  }
}

// ---------------------------------------------------------------- expectations

/** GET /recipes/{id}/expectations; an unknown status is PENDING (keeps polling, never READY). */
export function parseExpectations(data: unknown): ExpectationsResponse | null {
  if (!isObj(data)) return null
  const status = data.status === 'READY' || data.status === 'FAILED' ? data.status : 'PENDING'
  const error = isObj(data.error) && typeof data.error.code === 'string' ? { code: data.error.code, message: str(data.error.message) ?? undefined } : null
  return { recipe_id: str(data.recipe_id) ?? undefined, status, algo_version: str(data.algo_version), job_id: str(data.job_id),
    expectations: (isObj(data.expectations) ? data.expectations : null) as ExpectationsResponse['expectations'], error }
}

/**
 * The C05 `sherman_recipe_expectations_v1` document as a CompiledPartLite. Only
 * released (COMPILED) parts have one; its warnings become WARNING issues and the
 * recipe angle per bend is kept for the disagreement display.
 */
export function compiledFromExpectationsDoc(doc: unknown): CompiledPartLite | null {
  if (!isObj(doc) || doc.schema !== 'sherman_recipe_expectations_v1' || !Array.isArray(doc.bends) || !Array.isArray(doc.characteristics)) return null
  const bends = doc.bends.filter(isObj).flatMap(bend => (typeof bend.bend_uid === 'string' && num(bend.sequence_index) != null
    ? [{ ...(bend as unknown as CompiledBendLite), recipe_included_deg: num(bend.recipe_included_deg), recipe_angle_source: str(bend.recipe_angle_source) }] : []))
  const characteristics = doc.characteristics.filter(item => isObj(item) && typeof item.characteristic_id === 'string' && typeof item.kind === 'string') as unknown as CharacteristicNominalV1[]
  const warnings = Array.isArray(doc.warnings) ? doc.warnings.filter(isObj).flatMap(w => (typeof w.code === 'string'
    ? [{ code: w.code, severity: 'WARNING' as const, bend_uid: str(w.bend_uid), detail: str(w.detail) ?? undefined }] : [])) : []
  return {
    schema: 'sherman_recipe_expectations_v1', status: 'COMPILED', compiled_part_sha256: str(doc.compiled_part_sha256) ?? undefined,
    thickness: isObj(doc.thickness) ? doc.thickness as unknown as CompiledPartLite['thickness'] : null,
    k_factor: isObj(doc.k_factor) ? doc.k_factor as unknown as CompiledPartLite['k_factor'] : null,
    bends, characteristics, unmarked_graph_bends: [], issues: warnings,
  }
}
