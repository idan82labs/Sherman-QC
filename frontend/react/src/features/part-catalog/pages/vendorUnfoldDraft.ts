import type { VendorSetupDraft, VendorSetupResponse, VendorUnfoldPreview } from '../../../types'

function samePoint(a: unknown, b: unknown): boolean {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== 3 || b.length !== 3) return false
  for (let axis = 0; axis < 3; axis++) {
    if (typeof a[axis] !== 'number' || !Number.isFinite(a[axis]) || a[axis] !== b[axis]) return false
  }
  return true
}

function nullableText(value: unknown): value is string | null {
  return value === null || typeof value === 'string'
}

/** Editable CAD marks must still belong to this exact response, not just reuse IDs. */
export function canMapUnfoldDraft(data: VendorSetupResponse | undefined, draft: VendorSetupDraft | null,
  preview: VendorUnfoldPreview | null): boolean {
  try {
  if (!data || !data.part || !draft || !preview || !Array.isArray(data.candidates) || !Array.isArray(preview.bend_lines)
    || typeof data.part.cad_file_hash !== 'string' || !/^[0-9a-f]{64}$/.test(data.part.cad_file_hash)
    || preview.source_sha256 !== data.part.cad_file_hash
    || typeof data.candidate_generation !== 'string' || data.candidate_generation.length === 0
    || !nullableText(data.part.cad_uploaded_at) || !nullableText(draft.cad_uploaded_at)
    || !nullableText(data.part.revision) || !nullableText(draft.part_revision)
    || draft.is_stale || draft.cad_file_hash !== data.part.cad_file_hash
    || draft.cad_uploaded_at !== data.part.cad_uploaded_at || draft.candidate_generation !== data.candidate_generation
    || draft.part_revision !== data.part.revision || !Array.isArray(draft.marks)) return false
  const candidates = new Map(data.candidates.map(mark => [mark.feature_id, mark]))
  const flatIds = new Set(preview.bend_lines.map(line => line.feature_id))
  const ids = new Set<string>()
  for (const mark of draft.marks) {
    if (!mark || typeof mark.feature_id !== 'string' || ids.has(mark.feature_id)) return false
    ids.add(mark.feature_id)
    if (mark.status === 'removed') continue
    if (mark.status !== 'active' || mark.source !== 'cad_candidate' || !flatIds.has(mark.feature_id)) return false
    const candidate = candidates.get(mark.feature_id)
    if (!candidate || candidate.source !== 'cad_candidate'
      || !samePoint(mark.line_start, candidate.line_start) || !samePoint(mark.line_end, candidate.line_end)) return false
  }
  return true
  } catch {
    return false
  }
}
