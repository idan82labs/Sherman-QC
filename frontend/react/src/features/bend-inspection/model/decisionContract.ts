import type { BendDecisionContract, BendInspectionReport, BendMatch } from '../../../shared/api'

type DecisionState = 'FORMED' | 'UNFORMED' | 'UNKNOWN'
export type PresentationBendMatch = BendMatch & { decision_state: DecisionState }

type ReportInput = Pick<BendInspectionReport, 'matches' | 'summary' | 'decision_contract' | 'alignment'>

function effectiveCountableFlag(match: BendMatch): unknown {
  const nested = (match as BendMatch & { cad_bend?: { countable_in_regression?: boolean } }).cad_bend
  return match.countable_in_regression ?? nested?.countable_in_regression ?? true
}

export function isDecisionCountableBend(match: BendMatch): boolean {
  // Invalid flags remain potential targets, never silently excluded. The
  // versioned validator rejects them before exposing positive evidence.
  return effectiveCountableFlag(match) !== false
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringList(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((entry) => typeof entry === 'string')
}

function isCount(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
}

/** Validate the versioned boundary, not merely a truthy API object. */
function validatedContract(report: ReportInput): BendDecisionContract | null {
  const alignment: unknown = report.alignment
  if (alignment != null && !isRecord(alignment)) return null
  if (isRecord(alignment) && alignment.alignment_abstained != null
    && typeof alignment.alignment_abstained !== 'boolean') return null
  const raw: unknown = report.decision_contract
  if (!isRecord(raw) || raw.schema_version !== 1 || raw.kind !== 'BEND_DECISION_CONTRACT_V1'
    || raw.production_disposition !== 'NOT_AUTHORIZED'
    || typeof raw.registration !== 'string' || !['ABSTAINED', 'UNVERIFIED'].includes(raw.registration)
    || typeof raw.render_evidence !== 'string' || !['UNAVAILABLE', 'UNVERIFIED'].includes(raw.render_evidence)
    || !stringList(raw.reasons) || !Array.isArray(raw.bends) || !isRecord(raw.observed_counts)) return null
  if (isRecord(alignment) && alignment.alignment_abstained === true && raw.registration !== 'ABSTAINED') return null
  const counts = raw.observed_counts
  if (!['formed', 'unformed', 'unknown', 'total', 'partially_observed'].every((key) => isCount(counts[key]))) return null
  if (Number(counts.formed) + Number(counts.unformed) + Number(counts.unknown) !== counts.total
    || Number(counts.partially_observed) > Number(counts.unknown)) return null
  const rows = new Map<string, Record<string, unknown>>()
  for (const row of raw.bends) {
    if (!isRecord(row) || typeof row.bend_id !== 'string' || !row.bend_id.trim()
      || rows.has(row.bend_id) || typeof row.state !== 'string' || !['FORMED', 'UNFORMED', 'UNKNOWN'].includes(row.state)
      || typeof row.observation_detail !== 'string' || !stringList(row.reasons)) return null
    if (row.state === 'FORMED' && !['FORMED', 'OBSERVED_FORMED'].includes(row.observation_detail)) return null
    if (row.state === 'UNFORMED' && !['UNFORMED', 'OBSERVED_NOT_FORMED'].includes(row.observation_detail)) return null
    rows.set(row.bend_id, row)
  }
  const ids = new Set<string>()
  const expected = { formed: 0, unformed: 0, unknown: 0, total: 0, partially_observed: 0 }
  for (const match of report.matches) {
    if (typeof effectiveCountableFlag(match) !== 'boolean') return null
    if (!match.bend_id || ids.has(match.bend_id)) return null
    ids.add(match.bend_id)
    const row = rows.get(match.bend_id)
    if (!row) return null
    if (!isDecisionCountableBend(match)) continue
    expected.total += 1
    if (row.state === 'FORMED') expected.formed += 1
    else if (row.state === 'UNFORMED') expected.unformed += 1
    else {
      expected.unknown += 1
      if (row.observation_detail === 'PARTIALLY_OBSERVED') expected.partially_observed += 1
    }
  }
  if (rows.size !== ids.size || Object.entries(expected).some(([key, value]) => counts[key] !== value)) return null
  if (raw.registration === 'ABSTAINED' && [...rows.values()].some((row) => row.state !== 'UNKNOWN')) return null
  return raw as unknown as BendDecisionContract
}

export function decisionLabel(match: BendMatch): string {
  const state = (match as Partial<PresentationBendMatch>).decision_state
  if (state === 'FORMED') return 'Observed formed · unverified'
  if (state === 'UNFORMED') return 'Observed unformed · review'
  if (match.observability_detail_state === 'PARTIALLY_OBSERVED') return 'Partly observed · unknown'
  if (match.observability_detail_state === 'UNOBSERVED') return 'Not observed · unknown'
  return 'State unknown · review'
}

export function presentBendDecision(report: ReportInput) {
  const contract = validatedContract(report)
  const rows = new Map(contract?.bends.map((row) => [row.bend_id, row]) ?? [])
  const fallbackTotal = report.matches.filter(isDecisionCountableBend).length
  const observedCounts = contract?.observed_counts ?? {
    formed: 0, unformed: 0, unknown: fallbackTotal, total: fallbackTotal, partially_observed: 0,
  }
  const matches: PresentationBendMatch[] = report.matches.map((match) => {
    const row = rows.get(match.bend_id)
    const state = contract?.registration === 'ABSTAINED' ? 'UNKNOWN' : row?.state ?? 'UNKNOWN'
    return {
      ...match,
      decision_state: state,
      // V1 supplies neither calibrated registration nor authorized tolerance
      // disposition. Keep raw measurements, but never color them PASS/FAIL.
      status: 'NOT_DETECTED',
      observability_state: state,
      observability_detail_state: row?.observation_detail ?? 'UNKNOWN',
      action_item: state === 'UNFORMED'
        ? 'Review the observed unformed state; manufacturing stage and registration remain unverified.'
        : 'Review scan coverage and registration before any manufacturing decision.',
    }
  })
  const estimate = report.summary.structured_completed_bends
  return {
    available: contract !== null,
    counts: observedCounts,
    observedFormedRatio: contract ? `${observedCounts.formed}/${observedCounts.total}` : null,
    matches,
    registration: contract?.registration ?? 'UNVERIFIED',
    renderEvidence: contract?.render_evidence ?? 'UNAVAILABLE',
    reasons: contract?.reasons ?? ['DECISION_CONTRACT_UNAVAILABLE_OR_INVALID'],
    modelEstimate: isCount(estimate) ? estimate : null,
  }
}
