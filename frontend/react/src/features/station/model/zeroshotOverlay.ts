/**
 * The job-level zero-shot overlay (`sherman_zeroshot_overlay_v1`, `GET /api/line/jobs/{id}/zeroshot-overlay`).
 *
 * Display / REVIEW only: overlay lines are detections, never measurements, and never green. The only verdict
 * change the overlay may make is a pass-like E3 state (ALL_MEASURED_IN_TOLERANCE / STAGE_IN_TOLERANCE) lowered
 * to REVIEW_REQUIRED by an UNEXPECTED_FEATURE_REVIEW flag. `validateOverlay` re-checks that rule here, and an
 * overlay that names another job's scan or summary state is dropped (the screen then shows E3 alone).
 */
import type { ShopfloorInspectionResult, SummaryState } from './types'

export const OVERLAY_SCHEMA = 'sherman_zeroshot_overlay_v1'
const PASS_LIKE: readonly string[] = ['ALL_MEASURED_IN_TOLERANCE', 'STAGE_IN_TOLERANCE']
const KINDS = ['WITHHELD_BEND', 'UNEXPECTED_BEND', 'REFUSED_SCAN_DETECTION'] as const
export type OverlayLineKind = (typeof KINDS)[number]

export interface OverlayLine {
  line_id: string
  kind: OverlayLineKind
  decision: 'REVIEW' | 'UNVERIFIED'
  source: 'ZS_LANE_V1'
  bend_uid: string | null
  start_raw_mm: [number, number, number]
  end_raw_mm: [number, number, number]
  start_frame_mm: [number, number, number] | null
  end_frame_mm: [number, number, number] | null
  length_mm: number
  incl_deg: number | null
  /** NOT_MEASURED_JOGGLE: the zs angle is never shown as a number on a joggle. */
  angle_status?: 'SHOWN' | 'NOT_MEASURED_JOGGLE'
  score: number
}

export interface AngleCheck {
  code: 'INDEPENDENT_ANGLE_DISAGREES'
  bend_uid: string
  e3_angle_deg: number
  zs_angle_deg: number
  delta_deg: number
}

export interface ZeroshotOverlay {
  schema: typeof OVERLAY_SCHEMA
  overlay_version: string
  scan_sha256: string
  engine_result_sha256: string
  inspection_validity: string
  e3_summary_state: SummaryState
  effective_summary_state: SummaryState
  verdict_lowered: boolean
  review_reasons: string[]
  lines: OverlayLine[]
  flags: { code: 'UNEXPECTED_FEATURE_REVIEW'; line_ids: string[] }[]
  /** REVIEW reasons on E3 angle FAILs; they never change the verdict. */
  angle_checks?: AngleCheck[]
  withheld_bends: number
  withheld_bends_located: number
}

const isObj = (x: unknown): x is Record<string, unknown> => !!x && typeof x === 'object' && !Array.isArray(x)

/** The overlay if it is well formed, keeps the never-upgrade rule and belongs to `result`; else null. */
export function validateOverlay(raw: unknown, result: ShopfloorInspectionResult): ZeroshotOverlay | null {
  if (!isObj(raw) || raw.schema !== OVERLAY_SCHEMA) return null
  if (raw.scan_sha256 !== result.scan.sha256 || raw.e3_summary_state !== result.summary.state) return null
  if (raw.inspection_validity !== result.inspection_validity.state) return null
  const e3 = String(raw.e3_summary_state)
  const eff = String(raw.effective_summary_state)
  const flags = Array.isArray(raw.flags) ? raw.flags : null
  const lines = Array.isArray(raw.lines) ? raw.lines : null
  if (!flags || !lines) return null
  // never creates or upgrades a pass; only a flagged pass-like state becomes REVIEW_REQUIRED
  const want = flags.length > 0 && PASS_LIKE.includes(e3) ? 'REVIEW_REQUIRED' : e3
  if (eff !== want || raw.verdict_lowered !== (eff !== e3)) return null
  for (const l of lines) {
    if (!isObj(l) || !KINDS.includes(l.kind as OverlayLineKind)) return null
    if (l.decision !== 'REVIEW' && l.decision !== 'UNVERIFIED') return null
    if (typeof l.line_id !== 'string' || typeof l.length_mm !== 'number' || typeof l.score !== 'number') return null
    if (l.angle_status === 'NOT_MEASURED_JOGGLE' && l.incl_deg != null) return null
  }
  if (raw.angle_checks !== undefined && !Array.isArray(raw.angle_checks)) return null
  return raw as unknown as ZeroshotOverlay
}

/** The summary state to show: E3's, or REVIEW_REQUIRED when the overlay lowered a pass-like one. */
export function effectiveSummaryState(result: ShopfloorInspectionResult, overlay: ZeroshotOverlay | null): SummaryState {
  return overlay && overlay.verdict_lowered ? overlay.effective_summary_state : result.summary.state
}
