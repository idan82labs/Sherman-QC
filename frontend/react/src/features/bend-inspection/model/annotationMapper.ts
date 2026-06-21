import type { BendMatch } from '../../../shared/api'
import type { BendAnnotation } from '../../../shared/3d'

export function isCountableBend(match: BendMatch): boolean {
  if (match.countable_in_regression === false) return false
  if ((match.feature_type ?? '').toUpperCase() === 'ROLLED_SECTION') return false
  return true
}

export function parseLinePoint(raw?: [number, number, number] | number[] | null): [number, number, number] | null {
  if (!Array.isArray(raw) || raw.length < 3) return null
  const x = Number(raw[0])
  const y = Number(raw[1])
  const z = Number(raw[2])
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null
  return [x, y, z]
}

export function midpoint3D(
  a: [number, number, number],
  b: [number, number, number],
): [number, number, number] {
  return [
    (a[0] + b[0]) / 2,
    (a[1] + b[1]) / 2,
    (a[2] + b[2]) / 2,
  ]
}

function interpolate3D(
  a: [number, number, number],
  b: [number, number, number],
  t: number,
): [number, number, number] {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ]
}

export function midpointFromMatch(match: BendMatch): [number, number, number] | null {
  const a = parseLinePoint(match.display_detected_line_start ?? match.detected_line_start)
  const b = parseLinePoint(match.display_detected_line_end ?? match.detected_line_end)
  if (a && b) return midpoint3D(a, b)
  const c = parseLinePoint(match.display_cad_line_start ?? match.cad_line_start)
  const d = parseLinePoint(match.display_cad_line_end ?? match.cad_line_end)
  if (c && d) return midpoint3D(c, d)
  return null
}

function highestEndpoint(
  a: [number, number, number],
  b: [number, number, number],
): [number, number, number] {
  return a[2] >= b[2] ? a : b
}

export function toViewerStatus(status: BendMatch['status']): BendAnnotation['status'] {
  switch (status) {
    case 'PASS':
      return 'pass'
    case 'FAIL':
      return 'fail'
    case 'WARNING':
      return 'warning'
    default:
      return 'pending'
  }
}

export function buildBendAnnotations(matches: BendMatch[], focusedBendId?: string | null): BendAnnotation[] {
  const displayLabels = buildDisplayBendLabelMap(matches)
  const overlays: BendAnnotation[] = []
  for (const match of matches) {
    if (!isCountableBend(match)) {
      continue
    }
    const lineStart = parseLinePoint(match.display_cad_line_start ?? match.cad_line_start)
    const lineEnd = parseLinePoint(match.display_cad_line_end ?? match.cad_line_end)
    if (!lineStart || !lineEnd) continue
    const detectedLineStart = parseLinePoint(match.display_detected_line_start ?? match.detected_line_start)
    const detectedLineEnd = parseLinePoint(match.display_detected_line_end ?? match.detected_line_end)
    const featureType = match.feature_type ?? null
    const isRolledSection = featureType === 'ROLLED_SECTION'
    const rawMidpoint = midpoint3D(lineStart, lineEnd)
    const anchorStart = isRolledSection ? lineStart : (detectedLineStart ?? lineStart)
    const anchorEnd = isRolledSection ? lineEnd : (detectedLineEnd ?? lineEnd)
    const rolledAnchor = isRolledSection
      ? interpolate3D(highestEndpoint(anchorStart, anchorEnd), highestEndpoint(anchorStart, anchorEnd) === anchorStart ? anchorEnd : anchorStart, 0.38)
      : null
    overlays.push({
      id: match.bend_id,
      label: displayLabels.get(match.bend_id) ?? match.bend_id,
      position: rolledAnchor ?? rawMidpoint,
      calloutAnchor: rolledAnchor ?? rawMidpoint,
      featureType: featureType ?? undefined,
      lineStart: isRolledSection ? undefined : lineStart,
      lineEnd: isRolledSection ? undefined : lineEnd,
      detectedLineStart: isRolledSection ? undefined : detectedLineStart ?? undefined,
      detectedLineEnd: isRolledSection ? undefined : detectedLineEnd ?? undefined,
      expectedAngle: match.target_angle,
      measuredAngle: match.measured_angle,
      deviation: match.angle_deviation,
      status: toViewerStatus(match.status),
      active: focusedBendId === match.bend_id,
      displayGeometryCanonical: false,
      radiusDeviation: match.radius_deviation,
      lineCenterDeviationMm: match.line_center_deviation_mm,
      toleranceAngle: match.tolerance_angle,
      toleranceRadius: match.tolerance_radius,
    })
  }
  return overlays
}

export function buildDisplayBendLabelMap(matches: BendMatch[]): Map<string, string> {
  const labels = new Map<string, string>()
  let rolledIndex = 0
  let processIndex = 0
  const countableMatches = matches.filter(isCountableBend)
  const hasProcessFeatures = matches.length > countableMatches.length
  const countableOrder = new Map(countableMatches.map((match, index) => [match.bend_id, index]))

  const orderedCountableMatches = hasProcessFeatures && countableMatches.length === 2
    ? [...countableMatches].sort((a, b) => {
        const angleGap = Number(a.target_angle) - Number(b.target_angle)
        if (Math.abs(angleGap) > 0.25) return angleGap
        return (countableOrder.get(a.bend_id) ?? 0) - (countableOrder.get(b.bend_id) ?? 0)
      })
    : countableMatches

  orderedCountableMatches.forEach((match, index) => {
    labels.set(match.bend_id, `B${index + 1}`)
  })

  for (const match of matches) {
    if (isCountableBend(match)) continue

    if ((match.feature_type ?? '').toUpperCase() === 'ROLLED_SECTION' || (match.bend_form ?? '').toUpperCase() === 'ROLLED') {
      rolledIndex += 1
      labels.set(match.bend_id, `Rolled ${rolledIndex}`)
      continue
    }

    processIndex += 1
    labels.set(match.bend_id, `Process ${processIndex}`)
  }

  return labels
}

export function preferredFocusedBendId(
  matches: BendMatch[],
  operatorActions?: Array<{ bend_id: string }> | undefined,
): string | null {
  const countableMatches = matches.filter(isCountableBend)
  const issueMatches = countableMatches.filter((match) => match.status === 'FAIL' || match.status === 'WARNING')
  const operatorCountableBendId = operatorActions
    ?.find((action) => countableMatches.some((match) => match.bend_id === action.bend_id))
    ?.bend_id
  if (issueMatches.length > 1) {
    return null
  }
  return issueMatches[0]?.bend_id
    ?? countableMatches.find((match) => match.status === 'PASS')?.bend_id
    ?? countableMatches.find((match) => match.status !== 'NOT_DETECTED')?.bend_id
    ?? operatorCountableBendId
    ?? countableMatches[0]?.bend_id
    ?? null
}
