/** Screen-space presentation only. Never changes a bend ID or model coordinate. */
export const MANUAL_MARKER_LIMIT = 32
export const MANUAL_MARKER_GAP = 8
export const MANUAL_MARKER_MIN_SIZE = 48
const INSET = 8
const GRID_LIMIT = 16
const RINGS = 12

export interface MarkerRect { x: number; y: number; width: number; height: number }
export interface ManualMarkerAnchor {
  id: string
  anchorX: number
  anchorY: number
  width: number
  height: number
}
export interface ManualMarkerPlacement extends ManualMarkerAnchor, MarkerRect {}
export interface ManualMarkerLayout {
  mode: 'callouts' | 'tray'
  placements: ManualMarkerPlacement[]
  tray: MarkerRect | null
  contentHeight: number
  candidateChecks: number
}

/** Intrinsic text width is independent of an earlier assigned layout width. */
export function manualMarkerIntrinsicWidth(textWidths: readonly number[], horizontalInset: number) {
  if (textWidths.length > 3 || !textWidths.every(value => Number.isFinite(value) && value >= 0)
    || !Number.isFinite(horizontalInset) || horizontalInset < 0) throw new RangeError('Invalid marker text measurement')
  return Math.ceil(Math.max(48, Math.min(180, Math.max(0, ...textWidths) + horizontalInset)))
}

function overlaps(a: MarkerRect, b: MarkerRect, gap = MANUAL_MARKER_GAP) {
  return a.x < b.x + b.width + gap && a.x + a.width + gap > b.x
    && a.y < b.y + b.height + gap && a.y + a.height + gap > b.y
}

function pointCovered(rect: MarkerRect, x: number, y: number) {
  return x >= rect.x - 8 && x <= rect.x + rect.width + 8
    && y >= rect.y - 8 && y <= rect.y + rect.height + 8
}

/** Nearest point on the label border, for a leader to its unmodified anchor. */
export function manualMarkerLeaderEnd(rect: MarkerRect, anchorX: number, anchorY: number) {
  const centerX = rect.x + rect.width / 2, centerY = rect.y + rect.height / 2
  const dx = anchorX - centerX, dy = anchorY - centerY
  const scale = Math.max(Math.abs(dx) / (rect.width / 2), Math.abs(dy) / (rect.height / 2), 1)
  return { x: centerX + dx / scale, y: centerY + dy / scale }
}

/**
 * Bounded, deterministic packing. An uncrowded midpoint is retained exactly
 * (apart from integer-pixel rounding/clipping). A crowded layout prefers slots
 * not covering projected anchors. Work is <=32*(1+96+256) candidate scans per
 * pass, with one bounded uniform-grid recovery; no random or iterative solver.
 * A tray is explicit when targets cannot fit; it never claims all are visible.
 */
export function layoutManualMarkers(
  input: readonly ManualMarkerAnchor[],
  viewport: { width: number; height: number },
  reserved: readonly MarkerRect[] = [],
  forceTray = false,
): ManualMarkerLayout {
  if (input.length > MANUAL_MARKER_LIMIT || reserved.length > 16
    || !Number.isFinite(viewport.width) || !Number.isFinite(viewport.height)
    || viewport.width < 64 || viewport.height < 64) throw new RangeError('Unsupported marker layout viewport/count')
  const ids = new Set<string>()
  const items = input.map(item => {
    if (!item.id || ids.has(item.id) || ![item.anchorX, item.anchorY, item.width, item.height].every(Number.isFinite)
      || item.width < MANUAL_MARKER_MIN_SIZE || item.height < MANUAL_MARKER_MIN_SIZE) throw new RangeError('Invalid marker input')
    ids.add(item.id)
    return { ...item, width: Math.ceil(item.width), height: Math.ceil(item.height) }
  })
  for (const rect of reserved) {
    if (![rect.x, rect.y, rect.width, rect.height].every(Number.isFinite) || rect.width < 0 || rect.height < 0) {
      throw new RangeError('Invalid marker safe zone')
    }
  }
  let candidateChecks = 0
  const valid = (rect: MarkerRect, placed: readonly MarkerRect[]) => {
    candidateChecks++
    return rect.x >= INSET && rect.y >= INSET && rect.x + rect.width <= viewport.width - INSET
      && rect.y + rect.height <= viewport.height - INSET
      && !reserved.some(other => overlaps(rect, other)) && !placed.some(other => overlaps(rect, other))
  }
  const prefer = (item: ManualMarkerAnchor): ManualMarkerPlacement => ({ ...item,
    x: Math.max(INSET, Math.min(Math.floor(viewport.width - item.width - INSET), Math.round(item.anchorX - item.width / 2))),
    y: Math.max(INSET, Math.min(Math.floor(viewport.height - item.height - INSET), Math.round(item.anchorY - item.height / 2))) })
  const preferred = items.map(prefer)
  if (!forceTray && preferred.every((item, index) => valid(item, preferred.slice(0, index)))) {
    return { mode: 'callouts', placements: preferred, tray: null, contentHeight: viewport.height, candidateChecks }
  }
  const maxWidth = Math.max(48, ...items.map(item => item.width))
  const maxHeight = Math.max(48, ...items.map(item => item.height))
  const columns = Math.min(GRID_LIMIT, Math.floor((viewport.width - 2 * INSET + MANUAL_MARKER_GAP) / (maxWidth + MANUAL_MARKER_GAP)))
  const rows = Math.min(GRID_LIMIT, Math.floor((viewport.height - 2 * INSET + MANUAL_MARKER_GAP) / (maxHeight + MANUAL_MARKER_GAP)))
  const slots: MarkerRect[] = []
  for (let row = 0; row < rows; row++) for (let col = 0; col < columns; col++) {
    const rect = { x: INSET + col * (maxWidth + MANUAL_MARKER_GAP), y: INSET + row * (maxHeight + MANUAL_MARKER_GAP), width: maxWidth, height: maxHeight }
    if (valid(rect, [])) slots.push(rect)
  }
  const score = (rect: MarkerRect, item: ManualMarkerAnchor) => {
    const dx = rect.x + rect.width / 2 - item.anchorX, dy = rect.y + rect.height / 2 - item.anchorY
    const covered = items.reduce((count, anchor) => count + Number(pointCovered(rect, anchor.anchorX, anchor.anchorY)), 0)
    return dx * dx + dy * dy + covered * 40_000
  }
  const placements: ManualMarkerPlacement[] = []
  for (const item of items) {
    const candidates = [prefer(item)]
    for (let ring = 1; ring <= RINGS; ring++) for (let direction = 0; direction < 8; direction++) {
      const angle = direction * Math.PI / 4, distance = ring * (Math.max(item.width, item.height) + MANUAL_MARKER_GAP)
      candidates.push({ ...item, x: Math.round(item.anchorX + Math.cos(angle) * distance - item.width / 2),
        y: Math.round(item.anchorY + Math.sin(angle) * distance - item.height / 2) })
    }
    candidates.push(...slots.map(slot => ({ ...item, x: slot.x, y: slot.y })))
    let best: ManualMarkerPlacement | null = null, bestScore = Infinity
    for (const candidate of candidates) {
      if (!valid(candidate, placements)) continue
      const cost = score(candidate, item)
      if (!best || cost < bestScore) { best = candidate; bestScore = cost }
    }
    if (!best) break
    placements.push(best)
  }
  if (!forceTray && placements.length === items.length) return { mode: 'callouts', placements, tray: null, contentHeight: viewport.height, candidateChecks }
  // A uniform grid cannot fragment. If there are enough safe slots, recover
  // all markers rather than silently omitting a late marker in greedy packing.
  if (!forceTray && slots.length >= items.length) {
    const remaining = [...slots], recovered = items.map(item => {
      let best = 0
      for (let index = 1; index < remaining.length; index++) if (score(remaining[index], item) < score(remaining[best], item)) best = index
      const slot = remaining.splice(best, 1)[0]
      return { ...item, x: slot.x, y: slot.y }
    })
    return { mode: 'callouts', placements: recovered, tray: null, contentHeight: viewport.height, candidateChecks }
  }
  const top = Math.max(INSET, ...reserved.filter(rect => rect.y < viewport.height / 2).map(rect => rect.y + rect.height + MANUAL_MARKER_GAP))
  const bottom = Math.min(viewport.height - INSET, ...reserved.filter(rect => rect.y >= viewport.height / 2).map(rect => rect.y - MANUAL_MARKER_GAP))
  const trayCandidates: MarkerRect[] = [{ x: INSET, y: top, width: viewport.width - 2 * INSET, height: bottom - top }]
  const left = Math.max(INSET, ...reserved.filter(rect => rect.x + rect.width / 2 < viewport.width / 2).map(rect => rect.x + rect.width + MANUAL_MARKER_GAP))
  const right = Math.min(viewport.width - INSET, ...reserved.filter(rect => rect.x + rect.width / 2 >= viewport.width / 2).map(rect => rect.x - MANUAL_MARKER_GAP))
  trayCandidates.push({ x: left, y: INSET, width: right - left, height: viewport.height - 2 * INSET }, ...slots)
  const safeTrays = trayCandidates.filter(rect => rect.width >= 48 && rect.height >= maxHeight && valid(rect, []))
  safeTrays.sort((a, b) => b.width * b.height - a.width * a.height)
  const tray = safeTrays[0]
  if (!tray) throw new RangeError('No safe minimum-sized marker tray')
  const trayWidth = Math.min(maxWidth, tray.width), trayColumns = Math.max(1, Math.floor((tray.width + MANUAL_MARKER_GAP) / (trayWidth + MANUAL_MARKER_GAP)))
  const trayPlacements = items.map((item, index) => ({ ...item, width: Math.min(item.width, trayWidth),
    x: (index % trayColumns) * (trayWidth + MANUAL_MARKER_GAP), y: Math.floor(index / trayColumns) * (maxHeight + MANUAL_MARKER_GAP) }))
  return { mode: 'tray', placements: trayPlacements, tray, contentHeight: Math.ceil(items.length / trayColumns) * (maxHeight + MANUAL_MARKER_GAP) - MANUAL_MARKER_GAP, candidateChecks }
}
