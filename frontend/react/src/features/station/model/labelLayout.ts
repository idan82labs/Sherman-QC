/**
 * Screen-space layout of the 3D bend labels (pure; no DOM, no three.js).
 *
 * Bend anchors on real parts sit close together (joggle pairs, short bends
 * next to long ones), so labels drawn at their anchors cover each other and
 * the operator can read or tap the wrong bend. This places every label so
 * that no two visible labels overlap and none covers a viewer control:
 *
 * - placement order: the selected bend first (it always gets the spot nearest
 *   its anchor and is drawn on top), then bends that need attention, then the
 *   rest by sequence;
 * - each label tries its anchor, then offsets on growing rings around it
 *   (8 directions); the first free spot inside the viewport and outside every
 *   reserved rectangle (toolbar, legend, flags) wins;
 * - after the anchor itself, the offset a label used last frame is tried
 *   first, so labels do not jump around while the camera orbits;
 * - a label that finds no free spot takes the least-overlapping candidate and
 *   is reported (`overlapping`), so a check can fail on it;
 * - an offset label gets a leader line from its anchor to the nearest point of
 *   its box.
 *
 * Coordinates are CSS pixels with the origin at the viewer's top-left corner.
 */

export interface Rect {
  x: number
  y: number
  w: number
  h: number
}

export interface LabelInput {
  id: string
  /** Anchor on screen (CSS px). */
  ax: number
  ay: number
  /** Label box size (CSS px). */
  w: number
  h: number
  /** Anchor behind the camera or outside the clip volume. */
  hidden: boolean
  /** Placement priority: lower first. */
  priority: number
}

export interface LabelPlacement {
  id: string
  /** Top-left of the label box. */
  x: number
  y: number
  w: number
  h: number
  /** Index of the candidate used (0 = centred on the anchor). */
  candidate: number
  hidden: boolean
  /** No free spot was found: the label overlaps another label or a control. */
  overlapping: boolean
  /** Leader from the anchor to the box (null when centred on the anchor). */
  leader: { x1: number; y1: number; x2: number; y2: number } | null
}

export interface LayoutOptions {
  width: number
  height: number
  /** Viewer controls labels must not cover. */
  reserved: Rect[]
  /** Candidate index each label used last time (stability while orbiting). */
  previous?: ReadonlyMap<string, number>
  /** Gap kept between boxes (px). */
  gap?: number
  /** Distance kept from the viewer edge (px). */
  margin?: number
  /** Number of rings of offsets tried around the anchor. */
  rings?: number
}

/** Total overlap (px²) still counted as a free spot (floating-point noise). */
const FREE_EPS = 0.5

const DIRS: [number, number][] = [
  [0, -1], [0, 1], [1, 0], [-1, 0],
  [1, -1], [-1, -1], [1, 1], [-1, 1],
]

/** Candidate box centres for a label: the anchor, then `rings` rings of 8 offsets. */
function candidates(l: LabelInput, rings: number, gap: number): [number, number][] {
  const out: [number, number][] = [[l.ax, l.ay]]
  for (let k = 1; k <= rings; k++) {
    for (const [dx, dy] of DIRS) {
      out.push([l.ax + dx * k * (l.w * 0.55 + gap), l.ay + dy * k * (l.h + gap)])
    }
  }
  return out
}

function overlapArea(a: Rect, b: Rect, pad: number): number {
  const w = Math.min(a.x + a.w + pad, b.x + b.w + pad) - Math.max(a.x - pad, b.x - pad)
  const h = Math.min(a.y + a.h + pad, b.y + b.h + pad) - Math.max(a.y - pad, b.y - pad)
  return w > 0 && h > 0 ? w * h : 0
}

/** Area of `r` outside the box [margin, width-margin] x [margin, height-margin]. */
function outsideArea(r: Rect, width: number, height: number, margin: number): number {
  const ix = Math.max(0, Math.min(r.x + r.w, width - margin) - Math.max(r.x, margin))
  const iy = Math.max(0, Math.min(r.y + r.h, height - margin) - Math.max(r.y, margin))
  // Clamped: the subtraction of two equal floating-point areas is not exactly 0.
  const out = r.w * r.h - ix * iy
  return out > 0.5 ? out : 0
}

function leaderTo(ax: number, ay: number, r: Rect): { x1: number; y1: number; x2: number; y2: number } {
  const x2 = Math.max(r.x, Math.min(ax, r.x + r.w))
  const y2 = Math.max(r.y, Math.min(ay, r.y + r.h))
  return { x1: ax, y1: ay, x2, y2 }
}

export function layoutLabels(labels: readonly LabelInput[], opts: LayoutOptions): LabelPlacement[] {
  const gap = opts.gap ?? 4
  const margin = opts.margin ?? 6
  const rings = opts.rings ?? 4
  const order = labels
    .map((l, i) => ({ l, i }))
    .sort((a, b) => a.l.priority - b.l.priority || a.i - b.i)
  const placed: Rect[] = []
  const out = new Map<string, LabelPlacement>()
  for (const { l } of order) {
    if (l.hidden) {
      out.set(l.id, { id: l.id, x: l.ax - l.w / 2, y: l.ay - l.h / 2, w: l.w, h: l.h, candidate: 0, hidden: true, overlapping: false, leader: null })
      continue
    }
    const cands = candidates(l, rings, gap)
    const prev = opts.previous?.get(l.id)
    // The anchor itself first, then last frame's offset, then the rings.
    const tryOrder = prev != null && prev > 0 && prev < cands.length ? [0, prev, ...cands.keys()] : [...cands.keys()]
    let best = -1
    let bestCost = Infinity
    for (const c of tryOrder) {
      const [cx, cy] = cands[c]
      const r: Rect = { x: cx - l.w / 2, y: cy - l.h / 2, w: l.w, h: l.h }
      let cost = outsideArea(r, opts.width, opts.height, margin) * 4
      for (const p of placed) cost += overlapArea(r, p, gap / 2)
      for (const q of opts.reserved) cost += overlapArea(r, q, gap) * 4
      if (cost < FREE_EPS) {
        best = c
        bestCost = 0
        break
      }
      if (cost < bestCost) {
        best = c
        bestCost = cost
      }
    }
    const [cx, cy] = cands[best]
    const r: Rect = { x: cx - l.w / 2, y: cy - l.h / 2, w: l.w, h: l.h }
    placed.push(r)
    out.set(l.id, {
      id: l.id,
      ...r,
      candidate: best,
      hidden: false,
      overlapping: bestCost >= FREE_EPS,
      leader: best === 0 ? null : leaderTo(l.ax, l.ay, r),
    })
  }
  return labels.map(l => out.get(l.id) as LabelPlacement)
}

/** Pairs of visible placements whose boxes intersect (for checks). */
export function overlappingPairs(placements: readonly LabelPlacement[]): [string, string][] {
  const vis = placements.filter(p => !p.hidden)
  const pairs: [string, string][] = []
  for (let i = 0; i < vis.length; i++) {
    for (let j = i + 1; j < vis.length; j++) {
      if (overlapArea(vis[i], vis[j], 0) > 0) pairs.push([vis[i].id, vis[j].id])
    }
  }
  return pairs
}
