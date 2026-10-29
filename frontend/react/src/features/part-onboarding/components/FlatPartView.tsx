/**
 * Unfolded part drawn from the uploaded flat file (SVG, flat mm, not to scale).
 * Bend lines are dashed; each carries a label (flat line id, or the tagged order
 * number when the caller provides one). Unbound lines are amber.
 *
 * Labels sit at their line's midpoint unless that box would overlap a label
 * already placed (close parallel lines, e.g. a joggle): then the label slides
 * along its own line to the nearest free spot, so it stays on the line it names
 * and no label hides another.
 */
import { useMemo } from 'react'
import { fitProjector, flatGeometry, type Vec2 } from '../model/flat'
import type { FlatIntakeV1 } from '../model/types'

export interface FlatLineLabel {
  text: string
  bound: boolean
}

export default function FlatPartView({ intake, labels, selectedLineId, onSelectLine, ariaLabel, height = 360 }: {
  intake: FlatIntakeV1
  labels: Map<string, FlatLineLabel>
  selectedLineId?: string | null
  onSelectLine?: (lineId: string) => void
  ariaLabel: string
  height?: number
}) {
  const width = 900
  const geometry = useMemo(() => flatGeometry(intake), [intake])
  if (!geometry.bbox) return null
  const { project } = fitProjector(geometry.bbox, width, height, 44)
  const path = (loop: Vec2[]) => loop.map((point, index) => `${index ? 'L' : 'M'}${project(point).map(v => v.toFixed(1)).join(' ')}`).join(' ') + ' Z'
  const outlinePath = geometry.outline ? [geometry.outline, ...geometry.holes].map(path).join(' ') : null
  const placed: { x: number; y: number; w: number; h: number }[] = []
  const labelAt = new Map<string, { x: number; y: number; w: number }>()
  for (const line of geometry.lines) {
    const a = project(line.a), b = project(line.b)
    const text = (labels.get(line.id) ?? { text: line.id }).text
    const w = Math.max(28, text.length * 8.5 + 14)
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1
    const ux = (b[0] - a[0]) / len, uy = (b[1] - a[1]) / len
    const hits = (x: number, y: number) => placed.some(p => Math.abs(p.x - x) < (p.w + w) / 2 + 4 && Math.abs(p.y - y) < 30)
    let best = { x: (a[0] + b[0]) / 2, y: (a[1] + b[1]) / 2 }
    for (let k = 0; k <= 12; k++) {
      const d = (k % 2 ? 1 : -1) * Math.ceil(k / 2) * 32
      if (Math.abs(d) > len / 2 - 14) continue
      const x = (a[0] + b[0]) / 2 + ux * d, y = (a[1] + b[1]) / 2 + uy * d
      if (!hits(x, y)) { best = { x, y }; break }
    }
    placed.push({ ...best, w, h: 26 })
    labelAt.set(line.id, { ...best, w })
  }
  return (
    <svg className="onb-flat" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={ariaLabel} data-flat-view style={{ direction: 'ltr' }}>
      {outlinePath && <path className="flat-outline" d={outlinePath} fillRule="evenodd" />}
      {geometry.lines.map(line => {
        const a = project(line.a), b = project(line.b)
        const label = labels.get(line.id) ?? { text: line.id, bound: false }
        const spot = labelAt.get(line.id)!
        const mx = spot.x, my = spot.y
        const boxWidth = spot.w
        const interactive = !!onSelectLine
        return (
          <g key={line.id} data-flat-line={line.id} data-bound={label.bound}
            onClick={interactive ? () => onSelectLine!(line.id) : undefined}
            style={interactive ? { cursor: 'pointer' } : undefined}>
            <line className="bendline" x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} data-bound={label.bound} data-selected={selectedLineId === line.id} />
            {/* a wide transparent hit target keeps the line easy to tap */}
            {interactive && <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="transparent" strokeWidth={22} />}
            <rect className="lbl-bg" data-bound={label.bound} x={mx - boxWidth / 2} y={my - 13} width={boxWidth} height={26} rx={13} />
            <text className="lbl" x={mx} y={my + 4.5} textAnchor="middle">{label.text}</text>
          </g>
        )
      })}
    </svg>
  )
}
