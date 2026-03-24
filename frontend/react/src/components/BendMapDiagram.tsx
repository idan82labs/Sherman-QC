import { useMemo } from 'react'
import type { BendMatch } from '../services/api'

interface BendMapDiagramProps {
  matches: BendMatch[]
  focusedBendId?: string | null
  onFocusBend?: (bendId: string) => void
}

type Point3 = [number, number, number]

const STATUS_FILL: Record<string, string> = {
  PASS: '#34d399',
  FAIL: '#f87171',
  WARNING: '#fbbf24',
  NOT_DETECTED: '#64748b',
}

function parsePoint(v: Point3 | null | undefined): Point3 | null {
  if (!v || !Array.isArray(v) || v.length < 3) return null
  if (!Number.isFinite(v[0]) || !Number.isFinite(v[1]) || !Number.isFinite(v[2])) return null
  return v as Point3
}

/** Top-down projection: X → right, Z → up */
function projectXZ(p: Point3): [number, number] {
  return [p[0], p[2]]
}

export default function BendMapDiagram({ matches, focusedBendId, onFocusBend }: BendMapDiagramProps) {
  const WIDTH = 800
  const HEIGHT = 120
  const PAD = 20
  const DOT_R = 8

  const bendData = useMemo(() => {
    const items: Array<{
      id: string
      label: string
      mid2d: [number, number]
      start2d: [number, number]
      end2d: [number, number]
      status: string
    }> = []

    for (const m of matches) {
      const s = parsePoint(m.cad_line_start)
      const e = parsePoint(m.cad_line_end)
      if (!s || !e) continue
      const mid: Point3 = [(s[0] + e[0]) / 2, (s[1] + e[1]) / 2, (s[2] + e[2]) / 2]
      items.push({
        id: m.bend_id,
        label: m.bend_id.replace('CAD_', ''),
        mid2d: projectXZ(mid),
        start2d: projectXZ(s),
        end2d: projectXZ(e),
        status: m.status,
      })
    }
    return items
  }, [matches])

  const { normalized, hullPoints } = useMemo(() => {
    if (bendData.length === 0) return { normalized: [], hullPoints: '' }

    // Collect all 2D points for bounds
    const allPts: [number, number][] = []
    for (const b of bendData) {
      allPts.push(b.start2d, b.end2d, b.mid2d)
    }
    const xs = allPts.map(p => p[0])
    const ys = allPts.map(p => p[1])
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)
    const spanX = Math.max(1, maxX - minX)
    const spanY = Math.max(1, maxY - minY)

    // Normalize to SVG space
    const norm = (p: [number, number]): [number, number] => {
      const nx = (p[0] - minX) / spanX
      const ny = (p[1] - minY) / spanY
      return [
        PAD + nx * (WIDTH - PAD * 2),
        HEIGHT - PAD - ny * (HEIGHT - PAD * 2),
      ]
    }

    const normalized = bendData.map(b => ({
      ...b,
      cx: norm(b.mid2d)[0],
      cy: norm(b.mid2d)[1],
      sx: norm(b.start2d)[0],
      sy: norm(b.start2d)[1],
      ex: norm(b.end2d)[0],
      ey: norm(b.end2d)[1],
    }))

    // Convex hull of all normalized points for part silhouette
    const normPts = allPts.map(norm)
    const hull = convexHull(normPts)
    const hullPoints = hull.map(p => `${p[0]},${p[1]}`).join(' ')

    return { normalized, hullPoints }
  }, [bendData])

  if (normalized.length === 0) return null

  return (
    <div className="rounded-lg border border-dark-600 bg-dark-800/40 px-3 py-2">
      <p className="text-[10px] text-dark-500 uppercase tracking-wide mb-1">Bend Locations (Top-Down)</p>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full h-auto"
        style={{ maxHeight: 120 }}
      >
        {/* Part silhouette */}
        {hullPoints && (
          <polygon
            points={hullPoints}
            fill="#1e293b"
            stroke="#334155"
            strokeWidth={1}
            rx={4}
          />
        )}

        {/* Bend lines (subtle) */}
        {normalized.map(b => (
          <line
            key={`line-${b.id}`}
            x1={b.sx} y1={b.sy} x2={b.ex} y2={b.ey}
            stroke={STATUS_FILL[b.status] ?? '#64748b'}
            strokeWidth={2}
            strokeOpacity={0.4}
          />
        ))}

        {/* Bend dots + labels */}
        {normalized.map(b => {
          const isFocused = focusedBendId === b.id
          const fill = STATUS_FILL[b.status] ?? '#64748b'
          return (
            <g
              key={b.id}
              className="cursor-pointer"
              onClick={() => onFocusBend?.(b.id)}
            >
              {/* Focus ring */}
              {isFocused && (
                <circle
                  cx={b.cx} cy={b.cy} r={DOT_R + 4}
                  fill="none" stroke="white" strokeWidth={1.5} opacity={0.6}
                />
              )}
              {/* Status dot */}
              <circle
                cx={b.cx} cy={b.cy} r={isFocused ? DOT_R + 1 : DOT_R}
                fill={fill}
                stroke="#0f172a" strokeWidth={1.5}
              />
              {/* Label */}
              <text
                x={b.cx} y={b.cy - DOT_R - 4}
                textAnchor="middle"
                fill="#e2e8f0"
                fontSize={10}
                fontWeight="bold"
              >
                {b.label}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

/** Simple 2D convex hull (Graham scan) */
function convexHull(points: [number, number][]): [number, number][] {
  if (points.length < 3) return points
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1])
  const cross = (o: [number, number], a: [number, number], b: [number, number]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

  const lower: [number, number][] = []
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
      lower.pop()
    lower.push(p)
  }
  const upper: [number, number][] = []
  for (const p of pts.reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
      upper.pop()
    upper.push(p)
  }
  upper.pop()
  lower.pop()
  return lower.concat(upper)
}
