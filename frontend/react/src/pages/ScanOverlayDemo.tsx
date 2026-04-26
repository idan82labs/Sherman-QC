import { Suspense, lazy, useMemo, useState } from 'react'
import clsx from 'clsx'
import type { BendAnnotation } from '../components/ThreeViewer'
import fixture10839000 from '../fixtures/scanOverlay/10839000.json'
import fixtureYsherman from '../fixtures/scanOverlay/ysherman.json'

const ThreeViewer = lazy(() => import('../components/ThreeViewer'))

type DemoRenderableBend = {
  bend_id: string
  label?: string
  status?: string
  detail_segments?: string[] | null
  anchor?: [number, number, number] | null
  cad_line?: {
    raw_start?: [number, number, number] | null
    raw_end?: [number, number, number] | null
    display_start?: [number, number, number] | null
    display_end?: [number, number, number] | null
  } | null
  detected_line?: {
    raw_start?: [number, number, number] | null
    raw_end?: [number, number, number] | null
    display_start?: [number, number, number] | null
    display_end?: [number, number, number] | null
  } | null
  metadata?: Record<string, unknown> | null
}

type DemoFixture = {
  partId: string
  profileFamily: string
  estimatedBendCount: number | null
  estimatedBendCountRange: number[]
  abstained: boolean
  abstainReasons: string[]
  scanUrl: string
  renderableBendObjects: DemoRenderableBend[]
}

const FIXTURES: DemoFixture[] = [fixture10839000 as unknown as DemoFixture, fixtureYsherman as unknown as DemoFixture]

function displayRenderableBends(fixture: DemoFixture): DemoRenderableBend[] {
  const bends = fixture.renderableBendObjects || []
  if (typeof fixture.estimatedBendCount !== 'number' || !Number.isFinite(fixture.estimatedBendCount)) {
    return bends
  }
  const exactCount = Math.max(0, Math.trunc(fixture.estimatedBendCount))
  if (exactCount <= 0 || bends.length <= exactCount) {
    return bends
  }
  return bends.slice(0, exactCount)
}

function point3(raw?: [number, number, number] | number[] | null): [number, number, number] | undefined {
  if (!Array.isArray(raw) || raw.length < 3) return undefined
  const x = Number(raw[0])
  const y = Number(raw[1])
  const z = Number(raw[2])
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return undefined
  return [x, y, z]
}

function midpoint(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]
}

function toViewerStatus(status?: string | null): BendAnnotation['status'] {
  switch ((status || '').toUpperCase()) {
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

function metricRows(bend: DemoRenderableBend): string[] {
  const explicit = (bend.detail_segments || []).filter((item): item is string => typeof item === 'string' && !!item.trim())
  if (explicit.length) return explicit
  const metadata = bend.metadata || {}
  const rows: string[] = []
  if (typeof metadata.strip_family === 'string' && metadata.strip_family.trim()) rows.push(metadata.strip_family.trim().replace(/_/g, ' '))
  if (typeof metadata.dominant_angle_deg === 'number' && Number.isFinite(metadata.dominant_angle_deg)) rows.push(`θ≈${metadata.dominant_angle_deg.toFixed(1)}°`)
  if (typeof metadata.along_axis_span_mm === 'number' && Number.isFinite(metadata.along_axis_span_mm)) rows.push(`span ${metadata.along_axis_span_mm.toFixed(1)}mm`)
  if (typeof metadata.observability_status === 'string' && metadata.observability_status.trim()) rows.push(metadata.observability_status.trim())
  if (typeof metadata.confidence === 'number' && Number.isFinite(metadata.confidence)) rows.push(`conf ${metadata.confidence.toFixed(2)}`)
  return rows
}

function renderableToAnnotation(bend: DemoRenderableBend, activeId: string | null): BendAnnotation | null {
  const lineStart = point3(bend.detected_line?.display_start ?? bend.detected_line?.raw_start ?? bend.cad_line?.display_start ?? bend.cad_line?.raw_start)
  const lineEnd = point3(bend.detected_line?.display_end ?? bend.detected_line?.raw_end ?? bend.cad_line?.display_end ?? bend.cad_line?.raw_end)
  const position = point3(bend.anchor) ?? (lineStart && lineEnd ? midpoint(lineStart, lineEnd) : undefined)
  if (!lineStart || !lineEnd || !position) return null
  const metadata = bend.metadata || {}
  const expectedAngle = typeof metadata.dominant_angle_deg === 'number' && Number.isFinite(metadata.dominant_angle_deg)
    ? metadata.dominant_angle_deg
    : 0
  return {
    id: bend.bend_id,
    label: bend.label || bend.bend_id,
    position,
    calloutAnchor: position,
    lineStart,
    lineEnd,
    detectedLineStart: lineStart,
    detectedLineEnd: lineEnd,
    expectedAngle,
    measuredAngle: null,
    deviation: null,
    status: toViewerStatus(bend.status),
    active: activeId === bend.bend_id,
    displayGeometryCanonical: true,
    metricRows: metricRows(bend),
  }
}

export default function ScanOverlayDemo() {
  const [fixtureId, setFixtureId] = useState<string>(FIXTURES[0].partId)
  const fixture = useMemo(() => FIXTURES.find((item) => item.partId === fixtureId) ?? FIXTURES[0], [fixtureId])
  const visibleBends = useMemo(() => displayRenderableBends(fixture), [fixture])
  const [focusedBendId, setFocusedBendId] = useState<string | null>(visibleBends[0]?.bend_id ?? null)

  const bendAnnotations = useMemo(() => (
    visibleBends
      .map((bend) => renderableToAnnotation(bend, focusedBendId))
      .filter((item): item is BendAnnotation => item !== null)
  ), [visibleBends, focusedBendId])

  const focusedAnnotation = bendAnnotations.find((annotation) => String(annotation.id) === focusedBendId) ?? bendAnnotations[0] ?? null
  const focusedRenderable = visibleBends.find((bend) => bend.bend_id === focusedBendId) ?? visibleBends[0] ?? null
  const focusPoint = focusedAnnotation?.calloutAnchor ?? focusedAnnotation?.position ?? null
  const rangeLabel = `${fixture.estimatedBendCountRange[0]}-${fixture.estimatedBendCountRange[1]}`

  return (
    <div className="min-h-screen bg-dark-950 px-6 py-6 text-dark-100">
      <div className="mx-auto max-w-[1600px] space-y-4">
        <div className="flex flex-col gap-4 rounded-2xl border border-dark-700 bg-dark-900/70 p-5 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-2">
            <p className="text-xs uppercase tracking-[0.18em] text-sky-300">Scan-Only Bend Overlay Demo</p>
            <h1 className="text-2xl font-semibold text-dark-50">Clickable bend marks and bend cards</h1>
            <p className="max-w-3xl text-sm text-dark-300">
              This page exercises the scan-only zero-shot bend object contract directly. Click a bend mark in the 3D viewer
              or a bend chip below to focus that bend and inspect its card data.
            </p>
          </div>
          <div className="flex flex-col gap-3 rounded-xl border border-dark-700 bg-dark-950/60 p-4">
            <label className="text-xs font-semibold uppercase tracking-[0.16em] text-dark-400">Fixture</label>
            <select
              className="rounded-lg border border-dark-600 bg-dark-900 px-3 py-2 text-sm text-dark-100"
              value={fixture.partId}
              onChange={(event) => {
                const next = FIXTURES.find((item) => item.partId === event.target.value) ?? FIXTURES[0]
                setFixtureId(next.partId)
                setFocusedBendId(displayRenderableBends(next)[0]?.bend_id ?? null)
              }}
            >
              {FIXTURES.map((item) => (
                <option key={item.partId} value={item.partId}>
                  {item.partId} · {item.profileFamily}
                </option>
              ))}
            </select>
            <div className="grid grid-cols-2 gap-2 text-xs text-dark-300">
              <div className="rounded-lg border border-dark-700 bg-dark-900/70 px-3 py-2">
                <div className="uppercase tracking-[0.14em] text-dark-500">Count</div>
                <div className="mt-1 text-sm font-semibold text-dark-100">{fixture.estimatedBendCount ?? 'range only'}</div>
              </div>
              <div className="rounded-lg border border-dark-700 bg-dark-900/70 px-3 py-2">
                <div className="uppercase tracking-[0.14em] text-dark-500">Range</div>
                <div className="mt-1 text-sm font-semibold text-dark-100">{rangeLabel}</div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr),360px]">
          <div className="rounded-2xl border border-dark-700 bg-dark-900/70 p-3">
            <Suspense fallback={<div className="flex h-[760px] items-center justify-center text-dark-400">Loading 3D viewer…</div>}>
              <ThreeViewer
                modelUrl={fixture.scanUrl}
                bendAnnotations={bendAnnotations}
                focusPoint={focusPoint}
                focusAnnotationId={focusedBendId}
                onFocusAnnotationId={(id) => setFocusedBendId(id == null ? null : String(id))}
                calloutMode="focused"
                className="h-[760px]"
              />
            </Suspense>
          </div>

          <div className="space-y-4">
            <div className="rounded-2xl border border-dark-700 bg-dark-900/70 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-dark-400">Bend chips</p>
              <div className="mt-3 grid grid-cols-3 gap-2">
                {visibleBends.map((bend) => {
                  const active = bend.bend_id === focusedBendId
                  return (
                    <button
                      key={bend.bend_id}
                      type="button"
                      onClick={() => setFocusedBendId(bend.bend_id)}
                      className={clsx(
                        'rounded-lg border px-3 py-2 text-left text-sm font-semibold transition-colors',
                        active
                          ? 'border-sky-400 bg-sky-500/20 text-sky-100 ring-2 ring-sky-400/30'
                          : 'border-dark-600 bg-dark-950/70 text-dark-200 hover:border-dark-500 hover:bg-dark-900',
                      )}
                    >
                      {bend.bend_id}
                    </button>
                  )
                })}
              </div>
            </div>

            <div className="rounded-2xl border border-dark-700 bg-dark-900/70 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-dark-400">Selected bend card</p>
              {focusedRenderable ? (
                <div className="mt-3 space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-xl font-semibold text-dark-50">{focusedRenderable.bend_id}</div>
                      <div className="text-sm text-dark-300">{fixture.profileFamily}</div>
                    </div>
                    <span className="rounded-full border border-amber-400/35 bg-amber-500/12 px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-amber-100">
                      {focusedRenderable.status || 'UNKNOWN'}
                    </span>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-2">
                    {metricRows(focusedRenderable).map((row) => (
                      <div key={row} className="rounded-lg border border-dark-700 bg-dark-950/70 px-3 py-2 text-sm text-dark-100">
                        {row}
                      </div>
                    ))}
                  </div>
                  <div className="rounded-lg border border-dark-700 bg-dark-950/70 px-3 py-3 text-xs text-dark-300">
                    <div className="font-semibold uppercase tracking-[0.14em] text-dark-500">Notes</div>
                    <div className="mt-2 space-y-1">
                      <div>Scan-only bend cards are descriptive, not CAD-authoritative.</div>
                      {fixture.abstained && <div>Result abstained on exact count: {fixture.abstainReasons.join(', ') || 'reason unavailable'}.</div>}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="mt-3 text-sm text-dark-400">No bend selected.</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
