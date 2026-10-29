import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { Printer, Box, BarChart3 } from 'lucide-react'
import type { BendMatch } from '../services/api'
import BendMapDiagram from './BendMapDiagram'
import { AuthenticatedArtifactImage } from './AuthenticatedArtifact'
import { decisionLabel } from '../features/bend-inspection/model/decisionContract'

interface BenderViewProps {
  artifacts?: {
    bender_view_front_url?: string
    bender_view_profile_url?: string
    bender_view_top_url?: string
    bender_view_overall_url?: string
    reference_mesh_url?: string
    bend_overlay_overview_url?: string
  }
  matches: BendMatch[]
  jobId?: string
  focusedBendId?: string | null
  onFocusBend?: (bendId: string) => void
  scannedAt?: string | null
  observedFormedRatio?: string | null
}

export default function BenderView({ artifacts, matches, jobId, focusedBendId, onFocusBend, scannedAt, observedFormedRatio }: BenderViewProps) {
  const [expandedImage, setExpandedImage] = useState<string | null>(null)

  // Keyboard navigation: arrows cycle bends, Esc closes lightbox
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setExpandedImage(null)
      if ((e.key === 'ArrowRight' || e.key === 'ArrowLeft') && matches.length > 0) {
        const idx = matches.findIndex(m => m.bend_id === focusedBendId)
        const next = e.key === 'ArrowRight'
          ? (idx + 1) % matches.length
          : (idx - 1 + matches.length) % matches.length
        onFocusBend?.(matches[next].bend_id)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [focusedBendId, matches, onFocusBend])

  const views = [
    { label: 'Overall', url: artifacts?.bender_view_overall_url },
    { label: 'Front', url: artifacts?.bender_view_front_url },
    { label: 'Profile', url: artifacts?.bender_view_profile_url },
    { label: 'Top', url: artifacts?.bender_view_top_url },
  ]

  const hasViews = views.some(v => !!v.url)

  return (
    <div className="bender-view-container space-y-4">
      {/* Verdict Banner */}
      <div role="note" className="rounded-lg border border-dark-500 bg-dark-800 px-5 py-3">
        <p className="text-dark-100 font-bold text-lg">Research evidence only — not qualified for production accept/reject</p>
        <p className="text-dark-200 text-sm mt-1">Registration and render evidence are unverified. Operator review remains required; no machine adjustment is authorized.</p>
        <p className="text-dark-300 text-xs mt-1">
          {observedFormedRatio ? `${observedFormedRatio} observed formed` : 'Observed count unavailable'}
          {scannedAt && ` · ${new Date(scannedAt).toLocaleString()}`}
        </p>
      </div>

      {/* Action bar */}
      <div className="flex items-center gap-2 print:hidden">
        {jobId && (
          <a
            href={`/bend-inspection`}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-dark-600 bg-dark-800 text-dark-300 hover:bg-dark-700 transition-colors"
          >
            <Box className="w-3.5 h-3.5" />
            3D View
          </a>
        )}
        {artifacts?.bend_overlay_overview_url && (
          <button
            onClick={() => setExpandedImage(artifacts.bend_overlay_overview_url!)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-dark-600 bg-dark-800 text-dark-300 hover:bg-dark-700 transition-colors"
          >
            <BarChart3 className="w-3.5 h-3.5" />
            Heatmap
          </button>
        )}
        <button
          onClick={() => window.print()}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-dark-600 bg-dark-800 text-dark-300 hover:bg-dark-700 transition-colors"
        >
          <Printer className="w-3.5 h-3.5" />
          Print
        </button>
      </div>

      {/* Bend Progress Dots */}
      <div className="flex items-center gap-2 px-1">
        {matches.map(m => (
          <button
            key={m.bend_id}
            onClick={() => onFocusBend?.(m.bend_id)}
            className={clsx(
              'flex items-center gap-1.5 px-2 py-1 rounded-full text-[11px] font-medium transition-colors border',
              focusedBendId === m.bend_id
                ? 'border-white/40 bg-dark-700'
                : 'border-transparent hover:bg-dark-800',
            )}
            title={`${m.bend_id}: ${decisionLabel(m)}`}
          >
            <span className="w-2.5 h-2.5 rounded-full bg-slate-500" />
            <span className="text-dark-300">{m.bend_id.replace('CAD_', '')}</span>
          </button>
        ))}
      </div>

      {/* Bend-map diagram — shows physical bend locations */}
      <BendMapDiagram
        matches={matches.map((match) => ({ ...match, status: 'NOT_DETECTED' }))}
        focusedBendId={focusedBendId}
        onFocusBend={onFocusBend}
      />

      {/* Main layout: snapshots + table */}
      <div className="flex flex-col gap-4 lg:flex-row">
        {/* Snapshots — Overall as hero + 3 thumbnails */}
        <div className="flex-1 space-y-2">
          {hasViews ? (
            <>
              {/* Hero: Overall */}
              {views[0].url && (
                <div className="rounded-lg border border-dark-600 bg-dark-900/60 overflow-hidden">
                  <div className="px-2 py-1 text-[11px] text-dark-400 font-medium border-b border-dark-700">
                    {views[0].label}
                  </div>
                  <AuthenticatedArtifactImage
                    src={views[0].url}
                    alt={views[0].label}
                    className="w-full h-auto cursor-pointer hover:opacity-90 transition-opacity"
                    onClick={() => setExpandedImage(views[0].url!)}
                  />
                </div>
              )}
              {/* Secondary: Front, Profile, Top */}
              <div className="grid grid-cols-3 gap-2">
                {views.slice(1).map(v => (
                  <div key={v.label} className="rounded-lg border border-dark-600 bg-dark-900/60 overflow-hidden">
                    <div className="px-2 py-1 text-[10px] text-dark-500 font-medium border-b border-dark-700">
                      {v.label}
                    </div>
                    {v.url ? (
                      <AuthenticatedArtifactImage
                        src={v.url}
                        alt={v.label}
                        className="w-full h-auto cursor-pointer hover:opacity-90 transition-opacity"
                        onClick={() => setExpandedImage(v.url!)}
                      />
                    ) : (
                      <div className="aspect-[4/3] flex items-center justify-center text-dark-600 text-[10px]">
                        N/A
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="text-center text-dark-400 py-12 text-sm">
              Snapshots not available. Run a new inspection to generate them.
            </div>
          )}
        </div>

        {/* Right sidebar: metrics table */}
        <div className="w-full shrink-0 space-y-3 lg:w-[360px]">
          <div className="rounded-lg border border-dark-600 bg-dark-800/40 overflow-hidden">
            <div className="px-3 py-2 border-b border-dark-700">
              <p className="text-xs text-dark-400 font-medium uppercase tracking-wide">
                Bend Measurements
              </p>
            </div>
            <div className="overflow-y-auto max-h-[600px] divide-y divide-dark-700/50">
              {matches.map(m => {
                const isFocused = focusedBendId === m.bend_id
                return (
                  <div
                    key={m.bend_id}
                    className={clsx(
                      'px-3 py-2.5 cursor-pointer hover:bg-dark-700/30 transition-colors',
                      isFocused && 'bg-primary-500/10',
                    )}
                    onClick={() => onFocusBend?.(m.bend_id)}
                  >
                    {/* Row header: bend ID + status */}
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full bg-slate-500" />
                        <span className="text-dark-100 font-semibold text-sm">{m.bend_id}</span>
                      </div>
                      <span className="text-xs font-medium text-dark-200">
                        {decisionLabel(m)}
                      </span>
                    </div>

                      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                        {/* Angle */}
                        <div>
                          <span className="text-dark-500">Angle</span>
                          <div className="flex items-baseline gap-1.5">
                            <span className="text-dark-100 font-medium">
                              {m.measured_angle?.toFixed(1) ?? '-'}°
                            </span>
                            {m.angle_deviation != null && (
                              <span className="text-[11px] font-semibold text-dark-200">
                                {m.angle_deviation >= 0 ? '+' : ''}{m.angle_deviation.toFixed(1)}°
                              </span>
                            )}
                          </div>
                          <span className="text-dark-600 text-[10px]">
                            target {m.target_angle.toFixed(1)}° ± {m.tolerance_angle ?? 1}°
                          </span>
                        </div>

                        {/* Edge distance */}
                        <div>
                          <span className="text-dark-500">Edge dist</span>
                          <div className="flex items-baseline gap-1.5">
                            <span className="text-dark-100 font-medium">
                              {m.measured_edge_distance_mm?.toFixed(1) ?? '-'} mm
                            </span>
                            {m.edge_distance_deviation_mm != null && (
                              <span className="text-[11px] font-semibold text-dark-200">
                                {m.edge_distance_deviation_mm >= 0 ? '+' : ''}
                                {m.edge_distance_deviation_mm.toFixed(1)}
                              </span>
                            )}
                          </div>
                          {m.expected_edge_distance_mm != null && (
                            <span className="text-dark-600 text-[10px]">
                              target {m.expected_edge_distance_mm.toFixed(1)} mm
                            </span>
                          )}
                        </div>

                        <p className="col-span-2 mt-2 text-dark-300">Experimental measurements; tolerance disposition remains unverified.</p>
                      </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Expanded image lightbox */}
      {expandedImage && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center cursor-pointer print:hidden"
          onClick={() => setExpandedImage(null)}
        >
          <AuthenticatedArtifactImage
            src={expandedImage}
            alt="Expanded view"
            className="max-w-[90vw] max-h-[90vh] object-contain rounded-lg shadow-2xl"
          />
        </div>
      )}
    </div>
  )
}
