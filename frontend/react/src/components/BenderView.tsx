import { useState } from 'react'
import clsx from 'clsx'
import { Printer } from 'lucide-react'
import type { BendMatch } from '../services/api'

interface BenderViewProps {
  artifacts?: {
    bender_view_front_url?: string
    bender_view_profile_url?: string
    bender_view_top_url?: string
    bender_view_overall_url?: string
  }
  matches: BendMatch[]
  focusedBendId?: string | null
  onFocusBend?: (bendId: string) => void
}

const STATUS_BG: Record<string, string> = {
  PASS: 'bg-emerald-500',
  FAIL: 'bg-red-500',
  WARNING: 'bg-amber-500',
  NOT_DETECTED: 'bg-dark-600',
}

const STATUS_TEXT: Record<string, string> = {
  PASS: 'text-emerald-400',
  FAIL: 'text-red-400',
  WARNING: 'text-amber-400',
  NOT_DETECTED: 'text-dark-500',
}

export default function BenderView({ artifacts, matches, focusedBendId, onFocusBend }: BenderViewProps) {
  const [expandedImage, setExpandedImage] = useState<string | null>(null)

  const views = [
    { label: 'Overall', url: artifacts?.bender_view_overall_url },
    { label: 'Front', url: artifacts?.bender_view_front_url },
    { label: 'Profile', url: artifacts?.bender_view_profile_url },
    { label: 'Top', url: artifacts?.bender_view_top_url },
  ]

  const hasViews = views.some(v => !!v.url)

  // Compute verdict
  const scanned = matches.filter(m => m.status !== 'NOT_DETECTED')
  const failed = matches.filter(m => m.status === 'FAIL')
  const warned = matches.filter(m => m.status === 'WARNING')
  const notDetected = matches.filter(m => m.status === 'NOT_DETECTED')
  const allScanned = notDetected.length === 0
  let verdictBg = 'bg-emerald-600'
  let verdictText = `PASS — all ${matches.length} bends within tolerance`
  if (!allScanned) {
    verdictBg = 'bg-amber-600'
    verdictText = `INCOMPLETE — ${scanned.length}/${matches.length} bends scanned`
    if (failed.length > 0) {
      verdictBg = 'bg-red-600'
      verdictText += `, ${failed.length} out of spec`
    }
  } else if (failed.length > 0) {
    verdictBg = 'bg-red-600'
    const failIds = failed.map(m => m.bend_id.replace('CAD_', '')).join(', ')
    verdictText = `FAIL — ${failed.length} bend${failed.length > 1 ? 's' : ''} out of tolerance (${failIds})`
  } else if (warned.length > 0) {
    verdictBg = 'bg-amber-600'
    verdictText = `WARNING — ${warned.length} bend${warned.length > 1 ? 's' : ''} near tolerance limit`
  }

  return (
    <div className="bender-view-container space-y-4">
      {/* Verdict Banner */}
      <div className={clsx('rounded-lg px-5 py-3 flex items-center justify-between', verdictBg)}>
        <div>
          <p className="text-white font-bold text-lg">{verdictText}</p>
          <p className="text-white/70 text-xs mt-0.5">
            {scanned.length}/{matches.length} bends scanned
          </p>
        </div>
        <button
          onClick={() => window.print()}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-white/20 text-white hover:bg-white/30 transition-colors print:hidden"
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
            title={`${m.bend_id}: ${m.status}`}
          >
            <span className={clsx('w-2.5 h-2.5 rounded-full', STATUS_BG[m.status])} />
            <span className="text-dark-300">{m.bend_id.replace('CAD_', '')}</span>
          </button>
        ))}
      </div>

      {/* Main layout: snapshots + table */}
      <div className="flex gap-4">
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
                  <img
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
                      <img
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
        <div className="w-[360px] shrink-0 space-y-3">
          <div className="rounded-lg border border-dark-600 bg-dark-800/40 overflow-hidden">
            <div className="px-3 py-2 border-b border-dark-700">
              <p className="text-xs text-dark-400 font-medium uppercase tracking-wide">
                Bend Measurements
              </p>
            </div>
            <div className="overflow-y-auto max-h-[600px] divide-y divide-dark-700/50">
              {matches.map(m => {
                const isFocused = focusedBendId === m.bend_id
                const isIssue = m.status === 'FAIL' || m.status === 'WARNING'
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
                        <span className={clsx('w-2.5 h-2.5 rounded-full', STATUS_BG[m.status])} />
                        <span className="text-dark-100 font-semibold text-sm">{m.bend_id}</span>
                      </div>
                      <span className={clsx('text-xs font-bold', STATUS_TEXT[m.status])}>
                        {m.status === 'NOT_DETECTED' ? 'NOT SCANNED' : m.status}
                      </span>
                    </div>

                    {m.status !== 'NOT_DETECTED' ? (
                      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                        {/* Angle */}
                        <div>
                          <span className="text-dark-500">Angle</span>
                          <div className="flex items-baseline gap-1.5">
                            <span className="text-dark-100 font-medium">
                              {m.measured_angle?.toFixed(1) ?? '-'}°
                            </span>
                            {m.angle_deviation != null && (
                              <span className={clsx(
                                'text-[11px] font-semibold',
                                Math.abs(m.angle_deviation) <= (m.tolerance_angle ?? 1)
                                  ? 'text-emerald-400' : 'text-red-400',
                              )}>
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
                              <span className={clsx(
                                'text-[11px] font-semibold',
                                Math.abs(m.edge_distance_deviation_mm) <= 1.0
                                  ? 'text-emerald-400' : 'text-amber-400',
                              )}>
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

                        {/* Correction hint for failed bends */}
                        {isIssue && m.angle_deviation != null && (
                          <div className="col-span-2 mt-1.5 text-xs text-amber-200 bg-amber-500/15 border border-amber-500/20 rounded px-2.5 py-1.5 flex items-start gap-1.5">
                            <span className="text-amber-400 mt-px">⚠</span>
                            <div>
                              {m.angle_deviation > 0
                                ? `Over-bent by ${Math.abs(m.angle_deviation).toFixed(1)}° — reduce bend angle`
                                : `Under-bent by ${Math.abs(m.angle_deviation).toFixed(1)}° — increase bend angle`}
                              {m.edge_distance_deviation_mm != null && Math.abs(m.edge_distance_deviation_mm) > 1.0 && (
                                <span className="block mt-0.5 text-amber-300/70">
                                  Edge gap off by {m.edge_distance_deviation_mm > 0 ? '+' : ''}{m.edge_distance_deviation_mm.toFixed(1)}mm
                                  {' — '}adjust back stop accordingly
                                </span>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <p className="text-dark-500 text-xs">
                        Not captured in scan — reposition part and re-scan
                      </p>
                    )}
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
          <img
            src={expandedImage}
            alt="Expanded view"
            className="max-w-[90vw] max-h-[90vh] object-contain rounded-lg shadow-2xl"
          />
        </div>
      )}
    </div>
  )
}
