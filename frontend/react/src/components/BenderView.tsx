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

const STATUS_COLORS: Record<string, string> = {
  PASS: 'text-emerald-400',
  FAIL: 'text-red-400',
  WARNING: 'text-amber-400',
  NOT_DETECTED: 'text-dark-500',
}

const STATUS_SHORT: Record<string, string> = {
  PASS: 'PASS',
  FAIL: 'FAIL',
  WARNING: 'WARN',
  NOT_DETECTED: 'N/D',
}

export default function BenderView({ artifacts, matches, focusedBendId, onFocusBend }: BenderViewProps) {
  const [expandedImage, setExpandedImage] = useState<string | null>(null)

  const views = [
    { label: 'Front', url: artifacts?.bender_view_front_url },
    { label: 'Profile', url: artifacts?.bender_view_profile_url },
    { label: 'Top', url: artifacts?.bender_view_top_url },
    { label: 'Overall', url: artifacts?.bender_view_overall_url },
  ]

  const hasViews = views.some(v => !!v.url)

  if (!hasViews) {
    return (
      <div className="text-center text-dark-400 py-12 text-sm">
        Bender's View snapshots are not available for this inspection.
        <br />
        <span className="text-dark-500 text-xs">Run a new inspection to generate them.</span>
      </div>
    )
  }

  return (
    <div className="bender-view-container">
      {/* Toolbar */}
      <div className="flex items-center justify-between mb-3 print:hidden">
        <h3 className="text-sm font-semibold text-dark-100 uppercase tracking-wide">
          Bender's View
        </h3>
        <div className="flex gap-2">
          <button
            onClick={() => window.print()}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-dark-700 text-dark-300 hover:bg-dark-600 transition-colors"
          >
            <Printer className="w-3.5 h-3.5" />
            Print
          </button>
        </div>
      </div>

      {/* Main layout: images + sidebar */}
      <div className="flex gap-4">
        {/* 2x2 image grid */}
        <div className="flex-1 grid grid-cols-2 gap-2">
          {views.map((v) => (
            <div key={v.label} className="rounded-lg border border-dark-600 bg-dark-900/60 overflow-hidden">
              <div className="px-2 py-1 text-[11px] text-dark-400 font-medium border-b border-dark-700">
                {v.label}
              </div>
              {v.url ? (
                <img
                  src={v.url}
                  alt={`${v.label} view`}
                  className="w-full h-auto cursor-pointer hover:opacity-90 transition-opacity"
                  loading="lazy"
                  onClick={() => setExpandedImage(v.url!)}
                />
              ) : (
                <div className="aspect-[4/3] flex items-center justify-center text-dark-500 text-xs">
                  Not available
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Right sidebar: per-bend metrics table */}
        <div className="w-[320px] shrink-0">
          <div className="rounded-lg border border-dark-600 bg-dark-800/40 overflow-hidden">
            <div className="px-3 py-2 border-b border-dark-700">
              <p className="text-xs text-dark-400 font-medium uppercase tracking-wide">
                Dimensions (Exp / Act)
              </p>
            </div>
            <div className="overflow-y-auto max-h-[500px]">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-dark-800">
                  <tr className="text-left text-dark-500 border-b border-dark-700">
                    <th className="px-2 py-1.5">#</th>
                    <th className="px-2 py-1.5">Angle (°)</th>
                    <th className="px-2 py-1.5">Edge (mm)</th>
                    <th className="px-2 py-1.5">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {matches.map((m) => {
                    const isFocused = focusedBendId === m.bend_id
                    return (
                      <tr
                        key={m.bend_id}
                        className={clsx(
                          'border-b border-dark-700/50 cursor-pointer hover:bg-dark-700/30 transition-colors',
                          isFocused && 'bg-primary-500/10',
                        )}
                        onClick={() => onFocusBend?.(m.bend_id)}
                      >
                        <td className="px-2 py-1.5 font-medium text-dark-200">
                          {m.bend_id}
                        </td>
                        <td className="px-2 py-1.5">
                          <span className="text-dark-300">{m.target_angle.toFixed(1)}</span>
                          <span className="text-dark-500"> / </span>
                          <span className="text-dark-200">{m.measured_angle?.toFixed(1) ?? '-'}</span>
                          {m.angle_deviation != null && (
                            <div className={clsx(
                              'text-[10px]',
                              Math.abs(m.angle_deviation) <= (m.tolerance_angle ?? 1)
                                ? 'text-emerald-400'
                                : 'text-red-400',
                            )}>
                              {m.angle_deviation >= 0 ? '+' : ''}{m.angle_deviation.toFixed(2)}°
                            </div>
                          )}
                        </td>
                        <td className="px-2 py-1.5">
                          <span className="text-dark-300">
                            {m.expected_edge_distance_mm?.toFixed(1) ?? '-'}
                          </span>
                          <span className="text-dark-500"> / </span>
                          <span className="text-dark-200">
                            {m.measured_edge_distance_mm?.toFixed(1) ?? '-'}
                          </span>
                          {m.edge_distance_deviation_mm != null && (
                            <div className="text-[10px] text-dark-400">
                              {m.edge_distance_deviation_mm >= 0 ? '+' : ''}
                              {m.edge_distance_deviation_mm.toFixed(2)}mm
                            </div>
                          )}
                        </td>
                        <td className="px-2 py-1.5">
                          <span className={clsx('font-semibold', STATUS_COLORS[m.status])}>
                            {STATUS_SHORT[m.status] ?? m.status}
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Legend */}
          <div className="mt-3 flex flex-wrap gap-3 text-[11px] text-dark-400">
            <LegendItem color="bg-emerald-500" label="Pass" />
            <LegendItem color="bg-amber-500" label="Warning" />
            <LegendItem color="bg-red-500" label="Fail" />
            <LegendItem color="bg-slate-400" dashed label="Not Detected" />
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

function LegendItem({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      <span
        className={clsx('w-3 h-0.5 rounded-full', color, dashed && 'border border-dashed border-slate-400 bg-transparent')}
      />
      <span>{label}</span>
    </div>
  )
}
