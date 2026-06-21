import clsx from 'clsx'

import type { BendAnnotation, ScreenBendCallout } from '../bend-annotations/types'

export function BendCalloutOverlay({ callouts }: { callouts: ScreenBendCallout[] }) {
  const statusCardClasses: Record<BendAnnotation['status'], string> = {
    pass: 'border-emerald-300/85 bg-emerald-900/92 text-emerald-50',
    fail: 'border-red-300/90 bg-red-950/92 text-red-50',
    warning: 'border-amber-300/90 bg-amber-950/92 text-amber-50',
    pending: 'border-slate-300/70 bg-slate-900/92 text-slate-100',
  }
  const statusStroke: Record<BendAnnotation['status'], string> = {
    pass: '#34d399',
    fail: '#f87171',
    warning: '#fbbf24',
    pending: '#94a3b8',
  }

  return (
    <div className="pointer-events-none absolute inset-0 z-10 overflow-hidden">
      <svg className="absolute inset-0 h-full w-full" aria-hidden="true">
        {callouts.map((callout) => {
          const exitX = callout.side === 'left' ? callout.boxX + callout.width : callout.boxX
          const exitY = callout.boxY + callout.height / 2
          const elbowX = callout.side === 'left' ? exitX + 24 : exitX - 24
          const path = `M ${callout.anchorX.toFixed(1)} ${callout.anchorY.toFixed(1)} L ${elbowX.toFixed(1)} ${callout.anchorY.toFixed(1)} L ${elbowX.toFixed(1)} ${exitY.toFixed(1)} L ${exitX.toFixed(1)} ${exitY.toFixed(1)}`
          return (
            <path
              key={`${callout.id}-connector`}
              d={path}
              fill="none"
              stroke={statusStroke[callout.status]}
              strokeOpacity={callout.active ? 0.9 : 0.68}
              strokeWidth={callout.active ? 2.3 : 1.5}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )
        })}
      </svg>

      {callouts.map((callout) => (
        <div
          key={callout.id}
          className={clsx(
            'absolute rounded-xl border shadow-2xl',
            callout.active ? 'ring-2 ring-white/30' : 'ring-1 ring-white/10',
            statusCardClasses[callout.status],
          )}
          style={{
            left: `${callout.boxX}px`,
            top: `${callout.boxY}px`,
            width: `${callout.width}px`,
            minHeight: `${callout.height}px`,
          }}
        >
          <div className="px-3 py-2">
            <div className="flex items-center justify-between gap-2">
              <div className="text-[11px] font-semibold tracking-[0.12em] uppercase text-white/95">
                {callout.label}
              </div>
              {callout.active && (
                <span className="rounded-full border border-white/20 bg-white/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.14em] text-white/80">
                  Focus
                </span>
              )}
            </div>
            <div className="mt-1 space-y-0.5 text-[11px] font-mono leading-tight text-white/90">
              {callout.metricRows.map((row) => (
                <div key={`${callout.id}-${row}`}>{row}</div>
              ))}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

export default BendCalloutOverlay
