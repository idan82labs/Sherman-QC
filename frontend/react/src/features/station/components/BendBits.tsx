/**
 * Small shared pieces of the bender view: the verdict chip (icon + word,
 * never colour alone) and the tolerance bar (where the stored deviation sits
 * within ±tolerance, with its stored uncertainty as a whisker).
 */
import { AlertTriangle, Check, Clock, HelpCircle, Minus, RotateCcw, X, type LucideIcon } from 'lucide-react'

import { verdictText } from '../copy'
import type { BendVerdict, PartVerdict, ToleranceBar } from '../model/stationView'
import type { Lang } from '../model/types'

type AnyVerdict = BendVerdict | PartVerdict

export const VERDICT_ICON: Record<AnyVerdict, LucideIcon> = {
  OK: Check,
  OK_SO_FAR: Check,
  CHECK: AlertTriangle,
  FIX: X,
  UNSURE: HelpCircle,
  RESCAN: RotateCcw,
  NOT_BENT: Clock,
  NONE: Minus,
}

/** Tone class of a verdict: the three status colours, info blue for "OK so far", grey otherwise. */
export function verdictTone(v: AnyVerdict): 'pass' | 'review' | 'fail' | 'info' | 'unknown' | 'inactive' {
  switch (v) {
    case 'OK': return 'pass'
    case 'OK_SO_FAR': return 'info'
    case 'CHECK': return 'review'
    case 'FIX': return 'fail'
    case 'UNSURE':
    case 'RESCAN': return 'unknown'
    default: return 'inactive'
  }
}

export function VerdictChip({ verdict, lang, size = 'md' }: { verdict: AnyVerdict; lang: Lang; size?: 'md' | 'lg' }) {
  const Icon = VERDICT_ICON[verdict]
  return (
    <span className={`st-chip st-chip-${size} st-tone-${verdictTone(verdict)}`} data-verdict={verdict}>
      <Icon size={size === 'lg' ? 30 : 22} strokeWidth={2.75} aria-hidden />
      <span className="st-chip-word">{verdictText(verdict, lang)}</span>
    </span>
  )
}

/** Left to right: minus side, tolerance zone, zero, plus side (always LTR, like the numbers). */
export function ToleranceBarView({ bar, tone, title }: { bar: ToleranceBar; tone: string; title?: string }) {
  return (
    <span className={`st-tbar st-tone-${tone}`} dir="ltr" title={title} aria-hidden data-over={bar.over || undefined}>
      <span className="st-tbar-zone" style={{ left: `${bar.zoneFrom}%`, width: `${bar.zoneTo - bar.zoneFrom}%` }} />
      <span className="st-tbar-zero" style={{ left: `${bar.zero}%` }} />
      {bar.hi - bar.lo > 0.5 && <span className="st-tbar-whisker" style={{ left: `${bar.lo}%`, width: `${bar.hi - bar.lo}%` }} />}
      <span className="st-tbar-mark" style={{ left: `${bar.at}%` }} />
    </span>
  )
}
