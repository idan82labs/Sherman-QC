/**
 * Floating card for the selected bend, drawn over the 3D view at the end of
 * a leader line that rises from the bend's actual (measured) bend line. The
 * viewer positions it every frame (StationViewer `LabelLayoutEngine`); this
 * component only renders its content:
 *
 * - the studio bend name and one big verdict (OK / CHECK / FIX);
 * - flange A, flange B, then the angle as "expected → actual" with the
 *   signed deviation and the tolerance; a FAIL also says which way it is off
 *   (too long / too short, bent too far / not bent enough);
 * - the bend-line position and the backgauge when they need attention or exist;
 * - one sentence saying what to do. A bend that is not bent yet shows a
 *   plain "not bent yet" note and the expected values only.
 *
 * Every number is the stored result's (`cellView` formatting).
 */
import { forwardRef } from 'react'
import { ArrowRight, X } from 'lucide-react'

import { actionText, directionText, reasonText, t } from '../copy'
import { bendVerdict, characteristicVerdict, failDirection, type BendViewRow, type CellView } from '../model/stationView'
import type { Lang } from '../model/types'
import { VERDICT_ICON, VerdictChip, verdictTone } from './BendBits'

function Line({ title, cell, lang, unformed }: { title: string; cell: CellView; lang: Lang; unformed: boolean }) {
  const c = cell.characteristic
  const verdict = characteristicVerdict(c)
  const Icon = VERDICT_ICON[verdict]
  const unit = c.unit === 'mm' ? ' mm' : ''
  const dir = failDirection(c)
  return (
    <div className={`st-co-line st-tone-${verdictTone(verdict)}`} data-kind={c.kind} data-side={c.side ?? undefined} data-decision={c.decision}>
      <span className="st-co-title">{title}</span>
      {c.state === 'MEASURED' ? (
        <>
          <span className="st-co-values st-num" dir="ltr">
            <span className="st-co-expected">{cell.expectedText}</span>
            <ArrowRight size={18} aria-hidden className="st-co-arrow" />
            <span className="st-co-actual">{cell.actualText}{unit}</span>
          </span>
          <span className="st-co-delta st-num" dir="ltr">{cell.deltaText}</span>
          <span className="st-co-verdict" title={verdict}>
            <Icon size={22} strokeWidth={2.75} aria-hidden />
          </span>
          <span className="st-co-sub">
            <span className="st-num" dir="ltr">{t('toleranceLong', lang)} {cell.tolText}</span>
            {dir && <span className="st-co-dir">{directionText(dir, lang)}</span>}
          </span>
        </>
      ) : unformed || c.state === 'NOT_APPLICABLE' ? (
        <span className="st-co-values st-num" dir="ltr">
          <span className="st-co-expected">{cell.expectedText}{c.expected != null ? unit : ''}</span>
        </span>
      ) : (
        <>
          <span className="st-co-values st-co-unmeasured">
            <span className="st-num" dir="ltr">{cell.expectedText}{unit}</span>
            <ArrowRight size={18} aria-hidden className="st-co-arrow" />
            <span>{t('notMeasured', lang)}</span>
          </span>
          <span className="st-co-verdict">
            <Icon size={22} strokeWidth={2.75} aria-hidden />
          </span>
          {c.reason && <span className="st-co-sub">{reasonText(c.reason, lang)}</span>}
        </>
      )}
    </div>
  )
}

function needsLine(cell: CellView | null): boolean {
  if (!cell) return false
  const c = cell.characteristic
  if (c.state === 'NOT_APPLICABLE') return false
  return c.kind === 'BACKGAUGE_LENGTH' || c.decision === 'FAIL' || c.decision === 'REVIEW' || (c.required && c.decision === 'UNKNOWN')
}

export interface BendCalloutProps {
  row: BendViewRow
  lang: Lang
  onClose: () => void
}

export const BendCallout = forwardRef<HTMLDivElement, BendCalloutProps>(function BendCallout({ row, lang, onClose }, ref) {
  const verdict = bendVerdict(row)
  const unformed = row.state === 'NOT_YET_FORMED'
  const odd = row.state !== 'FORMED' && !unformed
  return (
    <div
      ref={ref}
      className={`st-callout st-tone-${verdictTone(verdict)}${unformed ? ' is-unformed' : ''}`}
      data-testid="bend-callout"
      data-bend-uid={row.bendUid}
      data-label={row.label}
      data-verdict={verdict}
      data-viewer-reserved
      data-callout
      style={{ visibility: 'hidden' }}
      onPointerDown={e => e.stopPropagation()}
      role="dialog"
      aria-label={`${t('colBend', lang)} ${row.label}`}
    >
      <div className="st-co-head">
        <span className="st-co-bend" dir="ltr">{row.label}</span>
        {row.moveRun && row.moveRun.size > 1 && <span className="st-move-tag">{t('move', lang)} {row.machineMoveIndex}</span>}
        <VerdictChip verdict={verdict} lang={lang} size="lg" />
        <button type="button" className="st-co-close" onClick={onClose} aria-label={t('closeCallout', lang)}>
          <X size={24} aria-hidden />
        </button>
      </div>
      {unformed && <p className="st-co-note">{actionText('NOT_BENT', lang)}</p>}
      {odd && (
        <p className="st-co-note st-co-state">
          {row.stateLabel}
          {row.bend.state_reasons.length > 0 && ` – ${row.bend.state_reasons.map(r => reasonText(r, lang)).join(', ')}`}
        </p>
      )}
      <div className="st-co-lines">
        <div className="st-co-colhead" aria-hidden>
          <span />
          {/* LTR like the values under it: expected on the left, actual on the right, in both languages. */}
          <span dir="ltr">
            {unformed ? t('expectedLong', lang) : <><bdi>{t('expectedLong', lang)}</bdi> → <bdi>{t('actualLong', lang)}</bdi></>}
          </span>
          {!unformed && <span>{t('deviationLong', lang)}</span>}
        </div>
        <Line title={t('colFlangeA', lang)} cell={row.flangeA} lang={lang} unformed={unformed} />
        <Line title={t('colFlangeB', lang)} cell={row.flangeB} lang={lang} unformed={unformed} />
        <Line title={t('colAngle', lang)} cell={row.angle} lang={lang} unformed={unformed} />
        {!unformed && needsLine(row.position) && <Line title={t('bendLine', lang)} cell={row.position} lang={lang} unformed={false} />}
        {!unformed && row.backgauge && needsLine(row.backgauge) && <Line title={t('backgauge', lang)} cell={row.backgauge} lang={lang} unformed={false} />}
      </div>
      {!unformed && <p className={`st-co-action st-tone-${verdictTone(verdict)}`}>{actionText(verdict, lang)}</p>}
    </div>
  )
})
