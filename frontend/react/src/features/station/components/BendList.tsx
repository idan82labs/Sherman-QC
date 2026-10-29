/**
 * Per-bend list for the bender (station v2), in press-brake order.
 *
 * One row per bend: the studio bend name, one verdict chip (OK / CHECK /
 * FIX, icon + word), then flange A, flange B and angle as the signed Δ (big)
 * with the measured value next to it and a small tolerance bar under it.
 * Expected values, tolerances and reasons live in the 3D callout, which
 * opens when a row (or a tag on the part) is tapped. NOT_YET_FORMED bends
 * are one short grey "Not bent yet" line. When the bend-line position or the
 * backgauge, not a flange or the angle, is what flags a bend, one extra line
 * says so. Bends of one machine move (מהלך) carry a "Move n" tag.
 *
 * Every value is the stored result's, formatted by `cellView` (the same
 * rounding as before); nothing is re-decided here. The list scrolls inside
 * the panel (the page never does) and keeps the selected row in view.
 */
import { memo, useEffect, useRef, type ReactNode } from 'react'
import { Clock } from 'lucide-react'

import { t, verdictText } from '../copy'
import { bendVerdict, cellView, characteristicVerdict, extraConcern, toleranceBar, type BendViewRow, type CellView } from '../model/stationView'
import type { GeneralMeasurementRow, Lang } from '../model/types'
import { ToleranceBarView, VerdictChip, verdictTone } from './BendBits'

export interface BendListProps {
  lang: Lang
  rows: BendViewRow[]
  generalMeasurements: GeneralMeasurementRow[]
  selectedUid: string | null
  onSelect: (uid: string) => void
  /** inspection_validity.state !== 'VALID': one line says nothing was measured. */
  inspectionInvalid?: boolean
  /** Rendered after the bends inside the scrolling list (the zero-shot detections, never measurements). */
  footer?: ReactNode
}

function MeasureCell({ cell, testId, lang }: { cell: CellView; testId: string; lang: Lang }) {
  const c = cell.characteristic
  const verdict = characteristicVerdict(c)
  const tone = verdictTone(verdict)
  if (c.state === 'MEASURED') {
    const bar = toleranceBar(c)
    return (
      <div className={`st-mcell st-tone-${tone}`} data-testid={testId} data-state={c.state} data-decision={c.decision}>
        <div className="st-mcell-top">
          <span className="st-delta st-num" dir="ltr">{cell.deltaText}</span>
          <span className="st-actual st-num" dir="ltr">{cell.actualText}</span>
        </div>
        {bar ? <ToleranceBarView bar={bar} tone={tone} title={`${t('toleranceLong', lang)} ${cell.tolText}`} /> : <span className="st-tbar-none" />}
      </div>
    )
  }
  if (c.state === 'NOT_MEASURED') {
    return (
      <div className="st-mcell st-tone-unknown is-unmeasured" data-testid={testId} data-state={c.state} data-decision={c.decision} title={cell.style.reason}>
        <span className="st-mcell-none">{t('notMeasured', lang)}</span>
      </div>
    )
  }
  return (
    <div className="st-mcell st-tone-inactive is-na" data-testid={testId} data-state={c.state} data-decision={c.decision} title={cell.style.reason}>
      <span className="st-mcell-none">—</span>
    </div>
  )
}

export const BendList = memo(function BendList({ lang, rows, generalMeasurements, selectedUid, onSelect, inspectionInvalid = false, footer = null }: BendListProps) {
  const wrapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!selectedUid) return
    const el = wrapRef.current?.querySelector<HTMLElement>(`[data-testid="bend-row"][data-bend-uid="${selectedUid}"]`)
    el?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [selectedUid])

  return (
    <div className="st-listbox">
      <div className="st-list-wrap" ref={wrapRef}>
        <div className="st-list-head" aria-hidden>
          <span>{t('colBend', lang)}</span>
          <span>{t('colVerdict', lang)}</span>
          <span>{t('colFlangeADelta', lang)} <small dir="ltr">Δ mm</small></span>
          <span>{t('colFlangeBDelta', lang)} <small dir="ltr">Δ mm</small></span>
          <span>{t('colAngleDelta', lang)} <small dir="ltr">Δ °</small></span>
        </div>
        <ol className="st-list" data-testid="bend-list">
          {inspectionInvalid && (
            <li className="st-list-note" data-testid="table-invalid-note">{t('invalidNoBends', lang)}</li>
          )}
          {rows.map(row => {
            const selected = row.bendUid === selectedUid
            const unformed = row.state === 'NOT_YET_FORMED'
            const verdict = bendVerdict(row)
            const extra = inspectionInvalid ? null : extraConcern(row)
            const run = row.moveRun
            return (
              <li
                key={row.bendUid}
                className={`st-brow st-tone-${verdictTone(verdict)}${selected ? ' is-selected' : ''}${unformed ? ' is-unformed' : ''}`}
                data-testid="bend-row"
                data-bend-uid={row.bendUid}
                data-sequence={row.sequenceIndex}
                data-label={row.label}
                data-verdict={verdict}
                aria-selected={selected}
                aria-label={`${t('colBend', lang)} ${row.label}`}
                role="button"
                tabIndex={0}
                onClick={() => onSelect(row.bendUid)}
                onKeyDown={e => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    onSelect(row.bendUid)
                  }
                }}
              >
                <div className="st-brow-bend">
                  <span className="st-bend-label" dir="ltr">{row.label}</span>
                  {run && run.size > 1 && <span className="st-move-tag">{t('move', lang)} {row.machineMoveIndex}</span>}
                </div>
                {unformed ? (
                  <div className="st-brow-unformed" data-testid={`cell-${row.sequenceIndex}-unformed`} data-state="NOT_APPLICABLE">
                    <Clock size={22} aria-hidden />
                    <span>{verdictText('NOT_BENT', lang)}</span>
                  </div>
                ) : (
                  <>
                    <VerdictChip verdict={verdict} lang={lang} />
                    <MeasureCell cell={row.flangeA} testId={`cell-${row.sequenceIndex}-flange-a`} lang={lang} />
                    <MeasureCell cell={row.flangeB} testId={`cell-${row.sequenceIndex}-flange-b`} lang={lang} />
                    <MeasureCell cell={row.angle} testId={`cell-${row.sequenceIndex}-angle`} lang={lang} />
                  </>
                )}
                {extra && (
                  <div className={`st-brow-extra st-tone-${verdictTone(characteristicVerdict(extra.characteristic))}`} data-kind={extra.characteristic.kind}>
                    <span>{extra.characteristic.kind === 'BACKGAUGE_LENGTH' ? t('backgauge', lang) : t('alsoPosition', lang)}</span>
                    <span className="st-num" dir="ltr">Δ {extra.deltaText}{extra.characteristic.unit === 'mm' ? ' mm' : ''}</span>
                    <span className="st-num st-tol" dir="ltr">{extra.tolText}</span>
                  </div>
                )}
              </li>
            )
          })}
        </ol>
        {generalMeasurements.length > 0 && (
          <section className="st-gm" aria-label={t('generalMeasurements', lang)}>
            <h3 className="st-gm-title">{t('generalMeasurements', lang)}</h3>
            <ol className="st-list">
              {generalMeasurements.map(gm => {
                const cell = cellView(gm.characteristic, lang)
                const verdict = characteristicVerdict(gm.characteristic)
                return (
                  <li key={gm.measurement_id} className={`st-brow st-gm-row st-tone-${verdictTone(verdict)}`} data-gm-id={gm.measurement_id}>
                    <span className="st-gm-label" dir="auto">{gm.label}</span>
                    {verdict === 'NONE' ? <span /> : <VerdictChip verdict={verdict} lang={lang} />}
                    <MeasureCell cell={cell} testId={`gm-${gm.measurement_id}`} lang={lang} />
                    <span className="st-gm-exp st-num" dir="ltr">{cell.expectedText} {cell.tolText}</span>
                  </li>
                )
              })}
            </ol>
          </section>
        )}
        {footer}
      </div>
      <div className="st-list-key" aria-hidden>
        <span className="st-key-item"><span className="st-key-zone" />{t('barKeyZone', lang)}</span>
        <span className="st-key-item"><span className="st-key-mark" />{t('barKeyMark', lang)}</span>
        <span className="st-key-item"><span className="st-key-whisker" />{t('barKeyWhisker', lang)}</span>
      </div>
    </div>
  )
})
