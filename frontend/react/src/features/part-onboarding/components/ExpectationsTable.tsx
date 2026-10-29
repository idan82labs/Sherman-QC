/**
 * Nominal expectations from the E1 CompiledPart, per bend in sequence order:
 * compiled included angle (and the recipe angle when they disagree), OML flange
 * length per side with its far-boundary kind, bend-line and backgauge
 * tolerances. Sides with no nominal are highlighted with the fix advice; a
 * NOT_INSPECTED side shows as such, never as a number.
 */
import { fill, type OnbCopy, type OnbLanguage } from '../copy'
import { formatBand, formatNominal, type BendExpectation, type ExpectationsView, type SideExpectation } from '../model/expectations'
import { Notice, ReasonRow } from './common'

function SideCell({ side, t }: { side: SideExpectation; t: OnbCopy }) {
  if (side.requirement === 'NOT_INSPECTED') {
    return <span className="onb-cell-na" data-side-state="not-inspected">{t.notInspectedShort}
      {side.nominal != null && <span className="onb-sub onb-num" dir="ltr">{formatNominal(side.nominal, 'mm')}</span>}</span>
  }
  if (side.unavailable) {
    return <span className="onb-cell-unavailable" data-side-state="unavailable" data-unavailable-reason={side.unavailable}>
      {side.unavailable === 'UNSUPPORTED_FEATURE' ? t.unsupported : t.unavailableNominal}
      <span className="onb-sub" style={{ color: 'var(--onb-ink)', fontWeight: 400 }}>{t.unavailableAdvice}</span></span>
  }
  const boundary = side.boundaryKind === 'ADJACENT_BEND' ? fill(t.boundaryADJACENT_BEND, { b: side.adjacentLabel ?? '?' })
    : side.boundaryKind === 'FREE_EDGE' ? t.boundaryFREE_EDGE : ''
  const band = formatBand(side.characteristic?.tolerance, 'mm')
  return <span data-side-state="nominal"><strong className="onb-num" dir="ltr">{formatNominal(side.nominal, 'mm')}</strong>
    <span className="onb-sub">{boundary}</span>
    <span className="onb-sub onb-num" dir="ltr">{band ?? t.noTolerance}</span></span>
}

function AngleCell({ bend, t }: { bend: BendExpectation; t: OnbCopy }) {
  const band = formatBand(bend.included?.tolerance, 'deg')
  if (bend.includedUnavailable) {
    return <span className="onb-cell-unavailable" data-angle-state="unavailable" data-unavailable-reason={bend.includedUnavailable}>
      {bend.includedUnavailable === 'UNSUPPORTED_FEATURE' ? t.unsupported : t.unavailableNominal}
      <span className="onb-sub" style={{ color: 'var(--onb-ink)', fontWeight: 400 }}>{t.unavailableAdvice}</span></span>
  }
  return <span data-angle-disagrees={bend.angleDisagrees} data-angle-state="nominal">
    <strong className="onb-num" dir="ltr">{formatNominal(bend.includedDeg, 'deg') ?? '—'}</strong>
    <span className="onb-sub onb-num" dir="ltr">{band ?? t.noTolerance}</span>
    {bend.angleDisagrees && <span className="onb-sub" style={{ color: 'var(--onb-amber)' }}>
      {t.compiledAngle} <bdi dir="ltr">{formatNominal(bend.includedDeg, 'deg')}</bdi>, {t.recipeAngle} <bdi dir="ltr">{formatNominal(bend.recipeIncludedDeg, 'deg') ?? '—'}</bdi></span>}
  </span>
}

export default function ExpectationsTable({ view, t, language, cadLabels }: {
  view: ExpectationsView; t: OnbCopy; language: OnbLanguage
  /**
   * The CAD mark label per sequence index (Bends & order), when the recipe was made from
   * the current setup. The recipe numbers bends by sequence (B1..Bn), so the CAD label is
   * shown beside it whenever the two differ (studio "B4" = recipe "B2").
   */
  cadLabels?: Map<number, string> | null
}) {
  return (
    <div className="grid gap-4" data-expectations data-compile-status={view.status}>
      {view.blockers.length > 0 && <section className="grid gap-2" data-blockers>
        <h3 className="onb-h3">{t.blockersTitle}</h3>
        {view.blockers.map((issue, index) => <ReasonRow key={`${issue.code}-${index}`} code={issue.code} language={language} detail={issueDetail(issue, view)} />)}
      </section>}
      {view.unmarkedGraphBends.length > 0 && <Notice tone="blocked"><span className="grid gap-1" data-unmarked>
        <strong>{t.unmarkedTitle}</strong><span>{t.unmarkedLead}</span>
        <bdi dir="ltr" className="onb-num">{view.unmarkedGraphBends.map(g => `${g.graph_bend_id} (${(180 - g.sweep_deg).toFixed(1)}°)`).join(', ')}</bdi></span></Notice>}

      <dl className="flex flex-wrap gap-x-8 gap-y-2">
        {view.thickness && <div><dt className="onb-help">{t.thicknessUsed}</dt><dd className="font-semibold onb-num" dir="ltr">{view.thickness.t_mm.toFixed(3)} mm</dd></div>}
        {view.kFactor && <div><dt className="onb-help">{t.kFactor}</dt><dd className="font-semibold"><bdi dir="ltr" className="onb-num">{view.kFactor.value.toFixed(3)} ± {view.kFactor.uncertainty.toFixed(3)}</bdi> <span className="onb-help">{view.kFactor.source === 'BOUND_FLAT' ? t.kFromFlat : t.kShop}</span></dd></div>}
        {view.flatCheck && view.flatCheck.status !== 'NOT_EVALUATED' && <div><dt className="onb-help">{t.flatCheck}</dt>
          <dd className="font-semibold">{t[`flatCheck${view.flatCheck.status}` as const]}{view.flatCheck.max_abs_delta_mm != null && <>{' '}<bdi dir="ltr" className="onb-num">({view.flatCheck.max_abs_delta_mm.toFixed(3)} mm)</bdi></>}</dd></div>}
      </dl>

      {view.bends.length > 0 && <div className="onb-table-wrap">
        <table className="onb-table" data-expectations-table>
          <thead><tr><th>{t.seq}</th><th>{t.bendCol}</th><th>{t.angle}</th><th>{t.flangeA}</th><th>{t.flangeB}</th><th>{t.bendLine}</th><th>{t.backgauge}</th></tr></thead>
          <tbody>
            {view.bends.map(bend => {
              const unavailable = !!bend.includedUnavailable || bend.sides.some(side => side.unavailable && side.requirement !== 'NOT_INSPECTED')
              const lineBand = formatBand(bend.bendLine?.tolerance, 'mm')
              return (
                <tr key={bend.bend_uid} data-expectation-row={bend.label} data-unavailable={unavailable}>
                  <td className="onb-num">{bend.sequence_index}{bend.machine_move_index ? <span className="onb-sub">{t.move} {bend.machine_move_index}</span> : null}</td>
                  <td><bdi dir="ltr" className="font-semibold">{bend.label}</bdi>
                    {cadLabels?.get(bend.sequence_index) && cadLabels.get(bend.sequence_index) !== bend.label
                      && <span className="onb-sub" data-cad-label={cadLabels.get(bend.sequence_index)}>{t.cadLabel} <bdi dir="ltr">{cadLabels.get(bend.sequence_index)}</bdi></span>}
                    <span className="onb-sub onb-num" dir="ltr">R {bend.innerRadiusMm.toFixed(2)}</span>
                    {!bend.supported && <span className="onb-sub" style={{ color: 'var(--onb-amber)' }}>{t.unsupported}</span>}</td>
                  <td><AngleCell bend={bend} t={t} /></td>
                  <td><SideCell side={bend.sides[0]} t={t} /></td>
                  <td><SideCell side={bend.sides[1]} t={t} /></td>
                  <td className="onb-num" dir="ltr">{lineBand ?? <span className="onb-cell-na">{t.noTolerance}</span>}</td>
                  <td>{bend.backgauge.length ? bend.backgauge.map(bg => <span key={bg.characteristic_id} className="block onb-num" dir="ltr">
                    {fill(t.stageN, { n: bg.stage_index ?? '?' })} {formatNominal(bg.nominal, 'mm') ?? t.unavailableNominal}</span>) : <span className="onb-cell-na">—</span>}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>}

      {view.generalMeasurements.length > 0 && <section className="grid gap-2"><h3 className="onb-h3">{t.generalTitle}</h3>
        <ul className="grid gap-1">{view.generalMeasurements.map(gm => <li key={gm.characteristic_id} className="flex flex-wrap gap-3">
          <bdi dir="ltr">{gm.measurement_id}</bdi><strong className="onb-num" dir="ltr">{formatNominal(gm.nominal, 'mm') ?? t.unavailableNominal}</strong>
          <span className="onb-help onb-num" dir="ltr">{formatBand(gm.tolerance, 'mm') ?? ''}</span></li>)}</ul></section>}

      {view.warnings.length > 0 && <section className="grid gap-2" data-warnings>
        <h3 className="onb-h3">{t.warningsTitle}</h3>
        {view.warnings.map((issue, index) => <ReasonRow key={`${issue.code}-${index}`} code={issue.code} language={language} severity="WARNING" detail={issueDetail(issue, view)} />)}
      </section>}
      <p className="onb-help">{t.conventionsNote}</p>
    </div>
  )
}

function issueDetail(issue: { bend_uid?: string | null; graph_bend_id?: string | null; detail?: string }, view: ExpectationsView): string | null {
  const label = issue.bend_uid ? view.bends.find(bend => bend.bend_uid === issue.bend_uid)?.label : null
  const parts = [label, issue.graph_bend_id, issue.detail].filter(Boolean)
  return parts.length ? parts.join(', ') : null
}
