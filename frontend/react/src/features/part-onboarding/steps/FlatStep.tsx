/**
 * Step 4: the unfolded part. Upload a flat DXF / flat STEP, see what the parser
 * found, ask E1 for a bend-line binding proposal (with every ambiguity it
 * reports), then decide: confirm a map (the proposal's, an alternative, or an
 * edited one) or keep the flat unbound. Only an operator-confirmed binding that
 * is current for the saved CAD and bend marks is ever used as the unfolded view
 * or as a K-factor source.
 *
 * C05 flow (`flat_intake.FlatIntakeService`): POST /flat (parse) -> POST
 * /flat/proposal -> PUT /flat/binding. Every write carries the flat record's
 * revision, so a decision made on an outdated screen is refused (409).
 */
import { useMemo, useState } from 'react'
import { ArrowRight, Ban, Cpu, RefreshCw, Trash2 } from 'lucide-react'
import { onboardingApi } from '../api/onboardingApi'
import { fill, type OnbCopy, type OnbLanguage } from '../copy'
import type { BindingMark } from '../model/bendIdentity'
import {
  compactBindingMap, contextFromSetup, contextMarks, flatBindingFreshness, flatGeometry, mirrorForConfirm, proposalIsCurrent,
  sameBindingMap, unboundMarks, validateBindingMap, type BindingMapProblem, type MirrorChoice,
} from '../model/flat'
import type { VendorSummary } from '../model/checklist'
import type { FlatBindingProposalV1, FlatIntakeV1, ProductionApiError } from '../model/types'
import { ApiErrorNotice, Disclosure, FileCard, FileDrop, Notice, ReasonRow, shortHash } from '../components/common'
import FlatPartView, { type FlatLineLabel } from '../components/FlatPartView'
import type { StepProps } from './stepProps'

function fmt(value: number | null | undefined, digits = 2): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—'
}

function ParseSummary({ intake, t, language }: { intake: FlatIntakeV1; t: OnbCopy; language: OnbLanguage }) {
  const parse = intake.parse
  if (parse.status !== 'PARSED') {
    return <div className="grid gap-2" data-flat-parse="failed">
      <strong>{t.flatParseFailed}</strong>
      <ReasonRow code={parse.code || 'FLAT_PARSE_FAILED'} language={language} detail={parse.message} />
    </div>
  }
  const rows: Array<[string, string]> = [
    [t.flatUnits, `${parse.units ?? '—'}`],
    [t.flatScale, fmt(parse.scale_to_mm, 4)],
    [t.flatArea, parse.outer_area_mm2 != null ? `${fmt(parse.outer_area_mm2, 0)} mm²` : '—'],
    [t.flatHoles, String(parse.inner_loops ?? 0)],
    [t.flatLines, String(parse.bend_lines.length)],
    [t.flatThickness, parse.thickness_mm != null ? `${fmt(parse.thickness_mm)} mm` : '—'],
  ]
  return (
    <div className="grid gap-3" data-flat-parse="parsed">
      <dl className="onb-facts">
        {rows.map(([label, value]) => <div key={label}><dt className="onb-help">{label}</dt><dd className="font-semibold onb-num" dir="ltr">{value}</dd></div>)}
      </dl>
      {parse.decimated && <p className="onb-help">{t.flatDecimated}</p>}
      {!!parse.notes?.length && <details><summary className="onb-help cursor-pointer">{t.flatNotes} ({parse.notes.length})</summary>
        <ul className="mt-2 list-disc ps-5 onb-help" dir="ltr">{parse.notes.map((note, i) => <li key={i}>{note}</li>)}</ul></details>}
    </div>
  )
}

function ProposalFacts({ proposal, t }: { proposal: FlatBindingProposalV1; t: OnbCopy }) {
  const fit = proposal.fit_choice
  return (
    <div className="grid gap-2" data-proposal>
      <Disclosure summary={<>{t.fitDetails} <span className="onb-help onb-num" dir="ltr">· {t.confidence} {fmt(proposal.confidence * 100, 0)}%</span></>}>
      <dl className="onb-facts">
        <div><dt className="onb-help">{t.confidence}</dt><dd className="font-semibold onb-num" dir="ltr">{fmt(proposal.confidence * 100, 0)}%</dd></div>
        <div><dt className="onb-help">{t.mirrored}</dt><dd className="font-semibold">{proposal.mirrored ? t.yes : t.no}</dd></div>
        <div><dt className="onb-help">{t.outlineRms}</dt><dd className="font-semibold onb-num" dir="ltr">{fmt(proposal.outline_rms_mm, 3)} mm</dd></div>
        <div><dt className="onb-help">{t.runnerUp}</dt><dd className="font-semibold onb-num" dir="ltr">{proposal.runner_up_rms_mm == null ? '—' : `${fmt(proposal.runner_up_rms_mm, 3)} mm`}</dd></div>
        <div><dt className="onb-help">{t.kFlat}</dt><dd className="font-semibold onb-num" dir="ltr">{fmt(proposal.k_factor, 3)} ± {fmt(proposal.k_uncertainty, 3)}</dd></div>
        <div><dt className="onb-help">{t.kBand}</dt><dd className="font-semibold onb-num" dir="ltr">{proposal.k_plausible_band ? `${fmt(proposal.k_plausible_band[0], 2)}–${fmt(proposal.k_plausible_band[1], 2)}` : '—'}</dd></div>
        <div><dt className="onb-help">{t.developedDelta}</dt><dd className="font-semibold onb-num" dir="ltr">{proposal.developed_max_abs_delta_mm == null ? '—' : `${fmt(proposal.developed_max_abs_delta_mm, 3)} mm`}</dd></div>
      </dl>
      </Disclosure>
      {proposal.needs_operator_review && <Notice tone="attention"><span data-warn="needs-review">{t.needsReview}</span></Notice>}
      {fit?.ambiguous && <Notice tone="attention"><span data-warn="ambiguous">{t.fitAmbiguous}</span></Notice>}
      {fit?.mirror_determined !== true && <Notice tone="attention"><span data-warn="mirror">{t.mirrorUndetermined}</span></Notice>}
      {proposal.k_clamped && <Notice tone="attention"><span data-warn="k-clamped">{t.kClamped}</span></Notice>}
    </div>
  )
}

function BindingEditor({ intake, marks, proposal, summary, t, language, onDone }: {
  intake: FlatIntakeV1; marks: BindingMark[]; proposal: FlatBindingProposalV1 | null; summary: VendorSummary
  t: OnbCopy; language: OnbLanguage; onDone: () => Promise<void>
}) {
  const confirmed = intake.state === 'CONFIRMED' && intake.binding?.state === 'OPERATOR_CONFIRMED' ? intake.binding : null
  const [map, setMap] = useState<Record<string, string>>(() => ({ ...(confirmed?.bend_line_map ?? proposal?.bend_line_map ?? {}) }))
  const [mirrorChoice, setMirrorChoice] = useState<MirrorChoice>(() =>
    confirmed && proposal && typeof confirmed.mirrored === 'boolean' ? (confirmed.mirrored === proposal.mirrored ? 'AS_PROPOSED' : 'FLIPPED') : null)
  const [selected, setSelected] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<ProductionApiError | null>(null)
  const [done, setDone] = useState(false)

  const stepOf = useMemo(() => new Map(summary.ordered.map(row => [row.feature_id, row.step])), [summary])
  const labelOfUid = useMemo(() => new Map(marks.map(mark => [mark.bend_uid, mark.label])), [marks])
  const lines = intake.parse.bend_lines
  const mirror = mirrorForConfirm(proposal, map, mirrorChoice)
  const problems: BindingMapProblem[] = [
    ...validateBindingMap(map, lines.map(line => line.line_id), marks.map(mark => mark.bend_uid)),
    ...(mirror.needsChoice ? ['MIRROR_UNDECIDED' as const] : []),
  ]
  const edited = !!proposal && !sameBindingMap(map, proposal.bend_line_map)
  const fit = proposal?.fit_choice
  const labels = useMemo(() => new Map<string, FlatLineLabel>(lines.map(line => {
    const uid = map[line.line_id]
    return [line.line_id, uid ? { text: `${line.line_id} → ${labelOfUid.get(uid) ?? '?'}`, bound: true } : { text: line.line_id, bound: false }]
  })), [lines, map, labelOfUid])

  const confirm = async () => {
    if (problems.length) return
    setBusy(true); setError(null); setDone(false)
    try {
      await onboardingApi.decideFlatBinding(intake.part_id || '', {
        expected_revision: intake.revision, state: 'OPERATOR_CONFIRMED', bend_line_map: compactBindingMap(map), mirrored: mirror.mirrored,
      })
      setDone(true)
      await onDone()
    } catch (err) {
      setError(err as ProductionApiError)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid gap-4" data-binding-editor data-binding-state={intake.state}>
      {flatGeometry(intake).bbox && <FlatPartView intake={intake} labels={labels} selectedLineId={selected} ariaLabel={t.flatViewAria}
        onSelectLine={id => { setSelected(id); document.getElementById(`bind-${id}`)?.focus() }} />}
      {!intake.parse.outline_mm && <p className="onb-help">{t.flatNoOutline}</p>}

      {!!fit?.alternative_maps?.length && <div className="grid gap-2" data-alternatives>
        <span className="onb-help">{t.alternatives}</span>
        {fit.alternative_maps.map((alt, index) => (
          <div key={index} className="flex flex-wrap items-center gap-2">
            <bdi dir="ltr" className="onb-num text-sm">{Object.entries(alt).map(([line, uid]) => `${line}→${labelOfUid.get(uid) ?? '?'}`).join('  ')}</bdi>
            <button type="button" className="onb-btn onb-btn-quiet" data-use-alternative={index}
              onClick={() => { setMap({ ...alt }); setMirrorChoice(null) }}>{t.useAlternative}</button>
          </div>
        ))}
      </div>}

      <div className="onb-table-wrap">
        <table className="onb-table" data-binding-table>
          <thead><tr><th>{t.flatLine}</th><th>{t.lineSource}</th><th>{t.lineOffset}</th><th>{t.cadBend}</th></tr></thead>
          <tbody>
            {lines.map(line => {
              const offset = proposal?.line_offsets_mm?.[line.line_id]
              return (
                <tr key={line.line_id} data-line-row={line.line_id}>
                  <td><bdi dir="ltr" className="font-semibold">{line.line_id}</bdi></td>
                  <td><span dir="ltr">{line.source}</span>{line.layer ? <span className="onb-sub" dir="ltr">{line.layer}</span> : null}
                    {line.annotation ? <span className="onb-sub" dir="ltr">{line.annotation}</span> : null}</td>
                  <td className="onb-num" dir="ltr">{offset == null ? '—' : `${fmt(offset, 2)} mm`}</td>
                  <td>
                    <select id={`bind-${line.line_id}`} className="onb-select" value={map[line.line_id] ?? ''} aria-label={`${t.cadBend} ${line.line_id}`}
                      onFocus={() => setSelected(line.line_id)}
                      onChange={event => setMap(previous => ({ ...previous, [line.line_id]: event.target.value }))}>
                      <option value="">{t.notABendLine}</option>
                      {marks.map(mark => <option key={mark.bend_uid} value={mark.bend_uid}>
                        {mark.label}{stepOf.has(mark.feature_id) ? ` (${t.bendStep} ${stepOf.get(mark.feature_id)})` : ''}</option>)}
                    </select>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {edited && <p className="onb-help" data-manual-edit>{t.manualEdited}</p>}

      {proposal && mirror.undetermined && <fieldset className="grid gap-2" data-mirror-choice={mirrorChoice ?? 'none'}>
        <legend className="font-semibold">{t.mirrorQuestion}</legend>
        <p className="onb-help">{t.mirrorHelp}</p>
        <div className="flex flex-wrap gap-4">
          {(['AS_PROPOSED', 'FLIPPED'] as const).map(choice => (
            <label key={choice} className="flex min-h-11 items-center gap-2">
              <input type="radio" name={`mirror-${intake.flat.sha256}`} value={choice} data-mirror={choice}
                checked={mirrorChoice === choice || (mirrorChoice === null && choice === 'AS_PROPOSED' && !edited)}
                onChange={() => setMirrorChoice(choice)} />
              <span>{choice === 'AS_PROPOSED' ? t.mirrorAsProposed : t.mirrorFlipped}
                {' '}<span className="onb-help">({t.mirrored}: {(choice === 'AS_PROPOSED' ? proposal.mirrored : !proposal.mirrored) ? t.yes : t.no})</span></span>
            </label>
          ))}
        </div>
      </fieldset>}

      {problems.length > 0 && <ul className="grid gap-1" data-binding-problems>{problems.map(problem =>
        <li key={problem} className="onb-error-text" data-binding-problem={problem}>{t[`bindingProblem${problem}` as const]}</li>)}</ul>}
      <div className="flex flex-wrap gap-2">
        <button type="button" className="onb-btn onb-btn-primary" data-confirm-binding disabled={busy || problems.length > 0} onClick={() => void confirm()}>
          {busy ? t.confirming : t.confirmBinding}
        </button>
      </div>
      {done && <Notice tone="info" role="status">{t.bindingConfirmed}</Notice>}
      <ApiErrorNotice error={error} language={language} />
    </div>
  )
}

export default function FlatStep(props: StepProps) {
  const { partId, snapshot, summary, t, language, refresh } = props
  const intake = snapshot.flat ?? null
  const [busy, setBusy] = useState<'upload' | 'propose' | 'remove' | 'unbound' | null>(null)
  const [progress, setProgress] = useState<number | null>(null)
  const [error, setError] = useState<ProductionApiError | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const run = async (kind: NonNullable<typeof busy>, action: () => Promise<unknown>, success?: string) => {
    setBusy(kind); setError(null); setMessage(null)
    try { await action(); if (success) setMessage(success) } catch (err) { setError(err as ProductionApiError) } finally {
      setBusy(null); setProgress(null)
      await refresh().catch(() => undefined)
    }
  }
  const ctx = useMemo(() => contextFromSetup(snapshot.vendorSetup), [snapshot.vendorSetup])
  const marks = useMemo(() => contextMarks(ctx), [ctx])
  const freshness = flatBindingFreshness(intake, snapshot.vendorSetup)
  const currentProposal = proposalIsCurrent(intake, ctx) ? intake?.proposal?.proposal ?? null : null
  const record = intake?.proposal ?? null
  const unbound = useMemo(() => unboundMarks(intake, ctx), [intake, ctx])
  const parsed = intake?.parse.status === 'PARSED'
  const goToBends = props.goTo
  return (
    <div className="onb-stack">
      {!intake && <p className="onb-lead">{t.flatOptionalLead}</p>}
      {intake && <div data-flat-current={intake.flat.sha256} data-flat-revision={intake.revision}>
        <FileCard name={intake.flat.original_filename || intake.flat.sha256} kind={intake.flat.format}
          facts={<>{t.fileHash} <bdi dir="ltr" className="onb-num">{shortHash(intake.flat.sha256)}</bdi></>}
          result={parsed ? `${t.flatLines}: ${intake.parse.bend_lines.length}` : t.flatParseFailed} resultTone={parsed ? 'done' : 'blocked'}
          actions={<button type="button" className="onb-btn onb-btn-danger" disabled={!!busy}
            onClick={() => { if (window.confirm(t.removeFlatConfirm)) void run('remove', () => onboardingApi.deleteFlat(partId, intake.revision), t.flatRemoved) }}>
            <Trash2 size={16} aria-hidden />{t.removeFlat}</button>} />
      </div>}
      <FileDrop accept=".dxf,.step,.stp" busy={busy === 'upload'} disabled={!!busy && busy !== 'upload'} progress={progress} compact={!!intake}
        label={intake ? t.replaceFile : t.chooseFile} title={t.dropTitleFlat} hint={t.flatAccept} t={t}
        onFile={file => void run('upload', () => onboardingApi.uploadFlat(partId, file, setProgress))} />
      {message && <Notice tone="info" role="status">{message}</Notice>}
      <ApiErrorNotice error={error} language={language} />
      {intake && <>
        <section className="onb-subsection"><h3 className="onb-h3">{parsed ? t.flatParsed : t.flatParseFailed}</h3>
          <ParseSummary intake={intake} t={t} language={language} /></section>
        {parsed && <section className="onb-subsection" data-binding-section>
          <div className="flex flex-wrap items-center gap-3"><h3 className="onb-h3 m-0">{t.bindingTitle}</h3>
            <span className="onb-chip" data-tone={freshness === 'FRESH' || freshness === 'DECLINED' ? 'done' : 'attention'} data-binding-chip={intake.state}>
              {t[`flatState${intake.state}` as const]}</span></div>
          <p className="onb-lead">{t.bindingLead}</p>
          {freshness === 'FRESH' && <Notice tone="info"><span data-binding-fresh>{t.bindingFresh}</span></Notice>}
          {freshness === 'FRESH' && unbound.length > 0 && <Notice tone="attention"><span data-partial-binding={unbound.length}>{fill(t.partialBinding, { n: unbound.length })}</span></Notice>}
          {freshness === 'DECLINED' && <Notice tone="info"><span data-flat-declined>{t.declinedNotice}</span></Notice>}
          {(freshness === 'STALE_CAD' || freshness === 'MARKS_CHANGED' || freshness === 'LINES_CHANGED') &&
            <Notice tone="attention"><span data-binding-stale={freshness}>{t[`bindingStale${freshness}` as const]}</span></Notice>}
          {record?.status === 'RECONCILIATION_BLOCKED' && <div className="grid gap-2" data-reconciliation-blocked>
            <ReasonRow code="FLAT_RECONCILIATION_BLOCKED" language={language} />
            {!!record.reconciliation?.mark_not_in_flange_graph?.length && <p className="onb-help">{t.marksNotInGraph}: <bdi dir="ltr">{record.reconciliation.mark_not_in_flange_graph
              .map(uid => record.bend_labels[uid] ?? uid).join(', ')}</bdi></p>}
            {!!record.reconciliation?.unmarked_bend && <p className="onb-help">{t.unmarkedBends}: <bdi className="onb-num">{record.reconciliation.unmarked_bend}</bdi></p>}
          </div>}
          {record && (record.status === 'PARSE_FAILED' || record.status === 'CAD_READ_FAILED') &&
            <ReasonRow code={record.reason || record.status} language={language} detail={record.detail} />}
          {record?.status === 'PROPOSED' && !currentProposal && <Notice tone="attention"><span data-proposal-stale>{t.proposalStale}</span></Notice>}
          {!record && <p className="onb-help" data-no-proposal>{t.noProposalYet}</p>}

          <div className="onb-actions">
            <button type="button" className={`onb-btn ${currentProposal ? 'onb-btn-quiet' : 'onb-btn-primary'}`} data-propose disabled={!!busy || !marks.length}
              aria-describedby={!marks.length ? 'onb-propose-why' : undefined} title={!marks.length ? t.proposeNeedsSetup : undefined}
              onClick={() => void run('propose', () => onboardingApi.proposeFlat(partId, intake.revision))}>
              {currentProposal ? <RefreshCw size={16} aria-hidden /> : <Cpu size={16} aria-hidden />}
              {busy === 'propose' ? t.proposing : currentProposal ? t.repropose : t.proposeBinding}</button>
            {intake.state !== 'DECLINED' && <button type="button" className="onb-btn onb-btn-quiet" data-keep-unbound disabled={!!busy} title={t.keepUnboundHint}
              onClick={() => void run('unbound', () => onboardingApi.decideFlatBinding(partId, { expected_revision: intake.revision, state: 'UNBOUND' }))}>
              <Ban size={16} aria-hidden />{t.keepUnbound}</button>}
          </div>
          {!marks.length && <div className="onb-disabled-why" role="note" data-propose-disabled-reason>
            <span id="onb-propose-why" data-marks-need-saving>{t.proposeNeedsSetup}</span>
            <button type="button" className="onb-btn onb-btn-link" onClick={() => goToBends('bends')}>{t.goToBends}<ArrowRight size={16} aria-hidden className={language === 'he' ? 'rotate-180' : ''} /></button>
          </div>}
          {currentProposal && <ProposalFacts proposal={currentProposal} t={t} />}
          {marks.length > 0 && <BindingEditor key={`${intake.flat.sha256}:${intake.revision}:${intake.state}`}
            intake={{ ...intake, part_id: intake.part_id || partId }} marks={marks} proposal={currentProposal} summary={summary} t={t} language={language}
            onDone={refresh} />}
        </section>}
      </>}
    </div>
  )
}
