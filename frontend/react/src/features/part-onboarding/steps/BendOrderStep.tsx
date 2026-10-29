/**
 * Step 5: bends and order. Tagging happens in the tagging studio (Vendor Setup
 * embedded in the wizard, `?step=bends&studio=1`, opened with "Tag bends"; the
 * full-page Vendor Setup with a return context stays available). This step shows
 * what was saved: draft revision, staleness (with its one-click fix), review and
 * order confirmation, the bend sequence and its מהלך groups, with a folded /
 * unfolded toggle. The unfolded view is drawn only from an uploaded flat whose
 * binding is confirmed and current; there is no CAD-derived fallback here, so
 * the operator always knows what the flat picture is made of.
 */
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { ExternalLink, FileUp, ScanLine } from 'lucide-react'
import ThreeViewer from '../../../shared/3d/ThreeViewer'
import { fill } from '../copy'
import { bendStepCodes, mergedPart } from '../model/checklist'
import { boundFlatLinesByFeature, contextFromSetup, vendorSetupHref } from '../model/flat'
import { EmptyState, ReasonList } from '../components/common'
import FlatPartView, { type FlatLineLabel } from '../components/FlatPartView'
import StaleDraftFix from '../components/StaleDraftFix'
import { GuidePreview } from '../vendorBridge'
import type { StepProps } from './stepProps'

const MOVE_COLORS = ['#4a6bf5', '#f59e0b', '#22b8cf', '#e879f9', '#a3e635', '#fb7185', '#fbbf24', '#60a5fa']

function colorOf(moveIndex: number | null, step: number): string {
  return moveIndex ? MOVE_COLORS[(moveIndex - 1) % MOVE_COLORS.length] : `hsl(${(step * 47) % 360}, 70%, 68%)`
}

export default function BendOrderStep(props: StepProps) {
  const { partId, snapshot, summary, t, language, goTo, openStudio } = props
  const [view, setView] = useState<'folded' | 'unfolded'>('folded')
  const setup = snapshot.vendorSetup
  const codes = bendStepCodes(summary)
  const flatLines = useMemo(() => boundFlatLinesByFeature(snapshot.flat, contextFromSetup(setup)), [snapshot.flat, setup])
  const canUnfold = !!flatLines?.size && !!snapshot.flat?.parse.outline_mm
  const shownView = canUnfold ? view : 'folded'

  const segments = useMemo(() => {
    const marks = new Map((setup?.draft?.marks ?? []).map(mark => [mark.feature_id, mark]))
    return summary.ordered.flatMap(row => {
      const mark = marks.get(row.feature_id)
      if (!mark?.line_start || !mark.line_end) return []
      return [{ id: row.feature_id, lineStart: mark.line_start, lineEnd: mark.line_end, label: String(row.step), accessibleLabel: row.label,
        color: colorOf(row.moveIndex, row.step), groupLabel: row.moveIndex ? fill(t.moveN, { n: row.moveIndex }) : undefined }]
    })
  }, [setup, summary, t.moveN])

  const flatLabels = useMemo(() => {
    const out = new Map<string, FlatLineLabel>()
    if (!flatLines) return out
    for (const row of summary.ordered) {
      const line = flatLines.get(row.feature_id)
      if (line) out.set(line.line_id, { text: String(row.step), bound: true })
    }
    return out
  }, [flatLines, summary])

  const part = setup?.part
  const previewUrl = part?.preview_mesh_path ? `/api/parts/${encodeURIComponent(partId)}/preview.ply?cad_sha256=${encodeURIComponent(part.cad_file_hash || '')}` : undefined
  const hasCad = !!mergedPart(snapshot)?.cad_file_hash
  const fullPage = <Link className="onb-btn onb-btn-link" to={vendorSetupHref(partId)} data-open-vendor-setup><ExternalLink size={16} aria-hidden />{t.openFullPage}</Link>

  if (!hasCad) return (
    <EmptyState data-bends-empty="no-cad" icon={<FileUp size={26} />} title={t.bendsEmptyTitle}
      actions={<button type="button" className="onb-btn onb-btn-primary" onClick={() => goTo('cad')}><FileUp size={18} aria-hidden />{t.goToCad}</button>}>
      <p>{t.bendsEmptyNoCad}</p>
    </EmptyState>
  )

  if (!summary.hasDraft) return (
    <EmptyState data-bends-empty="no-setup" icon={<ScanLine size={26} />} title={t.bendsEmptyTitle}
      actions={<>
        <button type="button" className="onb-btn onb-btn-primary onb-btn-lg" onClick={openStudio} data-tag-bends><ScanLine size={18} aria-hidden />{t.tagBends}</button>
        {fullPage}
      </>}>
      <p>{fill(t.bendsEmptyLead, { n: summary.candidateCount })}</p>
      <GuidePreview language={language} />
    </EmptyState>
  )

  const open = codes.filter(code => code !== 'DRAFT_STALE')
  return (
    <div className="onb-stack">
      <StaleDraftFix {...props} />
      <div className="onb-bends-head">
        <dl className="onb-stats" data-bend-summary>
          <div><dt>{t.bendReview}</dt>
            <dd className="onb-num" data-marks-reviewed={`${summary.confirmedCount}/${summary.activeCount}`}>{fill(t.marksReviewed, { c: summary.confirmedCount, n: summary.activeCount })}</dd></div>
          <div><dt>{t.sequenceConfirmed}</dt>
            <dd data-sequence-confirmed={summary.sequenceConfirmed} data-tone={summary.sequenceConfirmed ? 'done' : 'attention'}>{summary.sequenceConfirmed ? t.yes : t.sequenceNotConfirmed}</dd></div>
          <div><dt>{t.machineMoves}</dt><dd className="onb-num">{summary.machineMoves.length}</dd></div>
          <div><dt>{t.draftRevision}</dt><dd className="onb-num" data-draft-revision={summary.draftRevision}>{summary.draftRevision}</dd></div>
        </dl>
        <div className="onb-actions">
          <button type="button" className={`onb-btn ${open.length && !summary.stale ? 'onb-btn-primary' : 'onb-btn-quiet'}`} onClick={openStudio} data-tag-bends>
            <ScanLine size={18} aria-hidden />{t.editBends}</button>
          {fullPage}
        </div>
      </div>
      <ReasonList codes={open} language={language} t={t} severity="WARNING" />

      {summary.ordered.length > 0 && <section className="onb-bends-body">
        <div className="grid min-w-0 gap-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h3 className="onb-h3 m-0">{shownView === 'unfolded' ? t.flatViewTitle : t.folded}</h3>
            <div className="onb-toggle" role="group" aria-label={t.viewToggle}>
              <button type="button" aria-pressed={shownView === 'folded'} data-view="folded" onClick={() => setView('folded')}>{t.folded}</button>
              <button type="button" aria-pressed={shownView === 'unfolded'} data-view="unfolded" disabled={!canUnfold} title={canUnfold ? undefined : t.unfoldedNeedsFlat}
                onClick={() => setView('unfolded')}>{t.unfolded}</button>
            </div>
          </div>
          {shownView === 'unfolded' && snapshot.flat ? <>
            <FlatPartView intake={snapshot.flat} labels={flatLabels} ariaLabel={t.flatViewAria} />
            <p className="onb-help" data-unfold-source="uploaded-flat">{t.unfoldedFromFlat}</p>
          </> : previewUrl ? <div className="onb-viewer" dir="ltr">
            <ThreeViewer key={`${partId}:folded`} referenceUrl={previewUrl} referencePresentation="aluminum" manualBendSegments={segments} touchFriendly className="h-full" />
          </div> : <div className="onb-viewer onb-viewer-empty"><p className="onb-help">{t.foldedNoPreview}</p></div>}
          {!canUnfold && <p className="onb-help" data-unfold-unavailable>{t.unfoldedNeedsFlat}</p>}
        </div>

        <div className="grid content-start gap-3">
          <h3 className="onb-h3 m-0">{t.machineMoves}</h3>
          <div className="onb-moves" data-sequence>
            {summary.machineMoves.length === 0 && <p className="onb-help">{t.noMoves}</p>}
            {groupRows(summary).map(group => (
              <div key={group.key} className="onb-move" data-move={group.moveIndex ?? 'none'}
                style={{ borderInlineStartColor: group.moveIndex ? colorOf(group.moveIndex, 0) : 'var(--onb-line)' }}>
                {group.moveIndex && <strong className="onb-move-label">{fill(t.moveN, { n: group.moveIndex })}</strong>}
                <div className="flex flex-wrap gap-1.5">
                  {group.rows.map(row => <span key={row.feature_id} className="onb-bend-pill" data-bend-step={row.step}>
                    <b className="onb-num">{row.step}</b><bdi dir="ltr">{row.label}</bdi>
                    {row.source === 'manual' && <span className="onb-help">{t.manualMark}</span>}
                  </span>)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>}
    </div>
  )
}

/**
 * Consecutive rows grouped by move; consecutive bends outside any move share one
 * group (their pills wrap), so a long part's list stays short instead of one box
 * per bend running off the panel.
 */
function groupRows(summary: StepProps['summary']) {
  const groups: Array<{ key: string; moveIndex: number | null; rows: StepProps['summary']['ordered'] }> = []
  for (const row of summary.ordered) {
    const last = groups[groups.length - 1]
    if (last && last.moveIndex === row.moveIndex) last.rows.push(row)
    else groups.push({ key: row.feature_id, moveIndex: row.moveIndex, rows: [row] })
  }
  return groups
}
