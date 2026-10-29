/**
 * The two additive hooks Vendor Setup uses from onboarding:
 *
 * - `useBoundFlatUnfold`: when the part has an uploaded flat whose binding is
 *   OPERATOR_CONFIRMED and current (C05 basis: same CAD sha256, same set of bend
 *   identities) for the draft being edited, returns it in the `unfold_preview`
 *   shape so Vendor Setup's unfolded view is drawn from the operator's own file.
 *   It also refuses when the draft is stale or belongs to another CAD upload,
 *   candidate generation or part revision (the same guards Vendor Setup applies
 *   to its CAD development). Otherwise null, and Vendor Setup keeps its existing
 *   CAD-development view unchanged. It never throws: a malformed server body
 *   is null after `parseFlatRecord`, and the derivation is wrapped.
 * - `WizardReturnLink`: a "Done, back to onboarding" link when Vendor Setup was
 *   opened from the wizard (`?return=/production/parts/<id>/onboard?step=...`).
 *   The target is validated, so the parameter cannot redirect elsewhere.
 * - The tagging-studio guide (2026-10-17 polish): `WizardStudioBar` replaces
 *   Vendor Setup's header when it is embedded in the wizard (`wizard` prop);
 *   `WizardStandaloneBar` adds the breadcrumb, the same guide and the return
 *   link when the full page was opened from the wizard. Both show three
 *   sub-steps (confirm detected bends -> set order -> group into מהלך) and
 *   highlight the current one from the live editor state. Standalone Vendor
 *   Setup (no wizard context) renders neither and is unchanged.
 * - `useChiralityOverlay` (2026-10-28, display only): for a released part whose
 *   left/right map says it is left/right-sensitive, a warning above the
 *   workspace and one numbered marker per distinguishing feature on the folded
 *   CAD view ("L/R 1", ...; positions in the CAD model frame, mm, the frame the
 *   bend marks use). Nothing for an achiral, unreleased or uncomputed part; a
 *   failed read (e.g. no production line on this server) shows nothing.
 */
import { useMemo, type ReactNode } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, Check, ChevronLeft, ChevronRight } from 'lucide-react'
import { ShapeUtils, Vector2 } from 'three'
import { onboardingApi, onboardingKeys } from './api/onboardingApi'
import { fill, onbCopy, type OnbLanguage } from './copy'
import { vendorCopy } from '../part-catalog/pages/vendorSetupCopy'
import './onboarding.css'
import ChiralityNotice from './components/ChiralityNotice'
import { featureMarkers } from './model/chirality'
import { boundFlatUnfoldPreview, draftMatchesSetup, safeWizardReturn, type BridgeDraft, type BridgeSetupData, type FlatUnfoldPreview, type Triangulator } from './model/flat'

const triangulate: Triangulator = (contour, holes) =>
  ShapeUtils.triangulateShape(contour.map(([x, y]) => new Vector2(x, y)), holes.map(loop => loop.map(([x, y]) => new Vector2(x, y))))

export interface BoundFlatUnfold {
  preview: FlatUnfoldPreview
  flatSha256: string
}

export function useBoundFlatUnfold(partId: string, enabled: boolean, data: BridgeSetupData | null | undefined,
  draft: BridgeDraft | null | undefined): BoundFlatUnfold | null {
  const flat = useQuery({
    queryKey: onboardingKeys.flat(partId),
    queryFn: () => onboardingApi.getFlat(partId),
    enabled: enabled && !!partId,
    retry: false,
    refetchOnWindowFocus: false,
  })
  const intake = flat.data ?? null
  return useMemo(() => {
    try {
      if (!enabled || !intake || !draft || !draftMatchesSetup(data, draft)) return null
      const preview = boundFlatUnfoldPreview(intake, { cadSha256: data?.part.cad_file_hash, draft }, triangulate)
      return preview ? { preview, flatSha256: intake.flat.sha256 } : null
    } catch {
      return null
    }
  }, [enabled, intake, data, draft])
}

export function WizardReturnLink({ partId, language }: { partId: string; language: OnbLanguage }) {
  const [params] = useSearchParams()
  const target = safeWizardReturn(params.get('return'), partId)
  if (!target) return null
  const t = onbCopy[language]
  return (
    <Link to={target} data-return-to-wizard title={t.returnHint} className="onb-btn onb-btn-done">
      <ArrowLeft size={16} className={language === 'he' ? 'rotate-180' : ''} aria-hidden />{t.returnToWizard}
    </Link>
  )
}

/** Live editor state the guide reads from Vendor Setup. */
export interface StudioProgress {
  total: number
  confirmed: number
  orderConfirmed: boolean
  orderMode: boolean
  allAssigned: boolean
  moves: number
  dirty: boolean
}

/** What the wizard hands Vendor Setup when it is embedded as the tagging studio. */
export interface VendorWizardEmbed {
  language: OnbLanguage
  onDone: () => void
  onDirtyChange?: (dirty: boolean) => void
}

type SubState = 'done' | 'current' | 'todo'

/** Sub-step states and the hint index (0 confirm, 1 order, 2 moves, 3 save). */
export function studioSubSteps(p: StudioProgress): { states: [SubState, SubState, SubState]; hint: 0 | 1 | 2 | 3 } {
  const confirmDone = p.total > 0 && p.confirmed >= p.total
  const orderDone = p.orderConfirmed && !p.orderMode
  const movesDone = orderDone && p.moves > 0
  const current: 0 | 1 | 2 | 3 = p.orderMode ? (p.allAssigned ? 2 : 1) : !confirmDone ? 0 : !orderDone ? 1 : p.dirty || movesDone ? 3 : 2
  const state = (index: 0 | 1 | 2, done: boolean): SubState => (current === index ? 'current' : done ? 'done' : 'todo')
  return { states: [state(0, confirmDone), state(1, orderDone), state(2, movesDone)], hint: current }
}

function GuideSteps({ language, states }: { language: OnbLanguage; states: [SubState, SubState, SubState] }) {
  const t = onbCopy[language]
  const labels = [t.guideConfirm, t.guideOrder, t.guideMoves]
  return (
    <ol className="onb-guide" aria-label={t.studioTitle}>
      {labels.map((label, index) => (
        <li key={label} data-sub-step={index + 1} data-state={states[index]} aria-current={states[index] === 'current' ? 'step' : undefined}>
          <span className="onb-guide-dot" aria-hidden>{states[index] === 'done' ? <Check size={14} strokeWidth={3} /> : index + 1}</span>
          <span className="onb-guide-label">{label}{index === 2 && <span className="onb-guide-opt"> · {t.guideOptional}</span>}</span>
        </li>
      ))}
    </ol>
  )
}

function guideHint(language: OnbLanguage, hint: 0 | 1 | 2 | 3): string {
  const t = onbCopy[language]
  const v = vendorCopy[language]
  return [t.guideHintConfirm, fill(t.guideHintOrder, { b: v.tagOrder }), fill(t.guideHintMoves, { b: v.tagMachineMove }), t.guideHintSave][hint]
}

/** Static version of the guide, for the "no setup yet" state of the Bends step. */
export function GuidePreview({ language }: { language: OnbLanguage }) {
  return <div className="onb-guide-preview"><GuideSteps language={language} states={['todo', 'todo', 'todo']} /></div>
}

/** Header of the embedded tagging studio: the guide, the save state, Save, and "Done, back to the wizard". */
export function WizardStudioBar({ language, progress, save, onDone }: {
  language: OnbLanguage; progress: StudioProgress; save: ReactNode; onDone: () => void
}) {
  const t = onbCopy[language]
  const { states, hint } = studioSubSteps(progress)
  return (
    <div className="onb-root onb-studio-bar" dir={language === 'he' ? 'rtl' : 'ltr'} lang={language} data-studio-bar data-studio-hint={hint}>
      <div className="onb-studio-bar-row">
        <GuideSteps language={language} states={states} />
        <div className="onb-studio-bar-actions">
          {save}
          <button type="button" className="onb-btn onb-btn-done" onClick={onDone} data-studio-done>{t.doneBackToWizard}</button>
        </div>
      </div>
      <div className="onb-studio-hint-row">
        <p className="onb-studio-hint" role="status" aria-live="polite">{guideHint(language, hint)}</p>
        <span className="onb-save-state" data-dirty={progress.dirty}>{progress.dirty ? t.unsavedChanges : t.allSaved}</span>
      </div>
    </div>
  )
}

/** Full-page Vendor Setup opened from the wizard: breadcrumb, the guide, and the return link. Null otherwise. */
export function WizardStandaloneBar({ partId, language, progress }: { partId: string; language: OnbLanguage; progress: StudioProgress }) {
  const [params] = useSearchParams()
  const target = safeWizardReturn(params.get('return'), partId)
  if (!target) return null
  const t = onbCopy[language]
  const { states, hint } = studioSubSteps(progress)
  const Sep = language === 'he' ? ChevronLeft : ChevronRight
  return (
    <div className="onb-root onb-studio-bar onb-studio-bar-sticky" dir={language === 'he' ? 'rtl' : 'ltr'} lang={language} data-studio-bar="standalone" data-studio-hint={hint}>
      <nav className="onb-crumbs" aria-label={t.studioCrumb}>
        <Link to={target}>{t.studioCrumb}</Link><Sep size={14} aria-hidden /><Link to={target}>{t.studioStep}</Link><Sep size={14} aria-hidden /><span aria-current="page">{t.studioTitle}</span>
      </nav>
      <div className="onb-studio-bar-row">
        <GuideSteps language={language} states={states} />
        <div className="onb-studio-bar-actions">
          <WizardReturnLink partId={partId} language={language} />
        </div>
      </div>
      <div className="onb-studio-hint-row">
        <p className="onb-studio-hint" role="status" aria-live="polite">{guideHint(language, hint)}</p>
        <span className="onb-save-state" data-dirty={progress.dirty}>{progress.dirty ? t.unsavedChanges : t.allSaved}</span>
      </div>
    </div>
  )
}

export function boundFlatCopy(language: OnbLanguage) {
  const t = onbCopy[language]
  return { notice: t.vendorFlatNotice, pdfFoldedOnly: t.vendorPdfFoldedOnly }
}

export interface ChiralityViewerMarker {
  id: string
  lineStart: [number, number, number]
  lineEnd: [number, number, number]
  label: string
  color: string
}

export function useChiralityOverlay(partId: string, enabled: boolean, language: OnbLanguage): { notice: ReactNode; markers: ChiralityViewerMarker[] } {
  const query = useQuery({
    queryKey: [...onboardingKeys.chirality(partId), 'vendor-setup'],
    queryFn: () => onboardingApi.getChirality(partId),
    enabled: enabled && !!partId,
    retry: false,
    refetchOnWindowFocus: false,
    refetchInterval: q => (q.state.data?.status === 'PENDING' ? 3000 : false),
  })
  const status = query.data ?? null
  const markers = useMemo(() => featureMarkers(status, onbCopy[language]).map(m => ({
    // A point marker drawn as a 0.01 mm segment (the viewer's reference-dimension overlay).
    id: m.id, lineStart: m.point, lineEnd: [m.point[0] + 0.01, m.point[1], m.point[2]] as [number, number, number], label: m.label, color: '#f97316',
  })), [status, language])
  const notice = status ? (
    <div className="onb-root onb-chirality-vendor" dir={language === 'he' ? 'rtl' : 'ltr'} lang={language} data-vendor-chirality>
      <ChiralityNotice partId={partId} status={status} language={language} quiet markersShown={markers.length > 0} />
    </div>
  ) : null
  return { notice, markers }
}
