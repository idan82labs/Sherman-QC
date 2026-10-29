/**
 * /production/parts/:partId/onboard - the eight-step onboarding wizard:
 * Part, Folded CAD, Drawing, Unfolded part, Bends and order, Tolerances,
 * Review expectations, Release.
 *
 * Server reads: onboarding (C05), Vendor Setup (legacy catalog), flat intake,
 * release draft and compile preview (optional; C05 has none today, so the
 * Review step explains that E1 compiles at release). All of them are re-read on window focus
 * and when the operator comes back from Vendor Setup, so a saved order or a new
 * CAD shows up without a manual reload. The current step lives in `?step=`.
 *
 * `?step=bends&studio=1` opens the tagging studio: Vendor Setup embedded in the
 * panel with the rail still visible, its own guide and a "Done, back to the
 * wizard" button that closes it and re-reads everything. While it is open the
 * focus refresh is paused (it would reload the editor under the operator) and
 * leaving with unsaved changes asks first.
 */
import { Suspense, lazy, useCallback, useEffect, useMemo, useRef } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, ArrowRight } from 'lucide-react'
import { onboardingApi, onboardingKeys } from '../api/onboardingApi'
import { fill, onbCopy, stepCopy } from '../copy'
import {
  WIZARD_STEPS, canonicalSkuPreview, deriveSteps, firstOpenStep, isStepKey, mergedPart, summarizeVendorSetup,
  type WizardSnapshot, type WizardStepKey,
} from '../model/checklist'
import type { ProductionApiError } from '../model/types'
import { ApiErrorNotice, LanguageToggle, useOnboardingLanguage } from '../components/common'
import RouteRail from '../components/RouteRail'
import ToleranceEditor from '../components/ToleranceEditor'
import ReleasePanel from '../components/ReleasePanel'
import ChiralityNotice from '../components/ChiralityNotice'
import { useChirality } from '../api/useChirality'
import PartStep from '../steps/PartStep'
import { CadStep, DrawingStep } from '../steps/FileSteps'
import FlatStep from '../steps/FlatStep'
import BendOrderStep from '../steps/BendOrderStep'
import ReviewStep from '../steps/ReviewStep'
import type { StepProps } from '../steps/stepProps'
import '../onboarding.css'

// The tagging studio is the Vendor Setup page itself, embedded (loaded on first use).
const VendorSetupPage = lazy(() => import('../../part-catalog/pages/VendorSetupPage'))
const STEP_BODY: Record<WizardStepKey, (props: StepProps) => JSX.Element> = {
  part: PartStep,
  cad: CadStep,
  drawing: DrawingStep,
  flat: FlatStep,
  bends: BendOrderStep,
  tolerances: ToleranceEditor,
  review: ReviewStep,
  release: ReleasePanel,
}

function Arrow({ back, language }: { back?: boolean; language: 'en' | 'he' }) {
  const Icon = back ? ArrowLeft : ArrowRight
  return <Icon size={16} className={language === 'he' ? 'rotate-180' : ''} aria-hidden />
}

export default function OnboardingWizardPage() {
  const { partId = '' } = useParams()
  const [params, setParams] = useSearchParams()
  const [language, setLanguage] = useOnboardingLanguage()
  const t = onbCopy[language]
  const queryClient = useQueryClient()

  const onboarding = useQuery({ queryKey: onboardingKeys.onboarding(partId), queryFn: () => onboardingApi.getOnboarding(partId), enabled: !!partId, retry: 1 })
  const vendorSetup = useQuery({ queryKey: onboardingKeys.vendorSetup(partId), queryFn: () => onboardingApi.getVendorSetup(partId), enabled: !!partId, retry: 1, refetchOnWindowFocus: false })
  const flat = useQuery({ queryKey: onboardingKeys.flat(partId), queryFn: () => onboardingApi.getFlat(partId), enabled: !!partId, retry: 1 })
  const releaseDraft = useQuery({ queryKey: onboardingKeys.releaseDraft(partId), queryFn: () => onboardingApi.getReleaseDraft(partId), enabled: !!partId, retry: 1 })
  const compilePreview = useQuery({
    queryKey: onboardingKeys.compilePreview(partId), queryFn: () => onboardingApi.getCompilePreview(partId), enabled: !!partId, retry: 1,
    refetchInterval: query => (query.state.data?.kind === 'preview' && query.state.data.preview.status === 'PENDING' ? 2000 : false),
  })

  // Left/right map of the released compile (display only; none before the first release).
  const chirality = useChirality(partId, onboarding.data?.active_recipe?.recipe_id)

  const refresh = useCallback(async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: onboardingKeys.onboarding(partId) }),
      queryClient.invalidateQueries({ queryKey: onboardingKeys.vendorSetup(partId) }),
      queryClient.invalidateQueries({ queryKey: onboardingKeys.flat(partId) }),
      queryClient.invalidateQueries({ queryKey: onboardingKeys.releaseDraft(partId) }),
      queryClient.invalidateQueries({ queryKey: onboardingKeys.compilePreview(partId) }),
      queryClient.invalidateQueries({ queryKey: onboardingKeys.parts }),
    ])
  }, [queryClient, partId])

  const studioOpen = params.get('step') === 'bends' && params.get('studio') === '1'
  const studioDirty = useRef(false)
  // Returning from Vendor Setup (another tab, or this tab after navigation) re-reads everything.
  // Not while the embedded studio is open: a new Vendor Setup snapshot would reset its editor.
  useEffect(() => {
    if (studioOpen) return
    const onFocus = () => { void refresh() }
    const onVisible = () => { if (document.visibilityState === 'visible') void refresh() }
    window.addEventListener('focus', onFocus)
    document.addEventListener('visibilitychange', onVisible)
    return () => { window.removeEventListener('focus', onFocus); document.removeEventListener('visibilitychange', onVisible) }
  }, [refresh, studioOpen])

  const snapshot: WizardSnapshot = useMemo(() => ({
    onboarding: onboarding.data ?? null,
    vendorSetup: vendorSetup.data ?? null,
    flat: flat.data ?? null,
    releaseDraft: releaseDraft.data ?? null,
    compilePreview: compilePreview.data ?? null,
  }), [onboarding.data, vendorSetup.data, flat.data, releaseDraft.data, compilePreview.data])
  const summary = useMemo(() => summarizeVendorSetup(snapshot.vendorSetup), [snapshot.vendorSetup])
  const steps = useMemo(() => deriveSteps(snapshot), [snapshot])
  const requested = params.get('step')
  const loaded = onboarding.isSuccess && vendorSetup.isFetched
  // Every read the step statuses come from has answered once (success or error). Before that the
  // rail and the step body would show an empty part ("0 of 8 steps done", every step "To do"), so
  // both show a loading state instead (`data-loaded=false`).
  const settled = [onboarding, vendorSetup, flat, releaseDraft, compilePreview].every(query => query.isFetched)
  const current: WizardStepKey = isStepKey(requested) ? requested : loaded ? firstOpenStep(steps) : 'part'
  const index = WIZARD_STEPS.indexOf(current)
  const leaveStudioOk = useCallback(() => !studioOpen || !studioDirty.current || window.confirm(onbCopy[language].confirmLeaveStudio), [studioOpen, language])
  const goTo = useCallback((key: WizardStepKey) => {
    if (!leaveStudioOk()) return
    studioDirty.current = false
    setParams(previous => { const next = new URLSearchParams(previous); next.set('step', key); next.delete('studio'); return next }, { replace: false })
    window.scrollTo({ top: 0 })
  }, [setParams, leaveStudioOk])
  const openStudio = useCallback(() => {
    studioDirty.current = false
    setParams(previous => { const next = new URLSearchParams(previous); next.set('step', 'bends'); next.set('studio', '1'); return next }, { replace: false })
    window.scrollTo({ top: 0 })
  }, [setParams])
  const closeStudio = useCallback(() => {
    if (!leaveStudioOk()) return
    studioDirty.current = false
    setParams(previous => { const next = new URLSearchParams(previous); next.delete('studio'); return next }, { replace: false })
    void refresh()
  }, [setParams, leaveStudioOk, refresh])
  const onStudioDirty = useCallback((dirty: boolean) => { studioDirty.current = dirty }, [])
  // Leaving the page (reload, close) with unsaved studio work asks the browser to confirm.
  useEffect(() => {
    if (!studioOpen) return
    const onBeforeUnload = (event: BeforeUnloadEvent) => { if (studioDirty.current) event.preventDefault() }
    window.addEventListener('beforeunload', onBeforeUnload)
    return () => window.removeEventListener('beforeunload', onBeforeUnload)
  }, [studioOpen])

  const part = mergedPart(snapshot)
  const sku = onboarding.data?.sku ?? (part ? (canonicalSkuPreview(part.part_number).ok ? canonicalSkuPreview(part.part_number).sku : part.part_number) : '')
  const dir = language === 'he' ? 'rtl' : 'ltr'
  const Body = STEP_BODY[current]
  const loadError = (onboarding.error ?? null) as ProductionApiError | null
  const stepState = steps.find(step => step.key === current)!

  const following = index < WIZARD_STEPS.length - 1 ? WIZARD_STEPS[index + 1] : null
  const stepSettled = stepState.status === 'done' || stepState.status === 'optional'
  const props: StepProps = { partId, language, t, snapshot, summary, steps, refresh, goTo, openStudio }

  return (
    <div className="onb-root onb-wizard" dir={dir} lang={language} data-wizard data-current-step={current} data-loaded={settled ? 'true' : 'false'}
      aria-busy={!settled} data-studio={studioOpen ? 'open' : undefined}>
      <header className="onb-header">
        <div className="onb-header-id">
          <Link to="/production/parts" className="onb-back" onClick={event => { if (!leaveStudioOk()) event.preventDefault() }}>
            <Arrow back language={language} />{t.backToParts}</Link>
          {part && <div className="onb-tag" data-sku-tag>
            <bdi dir="ltr" className="onb-tag-sku">{sku}</bdi>
            <span className="onb-tag-meta">{part.part_name}{part.revision ? <> <bdi dir="ltr">{t.revision} {part.revision}</bdi></> : null}</span>
          </div>}
        </div>
        <LanguageToggle language={language} onChange={setLanguage} />
      </header>

      {loadError && <div className="mb-4"><ApiErrorNotice error={loadError} language={language} /></div>}

      <div className="onb-shell">
        <RouteRail steps={steps} current={current} language={language} onSelect={goTo} loading={!settled} />
        <main className="onb-panel onb-wizard-panel min-w-0" aria-labelledby="onb-step-title">
          <div className="onb-panel-head">
            <p className="onb-kicker onb-num">{fill(t.stepOf, { n: index + 1, total: WIZARD_STEPS.length })}{studioOpen ? <> <span aria-hidden>·</span> {t.studioTitle}</> : null}</p>
            <h1 id="onb-step-title" className="onb-step-title">{stepCopy[language][current].title}</h1>
            {!studioOpen && <p className="onb-step-purpose">{stepCopy[language][current].purpose}</p>}
          </div>
          {!studioOpen && settled && <ChiralityNotice partId={partId} status={chirality.data} language={language} quiet={current !== 'release'} />}
          {studioOpen ? (
            <div className="onb-body onb-studio" key="studio" data-studio-body>
              <Suspense fallback={<p className="onb-muted p-8">{t.loading}</p>}>
                <VendorSetupPage wizard={{ language, onDone: closeStudio, onDirtyChange: onStudioDirty }} />
              </Suspense>
            </div>
          ) : (
            <div className="onb-body" key={current} data-step-body={current} data-step-status={settled ? stepState.status : 'loading'}>
              {partId && settled ? <Body {...props} /> : <p className="onb-muted" role="status" data-step-loading>{t.loading}</p>}
            </div>
          )}
          {!studioOpen && <footer className="onb-footer">
            <button type="button" className="onb-btn onb-btn-link" disabled={index === 0} onClick={() => goTo(WIZARD_STEPS[index - 1])}>
              <Arrow back language={language} />{t.back}</button>
            {following && <div className="onb-footer-next">
              <button type="button" className={`onb-btn ${stepSettled ? 'onb-btn-primary' : 'onb-btn-quiet'}`} data-next onClick={() => goTo(following)}>
                {stepState.status === 'optional' ? t.skipOptional : fill(t.nextStepHint, { s: stepCopy[language][following].title })}<Arrow language={language} /></button>
            </div>}
          </footer>}
        </main>
      </div>
    </div>
  )
}
