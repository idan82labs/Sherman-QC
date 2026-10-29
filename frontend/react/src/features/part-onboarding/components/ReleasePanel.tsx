/**
 * Step 8: release. The button is enabled only when the server reports
 * `ready_to_release` with no blockers. A refused release lists its codes in
 * plain language, with the C05 specifics and the E1 compile report when the
 * compile ran, and keeps the button off until the server state changes. After
 * a release, and for a part released earlier, the active recipe's expectations
 * are polled until READY or FAILED and shown.
 */
import { useState } from 'react'
import { CheckCircle2, Lock, Rocket } from 'lucide-react'
import { onboardingApi } from '../api/onboardingApi'
import { fill, reasonFor } from '../copy'
import { buildReleaseRequest, emptyReleaseRequest, releaseReadiness } from '../model/checklist'
import type { ProductionApiError, ReleaseResponse } from '../model/types'
import { ActiveRecipeExpectations, CompileReportView } from '../steps/ReviewStep'
import type { StepProps } from '../steps/stepProps'
import ChecklistPanel from './ChecklistPanel'
import { Notice, ReasonRow, shortHash } from './common'
import StaleDraftFix from './StaleDraftFix'

/** A compile warning "CODE: detail" as a reason code and operator-readable detail ("+-" -> "±"). */
export function splitWarning(warning: string): { code: string; detail: string | null } {
  const match = /^([A-Z][A-Z0-9_]{2,}):\s*(.*)$/s.exec(warning.trim())
  const code = match ? match[1] : 'COMPILE_WARNING'
  const raw = match ? match[2] : warning
  const detail = raw.replace(/\+-|\+\/-/g, '±').replace(/\s+/g, ' ').trim()
  return { code, detail: detail || null }
}

function RefusalView({ error, props }: { error: ProductionApiError; props: StepProps }) {
  const { t, language } = props
  const issueCodes = new Set(error.compile?.issues.map(issue => issue.code) ?? [])
  return (
    <div className="grid gap-3" data-release-refused>
      <strong>{t.releaseFailed}</strong>
      <div role="alert" className="grid gap-2" data-api-error={error.status}>
        {error.codes.length ? error.codes.filter(code => !issueCodes.has(code)).map(code => <ReasonRow key={code} code={code} language={language}
          serverMessage={error.messages[code] || error.message} detail={error.details[code]?.join('; ') || null} />)
          : <Notice tone="blocked">{error.message || reasonFor('SERVER_ERROR', language).title}</Notice>}
      </div>
      {error.compile && <section className="grid gap-2" data-refusal-compile>
        <h3 className="onb-h3">{t.compileReportTitle}</h3>
        <CompileReportView report={error.compile} t={t} language={language} />
      </section>}
    </div>
  )
}

export default function ReleasePanel(props: StepProps) {
  const { partId, snapshot, summary, steps, t, language, refresh, goTo } = props
  const onboarding = snapshot.onboarding ?? null
  const readiness = releaseReadiness(onboarding)
  const stamp = JSON.stringify([onboarding?.blockers, onboarding?.ready_to_release, onboarding?.onboarding_state, onboarding?.active_recipe?.recipe_id])
  const [busy, setBusy] = useState(false)
  const [refusal, setRefusal] = useState<{ error: ProductionApiError; stamp: string } | null>(null)
  const [released, setReleased] = useState<(ReleaseResponse & { httpStatus: number }) | null>(null)
  const refusedNow = !!refusal && refusal.stamp === stamp
  const canRelease = readiness.ready && !busy && !refusedNow

  const release = async () => {
    if (!canRelease) return
    setBusy(true); setRefusal(null)
    try {
      const revision = snapshot.releaseDraft?.revision ?? null
      const request = buildReleaseRequest(snapshot.releaseDraft?.request ?? emptyReleaseRequest(summary.draftRevision, revision), summary, revision)
      const result = await onboardingApi.release(partId, request)
      setReleased(result)
      // The release is done: the button must not keep saying "Releasing…" while the reads refresh.
      setBusy(false)
      await refresh()
    } catch (err) {
      setRefusal({ error: err as ProductionApiError, stamp })
      await refresh().catch(() => undefined)
    } finally {
      setBusy(false)
    }
  }

  const active = onboarding?.active_recipe ?? null
  // Released and current, or a release would reproduce the active recipe: nothing to do, not "blocked".
  const nothingToRelease = !readiness.ready && readiness.codes.length === 0 && !!active
    && (readiness.unchanged || onboarding?.onboarding_state === 'RELEASED')
  const openCount = steps.filter(step => step.key !== 'release' && step.status !== 'done' && step.status !== 'optional').length
  const readyNow = readiness.ready && !refusedNow
  const isReleased = !!released || nothingToRelease
  return (
    <div className="onb-stack" data-release-panel data-ready={readiness.ready}>
      <div className="onb-release-card" data-tone={isReleased ? 'released' : readyNow ? 'ready' : 'blocked'}>
        <span className="onb-release-icon" aria-hidden>{isReleased ? <CheckCircle2 size={24} /> : readyNow ? <Rocket size={24} /> : <Lock size={22} />}</span>
        <div className="grid min-w-0 gap-1">
          <strong className="onb-release-title">{isReleased ? t.releaseDoneTitle : readyNow ? t.releaseReadyTitle : t.releaseNotReadyTitle}</strong>
          {!nothingToRelease && <p className={readyNow ? 'onb-lead' : 'onb-muted'} data-ready-text>{readyNow ? t.releaseLead
            : openCount > 1 ? fill(t.releaseOpenCount, { n: openCount }) : openCount === 1 ? t.releaseOpenOne : t.notReady}</p>}
          {nothingToRelease && !released && <p className="onb-lead" data-release-unchanged>{t.releaseUnchanged}</p>}
        </div>
        {isReleased && !busy
          ? <span className="onb-btn onb-btn-lg onb-btn-quiet" aria-disabled="true" data-release-button data-release-state="released">
            <CheckCircle2 size={16} aria-hidden />{t.releasedButton}</span>
          : <button type="button" className="onb-btn onb-btn-primary onb-btn-lg" disabled={!canRelease} onClick={() => void release()} data-release-button
            data-release-state={busy ? 'releasing' : canRelease ? 'ready' : 'locked'}>
            {!canRelease && <Lock size={16} aria-hidden />}{busy ? t.releasing : t.releaseButton}</button>}
      </div>
      <StaleDraftFix {...props} />
      {active && !released && <Notice tone="neutral"><span data-active-recipe={active.recipe_id}>{t.activeRecipeIs}: <bdi dir="ltr" className="font-semibold">{active.recipe_id}</bdi>{' '}
        <bdi dir="ltr" className="onb-num onb-help">{shortHash(active.recipe_sha256).slice(0, 8)}</bdi></span></Notice>}
      {onboarding?.onboarding_state === 'RELEASED_STALE' && !readiness.unchanged && <Notice tone="attention"><span data-released-stale>{t.releasedStale}</span></Notice>}

      {refusal && <RefusalView error={refusal.error} props={props} />}

      {released && <div className="onb-released" role="status" data-release-result={released.status} data-recipe-id={released.recipe.recipe_id}>
        <strong>{released.status === 'unchanged' ? fill(t.unchanged, { v: released.recipe.recipe_version }) : fill(t.releasedVersion, { v: released.recipe.recipe_version })}</strong>
        <span>{t.recipeLabel} <bdi dir="ltr" className="font-semibold">{released.recipe.recipe_id}</bdi></span>
        <span className="onb-help">{t.recipeHash} <bdi dir="ltr" className="onb-num">{released.recipe.recipe_sha256.slice(0, 8)}</bdi></span>
      </div>}
      {released && released.warnings.length > 0 && <section className="grid gap-2" data-release-warnings>
        <strong>{t.releaseWarnings}</strong>
        {released.warnings.map(w => {
          const { code, detail } = splitWarning(w)
          return <ReasonRow key={w} code={code} language={language} severity="WARNING" serverMessage={detail ?? undefined}
            detail={reasonFor(code, language).known ? detail : null} />
        })}
      </section>}

      <ChecklistPanel steps={steps} serverBlockers={readiness.codes.filter(code => code !== 'READY_FLAG_DISAGREES')}
        serverMessages={onboarding?.messages} language={language} onOpen={goTo} />
      {readiness.codes.includes('READY_FLAG_DISAGREES') && <ReasonRow code="READY_FLAG_DISAGREES" language={language} />}

      {/* The active recipe's nominals: the one just released (once onboarding reloads) or an earlier release. */}
      <ActiveRecipeExpectations props={props} />
    </div>
  )
}
