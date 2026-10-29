/**
 * Step 7: review expectations.
 *
 * - When the server offers a pre-release compile preview, the operator runs it
 *   here and sees the nominals and every reconciliation blocker before release.
 * - The C05 server in this tree has no preview route: E1 compiles at release.
 *   The step then says so plainly, shows the last release attempt's compile
 *   report (blockers in plain language, CAD bends with no mark) when it was made
 *   from the current inputs, and shows the active recipe's nominals when the
 *   part was released before. It never claims a review that did not happen.
 */
import { useMemo, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Cpu, ShieldCheck } from 'lucide-react'
import { onboardingApi, onboardingKeys } from '../api/onboardingApi'
import { fill, type OnbCopy, type OnbLanguage } from '../copy'
import { buildReleaseRequest, emptyReleaseRequest, previewIsCurrent } from '../model/checklist'
import { buildExpectations, compiledFrom } from '../model/expectations'
import type { CompileReportV1, ProductionApiError } from '../model/types'
import { ApiErrorNotice, EmptyState, Notice, ReasonRow } from '../components/common'
import ExpectationsTable from '../components/ExpectationsTable'
import type { StepProps } from './stepProps'

/** Recipe-side included angle per sequence index: CAD candidate sweep, or the manual entry. */
export function recipeAnglesBySequence(props: Pick<StepProps, 'summary' | 'snapshot'>): Map<number, number | null> {
  const overrides = props.snapshot.releaseDraft?.request?.bend_overrides ?? {}
  return new Map(props.summary.ordered.map(row => [row.step,
    row.source === 'manual' ? overrides[row.feature_id]?.manual_included_angle_deg ?? null : row.cadIncludedDeg]))
}

/** CAD mark label per sequence index from the saved Bends & order setup (shown beside the recipe label). */
export function cadLabelsBySequence(summary: StepProps['summary']): Map<number, string> {
  return new Map(summary.ordered.map(row => [row.step, row.label]))
}

/** E1 compile report (C05 `_compile_report`) in plain language: blockers, warnings, unmarked CAD bends. */
export function CompileReportView({ report, t, language }: { report: CompileReportV1; t: OnbCopy; language: OnbLanguage }) {
  const blockers = report.issues.filter(issue => issue.severity === 'BLOCKER')
  const warnings = report.issues.filter(issue => issue.severity === 'WARNING')
  const detail = (issue: CompileReportV1['issues'][number]) => [issue.bend_label, issue.graph_bend_id, issue.detail].filter(Boolean).join(', ') || null
  return (
    <div className="grid gap-3" data-compile-report={report.status ?? ''}>
      {blockers.length > 0 && <section className="grid gap-2" data-blockers>
        <h3 className="onb-h3">{t.blockersTitle}</h3>
        {blockers.map((issue, index) => <ReasonRow key={`${issue.code}-${index}`} code={issue.code} language={language} detail={detail(issue)} />)}
      </section>}
      {report.unmarked_graph_bends.length > 0 && <Notice tone="blocked"><span className="grid gap-1" data-unmarked>
        <strong>{t.unmarkedTitle}</strong><span>{t.unmarkedLead}</span>
        <bdi dir="ltr" className="onb-num">{report.unmarked_graph_bends.map(g => `${g.graph_bend_id} (${(180 - g.sweep_deg).toFixed(1)}°)`).join(', ')}</bdi></span></Notice>}
      {!!report.consistency_problems?.length && <ul className="grid gap-1 onb-help" dir="ltr">{report.consistency_problems.map(p => <li key={p}>{p}</li>)}</ul>}
      {report.detail && !blockers.length && <p className="onb-help" dir="auto">{report.detail}</p>}
      {warnings.length > 0 && <section className="grid gap-2" data-warnings>
        <h3 className="onb-h3">{t.warningsTitle}</h3>
        {warnings.map((issue, index) => <ReasonRow key={`${issue.code}-${index}`} code={issue.code} language={language} severity="WARNING" detail={detail(issue)} />)}
      </section>}
    </div>
  )
}

/** Nominals of the active recipe (C05 expectations), shown with their version so they are never mistaken for the current inputs. */
export function ActiveRecipeExpectations({ props }: { props: StepProps }) {
  const { snapshot, summary, t, language } = props
  const active = snapshot.onboarding?.active_recipe ?? null
  const expectations = useQuery({
    queryKey: onboardingKeys.expectations(active?.recipe_id ?? ''),
    queryFn: () => onboardingApi.getExpectations(active!.recipe_id),
    enabled: !!active,
    refetchInterval: query => (query.state.data?.status === 'PENDING' ? 2000 : false),
    retry: 1,
  })
  const angles = useMemo(() => recipeAnglesBySequence({ summary, snapshot }), [summary, snapshot])
  const view = useMemo(() => {
    const compiled = compiledFrom(expectations.data?.expectations ?? null)
    return compiled ? buildExpectations(compiled, angles) : null
  }, [expectations.data, angles])
  if (!active) return null
  return (
    <section className="grid gap-3" data-active-expectations={active.recipe_id} data-expectations-status={expectations.data?.status ?? (expectations.isError ? 'ERROR' : 'LOADING')}>
      <h3 className="onb-h3">{fill(t.activeExpectationsTitle, { v: active.version })}</h3>
      {snapshot.onboarding?.onboarding_state === 'RELEASED_STALE' && <Notice tone="attention">{t.activeExpectationsStale}</Notice>}
      {(!expectations.data || expectations.data.status === 'PENDING') && !expectations.isError && <Notice tone="info" role="status">{t.expectationsPending}</Notice>}
      {expectations.data?.status === 'FAILED' && <ReasonRow code={expectations.data.error?.code || 'EXPECTATIONS_FAILED'} language={language} detail={expectations.data.error?.message} />}
      {expectations.isError && <ApiErrorNotice error={expectations.error as unknown as ProductionApiError} language={language} />}
      {view && <ExpectationsTable view={view} t={t} language={language}
        cadLabels={snapshot.onboarding?.onboarding_state === 'RELEASED' && !summary.stale ? cadLabelsBySequence(summary) : null} />}
    </section>
  )
}

function UnavailableReview(props: StepProps) {
  const { snapshot, t, language } = props
  const last = snapshot.onboarding?.last_compile ?? null
  return (
    <div className="onb-stack" data-review-step data-preview-status="UNAVAILABLE">
      {!last?.applies_to_current_draft || last.status === 'COMPILED' ? <EmptyState icon={<ShieldCheck size={26} />} title={t.reviewCalmTitle}>
        <p data-preview-unavailable>{t.reviewUnavailableLead}</p>
      </EmptyState> : <p className="onb-lead" data-preview-unavailable>{t.reviewUnavailableLead}</p>}
      {last && <section className="onb-subsection" data-last-compile={last.status} data-last-compile-current={last.applies_to_current_draft}>
        <h3 className="onb-h3">{t.lastCompileTitle}</h3>
        <p className="onb-help">{last.applies_to_current_draft ? t.lastCompileCurrent : t.lastCompileOlder}</p>
        {last.applies_to_current_draft && last.status === 'COMPILED' && <Notice tone="info">{t.lastCompileOk}</Notice>}
        {last.applies_to_current_draft && last.status !== 'COMPILED' && <>
          {last.codes.filter(code => !last.report?.issues.some(issue => issue.code === code)).map(code =>
            <ReasonRow key={code} code={code} language={language} />)}
          {last.report && <CompileReportView report={last.report} t={t} language={language} />}
        </>}
      </section>}
      <ActiveRecipeExpectations props={props} />
    </div>
  )
}

export default function ReviewStep(props: StepProps) {
  const { partId, snapshot, summary, t, language, refresh } = props
  const queryClient = useQueryClient()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<ProductionApiError | null>(null)
  const state = snapshot.compilePreview ?? null
  const preview = state?.kind === 'preview' ? state.preview : null
  const current = previewIsCurrent(preview, snapshot)
  const compiled = useMemo(() => compiledFrom(preview?.compiled_part ?? null), [preview])
  const angles = useMemo(() => recipeAnglesBySequence({ summary, snapshot }), [summary, snapshot])
  const view = useMemo(() => (compiled ? buildExpectations(compiled, angles) : null), [compiled, angles])

  if (state?.kind === 'unavailable') return <UnavailableReview {...props} />

  const start = async () => {
    setBusy(true); setError(null)
    try {
      const revision = snapshot.releaseDraft?.revision ?? null
      const request = buildReleaseRequest(snapshot.releaseDraft?.request ?? emptyReleaseRequest(summary.draftRevision, revision), summary, revision)
      const response = await onboardingApi.startCompilePreview(partId, request)
      queryClient.setQueryData(onboardingKeys.compilePreview(partId), { kind: 'preview', preview: response })
      await refresh()
    } catch (err) {
      setError(err as ProductionApiError)
    } finally {
      setBusy(false)
    }
  }

  const pending = preview?.status === 'PENDING'
  return (
    <div className="grid gap-5" data-review-step data-preview-status={preview?.status ?? 'NONE'}>
      <p className="onb-lead">{t.reviewLead}</p>
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" className="onb-btn onb-btn-primary" disabled={busy || pending} onClick={() => void start()} data-compile-preview>
          <Cpu size={18} aria-hidden />{busy || pending ? t.compiling : preview ? t.recompile : t.compileNow}</button>
      </div>
      <ApiErrorNotice error={error} language={language} />
      {!preview && <p className="onb-muted">{t.previewMissing}</p>}
      {pending && <Notice tone="info" role="status">{t.compiling}</Notice>}
      {preview && !pending && !current && <Notice tone="attention"><span data-preview-stale>{t.previewStale}</span></Notice>}
      {preview?.status === 'FAILED' && <div className="grid gap-2" role="alert"><strong>{t.compileFailed}</strong>
        <ReasonRow code={preview.error?.code || 'INTERNAL_ERROR'} language={language} detail={preview.error?.message} /></div>}
      {!!preview?.recipe_codes?.length && <div className="grid gap-2" role="alert" data-recipe-codes>
        {preview.recipe_codes.map(code => <ReasonRow key={code} code={code} language={language} />)}</div>}
      {preview?.status === 'COMPILED' && current && <Notice tone="info" role="status"><span data-compiled-ok>{t.compiledOk}</span></Notice>}
      {preview?.status === 'BLOCKED' && <Notice tone="blocked" role="alert"><span data-compiled-blocked>{t.compiledBlocked}</span></Notice>}
      {view && <ExpectationsTable view={view} t={t} language={language} cadLabels={current ? cadLabelsBySequence(summary) : null} />}
    </div>
  )
}
