/**
 * The step rail. A vertical steel strip with one station per step; a completed
 * station shows a formed flange (rotated 90 degrees out of the strip), so the
 * rail reads as the part being bent as onboarding progresses.
 *
 * Where am I / what's next: a progress line on top ("3 of 8 steps done"), the
 * current station carries an accent edge and "You are here", and the first open
 * step after it is marked "Next".
 */
import { fill, onbCopy, stepCopy, type OnbLanguage } from '../copy'
import { nextOpenStep, type StepState, type WizardStepKey } from '../model/checklist'

function Joint({ status }: { status: StepState['status'] }) {
  const stroke = status === 'done' ? '#c9d2dc' : status === 'blocked' ? 'var(--onb-red)' : status === 'attention' ? 'var(--onb-amber)' : '#7d8a99'
  return (
    <span className="onb-joint" aria-hidden>
      <svg viewBox="0 0 50 44" width="50" height="44">
        {/* bend radius at the strip */}
        <circle cx="25" cy="22" r="7" fill="var(--onb-bg)" stroke={stroke} strokeWidth="2" />
        {/* the flange: lies along the strip (rotated) until the step is done, then stands out */}
        <rect className="onb-flange" x="25" y="20" width="22" height="4" rx="1.5" fill={stroke} />
        {status === 'blocked' && <path d="M21 18 L29 26 M29 18 L21 26" stroke="var(--onb-red)" strokeWidth="2" />}
        {status === 'attention' && <circle cx="25" cy="22" r="2.5" fill="var(--onb-amber)" />}
        {status === 'working' && <circle cx="25" cy="22" r="2.5" fill="var(--onb-dykem)" />}
      </svg>
    </span>
  )
}

export default function RouteRail({ steps, current, language, onSelect, loading = false }: {
  steps: StepState[]; current: WizardStepKey; language: OnbLanguage; onSelect: (key: WizardStepKey) => void
  /** The part's reads have not answered yet: no counts or statuses are claimed. */
  loading?: boolean
}) {
  const t = onbCopy[language]
  const done = steps.filter(step => step.status === 'done').length
  const next = loading ? null : nextOpenStep(steps, current)
  return (
    <nav className={`onb-rail${loading ? ' is-loading' : ''}`} aria-label={t.routeLabel} aria-busy={loading}>
      <div className="onb-rail-progress" data-rail-progress={loading ? 'loading' : `${done}/${steps.length}`}>
        <span className="onb-num">{loading ? t.loading : fill(t.railProgress, { d: done, n: steps.length })}</span>
        <span className="onb-rail-track" aria-hidden><span style={{ width: `${(done / Math.max(1, steps.length)) * 100}%` }} /></span>
      </div>
      <ol>
        {steps.map((step, index) => {
          const here = step.key === current
          return (
            <li key={step.key} className="onb-station" data-status={loading ? 'loading' : step.status} data-step={step.key} data-next={step.key === next ? 'true' : undefined}
              aria-current={here ? 'step' : undefined}>
              <button type="button" onClick={() => onSelect(step.key)}>
                <Joint status={loading ? 'todo' : step.status} />
                <span className="min-w-0">
                  <span className="onb-station-title"><span className="onb-num onb-station-n">{index + 1}</span>{stepCopy[language][step.key].title}</span>
                  <span className="onb-station-status">
                    {here ? <span className="onb-here">{t.youAreHere}</span> : null}
                    {here && !loading ? <span aria-hidden> · </span> : null}
                    {loading ? null : t[`status${step.status}` as const]}
                    {step.key === next && !here ? <span className="onb-next-pill">{t.nextUp}</span> : null}
                  </span>
                </span>
              </button>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
