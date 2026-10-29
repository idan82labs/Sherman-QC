/**
 * Release checklist: every wizard step with its state and, for anything open,
 * the plain-language reason and next action (the first reason in full, the rest
 * behind "N more"). Informational codes on optional steps (no compile preview on
 * this server) read as a quiet note, not a problem. The server's blocker list
 * is shown separately and is the release authority.
 */
import { ArrowRight } from 'lucide-react'
import { stepCopy, onbCopy, reasonFor, type OnbLanguage } from '../copy'
import type { StepState } from '../model/checklist'
import { ReasonList, ReasonRow, StatusChip } from './common'

export default function ChecklistPanel({ steps, serverBlockers, serverMessages, language, onOpen }: {
  steps: StepState[]; serverBlockers: string[]; serverMessages?: Record<string, string> | null; language: OnbLanguage
  onOpen: (key: StepState['key']) => void
}) {
  const t = onbCopy[language]
  return (
    <section className="onb-subsection" data-checklist>
      <h3 className="onb-h3">{t.checklistTitle}</h3>
      {serverBlockers.length > 0 && <div className="grid gap-2" data-server-blockers>
        <strong className="text-sm">{t.blockersTitle}</strong>
        <ReasonList codes={serverBlockers} language={language} t={t}
          render={code => <ReasonRow key={code} code={code} language={language} serverMessage={serverMessages?.[code]} />} />
      </div>}
      <ol className="onb-check">
        {steps.filter(step => step.key !== 'release').map(step => {
          const open = step.status !== 'done' && step.codes.length > 0
          return (
            <li key={step.key} className="onb-check-row" data-check={step.key} data-status={step.status}>
              <span className="onb-num onb-check-n">{steps.indexOf(step) + 1}</span>
              <div className="grid min-w-0 gap-2">
                <div className="onb-check-head">
                  <button type="button" className="onb-check-title" onClick={() => onOpen(step.key)}>{stepCopy[language][step.key].title}</button>
                  <StatusChip status={step.status} t={t} />
                  {open && step.status !== 'optional' && <button type="button" className="onb-btn onb-btn-link onb-check-open" onClick={() => onOpen(step.key)}>
                    {t.openStep}<ArrowRight size={14} aria-hidden className={language === 'he' ? 'rotate-180' : ''} /></button>}
                </div>
                {open && (step.status === 'optional'
                  ? <p className="onb-help" data-reason-note={step.codes[0]}>{reasonFor(step.codes[0], language).title} {reasonFor(step.codes[0], language).action}</p>
                  : <ReasonList codes={step.codes} language={language} t={t} severity={step.status === 'blocked' ? 'BLOCKER' : 'WARNING'} />)}
              </div>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
