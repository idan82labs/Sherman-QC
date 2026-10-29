/**
 * Recent scans (at most 10, newest first) as compact chips in their summary
 * colour; a failed job shows its failure code in grey (an engine failure is
 * never a part FAIL). Clicking a completed job opens it read-only.
 *
 * Chip colours are an index taken from the feed's job summaries; the result
 * itself is re-validated when a chip is opened. `verifiedStates` overrides a
 * chip with the validated verdict (the job on screen), or with null for a
 * result that was refused, which greys the chip instead of colouring it from
 * an unverified summary.
 */
import { DISPLAY_TOKENS } from '../model/vocabulary'
import { issueReasonText, t } from '../copy'
import type { Lang, LineJobSummary } from '../model/types'

export interface HistoryStripProps {
  lang: Lang
  jobs: LineJobSummary[]
  /** Job currently on screen. */
  shownJobId: string | null
  /** job_id -> validated summary state (null = result refused). */
  verifiedStates?: Record<string, string | null>
  onOpen: (job: LineJobSummary) => void
}

function clock(iso: string | null): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })
}

export function HistoryStrip({ lang, jobs, shownJobId, verifiedStates = {}, onOpen }: HistoryStripProps) {
  const list = jobs.filter(j => j.kind === 'inspect').slice(0, 10)
  return (
    <nav className="st-history" aria-label={t('history', lang)} data-testid="history-strip">
      <span className="st-history-title">{t('history', lang)}</span>
      {list.length === 0 && <span className="st-history-empty">{t('historyEmpty', lang)}</span>}
      <ol>
        {list.map(job => {
          const completed = job.execution_status === 'completed' && job.summary_state
          const verified = job.job_id in verifiedStates ? verifiedStates[job.job_id] : job.summary_state
          const refused = !!completed && verified == null
          const token = completed && verified ? (DISPLAY_TOKENS[verified] ?? null) : null
          const tone = token ? token.tone : job.execution_status === 'failed' ? 'unknown' : 'inactive'
          const label = token
            ? token[lang]
            : refused
              ? t('resultRefusedShort', lang)
              : job.execution_status === 'failed'
                ? issueReasonText(job.failure_code ?? 'FAILED', lang)
                : job.execution_status
          return (
            <li key={job.job_id}>
              <button
                type="button"
                className={`st-hchip st-tone-${tone}${job.job_id === shownJobId ? ' is-shown' : ''}`}
                disabled={!completed}
                onClick={() => onOpen(job)}
                data-job-id={job.job_id}
                data-verified={refused ? 'refused' : undefined}
                title={`${job.scan_file_name ?? job.job_id} – ${label}`}
              >
                <span className="st-hchip-dot" aria-hidden />
                <span className="st-hchip-serial" dir="ltr">{job.serial ?? job.job_id.slice(5, 11)}</span>
                <span className="st-hchip-time st-num" dir="ltr">{clock(job.finished_at ?? job.created_at)}</span>
              </button>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
