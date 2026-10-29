/**
 * Footer status ribbon: the job being measured right now (scan received /
 * preparing / measuring / building report, with progress and elapsed time),
 * else the latest issue (a quarantined scan or a failed job) with the
 * instruction "Rescan / call engineer", else an idle line.
 *
 * An issue is shown only when it is newer than the result on screen; an old
 * issue that has since been followed by a good inspection is not repeated.
 *
 * The elapsed time ticks in its own small component, so the rest of the
 * screen does not re-render every second.
 *
 * A running job whose stage and progress have not changed for
 * `stallAfterMs` (default 2 min) is flagged as possibly stalled (e.g. the
 * line service died with the job in flight): the ribbon turns amber and says
 * how long nothing has moved (`data-testid="ribbon-stalled"`). It never
 * claims the job failed; the server decides that.
 *
 * Held-back scans (station part picker): a quarantine issue offers "Choose
 * part" (or "Details" for a content problem that can only be dismissed)
 * instead of only "Rescan / call engineer"; while other scans are still held
 * back the idle ribbon says how many, with the same button.
 */
import { memo, useEffect, useState } from 'react'
import { Loader2, AlertOctagon } from 'lucide-react'

import { issueReasonText, jobStageText, t } from '../copy'
import type { Lang, LineJobSummary, StationIssue } from '../model/types'

export interface StatusRibbonProps {
  lang: Lang
  active: LineJobSummary | null
  issue: StationIssue | null
  /** finished_at of the result on screen, ISO (to hide stale issues). */
  shownFinishedAt: string | null
  /** server_time minus local time, ms (elapsed time is shown in server time) */
  serverOffsetMs: number
  /** A running job without any stage/progress change this long is flagged as possibly stalled. */
  stallAfterMs?: number
  /** Open held-back scans of this station (part picker). */
  heldCount?: number
  /** At least one held scan can be assigned to a part (not a content problem). */
  heldResolvable?: boolean
  /** Opens the part picker; absent: no picker on this screen. */
  onChoosePart?: () => void
}

export const STALL_AFTER_MS = 120_000

export function formatElapsed(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000))
  const m = Math.floor(s / 60)
  return `${m}:${String(s % 60).padStart(2, '0')}`
}

/** Elapsed time since `since` (ms epoch, server clock), ticking at 1 Hz. */
const Elapsed = memo(function Elapsed({ since, serverOffsetMs }: { since: number; serverOffsetMs: number }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [])
  return (
    <span className="st-num st-ribbon-elapsed" dir="ltr" data-testid="elapsed">
      {Number.isFinite(since) ? formatElapsed(now + serverOffsetMs - since) : '—'}
    </span>
  )
})

/** "No progress for m:ss" once `progressKey` has not changed for `stallAfterMs` (ticks at 1 Hz). */
const StallWatch = memo(function StallWatch({ progressKey, stallAfterMs, lang }: { progressKey: string; stallAfterMs: number; lang: Lang }) {
  const [now, setNow] = useState(() => Date.now())
  const [mark, setMark] = useState(() => ({ key: progressKey, at: now }))
  if (mark.key !== progressKey) setMark({ key: progressKey, at: now })
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [])
  const quiet = mark.key === progressKey ? now - mark.at : 0
  if (quiet < stallAfterMs) return null
  return (
    <span className="st-ribbon-stall" data-testid="ribbon-stalled" role="alert">
      {t('stalled', lang)} <span className="st-num" dir="ltr">{formatElapsed(quiet)}</span> – {t('stalledHint', lang)}
    </span>
  )
})

export const StatusRibbon = memo(function StatusRibbon({
  lang,
  active,
  issue,
  shownFinishedAt,
  serverOffsetMs,
  stallAfterMs = STALL_AFTER_MS,
  heldCount = 0,
  heldResolvable = false,
  onChoosePart,
}: StatusRibbonProps) {
  const pickButton = onChoosePart && heldCount > 0 && (
    <button type="button" className="st-ribbon-pick" onClick={onChoosePart} data-testid="choose-part" data-held={heldCount}>
      {heldResolvable ? t('choosePart', lang) : t('heldDetails', lang)}
      {heldCount > 1 && <span className="st-num st-ribbon-pick-count" dir="ltr">{heldCount}</span>}
    </button>
  )
  if (active) {
    const since = Date.parse(active.started_at ?? active.created_at)
    const progress = Math.max(0, Math.min(100, active.progress))
    return (
      <div className="st-ribbon st-ribbon-active" data-testid="status-ribbon" data-ribbon="active" data-job-stage={active.stage} role="status">
        <Loader2 className="st-spin" size={28} aria-hidden />
        <div className="st-ribbon-text">
          <strong>
            {t('processing', lang)}: {jobStageText(active.stage, lang)}
          </strong>
          <span className="st-ribbon-file" dir="ltr">{active.scan_file_name ?? active.sku}</span>
          {active.execution_status === 'running' && (
            <StallWatch key={active.job_id} progressKey={`${active.job_id}|${active.stage}|${active.progress}`} stallAfterMs={stallAfterMs} lang={lang} />
          )}
        </div>
        <div className="st-ribbon-progress" aria-label={`${progress}%`}>
          <div className="st-ribbon-bar" style={{ width: `${progress}%` }} />
        </div>
        <span className="st-num st-ribbon-pct" dir="ltr">{progress}%</span>
        <Elapsed since={since} serverOffsetMs={serverOffsetMs} />
      </div>
    )
  }
  const issueIsNew = issue && (!shownFinishedAt || Date.parse(issue.at) > Date.parse(shownFinishedAt))
  if (issue && issueIsNew) {
    return (
      <div className="st-ribbon st-ribbon-issue" data-testid="status-ribbon" data-ribbon="issue" data-reason={issue.reason_code} role="alert">
        <AlertOctagon size={28} aria-hidden />
        <div className="st-ribbon-text">
          <strong>
            {issue.kind === 'QUARANTINE' ? t('issueQuarantine', lang) : t('issueFailed', lang)}: {issueReasonText(issue.reason_code, lang)}
          </strong>
          {issue.file_name && <span className="st-ribbon-file" dir="auto">{issue.file_name}</span>}
        </div>
        {issue.kind === 'QUARANTINE' && pickButton ? pickButton : <span className="st-ribbon-action">{t('rescanOrCall', lang)}</span>}
      </div>
    )
  }
  return (
    <div className="st-ribbon st-ribbon-idle" data-testid="status-ribbon" data-ribbon="idle">
      <span className="st-idle-dot" aria-hidden />
      <span>{t('idle', lang)}</span>
      {pickButton && (
        <span className="st-ribbon-held" data-testid="ribbon-held">
          <span className="st-num" dir="ltr">{heldCount}</span> {t('heldCount', lang)}
        </span>
      )}
      {pickButton}
    </div>
  )
})
