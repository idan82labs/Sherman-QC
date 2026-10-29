/**
 * The saved bend setup went stale (catalog `is_stale`). Explains why in plain
 * words and offers the one fix that applies:
 * - only a new drawing was attached: one click re-saves the same setup against it
 *   (`onboardingApi.refreshStaleDraft`; the server re-checks CAD and candidates);
 * - the CAD side changed: open the tagging studio to review the bends again.
 */
import { useState } from 'react'
import { RefreshCw, ScanLine } from 'lucide-react'
import { onboardingApi } from '../api/onboardingApi'
import { staleDraftRepair } from '../model/checklist'
import type { ProductionApiError } from '../model/types'
import type { StepProps } from '../steps/stepProps'
import { ApiErrorNotice } from './common'

export default function StaleDraftFix({ partId, snapshot, summary, t, language, refresh, openStudio }: StepProps) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<ProductionApiError | null>(null)
  const [done, setDone] = useState(false)
  const repair = staleDraftRepair(snapshot.vendorSetup)
  if (done && !summary.stale) return <p className="onb-result" data-tone="done" role="status" data-setup-refreshed>{t.setupRefreshed}</p>
  if (!summary.stale || repair === 'NONE') return null
  const update = async () => {
    setBusy(true); setError(null)
    try {
      const outcome = await onboardingApi.refreshStaleDraft(partId)
      if (outcome === 'REVIEW') openStudio()
      else setDone(true)
      await refresh()
    } catch (err) {
      setError(err as ProductionApiError)
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="onb-callout" data-tone="attention" role="alert" data-stale-repair={repair}>
      <div className="grid gap-1">
        <strong data-draft-stale>{t.staleTitle}</strong>
        <span className="onb-lead">{repair === 'DRAWING_ONLY' ? t.staleDrawingOnly : t.staleCadChanged}</span>
        <ApiErrorNotice error={error} language={language} />
      </div>
      <div className="onb-callout-cta">
        {repair === 'DRAWING_ONLY'
          ? <button type="button" className="onb-btn onb-btn-primary" disabled={busy} onClick={() => void update()} data-refresh-setup>
            <RefreshCw size={16} aria-hidden className={busy ? 'onb-spin' : undefined} />{busy ? t.refreshingSetup : t.refreshSetup}</button>
          : <button type="button" className="onb-btn onb-btn-primary" onClick={openStudio} data-review-bends>
            <ScanLine size={16} aria-hidden />{t.reviewBendsAgain}</button>}
      </div>
    </div>
  )
}
