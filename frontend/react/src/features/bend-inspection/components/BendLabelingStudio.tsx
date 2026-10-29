import { Check, Download, MousePointer2, Save, Trash2, Undo2, X } from 'lucide-react'
import clsx from 'clsx'

import type {
  BendInstanceLabels,
  BendMarkerReviewDecision,
  ManualBendLabel,
} from '../../../services/api'

interface MarkerChoice {
  bendId: string
  label: string
}

interface BendLabelingStudioProps {
  enabled: boolean
  labels: BendInstanceLabels
  markers: MarkerChoice[]
  draftPoint?: [number, number, number] | null
  loading?: boolean
  saving?: boolean
  saved?: boolean
  error?: string | null
  onEnabledChange: (enabled: boolean) => void
  onSave: () => void
  onUndoDraft: () => void
  onDeleteManualBend: (bendId: string) => void
  onMarkerDecision: (bendId: string, decision: BendMarkerReviewDecision) => void
  onReviewStatusChange: (status: BendInstanceLabels['review_status']) => void
}

const decisionLabels: Array<{ decision: BendMarkerReviewDecision; label: string }> = [
  { decision: 'real_bend', label: 'Real bend' },
  { decision: 'not_bend_outline', label: 'Outline / not bend' },
  { decision: 'uncertain', label: 'Uncertain' },
]

function formatPoint(point: [number, number, number]) {
  return point.map((value) => value.toFixed(2)).join(', ')
}

function downloadLabels(labels: BendInstanceLabels) {
  const blob = new Blob([`${JSON.stringify(labels, null, 2)}\n`], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `${labels.job_id}-bend-instance-labels.json`
  anchor.click()
  URL.revokeObjectURL(url)
}

export default function BendLabelingStudio({
  enabled,
  labels,
  markers,
  draftPoint,
  loading,
  saving,
  saved,
  error,
  onEnabledChange,
  onSave,
  onUndoDraft,
  onDeleteManualBend,
  onMarkerDecision,
  onReviewStatusChange,
}: BendLabelingStudioProps) {
  const reviews = new Map(labels.marker_reviews.map((review) => [review.bend_id, review.decision]))
  const allMarkersReviewed = markers.every((marker) => reviews.has(marker.bendId))

  return (
    <div className="mb-3 rounded-xl border border-cyan-500/30 bg-cyan-500/5 p-3">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="flex items-center gap-2 text-sm font-semibold text-cyan-100">
            <MousePointer2 className="h-4 w-4" />
            Bend Label Studio
          </p>
          <p className="mt-1 text-xs text-dark-400">
            Create human bend axes in <code>reference_mesh_raw</code> and explicitly reject F1 outlines.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            className={clsx('btn', enabled ? 'btn-primary' : 'btn-secondary')}
            onClick={() => onEnabledChange(!enabled)}
          >
            {enabled ? 'Finish labeling' : 'Label bends'}
          </button>
          <button
            type="button"
            className="btn btn-secondary inline-flex items-center gap-2"
            onClick={() => downloadLabels(labels)}
            disabled={loading}
          >
            <Download className="h-4 w-4" />
            Export JSON
          </button>
          <button
            type="button"
            className="btn btn-secondary inline-flex items-center gap-2"
            onClick={onSave}
            disabled={loading || saving}
          >
            <Save className="h-4 w-4" />
            {saving ? 'Saving…' : 'Save review'}
          </button>
          <button
            type="button"
            className={clsx('btn', labels.review_status === 'complete' ? 'btn-secondary' : 'btn-primary')}
            onClick={() => onReviewStatusChange(labels.review_status === 'complete' ? 'in_progress' : 'complete')}
            disabled={labels.review_status !== 'complete' && !allMarkersReviewed}
          >
            {labels.review_status === 'complete' ? 'Reopen review' : 'Mark review complete'}
          </button>
        </div>
      </div>

      {error && (
        <div className="mt-3 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-200">
          {error}
        </div>
      )}
      {saved && !error && (
        <div className="mt-3 flex items-center gap-2 text-xs text-emerald-300">
          <Check className="h-4 w-4" /> Saved to this pipeline job.
        </div>
      )}
      {labels.review_status === 'complete' && (
        <div className="mt-3 flex items-center gap-2 text-xs text-cyan-200">
          <Check className="h-4 w-4" /> Manual review complete for this pipeline job.
        </div>
      )}
      {enabled && !allMarkersReviewed && (
        <p className="mt-3 text-xs text-amber-300">
          Review every F1 marker before completing this job. Add cyan axes for every missed bend.
        </p>
      )}

      {enabled && (
        <div className="mt-4 space-y-4 border-t border-cyan-500/20 pt-4">
          <div className="flex flex-col gap-2 rounded-lg border border-dark-600 bg-dark-900/60 px-3 py-2 text-xs text-dark-300 lg:flex-row lg:items-center lg:justify-between">
            <span>
              Rotate to inspect the local cross-section, then click the two endpoints of one real bend. Gold lines are your labels; the amber point is the pending first endpoint.
            </span>
            <button
              type="button"
              className="btn btn-secondary inline-flex items-center gap-2"
              onClick={onUndoDraft}
              disabled={!draftPoint}
            >
              <Undo2 className="h-4 w-4" />
              Undo first point
            </button>
          </div>

          {draftPoint && (
            <p className="text-xs text-amber-300">
              First endpoint: [{formatPoint(draftPoint)}]. Click the opposite endpoint to create the bend.
            </p>
          )}

          <div>
            <div className="mb-2 flex items-center justify-between">
              <p className="text-xs uppercase tracking-wide text-dark-400">Human bend axes</p>
              <span className="text-xs text-dark-500">{labels.manual_bends.length} labeled</span>
            </div>
            {labels.manual_bends.length === 0 ? (
              <p className="rounded-lg border border-dark-700 bg-dark-900/50 px-3 py-3 text-xs text-dark-500">
                No human bend axes yet. Click two surface points in the viewer.
              </p>
            ) : (
              <div className="grid gap-2 lg:grid-cols-2">
                {labels.manual_bends.map((bend: ManualBendLabel) => (
                  <div key={bend.bend_id} className="flex items-center justify-between gap-3 rounded-lg border border-dark-600 bg-dark-900/55 px-3 py-2">
                    <div className="min-w-0 text-xs">
                      <p className="font-medium text-cyan-200">{bend.bend_id}</p>
                      <p className="mt-1 truncate text-dark-500">
                        [{formatPoint(bend.line_start)}] → [{formatPoint(bend.line_end)}]
                      </p>
                    </div>
                    <button
                      type="button"
                      className="rounded p-2 text-dark-400 hover:bg-red-500/10 hover:text-red-300"
                      onClick={() => onDeleteManualBend(bend.bend_id)}
                      aria-label={`Delete ${bend.bend_id}`}
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div>
            <div className="mb-2 flex items-center justify-between">
              <p className="text-xs uppercase tracking-wide text-dark-400">Review F1 markers</p>
              <span className="text-xs text-dark-500">{reviews.size}/{markers.length} reviewed</span>
            </div>
            {markers.length === 0 ? (
              <p className="rounded-lg border border-dark-700 bg-dark-900/50 px-3 py-3 text-xs text-dark-500">
                F1 produced no displayable bend markers. Label every visible real bend manually.
              </p>
            ) : (
              <div className="space-y-2">
                {markers.map((marker) => (
                  <div key={marker.bendId} className="flex flex-col gap-2 rounded-lg border border-dark-600 bg-dark-900/55 px-3 py-2 lg:flex-row lg:items-center lg:justify-between">
                    <span className="text-sm font-medium text-dark-200">{marker.label}</span>
                    <div className="flex flex-wrap gap-2">
                      {decisionLabels.map(({ decision, label }) => (
                        <button
                          key={decision}
                          type="button"
                          className={clsx(
                            'rounded-lg border px-3 py-1.5 text-xs transition-colors',
                            reviews.get(marker.bendId) === decision
                              ? decision === 'real_bend'
                                ? 'border-emerald-400 bg-emerald-500/20 text-emerald-100'
                                : decision === 'not_bend_outline'
                                  ? 'border-red-400 bg-red-500/20 text-red-100'
                                  : 'border-amber-400 bg-amber-500/20 text-amber-100'
                              : 'border-dark-600 bg-dark-800 text-dark-400 hover:text-dark-200',
                          )}
                          onClick={() => onMarkerDecision(marker.bendId, decision)}
                        >
                          {decision === 'not_bend_outline' && <X className="mr-1 inline h-3.5 w-3.5" />}
                          {label}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
