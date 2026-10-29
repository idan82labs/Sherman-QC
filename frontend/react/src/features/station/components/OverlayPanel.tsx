/**
 * "Detections (unverified)": the zero-shot overlay of the job on screen (`sherman_zeroshot_overlay_v1`).
 *
 * Shown apart from the bend rows, with dashed borders and the review / muted colours only (never green):
 * these are lines the AI detection found in the scan, not measurements. A bend whose position E3 did not
 * measure gets its detected line ("Position not measured: detected line", check); a bend the recipe does
 * not explain is an "Unexpected bend" and is listed as a REVIEW reason; on a refused scan every line is
 * "Detected bend, unverified".
 */
import { memo } from 'react'
import { ScanSearch } from 'lucide-react'

import { t } from '../copy'
import type { OverlayLine, ZeroshotOverlay } from '../model/zeroshotOverlay'
import type { Lang } from '../model/types'

export interface OverlayPanelProps {
  lang: Lang
  overlay: ZeroshotOverlay
  /** bend_uid -> label shown in the list (studio label). */
  labels: Record<string, string>
}

function kindText(line: OverlayLine, lang: Lang): string {
  if (line.kind === 'WITHHELD_BEND') return t('zsKindWithheld', lang)
  if (line.kind === 'UNEXPECTED_BEND') return t('zsKindUnexpected', lang)
  return t('zsKindRefused', lang)
}

export const OverlayPanel = memo(function OverlayPanel({ lang, overlay, labels }: OverlayPanelProps) {
  const refused = overlay.inspection_validity !== 'VALID'
  return (
    <section
      className="st-zs"
      data-testid="zs-overlay"
      data-overlay-version={overlay.overlay_version}
      data-effective-state={overlay.effective_summary_state}
      data-verdict-lowered={overlay.verdict_lowered ? 'true' : 'false'}
      data-lines={overlay.lines.length}
      aria-label={t('zsTitle', lang)}
    >
      <h3 className="st-zs-title">
        <ScanSearch size={22} aria-hidden />
        <span>{t('zsTitle', lang)}</span>
      </h3>
      <p className="st-zs-note">{refused ? t('zsRefusedNote', lang) : t('zsNote', lang)}</p>
      {overlay.flags.map(f => (
        <p key={f.code} className="st-zs-reason" data-testid="zs-review-reason" data-code={f.code} data-line-ids={f.line_ids.join(',')}>
          {t('zsReasonUnexpected', lang)} <bdi dir="ltr">({f.line_ids.join(', ')})</bdi>
        </p>
      ))}
      {(overlay.angle_checks ?? []).map(a => (
        <p key={`angle-${a.bend_uid}`} className="st-zs-reason" data-testid="zs-review-reason" data-code={a.code} data-bend-uid={a.bend_uid}>
          {t('zsReasonAngle', lang)}{' '}
          <bdi dir="ltr">({labels[a.bend_uid] ?? a.bend_uid}: E3 {a.e3_angle_deg.toFixed(1)}° / zs {a.zs_angle_deg.toFixed(1)}°)</bdi>
        </p>
      ))}
      {overlay.lines.length > 0 && (
        <ol className="st-zs-list">
          {overlay.lines.map(line => (
            <li
              key={line.line_id}
              className={`st-zs-row is-${line.decision === 'REVIEW' ? 'review' : 'unverified'}`}
              data-testid="zs-line"
              data-line-id={line.line_id}
              data-kind={line.kind}
              data-decision={line.decision}
              data-bend-uid={line.bend_uid ?? ''}
            >
              <span className="st-zs-id" dir="ltr">{line.line_id}</span>
              <span className="st-zs-kind">
                {kindText(line, lang)}
                {line.bend_uid && labels[line.bend_uid] ? <bdi dir="ltr"> · {labels[line.bend_uid]}</bdi> : null}
              </span>
              <span className="st-zs-len st-num" dir="ltr" data-angle-status={line.angle_status ?? 'SHOWN'}>
                {line.length_mm.toFixed(0)} {t('zsLength', lang)}
                {line.angle_status === 'NOT_MEASURED_JOGGLE' ? ` · ${t('zsAngleJoggle', lang)}` : line.incl_deg != null ? ` · ${line.incl_deg.toFixed(1)}°` : ''}
              </span>
              <span className="st-zs-chip">{line.decision === 'REVIEW' ? t('zsCheck', lang) : t('zsUnverified', lang)}</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  )
})
