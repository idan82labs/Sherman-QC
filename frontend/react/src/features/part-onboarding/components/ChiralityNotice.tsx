/**
 * Left/right-sensitive part warning (display only), shown by the onboarding wizard and Vendor Setup.
 *
 * The map exists only for a released part (E1 compiles at release); the first read after a release
 * schedules it on the server and this polls while it is PENDING. Left/right-sensitive parts get a
 * clear warning with the distinguishing features listed (numbered like the viewer markers); an
 * achiral part, a pending map and a map that could not be computed are shown only where `quiet` is
 * false (the Release step), honestly as such. A failed read never blocks anything.
 */
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { onboardingApi, onboardingKeys } from '../api/onboardingApi'
import { onbCopy, type OnbLanguage } from '../copy'
import { featureText, isLeftRightSensitive, type ChiralityStatus } from '../model/chirality'
import { Notice } from './common'

export default function ChiralityNotice({ partId, status, language, quiet, markersShown }: {
  partId: string
  status: ChiralityStatus | null | undefined
  language: OnbLanguage
  /** Show only the left/right warning (nothing for achiral, pending or not computed). */
  quiet?: boolean
  /** The features are also marked on a 3D view ("L/R 1", ...). */
  markersShown?: boolean
}) {
  const t = onbCopy[language]
  const queryClient = useQueryClient()
  const retry = useMutation({
    mutationFn: () => onboardingApi.retryChirality(partId),
    onSettled: () => queryClient.invalidateQueries({ queryKey: onboardingKeys.chirality(partId) }),
  })
  if (isLeftRightSensitive(status)) {
    const summary = status!.summary!
    const features = summary.features
    const markers = features.filter(f => f.centre_final_mm).map((_, i) => t.chiralityMarker.replace('{n}', String(i + 1)))
    return (
      <div className="onb-chirality" role="alert" data-chirality={summary.part_class} lang={language}>
        <Notice tone="attention">
          <strong className="onb-chirality-title">{t.chiralityTitle}</strong>
          <p className="onb-chirality-lead">{t.chiralityLead} {features.length ? t.chiralityFeaturesLead : summary.part_class === 'CHIRAL_GROSS' ? t.chiralityGross : null}</p>
          {features.length > 0 && (
            <ol className="onb-chirality-list" data-chirality-features>
              {features.map((f, i) => (
                <li key={f.feature} data-chirality-feature={f.type}>
                  <span className="onb-chirality-num" dir="ltr">{t.chiralityMarker.replace('{n}', String(i + 1))}</span>
                  <span dir="auto">{featureText(f, t)}</span>
                </li>
              ))}
            </ol>
          )}
          {markersShown && markers.length > 0 && <p className="onb-help">{t.chiralityMarkedHint.replace('{m}', markers.join(', '))}</p>}
          {summary.part_class === 'CHIRAL_BY_FEATURES' && <p className="onb-help">{t.chiralityStationNote}</p>}
        </Notice>
      </div>
    )
  }
  if (quiet || !status) return null
  if (status.status === 'READY') {
    return <div className="onb-chirality" data-chirality="ACHIRAL"><Notice tone="neutral" role="status">{t.chiralityNotSensitive}</Notice></div>
  }
  if (status.status === 'PENDING') {
    return <div className="onb-chirality" data-chirality="PENDING"><Notice tone="neutral" role="status">{t.chiralityPending}</Notice></div>
  }
  return (
    <div className="onb-chirality" data-chirality="NOT_COMPUTED" data-chirality-reason={status.reason ?? ''}>
      <Notice tone="neutral" role="status">
        <span>{t.chiralityNotComputed}</span>{' '}
        {status.reason && <span className="onb-reason-code" dir="ltr">{status.reason}</span>}
        {status.status === 'FAILED' && (
          <button type="button" className="onb-btn onb-btn-link" disabled={retry.isPending} onClick={() => retry.mutate()} data-chirality-retry>
            {t.chiralityRetry}
          </button>
        )}
      </Notice>
    </div>
  )
}
