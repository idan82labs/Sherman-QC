/**
 * Signed deviation legend. Swatches and gradient come from the same stops
 * as the point colours (`model/heatmapScale.ts`), so the legend always
 * matches what is drawn. When the heatmap is unavailable the legend says
 * why instead of showing a scale.
 */
import { heatmapReasonText, t } from '../copy'
import { HEATMAP_NAN_HEX, legendGradientCss, legendSwatches } from '../model/heatmapScale'
import { formatQuantity } from '../model/stationView'
import type { Lang } from '../model/types'

export interface HeatmapLegendProps {
  lang: Lang
  band: number
  /** Null when the heatmap is drawn; otherwise the reason code. */
  issue: string | null
  /** Published but not loaded yet: shown dimmed as `data-heatmap=loading`, never as on. */
  loading?: boolean
  active: boolean
}

function issueText(issue: string, lang: Lang): string {
  if (issue === 'COUNT_MISMATCH') return t('heatmapCountMismatch', lang)
  return heatmapReasonText(issue, lang)
}

export function HeatmapLegend({ lang, band, issue, loading = false, active }: HeatmapLegendProps) {
  if (issue) {
    return (
      <div className="st-legend is-unavailable" data-testid="heatmap-legend" data-heatmap="unavailable" data-reason={issue} data-viewer-reserved>
        <strong>{t('heatmapUnavailable', lang)}</strong>
        <span>{issueText(issue, lang)}</span>
      </div>
    )
  }
  const swatches = legendSwatches(band).reverse() // top = +2b
  const decimals = band < 0.5 ? 2 : 1
  const mode = !active ? 'off' : loading ? 'loading' : 'on'
  // Off (the bender default): no legend on screen; the element stays for checks and screen readers.
  if (mode === 'off') return <div className="st-legend is-off" data-testid="heatmap-legend" data-heatmap="off" aria-hidden />
  return (
    <div
      className={`st-legend${mode === 'on' ? '' : ' is-dimmed'}`}
      data-testid="heatmap-legend"
      data-heatmap={mode}
      data-viewer-reserved
      aria-label={t('heatmapTitle', lang)}
      aria-busy={mode === 'loading'}
    >
      <div className="st-legend-title">{mode === 'loading' ? t('heatmapLoading', lang) : t('heatmapTitle', lang)}</div>
      <div className="st-legend-body">
        <div className="st-legend-bar" style={{ background: legendGradientCss('to top') }} aria-hidden />
        <ol className="st-legend-ticks">
          {swatches.map(s => (
            <li key={s.value} data-value={s.value} data-swatch={s.hex}>
              <span className="st-legend-swatch" style={{ background: s.hex }} aria-hidden />
              <span className="st-num" dir="ltr">
                {formatQuantity(s.value, 'mm', { signed: true, decimals })}
              </span>
            </li>
          ))}
        </ol>
      </div>
      <div className="st-legend-ends">
        <span>+ {t('heatmapOver', lang)}</span>
        <span>{'−'} {t('heatmapUnder', lang)}</span>
      </div>
      <div className="st-legend-nan">
        <span className="st-legend-swatch" style={{ background: HEATMAP_NAN_HEX }} aria-hidden />
        {t('heatmapNoValue', lang)}
      </div>
    </div>
  )
}
