/**
 * Top banner of the bender view: how far the part is ("8 / 12 bends done",
 * with one pip per planned bend) and one big part verdict (OK / OK so far /
 * CHECK / FIX / Rescan) taken from the engine's summary state, with how many
 * bends need fixing or checking and the permanent operator-assist note.
 * Colours follow the contract tokens (only a fully measured, in-tolerance
 * part is green; "OK so far" on an unfinished part is blue).
 */
import { stageStatusText, t, verdictText } from '../copy'
import { partVerdict, verdictCounts, type BannerView, type BendViewRow, type StageStripView } from '../model/stationView'
import type { Lang, ShopfloorInspectionResult } from '../model/types'
import { VERDICT_ICON, verdictTone } from './BendBits'

export interface TopBannerProps {
  lang: Lang
  banner: BannerView
  strip: StageStripView
  rows: BendViewRow[]
  result: ShopfloorInspectionResult
  isNew: boolean
  selectedUid: string | null
}

export function StageBox({ lang, strip, result, selectedUid }: Pick<TopBannerProps, 'lang' | 'strip' | 'result' | 'selectedUid'>) {
  const done = result.summary.bends_formed
  const total = result.summary.bends_total
  const flagged = strip.decision !== 'PASS'
  return (
    <section className="st-stagebox" data-testid="stage-strip" data-stage-status={strip.status} data-inferred-stage={strip.inferredStage ?? ''}>
      <div className="st-stagebox-count">
        <span className="st-num" dir="ltr">
          <strong>{done}</strong> / {total}
        </span>
        <span className="st-stagebox-word">{t('bendsDone', lang)}</span>
      </div>
      {/* Progress only (not a second verdict, not touch targets): tap a bend in the list or on the part. */}
      <ol className={`st-pips${strip.tiles.length > 14 ? ' is-dense' : ''}`} aria-hidden>
        {strip.tiles.map(tile => (
          <li
            key={tile.bendUid}
            className={`st-pip st-tone-${tile.tone}${tile.state === 'NOT_YET_FORMED' ? ' is-unformed' : ''}${tile.bendUid === selectedUid ? ' is-selected' : ''}`}
            data-bend-uid={tile.bendUid}
            data-state={tile.state}
            title={tile.label}
          />
        ))}
      </ol>
      {flagged && <span className="st-stagebox-note">{stageStatusText(strip.status, lang)}</span>}
    </section>
  )
}

export function VerdictBanner({ lang, banner, rows, result, isNew }: Pick<TopBannerProps, 'lang' | 'banner' | 'rows' | 'result' | 'isNew'>) {
  const verdict = partVerdict(banner.state)
  const Icon = VERDICT_ICON[verdict]
  const counts = verdictCounts(rows)
  const parts: { key: string; n: number; word: string }[] = [
    { key: 'fix', n: counts.FIX, word: t('bendsToFix', lang) },
    { key: 'check', n: counts.CHECK, word: t('bendsToCheck', lang) },
    { key: 'unsure', n: counts.UNSURE, word: t('bendsUnsure', lang) },
    { key: 'ok', n: counts.OK, word: t('bendsOk', lang) },
  ].filter(p => p.n > 0)
  // Bends that need nothing are counted only when nothing else does (the banner must fit at a glance).
  const shown = parts.some(p => p.key !== 'ok') ? parts.filter(p => p.key !== 'ok') : parts
  // The stage box already names a stage problem; the banner keeps validity and crease notes.
  const stageNote = stageStatusText(result.stage_decision.status, lang)
  const nothingFormed = result.bends.length > 0 && result.bends.every(b => b.state === 'NOT_YET_FORMED')
  return (
    <section
      className={`st-banner st-tone-${verdictTone(verdict)}${isNew ? ' is-new' : ''}`}
      data-testid="summary-banner"
      data-summary-state={banner.state}
      data-verdict={verdict}
      aria-live="polite"
    >
      <span className="st-banner-icon" aria-hidden>
        <Icon size={44} strokeWidth={3} />
      </span>
      <div className="st-banner-body">
        <div className="st-banner-line">
          <h2 className="st-banner-title">{verdictText(verdict, lang)}</h2>
          <span className="st-banner-counts">
            {nothingFormed ? (
              <span data-no-required-checks>{t('noBendsFormed', lang)}</span>
            ) : shown.length ? (
              shown.map(p => (
                <span key={p.key} className={`st-bcount st-bcount-${p.key}`} data-count={p.key}>
                  <strong className="st-num">{p.n}</strong> {p.word}
                </span>
              ))
            ) : null}
          </span>
        </div>
        <div className="st-banner-sub">
          <span className="st-banner-state">{banner.title}</span>
          {banner.details.filter(d => d !== stageNote).map(d => (
            <span key={d} className="st-banner-detail">{d}</span>
          ))}
        </div>
      </div>
    </section>
  )
}
